"""JEPA-style latent predictors and their loss functions.

The model predicts the *next* latent state from the fused (observation, action)
representation and is trained with

``L = L_latent + recon_weight * (L_recon + L_pred_x)
        + stability_weight * L_stability + curvature_weight * L_curvature``

Two collapse-prevention regularisers are provided:

* :func:`stability_loss` / :func:`stability_loss_running` - VICReg-style moment
  matching on random projections (mean 0, variance 1).
* :func:`sigreg_loss` - Sketched Isotropic Gaussian Regularization, the LeJEPA
  characteristic-function matching objective (Balestriero & LeCun, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig, SsmConfig
from ssm_latent.ssm import MultiScaleSsmBlock, MultiScaleSsmState, SsmBlock, SsmState
from ssm_latent.utils import ModelIOMixin, mean_squared_error_or_zero, normalize_projections

__all__ = [
    "Buffer",
    "MlpEncoder",
    "LatentState",
    "LatentLossArgs",
    "LatentPredictor",
    "MultiScaleLatentPredictor",
    "MultiscaleLatentState",
    "latent_loss",
    "stability_loss",
    "stability_loss_running",
    "sigreg_loss",
    "lejepa_loss",
    "curvature_loss",
    "curvature_loss_with_dt",
    "make_projections",
]

N_PROJECTIONS_DEFAULT = 16
SIGREG_FREQS_DEFAULT = (0.5, 1.0, 1.5, 2.0)


class Buffer(nnx.Variable):
    """Non-trainable array registered in the module state.

    Used for the frozen random projection matrices: they live inside the NNX
    graph (so they are checkpointed together with the parameters and moved with
    the model) but are invisible to ``wrt=nnx.Param`` optimiser filters.
    """


def make_projections(
    d_model: int, n_projections: int = N_PROJECTIONS_DEFAULT, *, seed: int = 0
) -> jax.Array:
    """Normalised, frozen random projection matrix ``[d_model, n_projections]``.

    Sampled from ``N(0, 1)`` and normalised column-wise; never updated by the
    optimiser (registered as a :class:`Buffer`, not a ``Param``).
    """
    key = jax.random.PRNGKey(seed)
    w = jax.random.normal(key, (d_model, n_projections), jnp.float32)
    return jax.lax.stop_gradient(normalize_projections(w))


# --------------------------------------------------------------------------- #
# loss functions
# --------------------------------------------------------------------------- #
def stability_loss(z: jax.Array, w: jax.Array) -> jax.Array:
    """VICReg-style moment matching on random projections.

    ``z``: ``[B, T, d_model]``; ``w``: ``[d_model, K]`` normalised projections.
    Penalises projected means away from 0 and projected variances away from 1.
    """
    batch, seq_len, d_model = z.shape
    z_flat = z.reshape(batch * seq_len, d_model)
    projections = (z_flat @ w).reshape(batch, seq_len, w.shape[1])
    mean = projections.mean(axis=1)  # [B, K]
    var = ((projections - mean[:, None, :]) ** 2).mean(axis=1)
    return (mean**2).mean() + ((var - 1.0) ** 2).mean()


def stability_loss_running(
    z: jax.Array,
    w: jax.Array,
    running_mean: jax.Array,
    running_var: jax.Array,
    momentum: float = 0.01,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Moment matching against exponential-moving-average statistics.

    Useful for the batch=1 / short-sequence regime: the loss still uses the
    *current* batch statistics (so it always pushes towards ``N(0, 1)``) while
    returning updated EMAs for monitoring.
    """
    batch, seq_len, d_model = z.shape
    z_flat = z.reshape(batch * seq_len, d_model)
    projections = (z_flat @ w).reshape(batch, seq_len, w.shape[1])

    batch_mean = projections.mean(axis=(0, 1))
    batch_var = ((projections - batch_mean[None, None, :]) ** 2).mean(axis=(0, 1))
    new_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
    new_var = (1.0 - momentum) * running_var + momentum * batch_var

    loss = (batch_mean**2).mean() + ((batch_var - 1.0) ** 2).mean()
    return loss, new_mean, new_var


def sigreg_loss(z: jax.Array, w: jax.Array, freqs=SIGREG_FREQS_DEFAULT) -> jax.Array:
    """Sketched Isotropic Gaussian Regularisation (LeJEPA, Balestriero & LeCun 2025).

    Projects embeddings onto ``K`` random directions, standardises each of them
    (zero mean / unit variance) and matches the empirical characteristic
    function to that of ``N(0, 1)``, ``phi(t) = exp(-t^2 / 2)``, at the given
    ``freqs``.  Because the projections are standardised the penalty depends
    only on *shape* (skew/kurtosis-like deviations from Gaussianity).
    """
    n = z.shape[0] * z.shape[1]
    u = z.reshape(n, z.shape[-1]) @ w
    mean = u.mean(axis=0, keepdims=True)
    var = ((u - mean) ** 2).mean(axis=0, keepdims=True)
    u = (u - mean) / (jnp.sqrt(var) + 1e-6)

    total = jnp.zeros((), dtype=z.dtype)
    for t in freqs:
        target = jnp.exp(-(jnp.asarray(t, z.dtype) ** 2) / 2.0)
        re = jnp.cos(u * t).mean(axis=0)
        im = jnp.sin(u * t).mean(axis=0)
        total = total + ((re - target) ** 2 + im**2).mean()
    return total / len(freqs)


def lejepa_loss(
    z: jax.Array,
    pred_z: jax.Array,
    projections: jax.Array,
    sigreg_weight: float = 1.0,
    freqs=SIGREG_FREQS_DEFAULT,
) -> jax.Array:
    """``L_pred + gamma * SIGReg(z)`` - the LeJEPA training objective."""
    _, seq_len = z.shape[:2]
    target = jax.lax.stop_gradient(z[:, 1:])
    mse_latent = mean_squared_error_or_zero(target - pred_z[:, : seq_len - 1])
    return mse_latent + sigreg_weight * sigreg_loss(z, projections, freqs)


def curvature_loss(z: jax.Array) -> jax.Array:
    """Temporal-straightening loss: mean squared 2nd-order difference of ``z``.

    Returns ``0`` for sequences shorter than 3 steps.
    """
    seq_len = z.shape[1]
    if seq_len < 3:
        return jnp.zeros((), dtype=z.dtype)
    accel = z[:, 2:] - 2.0 * z[:, 1:-1] + z[:, :-2]
    return (accel**2).mean()


def curvature_loss_with_dt(z: jax.Array, dt: jax.Array) -> jax.Array:
    """Time-delta aware 2nd-order difference for non-uniformly spaced data.

    ``dt``: ``[T]`` or ``[B, T]``; ``dt[t]`` is the step from ``t-1`` to ``t``.
    """
    seq_len = z.shape[1]
    if seq_len < 3:
        return jnp.zeros((), dtype=z.dtype)
    if dt.ndim == 1:
        dt = dt[None, :]
    dt = jnp.broadcast_to(dt, (z.shape[0], seq_len))[:, :, None]
    dt_t = jnp.maximum(dt[:, 2:], 1e-6)
    dt_t1 = jnp.maximum(dt[:, 1:-1], 1e-6)
    first = (z[:, 2:] - z[:, 1:-1]) / dt_t
    second = (z[:, 1:-1] - z[:, :-2]) / dt_t1
    return ((first - second) ** 2).mean()


# --------------------------------------------------------------------------- #
# MLP encoder
# --------------------------------------------------------------------------- #
class MlpEncoder(nnx.Module):
    """Nonlinear feed-forward encoder ``input_dim -> ... -> d_model``.

    Theorem 1 of Klindt, LeCun & Balestriero (2026) guarantees linear
    identifiability for *any* encoder (nonlinear included) that minimises the
    LeJEPA objective subject to Gaussian embeddings, so a nonlinear encoder can
    be used to probe the theory empirically.

    ``n_hidden == 0`` degenerates to a single linear layer.  Dropout is
    intentionally not implemented: it is a no-op at inference time and would
    break the determinism of the tests.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_hidden: int = 1,
        hidden_dim: int = 64,
        rngs: nnx.Rngs | None = None,
        *,
        dtype=jnp.float32,
    ):
        if n_hidden > 0 and hidden_dim == 0:
            raise ValueError("hidden_dim must be > 0 when n_hidden > 0")
        rngs = rngs or nnx.Rngs(0)
        self.n_hidden = n_hidden

        if n_hidden == 0:
            self.input_proj = nnx.Linear(input_dim, d_model, rngs=rngs, param_dtype=dtype)
            self.hidden: nnx.List = nnx.List([])
            self.output_proj: nnx.Linear | None = None
            return

        self.input_proj = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=dtype)
        self.hidden = nnx.List(
            [
                nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=dtype)
                for _ in range(max(n_hidden - 1, 0))
            ]
        )
        self.output_proj = nnx.Linear(hidden_dim, d_model, rngs=rngs, param_dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.forward(x)

    def forward(self, x: jax.Array) -> jax.Array:
        """``[B, T, input_dim] -> [B, T, d_model]``."""
        if self.n_hidden == 0:
            return self.input_proj(x)
        h = self.input_proj(x)
        for layer in self.hidden:
            h = layer(jax.nn.relu(h))
        assert self.output_proj is not None  # set whenever n_hidden > 0
        return self.output_proj(jax.nn.relu(h))

    def forward_single(self, x: jax.Array) -> jax.Array:
        """``[B, input_dim] -> [B, d_model]``."""
        if self.n_hidden == 0:
            return self.input_proj(x)
        h = self.input_proj(x)
        for layer in self.hidden:
            h = layer(jax.nn.relu(h))
        assert self.output_proj is not None  # set whenever n_hidden > 0
        return self.output_proj(jax.nn.relu(h))


# --------------------------------------------------------------------------- #
# single-scale latent predictor
# --------------------------------------------------------------------------- #
class LatentState(NamedTuple):
    """Streaming state of :class:`LatentPredictor`."""

    h: jax.Array
    prev_bx: jax.Array | None = None
    conv_state: jax.Array | None = None


@dataclass
class LatentLossArgs:
    """Inputs of :func:`latent_loss`."""

    z: jax.Array
    """Current latents ``[B, T, d_model]``."""
    pred_z: jax.Array
    """Predicted next latents ``[B, T, d_model]``."""
    reconstructed_x: jax.Array
    """Decoder output for the current latent."""
    predicted_x: jax.Array
    """Decoder output for the predicted latent."""
    original_x: jax.Array
    """Ground-truth observations."""
    stability_weight: float = 1.0
    curvature_weight: float = 0.5
    recon_weight: float = 1.0


def latent_loss(args: LatentLossArgs, projections: jax.Array) -> jax.Array:
    """Combined JEPA latent-prediction + reconstruction + regularisation loss."""
    _, seq_len = args.z.shape[:2]
    target_z = jax.lax.stop_gradient(args.z[:, 1:])
    mse_latent = mean_squared_error_or_zero(target_z - args.pred_z[:, : seq_len - 1])

    mse_recon = ((args.original_x - args.reconstructed_x) ** 2).mean()
    target_x = jax.lax.stop_gradient(args.original_x[:, 1:])
    mse_pred_x = mean_squared_error_or_zero(target_x - args.predicted_x[:, : seq_len - 1])

    reg = stability_loss(args.z, projections)
    curv = curvature_loss(args.z)
    return (
        mse_latent
        + args.recon_weight * (mse_recon + mse_pred_x)
        + args.stability_weight * reg
        + args.curvature_weight * curv
    )


class LatentPredictor(ModelIOMixin, nnx.Module):
    """Encode -> fuse with action -> SSM -> decode, all in latent space.

    Parameters
    ----------
    config:
        :class:`~ssm_latent.config.SsmConfig` for the inner SSM block.
    input_dim / action_dim:
        Observation and action dimensionality.
    n_projections:
        Number of probes used by the stability / SIGReg regularisers.
    """

    def __init__(
        self,
        config: SsmConfig,
        input_dim: int,
        action_dim: int,
        rngs: nnx.Rngs | None = None,
        *,
        n_projections: int = N_PROJECTIONS_DEFAULT,
        projection_seed: int = 0,
        dtype=jnp.float32,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.d_model = config.d_model

        self.encoder = nnx.Linear(input_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.decoder = nnx.Linear(config.d_model, input_dim, rngs=rngs, param_dtype=dtype)
        self.action_encoder = nnx.Linear(action_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.fusion = nnx.Linear(2 * config.d_model, config.d_model, rngs=rngs, param_dtype=dtype)
        self.ssm = SsmBlock(config, rngs, dtype=dtype)
        self.stability_projections = Buffer(
            jnp.asarray(make_projections(config.d_model, n_projections, seed=projection_seed))
        )

    # ------------------------------------------------------------- helpers
    def encode(self, observations: jax.Array) -> jax.Array:
        return self.encoder(observations)

    def decode(self, z: jax.Array) -> jax.Array:
        return self.decoder(z)

    def __call__(self, observations: jax.Array, actions: jax.Array):
        return self.forward(observations, actions)

    def forward(self, observations: jax.Array, actions: jax.Array, *, scan_mode: str = "parallel"):
        """Returns ``(z, pred_z, reconstructed_x, predicted_x)``."""
        z = self.encoder(observations)
        a = self.action_encoder(actions)
        u = self.fusion(jnp.concatenate([z, a], axis=-1))
        pred_z = self.ssm(u, scan_mode=scan_mode)
        return z, pred_z, self.decoder(z), self.decoder(pred_z)

    def zero_state(self, batch: int) -> LatentState:
        st = self.ssm.zero_state(batch)
        return LatentState(h=st.h, prev_bx=None, conv_state=st.conv_state)

    def step(self, z_prev: jax.Array, action: jax.Array, state: LatentState):
        """One autoregressive step.  Returns ``(pred_z, new_state)``."""
        a = self.action_encoder(action)
        u = self.fusion(jnp.concatenate([z_prev, a], axis=-1))
        prev_bx = state.prev_bx if state.prev_bx is not None else jnp.zeros_like(state.h)
        y, st = self.ssm.step(u, SsmState(h=state.h, prev_bx=prev_bx, conv_state=state.conv_state))
        return y, LatentState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state)

    def loss(self, args: LatentLossArgs) -> jax.Array:
        return latent_loss(args, jax.lax.stop_gradient(self.stability_projections[...]))


# --------------------------------------------------------------------------- #
# multi-scale latent predictor
# --------------------------------------------------------------------------- #
class MultiscaleLatentState(NamedTuple):
    """Streaming state of :class:`MultiScaleLatentPredictor` (alias of the SSM state)."""

    h: list[jax.Array]
    prev_bx: list[jax.Array]
    conv_state: list[jax.Array | None]


class MultiScaleLatentPredictor(ModelIOMixin, nnx.Module):
    """JEPA predictor whose dynamics come from a multi-timescale SSM stack.

    Set ``action_dim=0`` for the unsupervised (no-action) variant: the fusion
    layer then maps ``d_model -> d_model`` instead of concatenating actions.
    """

    def __init__(
        self,
        config: MultiScaleSsmConfig,
        input_dim: int,
        action_dim: int,
        rngs: nnx.Rngs | None = None,
        *,
        n_projections: int = N_PROJECTIONS_DEFAULT,
        projection_seed: int = 0,
        dtype=jnp.float32,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.d_model = config.d_model
        self.has_action = action_dim > 0

        self.encoder = nnx.Linear(input_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.decoder = nnx.Linear(config.d_model, input_dim, rngs=rngs, param_dtype=dtype)
        self.action_encoder: nnx.Linear | None
        if self.has_action:
            self.action_encoder = nnx.Linear(
                action_dim, config.d_model, rngs=rngs, param_dtype=dtype
            )
            self.fusion = nnx.Linear(
                2 * config.d_model, config.d_model, rngs=rngs, param_dtype=dtype
            )
        else:
            self.action_encoder = None
            self.fusion = nnx.Linear(config.d_model, config.d_model, rngs=rngs, param_dtype=dtype)

        self.ssm = MultiScaleSsmBlock(config, rngs, dtype=dtype)
        self.stability_projections = Buffer(
            jnp.asarray(make_projections(config.d_model, n_projections, seed=projection_seed))
        )

    # ------------------------------------------------------------- helpers
    def encode(self, observations: jax.Array) -> jax.Array:
        return self.encoder(observations)

    def decode(self, z: jax.Array) -> jax.Array:
        return self.decoder(z)

    def _fuse(self, obs: jax.Array, actions: jax.Array | None) -> jax.Array:
        z = self.encoder(obs)
        if self.action_encoder is not None:
            if actions is None:
                raise ValueError("this predictor was built with action conditioning; pass actions")
            return self.fusion(jnp.concatenate([z, self.action_encoder(actions)], axis=-1))
        return self.fusion(z)

    def __call__(self, observations: jax.Array, actions: jax.Array | None = None):
        return self.forward(observations, actions)

    def forward(
        self,
        observations: jax.Array,
        actions: jax.Array | None = None,
        *,
        scan_mode: str = "parallel",
    ):
        """Returns ``(z, pred_z, reconstructed_x, predicted_x)``."""
        z = self.encoder(observations)
        if self.action_encoder is not None:
            if actions is None:
                raise ValueError("this predictor was built with action conditioning; pass actions")
            u = self.fusion(jnp.concatenate([z, self.action_encoder(actions)], axis=-1))
        else:
            u = self.fusion(z)
        pred_z = self.ssm(u, scan_mode=scan_mode)
        return z, pred_z, self.decoder(z), self.decoder(pred_z)

    def forward_no_action(self, observations: jax.Array, *, scan_mode: str = "parallel"):
        """Forward pass without actions (valid for both variants)."""
        z = self.encoder(observations)
        pred_z = self.ssm(self.fusion(z), scan_mode=scan_mode)
        return z, pred_z, self.decoder(z), self.decoder(pred_z)

    def zero_state(self, batch: int) -> MultiscaleLatentState:
        st = self.ssm.zero_state(batch)
        return MultiscaleLatentState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state)

    def step(self, z_prev: jax.Array, action: jax.Array | None, state: MultiscaleLatentState):
        """One autoregressive step.  Returns ``(pred_z, new_state)``.

        For an action-conditioned model (``action_dim > 0``) an action must be
        supplied: silently falling back to the unconditioned path would drop the
        control input and quietly change the dynamics, so it raises instead.
        """
        if self.action_encoder is not None:
            if action is None:
                raise ValueError(
                    "this predictor was built with action conditioning; pass an action to step()"
                )
            u = self.fusion(jnp.concatenate([z_prev, self.action_encoder(action)], axis=-1))
        else:
            u = self.fusion(z_prev)
        y, st = self.ssm.step(u, MultiScaleSsmState(state.h, state.prev_bx, state.conv_state))
        return y, MultiscaleLatentState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state)

    def loss(self, args: LatentLossArgs) -> jax.Array:
        return latent_loss(args, jax.lax.stop_gradient(self.stability_projections[...]))
