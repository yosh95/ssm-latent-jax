"""Pure-Mamba predictors (no JEPA machinery) and the closed-loop "imagination" API.

* :class:`MambaPredictor` - one SSM block, observation -> next observation.
* :class:`MultiScaleMambaPredictor` - multi-timescale stack with an action
  fusion path, an ``h``-based closed loop and scheduled h-sampling training.

These are the models used by the anomaly-detection / streaming use cases, where
the SSM hidden state *is* the world model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig, SsmConfig
from ssm_latent.ssm import MultiScaleSsmBlock, MultiScaleSsmState, SsmBlock, SsmState
from ssm_latent.utils import ModelIOMixin, leaky_relu

__all__ = [
    "PredictorState",
    "MultiScalePredictorState",
    "MambaPredictor",
    "MultiScaleMambaPredictor",
]


class PredictorState(NamedTuple):
    """Streaming state of :class:`MambaPredictor`."""

    h: jax.Array
    prev_bx: jax.Array | None = None
    conv_state: jax.Array | None = None


class MultiScalePredictorState(NamedTuple):
    """Streaming state of :class:`MultiScaleMambaPredictor`."""

    h: list[jax.Array]
    prev_bx: list[jax.Array]
    conv_state: list[jax.Array | None]


class MambaPredictor(ModelIOMixin, nnx.Module):
    """``input_dim -> d_model -> SSM -> output_dim`` next-step predictor.

    ``predictions[t]`` estimates ``x[t+1]``; the loss therefore compares
    ``predictions[:, :-1]`` against ``targets[:, 1:]``.
    """

    def __init__(
        self,
        config: SsmConfig,
        input_dim: int,
        output_dim: int,
        rngs: nnx.Rngs | None = None,
        *,
        dtype=jnp.float32,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.input_proj = nnx.Linear(input_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.ssm = SsmBlock(config, rngs, dtype=dtype)
        self.output_proj = nnx.Linear(config.d_model, output_dim, rngs=rngs, param_dtype=dtype)
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.n_heads = config.n_heads
        self.d_state = config.d_state
        self.mimo_rank = config.mimo_rank
        self.use_conv = config.use_conv
        self.conv_kernel = config.conv_kernel

    def __call__(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        return self.forward(x, scan_mode=scan_mode)

    def forward(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        return self.output_proj(self.ssm(self.input_proj(x), scan_mode=scan_mode))

    def zero_state(self, batch: int, dtype=None) -> PredictorState:
        st = self.ssm.zero_state(batch, dtype)
        return PredictorState(h=st.h, prev_bx=None, conv_state=st.conv_state)

    def step(self, x: jax.Array, state: PredictorState):
        """``[B, input_dim] -> ([B, output_dim], new_state)``."""
        y, st = self.ssm.step(
            self.input_proj(x),
            SsmState(
                h=state.h,
                prev_bx=state.prev_bx if state.prev_bx is not None else jnp.zeros_like(state.h),
                conv_state=state.conv_state,
            ),
        )
        return self.output_proj(y), PredictorState(
            h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state
        )

    def scan_sequential(self, x: jax.Array) -> jax.Array:
        """Step-by-step forward over a whole sequence (for tests / streaming)."""
        batch, seq_len, _ = x.shape
        state = self.zero_state(batch, dtype=x.dtype)
        outs = []
        for t in range(seq_len):
            pred, state = self.step(x[:, t], state)
            outs.append(pred)
        return jnp.stack(outs, axis=1)

    @staticmethod
    def loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
        """Next-step MSE (``predictions[:, :-1]`` vs ``targets[:, 1:]``)."""
        return ((targets[:, 1:] - predictions[:, :-1]) ** 2).mean()


class MultiScaleMambaPredictor(ModelIOMixin, nnx.Module):
    """Multi-timescale Mamba predictor with action conditioning and closed loops.

    The model exposes four forward paths that all share ``obs_proj`` and
    ``imagine_fusion``:

    ==========================  ==========================================
    :meth:`forward`             observation only (zero action vector)
    :meth:`forward_with_action` observation + ground-truth action
    :meth:`step_imagine`        closed loop through observation space
    :meth:`step_imagine_h`      closed loop through the SSM hidden state
    ==========================  ==========================================
    """

    def __init__(
        self,
        config: MultiScaleSsmConfig,
        input_dim: int,
        output_dim: int,
        rngs: nnx.Rngs | None = None,
        *,
        dtype=jnp.float32,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.ssms = MultiScaleSsmBlock(config, rngs, dtype=dtype)
        self.output_proj = nnx.Linear(config.d_model, output_dim, rngs=rngs, param_dtype=dtype)
        self.imagine_fusion = nnx.Linear(
            2 * config.d_model, config.d_model, rngs=rngs, param_dtype=dtype
        )
        self.obs_proj = nnx.Linear(input_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.action_proj = nnx.Linear(output_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.h_proj = nnx.Linear(config.h_flat_dim, config.d_model, rngs=rngs, param_dtype=dtype)

        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.n_heads = config.n_heads
        self.d_state = config.d_state
        self.mimo_rank = config.mimo_rank
        self.n_layers = config.n_layers
        self.use_conv = config.use_conv
        self.conv_kernel = config.conv_kernel
        self.h_flat_dim = config.h_flat_dim

    # ------------------------------------------------------------- helpers
    def _h_flat(self, state: MultiScalePredictorState) -> jax.Array:
        h_last = state.h[self.n_layers - 1]
        return h_last.reshape(h_last.shape[0], -1)

    def _fuse(self, obs_like: jax.Array, action_like: jax.Array) -> jax.Array:
        return self.imagine_fusion(jnp.concatenate([obs_like, action_like], axis=-1))

    def zero_state(self, batch: int, dtype=None) -> MultiScalePredictorState:
        st = self.ssms.zero_state(batch, dtype)
        return MultiScalePredictorState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state)

    # ------------------------------------------------------ observation-only
    def __call__(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        return self.forward(x, scan_mode=scan_mode)

    def forward(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        batch, seq_len = x.shape[:2]
        obs_enc = self.obs_proj(x)
        zero_action = jnp.zeros((batch, seq_len, self.d_model), dtype=obs_enc.dtype)
        u = self._fuse(obs_enc, zero_action)
        return self.output_proj(self.ssms(u, scan_mode=scan_mode))

    def step(self, x: jax.Array, state: MultiScalePredictorState):
        batch = x.shape[0]
        obs_enc = self.obs_proj(x)
        zero_action = jnp.zeros((batch, self.d_model), dtype=obs_enc.dtype)
        u = self._fuse(obs_enc, zero_action)
        y, st = self.ssms.step(u, MultiScaleSsmState(state.h, state.prev_bx, state.conv_state))
        return self.output_proj(y), MultiScalePredictorState(
            h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state
        )

    def scan_sequential(self, x: jax.Array) -> jax.Array:
        """Step-by-step (observation-only) forward, for tests and streaming."""
        batch, seq_len = x.shape[:2]
        state = self.zero_state(batch, dtype=x.dtype)
        outs = []
        for t in range(seq_len):
            pred, state = self.step(x[:, t], state)
            outs.append(pred)
        return jnp.stack(outs, axis=1)

    # ------------------------------------------------------ action-conditioned
    def forward_with_action(
        self, observations: jax.Array, actions: jax.Array, *, scan_mode: str = "parallel"
    ) -> jax.Array:
        u = self._fuse(self.obs_proj(observations), self.action_proj(actions))
        return self.output_proj(self.ssms(u, scan_mode=scan_mode))

    # ---------------------------------------------------------- closed loops
    def step_imagine(
        self,
        y_prev: jax.Array,
        action: jax.Array,
        state: MultiScalePredictorState,
    ) -> tuple[jax.Array, jax.Array, MultiScalePredictorState]:
        """Closed loop through observation space (``y -> x -> z -> SSM``).

        ``y_prev`` must be the *SSM output* (``d_model`` wide), not the decoded
        prediction (``output_dim`` wide) - feeding the latter silently loses the
        information the world model carries in its latent state.
        """
        if y_prev.shape[-1] != self.d_model:
            raise ValueError(
                f"step_imagine expects the SSM output of width d_model={self.d_model}, "
                f"got width {y_prev.shape[-1]}"
            )
        y_obs = self.output_proj(y_prev)
        y_enc = self.obs_proj(y_obs)
        u = self._fuse(y_enc, self.action_proj(action))
        y_next, st = self.ssms.step(u, MultiScaleSsmState(state.h, state.prev_bx, state.conv_state))
        return (
            leaky_relu(y_obs),
            y_next,
            MultiScalePredictorState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state),
        )

    def step_imagine_h(
        self, state: MultiScalePredictorState, action: jax.Array
    ) -> tuple[jax.Array, jax.Array, MultiScalePredictorState]:
        """Closed loop through the last layer's hidden state ``h``."""
        h_enc = self.h_proj(self._h_flat(state))
        u = self._fuse(h_enc, self.action_proj(action))
        y_next, st = self.ssms.step(u, MultiScaleSsmState(state.h, state.prev_bx, state.conv_state))
        return (
            leaky_relu(self.output_proj(y_next)),
            y_next,
            MultiScalePredictorState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state),
        )

    # ----------------------------------------------------------------- training
    def forward_with_h_sampling(
        self,
        observations: jax.Array,
        actions: jax.Array,
        h_sampling_prob: float = 0.3,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Scheduled-sampling training of the ``step_imagine_h`` closed loop.

        With probability ``h_sampling_prob`` the SSM input at step ``t`` is
        derived from ``h_{t-1}`` instead of the ground-truth observation, which
        is what removes the exposure bias of the h-based loop.  The first step
        always uses the observation.
        """
        if h_sampling_prob <= 0.0:
            return self.forward_with_action(observations, actions)

        batch, seq_len = observations.shape[:2]
        key = key if key is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, seq_len)

        state = self.zero_state(batch, dtype=observations.dtype)
        preds = []
        for t in range(seq_len):
            obs_t = observations[:, t]
            act_t = actions[:, t]
            obs_enc = self.obs_proj(obs_t)
            use_h = (t > 0) & (jax.random.uniform(keys[t]) < h_sampling_prob)
            h_enc = self.h_proj(self._h_flat(state))
            input_enc = jnp.where(use_h, h_enc, obs_enc)
            u = self._fuse(input_enc, self.action_proj(act_t))
            y, st = self.ssms.step(u, MultiScaleSsmState(state.h, state.prev_bx, state.conv_state))
            preds.append(self.output_proj(y)[:, None])
            state = MultiScalePredictorState(h=st.h, prev_bx=st.prev_bx, conv_state=st.conv_state)
        return jnp.concatenate(preds, axis=1)

    @staticmethod
    def loss(predictions: jax.Array, targets: jax.Array) -> jax.Array:
        """Next-step MSE (``predictions[:, :-1]`` vs ``targets[:, 1:]``)."""
        return ((targets[:, 1:] - predictions[:, :-1]) ** 2).mean()
