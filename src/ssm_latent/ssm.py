"""Mamba-3 style selective state-space model (JAX / Flax NNX).

The recurrence, complex-valued rotation and BCNorm here implement Mamba-3
(Lahoti et al., 2026) directly in JAX.  The numerical conventions - the
head-wise ``theta`` rotation, the duplicated real/imaginary damping blocks and
the two-term exponential-trapezoidal recurrence - are shared with the sibling
Rust and PyTorch implementations so cross-language results stay comparable.

Recurrence implemented by :meth:`selective_scan` (Mamba-3, Lahoti et al., 2026)::

    h_t = exp(dt_t * A_t) . h_{t-1}
        + (1 - lam_t) * dt_t * exp(dt_t * A_t) * B_{t-1} x_{t-1}   (beta)
        + lam_t * dt_t * B_t x_t                                  (gamma)
    y_t = C_t . h_t

with ``A_t = a_re + i * (a_im * dt_t + theta_t)`` a *complex diagonal* state
transition matrix (so ``exp(dt A)`` rotates by ``a_im * dt + theta``), and

* ``dt_t = softplus(dt_proj(u_t))``      input-dependent step size
* ``lam_t = sigmoid(lambda_proj(u_t))``  trapezoidal gate
* ``B_t``/``C_t`` passed through BCNorm (RMSNorm) and head-wise biases

Two execution modes are provided and are mathematically equivalent:

* :meth:`SsmBlock.__call__` - Blelloch parallel scan, ``O(log T)`` depth
* :meth:`SsmBlock.step`     - single-step, ``O(1)`` state update for streaming

The ``normalize_projections`` / ``w / (‖w‖ + 1e-6)`` convention lives in
:mod:`ssm_latent.utils`; this module deliberately no longer re-exports it (it is
not used by the SSM itself), so there is exactly one definition to keep in sync
with the Rust / PyTorch siblings.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig, SsmConfig

__all__ = [
    "SsmState",
    "SsmBlock",
    "MultiScaleSsmState",
    "MultiScaleSsmBlock",
    "causal_conv1d",
    "parallel_scan",
    "sequential_scan",
]

_RMS_EPS = 1e-5  # RMSNorm epsilon (matches the other implementations)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def causal_conv1d(
    x: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
) -> jax.Array:
    """Depthwise causal 1-D convolution with ``"valid"`` output length.

    ``x`` is ``[B, T, C]``, ``weight`` is ``[C, 1, K]`` and the result is
    ``[B, T, C]`` with ``"valid"`` length semantics: ``K - 1`` steps of zero
    padding are prepended so that the length is preserved and the convolution
    never looks into the future.
    """
    batch, seq_len, channels = x.shape
    kernel = weight.shape[-1]
    xt = jnp.transpose(x, (0, 2, 1))  # [B, C, T]
    padded = jnp.pad(xt, ((0, 0), (0, 0), (kernel - 1, 0)))
    # depthwise: shift-and-scale accumulation (kernel is tiny, typically 4)
    out = jnp.zeros((batch, channels, seq_len), dtype=x.dtype)
    for k in range(kernel):
        w_k = weight[:, 0, k]  # [C]
        out = out + padded[:, :, k : k + seq_len] * w_k[None, :, None]
    if bias is not None:
        out = out + bias[None, :, None]
    return jnp.transpose(out, (0, 2, 1))


def sequential_scan(
    a: jax.Array,
    w: jax.Array,
    h0: jax.Array | None = None,
) -> jax.Array:
    """Reference ``O(T)`` scan used to cross-validate the parallel scan.

    ``a`` is ``[..., T, 2, 2]`` and ``w`` is ``[..., T, R, 2]``.  Returns the
    inclusive scan ``h[..., T, R, 2]`` with ``h_t = a_t @ h_{t-1} + w_t``.
    """
    t_len = a.shape[-3]
    if h0 is None:
        h0 = jnp.zeros(w.shape[:-3] + w.shape[-2:], dtype=w.dtype)
    hs = []
    h = h0
    for t in range(t_len):
        h = jnp.einsum("...ij,...dj->...di", a[:, t], h) + w[:, t]
        hs.append(h)
    return jnp.stack(hs, axis=-3)


def parallel_scan(
    a: jax.Array,
    w: jax.Array,
    mode: str = "parallel",
) -> jax.Array:
    """Inclusive scan of the affine recurrence ``h_t = a_t h_{t-1} + w_t``.

    Parameters
    ----------
    a:
        ``[..., T, 2, 2]`` transition matrices.
    w:
        ``[..., T, R, 2]`` additive terms.
    mode:
        ``"parallel"`` (default) uses the Blelloch/Hillis-Steele doubling scan
        (``O(log T)`` depth); ``"sequential"`` uses the naive loop and is useful
        as a slow but obviously-correct reference to cross-validate against.

    Returns
    -------
    ``h`` with shape ``[..., T, R, 2]``.
    """
    if mode == "sequential":
        return sequential_scan(a, w)
    if mode != "parallel":
        raise ValueError(f"unknown scan mode: {mode!r}")

    t_len = a.shape[-3]
    # move the scan axis (T) to the front: [T, ...]
    a_cum = jnp.moveaxis(a, -3, 0)
    w_cum = jnp.moveaxis(w, -3, 0)

    offset = 1
    while offset < t_len:
        num_pairs = t_len - offset
        # left operands  = cumulants at t < num_pairs   (indices 0 .. num_pairs-1)
        # right operands = cumulants at t >= offset     (indices offset .. T-1)
        a_left = a_cum[:num_pairs]
        w_left = w_cum[:num_pairs]
        a_right = a_cum[offset:]
        w_right = w_cum[offset:]
        # (A_r A_l, A_r w_l + w_r)
        a_new = a_right @ a_left
        w_new = jnp.einsum("...ij,...dj->...di", a_right, w_left) + w_right
        a_cum = jnp.concatenate([a_cum[:offset], a_new], axis=0)
        w_cum = jnp.concatenate([w_cum[:offset], w_new], axis=0)
        offset *= 2
    # back to the original layout: [..., T, R, 2]
    return jnp.moveaxis(w_cum, 0, -3)


# --------------------------------------------------------------------------- #
# state containers
# --------------------------------------------------------------------------- #
class SsmState(NamedTuple):
    """Streaming state of a single :class:`SsmBlock`."""

    h: jax.Array
    """Hidden state, ``[B, n_heads, d_state, d_head_mimo]``."""

    prev_bx: jax.Array
    """``B_{t} x_{t}`` of the previous step, same shape as ``h``."""

    conv_state: jax.Array | None = None
    """Tail of the causal convolution, ``[B, d_inner, K-1]`` (or ``None``)."""


class MultiScaleSsmState(NamedTuple):
    """Streaming state of a :class:`MultiScaleSsmBlock` (one entry per layer)."""

    h: list[jax.Array]
    prev_bx: list[jax.Array]
    conv_state: list[jax.Array | None]


# --------------------------------------------------------------------------- #
# single block
# --------------------------------------------------------------------------- #
class SsmBlock(nnx.Module):
    """One selective SSM block with complex rotation dynamics.

    Parameters
    ----------
    config:
        A :class:`~ssm_latent.config.SsmConfig`.
    rngs:
        Flax NNX RNG collection.  Defaults to ``nnx.Rngs(0)``.
    dtype:
        Parameter dtype (``jnp.float32`` by default).
    """

    def __init__(self, config: SsmConfig, rngs: nnx.Rngs | None = None, *, dtype=jnp.float32):
        config.validate()
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.dtype = dtype

        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.mimo_rank = config.mimo_rank
        self.d_inner = config.d_inner
        self.d_head = config.d_head
        self.d_head_mimo = config.d_head_mimo

        # projections ---------------------------------------------------------
        self.in_proj = nnx.Linear(config.d_model, 2 * self.d_inner, rngs=rngs, param_dtype=dtype)
        self.out_proj = nnx.Linear(self.d_inner, config.d_model, rngs=rngs, param_dtype=dtype)
        self.dt_proj = nnx.Linear(
            self.d_inner, config.n_heads, use_bias=False, rngs=rngs, param_dtype=dtype
        )
        self.dt_bias = nnx.Param(jnp.full((config.n_heads,), -3.1, dtype=dtype))
        self.lambda_proj = nnx.Linear(self.d_inner, config.n_heads, rngs=rngs, param_dtype=dtype)
        self.theta_proj = nnx.Linear(
            self.d_inner, config.n_heads * (config.d_state // 2), rngs=rngs, param_dtype=dtype
        )
        self.b_proj = nnx.Linear(
            self.d_inner,
            config.n_heads * config.mimo_rank * config.d_state,
            rngs=rngs,
            param_dtype=dtype,
        )
        self.c_proj = nnx.Linear(
            self.d_inner,
            config.n_heads * config.mimo_rank * config.d_state,
            rngs=rngs,
            param_dtype=dtype,
        )

        # BCNorm (QK normalisation, Mamba-3 sec. 3.4) ---------------------------
        bc_dim = config.n_heads * config.mimo_rank * config.d_state
        self.b_norm = nnx.RMSNorm(bc_dim, epsilon=_RMS_EPS, param_dtype=dtype, rngs=rngs)
        self.c_norm = nnx.RMSNorm(bc_dim, epsilon=_RMS_EPS, param_dtype=dtype, rngs=rngs)
        self.b_bias = nnx.Param(
            jnp.zeros((config.n_heads, config.mimo_rank, config.d_state), dtype)
        )
        self.c_bias = nnx.Param(
            jnp.zeros((config.n_heads, config.mimo_rank, config.d_state), dtype)
        )

        # state transition -----------------------------------------------------
        rng = rngs.params()
        self.a_re = nnx.Param(
            jax.random.uniform(
                rng,
                (config.n_heads, config.d_state // 2),
                dtype,
                minval=-1.0,
                maxval=-0.1,
            )
        )
        self.a_im = nnx.Param(
            jax.random.uniform(
                rng,
                (config.n_heads, config.d_state // 2),
                dtype,
                minval=0.0,
                maxval=2.0 * jnp.pi,
            )
        )

        self.d = nnx.Param(jnp.ones((self.d_inner,), dtype))
        self.norm = nnx.RMSNorm(self.d_inner, epsilon=_RMS_EPS, param_dtype=dtype, rngs=rngs)

        # optional causal convolution -----------------------------------------
        if config.use_conv:
            kernel = jax.random.normal(
                rng,
                (self.d_inner, 1, config.conv_kernel),
                dtype,
            ) / jnp.sqrt(config.conv_kernel)
            self.conv_weight = nnx.Param(kernel)
            self.conv_bias = nnx.Param(jnp.zeros((self.d_inner,), dtype))
            self.conv_kernel = config.conv_kernel
        else:
            self.conv_weight = None
            self.conv_bias = None
            self.conv_kernel = 0

    #: Optional causal-convolution parameters (``None`` when ``use_conv=False``).
    conv_weight: nnx.Param | None
    conv_bias: nnx.Param | None

    # ------------------------------------------------------------------ utils
    def _bc(self, u_silu: jax.Array):
        """BCNorm + biases for B and C, shaped ``[B, T, H, R, N]``."""
        b, c = self.b_proj(u_silu), self.c_proj(u_silu)
        bc_dim = self.n_heads * self.mimo_rank * self.d_state
        b = self.b_norm(b.reshape(*b.shape[:-1], bc_dim))
        c = self.c_norm(c.reshape(*c.shape[:-1], bc_dim))
        shape = (*b.shape[:-1], self.n_heads, self.mimo_rank, self.d_state)
        b = b.reshape(shape) + self.b_bias[...][None, None]
        c = c.reshape(shape) + self.c_bias[...][None, None]
        return b, c

    def _in_proj(self, x: jax.Array):
        """``in_proj`` + optional causal conv + SiLU. Returns ``(u_silu, evo_gate)``."""
        projected = self.in_proj(x)
        u, evo_gate = jnp.split(projected, 2, axis=-1)
        if self.conv_weight is not None:
            assert self.conv_bias is not None  # set together with conv_weight
            u = causal_conv1d(u, self.conv_weight[...], self.conv_bias[...])
        return jax.nn.silu(u), evo_gate

    def _gate_times(self, u_silu: jax.Array):
        """``(delta, lambda, theta)`` - all input dependent."""

        def _dt(x):
            return jax.nn.softplus(self.dt_proj(x) + self.dt_bias[...])

        delta = _dt(u_silu)
        lam = jax.nn.sigmoid(self.lambda_proj(u_silu))
        theta = self.theta_proj(u_silu)
        return delta, lam, theta

    # ------------------------------------------------------------- forward
    def __call__(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        """Parallel-scan forward pass for a whole sequence.

        ``x``: ``[B, T, d_model]`` -> ``[B, T, d_model]``.
        """
        return self.forward(x, scan_mode=scan_mode)

    def forward(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        x = jnp.asarray(x)
        u_silu, evo_gate = self._in_proj(x)
        delta, lam, theta = self._gate_times(u_silu)
        b, c = self._bc(u_silu)
        y_ssm = self._selective_scan(u_silu, delta, lam, theta, b, c, scan_mode=scan_mode)
        y_res = y_ssm + u_silu * self.d[...][None, None]
        y = self.norm(y_res)
        y = y * jax.nn.silu(evo_gate)
        return self.out_proj(y)

    def _selective_scan(
        self,
        u: jax.Array,
        delta: jax.Array,
        lam: jax.Array,
        theta: jax.Array,
        b: jax.Array,
        c: jax.Array,
        *,
        scan_mode: str = "parallel",
    ) -> jax.Array:
        """Core selective scan (parallel-scan variant)."""
        batch, seq_len, _ = u.shape
        h_heads, d_state, mimo_rank = self.n_heads, self.d_state, self.mimo_rank
        half = d_state // 2

        # --- per-head step size, gate and rotation ---------------------------
        # The rotation angle is defined head-wise (`dt_proj` and `theta_proj`
        # both emit n_heads values), so `da_re`, `da_im` and `theta` share the
        # same [B, T, H, half] shape.
        dt = delta[:, :, :, None]  # [B, T, H, 1]
        da_re = jnp.exp(self.a_re[...][None, None] * dt)  # [B, T, H, half]
        da_im = self.a_im[...][None, None] * dt  # [B, T, H, half]
        angle = da_im + theta.reshape(batch, seq_len, h_heads, half)

        cos, sin = jnp.cos(angle), jnp.sin(angle)
        a00 = da_re * cos
        a01 = -da_re * sin
        a10 = da_re * sin
        a11 = da_re * cos  # [B, T, H, half]
        a = jnp.stack([jnp.stack([a00, a01], -1), jnp.stack([a10, a11], -1)], -2)

        # --- B x and the trapezoidal weights ---------------------------------
        u_mimo = u.reshape(batch, seq_len, h_heads, mimo_rank, self.d_head_mimo)
        bx = jnp.einsum("bthrn,bthrd->bthnd", b, u_mimo)  # [B, T, H, N, D]
        bx_prev = jnp.concatenate([jnp.zeros_like(bx[:, :1]), bx[:, :-1]], axis=1)
        gamma = (delta * lam)[:, :, :, None, None]
        beta = (delta * (1.0 - lam))[:, :, :, None, None] * jnp.concatenate(
            [da_re, da_re], axis=-1
        )[..., None]
        w = gamma * bx + beta * bx_prev  # [B, T, H, N, D]

        # --- affine scan in the complex (2 x 2 real block) basis -------------
        h = self._affine_scan(a, w, scan_mode=scan_mode)  # [B, T, H, N, D] complex

        return jnp.einsum("bthrn,bthnd->bthrd", c, h).reshape(batch, seq_len, self.d_inner)

    def _affine_scan(
        self,
        a: jax.Array,
        w: jax.Array,
        *,
        scan_mode: str = "parallel",
    ) -> jax.Array:
        """Scan ``h_t = a_t h_{t-1} + w_t`` for complex ``h`` stored as 2 reals.

        ``a``: ``[B, T, H, half, 2, 2]``, ``w``: ``[B, T, H, N, D]`` real with the
        real/imaginary decomposition of ``w`` taken along the state axis (the
        ``N = d_state`` axis is split into real/imaginary halves).
        """
        batch, seq_len, h_heads, half = a.shape[:4]
        n_states, d_mimo = w.shape[-2], w.shape[-1]

        w_re, w_im = jnp.split(w, 2, axis=-2)  # [B, T, H, half, D]
        w_cplx = jnp.stack([w_re, w_im], axis=-1)  # [B, T, H, half, D, 2]

        a_grp = jnp.reshape(
            jnp.transpose(a, (0, 2, 3, 1, 4, 5)), (batch * h_heads * half, seq_len, 2, 2)
        )
        w_grp = jnp.reshape(
            jnp.transpose(w_cplx, (0, 2, 3, 1, 4, 5)),
            (batch * h_heads * half, seq_len, d_mimo, 2),
        )

        h_grp = parallel_scan(a_grp, w_grp, mode=scan_mode)  # [G, T, D, 2]
        h_grp = jnp.transpose(h_grp, (1, 0, 2, 3))  # [T, G, D, 2]
        # regroup into the *block* state layout used everywhere else:
        # state index 0..N/2-1 = real part, N/2..N-1 = imaginary part.
        h = jnp.reshape(h_grp, (seq_len, batch, h_heads, half, d_mimo, 2))
        h = jnp.transpose(h, (0, 1, 2, 5, 3, 4))  # [T, B, H, 2, half, D]
        h = jnp.reshape(h, (seq_len, batch, h_heads, n_states, d_mimo))
        return jnp.transpose(h, (1, 0, 2, 3, 4))  # [B, T, H, N, D]

    # ------------------------------------------------------------ streaming
    def zero_state(self, batch: int, dtype=None) -> SsmState:
        """Zero-initialised :class:`SsmState` for streaming inference."""
        dtype = dtype or self.dtype
        h = jnp.zeros((batch, self.n_heads, self.d_state, self.d_head_mimo), dtype=dtype)
        conv_state = (
            jnp.zeros((batch, self.d_inner, self.conv_kernel - 1), dtype=dtype)
            if self.conv_weight is not None
            else None
        )
        return SsmState(h=h, prev_bx=jnp.zeros_like(h), conv_state=conv_state)

    def step(self, x: jax.Array, state: SsmState) -> tuple[jax.Array, SsmState]:
        """Single-timestep update (``O(1)`` in sequence length).

        ``x``: ``[B, d_model]``; returns ``(y, new_state)``.
        """
        x = jnp.asarray(x)
        batch = x.shape[0]

        # --- input projection + causal conv (stateful) ------------------------
        projected = self.in_proj(x)
        u, evo_gate = jnp.split(projected, 2, axis=-1)
        next_conv_state: jax.Array | None = None
        if self.conv_weight is not None:
            assert state.conv_state is not None  # guaranteed by use_conv=True
            assert self.conv_bias is not None  # set together with conv_weight
            buf = jnp.concatenate([state.conv_state, u[:, :, None]], axis=-1)
            kernel = self.conv_weight.shape[-1]
            out = jnp.zeros((batch, self.d_inner), dtype=u.dtype)
            for k in range(kernel):
                out = out + buf[:, :, k] * self.conv_weight[...][:, 0, k]
            u = out + self.conv_bias[...]
            next_conv_state = buf[:, :, 1:]
        u_silu = jax.nn.silu(u)

        delta, lam, theta = self._gate_times(u_silu)
        half = self.d_state // 2

        dt = delta[:, :, None]  # [B, H, 1]
        da_re = jnp.exp(self.a_re[...][None] * dt)  # [B, H, half]
        da_im = self.a_im[...][None] * dt
        angle = da_im + theta.reshape(batch, self.n_heads, half)
        cos, sin = jnp.cos(angle)[..., None], jnp.sin(angle)[..., None]

        h_re, h_im = jnp.split(state.h, 2, axis=2)  # [B, H, half, D]
        h_rot = jnp.concatenate([h_re * cos - h_im * sin, h_re * sin + h_im * cos], axis=2)

        b_flat, c_flat = self._bc(u_silu[:, None])  # [B, 1, H, R, N]
        b_t, c_t = b_flat[:, 0], c_flat[:, 0]  # [B, H, R, N]
        u_mimo = u_silu.reshape(batch, self.n_heads, self.mimo_rank, self.d_head_mimo)
        current_bx = jnp.einsum("bhrn,bhrd->bhnd", b_t, u_mimo)  # [B, H, N, D]

        # --- apply damping to the stored state -------------------------------
        da_re_full = jnp.concatenate([da_re, da_re], axis=2)[..., None]  # [B, H, N, 1]
        gamma = (delta * lam)[:, :, None, None]
        beta = (delta * (1.0 - lam))[:, :, None, None] * da_re_full

        prev_bx = state.prev_bx
        if prev_bx is None:  # first step: beta term is zero by construction
            prev_bx = jnp.zeros_like(current_bx)
        h_next = da_re_full * h_rot + gamma * current_bx + beta * prev_bx

        y_ssm = jnp.einsum("bhrn,bhnd->bhrd", c_t, h_next).reshape(batch, self.d_inner)
        y_res = y_ssm + u_silu * self.d[...][None]
        y = self.norm(y_res) * jax.nn.silu(evo_gate)
        return self.out_proj(y), SsmState(h=h_next, prev_bx=current_bx, conv_state=next_conv_state)

    def scan_sequential(self, x: jax.Array) -> jax.Array:
        """Step-by-step forward, used by the equivalence tests."""
        batch, seq_len, _ = x.shape
        state = self.zero_state(batch, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            y, state = self.step(x[:, t], state)
            ys.append(y)
        return jnp.stack(ys, axis=1)


# --------------------------------------------------------------------------- #
# multi-scale stack
# --------------------------------------------------------------------------- #
class MultiScaleSsmBlock(nnx.Module):
    """``n_layers`` SSM blocks with per-layer timescale initialisation.

    Each layer is wrapped in a pre-LayerNorm residual branch whose contribution
    is scaled by ``1 / n_layers`` and clamped to ``[-100, 100]`` for numerical
    stability.
    """

    def __init__(
        self, config: MultiScaleSsmConfig, rngs: nnx.Rngs | None = None, *, dtype=jnp.float32
    ):
        config.validate()
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.dtype = dtype
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.n_heads = config.n_heads
        self.mimo_rank = config.mimo_rank

        self.layers: nnx.List = nnx.List([])
        self.norms: nnx.List = nnx.List([])
        base = SsmConfig(
            d_model=config.d_model,
            d_state=config.d_state,
            expand=config.expand,
            n_heads=config.n_heads,
            mimo_rank=config.mimo_rank,
            use_conv=config.use_conv,
            conv_kernel=config.conv_kernel,
        )
        for layer_idx in range(config.n_layers):
            layer_rngs = nnx.Rngs(rngs.params())
            block = SsmBlock(
                base.with_(use_conv=config.use_conv and layer_idx == 0),
                layer_rngs,
                dtype=dtype,
            )
            lo, hi = config.timescale_range(layer_idx)
            block.a_re = nnx.Param(
                jax.random.uniform(
                    layer_rngs.params(),
                    (config.n_heads, config.d_state // 2),
                    dtype,
                    minval=lo,
                    maxval=hi,
                )
            )
            self.layers.append(block)
            self.norms.append(
                nnx.RMSNorm(config.d_model, epsilon=_RMS_EPS, param_dtype=dtype, rngs=layer_rngs)
            )

    # ------------------------------------------------------------- forward
    def __call__(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        return self.forward(x, scan_mode=scan_mode)

    def forward(self, x: jax.Array, *, scan_mode: str = "parallel") -> jax.Array:
        scale = 1.0 / self.n_layers
        out = x
        for layer, norm in zip(self.layers, self.norms, strict=True):
            residual = out
            y = layer(norm(out), scan_mode=scan_mode)
            out = jnp.clip(y * scale + residual, -100.0, 100.0)
        return out

    def zero_state(self, batch: int, dtype=None) -> MultiScaleSsmState:
        dtype = dtype or self.dtype
        states = [layer.zero_state(batch, dtype) for layer in self.layers]
        return MultiScaleSsmState(
            h=[s.h for s in states],
            prev_bx=[s.prev_bx for s in states],
            conv_state=[s.conv_state for s in states],
        )
        # the first step is identical whether "no previous Bx yet" is zeros or
        # None, since beta * 0 = 0.

    def step(self, x: jax.Array, state: MultiScaleSsmState) -> tuple[jax.Array, MultiScaleSsmState]:
        """One timestep through every layer of the stack."""
        scale = 1.0 / self.n_layers
        out = x
        new_h, new_bx, new_conv = [], [], []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms, strict=True)):
            residual = out
            y, st = layer.step(
                norm(out),
                SsmState(h=state.h[i], prev_bx=state.prev_bx[i], conv_state=state.conv_state[i]),
            )
            out = jnp.clip(y * scale + residual, -100.0, 100.0)
            new_h.append(st.h)
            new_bx.append(st.prev_bx)
            new_conv.append(st.conv_state)
        return out, MultiScaleSsmState(h=new_h, prev_bx=new_bx, conv_state=new_conv)

    def scan_sequential(self, x: jax.Array) -> jax.Array:
        batch, seq_len, _ = x.shape
        state = self.zero_state(batch, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            y, state = self.step(x[:, t], state)
            ys.append(y)
        return jnp.stack(ys, axis=1)
