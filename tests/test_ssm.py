"""Equivalence guarantees for the selective SSM.

The central invariant (verified in Rust by `tests/equivalence_test.rs`) is that
the parallel-scan forward pass, the parallel scan driven by the sequential
`step`, and the step-by-step loop all agree to float32 precision - with and
without the causal convolution.
"""

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_close
from flax import nnx

from ssm_latent import MultiScaleSsmBlock, MultiScaleSsmConfig, SsmBlock, SsmConfig
from ssm_latent.ssm import causal_conv1d, parallel_scan, sequential_scan


# --------------------------------------------------------------------------- #
# forward / step equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_conv", [False, True])
@pytest.mark.parametrize("n_heads,mimo_rank", [(1, 1), (2, 1), (2, 2), (4, 2)])
def test_forward_step_equivalence(use_conv, n_heads, mimo_rank):
    cfg = SsmConfig(
        d_model=8,
        d_state=4,
        expand=2,
        n_heads=n_heads,
        mimo_rank=mimo_rank,
        use_conv=use_conv,
        conv_kernel=4,
    )
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 7, cfg.d_model))

    y_parallel = block.forward(x, scan_mode="parallel")
    y_scan = block.forward(x, scan_mode="sequential")
    y_step = block.scan_sequential(x)

    assert_close(y_parallel, y_scan, atol=1e-5, msg="parallel vs sequential scan")
    assert_close(y_parallel, y_step, atol=1e-5, msg="parallel vs step loop")


@pytest.mark.parametrize("batch,seq_len", [(1, 1), (1, 2), (3, 5), (2, 16)])
def test_shapes_and_dtypes(batch, seq_len):
    cfg = SsmConfig(d_model=6, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq_len, 6))
    y = block(x)
    assert y.shape == (batch, seq_len, 6)
    assert y.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(y)))


def test_state_shape_matches_reference_formula():
    cfg = SsmConfig(
        d_model=12, d_state=6, expand=2, n_heads=3, mimo_rank=2, use_conv=True, conv_kernel=5
    )
    block = SsmBlock(cfg, nnx.Rngs(0))
    state = block.zero_state(batch=4)
    # [batch, n_heads, d_state, d_head / mimo_rank]
    assert state.h.shape == (4, 3, 6, 4)
    assert state.conv_state.shape == (4, 24, 4)
    assert float(jnp.max(jnp.abs(state.h))) == 0.0


def test_no_conv_block_has_no_conv_state():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=False)
    block = SsmBlock(cfg, nnx.Rngs(0))
    assert block.conv_weight is None
    assert block.zero_state(2).conv_state is None


def test_batches_are_independent():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 6, 8))
    y = block(x)
    y0 = block(x[:1])
    assert_close(y[:1], y0, atol=1e-6, msg="batch independence")


# --------------------------------------------------------------------------- #
# scan primitive itself
# --------------------------------------------------------------------------- #
def test_parallel_scan_matches_sequential_reference():
    key = jax.random.PRNGKey(7)
    batch, seq_len, n_heads, half, d_mimo = 2, 9, 3, 2, 2
    k1, k2 = jax.random.split(key)
    angle = jax.random.normal(k1, (batch, seq_len, n_heads, half))
    r = jnp.abs(jax.random.normal(k2, (batch, seq_len, n_heads, half))) + 0.5
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    a = jnp.stack([jnp.stack([r * cos, -r * sin], -1), jnp.stack([r * sin, r * cos], -1)], -2)
    w = jax.random.normal(jax.random.PRNGKey(11), (batch, seq_len, n_heads, 2 * half, d_mimo))

    # regroup into the layout expected by the scan helper
    a_grp = jnp.reshape(
        jnp.transpose(a, (0, 2, 3, 1, 4, 5)), (batch * n_heads * half, seq_len, 2, 2)
    )
    w_cplx = jnp.stack(jnp.split(w, 2, axis=-2), axis=-1)
    w_grp = jnp.reshape(
        jnp.transpose(w_cplx, (0, 2, 3, 1, 4, 5)), (batch * n_heads * half, seq_len, d_mimo, 2)
    )
    assert_close(parallel_scan(a_grp, w_grp), sequential_scan(a_grp, w_grp), atol=1e-6)


def test_parallel_scan_rejects_unknown_mode():
    a = jnp.zeros((1, 3, 2, 2))
    w = jnp.zeros((1, 3, 1, 2))
    with pytest.raises(ValueError, match="unknown scan mode"):
        parallel_scan(a, w, mode="nope")


def test_parallel_scan_handles_odd_sequence_lengths():
    a = jnp.zeros((1, 5, 2, 2)) + jnp.eye(2) * 0.5
    w = jnp.ones((1, 5, 1, 2))
    h = parallel_scan(a, w)
    # h_t = 0.5 * h_{t-1} + 1 with h_{-1} = 0  ->  1, 1.5, 1.75, 1.875, ...
    expected = jnp.cumsum(0.5 ** jnp.arange(0, 5))
    assert_close(h[0, :, 0, 0], expected, atol=1e-6)


def test_causal_conv1d_matches_manual_convolution():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 6, 3))
    weight = jax.random.normal(jax.random.PRNGKey(1), (3, 1, 4))
    bias = jnp.arange(3.0)
    y = causal_conv1d(x, weight, bias)
    assert y.shape == x.shape
    for b in range(2):
        for t in range(6):
            for c in range(3):
                acc = bias[c]
                for k in range(4):
                    src = t + k - 3  # zero-padded window
                    if src >= 0:
                        acc = acc + x[b, src, c] * weight[c, 0, k]
                assert_close(y[b, t, c], acc, atol=1e-5)


def test_causal_conv1d_is_causal():
    """Changing x[t] must not affect y[<t]."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1, 8, 2))
    weight = jax.random.normal(key, (2, 1, 4))
    y = causal_conv1d(x, weight)
    x2 = x.at[0, 5, :].set(x[0, 5, :] + 5.0)
    y2 = causal_conv1d(x2, weight)
    assert float(jnp.max(jnp.abs(y[0, :5] - y2[0, :5]))) < 1e-6
    assert float(jnp.max(jnp.abs(y[0, 5:] - y2[0, 5:]))) > 1e-3


# --------------------------------------------------------------------------- #
# numerics / initialisation
# --------------------------------------------------------------------------- #
def test_a_re_is_damped_and_a_im_is_a_rotation():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    block = SsmBlock(cfg, nnx.Rngs(0))
    assert bool(jnp.all(block.a_re[...] < 0.0)), "a_re must be strictly negative (damping)"
    assert bool(jnp.all(block.a_re[...] >= -1.0))
    assert bool(jnp.all(block.a_im[...] >= 0.0))
    assert bool(jnp.all(block.a_im[...] <= 2.0 * jnp.pi))


def test_theta_and_rotation_change_the_dynamics():
    """Two blocks differing only in `theta_proj` weights must give different outputs."""
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    block_a = SsmBlock(cfg, nnx.Rngs(0))
    block_b = nnx.clone(block_a)
    block_b.theta_proj.kernel[...] = block_a.theta_proj.kernel[...] * 3.0
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 8))
    assert float(jnp.max(jnp.abs(block_a(x) - block_b(x)))) > 1e-4


def test_dt_bias_is_negative_so_delta_starts_small():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 8))
    u, _ = block._in_proj(x)
    delta, lam, _ = block._gate_times(u)
    assert float(jnp.max(delta)) < 0.1, "softplus(-3.1) ~ 0.045 keeps the initial step tiny"
    assert bool(jnp.all((lam > 0) & (lam < 1)))


def test_zero_input_produces_zero_state_trajectory():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=False)
    block = SsmBlock(cfg, nnx.Rngs(0))
    state = block.zero_state(2)
    x = jnp.zeros((2, 8))
    y, state = block.step(x, state)
    assert float(jnp.max(jnp.abs(state.h))) == 0.0
    assert float(jnp.max(jnp.abs(state.prev_bx))) == 0.0


def test_first_step_matches_parallel_scan_first_position():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=False)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 8))
    y_par = block.forward(x)
    y, _ = block.step(x[:, 0], block.zero_state(2))
    assert_close(y_par[:, 0], y, atol=1e-5)


# --------------------------------------------------------------------------- #
# multi-scale stack
# --------------------------------------------------------------------------- #
def test_multiscale_forward_step_equivalence():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )
    block = MultiScaleSsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(2), (2, 6, 8))
    assert_close(block.forward(x), block.forward(x, scan_mode="sequential"), atol=1e-5)
    assert_close(block.forward(x), block.scan_sequential(x), atol=1e-5)


def test_multiscale_residual_scaling_and_clamping():
    """With huge layer outputs the residual update must saturate at +/-100."""
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2)
    block = MultiScaleSsmBlock(cfg, nnx.Rngs(0))
    # scale up the output projections so the branch output explodes
    for layer in block.layers:
        layer.out_proj.kernel[...] = layer.out_proj.kernel[...] * 500.0
    x = jnp.ones((1, 4, 8)) * 3.0
    y = block(x)
    assert float(jnp.max(jnp.abs(y))) <= 100.0 + 1e-4
    assert float(jnp.max(jnp.abs(y))) > 50.0


def test_multiscale_layers_have_different_timescales():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )
    block = MultiScaleSsmBlock(cfg, nnx.Rngs(0))
    means = [float(jnp.mean(layer.a_re[...])) for layer in block.layers]
    assert means[0] < means[1] < means[2], f"expected fast < medium < slow, got {means}"
    # only the first layer owns a convolution
    assert block.layers[0].conv_weight is not None
    assert all(layer.conv_weight is None for layer in block.layers[1:])


def test_multiscale_state_is_per_layer():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )
    block = MultiScaleSsmBlock(cfg, nnx.Rngs(0))
    state = block.zero_state(2)
    assert len(state.h) == 3 and len(state.prev_bx) == 3 and len(state.conv_state) == 3
    assert state.conv_state[0] is not None and state.conv_state[1] is None


def test_multiscale_multi_batch_independence():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2, use_conv=True
    )
    block = MultiScaleSsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(9), (3, 5, 8))
    y = block(x)
    assert_close(y[1:2], block(x[1:2]), atol=1e-6)


# --------------------------------------------------------------------------- #
# jit / grad / vmap compatibility
# --------------------------------------------------------------------------- #
def test_jit_matches_eager():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 8))
    y_eager = block(x)

    @nnx.jit
    def run(block, x):
        return block(x)

    assert_close(run(block, x), y_eager, atol=1e-6)


def test_gradients_reach_every_parameter_group():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 8))

    def loss_fn(block):
        return jnp.mean(block(x) ** 2)

    grads = nnx.grad(loss_fn)(block)
    state = nnx.state(grads, nnx.Param)
    leaves = jax.tree.leaves(state)
    assert leaves, "no gradients found"
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in leaves)
    # named parameters that must receive gradients
    pure = nnx.to_pure_dict(state)
    for name in ("a_re", "a_im", "d", "b_bias", "c_bias"):
        assert name in pure["ssm"] if "ssm" in pure else name in pure
        grad = pure.get("ssm", pure)[name]
        assert bool(jnp.all(jnp.isfinite(grad)))


def test_jit_grad_and_vmap_are_composesable():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    block = SsmBlock(cfg, nnx.Rngs(0))

    @nnx.jit
    def loss(block, x):
        return jnp.mean(block(x) ** 2)

    x = jax.random.normal(jax.random.PRNGKey(0), (4, 5, 8))
    grads = nnx.grad(loss)(block, x)
    assert any(float(jnp.max(jnp.abs(g))) > 0 for g in jax.tree.leaves(nnx.state(grads, nnx.Param)))

    @nnx.jit
    def per_sample(block, x):
        return jax.vmap(lambda xi: jnp.mean(block(xi[None]) ** 2))(x)

    vals = per_sample(block, x)
    assert vals.shape == (4,)


def test_state_carrying_forward_is_jittable():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)

    @nnx.jit
    def scan_all(block, x):
        def body(state, xt):
            y, state = block.step(xt, state)
            return state, y

        init = block.zero_state(x.shape[0])
        return jax.lax.scan(body, init, jnp.swapaxes(x, 0, 1))

    block = SsmBlock(cfg, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, cfg.d_model))
    _, ys = scan_all(block, x)
    assert_close(jnp.swapaxes(ys, 0, 1), block(x), atol=1e-5)
