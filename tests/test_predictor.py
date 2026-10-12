"""Pure-Mamba predictors, streaming state and the closed-loop imagination paths."""

import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import assert_close
from flax import nnx

from ssm_latent import MultiScaleSsmConfig, SsmConfig
from ssm_latent.predictor import MambaPredictor, MultiScaleMambaPredictor
from ssm_latent.utils import count_parameters


# --------------------------------------------------------------------------- #
# MambaPredictor
# --------------------------------------------------------------------------- #
def test_mamba_predictor_shapes():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, input_dim=3, output_dim=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 7, 3))
    assert model(x).shape == (2, 7, 2)
    assert model.scan_sequential(x).shape == (2, 7, 2)


def test_mamba_predictor_forward_equals_step_loop():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, 3, 2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 3))
    assert_close(model(x), model.scan_sequential(x), atol=1e-5)


def test_mamba_predictor_step_advances_state():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, 3, 2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3))
    state = model.zero_state(2)
    assert float(jnp.max(jnp.abs(state.h))) == 0.0
    pred, state1 = model.step(x, state)
    assert pred.shape == (2, 2)
    assert float(jnp.max(jnp.abs(state1.h))) > 0.0
    assert state1.conv_state is not None
    pred2, state2 = model.step(x, state1)
    assert float(jnp.max(jnp.abs(state2.h - state1.h))) > 0.0
    assert not jnp.allclose(pred, pred2), "different history -> different prediction"


def test_mamba_predictor_output_depends_on_input():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    a = model(jnp.zeros((1, 4, 2)))
    b = model(jnp.ones((1, 4, 2)) * 5.0)
    assert float(jnp.max(jnp.abs(a - b))) > 1e-3


def test_mamba_predictor_loss_alignment():
    """``predictions[t]`` is compared against ``targets[t+1]``."""
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    targets = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 2))
    # predictions[t] must equal targets[t + 1]
    perfect = jnp.concatenate([targets[:, 1:], targets[:, -1:]], axis=1)
    assert_close(MambaPredictor.loss(perfect, targets), 0.0, atol=1e-6)
    assert float(MambaPredictor.loss(-perfect, targets)) > 0.0


def test_mamba_predictor_training_reduces_loss():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)
    targets = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 2))

    @nnx.jit
    def step(model, optimizer, targets):
        def loss_fn(model):
            return MambaPredictor.loss(model(targets), targets)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    first = float(step(model, optimizer, targets))
    for _ in range(40):
        last = float(step(model, optimizer, targets))
    assert last < first, f"{first} -> {last}"


def test_predictor_loss_on_a_single_step_is_zero_not_nan():
    """One step means no prediction/target pair -> the loss is 0, not NaN."""
    one = jnp.ones((2, 1, 2))
    assert float(MambaPredictor.loss(one, one)) == 0.0
    assert float(MultiScaleMambaPredictor.loss(one, one)) == 0.0


def test_mamba_predictor_backward_flow():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 2))
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)
    pure = nnx.to_pure_dict(nnx.state(grads, nnx.Param))
    # the conv weights of the first layer must be reached too
    assert float(jnp.max(jnp.abs(pure["ssm"]["conv_weight"]))) > 0.0


def test_mamba_predictor_save_load(tmp_path):
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2))
    expected = model(x)
    path = model.save(tmp_path / "mamba")
    assert path.endswith(".msgpack")
    other = MambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(9)).load(path)
    assert_close(other(x), expected, atol=1e-6)


def test_mamba_predictor_parameter_count_is_small():
    """~20K-50K parameters for the reference anomaly-detection config."""
    cfg = SsmConfig(d_model=64, d_state=16, expand=2, n_heads=4, mimo_rank=1)
    model = MambaPredictor(cfg, 1, 1, rngs=nnx.Rngs(0))
    n = count_parameters(model)
    assert 15_000 < n < 60_000, n


# --------------------------------------------------------------------------- #
# MultiScaleMambaPredictor
# --------------------------------------------------------------------------- #
def _ms_model(action_dim=2, **overrides):
    cfg = MultiScaleSsmConfig(
        d_model=8,
        d_state=4,
        expand=2,
        n_heads=2,
        mimo_rank=1,
        n_layers=3,
        use_conv=True,
        **overrides,
    )
    return MultiScaleMambaPredictor(cfg, 2, action_dim, rngs=nnx.Rngs(0))


def test_multiscale_predictor_forward_shapes():
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    assert model(x).shape == (2, 6, 2)
    assert model.forward_with_action(x, jnp.zeros((2, 6, 2))).shape == (2, 6, 2)


def test_multiscale_predictor_step_matches_forward():
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    state = model.zero_state(2)
    preds = []
    for t in range(6):
        pred, state = model.step(x[:, t], state)
        preds.append(pred)
    assert_close(model(x), jnp.stack(preds, axis=1), atol=1e-5)


def test_multiscale_predictor_zero_action_is_the_default_path():
    """`forward` (obs only) must equal `forward_with_action` with zero actions."""
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    assert_close(model(x), model.forward_with_action(x, jnp.zeros((2, 6, 2))), atol=1e-6)


def test_multiscale_predictor_imagine_loop_runs_and_stays_finite():
    model = _ms_model()
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 1, 2))
    act = jnp.zeros((2, 1, 2))
    # `step_imagine` is fed the SSM output (d_model), not the decoded prediction
    u = model.imagine_fusion(jnp.concatenate([model.obs_proj(obs), model.action_proj(act)], -1))
    y = model.ssms(u)[:, 0]
    state = model.zero_state(2)
    preds = []
    for _ in range(8):
        pred, y, state = model.step_imagine(y, jnp.zeros((2, 2)), state)
        preds.append(pred)
    preds = jnp.stack(preds)
    assert preds.shape == (8, 2, 2)
    assert bool(jnp.all(jnp.isfinite(preds)))


def test_multiscale_predictor_imagine_h_loop_runs_and_stays_finite():
    model = _ms_model()
    state = model.zero_state(2)
    preds = []
    for _ in range(5):
        pred, y, state = model.step_imagine_h(state, jnp.zeros((2, 2)))
        preds.append(pred)
    preds = jnp.stack(preds)
    assert preds.shape == (5, 2, 2)
    assert bool(jnp.all(jnp.isfinite(preds)))


def test_multiscale_predictor_h_sampling_prob_zero_equals_action_forward():
    model = _ms_model()
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    act = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 2))
    assert_close(
        model.forward_with_h_sampling(obs, act, 0.0),
        model.forward_with_action(obs, act),
        atol=1e-6,
    )


def test_multiscale_predictor_h_sampling_positive_is_deterministic_given_key():
    model = _ms_model()
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    act = jnp.zeros((2, 6, 2))
    a = model.forward_with_h_sampling(obs, act, 0.5, key=jax.random.PRNGKey(3))
    b = model.forward_with_h_sampling(obs, act, 0.5, key=jax.random.PRNGKey(3))
    c = model.forward_with_h_sampling(obs, act, 0.5, key=jax.random.PRNGKey(4))
    assert_close(a, b, atol=0.0)
    assert not jnp.allclose(a, c), "different keys must sample different h/obs mixes"


def test_multiscale_predictor_h_sampling_prob_one_differs_from_prob_zero():
    model = _ms_model()
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 2))
    act = jnp.zeros((2, 6, 2))
    never = model.forward_with_h_sampling(obs, act, 0.0)
    always = model.forward_with_h_sampling(obs, act, 1.0, key=jax.random.PRNGKey(0))
    # the first step always uses the observation, later steps use h
    assert_close(never[:, :1], always[:, :1], atol=1e-6)
    assert not jnp.allclose(never[:, 1:], always[:, 1:])


def test_multiscale_predictor_multi_batch_independence():
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 5, 2))
    y = model(x)
    assert_close(y[2:3], model(x[2:3]), atol=1e-6)


def test_multiscale_predictor_state_shape():
    model = _ms_model()
    state = model.zero_state(4)
    assert len(state.h) == 3
    assert state.h[0].shape == (4, 2, 4, model.config.d_head_mimo)
    assert state.conv_state[0] is not None and state.conv_state[1] is None


def test_multiscale_predictor_h_flat_dim_matches_state():
    """`h_proj` must accept exactly the flattened hidden state of the last layer."""
    model = _ms_model()
    state = model.zero_state(3)
    flat = state.h[-1].reshape(3, -1)
    assert flat.shape[1] == model.h_flat_dim
    assert model.h_proj.kernel[...].shape[0] == model.h_flat_dim
    proj = model.h_proj(flat)
    assert proj.shape == (3, model.d_model)


def test_multiscale_predictor_save_load(tmp_path):
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 2))
    expected = model(x)
    path = model.save(tmp_path / "ms-mamba")
    other = MultiScaleMambaPredictor(
        MultiScaleSsmConfig(
            d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
        ),
        2,
        2,
        rngs=nnx.Rngs(5),
    ).load(path)
    assert_close(other(x), expected, atol=1e-6)


def test_multiscale_predictor_imagine_rejects_a_decoded_prediction():
    """A wide-mismatch guard: `step_imagine` must not be fed `output_proj` outputs."""
    model = _ms_model()
    with pytest.raises(ValueError, match="SSM output"):
        model.step_imagine(jnp.zeros((2, 2)), jnp.zeros((2, 2)), model.zero_state(2))


def test_multiscale_predictor_loss_alignment():
    _ms_model()
    targets = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 2))
    perfect = jnp.concatenate([targets[:, 1:], targets[:, -1:]], axis=1)
    assert_close(MultiScaleMambaPredictor.loss(perfect, targets), 0.0, atol=1e-6)


def test_multiscale_predictor_backward_flow():
    model = _ms_model()
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 2))
    grads = nnx.grad(lambda m: jnp.mean(m(x) ** 2))(model)
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert leaves and all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_multiscale_predictor_supports_other_stack_depths(n_layers):
    cfg = MultiScaleSsmConfig(
        d_model=8,
        d_state=4,
        expand=2,
        n_heads=2,
        mimo_rank=1,
        n_layers=n_layers,
        use_conv=True,
    )
    model = MultiScaleMambaPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, 2))
    y = model(x)
    assert y.shape == (1, 4, 2)
    # step-by-step execution must reproduce the batched forward pass
    state = model.zero_state(1)
    stepwise = []
    for t in range(4):
        pred, state = model.step(x[:, t], state)
        stepwise.append(pred)
    assert_close(y, jnp.stack(stepwise, axis=1), atol=1e-5)
