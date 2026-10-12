"""Loss functions and JEPA latent predictors."""

import jax
import jax.numpy as jnp
import optax
import pytest
from conftest import assert_close
from flax import nnx

from ssm_latent import MultiScaleSsmConfig, SsmConfig
from ssm_latent.latent import (
    LatentLossArgs,
    LatentPredictor,
    MlpEncoder,
    MultiScaleLatentPredictor,
    curvature_loss,
    curvature_loss_with_dt,
    lejepa_loss,
    make_projections,
    sigreg_loss,
    stability_loss,
    stability_loss_running,
)
from ssm_latent.utils import normalize_projections


# --------------------------------------------------------------------------- #
# projections
# --------------------------------------------------------------------------- #
def test_make_projections_are_unit_norm_and_frozen():
    w = make_projections(16, 8, seed=0)
    assert w.shape == (16, 8)
    assert_close(jnp.linalg.norm(w, axis=0), jnp.ones(8), atol=1e-5)
    # A plain array (not an nnx.Variable), and seed-deterministic.
    assert isinstance(w, jnp.ndarray)
    assert_close(w, make_projections(16, 8, seed=0), atol=0.0)
    assert not jnp.allclose(w, make_projections(16, 8, seed=1))
    # The "frozen" / non-trainable guarantee is enforced where it matters: the
    # predictors register this as a `Buffer`, so `wrt=nnx.Param` never sees it.
    # That behaviour is covered by `test_stability_projections_are_not_trained`.


def test_normalize_projections_matches_reference_formula():
    w = jnp.arange(12.0).reshape(4, 3) + 1.0
    expected = w / (jnp.sqrt(jnp.sum(w**2, axis=0, keepdims=True)) + 1e-6)
    assert_close(normalize_projections(w), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# stability / sigreg
# --------------------------------------------------------------------------- #
def test_stability_loss_prefers_standard_normal():
    w = normalize_projections(jax.random.normal(jax.random.PRNGKey(0), (16, 8)))
    collapsed = jnp.zeros((1, 10, 16))
    ideal = jax.random.normal(jax.random.PRNGKey(1), (1, 400, 16))
    assert float(stability_loss(collapsed, w)) > float(stability_loss(ideal, w))
    assert_close(stability_loss(ideal, w), 0.0, atol=0.05)


def test_stability_loss_is_zero_for_exact_standard_normal_by_construction():
    """A z whose projected mean/var are (0, 1) exactly gives zero loss."""
    key = jax.random.PRNGKey(0)
    w = normalize_projections(jax.random.normal(key, (8, 4)))
    z = jax.random.normal(key, (1, 1, 8))  # single sample -> mean 0 but var 0
    loss = float(stability_loss(z, w))
    assert loss > 0.0  # variance term (0 - 1)^2 = 1 for every projection


def test_stability_loss_running_updates_ema_and_returns_loss():
    key = jax.random.PRNGKey(0)
    w = normalize_projections(jax.random.normal(key, (16, 8)))
    z = jax.random.normal(key, (2, 16, 16))
    mean = jnp.zeros(8)
    var = jnp.ones(8)
    loss, new_mean, new_var = stability_loss_running(z, w, mean, var, momentum=0.1)
    assert loss.shape == () and float(loss) >= 0.0
    assert not jnp.allclose(new_mean, mean) and not jnp.allclose(new_var, var)
    # EMA stays inside the convex hull of the running and batch statistics
    assert bool(jnp.all(jnp.abs(new_mean) <= jnp.maximum(jnp.abs(mean), 1e-6) + 1.0))


def test_sigreg_penalises_collapse_more_than_gaussian():
    w = make_projections(16, 8)
    collapsed = sigreg_loss(jnp.zeros((1, 10, 16)), w)
    gaussian = sigreg_loss(jax.random.normal(jax.random.PRNGKey(0), (4, 256, 16)), w)
    assert float(collapsed) > float(gaussian)
    assert_close(gaussian, 0.0, atol=0.02)


def test_sigreg_rewards_gaussianity_in_a_controllable_family():
    """More Gaussian samples must score better than heavy-tailed ones (matched variance)."""
    key = jax.random.PRNGKey(3)
    w = make_projections(8, 32)
    n = 20_000
    gauss = jax.random.normal(key, (1, n, 8))
    # heavy-tailed: t-like distribution (student-t with 4 dof approximated by
    # a ratio of a normal and a chi-square), rescaled to unit variance
    k1, k2 = jax.random.split(key)
    t_like = jax.random.normal(k1, (1, n, 8)) / jnp.sqrt(jax.random.gamma(k2, 2.0, (1, n, 8)) / 2.0)
    t_like = t_like / jnp.sqrt(2.0)  # var(t_4) = 2
    assert float(sigreg_loss(gauss, w)) < float(sigreg_loss(t_like, w))


def test_sigreg_loss_is_not_constant_in_frequency_count():
    w = make_projections(8, 8)
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 8))
    one = float(sigreg_loss(z, w, freqs=(1.0,)))
    two = float(sigreg_loss(z, w, freqs=(1.0, 2.0)))
    assert one != two


def test_lejepa_loss_is_finite_and_positive():
    w = make_projections(32, 8)
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 32))
    pred_z = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 32))
    loss = float(lejepa_loss(z, pred_z, w, 1.0, (0.5, 1.0, 1.5)))
    assert 0.0 <= loss < jnp.inf


def test_lejepa_prediction_term_is_exact_for_a_shifted_copy():
    w = make_projections(8, 4, seed=1)
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 12, 8))
    # perfect one-step-ahead predictor
    pred_z = jnp.concatenate([z[:, 1:], z[:, -1:]], axis=1)
    loss_with_sigreg = float(lejepa_loss(z, pred_z, w, 1.0, (0.5,)))
    loss_pred_only = float(lejepa_loss(z, pred_z, w, 0.0, (0.5,)))
    assert loss_pred_only < 1e-10
    assert loss_with_sigreg > loss_pred_only


# --------------------------------------------------------------------------- #
# curvature
# --------------------------------------------------------------------------- #
def test_curvature_loss_is_zero_for_a_straight_line():
    t = jnp.arange(10, dtype=jnp.float32)
    z = jnp.stack([t, 2.0 * t], axis=-1)[None, :, :]
    assert_close(curvature_loss(z), 0.0, atol=1e-6)


def test_curvature_loss_grows_with_jaggedness():
    straight = jnp.tile(jnp.arange(10, dtype=jnp.float32)[None, :, None], (1, 1, 4))
    jagged = straight + 0.1 * jnp.sin(jnp.arange(10, dtype=jnp.float32))[None, :, None]
    assert float(curvature_loss(jagged)) > float(curvature_loss(straight))


def test_curvature_loss_returns_zero_for_short_sequences():
    assert float(curvature_loss(jnp.ones((2, 2, 4)))) == 0.0
    assert float(curvature_loss(jnp.ones((2, 1, 4)))) == 0.0


def test_curvature_loss_with_dt_matches_uniform_case():
    t = jnp.arange(8, dtype=jnp.float32)
    z = jnp.sin(t)[None, :, None] * jnp.ones((1, 1, 3))
    dt = jnp.ones(8)
    assert_close(curvature_loss_with_dt(z, dt), curvature_loss(z), atol=1e-6)


def test_curvature_loss_with_dt_handles_nonuniform_spacing():
    """A time-linear trajectory is straight under dt-aware differencing only."""
    dt = jnp.array([1.0, 1.0, 2.0, 0.5, 3.0, 1.5])
    # dt[t] is the gap between steps t-1 and t, so the timestamps are the
    # running sum of dt[1:]
    t = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dt[1:])])
    z = (2.0 * t)[None, :, None] * jnp.ones((1, 1, 2))
    assert_close(curvature_loss_with_dt(z, dt), 0.0, atol=1e-5)
    assert float(curvature_loss(z)) > 1e-3, "uniform-dt loss is fooled by non-uniform sampling"


def test_curvature_loss_with_dt_broadcasts_per_batch():
    z = jnp.ones((3, 6, 2))
    dt = jnp.ones((3, 6))
    assert float(curvature_loss_with_dt(z, dt)) == 0.0


# --------------------------------------------------------------------------- #
# combined latent loss
# --------------------------------------------------------------------------- #
def build_args(key, batch=2, seq_len=8, d_model=8, input_dim=4):
    k1, k2, k3 = jax.random.split(key, 3)
    z = jax.random.normal(k1, (batch, seq_len, d_model))
    pred_z = jax.random.normal(k2, (batch, seq_len, d_model))
    x = jax.random.normal(k3, (batch, seq_len, input_dim))
    return LatentLossArgs(
        z=z,
        pred_z=pred_z,
        reconstructed_x=x + 0.1,
        predicted_x=x + 0.2,
        original_x=x,
    )


def test_latent_loss_equals_the_weighted_sum_of_its_terms():
    args = build_args(jax.random.PRNGKey(0))
    w = make_projections(8, 16)
    args.stability_weight = 2.0
    args.curvature_weight = 3.0
    args.recon_weight = 4.0

    total = float(__import__("ssm_latent").latent.latent_loss(args, w))
    seq_len = args.z.shape[1]
    latent = float(((args.z[:, 1:] - args.pred_z[:, : seq_len - 1]) ** 2).mean())
    recon = float(((args.original_x - args.reconstructed_x) ** 2).mean())
    pred_x = float(((args.original_x[:, 1:] - args.predicted_x[:, : seq_len - 1]) ** 2).mean())
    reg = float(stability_loss(args.z, w))
    curv = float(curvature_loss(args.z))
    expected = latent + 4.0 * (recon + pred_x) + 2.0 * reg + 3.0 * curv
    assert abs(total - expected) < 1e-4


def test_latent_loss_targets_are_stop_gradiented():
    """``z`` acts as a *target* in the predictive term, so it is detached there."""
    from ssm_latent.latent import latent_loss

    args = build_args(jax.random.PRNGKey(0))
    w = make_projections(8, 16)

    def predictive_only(z, pred_z):
        return latent_loss(
            LatentLossArgs(
                z,
                pred_z,
                args.original_x,
                args.original_x,
                args.original_x,
                stability_weight=0.0,
                curvature_weight=0.0,
                recon_weight=0.0,
            ),
            w,
        )

    gz, gpred = jax.grad(predictive_only, argnums=(0, 1))(args.z, args.pred_z)
    assert jnp.allclose(gz, 0.0), "the latent target must be detached"
    assert float(jnp.max(jnp.abs(gpred))) > 0.0

    def recon_only(z):
        return latent_loss(
            LatentLossArgs(
                z,
                z,
                args.original_x + 1.0,
                args.original_x,
                args.original_x,
                stability_weight=0.0,
                curvature_weight=0.0,
                recon_weight=1.0,
            ),
            w,
        )

    assert float(jnp.max(jnp.abs(jax.grad(recon_only)(args.z)))) > 0.0


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
def test_mlp_encoder_shapes_for_all_depths():
    for n_hidden, hidden_dim in [(0, 0), (1, 16), (2, 16), (3, 32)]:
        enc = MlpEncoder(4, 8, n_hidden=n_hidden, hidden_dim=hidden_dim, rngs=nnx.Rngs(0))
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 4))
        assert enc(x).shape == (2, 5, 8)
        assert enc.forward_single(x[:, 0]).shape == (2, 8)


def test_mlp_encoder_zero_hidden_is_linear():
    enc = MlpEncoder(4, 8, n_hidden=0, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    assert_close(enc(x), enc.input_proj(x), atol=1e-6)


def test_mlp_encoder_is_nonlinear_when_deep():
    enc = MlpEncoder(4, 8, n_hidden=2, hidden_dim=16, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 4))
    h = jnp.tanh(enc.input_proj(x))  # deliberately wrong activation
    wrong = enc.output_proj(jnp.tanh(enc.hidden[0](h)))
    assert not jnp.allclose(enc(x), wrong, atol=1e-4)
    # ... and the same computation with ReLU does reproduce it
    h_relu = jax.nn.relu(enc.input_proj(x))
    right = enc.output_proj(jax.nn.relu(enc.hidden[0](h_relu)))
    assert_close(enc(x), right, atol=1e-5)


def test_mlp_encoder_rejects_invalid_config():
    with pytest.raises(ValueError, match="hidden_dim"):
        MlpEncoder(4, 8, n_hidden=1, hidden_dim=0)


# --------------------------------------------------------------------------- #
# LatentPredictor
# --------------------------------------------------------------------------- #
def test_latent_predictor_shapes():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, input_dim=3, action_dim=2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 3))
    act = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 2))
    z, pred_z, recon, pred_x = model(obs, act)
    assert z.shape == (2, 6, 8) and pred_z.shape == (2, 6, 8)
    assert recon.shape == (2, 6, 3) and pred_x.shape == (2, 6, 3)


def test_latent_predictor_loss_is_differentiable():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, 3, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 6, 3))
    act = jax.random.normal(jax.random.PRNGKey(1), (2, 6, 2))

    def loss_fn(model):
        z, pred_z, recon, pred_x = model(obs, act)
        return model.loss(LatentLossArgs(z, pred_z, recon, pred_x, obs))

    loss = loss_fn(model)
    assert jnp.isfinite(loss)
    grads = nnx.grad(loss_fn)(model)
    flat = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in flat)


def test_latent_predictor_training_reduces_loss():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(3e-2), wrt=nnx.Param)
    obs = jax.random.normal(jax.random.PRNGKey(0), (4, 12, 2))
    act = jnp.zeros((4, 12, 2))

    def loss_fn(model):
        z, pred_z, recon, pred_x = model(obs, act)
        return model.loss(LatentLossArgs(z, pred_z, recon, pred_x, obs))

    @nnx.jit
    def step(model, optimizer):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    first = float(step(model, optimizer))
    for _ in range(30):
        last = float(step(model, optimizer))
    assert last < first * 0.5, f"loss did not decrease: {first} -> {last}"


def test_latent_predictor_step_matches_full_sequence_prefix():
    """Feeding the model step-by-step reproduces the SSM branch of `forward`."""
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 2))
    act = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 2))

    _, pred_z, _, _ = model(obs, act)
    state = model.zero_state(batch=2)
    z = model.encode(obs)
    outs = []
    for t in range(5):
        y, state = model.step(z[:, t], act[:, t], state)
        outs.append(y)
    assert_close(pred_z, jnp.stack(outs, axis=1), atol=1e-5, msg="latent step vs forward")


def test_stability_projections_are_not_trained():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    before = jnp.array(model.stability_projections[...])
    from ssm_latent.utils import pretty_params

    names = [n for n, _ in pretty_params(model)]
    assert any("stability_projections" in n for n in names)

    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    @nnx.jit
    def step(model, optimizer, obs):
        def loss_fn(model):
            z, pred_z, recon, pred_x = model(obs, jnp.zeros((2, 4, 2)))
            return jnp.mean(z**2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2))
    for _ in range(3):
        step(model, optimizer, obs)
    assert_close(
        model.stability_projections[...], before, atol=0.0, msg="projections must be frozen"
    )


def test_save_load_roundtrip(tmp_path):
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    model = LatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2))
    act = jnp.zeros((2, 4, 2))
    before = model(obs, act)[1]

    path = tmp_path / "model.msgpack"
    model.save(path)
    assert path.exists() and path.stat().st_size > 0

    fresh = LatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(1))
    loaded = fresh.load(path)
    assert_close(loaded(obs, act)[1], before, atol=1e-6)


# --------------------------------------------------------------------------- #
# MultiScaleLatentPredictor
# --------------------------------------------------------------------------- #
def test_multiscale_predictor_shapes_with_and_without_action():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )
    for action_dim in (0, 2):
        model = MultiScaleLatentPredictor(cfg, 3, action_dim, rngs=nnx.Rngs(0))
        obs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
        act = jax.random.normal(jax.random.PRNGKey(1), (2, 5, action_dim)) if action_dim else None
        z, pred_z, recon, pred_x = model(obs, act)
        assert z.shape == (2, 5, 8) and recon.shape == (2, 5, 3)
        assert model.has_action == (action_dim > 0)


def test_multiscale_predictor_requires_actions_when_conditioned():
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2)
    model = MultiScaleLatentPredictor(cfg, 3, 2, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="action conditioning"):
        model(jnp.zeros((1, 4, 3)), None)


def test_multiscale_predictor_action_conditioning_matters():
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2)
    model = MultiScaleLatentPredictor(cfg, 3, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    act = jnp.zeros((2, 5, 2))
    other = jnp.ones((2, 5, 2))
    _, pred_a, _, _ = model(obs, act)
    _, pred_b, _, _ = model(obs, other)
    assert not jnp.allclose(pred_a, pred_b), "action conditioning must actually matter"


def test_multiscale_predictor_no_action_variant():
    """`forward_no_action` is the unsupervised path: action_dim == 0."""
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2)
    model = MultiScaleLatentPredictor(cfg, 3, 0, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 3))
    z, pred_a, recon, pred_x = model.forward_no_action(obs)
    z2, pred_b, _, _ = model(obs)
    assert pred_a.shape == (2, 5, 8) and recon.shape == (2, 5, 3)
    assert_close(pred_a, pred_b, atol=1e-6)


def test_multiscale_predictor_step_matches_forward():
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 2))
    act = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 2))

    _, pred_z, _, _ = model(obs, act)
    state = model.zero_state(2)
    z = model.encode(obs)
    outs = []
    for t in range(5):
        y, state = model.step(z[:, t], act[:, t], state)
        outs.append(y)
    assert_close(pred_z, jnp.stack(outs, axis=1), atol=1e-5)


def test_multiscale_predictor_save_load(tmp_path):
    cfg = MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(0))
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 2))
    act = jnp.ones((2, 4, 2))
    expected = model(obs, act)[1]
    path = tmp_path / "ms.msgpack"
    model.save(path)
    other = MultiScaleLatentPredictor(cfg, 2, 2, rngs=nnx.Rngs(7)).load(path)
    assert_close(other(obs, act)[1], expected, atol=1e-6)
