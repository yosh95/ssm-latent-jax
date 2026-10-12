"""Meta-learning: tasks, adaptation strategies, Reptile and adaptation-speed metrics."""

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_close
from flax import nnx

from ssm_latent import MultiScaleSsmConfig
from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta import (
    AdaptationSpeedReport,
    AdaptationStrategy,
    CircleWorldDistribution,
    CircleWorldTask,
    CompositeDistribution,
    LossWeights,
    MetaPipelineConfig,
    ReptileMetaLearner,
    SineWaveDistribution,
    SineWaveTask,
    adapt,
    adapt_loss,
    adaptation_curve,
    adaptation_strategy_description,
    evaluate_adaptation_speed,
    inner_loop,
    run_meta_pipeline,
    strategy_mask,
    trainable_parameter_names,
)


def CFG(**kw):
    return MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=False, **kw
    )


def _model(seed=0, action_dim=2, input_dim=2, **kw):
    return MultiScaleLatentPredictor(CFG(**kw), input_dim, action_dim, rngs=nnx.Rngs(seed))


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
def test_circle_world_task_shapes_and_ids():
    task = CircleWorldTask(omega=1.5, n_support=8, n_query=16, batch_size=2)
    obs, act = task.support_set()
    assert obs.shape == (2, 8, 2) and act.shape == (2, 8, 2)
    q_obs, q_act = task.query_set()
    assert q_obs.shape == (2, 16, 2) and q_act.shape == (2, 16, 2)
    assert task.task_id == "CircleWorld(omega=1.500)"
    assert task.obs_dim == 2 and task.action_dim == 2
    assert task.success_threshold == 0.05


def test_circle_world_task_support_and_query_differ():
    # `noise_level=0` so that the support prefix is bit-identical to the query
    # set (the noise draw depends on the sequence length)
    task = CircleWorldTask(omega=1.0, noise_level=0.0, n_support=8, n_query=64, batch_size=2)
    support, _ = task.support_set()
    query, _ = task.query_set()
    assert support.shape[1] == 8 and query.shape[1] == 64
    # both sequences start at t = 0 and share the first `n_support` steps ...
    assert_close(support, query[:, :8], atol=1e-6)
    # ... but the query set extrapolates well past the adaptation horizon
    assert float(jnp.max(jnp.abs(query[:, 8:]))) > 0.1


def test_circle_world_task_data_is_on_the_unit_circle_and_deterministic():
    task = CircleWorldTask(omega=0.8, noise_level=0.0, n_support=6, batch_size=3)
    obs, _ = task.support_set()
    obs2, _ = task.support_set()
    assert_close(jnp.linalg.norm(obs, axis=-1), jnp.ones((3, 6)), atol=1e-5)
    assert_close(obs, obs2, atol=0.0)


def test_circle_world_distribution_is_key_deterministic_and_within_range():
    dist = CircleWorldDistribution(min_omega=0.5, max_omega=2.0)
    a = dist.sample_task(jax.random.PRNGKey(0))
    b = dist.sample_task(jax.random.PRNGKey(0))
    c = dist.sample_task(jax.random.PRNGKey(1))
    assert a.omega == b.omega and a.phase_shift == b.phase_shift
    assert (a.omega, a.phase_shift) != (c.omega, c.phase_shift)
    assert 0.5 <= a.omega <= 2.0
    assert 0.0 <= a.phase_shift <= 2 * jnp.pi
    assert dist.n_families == 1


def test_circle_world_distribution_covers_its_range():
    dist = CircleWorldDistribution(min_omega=0.5, max_omega=2.0)
    omegas = [dist.sample_task(jax.random.PRNGKey(i)).omega for i in range(50)]
    assert min(omegas) < 0.9 and max(omegas) > 1.7


def test_sine_wave_task_shapes():
    task = SineWaveTask(frequency=2.0, n_support=8, n_query=16, batch_size=2)
    obs, act = task.support_set()
    assert obs.shape == (2, 8, 1) and act.shape == (2, 8, 1)
    assert task.obs_dim == 1 and task.action_dim == 1
    assert task.task_id == "SineWave(f=2.00)"


def test_sine_wave_distribution_sampling():
    dist = SineWaveDistribution()
    task = dist.sample_task(jax.random.PRNGKey(0))
    assert 0.5 <= task.frequency <= 3.0
    assert 0.5 <= task.amplitude <= 2.0
    assert isinstance(task, SineWaveTask)


def test_composite_distribution_mixes_both_families():
    dist = CompositeDistribution()
    assert dist.n_families == 2
    kinds = {type(dist.sample_task(jax.random.PRNGKey(i))).__name__ for i in range(40)}
    assert kinds == {"CircleWorldTask", "SineWaveTask"}


# --------------------------------------------------------------------------- #
# adaptation strategies
# --------------------------------------------------------------------------- #
def test_strategy_descriptions_match_the_reference_strings():
    assert adaptation_strategy_description(AdaptationStrategy.FULL) == "All parameters"
    assert adaptation_strategy_description(AdaptationStrategy.ENCODER_ONLY) == "Encoder only"
    assert adaptation_strategy_description(AdaptationStrategy.DECODER_ONLY) == "Decoder only"
    assert (
        adaptation_strategy_description(AdaptationStrategy.SLOW_SSM_ONLY) == "Slow SSM layer only"
    )
    assert (
        adaptation_strategy_description(AdaptationStrategy.ENCODER_PLUS_SLOW_SSM)
        == "Encoder + Slow SSM"
    )


def test_strategy_masks_select_the_expected_parameter_groups():
    model = _model()

    full = trainable_parameter_names(model, AdaptationStrategy.FULL)
    assert len(full) == len(set(full)) > 10

    encoder = trainable_parameter_names(model, AdaptationStrategy.ENCODER_ONLY)
    assert encoder and all(n.startswith("encoder") for n in encoder)

    decoder = trainable_parameter_names(model, AdaptationStrategy.DECODER_ONLY)
    assert decoder and all(n.startswith("decoder") for n in decoder)

    slow = trainable_parameter_names(model, AdaptationStrategy.SLOW_SSM_ONLY)
    assert slow and all(n.startswith("ssm/layers/2/") for n in slow), slow

    both = trainable_parameter_names(model, AdaptationStrategy.ENCODER_PLUS_SLOW_SSM)
    assert set(both) == set(encoder) | set(slow)

    # strict subsets of the full parameter set
    for subset in (encoder, decoder, slow, both):
        assert set(subset) < set(full)


def test_strategy_mask_has_boolean_leaves_and_matches_the_names():
    model = _model()
    for strategy in AdaptationStrategy:
        mask = strategy_mask(model, strategy)
        leaves = jax.tree.leaves(mask)
        assert leaves and all(isinstance(v, bool) for v in leaves)
        assert sum(leaves) == len(trainable_parameter_names(model, strategy))


def test_slow_ssm_only_adapts_the_last_layer():
    """After adapting with SLOW_SSM_ONLY the fast layers must be bit-identical."""
    model = _model()
    before = [jnp.array(layer.a_re[...]) for layer in model.ssm.layers]
    task = CircleWorldTask(omega=1.2, n_support=8, n_query=16, batch_size=2)
    inner_loop(model, task, inner_lr=1e-2, inner_steps=4, strategy=AdaptationStrategy.SLOW_SSM_ONLY)
    after = [layer.a_re[...] for layer in model.ssm.layers]
    assert_close(after[0], before[0], atol=0.0)
    assert_close(after[1], before[1], atol=0.0)
    assert float(jnp.max(jnp.abs(after[2] - before[2]))) > 0.0


def test_encoder_only_leaves_the_ssm_untouched():
    model = _model()
    before = jnp.array(model.ssm.layers[0].a_re[...])
    task = CircleWorldTask(omega=1.2, n_support=8, n_query=16, batch_size=2)
    inner_loop(model, task, inner_lr=1e-2, inner_steps=3, strategy=AdaptationStrategy.ENCODER_ONLY)
    assert_close(model.ssm.layers[0].a_re[...], before, atol=0.0)


# --------------------------------------------------------------------------- #
# inner loop
# --------------------------------------------------------------------------- #
def test_inner_loop_reduces_the_support_loss():
    model = _model()
    task = CircleWorldTask(omega=1.0, noise_level=0.0, n_support=16, n_query=16, batch_size=2)
    _, result = inner_loop(model, task, inner_lr=1e-2, inner_steps=8)
    assert len(result.loss_curve) == 8
    assert result.steps_taken == 8
    assert result.loss_curve[-1] < result.loss_curve[0]
    assert result.adaptation_time_ms >= 0.0


def test_inner_loop_has_no_seed_parameter():
    """The inner loop is deterministic (`seed` used to be accepted and ignored)."""
    import inspect

    params = inspect.signature(inner_loop).parameters
    assert "seed" not in params


def test_inner_loop_with_zero_lr_does_not_move_but_still_reports():
    model = _model()
    before = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    task = CircleWorldTask(omega=1.0, n_support=8, n_query=8, batch_size=2)
    _, result = inner_loop(model, task, inner_lr=0.0, inner_steps=3)
    after = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    for key_tree, value in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        assert_close(value, key_tree, atol=0.0, msg="zero learning rate must not move parameters")
    assert_close(result.loss_curve[0], result.loss_curve[-1], atol=1e-6)
    assert result.initial_loss == pytest.approx(result.loss_curve[0], rel=1e-4)


def test_adapt_does_not_mutate_the_original_model():
    model = _model()
    before = jnp.array(model.encoder.kernel[...])
    task = CircleWorldTask(omega=1.0, n_support=8, n_query=8, batch_size=2)
    adapted, _ = adapt(model, task, inner_lr=1e-2, inner_steps=4)
    assert_close(model.encoder.kernel[...], before, atol=0.0)
    assert float(jnp.max(jnp.abs(adapted.encoder.kernel[...] - before))) > 0.0


def test_adapt_loss_uses_the_unsupervised_path_when_there_is_no_action():
    model = _model(action_dim=0)
    obs, _ = CircleWorldTask(omega=1.0, n_support=8, batch_size=2).support_set()
    value = float(adapt_loss(model, obs, None))
    assert jnp.isfinite(value) and value >= 0.0

    with_action = _model(action_dim=2)
    act = jnp.zeros_like(obs)
    assert jnp.isfinite(float(adapt_loss(with_action, obs, act)))


def test_loss_weights_are_honoured():
    model = _model()
    obs, act = CircleWorldTask(omega=1.0, n_support=8, batch_size=2).support_set()
    low = float(adapt_loss(model, obs, act, LossWeights(recon=0.0, stability=0.0, curvature=0.0)))
    high = float(
        adapt_loss(model, obs, act, LossWeights(recon=10.0, stability=10.0, curvature=10.0))
    )
    assert high > low


# --------------------------------------------------------------------------- #
# adaptation-speed report
# --------------------------------------------------------------------------- #
def test_adaptation_speed_report_from_curve_statistics():
    # NOTE: `evaluate_adaptation_speed` feeds this the *support* loss curve
    # (what `inner_loop` optimises); `adaptation_curve()` is the query-side twin.
    curve = [1.0, 0.5, 0.25, 0.01, 0.005]
    report = AdaptationSpeedReport.from_curve(curve, threshold=0.05)
    assert report.steps_to_success == 4  # 1-indexed
    assert report.loss_at_step_1 == 1.0
    assert report.success is True
    assert report.final_loss == 0.005
    assert report.initial_improvement_rate == pytest.approx(0.5)
    # trapezoidal AUC (step width 1), normalised by the number of points
    expected_auc = (
        0.5 * (1.0 + 0.5) + 0.5 * (0.5 + 0.25) + 0.5 * (0.25 + 0.01) + 0.5 * (0.01 + 0.005)
    ) / 5
    assert report.adaptation_auc == pytest.approx(expected_auc, rel=1e-5)


def test_adaptation_curve_is_the_query_side_twin_of_evaluate():
    """`adaptation_curve` scores the *held-out* loss at every step.

    It is the query-side twin of :func:`evaluate_adaptation_speed` (which scores
    the support curve the inner loop optimises).  On a deterministic task both
    must agree that adaptation generalises: the query loss falls as the model
    specialises.
    """
    model = _model()
    task = CircleWorldTask(omega=1.0, noise_level=0.0, n_support=16, n_query=32, batch_size=4)
    curve = adaptation_curve(model, task, inner_lr=1e-2, max_steps=6)
    assert len(curve) == 6
    assert all(v >= 0.0 for v in curve)
    assert curve[-1] < curve[0], f"query loss must fall over adaptation, got {curve}"

    report = evaluate_adaptation_speed(model, task, inner_lr=1e-2, max_steps=6)
    assert report.final_loss < report.adaptation_curve[0]


def test_adaptation_speed_report_defaults_are_safe():
    report = AdaptationSpeedReport.from_curve([])
    assert report.steps_to_success is None and report.success is False
    assert report.adaptation_auc == 0.0
    single = AdaptationSpeedReport.from_curve([0.5], threshold=0.1)
    assert single.adaptation_auc == 0.5
    assert single.initial_improvement_rate == 0.0
    assert single.success is False


def test_evaluate_adaptation_speed_runs_and_reports_reasonably():
    """A small world model should already adapt a circle task quite quickly."""
    model = _model()
    task = CircleWorldTask(omega=1.0, noise_level=0.0, n_support=16, n_query=32, batch_size=4)
    report = evaluate_adaptation_speed(model, task, inner_lr=1e-2, max_steps=10)
    assert len(report.adaptation_curve) == 10
    assert report.final_loss < report.adaptation_curve[0]
    assert report.adaptation_auc > 0.0


def test_evaluate_adaptation_speed_leaves_the_model_unchanged():
    model = _model()
    before = jnp.array(model.ssm.layers[0].a_re[...])
    evaluate_adaptation_speed(
        model, CircleWorldTask(n_support=8, n_query=8, batch_size=2), max_steps=2
    )
    assert_close(model.ssm.layers[0].a_re[...], before, atol=0.0)


# --------------------------------------------------------------------------- #
# Reptile
# --------------------------------------------------------------------------- #
def test_reptile_meta_train_step_reports_metrics():
    model = _model()
    learner = ReptileMetaLearner(
        model,
        CircleWorldDistribution(n_support=8, n_query=16, batch_size=2),
        inner_lr=1e-2,
        inner_steps=3,
        outer_lr=0.5,
    )
    metrics = learner.meta_train_step(iteration=1)
    assert metrics.iteration == 1
    assert metrics.success_rate in (0.0, 1.0)
    assert 0.0 <= metrics.success_rate <= 1.0
    assert metrics.avg_adaptation_time_ms >= 0.0
    assert jnp.isfinite(metrics.avg_final_loss)


def test_reptile_updates_the_base_weights_and_is_seed_deterministic():
    a = _model(seed=0)
    b = _model(seed=0)
    before = jnp.array(a.ssm.layers[0].a_re[...])

    def run(model):
        learner = ReptileMetaLearner(
            model,
            CircleWorldDistribution(
                min_omega=0.5, max_omega=2.0, n_support=8, n_query=8, batch_size=2
            ),
            inner_lr=1e-2,
            inner_steps=2,
            outer_lr=1.0,
            seed=7,
        )
        learner.meta_train_step()
        return model

    ra, rb = run(a), run(b)
    assert float(jnp.max(jnp.abs(ra.ssm.layers[0].a_re[...] - before))) > 0.0
    pure_a = nnx.to_pure_dict(nnx.state(ra, nnx.Param))
    pure_b = nnx.to_pure_dict(nnx.state(rb, nnx.Param))
    for x, y in zip(jax.tree.leaves(pure_a), jax.tree.leaves(pure_b), strict=True):
        assert_close(x, y, atol=1e-6)


def test_reptile_outer_lr_zero_is_a_no_op():
    # the sine-wave family is scalar-valued, so the model must match its IO dims
    model = _model(input_dim=1, action_dim=1)
    before = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    learner = ReptileMetaLearner(
        model,
        SineWaveDistribution(n_support=8, n_query=8, batch_size=2),
        inner_lr=1e-2,
        inner_steps=2,
        outer_lr=0.0,
    )
    learner.meta_train_step()
    after = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    for x, y in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        assert_close(x, y, atol=0.0)


def test_reptile_train_loop_returns_one_metric_per_iteration():
    model = _model()
    learner = ReptileMetaLearner(
        model,
        CircleWorldDistribution(n_support=8, n_query=8, batch_size=2),
        inner_lr=1e-2,
        inner_steps=2,
        outer_lr=0.2,
    )
    metrics = learner.train(3, verbose=False)
    assert [m.iteration for m in metrics] == [1, 2, 3]


def test_reptile_evaluate_on_holdout_tasks():
    model = _model()
    learner = ReptileMetaLearner(
        model, CircleWorldDistribution(n_support=8, n_query=8, batch_size=2), inner_steps=2
    )
    tasks = [
        CircleWorldTask(omega=0.3, n_support=8, n_query=8, batch_size=2),
        CircleWorldTask(omega=2.5, n_support=8, n_query=8, batch_size=2),
    ]
    reports = learner.evaluate(tasks, max_steps=3)
    assert len(reports) == 2
    assert all(isinstance(r, AdaptationSpeedReport) for r in reports)
    assert all(len(r.adaptation_curve) == 3 for r in reports)


def test_run_meta_pipeline_end_to_end():
    model = _model()
    config = MetaPipelineConfig(num_iterations=2, inner_steps=2, outer_lr=0.3, max_eval_steps=3)
    learner, train_metrics, reports = run_meta_pipeline(
        model,
        CircleWorldDistribution(n_support=8, n_query=8, batch_size=2),
        [CircleWorldTask(omega=0.4, n_support=8, n_query=8, batch_size=2)],
        config,
        verbose=False,
    )
    assert len(train_metrics) == 2 and len(reports) == 1
    assert isinstance(learner, ReptileMetaLearner)


@pytest.mark.slow
def test_reptile_improves_adaptation_on_an_unseen_task():
    """Meta-training must not make the holdout adaptation worse (sanity bound)."""
    model = _model()
    holdout = CircleWorldTask(omega=1.75, n_support=16, n_query=32, batch_size=4)
    before = evaluate_adaptation_speed(model, holdout, inner_lr=3e-3, max_steps=10)

    learner = ReptileMetaLearner(
        model,
        CircleWorldDistribution(
            min_omega=0.5, max_omega=2.0, n_support=16, n_query=32, batch_size=4
        ),
        inner_lr=3e-3,
        inner_steps=5,
        outer_lr=0.3,
        seed=1,
    )
    learner.train(5, verbose=False)
    after = evaluate_adaptation_speed(learner.model, holdout, inner_lr=3e-3, max_steps=10)
    assert after.adaptation_auc < before.adaptation_auc
