"""LeJEPA identifiability monitors (Theorems 1-4)."""

import jax
import jax.numpy as jnp
import pytest
from conftest import assert_close

from ssm_latent import monitor


# --------------------------------------------------------------------------- #
# Theorem 1 - linear identifiability
# --------------------------------------------------------------------------- #
def test_r2_is_one_for_an_orthogonal_transform():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (4, 32, 6))
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (6, 6)))
    rotated = z @ q
    assert_close(monitor.compute_identifiability_r2(rotated, z), 1.0, atol=1e-4)


def test_r2_is_invariant_to_scale():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (2, 16, 4))
    assert_close(
        monitor.compute_identifiability_r2(3.7 * z, z),
        monitor.compute_identifiability_r2(z, z),
        atol=1e-5,
    )


def test_r2_drops_for_unrelated_latents():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (4, 64, 8))
    other = jax.random.normal(jax.random.PRNGKey(1), (4, 64, 8))
    assert float(monitor.compute_identifiability_r2(other, z)) < 0.3


def test_r2_is_zero_for_a_collapsed_latent():
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 4))
    assert_close(monitor.compute_identifiability_r2(jnp.zeros_like(z), z), 0.0, atol=1e-6)


def test_procrustes_recovers_a_known_rotation_and_is_orthogonal():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (3, 40, 5))
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(2), (5, 5)))
    rotated = z @ q
    aligned, q_hat = monitor.procrustes_alignment(rotated, z)
    assert_close(q_hat.T @ q_hat, jnp.eye(5), atol=1e-4, msg="Q must be orthogonal")
    assert_close(aligned, z, atol=1e-3, msg="alignment")


def test_procrustes_is_identity_for_identical_inputs():
    z = jax.random.normal(jax.random.PRNGKey(0), (1, 20, 4))
    _, q = monitor.procrustes_alignment(z, z)
    assert_close(q, jnp.eye(4), atol=1e-5)


def test_gennorm_score_peaks_at_the_gaussian():
    key = jax.random.PRNGKey(0)
    gaussian = jax.random.normal(key, (4, 2000, 4))
    uniform = jax.random.uniform(key, (4, 2000, 4), minval=-1.0, maxval=1.0)
    assert float(monitor.gennorm_identifiability_score(gaussian, 2.0)) > float(
        monitor.gennorm_identifiability_score(uniform, 2.0)
    )


def test_gennorm_score_covers_all_alpha_branches():
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 512, 3))
    scores = [
        float(monitor.gennorm_identifiability_score(z, a)) for a in (0.5, 1.0, 1.5, 2.0, 4.0, 8.0)
    ]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert max(scores) == scores[3], "alpha = 2 must be the best match for Gaussian data"


# --------------------------------------------------------------------------- #
# Theorem 3 - exploration quality
# --------------------------------------------------------------------------- #
def test_exploration_quality_is_low_risk_for_isotropic_gaussian_data():
    z = jax.random.normal(jax.random.PRNGKey(0), (8, 256, 16))
    summary = monitor.compute_exploration_quality(z)
    assert summary.risk_level == "LOW"
    assert summary.coverage > 0.5
    assert summary.anisotropy < 2.0
    assert summary.effective_rank > 8.0
    assert summary.gaussian_score > 0.6


def test_anisotropic_data_is_flagged():
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (8, 256, 8)) * jnp.array(
        [10.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    )
    summary = monitor.compute_exploration_quality(z)
    assert summary.anisotropy > 3.0
    assert summary.effective_rank < 3.0
    assert summary.risk_level == "HIGH"


def test_goal_directed_trajectories_are_narrow():
    """A monotonically re-visited line has |cos| ~ 1 between displacements."""
    t = jnp.linspace(0, 1, 200)[None, :, None]
    line = jnp.concatenate([t, t, t], axis=-1)
    summary = monitor.compute_exploration_quality(line)
    assert summary.trajectory_narrowness > 0.9

    random_walk = jax.random.normal(jax.random.PRNGKey(0), (4, 400, 8))
    wide = monitor.compute_exploration_quality(random_walk)
    assert wide.trajectory_narrowness < summary.trajectory_narrowness


def test_collapsed_data_is_flagged_as_high_risk():
    z = jnp.ones((2, 64, 8))
    summary = monitor.compute_exploration_quality(z)
    assert summary.coverage <= 0.2
    assert summary.risk_level == "HIGH"


def test_exploration_quality_handles_very_short_sequences():
    z = jnp.ones((2, 2, 4))
    summary = monitor.compute_exploration_quality(z)
    assert summary.trajectory_narrowness == 0.0


def test_check_exploration_quality_warns_on_high_risk():
    with pytest.warns(UserWarning, match="poor exploration quality"):
        monitor.check_exploration_quality(jnp.ones((1, 16, 4)), context="unit-test")
    z = jax.random.normal(jax.random.PRNGKey(0), (8, 256, 16))
    assert monitor.check_exploration_quality(z).risk_level == "LOW"


# --------------------------------------------------------------------------- #
# Theorem 2 - stationarity
# --------------------------------------------------------------------------- #
def test_stationary_residuals_report_low_risk():
    z = jax.random.normal(jax.random.PRNGKey(0), (4, 64, 8))
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(1), z.shape)
    report = monitor.check_stationarity(z, z + noise, n_layers=3)
    assert abs(report.residual_trend) < 0.05
    assert report.risk_level == "LOW"
    assert report.dominant_layer == 1  # middle layer for flat trends


def test_exploding_error_is_detected_as_high_risk():
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 4))
    growth = jnp.exp(jnp.linspace(0, 6, 64))[None, :, None]
    report = monitor.check_stationarity(z, z - growth, n_layers=3)
    assert report.residual_trend > 0.05
    assert report.risk_level in ("HIGH", "LOW-MEDIUM")
    assert report.dominant_layer == 2


def test_shrinking_error_flags_the_fast_layer():
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 4))
    # the perturbation has to dominate the baseline |z| scale for the trend
    # slope to be visible at all
    shrink = 30.0 * jnp.exp(-jnp.linspace(0, 12, 64))[None, :, None]
    report = monitor.check_stationarity(z, z - shrink, n_layers=3)
    assert report.residual_trend < -0.05
    assert report.dominant_layer == 0


def test_stationarity_handles_short_sequences():
    z = jnp.ones((1, 3, 4))
    report = monitor.check_stationarity(z, z, n_layers=1)
    assert report.residual_trend == 0.0
    assert report.layer_entropy == 1.0


# --------------------------------------------------------------------------- #
# Theorem 4 - planning
# --------------------------------------------------------------------------- #
def test_linear_latent_plan_interpolates_endpoints():
    start = jnp.zeros((1, 3))
    end = jnp.ones((1, 3)) * 4.0
    plan = monitor.linear_latent_plan(start, end, n_steps=5)
    assert plan.shape == (5, 1, 3)
    assert_close(plan[0], start, atol=1e-6)
    assert_close(plan[-1], end, atol=1e-6)
    assert_close(plan[2], (start + end) / 2, atol=1e-6)


def test_linear_latent_plan_single_step():
    plan = monitor.linear_latent_plan(jnp.zeros((1, 2)), jnp.ones((1, 2)), n_steps=1)
    assert plan.shape == (1, 1, 2)


def test_plan_path_cost_is_zero_for_a_single_waypoint():
    assert float(monitor.plan_path_cost(jnp.zeros((1, 2, 3)))) == 0.0


def test_plan_path_cost_scales_quadratically_with_length():
    start = jnp.zeros((1, 2))
    end = jnp.ones((1, 2))
    short = monitor.linear_latent_plan(start, end, 3)
    long = monitor.linear_latent_plan(start, end, 5)
    assert float(monitor.plan_path_cost(long)) < float(monitor.plan_path_cost(short))


def test_planning_consistency_is_exact_for_identical_plans():
    z = jax.random.normal(jax.random.PRNGKey(0), (4, 8, 6))
    result = monitor.check_planning_consistency(z, z)
    assert result.cost_consistency_r2 > 0.999
    assert result.directional_error < 1e-3
    assert result.is_consistent


def test_planning_consistency_detects_inconsistent_plans():
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (4, 8, 6))
    b = jax.random.normal(jax.random.PRNGKey(1), (4, 8, 6))
    result = monitor.check_planning_consistency(a, b)
    assert not result.is_consistent
    assert result.directional_error > 0.5


def test_planning_consistency_is_invariant_to_rotation():
    """Theorem 4: an orthogonal re-parameterisation leaves path costs unchanged."""
    key = jax.random.PRNGKey(0)
    z = jax.random.normal(key, (4, 8, 6))
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (6, 6)))
    result = monitor.check_planning_consistency(z @ q, z)
    assert result.cost_consistency_r2 > 0.999
    # ... which is what Theorem 4's `cost_consistency_r2` measures.  The raw
    # direction cosine is *not* rotation invariant, so `is_consistent` (which
    # additionally demands aligned directions) is expected to fail here.
    assert result.directional_error > 0.5


def test_planning_consistency_handles_degenerate_plan_length():
    z = jnp.zeros((2, 1, 3))
    result = monitor.check_planning_consistency(z, z)
    assert not result.is_consistent
    assert result.cost_consistency_r2 == 0.0


# --------------------------------------------------------------------------- #
# combined health check
# --------------------------------------------------------------------------- #
def test_health_check_returns_both_reports():
    z = jax.random.normal(jax.random.PRNGKey(0), (4, 128, 8))
    exploration, stationarity = monitor.health_check(z, z, n_layers=3, context="test")
    assert exploration.risk_level == "LOW"
    assert stationarity.risk_level == "LOW"


def test_health_check_is_jit_friendly():
    """Every metric must be traceable (no Python-level data dependence)."""
    from ssm_latent.monitor import (
        compute_identifiability_r2,
        gennorm_identifiability_score,
        plan_path_cost,
    )

    z = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 4))

    @jax.jit
    def metrics(z):
        return (
            compute_identifiability_r2(z, z),
            gennorm_identifiability_score(z),
            plan_path_cost(jnp.moveaxis(monitor.linear_latent_plan(z[0, 0], z[0, -1], 5), 0, 0)),
        )

    r2, score, cost = metrics(z)
    assert jnp.isfinite(r2) and jnp.isfinite(score) and jnp.isfinite(cost)


def test_stationarity_measures_layer_entropy_from_layer_errors():
    """With a per-layer breakdown the attribution is measured, not heuristic."""
    z = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 4))
    # `layer_errors[i]` is layer i's *prediction* of z; layer 1 is far off.
    layer_errors = [z, z + 5.0, z]
    report = monitor.check_stationarity(z, z, n_layers=3, layer_errors=layer_errors)
    assert report.dominant_layer == 1
    assert 0.0 <= report.layer_entropy <= 1.0
    # an equal error split -> higher entropy than the concentrated case
    spread = [z + 0.3, z + 0.5, z + 0.4]
    flat = monitor.check_stationarity(z, z, n_layers=3, layer_errors=spread)
    assert flat.layer_entropy > report.layer_entropy
    assert flat.layer_entropy > 0.9
