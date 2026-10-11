"""LeJEPA identifiability monitors (Klindt, LeCun & Balestriero, 2026).

Port of the monitoring helpers in `ssm_latent_model::latent`:

* Theorem 1 - Gaussian latents  -> :func:`gennorm_identifiability_score`,
  :func:`compute_identifiability_r2`, :func:`procrustes_alignment`
* Theorem 2 - stationary dynamics -> :func:`check_stationarity`
* Theorem 3 - isotropic exploration -> :func:`compute_exploration_quality`
* Theorem 4 - planning consistency -> :func:`check_planning_consistency`
  (plus :func:`linear_latent_plan`, :func:`plan_path_cost`)

:func:`health_check` bundles conditions 2 and 3 for use inside a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "ExplorationSummary",
    "StationarityReport",
    "PlanningConsistency",
    "compute_identifiability_r2",
    "procrustes_alignment",
    "gennorm_identifiability_score",
    "linear_latent_plan",
    "plan_path_cost",
    "compute_exploration_quality",
    "check_exploration_quality",
    "check_stationarity",
    "check_planning_consistency",
    "health_check",
]


# --------------------------------------------------------------------------- #
# Theorem 1 - linear identifiability
# --------------------------------------------------------------------------- #
def compute_identifiability_r2(z_hat: jax.Array, z_true: jax.Array) -> jax.Array:
    """Scale-free identifiability score in ``[0, 1]`` (squared) or ``[-1, 1]``.

    Returns ``tr(C^T C) / tr(A^T A)`` where ``C`` is the cross-covariance and
    ``A`` the auto-covariance of the true latent.  Equals 1 when the learned
    latent is an exact rotation of the true latent (linear identifiability).
    """
    b, t, d = z_hat.shape
    n = float(b * t)
    x = z_hat.reshape(b * t, d)
    y = z_true.reshape(b * t, d)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    c = (x.T @ y) / n
    a = (y.T @ y) / n
    return jnp.clip((c**2).sum() / ((a**2).sum() + 1e-8), 0.0, 1.0)


def procrustes_alignment(
    z_hat: jax.Array, z_true: jax.Array, n_iter: int = 30
) -> tuple[jax.Array, jax.Array]:
    """Align ``z_hat`` to ``z_true`` with the nearest orthogonal transform.

    The orthogonal factor is found with the Newton-Schulz polar iteration, which
    avoids an SVD and keeps the whole computation differentiable / jittable.
    Returns ``(z_aligned, q)``.
    """
    b, t, d = z_hat.shape
    n = float(b * t)
    x = z_hat.reshape(b * t, d)
    y = z_true.reshape(b * t, d)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc, yc = x - x_mean, y - y_mean

    c = (xc.T @ yc) / n
    q = c * (jnp.sqrt(float(d)) / jnp.maximum(jnp.linalg.norm(c), 1e-8))
    eye = jnp.eye(d, dtype=q.dtype)

    def body(_, q):
        qtq = q.T @ q
        return 0.5 * (q @ (3.0 * eye - qtq))

    q = jax.lax.fori_loop(0, n_iter, body, q)
    z_aligned = (xc @ q + y_mean).reshape(b, t, d)
    return z_aligned, q


def gennorm_identifiability_score(z: jax.Array, alpha: float = 2.0) -> jax.Array:
    """How Gaussian the latent distribution is (peaks at ``alpha == 2``).

    Uses sample excess kurtosis compared against the target kurtosis of a
    generalised normal with shape ``alpha``; the score is
    ``exp(-(kappa - target)^2 / 2)`` averaged over latent dimensions.
    """
    d = z.shape[-1]
    flat = z.reshape(-1, d)
    centred = flat - flat.mean(axis=0, keepdims=True)
    std = jnp.sqrt(centred.var(axis=0) + 1e-8)
    kurtosis = ((centred / std) ** 4).mean(axis=0)

    if abs(alpha - 2.0) < 0.01:
        target = 3.0
    elif alpha <= 1.0:
        target = 12.0 + (6.0 - 12.0) * (alpha - 0.5) / 0.5
    elif alpha <= 2.0:
        target = 6.0 + (3.0 - 6.0) * (alpha - 1.0)
    elif alpha <= 5.0:
        target = 3.0 + (2.0 - 3.0) * (alpha - 2.0) / 3.0
    else:
        target = 1.8
    return jnp.exp(-((kurtosis - target) ** 2) / 2.0).mean()


# --------------------------------------------------------------------------- #
# Theorem 4 - planning
# --------------------------------------------------------------------------- #
def linear_latent_plan(z_start: jax.Array, z_end: jax.Array, n_steps: int = 10) -> jax.Array:
    """Straight-line interpolation in latent space: ``[n_steps, B, d_model]``."""
    n = max(n_steps - 1, 1)
    alphas = jnp.arange(n_steps, dtype=z_start.dtype)[:, None, None] / n
    return (1.0 - alphas) * z_start[None] + alphas * z_end[None]


def plan_path_cost(plan: jax.Array) -> jax.Array:
    """Sum of squared step sizes along a plan of shape ``[T, B, d_model]``."""
    if plan.shape[0] < 2:
        return jnp.zeros((), dtype=plan.dtype)
    diffs = plan[1:] - plan[:-1]
    return (diffs**2).mean(axis=(1, 2)).sum()


def check_planning_consistency(
    z_plans: jax.Array, z_from_x_plans: jax.Array
) -> PlanningConsistency:
    """Correlate latent-space and observation-space path costs (Theorem 4)."""
    _, plan_len, d = z_plans.shape
    if plan_len < 2:
        return PlanningConsistency(0.0, jnp.pi / 2, False)

    def costs(plan):
        diffs = plan[:, 1:] - plan[:, :-1]
        return (diffs**2).sum(axis=-1).mean(axis=1)

    zc, xc = costs(z_plans), costs(z_from_x_plans)
    zc = zc - zc.mean()
    xc = xc - xc.mean()
    r2 = float(jnp.clip((zc @ xc) ** 2 / ((zc @ zc) * (xc @ xc) + 1e-12), 0.0, 1.0))

    dz = (z_plans[:, 1:] - z_plans[:, :-1]).reshape(-1, d)
    dx = (z_from_x_plans[:, 1:] - z_from_x_plans[:, :-1]).reshape(-1, d)
    cos_theta = jnp.sum(dz * dx, axis=1) / (
        jnp.linalg.norm(dz, axis=1) * jnp.linalg.norm(dx, axis=1) + 1e-8
    )
    angle = float(jnp.arccos(jnp.clip(cos_theta.mean(), -1.0, 1.0)))
    return PlanningConsistency(r2, angle, r2 > 0.8 and angle < 0.5)


# --------------------------------------------------------------------------- #
# Theorem 3 - exploration quality
# --------------------------------------------------------------------------- #
@dataclass
class ExplorationSummary:
    """Structured exploration-quality report (Theorem 3)."""

    coverage: float
    anisotropy: float
    effective_rank: float
    trajectory_narrowness: float
    gaussian_score: float
    risk_level: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"coverage={self.coverage:.3f} anisotropy={self.anisotropy:.3f} "
            f"effective_rank={self.effective_rank:.2f} "
            f"narrowness={self.trajectory_narrowness:.3f} "
            f"gaussian_score={self.gaussian_score:.3f} -> {self.risk_level}"
        )


def compute_exploration_quality(z: jax.Array) -> ExplorationSummary:
    """Coverage, anisotropy, effective rank, narrowness and Gaussianity."""
    batch, t, d = z.shape
    n = float(batch * t)
    flat = z.reshape(batch * t, d)
    centred = flat - flat.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / n

    trace = float(jnp.trace(cov))
    frob_sq = float((cov**2).sum())
    effective_rank = float(min(max(trace**2 / max(frob_sq, 1e-8), 1.0), float(d)))
    avg_eig = trace / d

    # dominant eigenvalue by power iteration (deterministic start vector)
    v = jnp.ones((d, 1), dtype=cov.dtype) / jnp.sqrt(float(d))

    def body(_, v):
        w = cov @ v
        return w / (jnp.linalg.norm(w) + 1e-8)

    v = jax.lax.fori_loop(0, 20, body, v)
    lambda_max = float((v.T @ cov @ v)[0, 0])
    anisotropy = lambda_max / avg_eig if avg_eig > 1e-8 else 1.0
    coverage = min(max(effective_rank / d, 0.0), 1.0)

    # trajectory narrowness: |cos| between consecutive displacements
    if t >= 3:
        disp = z[:, 1:] - z[:, :-1]
        d1, d2 = disp[:, :-1], disp[:, 1:]
        dot = (d1 * d2).sum(axis=-1)
        norm = jnp.linalg.norm(d1, axis=-1) * jnp.linalg.norm(d2, axis=-1) + 1e-8
        narrowness = float(jnp.abs(dot / norm).mean())
    else:
        narrowness = 0.0

    std = jnp.sqrt(centred.var(axis=0) + 1e-8)
    excess = float(jnp.abs(((centred / std) ** 4).mean(axis=0) - 3.0).mean())
    gaussian_score = float(jnp.clip(jnp.exp(-excess * 0.5), 0.0, 1.0))

    if coverage < 0.2 or anisotropy > 3.0 or gaussian_score < 0.4:
        risk = "HIGH"
    elif coverage < 0.4 or anisotropy > 2.0 or gaussian_score < 0.6:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return ExplorationSummary(
        coverage=coverage,
        anisotropy=anisotropy,
        effective_rank=effective_rank,
        trajectory_narrowness=narrowness,
        gaussian_score=gaussian_score,
        risk_level=risk,
    )


def check_exploration_quality(
    z: jax.Array, *, context: str = "", threshold: str = "HIGH"
) -> ExplorationSummary:
    """As :func:`compute_exploration_quality`, warning when risk is high."""
    import warnings

    summary = compute_exploration_quality(z)
    if summary.risk_level == "HIGH":
        warnings.warn(
            f"poor exploration quality{' (' + context + ')' if context else ''}: {summary}. "
            "LeJEPA identifiability guarantees may not hold.",
            stacklevel=2,
        )
    return summary


# --------------------------------------------------------------------------- #
# Theorem 2 - stationarity
# --------------------------------------------------------------------------- #
@dataclass
class StationarityReport:
    """Prediction-error trend and risk assessment (Theorem 2)."""

    residual_trend: float
    dominant_layer: int
    layer_entropy: float
    risk_level: str


def check_stationarity(
    z: jax.Array, predicted_z: jax.Array, n_layers: int = 3
) -> StationarityReport:
    """Slope of ``log`` prediction error against time.

    A flat trend means the dynamics look stationary (Theorem 2's assumption);
    a strongly positive trend means the model is progressively losing track.
    """
    _, t = z.shape[:2]
    trend = 0.0
    if t >= 4:
        err = ((z[:, 1:] - predicted_z[:, :-1]) ** 2).sum(axis=-1).mean(axis=0)  # [T-1]
        err = jnp.asarray(err)
        log_err = jnp.log(err + 1e-10)
        steps = jnp.arange(1, log_err.shape[0] + 1, dtype=log_err.dtype)
        steps_c = steps - steps.mean()
        trend = float((steps_c @ (log_err - log_err.mean())) / (steps_c @ steps_c))

    layer_entropy = 0.85 if n_layers > 1 else 1.0
    if trend > 0.05:
        dominant = n_layers - 1
    elif trend < -0.05:
        dominant = 0
    else:
        dominant = n_layers // 2

    if abs(trend) > 0.1:
        risk = "HIGH"
    elif abs(trend) > 0.05:
        risk = "LOW-MEDIUM"
    else:
        risk = "LOW"
    return StationarityReport(trend, dominant, layer_entropy, risk)


@dataclass
class PlanningConsistency:
    """Result of a planning-consistency check (Theorem 4)."""

    cost_consistency_r2: float
    directional_error: float
    is_consistent: bool


def health_check(
    z: jax.Array, predicted_z: jax.Array, n_layers: int = 3, *, context: str = ""
) -> tuple[ExplorationSummary, StationarityReport]:
    """Run the exploration-quality and stationarity checks together."""
    return (
        check_exploration_quality(z, context=context),
        check_stationarity(z, predicted_z, n_layers),
    )
