"""Example 5 - LeJEPA identifiability monitors in action.

The four conditions of Klindt, LeCun & Balestriero (2026) are checked on
*deliberately healthy and unhealthy* latent spaces, so that the thresholds are
readable rather than magic numbers:

* Theorem 1 - Gaussian latents      -> identifiability R^2, Procrustes, kurtosis
* Theorem 2 - stationary dynamics   -> prediction-error trend
* Theorem 3 - isotropic exploration -> coverage / anisotropy / effective rank
* Theorem 4 - planning consistency  -> latent vs. observation path costs

Run: ``python examples/05_identifiability_monitors.py``
"""

import jax
import jax.numpy as jnp
from common import banner, section

from ssm_latent.monitor import (
    check_exploration_quality,
    check_planning_consistency,
    check_stationarity,
    compute_exploration_quality,
    compute_identifiability_r2,
    gennorm_identifiability_score,
    health_check,
    linear_latent_plan,
    plan_path_cost,
    procrustes_alignment,
)

KEY = jax.random.PRNGKey(0)


def _row(name, value, fmt="{:.4f}"):
    return f"  {name:<34}{fmt.format(value)}"


def main() -> None:
    banner("LeJEPA identifiability monitors")

    # ------------------------------------------------------------------ T1
    section("Theorem 1 - Gaussianity and linear identifiability")
    z_true = jax.random.normal(KEY, (8, 256, 16))
    q, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (16, 16)))
    rotated = z_true @ q
    scaled = 5.0 * rotated + jnp.arange(16.0)

    print(
        _row("R^2 (exact rotation)", float(compute_identifiability_r2(rotated, z_true)), "{:.6f}")
    )
    print(
        _row(
            "R^2 (rotation + shift + scale)",
            float(compute_identifiability_r2(scaled, z_true)),
            "{:.6f}",
        )
    )
    other = jax.random.normal(jax.random.PRNGKey(2), (8, 256, 16))
    print(
        _row("R^2 (unrelated latents)", float(compute_identifiability_r2(other, z_true)), "{:.6f}")
    )

    aligned, q_hat = procrustes_alignment(rotated, z_true)
    print(_row("Procrustes reconstruction MSE", float(jnp.mean((aligned - z_true) ** 2)), "{:.3e}"))
    print(
        _row(
            "Procrustes ||Q^T Q - I||_inf",
            float(jnp.abs(q_hat.T @ q_hat - jnp.eye(16)).max()),
            "{:.3e}",
        )
    )

    uniform = jax.random.uniform(KEY, (8, 4096, 16))
    print(
        _row(
            "gennorm score (Gaussian data)",
            float(gennorm_identifiability_score(z_true, 2.0)),
            "{:.4f}",
        )
    )
    print(
        _row(
            "gennorm score (uniform data)",
            float(gennorm_identifiability_score(uniform, 2.0)),
            "{:.4f}",
        )
    )

    # ------------------------------------------------------------------ T3
    section("Theorem 3 - isotropic exploration")
    healthy = jax.random.normal(KEY, (16, 512, 32))
    anisotropic = healthy * jnp.concatenate([jnp.full((1,), 20.0), jnp.ones((31,)) * 0.05])
    # a narrow trajectory: 2D (a line) embedded in 32D with a little noise
    t = jnp.linspace(0.0, 1.0, 512)[None, :, None]
    narrow = jnp.concatenate([t + 1e-3 * healthy[:, :, :1], 1e-3 * healthy[:, :, 1:]], axis=-1)
    collapsed = jnp.full((16, 512, 32), 1e-3)

    print(
        f"  {'case':<14}{'coverage':>10}{'anisotropy':>12}{'eff. rank':>11}{'narrow':>9}{'gauss':>8}  risk"
    )
    for name, z in (
        ("healthy", healthy),
        ("anisotropic", anisotropic),
        ("narrow", narrow),
        ("collapsed", collapsed),
    ):
        summary = compute_exploration_quality(z)
        print(
            f"  {name:<14}{summary.coverage:>10.3f}{summary.anisotropy:>12.2f}"
            f"{summary.effective_rank:>11.2f}{summary.trajectory_narrowness:>9.3f}"
            f"{summary.gaussian_score:>8.3f}  {summary.risk_level}"
        )

    # ------------------------------------------------------------------ T2
    section("Theorem 2 - stationary vs. drifting dynamics")
    z = jax.random.normal(KEY, (4, 128, 8))
    stationary = z + 0.05 * jax.random.normal(jax.random.PRNGKey(3), z.shape)
    drift = z - jnp.exp(jnp.linspace(0.0, 8.0, 128))[None, :, None]
    for name, pred in (("stationary", stationary), ("drifting", drift)):
        r = check_stationarity(z, pred, n_layers=3)
        print(
            f"  {name:<14} trend={r.residual_trend:+.5f}  dominant_layer={r.dominant_layer}  {r.risk_level}"
        )

    # ------------------------------------------------------------------ T4
    section("Theorem 4 - planning consistency")
    plans = jax.random.normal(KEY, (6, 12, 16))
    print(
        _row(
            "identical plans R^2",
            check_planning_consistency(plans, plans).cost_consistency_r2,
            "{:.6f}",
        )
    )

    smooth = linear_latent_plan(plans[:, 0], plans[:, -1], n_steps=12)
    smooth = jnp.moveaxis(smooth, 0, 1)
    noisy = smooth + 0.5 * jax.random.normal(jax.random.PRNGKey(9), smooth.shape)
    for name, other_plan in (("straight plan", smooth), ("noisy plan", noisy)):
        c = check_planning_consistency(smooth, other_plan)
        print(
            f"  {name:<16} cost R^2={c.cost_consistency_r2:.4f}  angle={c.directional_error:.4f}  consistent={c.is_consistent}"
        )
    print(_row("straight plan path cost", float(plan_path_cost(smooth)), "{:.4f}"))
    print("  (the straight latent plan is by construction the cost-minimal one, which is")
    print("   why temporal straightening is the useful inductive bias for latent planning)")

    # ------------------------------------------------------------ combined
    section("combined health check")
    exploration, stationarity = health_check(healthy, healthy, n_layers=3, context="example")
    print(f"  exploration: {exploration}")
    print(f"  stationarity: {stationarity}")


if __name__ == "__main__":
    main()
