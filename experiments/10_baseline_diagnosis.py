"""Experiment 10 - is the out-of-range breakdown the model's fault, or the baseline's?

Experiment 09 showed no mechanism lever helps, which left a suspicion: maybe the
evaluation is unfair at low omega.  At omega=0.25 the point barely moves, so "repeat
the last frame" (persistence) is nearly correct, and a *linear* extrapolation (assume
the last velocity continues) is even better.  This experiment separates the two:

  1. how strong the baselines are as a function of omega (`persistence`, `linear`),
  2. whether the model's error is in *latent* space or added by the decoder, and
  3. whether the latent actually drifts when it should stay still.

Finding (see FINDINGS.md): at omega=0.25 persistence (0.32) and linear extrapolation
(0.11) are both much stronger than the model (0.70) - so a "model loses to persistence"
headline is partly an artefact of an easy baseline.  But the model *also* has a real
defect: its latent travels ~0.38 while the true latent barely moves, and the rollout
error (0.79) exceeds that travel - the model fails to *hold still* on a low-omega task.

Run:  python experiments/10_baseline_diagnosis.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import methods as M

RESULTS = Path(__file__).resolve().parent / "results" / "10_baseline.json"
CURRICULUM = [0.8, 1.2, 1.6]
OMEGAS = (0.25, 0.5, 1.2, 2.0, 3.0)


def rmse(a, b) -> float:
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


def measure(model, omega, *, horizon=10, seq=32, batch=32) -> dict:
    obs = M.circle_world(batch, seq, omega=omega)
    act = M.circle_world_actions(batch, seq, omega=omega)
    future = M.circle_world(batch, horizon, omega=omega, phase_shift=omega * 0.3 * seq)

    # model, closed loop
    z = model.encode(obs)
    state = model.zero_state(batch)
    z_run = z[:, -1]
    decoded = []
    for _ in range(horizon):
        z_run, state = model.step(z_run, act[:, -1], state)
        decoded.append(model.decode(z_run))
    rollout = jnp.stack(decoded, axis=1)

    # baselines
    persistence = jnp.repeat(obs[:, -1:], horizon, axis=1)
    last_v = obs[:, -1] - obs[:, -2]
    steps = jnp.arange(1, horizon + 1, dtype=last_v.dtype)[None, :, None]
    linear = obs[:, -1][:, None, :] + last_v[:, None, :] * steps

    # latent diagnostics
    z_true = model.encode(future)
    z_roll = []
    zzz = model.encode(obs)
    state = model.zero_state(batch)
    z_run = zzz[:, -1]
    for _ in range(horizon):
        z_run, state = model.step(z_run, act[:, -1], state)
        z_roll.append(z_run)
    z_roll = jnp.stack(z_roll, axis=1)

    return {
        "omega": omega,
        "model_rmse": rmse(rollout, future),
        "persistence_rmse": rmse(persistence, future),
        "linear_rmse": rmse(linear, future),
        "latent_rollout_err": rmse(z_roll, z_true),
        "latent_travel": rmse(z_true, zzz[:, -1:]),  # how far the true latent moves
    }


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        model = M.build_model(0)
        M.reptile_train(model, CURRICULUM, objective="oneshot", iters=40)
        rows = [measure(model, o) for o in OMEGAS]
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    print("\nClosed-loop error vs baselines (lower is better):")
    print(f"  {'omega':>6}{'model':>10}{'persist':>10}{'linear':>10}")
    for r in rows:
        print(
            f"  {r['omega']:>6.2f}{r['model_rmse']:>10.4f}{r['persistence_rmse']:>10.4f}"
            f"{r['linear_rmse']:>10.4f}"
        )

    print("\nLatent diagnostics:")
    print(f"  {'omega':>6}{'latent roll err':>17}{'latent travel':>15}{'err/travel':>12}")
    for r in rows:
        ratio = r["latent_rollout_err"] / (r["latent_travel"] + 1e-9)
        print(
            f"  {r['omega']:>6.2f}{r['latent_rollout_err']:>17.4f}{r['latent_travel']:>15.4f}"
            f"{ratio:>12.2f}"
        )

    print("\n  Read: 'linear' is a strong baseline at low omega (the task is nearly")
    print("  straight-line), so 'loses to persistence' overstates the failure there.")
    print("  But err/travel >> 1 means the latent drifts more than the task moves -")
    print("  a genuine defect: the model cannot hold still when it should.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
