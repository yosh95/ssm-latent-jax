"""Experiment 09 - (B): do the *mechanism* levers fix the out-of-range breakdown?

Experiments 05-08 showed the search over *settings* (curriculum, adaptation budget)
hits a ceiling: no setting made closed-loop imagination usable out of range.  So this
experiment changes the two things that could actually be the cause, holding the
curriculum fixed:

  objective  in {"oneshot", "rollout"}   - train to predict the next step, or train to
                                           roll out (the thing we score)
  adapt      in {"weight", "warmup"}      - a few gradient steps on a demo, or a
                                           weight-free state warm-up (in-context)

This is a 2x2 grid, each cell over 2 seeds, on fixed in-range {0.8, 1.6} and
out-of-range {0.25, 3.0} targets.  We report the mean out-of-range imagination gain
(persistence - closed) and whether it is usable (both targets positive).

Pre-declared: a lever "works" if the mean out-of-range gain is positive AND beats the
baseline cell (oneshot + weight) on the same targets.  Because exp08 taught us
single-seed conclusions do not hold, every cell is reported per seed.

Run:  python experiments/09_mechanism_ablation.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import methods as M

RESULTS = Path(__file__).resolve().parent / "results" / "09_mechanism.json"
CURRICULUM = [0.8, 1.2, 1.6]  # fixed, in-range only
IN_TARGETS = (1.2,)
OUT_TARGETS = (0.25, 3.0)
SEEDS = (0, 1)
ITERS = 40
DEMO_OMEGA = 0.25  # the demonstration the driver hands over


def cell(objective: str, adapt: str, seed: int) -> dict:
    model = M.build_model(seed)
    M.reptile_train(model, CURRICULUM, objective=objective, iters=ITERS, seed=seed)

    gain = {}
    for omega in IN_TARGETS + OUT_TARGETS:
        if adapt == "weight":
            # a few gradient steps on a demonstration of the target dynamics
            demo = M.circle_world(4, M.DEMO_LEN if hasattr(M, "DEMO_LEN") else 16, omega=DEMO_OMEGA)
            demo_a = M.circle_world_actions(4, 16, omega=DEMO_OMEGA)
            adapted = M.adapt_weight(model, demo, demo_a, k=5, objective=objective)
            gain[omega] = M.evaluate(adapted, omega)["imagination_gain"]
        elif adapt == "warmup":
            gain[omega] = M.evaluate(model, omega, warm_from=(DEMO_OMEGA, 16))["imagination_gain"]
        else:
            raise ValueError(adapt)
    ood = [gain[o] for o in OUT_TARGETS]
    return {
        "objective": objective,
        "adapt": adapt,
        "seed": seed,
        "in_range_gain": gain[IN_TARGETS[0]],
        "mean_ood_gain": sum(ood) / len(ood),
        "usable_ood": all(g > 0 for g in ood),
        "per_target": {str(k): v for k, v in gain.items()},
    }


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        rows = []
        for objective in ("oneshot", "rollout"):
            for adapt in ("weight", "warmup"):
                for seed in SEEDS:
                    r = cell(objective, adapt, seed)
                    rows.append(r)
                    print(
                        f"  {objective:<8} + {adapt:<7} seed={seed}: "
                        f"meanOOD={r['mean_ood_gain']:+.3f} "
                        f"(in={r['in_range_gain']:+.3f}, usable={r['usable_ood']})",
                        flush=True,
                    )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    def vals(objective, adapt):
        return [
            r["mean_ood_gain"] for r in rows if r["objective"] == objective and r["adapt"] == adapt
        ]

    print("\nMean out-of-range gain, 2x2 grid (per-seed, then mean):")
    print(f"  {'objective':<9}{'adapt':<9}{'seed0':>9}{'seed1':>9}{'mean':>9}{'usable':>8}")
    baseline = None
    for objective in ("oneshot", "rollout"):
        for adapt in ("weight", "warmup"):
            v = vals(objective, adapt)
            usable = sum(
                r["usable_ood"] for r in rows if r["objective"] == objective and r["adapt"] == adapt
            )
            print(
                f"  {objective:<9}{adapt:<9}{v[0]:>+9.3f}{v[1]:>+9.3f}{sum(v) / len(v):>+9.3f}"
                f"{usable:>5}/{len(v)}"
            )
            if objective == "oneshot" and adapt == "weight":
                baseline = sum(v) / len(v)

    print(f"\nBaseline (oneshot + weight): mean out-of-range gain = {baseline:+.3f}")
    print("A lever 'works' if its mean is positive AND below-baseline is negative:")
    for objective in ("oneshot", "rollout"):
        for adapt in ("weight", "warmup"):
            v = sum(vals(objective, adapt)) / len(vals(objective, adapt))
            tag = (
                "baseline"
                if (objective, adapt) == ("oneshot", "weight")
                else ("WORKS" if v > 0 and v > baseline else "no")
            )
            print(f"  {objective:<9}+ {adapt:<8} meanOOD={v:+.3f}  -> {tag}")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
