"""Experiment 03 - robustness sweep for the adaptation-speed vs. imagination gap.

Experiment 01 found, on a single seed, that adaptation speed and one-step accuracy
track each other (r=+0.98) but neither predicts closed-loop imagination, and that
imagination collapses to the persistence baseline out of the meta-training range.
This script hardens that into a **pre-declared protocol** and reports uncertainty.

Protocol (fixed before running; do not tune post hoc)
-----------------------------------------------------
Task          : CircleWorld, horizon=10, context seq=32, batch=32, obs noise=0.02
Meta-training : omega ~ U[0.5, 2.0], 40 Reptile iters, inner_lr=3e-3,
                inner_steps=5, outer_lr=0.3  (the repository defaults)
Evaluation    : in-range  omega in {0.5, 0.8, 1.2, 1.6, 2.0}
                out-range omega in {0.2, 0.3, 2.6, 3.2}
Models        : d_model in {16, 32}   (d_state=8, expand=2, n_heads=4, n_layers=3)
Seeds         : {0, 1, 2}
Reported      : steps_to_success, adaptation AUC, one-step RMSE, closed-loop RMSE,
                persistence RMSE, imagination gain = persistence - closed_loop
Uncertainty   : per-cell mean +/- std over seeds; bootstrap 95% CI for correlations

Raw results are cached in ``experiments/results/03_robustness.json`` (re-runs reuse
them unless ``--force`` is passed).

Run:  python experiments/03_robustness_sweep.py [--force]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

from common import evaluate_latent_world_model
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta import (
    CircleWorldDistribution,
    CircleWorldTask,
    MetaPipelineConfig,
    evaluate_adaptation_speed,
    run_meta_pipeline,
)

# ---- pre-declared protocol ------------------------------------------------- #
HORIZON = 10
SEQ, BATCH, NOISE = 32, 32, 0.02
IN_RANGE = (0.5, 0.8, 1.2, 1.6, 2.0)
OUT_RANGE = (0.2, 0.3, 2.6, 3.2)
SIZES = (16, 32)
SEEDS = (0, 1, 2)
META_ITERS, INNER_LR, INNER_STEPS, OUTER_LR = 40, 3e-3, 5, 0.3
RESULTS = Path(__file__).resolve().parent / "results" / "03_robustness.json"


def build_model(d_model: int, seed: int) -> MultiScaleLatentPredictor:
    cfg = MultiScaleSsmConfig(
        d_model=d_model, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    return MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))


def task(omega: float) -> CircleWorldTask:
    return CircleWorldTask(omega=omega, noise_level=NOISE, n_support=16, n_query=32, batch_size=4)


def measure(model, omega: float) -> dict:
    rep = evaluate_adaptation_speed(model, task(omega), inner_lr=INNER_LR, max_steps=20)
    wm = evaluate_latent_world_model(model, omega=omega, horizon=HORIZON, seq=SEQ, batch=BATCH)
    return {
        "steps_to_success": rep.steps_to_success,
        "adapt_auc": rep.adaptation_auc,
        "one_step_rmse": wm["one_step_rmse"],
        "closed_loop_rmse": wm["closed_loop_rmse"],
        "persistence_rmse": wm["persistence_rmse"],
        "imagination_gain": wm["imagination_gain"],
    }


def run_sweep() -> list[dict]:
    rows: list[dict] = []
    omegas = IN_RANGE + OUT_RANGE
    for d_model in SIZES:
        for seed in SEEDS:
            t0 = time.time()
            model = build_model(d_model, seed)
            before = {o: measure(model, o) for o in omegas}
            dist = CircleWorldDistribution(
                min_omega=0.5, max_omega=2.0, n_support=16, n_query=32, batch_size=4
            )
            pcfg = MetaPipelineConfig(
                num_iterations=META_ITERS,
                inner_lr=INNER_LR,
                inner_steps=INNER_STEPS,
                outer_lr=OUTER_LR,
                max_eval_steps=20,
            )
            run_meta_pipeline(model, dist, [task(o) for o in IN_RANGE], pcfg, verbose=False)
            after = {o: measure(model, o) for o in omegas}
            for o in omegas:
                rows.append(
                    {
                        "d_model": d_model,
                        "seed": seed,
                        "omega": o,
                        "in_range": o in IN_RANGE,
                        "before": before[o],
                        "after": after[o],
                    }
                )
            print(f"  d_model={d_model} seed={seed}: {time.time() - t0:.0f}s", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# statistics (stdlib only, so this stays dependency-free)
# --------------------------------------------------------------------------- #
def mean(xs) -> float:
    return sum(xs) / len(xs)


def std(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(a, b) -> float:
    ma, mb = mean(a), mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) + 1e-12
    return num / den


def bootstrap_ci(a, b, *, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the Pearson correlation of paired samples."""
    import random

    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        aa = [a[i] for i in idx]
        bb = [b[i] for i in idx]
        if len(set(aa)) < 2 or len(set(bb)) < 2:
            continue
        vals.append(pearson(aa, bb))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return lo, hi


def summarize(rows: list[dict]) -> None:
    print("\nPer-cell mean +/- std over seeds (after meta-training):")
    print(
        f"  {'d_model':>7}{'omega':>7}{'range':>10}{'steps':>8}{'AUC':>9}"
        f"{'1-step':>9}{'closed':>9}{'persist':>9}{'gain':>9}"
    )
    for d_model in SIZES:
        for o in IN_RANGE + OUT_RANGE:
            cell = [r["after"] for r in rows if r["d_model"] == d_model and r["omega"] == o]

            gains = [c["imagination_gain"] for c in cell]
            steps = {c["steps_to_success"] for c in cell}
            steps_str = "never" if None in steps else str(min(steps))
            print(
                f"  {d_model:>7}{o:>7.1f}{'in' if o in IN_RANGE else 'out':>10}"
                f"{steps_str:>8}{mean([c['adapt_auc'] for c in cell]):>9.4f}"
                f"{mean([c['one_step_rmse'] for c in cell]):>9.4f}"
                f"{mean([c['closed_loop_rmse'] for c in cell]):>9.4f}"
                f"{mean([c['persistence_rmse'] for c in cell]):>9.4f}"
                f"{mean(gains):>+9.4f}"
            )

    # correlations over every (size, seed, omega) after meta-training
    aft = [r["after"] for r in rows]
    auc = [a["adapt_auc"] for a in aft]
    one = [a["one_step_rmse"] for a in aft]
    gain = [a["imagination_gain"] for a in aft]

    print(f"\nCorrelations after meta-training (n = {len(aft)}, bootstrap 95% CI):")
    for name, x, y in (
        ("adapt AUC   ~ one-step RMSE", auc, one),
        ("adapt AUC   ~ imagination gain", auc, gain),
        ("one-step    ~ imagination gain", one, gain),
    ):
        lo, hi = bootstrap_ci(x, y)
        print(f"  {name:<32} r = {pearson(x, y):+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\nIn-range vs out-of-range imagination gain (after meta-training):")
    for name, sel in (
        ("in-range", lambda r: r["in_range"]),
        ("out-of-range", lambda r: not r["in_range"]),
    ):
        g = [r["after"]["imagination_gain"] for r in rows if sel(r)]
        pos = sum(x > 0 for x in g)
        print(
            f"  {name:<13} mean = {mean(g):+.4f} +/- {std(g):.4f}"
            f"   beats persistence in {pos}/{len(g)} cells"
        )

    print("\nSame, before meta-training (control):")
    for name, sel in (
        ("in-range", lambda r: r["in_range"]),
        ("out-of-range", lambda r: not r["in_range"]),
    ):
        g = [r["before"]["imagination_gain"] for r in rows if sel(r)]
        pos = sum(x > 0 for x in g)
        print(f"  {name:<13} mean = {mean(g):+.4f}   beats persistence in {pos}/{len(g)} cells")


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print(
            f"loaded {len(rows)} cached rows from {RESULTS.relative_to(RESULTS.parents[2])}"
            " (pass --force to recompute)"
        )
    else:
        print(
            f"running sweep: |sizes|={len(SIZES)} |seeds|={len(SEEDS)} "
            f"|omegas|={len(IN_RANGE) + len(OUT_RANGE)}"
        )
        rows = run_sweep()
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS.relative_to(RESULTS.parents[2])}")
    summarize(rows)
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
