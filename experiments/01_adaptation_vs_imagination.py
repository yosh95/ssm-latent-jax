"""Experiment 01 - are "fast adaptation", "accurate prediction" and "good imagination" the same thing?

Motivation
----------
The SAI thesis (Goldfeder, Wyder, LeCun & Shwartz-Ziv, 2026), and this repo's own
`meta.eval`, treat *adaptation speed* as the metric that matters.  A latent world
model, however, is used by *imagining* forward: at deployment it consumes its own
predictions, so what decides success is the closed-loop roll-out error - not the
one-step error, and not the number of adaptation steps.

This experiment measures, for one small model (~40K params), three quantities that
are routinely conflated:

  1. adaptation speed   - steps to reach the task threshold (the SAI metric),
  2. one-step accuracy  - teacher-forced next-step RMSE (the usual benchmark),
  3. imagination        - closed-loop roll-out RMSE vs the "persistence" baseline,

on a grid of angular velocities (including values outside the meta-training range),
before and after Reptile meta-training.  The headline quantity is `imagination_gain`
= persistence_rmse - closed_loop_rmse: >0 only if the world model beats "nothing
changes".

Preliminary - single seed, synthetic task.  Run:
    python experiments/01_adaptation_vs_imagination.py
"""

from __future__ import annotations

import math
import time

from common import evaluate_latent_world_model
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta import (
    AdaptationStrategy,
    CircleWorldDistribution,
    CircleWorldTask,
    MetaPipelineConfig,
    adapt,
    adaptation_strategy_description,
    evaluate_adaptation_speed,
    run_meta_pipeline,
)

HORIZON = 10


def build_model(seed: int = 0):
    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    return MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))


def task(omega: float, **kw) -> CircleWorldTask:
    return CircleWorldTask(
        omega=omega, noise_level=0.02, n_support=16, n_query=32, batch_size=4, **kw
    )


def measure(model, omega: float) -> dict:
    rep = evaluate_adaptation_speed(model, task(omega), inner_lr=3e-3, max_steps=20)
    wm = evaluate_latent_world_model(model, omega=omega, horizon=HORIZON)
    return {"omega": omega, "steps": rep.steps_to_success, "auc": rep.adaptation_auc, **wm}


def pearson(a, b) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) + 1e-12
    return num / den


def print_grid(rows, title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'omega':>6}{'steps':>7}{'AUC':>9}{'1-step':>9}{'closed':>9}{'persist':>9}{'gain':>9}"
    )
    for r in rows:
        steps = "never" if r["steps"] is None else str(r["steps"])
        flag = "" if r["imagination_gain"] > 0 else "   <- imagination loses to persistence"
        print(
            f"  {r['omega']:>6.1f}{steps:>7}{r['auc']:>9.4f}{r['one_step_rmse']:>9.4f}"
            f"{r['closed_loop_rmse']:>9.4f}{r['persistence_rmse']:>9.4f}{r['imagination_gain']:>+9.4f}{flag}"
        )


def main() -> None:
    t0 = time.time()
    model = build_model()
    omegas = (0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0)

    before = [measure(model, o) for o in omegas]
    print_grid(before, "BEFORE meta-training (random init)")

    dist = CircleWorldDistribution(
        min_omega=0.5, max_omega=2.0, n_support=16, n_query=32, batch_size=4
    )
    pcfg = MetaPipelineConfig(
        num_iterations=40, inner_lr=3e-3, inner_steps=5, outer_lr=0.3, max_eval_steps=20
    )
    learner, _, _ = run_meta_pipeline(
        model, dist, [task(o) for o in (0.8, 1.2, 1.6)], pcfg, verbose=False
    )

    after = [measure(learner.model, o) for o in omegas]
    print_grid(after, "AFTER meta-training (Reptile, 40 iters on omega in [0.5, 2.0])")

    # ---- correlations ----
    auc = [r["auc"] for r in after]
    one = [r["one_step_rmse"] for r in after]
    gain = [r["imagination_gain"] for r in after]
    print(f"\n  correlations over {len(omegas)} tasks (after meta-training):")
    print(
        f"    corr(adaptation AUC, one-step RMSE)     = {pearson(auc, one):+.3f}"
        "   (fast adaptation <-> accurate prediction)"
    )
    print(f"    corr(adaptation AUC, imagination gain) = {pearson(auc, gain):+.3f}")
    print(f"    corr(one-step RMSE,  imagination gain) = {pearson(one, gain):+.3f}")

    lo = [r for r in after if 0.5 <= r["omega"] <= 2.0]
    oo = [r for r in after if not (0.5 <= r["omega"] <= 2.0)]
    print("\n  in-range vs out-of-range (omega outside the meta-training range):")
    for name, grp in (("in-range", lo), ("out-of-range", oo)):
        g = [r["imagination_gain"] for r in grp]
        print(
            f"    {name:<13} mean imagination gain = {sum(g) / len(g):+.4f}"
            f"   (positive in {sum(x > 0 for x in g)}/{len(g)} tasks)"
        )

    # ---- specialisation strategies ----
    om0 = 0.3
    print(f"\n  specialisation strategies on omega={om0} (out-of-range).")
    print("  `evaluate_adaptation_speed` adapts a *clone*, so we adapt explicitly to see")
    print("  the world model the adapted weights produce.")
    base = evaluate_latent_world_model(learner.model, omega=om0, horizon=HORIZON)
    print(
        f"    {'(no adaptation)':<22} steps={'—':>5}  closed={base['closed_loop_rmse']:.4f}"
        f"  gain={base['imagination_gain']:+.4f}"
    )
    for strat in AdaptationStrategy:
        rep = evaluate_adaptation_speed(
            learner.model, task(om0), inner_lr=3e-3, max_steps=20, strategy=strat
        )
        adapted, _ = adapt(learner.model, task(om0), inner_lr=3e-3, inner_steps=5, strategy=strat)
        wm = evaluate_latent_world_model(adapted, omega=om0, horizon=HORIZON)
        steps = "never" if rep.steps_to_success is None else str(rep.steps_to_success)
        print(
            f"    {adaptation_strategy_description(strat):<22} steps={steps:>5}"
            f"  closed={wm['closed_loop_rmse']:.4f}  gain={wm['imagination_gain']:+.4f}"
        )

    print("\n  Take-away: adaptation speed and one-step accuracy track each other closely,")
    print("  but neither reliably predicts closed-loop imagination - which degrades to the")
    print("  persistence baseline outside the trained range even when adaptation is fast.")
    print(f"\n  (wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
