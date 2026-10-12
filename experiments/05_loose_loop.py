"""Experiment 05 - the loose loop: can a *few* adaptation steps on a short demo
make a pretrained world model usable (closed loop) on a new task, including OOD?

This is the experiment the whole line of work has been building to.  The picture is
the "LLM drives a world model" one, but with the driver scripted for now:

    driver                       world model
    ------                       -----------
    declares a target task  -->  (a generic, pretrained model)
    hands over a short demo  -->  adapts for k gradient steps on the demo
    reads back numbers       <--  (how well it now *imagines* that task)

Crucially the coupling is *loose*: the driver only picks a task, sends a short
demonstration, and reads scalars.  No logits, no gradients, no shared weights - so
whatever the pair achieves is not bounded by the base model's own knowledge (cf.
arXiv:2504.13837, where RL merely re-weights what the base already has and the base
catches up as k grows).

Question
--------
Adaptation is measured the way the SAI thesis measures it (a handful of steps on a
short support set), but *success is judged in closed loop* - "did the model's
imagination beat the persistence baseline on a held-out query?" - and we split the
answer by in-range vs out-of-range omegas.  If a few steps suffice in-distribution
but not out-of-distribution, then a driver can only steer the world model near the
data it was pretrained on; if it works out-of-range too, the pair genuinely reaches
somewhere the pretrained model could not go on its own.

Protocol (fixed before running)
-------------------------------
Pretrain  : Reptile meta-training on omega ~ U[0.5, 2.0] (repository defaults)
Targets   : in-range  {0.8, 1.2, 1.8};  out-range {0.25, 3.0}
Demand    : k in {0, 1, 2, 3, 5, 10} adaptation steps on the demo, inner_lr=3e-3
Judge     : closed-loop imagination gain on the *query* (unseen phase), horizon 10
Metric    : "steps to make imagination beat persistence" per target
Raw results cached in experiments/results/05_loose_loop.json (--force to recompute)

Run:  python experiments/05_loose_loop.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
from agent import DemoTask, ScriptedDriver, TaskSpec, build_world_model
from common import evaluate_latent_world_model

from ssm_latent.meta import (
    CircleWorldDistribution,
    CircleWorldTask,
    MetaPipelineConfig,
    adapt,
    run_meta_pipeline,
)

IN_RANGE = (0.8, 1.2, 1.8)
OUT_RANGE = (0.25, 3.0)
K_STEPS = (0, 1, 2, 3, 5, 10)
SEEDS = (0, 1)
HORIZON = 10
RESULTS = Path(__file__).resolve().parent / "results" / "05_loose_loop.json"


def pretrain(seed: int):
    """The generic world model a driver would be handed."""
    model = build_world_model(seed)
    dist = CircleWorldDistribution(
        min_omega=0.5, max_omega=2.0, n_support=16, n_query=32, batch_size=4
    )
    pcfg = MetaPipelineConfig(
        num_iterations=40, inner_lr=3e-3, inner_steps=5, outer_lr=0.3, max_eval_steps=20
    )
    train_tasks = [
        CircleWorldTask(omega=o, n_support=16, n_query=32, batch_size=4) for o in (0.8, 1.2, 1.6)
    ]
    learner, _, _ = run_meta_pipeline(model, dist, train_tasks, pcfg, verbose=False)
    return learner.model


def one_cell(model, driver, spec: TaskSpec, k: int, seed: int) -> dict:
    key = jax.random.fold_in(jax.random.PRNGKey(seed), int(spec.omega * 1000) + k)
    demo = driver.demonstrate(spec, key=key)
    task = DemoTask(spec, demo)
    if k > 0:
        # `adapt` clones internally, so `model` (the pretrained prior) is untouched
        adapted, result = adapt(model, task, inner_lr=3e-3, inner_steps=k)
        support_loss = result.loss_curve[-1]
    else:
        adapted, support_loss = model, None  # k=0: the driver said nothing useful
    wm = evaluate_latent_world_model(adapted, omega=spec.omega, horizon=HORIZON)
    return {
        "omega": spec.omega,
        "in_range": spec.omega in IN_RANGE,
        "k": k,
        "seed": seed,
        "adapted_support_loss": support_loss,
        **wm,
    }


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print(f"loaded {len(rows)} cached rows (pass --force to recompute)")
    else:
        rows = []
        for seed in SEEDS:
            model = pretrain(seed)
            print(f"  pretrained seed={seed} ({time.time() - t0:.0f}s)", flush=True)
            for omega in IN_RANGE + OUT_RANGE:
                driver = ScriptedDriver([omega])
                for k in K_STEPS:
                    rows.append(one_cell(model, driver, TaskSpec(omega=omega), k, seed))
                    print(f"    omega={omega} k={k:2d} ({time.time() - t0:.0f}s)", flush=True)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    # ---- report ----
    def mean(xs):
        return sum(xs) / len(xs)

    print("\nImagination gain (persistence - closed loop) on the query; higher is better.")
    print("A cell is 'usable' when the gain is positive.")
    print(f"  {'target':>10}{'range':>6}" + "".join(f"{'k=' + str(k):>9}" for k in K_STEPS))
    for omega in IN_RANGE + OUT_RANGE:
        cells = []
        for k in K_STEPS:
            xs = [r["imagination_gain"] for r in rows if r["omega"] == omega and r["k"] == k]
            cells.append(mean(xs))
        print(
            f"  {omega:>10.2f}{'in' if omega in IN_RANGE else 'out':>6}"
            + "".join(f"{c:>+9.3f}" for c in cells)
        )

    print("\n'Usable' = imagination beats persistence (mean gain > 0), and steps to get there:")
    for omega in IN_RANGE + OUT_RANGE:
        steps = next(
            (
                k
                for k in K_STEPS
                if mean(
                    [r["imagination_gain"] for r in rows if r["omega"] == omega and r["k"] == k]
                )
                > 0
            ),
            None,
        )
        print(
            f"  omega={omega:>5.2f} ({'in ' if omega in IN_RANGE else 'out'}): "
            f"{'never' if steps is None else str(steps) + ' steps'}"
        )

    print("\n  Reading: if the OOD targets only become usable at large k (or never), the")
    print("  driver can steer the model within its pretraining range but not beyond it;")
    print("  if they become usable in a few steps, the *pair* reaches further than either")
    print("  the frozen model or a weight-free base model could.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
