"""A tiny, sandboxed experiment tool the LLM can *call* to run the world-model loop.

This is the "steering wheel": the LLM does not get gradients or weights, only the
ability to launch a trial and read back scalars - exactly the surface a human
researcher (or a script) would use.  Keeping it a plain function/CLI means the same
tool works whether the caller is an LLM, a shell script, or a human.

A trial is: (curriculum, adaptation budget) -> closed-loop imagination quality on a
fixed set of target tasks.  The result is a single JSON blob.

Usage:
    python experiments/run_trial.py --omegas 0.5 0.9 1.3 1.7 2.0 --k 5
    python experiments/run_trial.py --spec '{"omegas":[...], "k":5, "lr":0.003, "iters":40}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
from agent import build_world_model
from common import evaluate_latent_world_model

from ssm_latent.meta import (
    CircleWorldTask,
    MetaPipelineConfig,
    run_meta_pipeline,
)

# fixed scoring targets (never change these - the score must be comparable across trials)
SCORE_TARGETS = (0.25, 1.2, 3.0)
HORIZON = 10


def _fixed_dist(omegas):
    from ssm_latent.meta import TaskDistribution

    class _D(TaskDistribution):
        def __init__(self):
            self.omegas = list(omegas)
            self.description = "trial omegas"
            self.n_families = 1

        def sample_task(self, key):
            i = int(jax.random.randint(key, (), 0, len(self.omegas)))
            return CircleWorldTask(omega=self.omegas[i], n_support=16, n_query=32, batch_size=4)

    return _D()


def run_trial(spec: dict) -> dict:
    """Run one trial and return its score.  Pure function of `spec` (deterministic)."""
    omegas = [float(o) for o in spec["omegas"]]
    k = int(spec.get("k", 5))
    lr = float(spec.get("lr", 3e-3))
    iters = int(spec.get("iters", 40))
    seed = int(spec.get("seed", 0))

    model = build_world_model(seed)
    dist = _fixed_dist(omegas)
    pcfg = MetaPipelineConfig(
        num_iterations=iters, inner_lr=lr, inner_steps=k, outer_lr=0.3, max_eval_steps=20
    )
    train = [CircleWorldTask(omega=o, n_support=16, n_query=32, batch_size=4) for o in omegas[:3]]
    run_meta_pipeline(model, dist, train, pcfg, verbose=False)

    scores = {}
    for omega in SCORE_TARGETS:
        wm = evaluate_latent_world_model(model, omega=omega, horizon=HORIZON)
        scores[str(omega)] = {"gain": wm["imagination_gain"], "closed": wm["closed_loop_rmse"]}

    ood = [scores[str(o)]["gain"] for o in SCORE_TARGETS if o != 1.2]
    return {
        "spec": spec,
        "scores": scores,
        "summary": {
            "in_range_gain": scores["1.2"]["gain"],
            "mean_out_of_range_gain": sum(ood) / len(ood),
            "usable_out_of_range": all(g > 0 for g in ood),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omegas", type=float, nargs="+")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spec", type=str)
    args = ap.parse_args()
    spec = (
        json.loads(args.spec)
        if args.spec
        else {
            "omegas": args.omegas,
            "k": args.k,
            "lr": args.lr,
            "iters": args.iters,
            "seed": args.seed,
        }
    )
    print(json.dumps(run_trial(spec)))


if __name__ == "__main__":
    main()
