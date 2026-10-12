"""Experiment 08 - is the LLM's "k=10 is better" finding robust, or a seed-0 fluke?

Experiment 07 let a frozen LLM search and it found that raising the adaptation budget
(k) from 5 to 10 improved mean out-of-range imagination gain, beating the scripted
"wide" baseline - but that was seed 0, one trial per setting.  Before building on it,
check it across seeds.

Pre-declared: the finding is considered ROBUST only if, for every seed, the LLM's best
curriculum at k=10 beats (a) the same curriculum at k=5 and (b) the "wide" baseline at
its own best k.

Run:  python experiments/08_k_robustness.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from run_trial import run_trial

RESULTS = Path(__file__).resolve().parent / "results" / "08_k_robustness.json"
CURRICULA = {
    "llm-best": [0.25, 1.2, 3.0, 0.5, 2.5],  # the winner from experiment 07
    "wide": [0.2, 0.7, 1.2, 2.0, 3.2],  # the scripted human heuristic
    "default": [0.5, 0.9, 1.3, 1.7, 2.0],  # the repository curriculum
}
KS = (5, 10)
SEEDS = (0, 1, 2)


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        rows = []
        for name, omegas in CURRICULA.items():
            for k in KS:
                for seed in SEEDS:
                    spec = {"omegas": omegas, "k": k, "iters": 40, "seed": seed}
                    r = run_trial(spec)
                    rows.append(
                        {
                            "curriculum": name,
                            "k": k,
                            "seed": seed,
                            "mean_ood": r["summary"]["mean_out_of_range_gain"],
                            "in_range": r["summary"]["in_range_gain"],
                        }
                    )
                    print(
                        f"  {name:<9} k={k:2d} seed={seed}: "
                        f"meanOOD={r['summary']['mean_out_of_range_gain']:+.3f}",
                        flush=True,
                    )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    def val(name, k, seed):
        return next(
            r["mean_ood"]
            for r in rows
            if r["curriculum"] == name and r["k"] == k and r["seed"] == seed
        )

    print("\nmean out-of-range gain (positive = imagination beats persistence):")
    print(f"  {'curriculum':<10}{'k':>4}" + "".join(f"{'s' + str(s):>10}" for s in SEEDS))
    for name in CURRICULA:
        for k in KS:
            cells = [val(name, k, s) for s in SEEDS]
            print(f"  {name:<10}{k:>4}" + "".join(f"{c:>+10.3f}" for c in cells))

    print("\nDoes k=10 beat k=5 for the LLM's curriculum, per seed?")
    wins5 = 0
    for s in SEEDS:
        w5 = val("llm-best", 10, s) > val("llm-best", 5, s)
        wins5 += w5
        print(
            f"  seed {s}: k=10 {val('llm-best', 10, s):+.3f} vs k=5 "
            f"{val('llm-best', 5, s):+.3f}  -> {'yes' if w5 else 'NO'}"
        )

    print("\nDoes the LLM's best beat the 'wide' baseline (at its own best k), per seed?")
    winsw = 0
    for s in SEEDS:
        wide_best = max(val("wide", k, s) for k in KS)
        w = val("llm-best", 10, s) > wide_best
        winsw += w
        print(
            f"  seed {s}: llm-best(k=10) {val('llm-best', 10, s):+.3f} vs wide-best "
            f"{wide_best:+.3f}  -> {'yes' if w else 'NO'}"
        )

    robust = (wins5 == len(SEEDS)) and (winsw == len(SEEDS))
    print(f"\n  k=10 over k=5:    {wins5}/{len(SEEDS)} seeds")
    print(f"  beats wide:       {winsw}/{len(SEEDS)} seeds")
    print(f"  VERDICT: finding is {'ROBUST' if robust else 'NOT robust'} across seeds.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
