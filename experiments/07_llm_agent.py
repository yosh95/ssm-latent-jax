"""Experiment 07 - the LLM *runs the experiments* (hypothesis -> trial -> revise).

Experiment 06 asked the LLM for one curriculum, up front, and it landed on the same
obvious "widen the range" move a human would make - no better.  A single answer is
not "driving".  This experiment gives the frozen LLM a *steering wheel*: a
`run_trial` tool it can call as many times as its budget allows, reading the score of
each attempt and choosing the next one itself.  Nothing is shared with the world
model but text: the LLM never sees a gradient, a weight, or a logit.

The question is not "does the LLM solve it" but "does *operating a simulator* let a
frozen base model reach somewhere it could not by answering alone?"  So we measure the
best trial the LLM finds under a fixed budget, against:

  * ``default`` - the repository's own curriculum,
  * ``wide``    - the obvious human heuristic (widen the training range),
  * ``llm-1shot`` - the LLM's single-shot answer from experiment 06,

with identical compute per trial.  Honest outcome either way: if the closed-loop
driver cannot beat the scripted baselines, loose coupling has not yet bought anything
on this task, and the next lever is not "ask the LLM better" but the adaptation
mechanism itself.

Protocol
--------
Fixed per trial : 40 Reptile iters, inner_lr 3e-3, seed 0 (so trials are comparable)
LLM controls    : the training omegas and the adaptation budget k only
Budget          : at most 10 trials
Score           : mean out-of-range imagination gain on omega = {0.25, 3.0}
                  (positive = imagination beats persistence on both)
Raw results cached in experiments/results/07_llm_agent.json (--force to recompute)

Run:  python experiments/07_llm_agent.py [--force]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from run_trial import run_trial

RESULTS = Path(__file__).resolve().parent / "results" / "07_llm_agent.json"
MAX_TRIALS = 10

TOOL = [
    {
        "type": "function",
        "function": {
            "name": "run_trial",
            "description": (
                "Meta-train a small latent world model on a chosen set of angular "
                "velocities and score how well it *imagines* three fixed target tasks in "
                "closed loop. Returns per-target gains (persistence_rmse - closed_loop_rmse; "
                "positive = the model beats the 'nothing changes' baseline) and a summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "omegas": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "training angular velocities, 4-6 values",
                    },
                    "k": {
                        "type": "integer",
                        "description": "adaptation steps allowed, 0-10 (default 5)",
                    },
                },
                "required": ["omegas"],
            },
        },
    }
]

SYSTEM = (
    "You are tuning a small latent world model of 2-D circular motion so that it can "
    "IMAGINE (roll out in closed loop) three target dynamics it was not trained on: "
    "omega = 0.25, 1.2 and 3.0. You do not have the model's weights; you can only call "
    "run_trial to try a training curriculum and read back scores. Your goal is to make "
    "mean_out_of_range_gain (averaged over omega = 0.25 and 3.0) as high as possible, "
    "and ideally positive. Use your trial budget (at most 10) to probe, then exploit. "
    "When you are done, stop calling tools and reply with a one-line summary of what "
    "worked."
)


def _chat(messages: list[dict]) -> dict:
    url = os.environ["LLM_CLI_API_URL"].rstrip("/")
    payload = {
        "model": os.environ["LLM_CLI_MODEL"],
        "messages": messages,
        "tools": TOOL,
        "max_tokens": 8000,
        "temperature": 0.3,
    }
    req = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["LLM_CLI_API_KEY"],
        },
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]


def run_agent() -> dict:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Begin. Call run_trial to try your first curriculum."},
    ]
    trials = []
    for _turn in range(MAX_TRIALS * 2):  # allow assistant<->tool turns
        msg = _chat(messages)
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
        if not calls:
            return {
                "trials": trials,
                "final_message": msg.get("content"),
                "stopped": "no_tool_call",
            }
        for call in calls:
            spec = json.loads(call["function"]["arguments"])
            spec.setdefault("k", 5)
            spec["iters"] = 40
            spec["seed"] = 0
            result = run_trial(spec)
            trials.append(result)
            print(
                f"  trial {len(trials):2d}: omegas={spec['omegas']} k={spec['k']} "
                f"-> meanOOD={result['summary']['mean_out_of_range_gain']:+.3f}",
                flush=True,
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result["summary"]),
                }
            )
            if len(trials) >= MAX_TRIALS:
                messages.append(
                    {
                        "role": "user",
                        "content": "Trial budget exhausted. Stop calling tools and summarise.",
                    }
                )
    return {"trials": trials, "final_message": None, "stopped": "budget"}


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        data = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        data = {"agent": run_agent()}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(data, indent=1))
        print(f"saved raw results to {RESULTS}")

    trials = data["agent"]["trials"]
    print(f"\nLLM ran {len(trials)} trials. Best curricula by mean out-of-range gain:")
    ranked = sorted(trials, key=lambda r: r["summary"]["mean_out_of_range_gain"], reverse=True)
    for r in ranked[:5]:
        s = r["summary"]
        print(
            f"  omegas={r['spec']['omegas']} k={r['spec']['k']}  "
            f"meanOOD={s['mean_out_of_range_gain']:+.3f}  "
            f"usableOOD={s['usable_out_of_range']}  in={s['in_range_gain']:+.3f}"
        )

    best = ranked[0]["summary"]["mean_out_of_range_gain"]
    print(f"\nBest mean out-of-range gain found by the LLM: {best:+.3f}")
    print("Compare (from experiments/06_llm_driver.py, same per-trial compute):")
    print("  default  meanOOD = -0.268")
    print("  wide     meanOOD = -0.252   (the obvious human heuristic)")
    print("  llm-1shot meanOOD = -0.289  (single answer, no experiments)")
    verdict = "BEATS" if best > -0.252 else "does NOT beat"
    print(f"  <- the searching LLM {verdict} the scripted 'wide' baseline.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
