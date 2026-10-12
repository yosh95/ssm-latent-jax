"""Experiment 06 - the LLM *drives* the world model (loose coupling, closed loop).

The picture
-----------
A frozen LLM is the "human"; the world model is the "car".  The human cannot run any
faster than a human, but with a car it reaches places it could not otherwise go.
Here the LLM does not touch a single weight of the language model and shares nothing
with the world model - it only *chooses* and *reads back numbers*.  What it chooses is
the **training curriculum** for the world model (which dynamics to meta-train on) and
the adaptation budget, and what it reads back is how well the resulting model
*imagines* the target tasks in closed loop.

Why curriculum, and not a differentiable hook
---------------------------------------------
An algorithmic seam (logits -> world model) would be bounded by the base model's own
knowledge.  Asking the LLM to *operate* an external simulator instead is the loose
seam: the value shows up as tasks the pair can reach, not as gradients.

The honest control
------------------
The obvious human heuristic for "make it work outside the training range" is to
*widen the range*.  So we race the LLM against it:

  * ``default``  - omega ~ U[0.5, 2.0] (the repository's own curriculum)
  * ``wide``     - omega spread over U[0.2, 3.2] (the obvious human move)
  * ``llm-r1``   - the LLM's first proposal, given the targets and no feedback
  * ``llm-r2``   - the LLM's revised proposal, after seeing the round-1 numbers

If the LLM does not beat ``wide``, it added nothing over an obvious human heuristic -
which is the result that would matter most (in either direction).

Protocol (fixed before running)
-------------------------------
Targets : in-range {1.2}; out-of-range {0.25, 3.0}
Budget  : 40 Reptile iters, inner 5 steps, lr 3e-3, outer 0.3 (repository defaults)
Judge   : closed-loop imagination gain (persistence - closed) on the query, horizon 10
Rounds  : 2 (the LLM sees round-1 numbers before proposing round 2)
Raw results cached in experiments/results/06_llm_driver.json (--force to recompute)

Run:  python experiments/06_llm_driver.py [--force]
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import jax
from agent import build_world_model
from common import evaluate_latent_world_model

from ssm_latent.meta import (
    CircleWorldTask,
    MetaPipelineConfig,
    TaskDistribution,
    run_meta_pipeline,
)

TARGETS = {1.2: "in", 0.25: "out", 3.0: "out"}
HORIZON = 10
RESULTS = Path(__file__).resolve().parent / "results" / "06_llm_driver.json"


# --------------------------------------------------------------------------- #
# a distribution over an explicit list of omegas (what the driver picks)
# --------------------------------------------------------------------------- #
class FixedDistribution(TaskDistribution):
    """Samples tasks from a fixed list of omegas the driver chose (key-deterministic)."""

    def __init__(self, omegas):
        self.omegas = [float(o) for o in omegas]
        self.description = "driver-chosen omegas " + ", ".join(f"{o:.2f}" for o in self.omegas)
        self.n_families = 1

    def sample_task(self, key: jax.Array) -> CircleWorldTask:
        idx = int(jax.random.randint(key, (), 0, len(self.omegas)))
        return CircleWorldTask(omega=self.omegas[idx], n_support=16, n_query=32, batch_size=4)


# --------------------------------------------------------------------------- #
# the LLM transport (kept deliberately dumb: send text, get text)
# --------------------------------------------------------------------------- #
def chat(prompt: str, *, max_tokens: int = 8000, temperature: float = 0.3) -> str:
    """Send one prompt, return the best JSON-ish answer we can find.

    The endpoint here serves a *reasoning* model that can spend the whole budget
    thinking and return an empty ``content`` (``finish_reason="length"``).  We give it
    a generous budget and, if ``content`` is still empty, fall back to the tail of
    ``reasoning`` - which is where it eventually writes the list.  The transport stays
    dumb: text in, text out.
    """
    url = os.environ["LLM_CLI_API_URL"].rstrip("/")
    payload = {
        "model": os.environ["LLM_CLI_MODEL"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
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
        msg = json.load(r)["choices"][0]["message"]
    return (msg.get("content") or "") + "\n" + (msg.get("reasoning") or "")


def _extract_omegas(text: str) -> list[float]:
    """Pull a JSON list of floats out of a free-form reply (tolerate prose/fences)."""
    import re

    matches = re.findall(r"\[[^\[\]]*\]", text)
    for m in reversed(matches):  # prefer the last list (the conclusion, not an example)
        try:
            vals = [float(x) for x in json.loads(m)]
        except (ValueError, TypeError):
            continue
        if 3 <= len(vals) <= 8:
            return vals
    raise ValueError(f"no JSON list of 3-8 floats in reply: {text[-300:]!r}")


def prompt_curriculum(round_idx: int, prior: str | None) -> str:
    base = (
        "You are choosing the TRAINING CURRICULUM for a small latent world-model of "
        "2-D circular motion. The model is meta-trained on a set of angular velocities "
        "'omega', then adapted to a target with a few gradient steps.\n"
        "Targets it must work on: omega = 0.25, 1.2, 3.0 (it will be scored on closed-loop "
        "prediction quality).\n"
        "You may pick 4 to 6 training omegas; each meta-training run is cheap but not free.\n"
        "Reply with ONLY a JSON list of numbers, e.g. [0.6, 1.0, 1.4, 1.8]."
    )
    if prior:
        base += (
            "\n\nFeedback from the previous attempt:\n"
            + prior
            + "\nRevise the curriculum to do better. Reply with ONLY the JSON list."
        )
    return base


def run_curriculum(model_seed: int, omegas, targets) -> dict:
    model = build_world_model(model_seed)
    dist = FixedDistribution(omegas)
    pcfg = MetaPipelineConfig(
        num_iterations=40, inner_lr=3e-3, inner_steps=5, outer_lr=0.3, max_eval_steps=20
    )
    train = [
        CircleWorldTask(omega=o, n_support=16, n_query=32, batch_size=4) for o in omegas[:3]
    ] or [CircleWorldTask(omega=1.0, n_support=16, n_query=32, batch_size=4)]
    learner, _, _ = run_meta_pipeline(model, dist, train, pcfg, verbose=False)
    out = {}
    for omega in targets:
        out[omega] = evaluate_latent_world_model(learner.model, omega=omega, horizon=HORIZON)
    return out


def summarize(name: str, res: dict) -> str:
    parts = []
    for omega, label in TARGETS.items():
        g = res[omega]["imagination_gain"]
        parts.append(f"{label} omega={omega}: gain={g:+.3f}")
    mean_out = sum(res[o]["imagination_gain"] for o in TARGETS if TARGETS[o] == "out") / 2
    mean_in = res[1.2]["imagination_gain"]
    return f"{name:<10} in(1.2)={mean_in:+.3f}  mean OOD={mean_out:+.3f}  | " + "  ".join(parts)


def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        data = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        data = {"runs": {}, "driver": {}}

        for name, omegas in (
            ("default", [0.5, 0.9, 1.3, 1.7, 2.0]),
            ("wide", [0.2, 0.7, 1.2, 2.0, 3.2]),
        ):
            data["runs"][name] = {
                "omegas": omegas,
                "result": _serialise(run_curriculum(0, omegas, TARGETS)),
            }
            print(f"  {name} done ({time.time() - t0:.0f}s)", flush=True)

        prior = None
        for round_idx in (1, 2):
            reply = chat(prompt_curriculum(round_idx, prior))
            omegas = _extract_omegas(reply)
            res = run_curriculum(0, omegas, TARGETS)
            data["driver"][f"llm-r{round_idx}"] = {
                "omegas": omegas,
                "reply": reply,
                "result": _serialise(res),
            }
            print(f"  llm-r{round_idx}: omegas={omegas} ({time.time() - t0:.0f}s)", flush=True)
            prior = summarize(f"llm-r{round_idx}", res)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(data, indent=1))
        print(f"saved raw results to {RESULTS}")

    print("\nClosed-loop imagination gain (persistence - closed); higher is better.")
    print("  'mean OOD' averages omega=0.25 and 3.0 (the targets outside the default range).")
    for name in ("default", "wide", "llm-r1", "llm-r2"):
        entry = data["runs"].get(name) or data["driver"].get(name)
        if entry:
            print("  " + summarize(name, _deserialise(entry["result"])))
            print(f"      omegas: {entry['omegas']}")
    print("\\n  If 'llm-r*' does not beat 'wide', the driver added nothing over the obvious")
    print("  human heuristic; if it does, the LLM is doing more than being handed the answer.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


# json round-trip helpers (keys are floats; json needs strings)
def _serialise(res: dict) -> dict:
    return {str(k): v for k, v in res.items()}


def _deserialise(res: dict) -> dict:
    return {float(k): v for k, v in res.items()}


if __name__ == "__main__":
    main()
