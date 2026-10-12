"""Experiment 02 - does scheduled h-sampling actually reduce closed-loop drift?

`examples/03_circle_world_planning.py` notes that a JEPA/Mamba model trained one
step ahead drifts in closed loop and points at
`MultiScaleMambaPredictor.forward_with_h_sampling` as "the remedy" - but that claim
was never measured in the repo.  This experiment measures it.

We train the same observation-space multi-timescale Mamba predictor on CircleWorld
with `h_sampling_prob` in {0.0 (off), 0.3, 0.6}, over several seeds, and report

  * one-step (teacher-forced) RMSE, and
  * closed-loop roll-out RMSE (the model consumes its own predictions),

at a horizon of 10.  If scheduled sampling helps, closed-loop RMSE should fall as
`h_sampling_prob` rises, without one-step RMSE blowing up.

Preliminary - small model, short training.  Run:
    python experiments/02_h_sampling_remedy.py
"""

from __future__ import annotations

import time

import jax
import optax
from common import evaluate_mamba_predictor
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.data import circle_world, circle_world_actions
from ssm_latent.predictor import MultiScaleMambaPredictor

STEPS = 600
BATCH, SEQ, OMEGA = 32, 32, 1.6
HORIZON = 10


def train(h_prob: float, seed: int):
    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    model = MultiScaleMambaPredictor(cfg, input_dim=2, output_dim=2, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = circle_world(BATCH, SEQ, omega=OMEGA)
    act = circle_world_actions(BATCH, SEQ, omega=OMEGA)
    base_key = jax.random.PRNGKey(seed)

    @nnx.jit
    def step(model, opt, obs, act, key):
        def loss_fn(model):
            if h_prob <= 0.0:
                pred = model.forward_with_action(obs, act)
            else:
                pred = model.forward_with_h_sampling(obs, act, h_prob, key=key)
            return MultiScaleMambaPredictor.loss(pred, obs)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for i in range(STEPS):
        step(model, opt, obs, act, jax.random.fold_in(base_key, i))
    return model


def main() -> None:
    t0 = time.time()
    probs = (0.0, 0.3, 0.6)
    seeds = (0, 1, 2)
    print("h_sampling  seed   one-step   closed-loop")
    agg = {p: {"one": [], "cl": []} for p in probs}
    for p in probs:
        for s in seeds:
            model = train(p, s)
            m = evaluate_mamba_predictor(model, omega=OMEGA, horizon=HORIZON)
            agg[p]["one"].append(m["one_step_rmse"])
            agg[p]["cl"].append(m["closed_loop_rmse"])
            print(f"{p:>9.1f}  {s:>4}   {m['one_step_rmse']:.4f}     {m['closed_loop_rmse']:.4f}")

    print("\n  mean over seeds:")
    print(f"  {'h_sampling':>11}{'one-step':>11}{'closed-loop':>13}")
    for p in probs:
        one = sum(agg[p]["one"]) / len(seeds)
        cl = sum(agg[p]["cl"]) / len(seeds)
        print(f"  {p:>11.1f}{one:>11.4f}{cl:>13.4f}")

    lo, hi = agg[0.0]["cl"], agg[0.6]["cl"]
    delta = sum(lo) / len(lo) - sum(hi) / len(hi)
    print(
        f"\n  closed-loop change from h_sampling=0.0 to 0.6: {delta:+.4f}"
        f"  ({'helps' if delta > 0 else 'does NOT help'})"
    )
    print(f"  (wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
