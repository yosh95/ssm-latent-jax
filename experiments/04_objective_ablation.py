"""Experiment 04 - is the closed-loop breakdown a *property of the objective*?

Experiment 01/03 showed that a JEPA world model can adapt fast, score well one step
ahead, and still fail to imagine.  Before calling that a property of JEPA, we have to
rule out the boring explanation: maybe closing the loop is just hard, full stop.

This experiment trains the *same architecture* with three objectives, on the same
CircleWorld distribution, and measures them identically:

  * ``mse``    - observation-space multi-timescale Mamba predictor (no latent, no
                 JEPA, no regulariser): the plain control.  Scored in obs space.
  * ``vicreg`` - the JEPA latent model with its default `stability_loss` (VICReg
                 moment matching) regulariser.
  * ``sigreg`` - the same JEPA latent model, but the regulariser replaced by the
                 SIGReg / LeJEPA characteristic-function objective.

If the closed-loop breakdown is *specific to the objective*, the three will differ;
if not, all three collapse to the persistence baseline out of range and the story is
"closing the loop is hard", not "JEPA is broken".

We also report a representation-collapse probe (`sigreg_loss` of the learned
latents, and their per-dimension variance) so that "did the objective do its job?"
and "can it imagine?" can be told apart.

NOTE: the latent models are scored in *latent-decoded* observation space, so their
one-step numbers are not directly comparable to the obs-space `mse` model's (a
different decoder).  The comparison that matters here is the *closed-loop* column.

Preliminary.  Run:  python experiments/04_objective_ablation.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import optax
from common import (
    evaluate_latent_world_model,
    evaluate_mamba_predictor,
)
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.data import circle_world, circle_world_actions
from ssm_latent.latent import (
    LatentLossArgs,
    MultiScaleLatentPredictor,
    lejepa_loss,
    sigreg_loss,
)
from ssm_latent.predictor import MultiScaleMambaPredictor

STEPS = 600
BATCH, SEQ = 32, 32
HORIZON = 10
OMEGA_GRID = (0.3, 0.8, 1.2, 1.6, 2.5)
OMEGA_TRAIN = 1.6
SEEDS = (0, 1)


def _cfg():
    return MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_latent(objective: str, seed: int):
    model = MultiScaleLatentPredictor(_cfg(), input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = circle_world(BATCH, SEQ, omega=OMEGA_TRAIN)
    act = circle_world_actions(BATCH, SEQ, omega=OMEGA_TRAIN)
    proj = jax.lax.stop_gradient(model.stability_projections[...])

    @nnx.jit
    def step(model, opt, obs, act):
        def loss_fn(model):
            z, pred_z, recon_x, pred_x = model(obs, act)
            base = LatentLossArgs(z, pred_z, recon_x, pred_x, obs)
            if objective == "sigreg":
                # keep the *same* scaffolding as `latent_loss`, swap VICReg -> SIGReg
                return lejepa_loss(z, pred_z, proj, sigreg_weight=1.0)
            # vicreg: the library's own objective (SIGReg-free)
            return model.loss(base)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for _ in range(STEPS):
        step(model, opt, obs, act)
    return model


def train_mse(seed: int):
    model = MultiScaleMambaPredictor(_cfg(), input_dim=2, output_dim=2, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = circle_world(BATCH, SEQ, omega=OMEGA_TRAIN)
    act = circle_world_actions(BATCH, SEQ, omega=OMEGA_TRAIN)

    @nnx.jit
    def step(model, opt, obs, act):
        def loss_fn(model):
            return MultiScaleMambaPredictor.loss(model.forward_with_action(obs, act), obs)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for _ in range(STEPS):
        step(model, opt, obs, act)
    return model


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def latent_probe(model, omega: float) -> dict:
    obs = circle_world(BATCH, SEQ, omega=omega)
    z = model.encode(obs)
    var = jnp.var(z.reshape(-1, z.shape[-1]), axis=0)
    return {
        "sigreg_on_latents": float(sigreg_loss(z, model.stability_projections[...])),
        "latent_var_mean": float(var.mean()),
        "latent_var_min": float(var.min()),
    }


def run_latent(objective: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        model = train_latent(objective, seed)
        for omega in OMEGA_GRID:
            wm = evaluate_latent_world_model(model, omega=omega, horizon=HORIZON)
            probe = latent_probe(model, omega)
            rows.append({"objective": objective, "seed": seed, "omega": omega, **wm, **probe})
    return rows


def run_mse() -> list[dict]:
    rows = []
    for seed in SEEDS:
        model = train_mse(seed)
        for omega in OMEGA_GRID:
            wm = evaluate_mamba_predictor(model, omega=omega, horizon=HORIZON)
            rows.append({"objective": "mse", "seed": seed, "omega": omega, **wm})
    return rows


def mean(xs) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    t0 = time.time()
    rows = run_mse() + run_latent("vicreg") + run_latent("sigreg")

    print("Closed-loop imagination gain (persistence - closed), mean over seeds:")
    print(f"  {'objective':<9}" + "".join(f"{o:>9.1f}" for o in OMEGA_GRID))
    for obj in ("mse", "vicreg", "sigreg"):
        cells = [
            mean([r["imagination_gain"] for r in rows if r["objective"] == obj and r["omega"] == o])
            for o in OMEGA_GRID
        ]
        print(f"  {obj:<9}" + "".join(f"{c:>+9.4f}" for c in cells))

    print("\nClosed-loop RMSE (lower is better; persistence is the baseline):")
    pers = [mean([r["persistence_rmse"] for r in rows if r["omega"] == o]) for o in OMEGA_GRID]
    print(f"  {'persist':<9}" + "".join(f"{p:>9.4f}" for p in pers))
    for obj in ("mse", "vicreg", "sigreg"):
        cells = [
            mean([r["closed_loop_rmse"] for r in rows if r["objective"] == obj and r["omega"] == o])
            for o in OMEGA_GRID
        ]
        print(f"  {obj:<9}" + "".join(f"{c:>9.4f}" for c in cells))

    print("\nRepresentation probe (latent objectives only):")
    print(f"  {'objective':<9}{'sigreg(z)':>12}{'var_mean':>11}{'var_min':>10}")
    for obj in ("vicreg", "sigreg"):
        s = mean([r["sigreg_on_latents"] for r in rows if r["objective"] == obj])
        vm = mean([r["latent_var_mean"] for r in rows if r["objective"] == obj])
        vn = min(r["latent_var_min"] for r in rows if r["objective"] == obj)
        print(f"  {obj:<9}{s:>12.4f}{vm:>11.4f}{vn:>10.4f}")
    print("  (sigreg(z) ~ 0 => latents look Gaussian; large var_min => no dead dims)")

    print("\n  NOTE: the latent models are scored through a decoder, so their one-step")
    print("  numbers are not comparable to the obs-space `mse` model.  Compare the")
    print("  closed-loop columns, and the *sign* of the gain.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
