"""Example 3 - the CircleWorld benchmark and latent planning.

Predict the (x, y) position of a point rotating at constant angular velocity,
20 steps into the future - the deceptively simple physical task that shows
*neither* a plain Mamba predictor in observation space *nor* a single-timescale
latent predictor suffices, while a multi-timescale SSM x JEPA model does.

The script trains all three variants, reports one-step and 20-step errors, and
then exercises the latent-planning utilities (temporal straightening, plan path
cost and planning consistency, Theorem 4 of Klindt/LeCun/Balestriero 2026).

Run: ``python examples/03_circle_world_planning.py``
"""

import jax
import jax.numpy as jnp
import optax
from common import banner, rmse, section
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig, SsmConfig
from ssm_latent.data import circle_world, circle_world_actions
from ssm_latent.latent import LatentLossArgs, MultiScaleLatentPredictor, curvature_loss
from ssm_latent.monitor import check_planning_consistency, linear_latent_plan, plan_path_cost
from ssm_latent.predictor import MambaPredictor
from ssm_latent.utils import count_parameters

OMEGA = 1.6
STEPS = 800
BATCH = 32
SEQ = 32


def train_mamba_predictor(seed: int = 0):
    """Reference baseline: next-step MSE directly in observation space."""
    cfg = SsmConfig(d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, input_dim=2, output_dim=2, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = circle_world(BATCH, SEQ, omega=OMEGA)

    @nnx.jit
    def step(model, optimizer, obs):
        def loss_fn(model):
            return MambaPredictor.loss(model(obs), obs)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    loss = float("nan")
    for _ in range(STEPS):
        loss = float(step(model, optimizer, obs))
    return model, loss


def train_latent_predictor(n_layers: int, seed: int = 0):
    """JEPA objective with ``n_layers`` (multi-)timescale SSM layers."""
    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=n_layers, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = circle_world(BATCH, SEQ, omega=OMEGA)
    act = circle_world_actions(BATCH, SEQ, omega=OMEGA)

    @nnx.jit
    def step(model, optimizer, obs, act):
        def loss_fn(model):
            z, pred_z, recon_x, pred_x = model(obs, act)
            return model.loss(
                LatentLossArgs(
                    z=z,
                    pred_z=pred_z,
                    reconstructed_x=recon_x,
                    predicted_x=pred_x,
                    original_x=obs,
                    stability_weight=0.05,
                    curvature_weight=0.01,
                    recon_weight=1.0,
                )
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    loss = float("nan")
    for _ in range(STEPS):
        loss = float(step(model, optimizer, obs, act))
    return model, loss


def latent_rollout(model, obs, act, horizon: int):
    """Closed-loop latent rollout decoded back into observation space.

    Returns ``[B, horizon, 2]`` predictions for the steps *after* ``obs``.
    """
    z, _, _, _ = model(obs, act)
    state = model.zero_state(obs.shape[0])
    z_run = z[:, -1]
    decoded = []
    for _ in range(horizon):
        z_run, state = model.step(z_run, act[:, -1], state)
        decoded.append(model.decode(z_run))
    return jnp.stack(decoded, axis=1)


def main() -> None:
    banner("CircleWorld: phase-locked prediction and latent planning")

    section("training the three variants")
    mamba, mamba_loss = train_mamba_predictor()
    single, single_loss = train_latent_predictor(n_layers=1)
    multi, multi_loss = train_latent_predictor(n_layers=3)
    print(
        f"  Mamba-only (obs space)  params={count_parameters(mamba):6d}  final train loss={mamba_loss:.5f}"
    )
    print(
        f"  JEPA single-scale       params={count_parameters(single):6d}  final train loss={single_loss:.5f}"
    )
    print(
        f"  JEPA multi-scale        params={count_parameters(multi):6d}  final train loss={multi_loss:.5f}"
    )

    section("prediction error on an unseen phase")
    obs = circle_world(BATCH, SEQ, omega=OMEGA, phase_shift=1.1)
    act = circle_world_actions(BATCH, SEQ, omega=OMEGA, phase_shift=1.1)

    mamba_pred = mamba(obs)
    single_pred = single(obs, act)[3]
    multi_pred = multi(obs, act)[3]

    print(
        f"  one-step RMSE : mamba={rmse(mamba_pred[:, :-1], obs[:, 1:]):.4f}  "
        f"jepa-1x={rmse(single_pred[:, :-1], obs[:, 1:]):.4f}  "
        f"jepa-3x={rmse(multi_pred[:, :-1], obs[:, 1:]):.4f}"
    )

    print("  closed-loop latent rollout (the model consumes its own predictions):")
    for horizon in (5, 10, 20):
        phase0 = 1.1 + OMEGA * 0.3 * SEQ
        truth = circle_world(BATCH, horizon, omega=OMEGA, phase_shift=phase0)
        persistence = obs[:, -1:].repeat(horizon, axis=1)  # "nothing changes" baseline
        row = [f"    horizon {horizon:2d}: persistence={rmse(persistence, truth):.4f}"]
        for name, model in (("jepa-1x", single), ("jepa-3x", multi)):
            pred = latent_rollout(model, obs, act, horizon)
            row.append(f"{name}={rmse(pred, truth):.4f}")
        print("  ".join(row))
    print("  note: a JEPA model trained one step ahead has no mechanism against")
    print("  exposure bias, so closed-loop drift is expected.")
    print("  MultiScaleMambaPredictor.forward_with_h_sampling is one candidate remedy,")
    print("  but experiments/02_h_sampling_remedy.py does NOT find it helping here: in")
    print("  our runs it made both one-step and closed-loop error worse.")

    section("latent trajectory straightness (temporal straightening)")
    z_multi = multi(obs, act)[0]
    z_single = single(obs, act)[0]
    print(f"  curvature_loss(z) multi-scale : {float(curvature_loss(z_multi)):.6f}")
    print(f"  curvature_loss(z) single-scale: {float(curvature_loss(z_single)):.6f}")

    section("latent planning (Theorem 4)")
    plan = linear_latent_plan(z_multi[:, 0], z_multi[:, -1], n_steps=10)
    print(
        f"  straight-line plan shape: {tuple(plan.shape)}, path cost {float(plan_path_cost(plan))}"
    )

    # round-trip the plan through observation space and re-encode it
    flat = plan.reshape(-1, plan.shape[-1])
    x_plan = multi.decode(flat).reshape(plan.shape[0], plan.shape[1], -1)
    z_from_x = multi.encode(x_plan)
    # `check_planning_consistency` expects [n_plans, plan_len, d_model] for both
    consistency = check_planning_consistency(jnp.moveaxis(plan, 0, 1), jnp.moveaxis(z_from_x, 0, 1))
    print(f"  cost consistency R^2     : {consistency.cost_consistency_r2:.4f}")
    print(f"  directional error (rad)  : {consistency.directional_error:.4f}")
    print(f"  consistent               : {consistency.is_consistent}")


if __name__ == "__main__":
    main()
