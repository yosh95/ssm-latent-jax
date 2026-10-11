"""Example 2 - JEPA latent predictor on a damped oscillator.

Trains a small ``MultiScaleLatentPredictor`` end to end:
encode -> fuse with action -> multi-scale SSM -> decode, with the JEPA objective
(latent prediction + reconstruction + stability + temporal straightening).

Run: ``python examples/02_latent_predictor.py``
"""

import jax
import jax.numpy as jnp
import optax
from common import banner, rmse, section
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.data import damped_oscillator
from ssm_latent.latent import LatentLossArgs, MultiScaleLatentPredictor, lejepa_loss
from ssm_latent.monitor import compute_exploration_quality
from ssm_latent.utils import count_parameters


def main() -> None:
    banner("JEPA latent world model - damped oscillator")

    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=0, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    print(f"parameters: {count_parameters(model)}")

    def make_batch(seed):
        obs = damped_oscillator(batch_size=16, seq_len=32, freq=1.0, decay=0.3, seed=seed)
        return obs

    def loss_fn(model, obs):
        z, pred_z, recon_x, pred_x = model.forward_no_action(obs)
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

    @nnx.jit
    def train_step(model, optimizer, obs):
        loss, grads = nnx.value_and_grad(loss_fn)(model, obs)
        optimizer.update(model, grads)
        return loss

    section("training")
    obs = make_batch(0)
    for step in range(1, 201):
        loss = float(train_step(model, optimizer, obs))
        if step % 40 == 0:
            print(f"  step {step:4d}  loss = {loss:.5f}")

    section("one-step-ahead predictions (latent space)")
    obs = make_batch(1)
    z, pred_z, recon_x, pred_x = model.forward_no_action(obs)
    print(f"  latent pred MSE    : {float(jnp.mean((z[:, 1:] - pred_z[:, :-1]) ** 2)):.5f}")
    print(f"  latent pred R^2    : {_r2(pred_z[:, :-1], z[:, 1:]):.4f}")
    print(
        f"  reconstruction RMSE: {rmse(recon_x, obs):.4f}  (signal RMS {float(jnp.sqrt(jnp.mean(obs**2))):.4f})"
    )
    print(f"  one-step obs RMSE  : {rmse(pred_x[:, :-1], obs[:, 1:]):.4f}")

    section("LeJEPA health check on the learned latents")
    from ssm_latent.monitor import check_stationarity

    print(f"  {compute_exploration_quality(z)}")
    print(f"  {check_stationarity(z, pred_z, n_layers=cfg.n_layers)}")
    print(
        "  note: a *deterministic* damped oscillator is by construction a narrow,\n"
        "        1-D trajectory, so Theorem 3 ('isotropic exploration') is violated:\n"
        "        this is the monitor doing its job, not a training bug."
    )

    section("SIGReg (LeJEPA) objective value")
    print(
        f"  lejepa_loss = {float(lejepa_loss(z, pred_z, model.stability_projections[...], 0.1)):.5f}"
    )

    section("streaming inference (O(1) per step)")
    state = model.zero_state(batch=1)
    z0 = model.encode(obs[:1, :1])
    preds = []
    for _ in range(8):
        y, state = model.step(z0[:, 0], None, state)
        preds.append(y[:, 0])
    print(f"  rolled out {len(preds)} steps, output {tuple(jnp.stack(preds).shape)}")


def _r2(pred, target) -> float:
    pred = jnp.asarray(pred)
    target = jnp.asarray(target)
    var = float(jnp.mean((target - target.mean()) ** 2)) + 1e-12
    return 1.0 - float(jnp.mean((pred - target) ** 2)) / var


if __name__ == "__main__":
    main()
