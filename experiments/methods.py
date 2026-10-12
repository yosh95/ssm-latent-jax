"""Two levers that act on the *cause* of the out-of-range breakdown.

Experiments 05-08 showed that searching the *settings* (curriculum, budget) hits a
ceiling: no setting made the closed-loop imagination usable out of range.  So (B)
hands the driver two variables that act on the mechanism instead:

  objective = "oneshot"  - the library's one-step JEPA objective (train to predict
                           the *next* latent; imagine later, and hope).
  objective = "rollout"  - additionally minimise the error of a *closed-loop* rollout
                           (the model consumes its own predictions during training),
                           i.e. train for the thing we actually measure.

  adapt     = "weight"   - a few gradient steps on the demonstration (as in exp05).
  adapt     = "warmup"   - **no weight change**: run the demonstration through the
                           recurrent state and start the query rollout from there
                           (in-context conditioning).  Cheaper and, if it works,
                           a very different story from "the LLM fine-tunes the model".

Everything is a pure function of a spec, so the same code backs the CLI tool the LLM
calls and the scripted baselines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.data import circle_world, circle_world_actions
from ssm_latent.latent import (
    LatentLossArgs,
    MultiScaleLatentPredictor,
)

D_MODEL, D_STATE, N_HEADS, N_LAYERS = 32, 8, 4, 3
TRAIN_SEQ, TRAIN_BATCH = 24, 24


def build_model(seed: int) -> MultiScaleLatentPredictor:
    cfg = MultiScaleSsmConfig(
        d_model=D_MODEL,
        d_state=D_STATE,
        expand=2,
        n_heads=N_HEADS,
        mimo_rank=1,
        n_layers=N_LAYERS,
        use_conv=True,
    )
    return MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))


# --------------------------------------------------------------------------- #
# objectives
# --------------------------------------------------------------------------- #
def oneshot_loss(model, obs, act) -> jax.Array:
    """One-step JEPA objective: predict the next latent (the library's own loss)."""
    z, pred_z, recon_x, pred_x = model(obs, act)
    return model.loss(LatentLossArgs(z, pred_z, recon_x, pred_x, obs))


def rollout_loss(model, obs, act) -> jax.Array:
    """Closed-loop objective: roll the model out and match the *whole* trajectory.

    The model is fed its own predicted latent at every step (no teacher forcing), and
    decoded predictions are compared to the ground-truth observations, so the loss
    sees exactly the quantity we score (imagination) rather than a one-step proxy.
    """
    batch, seq_len = obs.shape[:2]
    z = model.encode(obs)
    state = model.zero_state(batch)
    z_run = z[:, 0]
    preds = []
    for t in range(seq_len - 1):
        z_run, state = model.step(z_run, act[:, t], state)
        preds.append(model.decode(z_run))
    preds = jnp.stack(preds, axis=1)  # [B, T-1, 2]
    target = jax.lax.stop_gradient(obs[:, 1:])
    mse = ((preds - target) ** 2).mean()
    # keep the representation healthy with the library's own one-step regulariser
    z2, pred_z, recon_x, pred_x = model(obs, act)
    reg = model.loss(LatentLossArgs(z2, pred_z, recon_x, pred_x, obs, recon_weight=0.0))
    return mse + 0.1 * reg


LOSSES = {"oneshot": oneshot_loss, "rollout": rollout_loss}


# --------------------------------------------------------------------------- #
# Reptile meta-training (generic over the objective)
# --------------------------------------------------------------------------- #
def _make_inner_step(objective: str):
    """Build the jitted inner update *once* (re-creating it per iteration recompiles)."""
    loss_fn = LOSSES[objective]

    @nnx.jit
    def inner(clone, opt, obs, act):
        loss, grads = nnx.value_and_grad(lambda m: loss_fn(m, obs, act))(clone)
        opt.update(clone, grads)
        return loss

    return inner


def reptile_train(
    model,
    omegas,
    *,
    objective="oneshot",
    iters=40,
    inner_steps=5,
    inner_lr=3e-3,
    outer_lr=0.3,
    seed=0,
) -> MultiScaleLatentPredictor:
    """Minimal Reptile loop over a fixed list of omegas (deterministic per seed).

    One clone + optimizer is reused across iterations (resetting the clone to the
    current base each time) - creating a fresh ``nnx.Module`` per iteration forces the
    JIT trace to recompile every step, which is ~100x slower.
    """
    inner = _make_inner_step(objective)
    omegas = [float(o) for o in omegas]
    clone = nnx.clone(model)
    opt = nnx.Optimizer(clone, optax.adam(inner_lr), wrt=nnx.Param)

    for it in range(iters):
        nnx.update(clone, nnx.state(model))  # start the inner loop from the base weights
        omega = omegas[it % len(omegas)]
        obs = circle_world(TRAIN_BATCH, TRAIN_SEQ, omega=omega)
        act = circle_world_actions(TRAIN_BATCH, TRAIN_SEQ, omega=omega)

        for _ in range(inner_steps):
            inner(clone, opt, obs, act)

        base = nnx.state(model).to_pure_dict()
        target = nnx.state(clone).to_pure_dict()
        new = jax.tree.map(lambda b, t: b + outer_lr * (t - b), base, target)
        st = nnx.state(model)
        st.replace_by_pure_dict(new)
        nnx.update(model, st)
    return model


# --------------------------------------------------------------------------- #
# adaptation mechanisms
# --------------------------------------------------------------------------- #
def adapt_weight(model, demo_obs, demo_act, *, k, lr=3e-3, objective="oneshot"):
    """A few gradient steps on the demonstration (the exp05 mechanism)."""
    if k <= 0:
        return model
    loss_fn = LOSSES[objective]
    clone = nnx.clone(model)
    opt = nnx.Optimizer(clone, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def step(clone, opt, obs, act):
        loss, grads = nnx.value_and_grad(lambda m: loss_fn(m, obs, act))(clone)
        opt.update(clone, grads)
        return loss

    for _ in range(k):
        step(clone, opt, demo_obs, demo_act)
    return clone


def warm_state(model, demo_obs, demo_act):
    """Run the demonstration through the recurrence - no gradients, no weight change."""
    batch = demo_obs.shape[0]
    z = model.encode(demo_obs)
    state = model.zero_state(batch)
    z_run = z[:, 0]
    for t in range(demo_obs.shape[1] - 1):
        z_run, state = model.step(z_run, demo_act[:, t], state)
    return z_run, state


# --------------------------------------------------------------------------- #
# evaluation (closed-loop imagination, optionally from a warmed state)
# --------------------------------------------------------------------------- #
def evaluate(model, omega, *, horizon=10, seq=32, batch=32, warm_from=None, demo_seq=16):
    """Closed-loop imagination on ``omega``, scored against a persistence baseline.

    ``warm_from`` is an optional ``(demo_omega, demo_seq)`` pair: when given, the
    model is first run (forward only, no gradients) over a demonstration of that
    task at the *same batch*, and the query rollout starts from the resulting state.
    This is the weight-free "warmup" mechanism.
    """
    obs = circle_world(batch, seq, omega=omega)
    act = circle_world_actions(batch, seq, omega=omega)
    future = circle_world(batch, horizon, omega=omega, phase_shift=omega * 0.3 * seq)

    if warm_from is None:
        z = model.encode(obs)
        state = model.zero_state(batch)
        z_run = z[:, -1]
    else:  # start from the state the demonstration left behind (same batch!)
        demo_omega, demo_len = warm_from
        demo_obs = circle_world(batch, demo_len, omega=demo_omega)
        demo_act = circle_world_actions(batch, demo_len, omega=demo_omega)
        z_run, state = warm_state(model, demo_obs, demo_act)

    decoded = []
    for _ in range(horizon):
        z_run, state = model.step(z_run, act[:, -1], state)
        decoded.append(model.decode(z_run))
    rollout = jnp.stack(decoded, axis=1)

    persistence = jnp.repeat(obs[:, -1:], horizon, axis=1)
    cl = float(jnp.sqrt(jnp.mean((rollout - future) ** 2)))
    pers = float(jnp.sqrt(jnp.mean((persistence - future) ** 2)))
    return {"closed_loop_rmse": cl, "persistence_rmse": pers, "imagination_gain": pers - cl}
