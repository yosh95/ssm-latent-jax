"""Shared metrics for the research experiments.

Not part of the published package: this directory is a scratchpad for
*not-yet-written-up* results, kept next to the library so every number in the
write-ups can be regenerated exactly.

Importing this module inserts ``src/`` on ``sys.path`` (like ``examples/common.py``)
so a fresh checkout works without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if jax.devices()[0].platform != "cpu":  # keep runs comparable / reproducible
    jax.config.update("jax_platform_name", "cpu")


# --------------------------------------------------------------------------- #
# error metrics
# --------------------------------------------------------------------------- #
def rmse(pred, target) -> float:
    pred, target = jnp.asarray(pred), jnp.asarray(target)
    return float(jnp.sqrt(jnp.mean((pred - target) ** 2)))


def r2(pred, target) -> float:
    pred, target = jnp.asarray(pred), jnp.asarray(target)
    var = float(jnp.mean((target - target.mean()) ** 2)) + 1e-12
    return 1.0 - float(jnp.mean((pred - target) ** 2)) / var


# --------------------------------------------------------------------------- #
# world-model evaluation (mirrors examples/03_circle_world_planning.py)
# --------------------------------------------------------------------------- #
def context_and_truth(
    batch: int,
    seq: int,
    horizon: int,
    *,
    omega: float,
    phase_shift: float = 0.0,
    dt: float = 0.3,
    noise: float = 0.0,
):
    """A context window and its *true* continuation (the generators are deterministic).

    ``obs`` is ``[B, seq, 2]``; ``future`` is ``[B, horizon, 2]`` starting exactly
    where the context ends (``phase_shift`` is advanced by ``omega * dt * seq``).
    """
    from ssm_latent.data import circle_world

    obs = circle_world(batch, seq, omega=omega, phase_shift=phase_shift, dt=dt, noise=noise)
    future = circle_world(
        batch, horizon, omega=omega, phase_shift=phase_shift + omega * dt * seq, dt=dt
    )
    return obs, future


def persistence_baseline(obs, horizon: int):
    """The "nothing changes" baseline: repeat the last observed frame."""
    return jnp.repeat(obs[:, -1:], horizon, axis=1)


def latent_rollout(model, obs, act, horizon: int):
    """Autoregressive latent imagination, decoded back to observation space.

    Identical in spirit to ``examples/03``: the model consumes its *own* predicted
    latent at every step (no teacher forcing), with the context's last action
    applied throughout.
    """
    z = model.encode(obs)
    state = model.zero_state(obs.shape[0])
    z_run = z[:, -1]
    decoded = []
    for _ in range(horizon):
        z_run, state = model.step(z_run, act[:, -1], state)
        decoded.append(model.decode(z_run))
    return jnp.stack(decoded, axis=1)


def evaluate_latent_world_model(
    model, *, omega: float, horizon: int, seq: int = 32, batch: int = 32, phase_shift: float = 1.1
):
    """One-step accuracy + closed-loop imagination, on a phase the model never saw.

    Returns ``{"one_step_rmse", "closed_loop_rmse", "persistence_rmse",
    "imagination_gain"}`` where ``imagination_gain = persistence_rmse -
    closed_loop_rmse`` (>0 means the world model beats "nothing changes").
    """
    from ssm_latent.data import circle_world_actions

    obs, future = context_and_truth(batch, seq, horizon, omega=omega, phase_shift=phase_shift)
    act = circle_world_actions(batch, seq, omega=omega, phase_shift=phase_shift)

    one_step = rmse(model(obs, act)[3][:, :-1], obs[:, 1:])
    rollout = latent_rollout(model, obs, act, horizon)
    cl = rmse(rollout, future)
    pers = rmse(persistence_baseline(obs, horizon), future)
    return {
        "one_step_rmse": one_step,
        "closed_loop_rmse": cl,
        "persistence_rmse": pers,
        "imagination_gain": pers - cl,
    }


def predictor_rollout(model, obs, horizon: int):
    """Closed-loop rollout of an observation-space ``MambaPredictor``.

    The predictor maps obs -> next obs, so rolling it means feeding its own
    decoded output back in (no action input).
    """
    x = obs[:, -1]
    state = model.zero_state(obs.shape[0])
    preds = []
    for _ in range(horizon):
        x, state = model.step(x, state)
        preds.append(x)
    return jnp.stack(preds, axis=1)


def evaluate_mamba_predictor(
    model, *, omega: float, horizon: int, seq: int = 32, batch: int = 32, phase_shift: float = 1.1
):
    """One-step accuracy + closed-loop rollout for the observation-space predictor."""
    obs, future = context_and_truth(batch, seq, horizon, omega=omega, phase_shift=phase_shift)
    one_step = rmse(model(obs)[:, :-1], obs[:, 1:])
    rollout = predictor_rollout(model, obs, horizon)
    cl = rmse(rollout, future)
    pers = rmse(persistence_baseline(obs, horizon), future)
    return {
        "one_step_rmse": one_step,
        "closed_loop_rmse": cl,
        "persistence_rmse": pers,
        "imagination_gain": pers - cl,
    }
