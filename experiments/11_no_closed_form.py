"""Experiment 11 - on a task with *no closed form*, does a learned world model earn its keep?

The whole line so far used closed-form generators, where an *exact* Python tool already
predicts to ~1e-6 (see FINDINGS.md) - so a learned model could never "beat code" there.
This experiment moves to systems that are only numerically integrable:

  * ``driven_pendulum`` - damped, sinusoidally forced; aperiodic.
  * ``double_pendulum`` - chaotic; nearby trajectories diverge exponentially.

Three forecasters, scored identically on closed-loop roll-out error (lower is better):

  * ``persistence`` - repeat the last frame (a null model that needs no knowledge),
  * ``learned``     - a small JEPA/SSM world model trained on observations only,
  * ``code``        - the *known* equations integrated forward, starting from a state
                      estimated from the observations alone (the honest "I wrote the
                      simulator" baseline: knowing the physics still leaves you
                      estimating the initial state, and chaotic systems punish that).

The interesting quantity is where ``learned`` sits between the trivial baseline and the
code baseline, and how the gaps grow with the forecast horizon.

Run:  python experiments/11_no_closed_form.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import dynamics as D
import jax.numpy as jnp

RESULTS = Path(__file__).resolve().parent / "results" / "11_no_closed_form.json"
SYSTEMS = {"driven_pendulum": 2, "double_pendulum": 4}
TRAIN_SEQ, BATCH = 32, 32
HORIZONS = (5, 10, 20)
TRAIN_STEPS = 400


def rmse(a, b) -> float:
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# learned world model (observations only, no actions: the dynamics are autonomous)
# --------------------------------------------------------------------------- #
def train(system: str, obs_dim: int, *, seed: int = 0):
    import optax
    from flax import nnx

    from ssm_latent.config import MultiScaleSsmConfig
    from ssm_latent.latent import LatentLossArgs, MultiScaleLatentPredictor

    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, input_dim=obs_dim, action_dim=0, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    obs = D.sample(system, BATCH, TRAIN_SEQ)

    @nnx.jit
    def step(model, opt, obs):
        def loss_fn(model):
            z, pred_z, recon_x, pred_x = model.forward_no_action(obs)
            return model.loss(LatentLossArgs(z, pred_z, recon_x, pred_x, obs))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for _ in range(TRAIN_STEPS):
        step(model, opt, obs)
    return model


def learned_rollout(model, context, horizon: int):
    z = model.encode(context)
    state = model.zero_state(context.shape[0])
    z_run = z[:, -1]
    decoded = []
    for _ in range(horizon):
        z_run, state = model.step(z_run, None, state)
        decoded.append(model.decode(z_run))
    return jnp.stack(decoded, axis=1)


# --------------------------------------------------------------------------- #
# the "I wrote the simulator" baseline: true equations + estimated initial state
# --------------------------------------------------------------------------- #
def _estimate_state(system: str, context):
    """Recover the integrator state from observations alone (finite differences)."""
    last, prev = context[:, -1], context[:, -2]
    if system == "driven_pendulum":  # obs = (sin th, cos th)
        th = jnp.arctan2(last[:, 0], last[:, 1])
        th_p = jnp.arctan2(prev[:, 0], prev[:, 1])
        w = _unwrap_diff(th, th_p) / 0.05
        return jnp.stack([th, w])
    # double pendulum: obs = (x1, y1, x2, y2)
    th1 = jnp.arctan2(last[:, 0], -last[:, 1])
    th2 = jnp.arctan2(last[:, 2] - last[:, 0], -(last[:, 3] - last[:, 1]))
    th1p = jnp.arctan2(prev[:, 0], -prev[:, 1])
    th2p = jnp.arctan2(prev[:, 2] - prev[:, 0], -(prev[:, 3] - prev[:, 1]))
    w1 = _unwrap_diff(th1, th1p) / 0.02
    w2 = _unwrap_diff(th2, th2p) / 0.02
    return jnp.stack([th1, w1, th2, w2])


def _unwrap_diff(a, b):
    d = a - b
    return (d + jnp.pi) % (2 * jnp.pi) - jnp.pi


def code_rollout(system: str, context, horizon: int):
    """Integrate the *true* equations forward from the estimated state."""
    state = _estimate_state(system, context)

    if system == "driven_pendulum":
        g_par = {"b": 0.2, "amp": 1.2, "drive": 2.0, "L": 1.0, "dt": 0.05}

        def deriv(s, t):
            th, w = s[0], s[1]
            return jnp.stack(
                [
                    w,
                    -g_par["b"] * w
                    - (D.G / g_par["L"]) * jnp.sin(th)
                    + g_par["amp"] * jnp.cos(g_par["drive"] * t),
                ]
            )

        def obs_of(s):
            return jnp.stack([jnp.sin(s[0]), jnp.cos(s[0])], axis=-1)

        t0 = 0.05 * 32
        dt = g_par["dt"]
    else:

        def deriv(s):
            t1, w1, t2, w2 = s[0], s[1], s[2], s[3]
            d = t2 - t1
            den = 2 * 1 + 1 - 1 * jnp.cos(2 * d)
            a1 = (
                -D.G * 3 * jnp.sin(t1)
                - D.G * jnp.sin(t1 - 2 * t2)
                - 2 * jnp.sin(d) * (w2**2 + w1**2 * jnp.cos(d))
            ) / den
            a2 = (2 * jnp.sin(d) * (w1**2 * 2 + D.G * 2 * jnp.cos(t1) + w2**2 * jnp.cos(d))) / den
            return jnp.stack([w1, a1, w2, a2])

        def obs_of(s):
            x1, y1 = jnp.sin(s[0]), -jnp.cos(s[0])
            x2, y2 = x1 + jnp.sin(s[2]), y1 - jnp.cos(s[2])
            return jnp.stack([x1, y1, x2, y2], axis=-1)

        t0 = 0.02 * 32
        dt = 0.02

    out = []
    s = state
    for i in range(horizon):
        k1 = deriv(s) if system != "driven_pendulum" else deriv(s, t0 + i * dt)
        k2 = (
            deriv(s + 0.5 * dt * k1)
            if system != "driven_pendulum"
            else deriv(s + 0.5 * dt * k1, t0 + (i + 0.5) * dt)
        )
        k3 = (
            deriv(s + 0.5 * dt * k2)
            if system != "driven_pendulum"
            else deriv(s + 0.5 * dt * k2, t0 + (i + 0.5) * dt)
        )
        k4 = (
            deriv(s + dt * k3)
            if system != "driven_pendulum"
            else deriv(s + dt * k3, t0 + (i + 1) * dt)
        )
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(obs_of(s))
    return jnp.stack(out, axis=1)


# --------------------------------------------------------------------------- #
def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        rows = []
        for system, obs_dim in SYSTEMS.items():
            model = train(system, obs_dim)
            for horizon in HORIZONS:
                context = D.sample(system, 32, 32)
                future = D.sample(system, 32, 32 + horizon)[:, 32:]
                learned = learned_rollout(model, context, horizon)
                code = code_rollout(system, context, horizon)
                persistence = jnp.repeat(context[:, -1:], horizon, axis=1)
                rows.append(
                    {
                        "system": system,
                        "horizon": horizon,
                        "persistence_rmse": rmse(persistence, future),
                        "learned_rmse": rmse(learned, future),
                        "code_rmse": rmse(code, future),
                    }
                )
                r = rows[-1]
                print(
                    f"  {system:<16} h={horizon:2d}: persistence={r['persistence_rmse']:.4f} "
                    f"learned={r['learned_rmse']:.4f} code={r['code_rmse']:.4f}",
                    flush=True,
                )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    print("\nClosed-loop RMSE by horizon (lower is better):")
    for system in SYSTEMS:
        print(f"  {system}")
        print(
            f"    {'horizon':>8}{'persistence':>13}{'learned':>10}{'code':>9}{'learned gain':>14}"
        )
        for r in rows:
            if r["system"] != system:
                continue
            gain = r["persistence_rmse"] - r["learned_rmse"]
            print(
                f"    {r['horizon']:>8}{r['persistence_rmse']:>13.4f}"
                f"{r['learned_rmse']:>10.4f}{r['code_rmse']:>9.4f}{gain:>+14.4f}"
            )
    print("\n  'code' = the true equations integrated from an estimated state.  Code only")
    print("  wins if the learned model cannot beat persistence, or if the gap to code is")
    print("  large and grows; on a chaotic system even code loses accuracy past the Lyapunov")
    print("  time, which sets the horizon over which *any* forecaster is useful.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
