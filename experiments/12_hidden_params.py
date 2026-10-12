"""Experiment 12 - where code stops winning: hidden parameters + partial observability.

exp11 showed code beats a learned world model whenever the dynamics are writable as
equations.  This experiment constructs the case exp11 predicted would be different:
a damped driven pendulum whose *(b, amp, drive)* vary per episode and are **never
observed** - only ``(sin theta, cos theta)`` is - so the equations are not writable
from what the agent sees.

Three forecasters, on episodes whose hidden parameters are unknown:

  * ``persistence``     - repeat the last frame,
  * ``learned``         - a JEPA/SSM world model trained across the hidden-parameter
                          band (it can, in principle, infer the dynamics from context),
  * ``code+sysid``      - the *strongest* code baseline: fit the three hidden parameters
                          to the context by gradient descent, then integrate the true
                          equations.  This is what a competent engineer would write.

We sweep the band width ``spread``: at spread=0 there is one fixed dynamics (code has
nothing to identify, exp11 applies); as spread grows, identification gets harder for
both sides.  The question: does the learned model, which sees the same context, hold up
where parameter fitting degrades?

Run:  python experiments/12_hidden_params.py [--force]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import dynamics as D
import jax
import jax.numpy as jnp
import optax
from flax import nnx

RESULTS = Path(__file__).resolve().parent / "results" / "12_hidden_params.json"
SPREADS = (0.0, 0.5, 1.0)
HORIZONS = (5, 10, 20)
CONTEXT_SEQ, BATCH, DT = 32, 32, 0.05
TRAIN_STEPS = 500


def rmse(a, b) -> float:
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


# --------------------------------------------------------------------------- #
# learned world model: trained on a *mixture* of hidden-parameter episodes
# --------------------------------------------------------------------------- #
def train_learned(spread: float, *, seed: int = 0):
    from ssm_latent.config import MultiScaleSsmConfig
    from ssm_latent.latent import LatentLossArgs, MultiScaleLatentPredictor

    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    model = MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=0, rngs=nnx.Rngs(seed))
    opt = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

    # sample a fresh *mixture* of episodes each step (different hidden params)
    def batch(step_key):
        return D.hidden_pendulum(BATCH, CONTEXT_SEQ, spread=spread, seed=int(step_key))

    @nnx.jit
    def step(model, opt, obs):
        def loss_fn(model):
            z, pred_z, recon_x, pred_x = model.forward_no_action(obs)
            return model.loss(LatentLossArgs(z, pred_z, recon_x, pred_x, obs))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    for i in range(TRAIN_STEPS):
        step(model, opt, batch(i))
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
# code + system identification: fit (b, amp, drive) to the context, then integrate
# --------------------------------------------------------------------------- #
def code_sysid(context, horizon: int):
    """Gradient-fit the three hidden parameters per episode, then integrate RK4."""
    # initial guess: middle of the band; fit against the context via finite differences
    theta = jnp.arctan2(context[:, :, 0], context[:, :, 1])
    dtheta = jnp.diff(theta, axis=1)
    dtheta = (dtheta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    w = dtheta / DT  # observed angular velocity [B, T-1]

    # fit (b, amp, drive) by matching the observed angular acceleration
    w_dot_obs = (w[:, 1:] - w[:, :-1]) / DT  # [B, T-2]
    th_mid = theta[:, 1:-1]  # [B, T-2]
    w_mid = w[:, :-1]  # [B, T-2]
    t_mid = jnp.arange(1, theta.shape[1] - 1, dtype=theta.dtype) * DT  # [T-2]

    def loss(params):
        b, amp, drive = params
        w_dot_model = -b * w_mid - D.G * jnp.sin(th_mid) + amp * jnp.cos(drive * t_mid)[None, :]
        return jnp.mean((w_dot_model - w_dot_obs) ** 2)

    params = (jnp.array(0.2), jnp.array(1.0), jnp.array(2.0))  # (b, amp, drive)
    opt = optax.adam(5e-2)
    state = opt.init(params)
    grad_fn = jax.value_and_grad(loss)
    for _ in range(200):
        _, grads = grad_fn(params)
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)

    th_last = jnp.arctan2(context[:, -1, 0], context[:, -1, 1])
    s = jnp.stack([th_last, w[:, -1]])
    b, amp, drive = params
    t0 = DT * CONTEXT_SEQ
    out = []
    for i in range(horizon):
        t = t0 + i * DT
        k1 = _deriv(s, t, b, amp, drive)
        k2 = _deriv(s + 0.5 * DT * k1, t + 0.5 * DT, b, amp, drive)
        k3 = _deriv(s + 0.5 * DT * k2, t + 0.5 * DT, b, amp, drive)
        k4 = _deriv(s + DT * k3, t + DT, b, amp, drive)
        s = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(jnp.stack([jnp.sin(s[0]), jnp.cos(s[0])], axis=-1))
    return jnp.stack(out, axis=1)


def _deriv(s, t, b, amp, drive):
    th, w = s[0], s[1]
    return jnp.stack([w, -b * w - D.G * jnp.sin(th) + amp * jnp.cos(drive * t)])


# --------------------------------------------------------------------------- #
def main() -> None:
    force = "--force" in sys.argv
    t0 = time.time()
    if RESULTS.exists() and not force:
        rows = json.loads(RESULTS.read_text())
        print("loaded cached results (pass --force to recompute)")
    else:
        rows = []
        for spread in SPREADS:
            model = train_learned(spread)
            context = D.hidden_pendulum(BATCH, CONTEXT_SEQ, spread=spread, seed=999)
            full = D.hidden_pendulum(BATCH, CONTEXT_SEQ + max(HORIZONS), spread=spread, seed=999)
            for horizon in HORIZONS:
                future = full[:, CONTEXT_SEQ : CONTEXT_SEQ + horizon]
                learned = learned_rollout(model, context, horizon)
                code = code_sysid(context, horizon)
                persistence = jnp.repeat(context[:, -1:], horizon, axis=1)
                rows.append(
                    {
                        "spread": spread,
                        "horizon": horizon,
                        "persistence_rmse": rmse(persistence, future),
                        "learned_rmse": rmse(learned, future),
                        "code_rmse": rmse(code, future),
                    }
                )
                r = rows[-1]
                print(
                    f"  spread={spread:.1f} h={horizon:2d}: persistence={r['persistence_rmse']:.4f} "
                    f"learned={r['learned_rmse']:.4f} code_sysid={r['code_rmse']:.4f}",
                    flush=True,
                )
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(rows, indent=1))
        print(f"saved raw results to {RESULTS}")

    print("\nClosed-loop RMSE (lower is better).  'code+sysid' fits the hidden params")
    print("to the context by gradient descent, then integrates the true equations.")
    for spread in SPREADS:
        print(f"  spread={spread:.1f}  (0 = one fixed dynamics, 1 = wide hidden band)")
        print(f"    {'horizon':>8}{'persistence':>13}{'learned':>10}{'code+sysid':>12}")
        for r in rows:
            if r["spread"] != spread:
                continue
            print(
                f"    {r['horizon']:>8}{r['persistence_rmse']:>13.4f}"
                f"{r['learned_rmse']:>10.4f}{r['code_rmse']:>12.4f}"
            )
    print("\n  Watch the gap between 'learned' and 'code+sysid' as spread grows: if it")
    print("  narrows, the learned model competes exactly where system identification")
    print("  gets hard - the niche exp11 predicted.")
    print(f"\n(wall clock {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
