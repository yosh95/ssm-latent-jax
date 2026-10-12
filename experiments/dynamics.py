"""Dynamical systems with **no closed-form solution**, for the difficulty sweep.

`ssm_latent.data` ships closed-form generators (circle, sine, damped oscillator), which
turned out to be the wrong testbed for "does a learned world model earn its keep?" - an
*exact* Python tool already predicts them to ~1e-6 (see FINDINGS.md).  These are the
opposite: only numerically integrable.

  * ``driven_pendulum`` - damped, sinusoidally forced; deterministic, aperiodic.
  * ``double_pendulum`` - chaotic; tiny parameter changes diverge exponentially.

Ground truth is a fixed-step RK4 integrator - "the physics", not a closed form.  The
same integrator is what an oracle "code" baseline gets, so the comparison in exp11 is
fair: code can only win here by *knowing the dynamics*, and a learned model has to
recover them from observations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

G = 9.81


# --------------------------------------------------------------------------- #
# damped driven pendulum:  theta'' = -b*theta' - (g/L) sin(theta) + A cos(w t)
# --------------------------------------------------------------------------- #
def driven_pendulum(
    batch=4, seq=64, *, b=0.2, amp=1.2, drive=2.0, L=1.0, dt=0.05, theta0=0.6, seed=0
):
    """Observations ``(sin theta, cos theta)``; state ``(theta, omega)``."""
    phases = theta0 + jnp.arange(batch) * 0.5

    def deriv(s, t):
        th, w = s[0], s[1]
        return jnp.stack([w, -b * w - (G / L) * jnp.sin(th) + amp * jnp.cos(drive * t)])

    def step(s, t):
        k1 = deriv(s, t)
        k2 = deriv(s + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = deriv(s + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = deriv(s + dt * k3, t + dt)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    s = jnp.stack([phases, jnp.zeros_like(phases)])  # [2, B]
    obs = []
    for i in range(seq):
        obs.append(jnp.stack([jnp.sin(s[0]), jnp.cos(s[0])], axis=-1))  # [B, 2]
        s = step(s, i * dt)
    return jnp.stack(obs, axis=1)  # [B, T, 2]


# --------------------------------------------------------------------------- #
# double pendulum: chaotic.  state (th1, w1, th2, w2)
# --------------------------------------------------------------------------- #
def double_pendulum(
    batch=4, seq=64, *, L1=1.0, L2=1.0, m1=1.0, m2=1.0, dt=0.02, theta0=2.0, seed=0
):
    """Observations ``(x1, y1, x2, y2)`` of the two bobs."""
    th1 = theta0 + jnp.arange(batch) * 0.01  # near-identical starts -> chaos
    s = jnp.stack([th1, jnp.zeros_like(th1), th1 * 0.5, jnp.zeros_like(th1)])

    def deriv(s):
        t1, w1, t2, w2 = s[0], s[1], s[2], s[3]
        d = t2 - t1
        den = 2 * m1 + m2 - m2 * jnp.cos(2 * d)
        a1 = (
            -G * (2 * m1 + m2) * jnp.sin(t1)
            - m2 * G * jnp.sin(t1 - 2 * t2)
            - 2 * jnp.sin(d) * m2 * (w2**2 * L2 + w1**2 * L1 * jnp.cos(d))
        ) / (L1 * den)
        a2 = (
            2
            * jnp.sin(d)
            * (w1**2 * L1 * (m1 + m2) + G * (m1 + m2) * jnp.cos(t1) + w2**2 * L2 * m2 * jnp.cos(d))
        ) / (L2 * den)
        return jnp.stack([w1, a1, w2, a2])

    def step(s):
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    obs = []
    for _ in range(seq):
        t1, t2 = s[0], s[2]
        x1, y1 = L1 * jnp.sin(t1), -L1 * jnp.cos(t1)
        x2 = x1 + L2 * jnp.sin(t2)
        y2 = y1 - L2 * jnp.cos(t2)
        obs.append(jnp.stack([x1, y1, x2, y2], axis=-1))
        s = step(s)
    return jnp.stack(obs, axis=1)


# --------------------------------------------------------------------------- #
# a system where the equations are NOT writable from the observation:
#   damped driven pendulum with per-episode *hidden* parameters, and only the
#   angle observed (theta, not the angular velocity).  Code cannot write this down
#   without first identifying the parameters AND the hidden velocity from data.
# --------------------------------------------------------------------------- #
def hidden_pendulum(batch=8, seq=64, *, dt=0.05, seed=0, spread=1.0):
    """Observations ``(sin theta, cos theta)``; hidden params ``(b, amp, drive)``.

    ``spread`` scales how far the hidden parameters vary across the batch:
    ``spread=0`` gives a single fixed dynamics (code has nothing to identify),
    ``spread=1`` gives a wide band that must be inferred per episode.
    """
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(seed), 5)
    b = jax.random.uniform(k1, (batch,), minval=0.05, maxval=0.05 + 0.45 * spread)
    amp = jax.random.uniform(k2, (batch,), minval=0.5, maxval=0.5 + 1.5 * spread)
    drive = jax.random.uniform(k3, (batch,), minval=1.5, maxval=1.5 + 1.5 * spread)
    th0 = jax.random.uniform(k4, (batch,), minval=-jnp.pi, maxval=jnp.pi)
    w0 = jax.random.uniform(k5, (batch,), minval=-1.0, maxval=1.0)

    def deriv(s, t, b, amp, drive):
        th, w = s[0], s[1]
        return jnp.stack([w, -b * w - G * jnp.sin(th) + amp * jnp.cos(drive * t)])

    def step(s, t):
        kk1 = deriv(s, t, b, amp, drive)
        kk2 = deriv(s + 0.5 * dt * kk1, t + 0.5 * dt, b, amp, drive)
        kk3 = deriv(s + 0.5 * dt * kk2, t + 0.5 * dt, b, amp, drive)
        kk4 = deriv(s + dt * kk3, t + dt, b, amp, drive)
        return s + (dt / 6.0) * (kk1 + 2 * kk2 + 2 * kk3 + kk4)

    s = jnp.stack([th0, w0])
    obs = []
    for i in range(seq):
        obs.append(jnp.stack([jnp.sin(s[0]), jnp.cos(s[0])], axis=-1))
        s = step(s, i * dt)
    return jnp.stack(obs, axis=1)


SYSTEMS = {
    "driven_pendulum": driven_pendulum,
    "double_pendulum": double_pendulum,
    "hidden_pendulum": hidden_pendulum,
}


def sample(system: str, batch: int, seq: int, *, dt: float | None = None, **kw):
    if dt is not None:
        return SYSTEMS[system](batch, seq, dt=dt, **kw)
    return SYSTEMS[system](batch, seq, **kw)
