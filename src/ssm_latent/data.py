"""Synthetic physical-dynamics datasets used for training and testing.

Ports `ssm_latent_model::meta::tasks` (CircleWorld, SineWave) and the data
generators used by the Rust demos (damped oscillator, Lissajous, chaotic
variants, tokenised "tiny stories").
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = [
    "circle_world",
    "circle_world_actions",
    "sine_wave",
    "damped_oscillator",
    "lissajous",
    "chaotic_circle",
    "make_demo_data",
    "TinyStories",
    "TINY_STORIES",
]


# --------------------------------------------------------------------------- #
# circle world (the canonical demo task of the Rust crate)
# --------------------------------------------------------------------------- #
def circle_world(
    batch_size: int = 4,
    seq_len: int = 32,
    omega: float = 1.0,
    phase_shift: float = 0.0,
    dt: float = 0.3,
    noise: float = 0.0,
    *,
    seed: int = 0,
) -> jax.Array:
    """Observations ``(cos, sin)`` of a point moving on a unit circle.

    Mirrors `CircleWorldTask::generate_data`: the per-batch phase offset is
    ``phase_shift + b * 1.25`` and the sample times are ``t * dt``.
    """
    b = jnp.arange(batch_size)[:, None]
    t = jnp.arange(seq_len)[None, :]
    angle = omega * (t.astype(jnp.float32) * dt) + phase_shift + b.astype(jnp.float32) * 1.25
    obs = jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)
    if noise > 0.0:
        obs = obs + noise * jax.random.normal(jax.random.PRNGKey(seed), obs.shape)
    return obs


def circle_world_actions(
    batch_size: int = 4,
    seq_len: int = 32,
    omega: float = 1.0,
    phase_shift: float = 0.0,
    dt: float = 0.3,
) -> jax.Array:
    """Analytic velocity (tangent vector) for :func:`circle_world`."""
    b = jnp.arange(batch_size)[:, None]
    t = jnp.arange(seq_len)[None, :]
    angle = omega * (t.astype(jnp.float32) * dt) + phase_shift + b.astype(jnp.float32) * 1.25
    return jnp.stack([-omega * dt * jnp.sin(angle), omega * dt * jnp.cos(angle)], axis=-1)


def sine_wave(
    batch_size: int = 4,
    seq_len: int = 32,
    freq: float = 1.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    dt: float = 0.1,
    noise: float = 0.0,
    *,
    seed: int = 0,
) -> jax.Array:
    """``amplitude * sin(freq * t * dt + phase)`` with a per-batch phase offset."""
    b = jnp.arange(batch_size)[:, None]
    t = jnp.arange(seq_len)[None, :]
    angle = freq * (t.astype(jnp.float32) * dt) + phase + b.astype(jnp.float32) * 0.5
    obs = (amplitude * jnp.sin(angle))[..., None]
    if noise > 0.0:
        obs = obs + noise * jax.random.normal(jax.random.PRNGKey(seed), obs.shape)
    return obs


def damped_oscillator(
    batch_size: int = 4,
    seq_len: int = 64,
    freq: float = 1.0,
    decay: float = 0.5,
    dt: float = 0.1,
    *,
    seed: int = 0,  # accepted for API uniformity (this generator is deterministic)
) -> jax.Array:
    """Damped harmonic oscillator state ``(x, x_dot)``."""
    b = jnp.arange(batch_size)[:, None]
    t = jnp.arange(seq_len)[None, :].astype(jnp.float32) * dt
    phase = 1.25 * b.astype(jnp.float32)
    env = jnp.exp(-decay * t)
    x = env * jnp.cos(freq * t + phase)
    v = env * (-decay * jnp.cos(freq * t + phase) - freq * jnp.sin(freq * t + phase))
    return jnp.stack([x, v], axis=-1)


def lissajous(
    batch_size: int = 4,
    seq_len: int = 64,
    a: float = 1.0,
    b_: float = 2.0,
    delta: float = 0.5,
    dt: float = 0.1,
    *,
    seed: int = 0,  # accepted for API uniformity (this generator is deterministic)
) -> jax.Array:
    """Lissajous curve ``(sin(a t), sin(b t + delta))``."""
    t = jnp.arange(seq_len)[None, :].astype(jnp.float32) * dt
    phase = 1.25 * jnp.arange(batch_size)[:, None].astype(jnp.float32)
    return jnp.stack([jnp.sin(a * t + phase), jnp.sin(b_ * t + delta + phase)], axis=-1)


def chaotic_circle(
    batch_size: int = 4,
    seq_len: int = 64,
    omega: float = 1.3,
    chaos: float = 0.35,
    dt: float = 0.3,
    *,
    seed: int = 0,
) -> jax.Array:
    """Circle whose angular velocity oscillates (a mild chaotic modulation)."""
    b = jnp.arange(batch_size)[:, None]
    t = jnp.arange(seq_len)[None, :].astype(jnp.float32)
    phase = 1.25 * b.astype(jnp.float32)
    angle = omega * (t * dt + chaos * jnp.sin(0.7 * t)) + phase
    return jnp.stack([jnp.cos(angle), jnp.sin(angle)], axis=-1)


def make_demo_data(
    kind: str = "circle",
    batch_size: int = 4,
    seq_len: int = 32,
    *,
    seed: int = 0,
    **kwargs,
) -> jax.Array:
    """Factory dispatching to one of the generators above."""
    generators = {
        "circle": circle_world,
        "sine": sine_wave,
        "damped": damped_oscillator,
        "lissajous": lissajous,
        "chaotic": chaotic_circle,
    }
    if kind not in generators:
        raise ValueError(f"unknown dataset {kind!r}; choose from {sorted(generators)}")
    return generators[kind](batch_size=batch_size, seq_len=seq_len, seed=seed, **kwargs)


# --------------------------------------------------------------------------- #
# tiny stories (character-level language-modelling demo)
# --------------------------------------------------------------------------- #
TINY_STORIES = (
    "the little robot learned to walk in the garden . "
    "it stumbled , then stood , then ran . "
    "the cat watched from the wall and purred . "
    "one day the robot drew a circle in the sand . "
    "the circle became a gate , and the gate became a road . "
    "the robot followed the road to the sea . "
    "at the sea the robot learned to swim . "
    "the waves taught it a rhythm it had never heard . "
)


@dataclass
class TinyStories:
    """Character-level tokeniser + next-token batcher for :data:`TINY_STORIES`.

    ``vocab_size`` is exposed so a model head can be sized directly; ``0`` is
    reserved as the padding token (as in `tiny-stories-jepa-demo`).
    """

    text: str = TINY_STORIES
    max_seq_len: int = 32

    def __post_init__(self):
        chars = sorted(set(self.text))
        self.itos = {i + 1: c for i, c in enumerate(chars)}
        self.stoi = {c: i for i, c in self.itos.items()}
        self.ids = jnp.asarray([self.stoi[c] for c in self.text], dtype=jnp.int32)

    @property
    def vocab_size(self) -> int:
        return len(self.itos) + 1  # +1 for the padding id 0

    def batch(self, batch_size: int = 4, seq_len: int | None = None) -> tuple[jax.Array, jax.Array]:
        """Return ``(inputs, targets)`` of shape ``[batch_size, seq_len]``."""
        seq_len = min(seq_len or self.max_seq_len, self.ids.shape[0] - 1)
        starts = jnp.arange(batch_size) * 7 % max(self.ids.shape[0] - seq_len - 1, 1)
        inputs = jnp.stack([self.ids[s : s + seq_len] for s in starts])
        targets = jnp.stack([self.ids[s + 1 : s + seq_len + 1] for s in starts])
        return inputs, targets

    def decode(self, ids) -> str:
        import numpy as np

        return "".join(self.itos.get(int(i), "") for i in np.asarray(ids).ravel())

    def batches(
        self, batch_size: int = 4, seq_len: int | None = None, n_batches: int = 8
    ) -> Iterator:
        """Deterministic stream of ``n_batches`` batches."""
        for i in range(n_batches):
            inputs, targets = self.batch(batch_size, seq_len)
            yield jnp.roll(inputs, i * 3, axis=1), jnp.roll(targets, i * 3, axis=1)
