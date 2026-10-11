"""Task definitions and distributions for meta-learning.

Ported from `ssm_latent_model::meta::tasks`.  A *task* is a concrete dynamics
model (support set to adapt on, query set to score the adaptation) and a
*distribution* is a key-deterministic sampler of those tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ssm_latent import data as _data

__all__ = [
    "Task",
    "TaskDistribution",
    "CircleWorldTask",
    "CircleWorldDistribution",
    "SineWaveTask",
    "SineWaveDistribution",
    "CompositeDistribution",
]


class Task(ABC):
    """A single meta-learning task."""

    task_id: str
    obs_dim: int = 1
    action_dim: int = 1
    success_threshold: float = 0.05

    @abstractmethod
    def support_set(self) -> tuple[jax.Array, jax.Array]:
        """``(observations, actions)`` used by the inner adaptation loop."""

    @abstractmethod
    def query_set(self) -> tuple[jax.Array, jax.Array]:
        """Held-out ``(observations, actions)`` used to score the adapted model."""


class TaskDistribution(ABC):
    """Distribution over tasks; ``sample_task(key)`` must be key-deterministic."""

    description: str = ""
    n_families: int = 1

    @abstractmethod
    def sample_task(self, key: jax.Array) -> Task: ...


# --------------------------------------------------------------------------- #
# CircleWorld
# --------------------------------------------------------------------------- #
@dataclass
class CircleWorldTask(Task):
    """A point moving on a unit circle with angular velocity ``omega``.

    The action is the analytic tangent, so the whole task is determined by
    ``omega`` - which makes it the canonical *adaptation-speed* benchmark: the
    model must infer the angular velocity from the support set and extrapolate.
    """

    omega: float = 1.0
    noise_level: float = 0.02
    phase_shift: float = 0.0
    n_support: int = 16
    n_query: int = 64
    batch_size: int = 4

    obs_dim: int = 2
    action_dim: int = 2
    success_threshold: float = 0.05

    def __post_init__(self):
        self.task_id = f"CircleWorld(omega={self.omega:.3f})"

    def _generate(self, n_steps: int):
        obs = _data.circle_world(
            batch_size=self.batch_size,
            seq_len=n_steps,
            omega=self.omega,
            phase_shift=self.phase_shift,
            noise=self.noise_level,
        )
        act = _data.circle_world_actions(
            batch_size=self.batch_size,
            seq_len=n_steps,
            omega=self.omega,
            phase_shift=self.phase_shift,
        )
        return obs, act

    def support_set(self):
        return self._generate(self.n_support)

    def query_set(self):
        return self._generate(self.n_query)


@dataclass
class CircleWorldDistribution(TaskDistribution):
    """``omega ~ U(min_omega, max_omega)`` with a random phase per task."""

    min_omega: float = 0.5
    max_omega: float = 2.0
    noise_level: float = 0.02
    n_support: int = 16
    n_query: int = 64
    batch_size: int = 4
    description: str = "CircleWorld(omega=[0.5, 2.0])"
    n_families: int = 1

    def sample_task(self, key: jax.Array) -> CircleWorldTask:
        k_omega, k_phase = jax.random.split(key)
        omega = jax.random.uniform(k_omega, (), minval=self.min_omega, maxval=self.max_omega)
        phase = jax.random.uniform(k_phase, (), minval=0.0, maxval=2.0 * jnp.pi)
        return CircleWorldTask(
            omega=float(omega),
            noise_level=self.noise_level,
            phase_shift=float(phase),
            n_support=self.n_support,
            n_query=self.n_query,
            batch_size=self.batch_size,
        )


# --------------------------------------------------------------------------- #
# SineWave
# --------------------------------------------------------------------------- #
@dataclass
class SineWaveTask(Task):
    """``amplitude * sin(freq * t * 0.1 + phase)`` (scalar observation)."""

    frequency: float = 1.0
    amplitude: float = 1.0
    phase: float = 0.0
    noise_level: float = 0.02
    n_support: int = 16
    n_query: int = 64
    batch_size: int = 4

    obs_dim: int = 1
    action_dim: int = 1
    success_threshold: float = 0.05

    def __post_init__(self):
        self.task_id = f"SineWave(f={self.frequency:.2f})"

    def _generate(self, n_steps: int):
        obs = _data.sine_wave(
            batch_size=self.batch_size,
            seq_len=n_steps,
            freq=self.frequency,
            amplitude=self.amplitude,
            phase=self.phase,
            noise=self.noise_level,
        )
        # the analytic time-derivative of the observation plays the role of the
        # action (same convention as CircleWorldTask)
        b = jnp.arange(self.batch_size, dtype=jnp.float32)[:, None]
        t = jnp.arange(n_steps, dtype=jnp.float32)[None, :] * 0.1
        angle = self.frequency * t + self.phase + b * 0.5
        act = (self.amplitude * self.frequency * jnp.cos(angle))[..., None]
        return obs, act

    def support_set(self):
        return self._generate(self.n_support)

    def query_set(self):
        return self._generate(self.n_query)


@dataclass
class SineWaveDistribution(TaskDistribution):
    """``freq ~ U(min_freq, max_freq)``, ``amplitude ~ U(min_amp, max_amp)``."""

    min_freq: float = 0.5
    max_freq: float = 3.0
    min_amplitude: float = 0.5
    max_amplitude: float = 2.0
    noise_level: float = 0.02
    n_support: int = 16
    n_query: int = 64
    batch_size: int = 4
    description: str = "SineWave(f=[0.5, 3.0])"
    n_families: int = 1

    def sample_task(self, key: jax.Array) -> SineWaveTask:
        k_freq, k_amp, k_phase = jax.random.split(key, 3)
        return SineWaveTask(
            frequency=float(
                jax.random.uniform(k_freq, (), minval=self.min_freq, maxval=self.max_freq)
            ),
            amplitude=float(
                jax.random.uniform(k_amp, (), minval=self.min_amplitude, maxval=self.max_amplitude)
            ),
            phase=float(jax.random.uniform(k_phase, (), minval=0.0, maxval=2.0 * jnp.pi)),
            noise_level=self.noise_level,
            n_support=self.n_support,
            n_query=self.n_query,
            batch_size=self.batch_size,
        )


@dataclass
class CompositeDistribution(TaskDistribution):
    """Uniform mixture of :class:`CircleWorldDistribution` and :class:`SineWaveDistribution`."""

    circle_dist: CircleWorldDistribution | None = None
    sine_dist: SineWaveDistribution | None = None
    description: str = "Composite[CircleWorld + SineWave]"
    n_families: int = 2

    def __post_init__(self):
        self.circle_dist = self.circle_dist or CircleWorldDistribution()
        self.sine_dist = self.sine_dist or SineWaveDistribution()

    def sample_task(self, key: jax.Array) -> Task:
        k_choice, k_task = jax.random.split(key)
        if float(jax.random.uniform(k_choice)) < 0.5:
            return self.circle_dist.sample_task(k_task)
        return self.sine_dist.sample_task(k_task)
