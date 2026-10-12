"""The LLM side of the loop - a *loosely coupled* transport, not an algorithm.

Design note (this is the whole point of the experiment)
--------------------------------------------------------
We are **not** wiring a language model's parameters, logits or hidden states into
the world model.  That would be an algorithmic coupling: the LLM would be a
differentiable layer, and its contribution would be bounded by the base model's
knowledge (RL/FT merely re-weights the paths the base model already has - see
arXiv:2504.13837, where the base model catches up as k grows).

Instead the coupling is **loose**, the way a driver is coupled to a car: the LLM
only *acts* on the world model through the same surface a human would use - it
chooses a task, hands over a short demonstration, and reads back numbers.  The
world model is a tool that lets the (frozen) LLM reach tasks it could not otherwise
actuating.  The claim under test is that this extends what the *pair* can do beyond
what the base model can do alone - without touching a single weight.

This module is the seam.  It defines the two messages that cross it and a fake
driver used for tests; the real transports (an OpenAI-compatible endpoint, or
`llm-cli-py`) implement the same one method later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import jax
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.latent import MultiScaleLatentPredictor

# The world model the driver is allowed to use, and the observation it gets.
D_MODEL, D_STATE, N_HEADS, N_LAYERS = 32, 8, 4, 3
N_SUPPORT, N_QUERY = 16, 32
BATCH = 4


# --------------------------------------------------------------------------- #
# messages that cross the loose seam
# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec:
    """What the driver *chooses*: one task, declared by its physical parameter."""

    omega: float
    noise_level: float = 0.02
    phase_shift: float = 0.0

    def describe(self) -> str:
        return (
            f"target dynamics: a point on the unit circle with angular velocity "
            f"omega={self.omega:.3f} (noise={self.noise_level:.3f})"
        )


@dataclass
class Demo:
    """The driver's demonstration - the *only* physical information it hands over."""

    observations: jax.Array  # [B, N_SUPPORT, 2]
    actions: jax.Array  # [B, N_SUPPORT, 2]


@dataclass
class Feedback:
    """The driver's *observation* after acting, so it can close its own loop."""

    adaptation_steps: int | None
    adapted_loss: float
    persistence_rmse: float
    imagination_rmse: float
    imagination_gain: float
    detail: dict = field(default_factory=dict)

    def as_text(self) -> str:
        steps = "never" if self.adaptation_steps is None else str(self.adaptation_steps)
        return (
            f"steps_to_adapt={steps} adapted_loss={self.adapted_loss:.4f} "
            f"imagination_rmse={self.imagination_rmse:.4f} "
            f"persistence_rmse={self.persistence_rmse:.4f} "
            f"gain={self.imagination_gain:+.4f}"
        )


def build_world_model(seed: int = 0) -> MultiScaleLatentPredictor:
    """A fresh, *untrained* world model - the generic starting point the driver gets."""
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


class Driver(Protocol):
    """The loose seam.  A driver picks tasks and sketches a demonstration for each."""

    name: str

    def propose_tasks(self, n: int, *, key: jax.Array) -> list[TaskSpec]: ...

    def demonstrate(self, task: TaskSpec, *, key: jax.Array) -> Demo: ...


def _gen(task: TaskSpec, n_steps: int, batch: int, key: jax.Array) -> tuple[jax.Array, jax.Array]:
    from ssm_latent.data import circle_world, circle_world_actions

    obs = circle_world(
        batch,
        n_steps,
        omega=task.omega,
        phase_shift=task.phase_shift,
        noise=task.noise_level,
        seed=int(key[0]) if task.noise_level else 0,
    )
    act = circle_world_actions(batch, n_steps, omega=task.omega, phase_shift=task.phase_shift)
    return obs, act


class DemoTask:
    """Adapts the loose-seam `Demo` into the library's `Task` interface.

    support = exactly the driver's demonstration (what the LLM hands over);
    query   = a fresh draw at the same physical parameters (what "did it work?"
    means operationally - an unseen phase of the *same* task).
    """

    success_threshold = 0.05

    def __init__(self, spec: TaskSpec, demo: Demo, *, n_query: int = N_QUERY, batch: int = BATCH):
        self.task_id = f"LLM-demo(omega={spec.omega:.3f})"
        self.omega = spec.omega
        self.obs_dim = 2
        self.action_dim = 2
        self.noise_level = spec.noise_level
        self.phase_shift = spec.phase_shift
        self._support = (demo.observations, demo.actions)
        self._n_query = n_query
        self._batch = batch

    def support_set(self):
        return self._support

    def query_set(self):
        obs, act = _gen(
            TaskSpec(self.omega, self.noise_level, self.phase_shift),
            self._n_query,
            self._batch,
            jax.random.PRNGKey(12345),
        )
        return obs, act


# --------------------------------------------------------------------------- #
# a deterministic, scripted driver - the reference "human" for tests/baselines
# --------------------------------------------------------------------------- #
class ScriptedDriver:
    """Sweeps the requested omegas in order.  Deterministic, no model, no network.

    This is the control: it shows what the loop can do when the task choice is not
    the interesting variable.  A real LLM driver should beat it precisely by
    *choosing well* (the tasks and demonstrations that maximise what the pair can
    do), which is what makes this a fair thing to hold fixed.
    """

    name = "scripted"

    def __init__(self, omegas):
        self.omegas = list(omegas)

    def propose_tasks(self, n: int, *, key: jax.Array) -> list[TaskSpec]:
        return [TaskSpec(omega=w) for w in self.omegas[:n]]

    def demonstrate(self, task: TaskSpec, *, key: jax.Array) -> Demo:
        obs, act = _gen(task, N_SUPPORT, BATCH, key)
        return Demo(obs, act)


# --------------------------------------------------------------------------- #
# the (future) LLM driver - declared here so the seam is honest about its shape
# --------------------------------------------------------------------------- #
class LLMDriver:
    """Asks a *frozen* LLM (via an OpenAI-compatible endpoint) to choose and demonstrate.

    Not wired to a network yet on purpose: the point of the first experiment is to
    fix the *protocol* (what the driver sees, what it sends) with the scripted
    control, so that when the LLM is dropped in, a bad result can be attributed to
    the driver rather than to the transport.  Environment variables (``LLM_CLI_*``)
    are read only at construction so importing this file never touches the network.
    """

    def __init__(
        self, model: str | None = None, api_url: str | None = None, api_key: str | None = None
    ):
        self.model = model or os.environ.get("LLM_CLI_MODEL", "")
        self.api_url = api_url or os.environ.get("LLM_CLI_API_URL", "")
        self.api_key = api_key or os.environ.get("LLM_CLI_API_KEY", "")
        self.name = f"llm:{self.model or 'unset'}"

    def propose_tasks(self, n: int, *, key: jax.Array) -> list[TaskSpec]:
        raise NotImplementedError(
            "LLM transport not wired yet - run with the scripted control first; see "
            "experiments/README.md"
        )

    def demonstrate(self, task: TaskSpec, *, key: jax.Array) -> Demo:
        raise NotImplementedError
