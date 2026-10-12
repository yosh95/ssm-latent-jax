"""Adaptation-speed evaluation - the core metric of SAI.

Goldfeder, Wyder, LeCun & Shwartz-Ziv (2026) argue that the right measure of
intelligence is not generality but *adaptation speed*: how many gradient steps
does an agent need to master a new task?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from flax import nnx

from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta.adapt import AdaptationStrategy, LossWeights, adapt
from ssm_latent.meta.tasks import Task

__all__ = ["AdaptationSpeedReport", "adaptation_curve", "evaluate_adaptation_speed"]


@dataclass
class AdaptationSpeedReport:
    """Curve and summary statistics of an adaptation run."""

    adaptation_curve: list[float] = field(default_factory=list)
    loss_at_step_1: float = 0.0
    loss_at_step_5: float = 0.0
    loss_at_step_10: float = 0.0
    loss_at_step_20: float = 0.0
    steps_to_success: int | None = None
    initial_improvement_rate: float = 0.0
    adaptation_auc: float = 0.0
    final_loss: float = 0.0
    success: bool = False

    @classmethod
    def from_curve(cls, curve, threshold: float = 0.05) -> AdaptationSpeedReport:
        """Build a report from a loss curve (1-indexed ``steps_to_success``)."""
        curve = [float(c) for c in np.asarray(curve).ravel()]
        if not curve:
            return cls()

        steps_to_success = next((i + 1 for i, loss in enumerate(curve) if loss < threshold), None)
        init_rate = (curve[0] - curve[1]) / curve[0] if len(curve) >= 2 and curve[0] > 0 else 0.0
        if len(curve) >= 2:
            auc = float(np.trapezoid(np.asarray(curve))) / len(curve)
        else:
            auc = curve[0]

        def _at(i):
            return curve[i - 1] if len(curve) >= i else 0.0

        return cls(
            adaptation_curve=curve,
            loss_at_step_1=_at(1),
            loss_at_step_5=_at(5),
            loss_at_step_10=_at(10),
            loss_at_step_20=_at(20),
            steps_to_success=steps_to_success,
            initial_improvement_rate=init_rate,
            adaptation_auc=auc,
            final_loss=curve[-1],
            success=steps_to_success is not None,
        )


def adaptation_curve(
    model: MultiScaleLatentPredictor,
    task: Task,
    *,
    inner_lr: float = 1e-3,
    max_steps: int = 20,
    loss_weights: LossWeights = LossWeights(),
    strategy: AdaptationStrategy = AdaptationStrategy.FULL,
) -> list[float]:
    """Held-out (query) loss after each of ``max_steps`` adaptation steps.

    This is the query-side twin of
    :func:`~ssm_latent.meta.adapt.inner_loop`: the *support* set drives the
    updates while the *query* set is scored, because the SAI metric is about
    generalisation speed, not how far the optimiser moves on the support set.

    ``model`` is adapted **in place**; pass :func:`flax.nnx.clone` if you need to
    keep the original.  Both this and
    :func:`~ssm_latent.meta.adapt.inner_loop` share one jitted stepping routine
    (:func:`~ssm_latent.meta.adapt.make_masked_train_step`), so they can never
    drift apart.
    """
    import optax

    from ssm_latent.meta.adapt import make_masked_train_step, strategy_mask

    strategy = AdaptationStrategy(strategy)
    keep_mask = strategy_mask(model, strategy)
    optimizer = nnx.Optimizer(model, optax.adam(inner_lr), wrt=nnx.Param)

    support_obs, support_act = task.support_set()
    query_obs, query_act = task.query_set()

    train_step, loss_fn = make_masked_train_step(model, optimizer, keep_mask, loss_weights)

    curve = []
    for _ in range(max_steps):
        train_step(model, optimizer, support_obs, support_act)
        curve.append(float(loss_fn(model, query_obs, query_act)))
    return curve


def evaluate_adaptation_speed(
    model: MultiScaleLatentPredictor,
    task: Task,
    *,
    inner_lr: float = 1e-3,
    max_steps: int = 20,
    loss_weights: LossWeights = LossWeights(),
    strategy: AdaptationStrategy = AdaptationStrategy.FULL,
) -> AdaptationSpeedReport:
    """Evaluate how quickly a *copy* of ``model`` adapts to ``task``.

    This is the function used to validate the SAI claim in the README: a small
    SSM x JEPA world model should reach the task threshold in a handful of steps.
    """
    clone = nnx.clone(model)
    _, result = adapt(
        clone,
        task,
        inner_lr=inner_lr,
        inner_steps=max_steps,
        strategy=strategy,
        loss_weights=loss_weights,
    )
    return AdaptationSpeedReport.from_curve(result.loss_curve, task.success_threshold)
