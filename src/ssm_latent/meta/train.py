"""Reptile meta-training: optimising the *initialisation* for fast adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import jax
from flax import nnx

from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta.adapt import AdaptationStrategy, LossWeights, adapt
from ssm_latent.meta.eval import AdaptationSpeedReport, evaluate_adaptation_speed
from ssm_latent.meta.tasks import Task, TaskDistribution

__all__ = [
    "ReptileMetaLearner",
    "MetaTrainMetrics",
    "MetaPipelineConfig",
    "run_meta_pipeline",
]


@dataclass
class MetaTrainMetrics:
    """Per-iteration diagnostics of :class:`ReptileMetaLearner`.

    Attributes
    ----------
    support_improvement:
        Mean reduction of the *inner-loop* (support) loss, i.e. how much the
        few adaptation steps move the parameters.  Always non-negative in
        expectation and is what the outer loop optimises.
    query_improvement:
        Mean reduction of the *post-adaptation query* loss (generalisation).
    avg_support_loss, avg_query_loss:
        Mean support / query losses after adaptation.
    success_rate:
        Fraction of sampled tasks whose post-adaptation *query* loss dropped
        below the task's ``success_threshold`` (the SAI "did it learn it"
        signal).
    avg_adaptation_time_ms:
        Wall-clock time of the inner loops.
    """

    iteration: int = 0
    support_improvement: float = 0.0
    query_improvement: float = 0.0
    avg_support_loss: float = 0.0
    avg_query_loss: float = 0.0
    success_rate: float = 0.0
    avg_adaptation_time_ms: float = 0.0

    # ----- backwards-compatible aliases (deprecated) ----------------------
    @property
    def avg_improvement(self) -> float:
        """Deprecated alias of :attr:`support_improvement`."""
        return self.support_improvement

    @property
    def avg_final_loss(self) -> float:
        """Deprecated alias of :attr:`avg_query_loss`."""
        return self.avg_query_loss


class ReptileMetaLearner:
    """Reptile (Nichol et al., 2018) outer loop over an SSM x JEPA world model.

    For every meta-iteration a task is sampled, the model is adapted to it
    (inner loop, ``inner_steps`` Adam steps) and the base parameters are moved a
    fraction ``outer_lr`` towards the adapted ones::

        theta <- theta + outer_lr * (adapted_theta - theta)

    ``outer_lr = 1.0`` recovers a hard copy of the adapted weights, so the full
    range from "gentle interpolation" to "copy" is available.
    """

    def __init__(
        self,
        model: MultiScaleLatentPredictor,
        task_dist: TaskDistribution | None = None,
        *,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        outer_lr: float = 0.1,
        tasks_per_batch: int = 1,
        loss_weights: LossWeights = LossWeights(),
        seed: int = 42,
        strategy: AdaptationStrategy = AdaptationStrategy.FULL,
    ):
        self.model = model
        self.task_dist = task_dist or _default_distribution()
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_lr = outer_lr
        self.tasks_per_batch = tasks_per_batch
        self.loss_weights = loss_weights
        self.seed = seed
        self.strategy = AdaptationStrategy(strategy)

    # ------------------------------------------------------------------ step
    def meta_train_step(self, iteration: int = 0) -> MetaTrainMetrics:
        """One Reptile iteration (``tasks_per_batch`` sampled tasks)."""
        self.seed += 1
        key = jax.random.PRNGKey(self.seed)

        sup_improvement = qry_improvement = 0.0
        sup_loss = qry_loss = success = adapt_time = 0.0
        for i in range(self.tasks_per_batch):
            task = self.task_dist.sample_task(jax.random.fold_in(key, i))
            adapted, result = adapt(
                self.model,
                task,
                inner_lr=self.inner_lr,
                inner_steps=self.inner_steps,
                strategy=self.strategy,
                loss_weights=self.loss_weights,
            )
            # support = inner-loop training loss (monotone-ish by construction);
            # query  = held-out loss whose *initial* value is the pre-adaptation
            # query loss.  The query improvement is therefore the SAI quantity
            # of interest and can legitimately be negative for an over-shooting
            # inner loop - which is exactly what we want to surface.
            sup_before = result.loss_curve[0] if result.loss_curve else result.initial_loss
            sup_improvement += sup_before - result.loss_curve[-1] if result.loss_curve else 0.0
            qry_improvement += result.initial_query_loss - result.final_loss
            sup_loss += result.loss_curve[-1] if result.loss_curve else sup_before
            qry_loss += result.final_loss
            success += float(result.success)
            adapt_time += result.adaptation_time_ms

            # Reptile interpolation towards the adapted parameters.
            base_nnx = nnx.state(self.model)
            base = base_nnx.to_pure_dict()
            target = nnx.state(adapted).to_pure_dict()
            new = jax.tree.map(lambda b, t: b + self.outer_lr * (t - b), base, target)
            base_nnx.replace_by_pure_dict(new)
            nnx.update(self.model, base_nnx)

        n = float(self.tasks_per_batch)
        return MetaTrainMetrics(
            iteration=iteration,
            support_improvement=sup_improvement / n,
            query_improvement=qry_improvement / n,
            avg_support_loss=sup_loss / n,
            avg_query_loss=qry_loss / n,
            success_rate=success / n,
            avg_adaptation_time_ms=adapt_time / n,
        )

    def train(self, num_iterations: int = 30, *, verbose: bool = True) -> list[MetaTrainMetrics]:
        """Run ``num_iterations`` Reptile steps."""
        metrics = []
        for i in range(num_iterations):
            m = self.meta_train_step(iteration=i + 1)
            metrics.append(m)
            if verbose and ((i + 1) % 10 == 0 or i == 0):
                print(
                    f"[Meta-Train {i + 1:3d}/{num_iterations}] "
                    f"support_gain={m.support_improvement:+.6f} "
                    f"query_gain={m.query_improvement:+.6f} "
                    f"query_loss={m.avg_query_loss:.6f} "
                    f"success_rate={m.success_rate:.2f}"
                )
        return metrics

    def evaluate(self, tasks, *, max_steps: int = 20) -> list[AdaptationSpeedReport]:
        """Adaptation-speed reports for a list of holdout tasks."""
        return [
            evaluate_adaptation_speed(
                self.model,
                task,
                inner_lr=self.inner_lr,
                max_steps=max_steps,
                loss_weights=self.loss_weights,
                strategy=self.strategy,
            )
            for task in tasks
        ]


@dataclass
class MetaPipelineConfig:
    """Hyper-parameters of :func:`run_meta_pipeline`."""

    num_iterations: int = 30
    inner_lr: float = 1e-3
    inner_steps: int = 5
    outer_lr: float = 0.1
    max_eval_steps: int = 20
    tasks_per_batch: int = 1


def _default_distribution() -> TaskDistribution:
    from ssm_latent.meta.tasks import CircleWorldDistribution

    return CircleWorldDistribution()


def run_meta_pipeline(
    model: MultiScaleLatentPredictor,
    task_dist: TaskDistribution,
    eval_tasks: list[Task],
    config: MetaPipelineConfig | None = None,
    *,
    verbose: bool = True,
) -> tuple[ReptileMetaLearner, list[MetaTrainMetrics], list[AdaptationSpeedReport]]:
    """Meta-train a model and evaluate adaptation speed on holdout tasks."""
    config = config or MetaPipelineConfig()
    learner = ReptileMetaLearner(
        model,
        task_dist,
        inner_lr=config.inner_lr,
        inner_steps=config.inner_steps,
        outer_lr=config.outer_lr,
        tasks_per_batch=config.tasks_per_batch,
    )
    if verbose:
        print("=== Meta-Training ===")
        print(f"Task distribution: {task_dist.description}")
        print(
            f"inner_lr={config.inner_lr} inner_steps={config.inner_steps} outer_lr={config.outer_lr}"
        )
    train_metrics = learner.train(config.num_iterations, verbose=verbose)
    reports = learner.evaluate(eval_tasks, max_steps=config.max_eval_steps)
    return learner, train_metrics, reports
