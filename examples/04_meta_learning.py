"""Example 4 - meta-learning for SAI: adaptation *speed* as the metric.

Goldfeder, Wyder, LeCun & Shwartz-Ziv (2026) argue that the quantity that
matters is not generality but how *fast* an agent acquires new skills.  This
example

1. measures the adaptation speed of a freshly initialised world model on
   CircleWorld tasks (including two outside the training range),
2. meta-trains it with Reptile,
3. re-measures, and
4. compares the parameter-subset adaptation strategies (encoder only, slow SSM
   layer only, ...) that SAI's "specialisation" thesis suggests.

Run: ``python examples/04_meta_learning.py``
"""

import jax
import jax.numpy as jnp
from common import banner, section
from flax import nnx

from ssm_latent.config import MultiScaleSsmConfig
from ssm_latent.latent import MultiScaleLatentPredictor
from ssm_latent.meta import (
    AdaptationStrategy,
    CircleWorldDistribution,
    CircleWorldTask,
    MetaPipelineConfig,
    ReptileMetaLearner,
    adaptation_strategy_description,
    evaluate_adaptation_speed,
    run_meta_pipeline,
    trainable_parameter_names,
)
from ssm_latent.utils import count_parameters


def build_model(seed: int = 0):
    cfg = MultiScaleSsmConfig(
        d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, n_layers=3, use_conv=True
    )
    return MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(seed))


def holdout_tasks():
    """Two extrapolation tasks plus one robustness task, all unseen."""
    return [
        CircleWorldTask(
            omega=0.3, noise_level=0.02, phase_shift=0.0, n_support=16, n_query=32, batch_size=4
        ),
        CircleWorldTask(
            omega=2.5, noise_level=0.02, phase_shift=1.5, n_support=16, n_query=32, batch_size=4
        ),
        CircleWorldTask(
            omega=1.2, noise_level=0.05, phase_shift=3.0, n_support=16, n_query=32, batch_size=4
        ),
    ]


def report(model, tasks, label: str, max_steps: int = 20):
    print(f"\n  {label}")
    print(f"  {'task':<28}{'steps->success':>16}{'AUC (lower=faster)':>22}{'init rate':>12}")
    for task in tasks:
        rep = evaluate_adaptation_speed(model, task, inner_lr=3e-3, max_steps=max_steps)
        steps = "never" if rep.steps_to_success is None else str(rep.steps_to_success)
        print(
            f"  {task.task_id:<28}{steps:>16}{rep.adaptation_auc:>22.4f}{rep.initial_improvement_rate:>12.4f}"
        )
    return [
        evaluate_adaptation_speed(model, task, inner_lr=3e-3, max_steps=max_steps) for task in tasks
    ]


def main() -> None:
    banner("SAI: optimising adaptation speed with Reptile")

    model = build_model()
    print(f"world model: {count_parameters(model)} parameters")
    tasks = holdout_tasks()

    section("before meta-training (random initialisation)")
    before = report(model, tasks, "from scratch")

    section("meta-training (Reptile over CircleWorld)")
    dist = CircleWorldDistribution(
        min_omega=0.5, max_omega=2.0, n_support=16, n_query=32, batch_size=4
    )
    config = MetaPipelineConfig(
        num_iterations=40, inner_lr=3e-3, inner_steps=5, outer_lr=0.3, max_eval_steps=20
    )
    learner, metrics, _ = run_meta_pipeline(model, dist, tasks, config, verbose=True)

    section("after meta-training")
    after = report(learner.model, tasks, "meta-learned")

    section("summary")
    print(f"  {'task':<28}{'AUC before':>12}{'AUC after':>12}{'delta':>10}")
    for task, b, a in zip(tasks, before, after, strict=True):
        print(
            f"  {task.task_id:<28}{b.adaptation_auc:>12.4f}{a.adaptation_auc:>12.4f}"
            f"{a.adaptation_auc - b.adaptation_auc:>+10.4f}"
        )
    print(
        f"\n  meta-training loss curve (first/last): "
        f"{metrics[0].avg_final_loss:.4f} -> {metrics[-1].avg_final_loss:.4f}"
    )

    section("parameter-subset adaptation strategies (SAI specialisation)")
    for strategy in AdaptationStrategy:
        names = trainable_parameter_names(learner.model, strategy)
        rep = evaluate_adaptation_speed(
            learner.model, tasks[0], inner_lr=3e-3, max_steps=20, strategy=strategy
        )
        steps = "never" if rep.steps_to_success is None else str(rep.steps_to_success)
        print(
            f"  {adaptation_strategy_description(strategy):<24} "
            f"params={len(names):3d}  steps->success={steps:>5}  AUC={rep.adaptation_auc:.4f}"
        )

    section("take-away")
    print("  Meta-training moves the *initialisation* into a region from which a few")
    print("  gradient steps suffice - the operational meaning of adaptation speed.")
    print("  Updating only a subset of parameters can be markedly faster, which is the")
    print("  empirical counterpart of SAI's specialisation argument.")


if __name__ == "__main__":
    main()
