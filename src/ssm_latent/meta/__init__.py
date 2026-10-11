"""Meta-learning for SAI ("Superhuman Adaptable Intelligence").

* :mod:`~ssm_latent.meta.tasks`  - task definitions and distributions
* :mod:`~ssm_latent.meta.adapt`  - inner-loop adaptation + strategies
* :mod:`~ssm_latent.meta.eval`   - adaptation-speed metrics (the SAI core metric)
* :mod:`~ssm_latent.meta.train`  - Reptile outer loop
"""

from ssm_latent.meta.adapt import (
    ADAPTATION_STRATEGIES,
    AdaptationResult,
    AdaptationStrategy,
    LossWeights,
    adapt,
    adapt_loss,
    adaptation_strategy_description,
    inner_loop,
    strategy_mask,
    strategy_path_predicate,
    trainable_parameter_names,
)
from ssm_latent.meta.eval import (
    AdaptationSpeedReport,
    evaluate_adaptation_speed,
)
from ssm_latent.meta.tasks import (
    CircleWorldDistribution,
    CircleWorldTask,
    CompositeDistribution,
    SineWaveDistribution,
    SineWaveTask,
    Task,
    TaskDistribution,
)
from ssm_latent.meta.train import (
    MetaPipelineConfig,
    MetaTrainMetrics,
    ReptileMetaLearner,
    run_meta_pipeline,
)

__all__ = [
    "Task",
    "TaskDistribution",
    "CircleWorldTask",
    "CircleWorldDistribution",
    "SineWaveTask",
    "SineWaveDistribution",
    "CompositeDistribution",
    "ADAPTATION_STRATEGIES",
    "AdaptationStrategy",
    "adaptation_strategy_description",
    "strategy_mask",
    "strategy_path_predicate",
    "trainable_parameter_names",
    "AdaptationResult",
    "LossWeights",
    "adapt_loss",
    "inner_loop",
    "adapt",
    "AdaptationSpeedReport",
    "evaluate_adaptation_speed",
    "ReptileMetaLearner",
    "MetaTrainMetrics",
    "MetaPipelineConfig",
    "run_meta_pipeline",
]
