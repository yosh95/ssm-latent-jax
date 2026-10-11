"""ssm_latent — Mamba-3 x JEPA latent world models in JAX / Flax NNX.

A research toolkit for *latent world modelling*: selective state-space dynamics
(Mamba-3) fused with a Joint-Embedding Predictive Architecture (JEPA), plus the
LeJEPA collapse-prevention objective and the identifiability monitors that go
with it.  Everything is built on the functional JAX core (``jit`` / ``grad`` /
``vmap`` / ``lax.scan``) and Flax NNX, so models are hackable in a notebook,
composable, and verifiable with ``pytest``.

Layout
------
- :mod:`ssm_latent.ssm`          selective SSM block (parallel scan + O(1) step)
- :mod:`ssm_latent.latent`       JEPA latent predictors and loss functions
- :mod:`ssm_latent.monitor`      LeJEPA identifiability monitors (Theorems 1-4)
- :mod:`ssm_latent.meta`         Reptile meta-learning and adaptation-speed eval
- :mod:`ssm_latent.multimodal`   vision + sensor + action fusion
- :mod:`ssm_latent.data`         synthetic physical-dynamics datasets

This package is the primary reference implementation.  A Rust/Burn sibling
(`ssm-latent-rs`) and a small PyTorch sibling (`ssm-latent-py`) exist for
deployment and comparison; the numerical semantics are intentionally shared so
results across the three stay comparable.
"""

from ssm_latent import (
    data,
    latent,
    meta,
    monitor,
    multimodal,
    predictor,
    ssm,
    utils,
)
from ssm_latent.config import MultiScaleSsmConfig, SsmConfig
from ssm_latent.latent import (
    LatentLossArgs,
    LatentPredictor,
    LatentState,
    MultiScaleLatentPredictor,
    MultiscaleLatentState,
    latent_loss,
)
from ssm_latent.predictor import (
    MambaPredictor,
    MultiScaleMambaPredictor,
    MultiScalePredictorState,
    PredictorState,
)
from ssm_latent.ssm import (
    MultiScaleSsmBlock,
    MultiScaleSsmState,
    SsmBlock,
    SsmState,
)

__version__ = "0.1.0"

__all__ = [
    "MultiScaleLatentPredictor",
    "LatentPredictor",
    "LatentState",
    "MultiscaleLatentState",
    "LatentLossArgs",
    "latent_loss",
    "SsmBlock",
    "SsmState",
    "MultiScaleSsmBlock",
    "MultiScaleSsmState",
    "SsmConfig",
    "MultiScaleSsmConfig",
    "MambaPredictor",
    "MultiScaleMambaPredictor",
    "PredictorState",
    "MultiScalePredictorState",
    "data",
    "latent",
    "meta",
    "monitor",
    "multimodal",
    "predictor",
    "ssm",
    "utils",
    "__version__",
]
