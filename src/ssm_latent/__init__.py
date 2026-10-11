"""ssm_latent — a JAX/Flax NNX port of the Rust `ssm-latent-model` crate.

Mamba-3 (selective state space model) x JEPA (joint-embedding predictive
architecture) for latent world modelling, targeting research, verification and
prototyping workflows where Rust's compile times and intermediate artefacts get
in the way.

Layout
------
- :mod:`ssm_latent.ssm`          selective SSM block (parallel scan + O(1) step)
- :mod:`ssm_latent.latent`       JEPA latent predictors and loss functions
- :mod:`ssm_latent.monitor`      LeJEPA identifiability monitors (Theorems 1-4)
- :mod:`ssm_latent.meta`         Reptile meta-learning and adaptation-speed eval
- :mod:`ssm_latent.multimodal`   vision + sensor + action fusion
- :mod:`ssm_latent.data`         synthetic physical-dynamics datasets

The numerical semantics (including quirks such as the head-wise `theta`
parameter and the multi-scale `1/n_layers` output scaling) deliberately mirror
the reference Rust implementation so that results are directly comparable.
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
