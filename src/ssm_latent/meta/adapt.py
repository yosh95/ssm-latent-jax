"""Inner-loop adaptation: the fast specialisation step of meta-learning."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import jax
import optax
from flax import nnx

from ssm_latent.latent import LatentLossArgs, MultiScaleLatentPredictor
from ssm_latent.meta.tasks import Task

__all__ = [
    "AdaptationStrategy",
    "AdaptationResult",
    "LossWeights",
    "adapt_loss",
    "inner_loop",
    "adapt",
    "ADAPTATION_STRATEGIES",
    "strategy_path_predicate",
    "strategy_mask",
    "mask_grads",
    "make_masked_train_step",
    "trainable_parameter_names",
]


class AdaptationStrategy(StrEnum):
    """Which parameter subsets are updated during inner-loop adaptation.

    This mirrors SAI's emphasis on *specialisation*: adapting only the encoder
    (new sensors) or only the slowest SSM layer (new environmental dynamics) can
    be far faster than adapting everything.
    """

    ENCODER_ONLY = "encoder_only"
    DECODER_ONLY = "decoder_only"
    SLOW_SSM_ONLY = "slow_ssm_only"
    FULL = "full"
    ENCODER_PLUS_SLOW_SSM = "encoder_plus_slow_ssm"

    @property
    def description(self) -> str:
        return {
            "encoder_only": "Encoder only",
            "decoder_only": "Decoder only",
            "slow_ssm_only": "Slow SSM layer only",
            "full": "All parameters",
            "encoder_plus_slow_ssm": "Encoder + Slow SSM",
        }[self.value]


ADAPTATION_STRATEGIES = tuple(AdaptationStrategy)


@dataclass
class LossWeights:
    """Weights of the JEPA loss used to drive adaptation."""

    recon: float = 1.0
    stability: float = 0.01
    curvature: float = 0.005


@dataclass
class AdaptationResult:
    """Outcome of one inner-loop adaptation run.

    Loss bookkeeping (all on the model's JEPA objective):

    * ``initial_support_loss`` - support loss before any step.
    * ``loss_curve``           - support loss recorded at the end of each step
                                 (so ``loss_curve[0]`` is the loss *after* one
                                 gradient step; the pre-step value is
                                 ``initial_support_loss``).
    * ``initial_query_loss``   - held-out (query) loss before any step.
    * ``final_loss``           - query loss after the final step; this is the
                                 generalisation number the SAI metric cares
                                 about and drives ``success``.
    """

    initial_support_loss: float = 0.0
    final_loss: float = 0.0
    loss_curve: list[float] = field(default_factory=list)
    initial_query_loss: float = 0.0
    steps_taken: int = 0
    success: bool = False
    adaptation_time_ms: float = 0.0

    @property
    def initial_loss(self) -> float:
        """Deprecated alias of :attr:`initial_support_loss`."""
        return self.initial_support_loss


def adaptation_strategy_description(strategy: AdaptationStrategy) -> str:
    """Human-readable description of a strategy."""
    return AdaptationStrategy(strategy).description


def adapt_loss(
    model: MultiScaleLatentPredictor,
    observations: jax.Array,
    actions: jax.Array | None,
    weights: LossWeights = LossWeights(),
) -> jax.Array:
    """JEPA loss of ``model`` on a batch, ready to be differentiated."""
    if model.action_encoder is not None and actions is not None:
        z, pred_z, recon_x, pred_x = model.forward(observations, actions)
    else:
        z, pred_z, recon_x, pred_x = model.forward_no_action(observations)
    args = LatentLossArgs(
        z=z,
        pred_z=pred_z,
        reconstructed_x=recon_x,
        predicted_x=pred_x,
        original_x=observations,
        stability_weight=weights.stability,
        curvature_weight=weights.curvature,
        recon_weight=weights.recon,
    )
    return model.loss(args)


def strategy_path_predicate(strategy: AdaptationStrategy, n_layers: int):
    """Return ``predicate(path, value) -> bool`` selecting a strategy's parameters.

    ``path`` is the tuple of attribute names inside the ``nnx.Module`` (e.g.
    ``("ssm", "layers", "2", "a_re")``), so the predicate is independent of the
    array shapes and can be reused as an :mod:`optax` mask.
    """
    strategy = AdaptationStrategy(strategy)

    def predicate(path: tuple, value=None) -> bool:
        p = tuple(str(k) for k in path)
        if strategy is AdaptationStrategy.FULL:
            return True
        if strategy is AdaptationStrategy.ENCODER_ONLY:
            return p[0] == "encoder"
        if strategy is AdaptationStrategy.DECODER_ONLY:
            return p[0] == "decoder"
        if strategy is AdaptationStrategy.SLOW_SSM_ONLY:
            if p[0] != "ssm" or p[1] != "layers":
                return False
            # "slow" == the last layer of the multi-scale stack
            return len(p) > 2 and p[2] == str(n_layers - 1)
        if strategy is AdaptationStrategy.ENCODER_PLUS_SLOW_SSM:
            if p[0] == "encoder":
                return True
            if p[0] != "ssm" or p[1] != "layers":
                return False
            return len(p) > 2 and p[2] == str(n_layers - 1)
        raise ValueError(f"unknown strategy {strategy!r}")

    return predicate


def strategy_mask(model: MultiScaleLatentPredictor, strategy: AdaptationStrategy):
    """Boolean pytree (plain nested dicts) selecting a strategy's parameters.

    The mask mirrors the structure produced by ``nnx.to_pure_dict`` - which is
    the layout :class:`flax.nnx.Optimizer` hands to Optax - so it can be passed
    straight to :func:`optax.masked`.
    """
    predicate = strategy_path_predicate(strategy, model.ssm.n_layers)
    # `nnx.as_pure` strips the Variable wrappers: this is exactly the layout the
    # NNX optimiser hands to Optax, so the mask lines up leaf-by-leaf.
    pure = nnx.as_pure(nnx.state(model, nnx.Param))
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: bool(predicate(tuple(getattr(k, "key", k) for k in path), leaf)),
        pure,
        is_leaf=lambda x: hasattr(x, "shape"),
    )


def trainable_parameter_names(
    model: MultiScaleLatentPredictor, strategy: AdaptationStrategy
) -> list[str]:
    """Human-readable list of the parameter paths a strategy updates."""
    predicate = strategy_path_predicate(strategy, model.ssm.n_layers)
    flat = jax.tree_util.tree_flatten_with_path(nnx.as_pure(nnx.state(model, nnx.Param)))[0]
    return [
        "/".join(str(getattr(k, "key", k)) for k in path)
        for path, leaf in flat
        if predicate(tuple(getattr(k, "key", k) for k in path), leaf)
    ]


def mask_grads(grads, mask):
    """Zero the gradient leaves whose entry in ``mask`` is ``False``.

    ``grads`` is the ``nnx.Param`` state produced by ``nnx.grad``; ``mask`` is a
    matching pytree of Python booleans from :func:`strategy_mask`.  The result is
    again an ``nnx.Param`` state, i.e. directly consumable by ``Optimizer.update``.

    Because Adam's update is proportional to the gradient, zeroed leaves stay
    *bit-identical*: the excluded parameters never move and no optimiser state is
    created for them in a way that could leak an update.

    ``None`` gradients (parameters that do not affect the loss) are preserved.
    """
    return jax.tree_util.tree_map(
        lambda g, k: None if g is None else nnx.Param(g[...] * k),
        grads,
        mask,
        is_leaf=lambda x: isinstance(x, (bool, nnx.Param)),
    )


def make_masked_train_step(
    model: MultiScaleLatentPredictor,
    optimizer,
    keep_mask,
    loss_weights: LossWeights,
):
    """Build the jitted ``(model, optimizer, obs, act) -> loss`` adaptation step.

    Factored out so that :func:`inner_loop` (support-loss curve) and
    :func:`~ssm_latent.meta.eval.adaptation_curve` (held-out query curve) share
    *one* implementation - the two only differ in which set they score.
    """

    def loss_fn(model, obs, act):
        return adapt_loss(model, obs, act, loss_weights)

    @nnx.jit
    def train_step(model, optimizer, obs, act):
        loss, grads = nnx.value_and_grad(loss_fn)(model, obs, act)
        optimizer.update(model, mask_grads(grads, keep_mask))
        return loss

    return train_step, loss_fn


def inner_loop(
    model: MultiScaleLatentPredictor,
    task: Task,
    *,
    inner_lr: float = 1e-3,
    inner_steps: int = 5,
    strategy: AdaptationStrategy = AdaptationStrategy.FULL,
    loss_weights: LossWeights = LossWeights(),
) -> tuple[MultiScaleLatentPredictor, AdaptationResult]:
    """Adapt ``model`` to ``task`` with ``inner_steps`` of Adam.

    The inner loop is deterministic (its only randomness is the task's own data),
    so there is no ``seed``; parameter updates happen in-place through the model's
    NNX state and only the subset selected by ``strategy`` receives gradients.

    Returns
    -------
    ``(adapted_model, AdaptationResult)`` - ``adapted_model`` is ``model`` itself
    (mutated in place); use :func:`~ssm_latent.meta.adapt.adapt` to adapt a copy.
    """
    start = time.perf_counter()
    strategy = AdaptationStrategy(strategy)

    # Strategy selection is implemented by *masking the gradients* rather than
    # by using `optax.masked`: with zero gradients Adam produces exactly zero
    # updates, so the excluded parameters stay bit-identical and the optimiser
    # state keeps the same dense structure as the parameters.
    keep_mask = strategy_mask(model, strategy)
    optimizer = nnx.Optimizer(model, optax.adam(inner_lr), wrt=nnx.Param)

    support_obs, support_act = task.support_set()
    query_obs, query_act = task.query_set()

    train_step, loss_fn = make_masked_train_step(model, optimizer, keep_mask, loss_weights)

    initial_support_loss = float(loss_fn(model, support_obs, support_act))
    initial_query_loss = float(loss_fn(model, query_obs, query_act))
    curve: list[float] = []

    for _ in range(inner_steps):
        loss = train_step(model, optimizer, support_obs, support_act)
        curve.append(float(loss))

    final_loss = float(loss_fn(model, query_obs, query_act))
    return model, AdaptationResult(
        initial_support_loss=initial_support_loss,
        final_loss=final_loss,
        loss_curve=curve,
        initial_query_loss=initial_query_loss,
        steps_taken=inner_steps,
        success=final_loss < task.success_threshold,
        adaptation_time_ms=(time.perf_counter() - start) * 1000.0,
    )


def adapt(
    model: MultiScaleLatentPredictor,
    task: Task,
    *,
    inner_lr: float = 1e-3,
    inner_steps: int = 5,
    strategy: AdaptationStrategy = AdaptationStrategy.FULL,
    loss_weights: LossWeights = LossWeights(),
) -> tuple[MultiScaleLatentPredictor, AdaptationResult]:
    """Like :func:`inner_loop` but adapts a *clone*, leaving ``model`` untouched."""
    clone = nnx.clone(model)
    return inner_loop(
        clone,
        task,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        strategy=strategy,
        loss_weights=loss_weights,
    )
