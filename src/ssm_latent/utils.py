"""Small numerical helpers shared across the package."""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "normalize_projections",
    "leaky_relu",
    "tree_l2_distance",
    "count_parameters",
    "pretty_params",
    "save_model",
    "load_model",
    "ModelIOMixin",
]


def normalize_projections(w: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Normalise every column of a random projection matrix to unit length.

    The ``eps`` is added to the norm *before* dividing (rather than clamping the
    norm from below).  This keeps the operation smooth and finite even for a
    zero column and matches the semantics used by the sibling implementations.
    """
    w = jnp.asarray(w)
    norm = jnp.sqrt(jnp.sum(w**2, axis=0, keepdims=True)) + eps
    return w / norm


def leaky_relu(x: jax.Array, slope: float = 0.01) -> jax.Array:
    """Leaky ReLU, used by the closed-loop "imagination" paths."""
    x = jnp.asarray(x)
    return jnp.where(x >= 0, x, slope * x)


def tree_l2_distance(tree_a, tree_b) -> float:
    """L2 distance between two pytrees of arrays with identical structure."""
    leaves_a = jax.tree.leaves(tree_a)
    leaves_b = jax.tree.leaves(tree_b)
    if len(leaves_a) != len(leaves_b):
        raise ValueError("pytrees have different structure")
    total = 0.0
    for a, b in zip(leaves_a, leaves_b, strict=True):
        diff = jnp.asarray(a) - jnp.asarray(b)
        total += float(jnp.sum(diff**2))
    return float(jnp.sqrt(total))


def count_parameters(model) -> int:
    """Number of scalar parameters stored in a flax ``nnx.Module``."""
    from flax import nnx

    state = nnx.state(model, nnx.Param)

    def _n(x):
        return int(jnp.size(x)) if hasattr(x, "shape") else 0

    return sum(_n(x) for x in jax.tree.leaves(state))


def pretty_params(model, filter=None) -> list[tuple[str, tuple[int, ...]]]:
    """``[(path, shape), ...]`` for every array variable of an ``nnx.Module``.

    By default *all* :class:`flax.nnx.Variable` collections are listed (both
    trainable ``Param``s and frozen buffers such as the random projections).
    """
    from flax import nnx

    state = nnx.state(model, filter if filter is not None else nnx.Variable)
    flat = jax.tree_util.tree_flatten_with_path(state)[0]
    return [
        ("/".join(str(getattr(k, "key", k)) for k in path), tuple(jnp.shape(leaf)))
        for path, leaf in flat
    ]


# --------------------------------------------------------------------------- #
# checkpointing
#
# One binary file holding the whole NNX state (parameters *and* buffers) via
# `flax.serialization` + msgpack.
# --------------------------------------------------------------------------- #
def _as_numpy(tree):
    import numpy as np

    return jax.tree.map(lambda x: np.asarray(x), tree)


def save_model(model, path) -> str:
    """Serialise the full NNX state (params *and* buffers) to ``path``.

    Returns the path as a string.  ``.msgpack`` is appended when the path has no
    suffix.
    """
    from pathlib import Path

    from flax import nnx, serialization

    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".msgpack")
    # `to_pure_dict` drops the Variable wrappers, leaving a plain dict of arrays
    # that msgpack can serialise.
    state = nnx.to_pure_dict(nnx.state(model))
    path.write_bytes(serialization.msgpack_serialize(_as_numpy(state)))
    return str(path)


def load_model(model, path):
    """Load a checkpoint written by :func:`save_model` into ``model`` (in place).

    ``model`` must have been constructed with the same architecture and is
    overwritten in place by the file's values; the module is also returned for
    convenience.
    """
    from pathlib import Path

    from flax import nnx, serialization

    path = Path(path)
    if not path.exists() and not path.suffix:
        path = path.with_suffix(".msgpack")
    restored = serialization.from_state_dict(
        nnx.state(model), serialization.msgpack_restore(path.read_bytes())
    )
    nnx.update(model, restored)
    return model


class ModelIOMixin:
    """Mixin adding a one-call ``save``/``load`` checkpointing API."""

    def save(self, path) -> str:
        """Save this module's state to ``path`` (``.msgpack`` appended if absent)."""
        return save_model(self, path)

    def load(self, path):
        """Load a checkpoint into this module and return it."""
        return load_model(self, path)
