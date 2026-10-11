"""Shared helpers for the example scripts.

Every example is runnable directly::

    python examples/02_latent_predictor.py

They insert ``src/`` into ``sys.path`` so that the package works from a fresh
checkout without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

if jax.devices()[0].platform != "cpu":
    jax.config.update("jax_platform_name", "cpu")


def banner(title: str) -> None:
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def mse(pred, target) -> float:
    return float(jnp.mean((jnp.asarray(pred) - jnp.asarray(target)) ** 2))


def rmse(pred, target) -> float:
    return float(jnp.sqrt(mse(pred, target)))
