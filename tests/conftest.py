"""Shared fixtures and helpers for the test suite."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Make ``src/`` importable without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssm_latent import MultiScaleSsmConfig, SsmConfig  # noqa: E402


@pytest.fixture(scope="session")
def key():
    """A fixed PRNG key so every test is deterministic."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def small_config():
    """Small single-block config (no convolution)."""
    return SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=False)


@pytest.fixture
def small_config_conv():
    """Small single-block config with a causal convolution."""
    return SsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True, conv_kernel=4
    )


@pytest.fixture
def small_multiscale_config():
    """Three-layer multi-timescale config."""
    return MultiScaleSsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=3, use_conv=True
    )


def assert_close(a, b, atol=1e-5, rtol=1e-5, msg=""):
    """Assert two arrays are numerically close, with a helpful message."""
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    assert a.shape == b.shape, f"{msg} shape mismatch: {a.shape} vs {b.shape}"
    max_diff = float(jnp.max(jnp.abs(a - b)))
    assert jnp.allclose(a, b, atol=atol, rtol=rtol), f"{msg} max|diff|={max_diff} (atol={atol})"
    return max_diff


def random_sequence(key, batch, seq_len, d_model, dtype=jnp.float32):
    return jax.random.normal(key, (batch, seq_len, d_model), dtype)
