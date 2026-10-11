"""Configuration validation rules and error messages."""

import pytest

from ssm_latent import MultiScaleSsmConfig, SsmConfig
from ssm_latent.config import ConfigError


def test_derived_dimensions():
    cfg = SsmConfig(d_model=16, d_state=8, expand=2, n_heads=4, mimo_rank=2)
    assert cfg.d_inner == 32
    assert cfg.d_head == 8
    assert cfg.d_head_mimo == 4
    assert cfg.h_flat_dim == 4 * 8 * 4


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(d_model=0, d_state=4, expand=2, n_heads=1, mimo_rank=1), "d_model"),
        (dict(d_model=8, d_state=0, expand=2, n_heads=1, mimo_rank=1), "d_state"),
        (dict(d_model=8, d_state=4, expand=0, n_heads=1, mimo_rank=1), "expand"),
        (dict(d_model=8, d_state=4, expand=2, n_heads=0, mimo_rank=1), "n_heads"),
        (dict(d_model=8, d_state=4, expand=2, n_heads=1, mimo_rank=0), "mimo_rank"),
        (dict(d_model=8, d_state=7, expand=2, n_heads=2, mimo_rank=1), "must be even"),
        (dict(d_model=8, d_state=4, expand=2, n_heads=7, mimo_rank=1), "divisible by n_heads"),
        (dict(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=3), "divisible by mimo_rank"),
    ],
)
def test_invalid_configs_raise(kwargs, expected):
    with pytest.raises(ConfigError, match=expected):
        SsmConfig(**kwargs).validate()


def test_valid_config_passes():
    cfg = SsmConfig(d_model=16, d_state=8, expand=2, n_heads=2, mimo_rank=1)
    assert cfg.validate() is cfg


def test_conv_kernel_validation():
    SsmConfig(
        d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=False, conv_kernel=0
    ).validate()
    with pytest.raises(ConfigError, match="conv_kernel"):
        SsmConfig(
            d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True, conv_kernel=0
        ).validate()


def test_with_returns_a_copy():
    cfg = SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1)
    other = cfg.with_(use_conv=True, conv_kernel=3)
    assert cfg.use_conv is False and other.use_conv is True
    assert other.conv_kernel == 3


def test_multiscale_requires_layers():
    with pytest.raises(ConfigError, match="n_layers"):
        MultiScaleSsmConfig(
            d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=0
        ).validate()


def test_multiscale_timescale_ranges():
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=4)
    assert cfg.timescale_range(0) == (-1.0, -0.3)
    assert cfg.timescale_range(1) == (-0.3, -0.05)
    assert cfg.timescale_range(2) == (-0.05, -0.005)
    # every layer beyond the first two is "slow"
    assert cfg.timescale_range(3) == cfg.timescale_range(2)


def test_to_dict_roundtrip():
    cfg = MultiScaleSsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, n_layers=2)
    d = cfg.to_dict()
    assert d["n_layers"] == 2 and d["d_model"] == 8
