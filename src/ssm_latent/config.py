"""Configuration objects mirroring the Rust `SsmConfig` / `MultiScaleSsmConfig`.

Validation rules are identical to the reference implementation so that a
configuration accepted (or rejected) in Rust behaves the same way here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["ConfigError", "SsmConfig", "MultiScaleSsmConfig"]


class ConfigError(ValueError):
    """Raised when an SSM configuration violates one of its constraints."""


@dataclass(frozen=True)
class SsmConfig:
    """Configuration of a single selective SSM block (Mamba-3 style).

    Parameters
    ----------
    d_model:
        Model dimension (input and output width).
    d_state:
        State-space dimension. Must be even so that the state can be split
        into ``d_state / 2`` real / imaginary halves (complex rotation).
    expand:
        Inner-dimension multiplier, ``d_inner = d_model * expand``.
    n_heads:
        Number of heads. ``d_inner`` must be divisible by ``n_heads``.
    mimo_rank:
        MIMO rank; ``d_head = d_inner // n_heads`` must be divisible by it.
    use_conv:
        Whether to apply a causal depthwise Conv1d before the selective scan.
    conv_kernel:
        Kernel width of that convolution.
    """

    d_model: int
    d_state: int
    expand: int = 2
    n_heads: int = 1
    mimo_rank: int = 1
    use_conv: bool = False
    conv_kernel: int = 4

    # ----- derived quantities -------------------------------------------------
    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def d_head(self) -> int:
        return self.d_inner // self.n_heads

    @property
    def d_head_mimo(self) -> int:
        return self.d_head // self.mimo_rank

    @property
    def h_flat_dim(self) -> int:
        """Flattened size of the MIMO hidden state, ``H * N * d_head_mimo``."""
        return self.n_heads * self.d_state * self.d_head_mimo

    # ----- validation ---------------------------------------------------------
    def validate(self) -> SsmConfig:
        """Validate the configuration, raising :class:`ConfigError` on failure."""
        if self.mimo_rank == 0:
            raise ConfigError("mimo_rank must be greater than 0")
        if self.n_heads == 0:
            raise ConfigError("n_heads must be greater than 0")
        if self.d_model == 0:
            raise ConfigError("d_model must be greater than 0")
        if self.d_state == 0:
            raise ConfigError("d_state must be greater than 0")
        if self.expand == 0:
            raise ConfigError("expand must be greater than 0")
        if self.d_state % 2 != 0:
            raise ConfigError(f"d_state must be even for complex rotation (got {self.d_state})")
        d_inner = self.d_model * self.expand
        if d_inner % self.n_heads != 0:
            raise ConfigError(f"d_inner ({d_inner}) must be divisible by n_heads ({self.n_heads})")
        d_head = d_inner // self.n_heads
        if d_head % self.mimo_rank != 0:
            raise ConfigError(
                f"d_head ({d_head}) must be divisible by mimo_rank ({self.mimo_rank})"
            )
        if self.use_conv and self.conv_kernel == 0:
            raise ConfigError("conv_kernel must be greater than 0 when use_conv is true")
        return self

    def with_(self, **kwargs) -> SsmConfig:
        """Return a copy with the given fields replaced (builder-style)."""
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "d_state": self.d_state,
            "expand": self.expand,
            "n_heads": self.n_heads,
            "mimo_rank": self.mimo_rank,
            "use_conv": self.use_conv,
            "conv_kernel": self.conv_kernel,
        }


@dataclass(frozen=True)
class MultiScaleSsmConfig(SsmConfig):
    """Configuration of a stack of SSM blocks with different timescales.

    Layer 0 is "fast" (``a_re ~ U(-1.0, -0.3)``), layer 1 "medium"
    (``U(-0.3, -0.05)``) and every subsequent layer "slow"
    (``U(-0.05, -0.005)``).  As in the Rust reference the causal convolution is
    only instantiated on the first layer.
    """

    n_layers: int = 3
    dropout: float = 0.0

    def validate(self) -> MultiScaleSsmConfig:
        if self.n_layers == 0:
            raise ConfigError("n_layers must be > 0")
        SsmConfig(
            d_model=self.d_model,
            d_state=self.d_state,
            expand=self.expand,
            n_heads=self.n_heads,
            mimo_rank=self.mimo_rank,
            use_conv=self.use_conv,
            conv_kernel=self.conv_kernel,
        ).validate()
        return self

    def timescale_range(self, layer_idx: int) -> tuple[float, float]:
        """Uniform ``a_re`` (damping) initialisation range for a given layer."""
        if layer_idx == 0:
            return (-1.0, -0.3)
        if layer_idx == 1:
            return (-0.3, -0.05)
        return (-0.05, -0.005)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"n_layers": self.n_layers, "dropout": self.dropout})
        return d
