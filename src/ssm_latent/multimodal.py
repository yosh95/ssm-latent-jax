"""Multimodal (vision + sensor + action) latent predictor.

Port of `ssm_latent_model::multimodal`.  Images are encoded by a small CNN,
sensor and action streams by linear layers; everything is fused and fed to the
SSM, whose output is decoded back into both modalities for the auxiliary
reconstruction losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from ssm_latent.config import SsmConfig
from ssm_latent.latent import N_PROJECTIONS_DEFAULT, Buffer, stability_loss
from ssm_latent.ssm import SsmBlock
from ssm_latent.utils import ModelIOMixin

__all__ = [
    "VisionEncoder",
    "VisionDecoder",
    "MultimodalLossInput",
    "MultimodalLatentPredictor",
]


class VisionEncoder(nnx.Module):
    """``[B, 16, 16, C] -> [B, d_model]`` via two strided convolutions (channels-last)."""

    def __init__(
        self, channels: int, d_model: int, rngs: nnx.Rngs | None = None, *, dtype=jnp.float32
    ):
        rngs = rngs or nnx.Rngs(0)
        self.conv1 = nnx.Conv(
            channels,
            16,
            (3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
            param_dtype=dtype,
        )
        self.conv2 = nnx.Conv(
            16, 32, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)), rngs=rngs, param_dtype=dtype
        )
        self.linear = nnx.Linear(32 * 4 * 4, d_model, rngs=rngs, param_dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.conv1(x))  # [B, 8, 8, 16]
        x = jax.nn.relu(self.conv2(x))  # [B, 4, 4, 32]
        return self.linear(x.reshape(x.shape[0], -1))


class VisionDecoder(nnx.Module):
    """``[B, d_model] -> [B, 16, 16, C]`` with transposed convolutions (channels-last)."""

    def __init__(
        self, d_model: int, channels: int, rngs: nnx.Rngs | None = None, *, dtype=jnp.float32
    ):
        rngs = rngs or nnx.Rngs(0)
        self.linear = nnx.Linear(d_model, 32 * 4 * 4, rngs=rngs, param_dtype=dtype)
        # "SAME" gives the 4x4 -> 8x8 -> 16x16 upsampling of the Rust
        # ConvTranspose2d(stride=2, padding=1, padding_out=1).
        self.deconv1 = nnx.ConvTranspose(
            32, 16, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs, param_dtype=dtype
        )
        self.deconv2 = nnx.ConvTranspose(
            16, 8, (3, 3), strides=(2, 2), padding="SAME", rngs=rngs, param_dtype=dtype
        )
        self.final_conv = nnx.Conv(8, channels, (1, 1), rngs=rngs, param_dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear(x).reshape(x.shape[0], 4, 4, 32)  # channels-last
        x = jax.nn.relu(self.deconv1(x))
        x = jax.nn.relu(self.deconv2(x))
        return self.final_conv(x)


@dataclass
class MultimodalLossInput:
    """Inputs of :meth:`MultimodalLatentPredictor.loss`."""

    z: jax.Array
    pred_z: jax.Array
    recons_img: jax.Array
    orig_img: jax.Array
    recons_sens: jax.Array
    orig_sens: jax.Array
    stability_weight: float = 1.0


class MultimodalLatentPredictor(ModelIOMixin, nnx.Module):
    """Joint vision/sensor/action latent world model.

    Shapes: ``images [B, T, C, H, W]``, ``sensors [B, T, sensor_dim]``,
    ``actions [B, T, action_dim]``.  :meth:`forward` returns
    ``(z_vis, pred_z, decoded_img, decoded_sensor)``.
    """

    def __init__(
        self,
        config: SsmConfig,
        img_channels: int,
        sensor_dim: int,
        action_dim: int,
        rngs: nnx.Rngs | None = None,
        *,
        n_projections: int = N_PROJECTIONS_DEFAULT,
        projection_seed: int = 0,
        dtype=jnp.float32,
    ):
        rngs = rngs or nnx.Rngs(0)
        self.config = config
        self.d_model = config.d_model
        self.img_shape = (img_channels, 16, 16)

        self.vision_encoder = VisionEncoder(img_channels, config.d_model, rngs, dtype=dtype)
        self.sensor_encoder = nnx.Linear(sensor_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.action_encoder = nnx.Linear(action_dim, config.d_model, rngs=rngs, param_dtype=dtype)
        self.fusion = nnx.Linear(3 * config.d_model, config.d_model, rngs=rngs, param_dtype=dtype)
        self.ssm = SsmBlock(config, rngs, dtype=dtype)
        self.vision_decoder = VisionDecoder(config.d_model, img_channels, rngs, dtype=dtype)
        self.sensor_decoder = nnx.Linear(config.d_model, sensor_dim, rngs=rngs, param_dtype=dtype)

        from ssm_latent.latent import make_projections

        self.stability_projections = Buffer(
            jnp.asarray(make_projections(config.d_model, n_projections, seed=projection_seed))
        )

    def __call__(
        self,
        images: jax.Array,
        sensors: jax.Array,
        actions: jax.Array,
        *,
        scan_mode: str = "parallel",
    ):
        return self.forward(images, sensors, actions, scan_mode=scan_mode)

    def forward(
        self,
        images: jax.Array,
        sensors: jax.Array,
        actions: jax.Array,
        *,
        scan_mode: str = "parallel",
    ):
        """Images are given channels-first (``[B, T, C, H, W]``, as in the Rust API).

        They are transposed to the channels-last layout expected by the flax
        convolutions internally, and the decoded images are transposed back.
        """
        batch, seq_len = images.shape[:2]
        channels, height, width = self.img_shape
        img_flat = jnp.transpose(
            images.reshape(batch * seq_len, channels, height, width), (0, 2, 3, 1)
        )
        z_vis = self.vision_encoder(img_flat).reshape(batch, seq_len, self.d_model)
        z_sens = self.sensor_encoder(sensors)
        z_act = self.action_encoder(actions)

        u = self.fusion(jnp.concatenate([z_vis, z_sens, z_act], axis=-1))
        pred_z = self.ssm(u, scan_mode=scan_mode)

        decoded = self.vision_decoder(pred_z.reshape(batch * seq_len, self.d_model))
        decoded_img = jnp.transpose(decoded, (0, 3, 1, 2)).reshape(
            batch, seq_len, channels, height, width
        )
        decoded_sens = self.sensor_decoder(pred_z)
        return z_vis, pred_z, decoded_img, decoded_sens

    def loss(self, args: MultimodalLossInput) -> jax.Array:
        seq_len = args.z.shape[1]
        mse_latent = (
            (jax.lax.stop_gradient(args.z[:, 1:]) - args.pred_z[:, : seq_len - 1]) ** 2
        ).mean()
        mse_img = ((args.orig_img - args.recons_img) ** 2).mean()
        mse_sens = ((args.orig_sens - args.recons_sens) ** 2).mean()
        reg = stability_loss(args.z, jax.lax.stop_gradient(self.stability_projections[...]))
        return mse_latent + mse_img + mse_sens + args.stability_weight * reg
