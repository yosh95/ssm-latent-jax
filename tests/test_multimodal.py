"""Vision + sensor + action fusion (`src/multimodal.rs`)."""

import jax
import jax.numpy as jnp
from conftest import assert_close
from flax import nnx

from ssm_latent import SsmConfig
from ssm_latent.multimodal import (
    MultimodalLatentPredictor,
    MultimodalLossInput,
    VisionDecoder,
    VisionEncoder,
)


def _cfg():
    return SsmConfig(d_model=8, d_state=4, expand=2, n_heads=2, mimo_rank=1, use_conv=True)


def test_vision_encoder_shape_contract():
    enc = VisionEncoder(channels=1, d_model=8, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 16, 16, 1))
    y = enc(x)
    assert y.shape == (3, 8)
    assert bool(jnp.all(jnp.isfinite(y)))


def test_vision_decoder_inverts_the_encoder_spatial_shape():
    dec = VisionDecoder(d_model=8, channels=1, rngs=nnx.Rngs(0))
    z = jax.random.normal(jax.random.PRNGKey(0), (3, 8))
    img = dec(z)
    assert img.shape == (3, 16, 16, 1)


def test_vision_encoder_decoder_roundtrip_is_finite_and_non_trivial():
    enc = VisionEncoder(1, 8, nnx.Rngs(0))
    dec = VisionDecoder(8, 1, nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 16, 16, 1))
    recon = dec(enc(x))
    assert recon.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(recon)))
    assert float(jnp.max(jnp.abs(recon))) > 0.0


def test_multimodal_forward_shapes_channels_first_api():
    model = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(0))
    images = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 1, 16, 16))
    sensors = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 3))
    actions = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 2))
    z_vis, pred_z, dec_img, dec_sens = model(images, sensors, actions)
    assert z_vis.shape == (2, 4, 8)
    assert pred_z.shape == (2, 4, 8)
    assert dec_img.shape == (2, 4, 1, 16, 16)
    assert dec_sens.shape == (2, 4, 3)


def test_multimodal_loss_is_finite_non_negative_and_differentiable():
    model = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(0))
    images = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 1, 16, 16))
    sensors = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 3))
    actions = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 2))

    def loss_fn(model):
        z, pred_z, dec_img, dec_sens = model(images, sensors, actions)
        return model.loss(MultimodalLossInput(z, pred_z, dec_img, images, dec_sens, sensors))

    loss = loss_fn(model)
    assert jnp.isfinite(loss) and float(loss) >= 0.0
    grads = nnx.grad(loss_fn)(model)
    leaves = jax.tree.leaves(nnx.state(grads, nnx.Param))
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in leaves)


def test_multimodal_perfect_reconstructions_give_zero_reconstruction_error():
    model = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(0))
    z = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 8))
    pred_z = z
    images = jnp.zeros((1, 3, 1, 16, 16))
    sensors = jnp.zeros((1, 3, 3))
    loss = model.loss(
        MultimodalLossInput(z, pred_z, images, images, sensors, sensors, stability_weight=0.0)
    )
    seq_len = 3
    expected_latent = float(((z[:, 1:] - pred_z[:, : seq_len - 1]) ** 2).mean())
    assert abs(float(loss) - expected_latent) < 1e-6


def test_multimodal_modalities_all_influence_the_latent():
    model = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(0))
    images = jnp.zeros((1, 2, 1, 16, 16))
    sensors = jnp.zeros((1, 2, 3))
    actions = jnp.zeros((1, 2, 2))
    base = model(images, sensors, actions)[1]
    assert not jnp.allclose(base, model(images + 1.0, sensors, actions)[1])
    assert not jnp.allclose(base, model(images, sensors + 1.0, actions)[1])
    assert not jnp.allclose(base, model(images, sensors, actions + 1.0)[1])


def test_multimodal_save_load(tmp_path):
    model = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(0))
    images = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 1, 16, 16))
    sensors = jnp.zeros((1, 2, 3))
    actions = jnp.zeros((1, 2, 2))
    expected = model(images, sensors, actions)[1]
    path = model.save(tmp_path / "multimodal")
    other = MultimodalLatentPredictor(_cfg(), 1, 3, 2, rngs=nnx.Rngs(4)).load(path)
    assert_close(other(images, sensors, actions)[1], expected, atol=1e-6)
