"""Synthetic dataset generators."""

import jax.numpy as jnp
import pytest
from conftest import assert_close

from ssm_latent import data


def test_circle_world_is_on_the_unit_circle():
    obs = data.circle_world(batch_size=3, seq_len=10, omega=1.0)
    assert obs.shape == (3, 10, 2)
    radius = jnp.linalg.norm(obs, axis=-1)
    assert_close(radius, jnp.ones_like(radius), atol=1e-5)


def test_circle_world_actions_are_the_analytic_derivative():
    """`act` is the continuous-time tangent vector of the circle, times `dt`."""
    dt, omega = 0.3, 1.4
    obs = data.circle_world(2, 9, omega=omega, dt=dt)
    act = data.circle_world_actions(2, 9, omega=omega, dt=dt)
    # the angles are exactly recoverable from the observations
    angle = jnp.arctan2(obs[..., 1], obs[..., 0])
    expected = jnp.stack([-omega * dt * jnp.sin(angle), omega * dt * jnp.cos(angle)], axis=-1)
    assert_close(act, expected, atol=1e-5)


def test_circle_world_action_is_a_first_or_second_order_discretisation():
    """The analytic action approximates the discrete step to O(dt^2)."""
    omega = 1.4

    def residual(dt):
        obs = data.circle_world(1, 6, omega=omega, dt=dt)
        act = data.circle_world_actions(1, 6, omega=omega, dt=dt)
        return float(jnp.max(jnp.abs((obs[:, 1:] - obs[:, :-1]) - act[:, :-1])))

    coarse, fine = residual(0.3), residual(0.15)
    assert fine < coarse
    assert 3.0 < coarse / fine < 5.0, f"expected an O(dt^2) residual, got ratio {coarse / fine}"


def test_circle_world_batch_phases_are_distinct():
    obs = data.circle_world(batch_size=4, seq_len=3)
    assert not jnp.allclose(obs[0], obs[1])
    assert_close(obs[0, 0], obs[0, 0], atol=1e-6)


def test_sine_wave_amplitude_and_range():
    obs = data.sine_wave(2, 64, freq=1.0, amplitude=2.5)[..., 0]
    assert float(jnp.max(jnp.abs(obs))) <= 2.5 + 1e-5
    assert float(jnp.max(jnp.abs(obs))) > 2.0


def test_damped_oscillator_decays():
    obs = data.damped_oscillator(1, 64, decay=0.5)
    energy = jnp.linalg.norm(obs, axis=-1)[0]
    assert float(energy[0]) > float(energy[-1])


def test_lissajous_is_bounded_and_two_dimensional():
    obs = data.lissajous(1, 64)
    assert obs.shape == (1, 64, 2)
    assert float(jnp.max(jnp.abs(obs))) <= 1.0 + 1e-5


def test_chaotic_circle_is_on_the_unit_circle():
    obs = data.chaotic_circle(2, 32)
    assert_close(jnp.linalg.norm(obs, axis=-1), jnp.ones((2, 32)), atol=1e-5)


def test_make_demo_data_dispatch_and_errors():
    for kind in ("circle", "sine", "damped", "lissajous", "chaotic"):
        out = data.make_demo_data(kind, batch_size=2, seq_len=6)
        assert out.shape[0] == 2 and out.shape[1] == 6
    with pytest.raises(ValueError, match="unknown dataset"):
        data.make_demo_data("nope")


def test_noise_is_reproducible_and_actually_noisy():
    a = data.circle_world(2, 8, noise=0.1, seed=3)
    b = data.circle_world(2, 8, noise=0.1, seed=3)
    c = data.circle_world(2, 8, noise=0.1, seed=4)
    assert_close(a, b, atol=0.0)
    assert not jnp.allclose(a, c)


# --------------------------------------------------------------------------- #
# tiny stories
# --------------------------------------------------------------------------- #
def test_tiny_stories_vocab_and_ids():
    stories = data.TinyStories(max_seq_len=16)
    assert stories.vocab_size == len(set(data.TINY_STORIES)) + 1
    assert int(stories.ids.min()) >= 1, "0 is reserved for padding"
    assert int(stories.ids.max()) < stories.vocab_size


def test_tiny_stories_batches_are_shifted_by_one():
    stories = data.TinyStories(max_seq_len=12)
    inputs, targets = stories.batch(batch_size=2, seq_len=12)
    assert inputs.shape == targets.shape == (2, 12)
    # targets of every sequence are its inputs shifted left by one
    for b in range(2):
        assert stories.decode(inputs[b])[1:] == stories.decode(targets[b])[:-1]


def test_tiny_stories_decode_roundtrip():
    stories = data.TinyStories()
    assert stories.decode(stories.ids) == stories.text


def test_tiny_stories_batch_stream_is_deterministic():
    stories = data.TinyStories(max_seq_len=8)
    a = list(stories.batches(2, 8, n_batches=3))
    b = list(stories.batches(2, 8, n_batches=3))
    for (ia, ta), (ib, tb) in zip(a, b, strict=True):
        assert_close(ia, ib, atol=0.0)
        assert_close(ta, tb, atol=0.0)
