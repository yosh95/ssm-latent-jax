"""Example 6 - streaming (O(1)) anomaly detection with a pure-Mamba predictor.

The JEPA machinery is deliberately dropped here: for univariate sensor streams a
small ``MambaPredictor`` in observation space is enough, and its constant-time
``step`` makes it usable inside a control loop.  The example trains on a clean
signal, then flags injected anomalies with a robust (MAD-based) threshold on the
prediction residual, in streaming mode.

Run: ``python examples/06_anomaly_detection_streaming.py``
"""

import jax
import jax.numpy as jnp
import optax
from common import banner, section
from flax import nnx

from ssm_latent.config import SsmConfig
from ssm_latent.data import sine_wave
from ssm_latent.predictor import MambaPredictor
from ssm_latent.utils import count_parameters

SEQ = 64
BATCH = 16


def make_signal(seed: int, noise: float = 0.002):
    return sine_wave(BATCH, SEQ, freq=0.8, amplitude=1.0, noise=noise, seed=seed)


def main() -> None:
    banner("Streaming anomaly detection with a pure-Mamba world model")

    cfg = SsmConfig(d_model=32, d_state=8, expand=2, n_heads=4, mimo_rank=1, use_conv=True)
    model = MambaPredictor(cfg, input_dim=1, output_dim=1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)
    print(f"parameters: {count_parameters(model)}")

    train = make_signal(0)

    @nnx.jit
    def train_step(model, optimizer, x):
        def loss_fn(model):
            return MambaPredictor.loss(model(x), x)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    section("training on the clean signal")
    for step in range(1, 801):
        loss = float(train_step(model, optimizer, train))
        if step % 100 == 0:
            print(f"  step {step:4d}  one-step MSE = {loss:.6e}")

    section("calibrating the residual threshold on held-out clean data")
    clean = make_signal(1)
    clean_residual = _residuals(model, clean)
    median = float(jnp.median(clean_residual))
    mad = float(jnp.median(jnp.abs(clean_residual - median))) + 1e-12

    # pick the operating point by a target false-positive budget, which is the
    # honest way to choose a threshold: 1% of the *clean* residuals may fire.
    target_fp = 0.01
    threshold = float(jnp.quantile(clean_residual, 1.0 - target_fp))
    print(f"  median residual         = {median:.3e}")
    print(f"  scaled MAD (1.4826*MAD) = {1.4826 * mad:.3e}")
    print(f"  threshold (99th pct.)   = {threshold:.3e}")
    print(f"  MAD-equivalent k        = {(threshold - median) / (1.4826 * mad):.2f}")
    fp_rate_clean = float(jnp.mean(clean_residual > threshold))
    print(f"  false-positive rate on held-out clean data = {fp_rate_clean:.2%}")

    section("streaming detection")
    # inject three anomalies (a spike, a level shift, a dropout) into a fresh signal
    x = make_signal(2)
    x = x.at[:, 20, 0].add(1.0)  # spike
    x = x.at[:, 35:40, 0].add(0.5)  # level shift
    x = x.at[:, 55, 0].set(0.0)  # dropout

    anomalies = _detect(model, x, threshold)
    injected = (
        jnp.zeros((SEQ,), dtype=bool)
        .at[20]
        .set(True)
        .at[jnp.arange(35, 40)]
        .set(True)
        .at[55]
        .set(True)
    )

    # `flags[t]` says "the residual at step t is anomalous"; a positive is
    # credited to the injected window it falls in, so allow the two steps of
    # slack the recurrent model needs to react to a step change.
    window = jnp.zeros((SEQ,), dtype=bool)
    for t in (20, 35, 36, 37, 38, 39, 55):
        window = window.at[max(t - 1, 0) : t + 3].set(True)
    tp = int(jnp.sum(anomalies & window))
    fp = int(jnp.sum(anomalies & ~window))
    fn = int(jnp.sum(~anomalies & injected))
    print("  injected anomalies     : t=20 (spike), t=35..39 (level shift), t=55 (dropout)")
    print(f"  true positives         : {tp}   (any flag within 1 step of an anomaly)")
    print(f"  missed anomalies       : {fn}")
    print(f"  spurious flags         : {fp}")
    print(f"  anomaly recall         : {tp / max(tp + fn, 1):.2%}")
    print(f"  flags per stream       : {float(jnp.mean(jnp.sum(anomalies, axis=1))):.2f}")
    first = [int(t) for t in range(SEQ) if bool(anomalies[0, t])]
    print(f"  stream 0 flagged steps : {first}")

    section("streaming cost")
    state = model.zero_state(batch=1)
    pred, state = model.step(x[:1, 0], state)
    print(f"  one step -> prediction {tuple(pred.shape)}, state h{tuple(state.h.shape)}")
    print("  cost per step is O(1) in the sequence length - no re-processing of history")


def _residuals(model, x):
    """|x[t+1] - pred[t]| for every stream and step."""
    pred = model(x)
    return jnp.abs(x[:, 1:] - pred[:, :-1])


def _detect(model, x, threshold):
    """Streaming detection using only the recurrent state (no look-ahead)."""
    batch, seq_len, _ = x.shape
    state = model.zero_state(batch)
    flags = jnp.zeros((batch, seq_len), dtype=bool)
    prev_pred = None
    for t in range(seq_len):
        pred, state = model.step(x[:, t], state)
        if prev_pred is not None:
            flags = flags.at[:, t].set(jnp.abs(x[:, t, 0] - prev_pred[:, 0]) > threshold)
        prev_pred = pred
    return flags


if __name__ == "__main__":
    main()
