# ssm-latent-jax ⚡

### Mamba-3 × JEPA latent world models in JAX / Flax NNX

[![CI](https://github.com/yosh95/ssm-latent-jax/actions/workflows/ci.yml/badge.svg)](https://github.com/yosh95/ssm-latent-jax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11%E2%80%933.14](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue.svg)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-%2BFlax%20NNX-9cf.svg)](https://github.com/google/jax)

A **Python-first research toolkit** for JEPA latent world models, built on **JAX / Flax
NNX**: Mamba-3 selective state-space dynamics fused with a Joint-Embedding Predictive
Architecture, the LeJEPA collapse-prevention objective, and the identifiability monitors
and meta-learning (SAI) machinery that go with them.

> **Python is the primary implementation.** This package is where the algorithms are
> developed and studied. A Rust / Burn port ([`ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs))
> and a small PyTorch port ([`ssm-latent-py`](https://github.com/yosh95/ssm-latent-py))
> exist for *deployment* and *cross-language comparison*; they share this codebase's
> numerical conventions so results stay comparable, but they follow this one.

| Language | Repository | Role |
|---|---|---|
| ⚡ **JAX (this repo)** | `ssm-latent-jax` | **primary** — research, prototyping, verification |
| 🦀 Rust (Burn) | [`ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs) | deployment port; includes WASM demos |
| 🔥 PyTorch | [`ssm-latent-py`](https://github.com/yosh95/ssm-latent-py) | minimal PyTorch re-implementation of the SSM + latent predictors |

> **Why JAX / Python?** JAX gives a functional, composable core (`jit` / `grad` / `vmap` /
> `lax.scan`) and XLA compilation, and the whole codebase is explorable in a notebook,
> hackable in one line, and verifiable with `pytest`. The algorithms live here; the other
> languages consume them. Shared numerical semantics are documented in
> [`docs/DESIGN.md`](docs/DESIGN.md).

---

## 📦 What is inside

| Module | What it does |
|---|---|
| `ssm_latent.ssm` | Mamba-3 selective SSM: exponential-trapezoidal recurrence, complex-valued rotation, MIMO, BCNorm, causal conv; parallel scan **and** `O(1)` streaming step |
| `ssm_latent.latent` | JEPA latent predictors, recon/stability/curvature losses, SIGReg (LeJEPA) objective, MLP encoder |
| `ssm_latent.monitor` | LeJEPA identifiability monitors: Theorems 1–4 (Gaussianity, stationarity, isotropic exploration, planning consistency) |
| `ssm_latent.predictor` | Pure-Mamba predictors, closed-loop "imagination" (`d_model` and `h`-based), scheduled h-sampling training |
| `ssm_latent.multimodal` | Vision (CNN) + sensor + action fusion |
| `ssm_latent.meta` | Task distributions, adaptation strategies, Reptile meta-learning, adaptation-speed evaluation (the SAI core metric) |
| `ssm_latent.data` | CircleWorld, SineWave, damped oscillator, Lissajous, chaotic circle, tiny-stories corpus |

---

## 🚀 Install

```bash
git clone https://github.com/yosh95/ssm-latent-jax
cd ssm-latent-jax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CPU JAX; use ".[cuda]" for CUDA 12
# requires Python >= 3.11, < 3.15 (flax >= 0.11 for `nnx.List`)
```

Using [`uv`](https://docs.astral.sh/uv/) (which reads `.python-version`, so the
interpreter is picked for you):

```bash
uv venv && uv pip install -e ".[dev]"     # or: uv sync, if you keep a lockfile
```

> **Python version.** `requires-python` is `>=3.11,<3.15`. `flax` pulls in
> `tensorstore`, which publishes wheels up to **cp314** only; on Python 3.15+ uv
> falls back to a Bazel source build of `tensorstore` that fails. Pin 3.11–3.14
> (`.python-version` says 3.14).

or, without installing:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## ⚡ Quickstart

```python
import jax.numpy as jnp
import optax
from flax import nnx

from ssm_latent import MultiScaleSsmConfig, MultiScaleLatentPredictor
from ssm_latent.data import circle_world, circle_world_actions
from ssm_latent.latent import LatentLossArgs

# multi-timescale SSM × JEPA world model
cfg = MultiScaleSsmConfig(d_model=32, d_state=8, expand=2, n_heads=4,
                          mimo_rank=1, n_layers=3, use_conv=True)
model = MultiScaleLatentPredictor(cfg, input_dim=2, action_dim=2, rngs=nnx.Rngs(0))

obs = circle_world(batch_size=8, seq_len=32, omega=1.5)         # [B, T, 2]
act = circle_world_actions(batch_size=8, seq_len=32, omega=1.5)

def loss_fn(model, obs, act):
    z, pred_z, recon_x, pred_x = model(obs, act)
    return model.loss(LatentLossArgs(z, pred_z, recon_x, pred_x, obs))

optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

@nnx.jit
def train_step(model, optimizer, obs, act):
    loss, grads = nnx.value_and_grad(loss_fn)(model, obs, act)
    optimizer.update(model, grads)
    return loss

for _ in range(500):
    loss = train_step(model, optimizer, obs, act)

# O(1) streaming inference
state = model.zero_state(8)
z = model.encode(obs)
pred_z, state = model.step(z[:, -1], act[:, -1], state)
```

---

## 🔁 Parallel scan ≡ O(1) streaming step

The central invariant — parallel scan ≡ streaming step — is preserved and tested:

```python
y_parallel = block.forward(x, scan_mode="parallel")    # O(log T) depth
y_scan     = block.forward(x, scan_mode="sequential")  # naive scan
y_step     = block.scan_sequential(x)                  # one step at a time
assert jnp.allclose(y_parallel, y_step, atol=1e-5)      # ~1e-7 in practice
```

---

## 🧪 Tests

```bash
pytest -q                 # 189 tests, ~3 min on 16 CPU cores
pytest -q -m "not slow"   # skip the longer meta-learning check
```

| Suite | Tests | What it guarantees |
|---|---:|---|
| `test_config.py` | 15 | every `SsmConfig` validation rule and its error message |
| `test_ssm.py` | 34 | parallel ≡ step ≡ `lax.scan`; conv causality; state shapes; damping/rotation init; JIT + grad + vmap composition |
| `test_latent.py` | 34 | loss decomposition, `stop_gradient` placement, VICReg/SIGReg properties, frozen projection buffers, checkpoint round-trip |
| `test_monitor.py` | 29 | Theorems 1–4: rotation invariance, Procrustes, kurtosis, exploration risk levels, planning consistency |
| `test_predictor.py` | 27 | streaming state advance, closed-loop imagination, h-sampling determinism, parameter counts, save/load |
| `test_meta.py` | 29 | task shapes/distributions, strategy masks move **only** the intended parameters, Reptile interpolation, adaptation-speed metrics |
| `test_multimodal.py` | 8 | vision encoder/decoder shape contract, modality ablation, loss decomposition |
| `test_data.py` | 14 | analytic derivatives of the generators, unit-circle invariants, tokenizer round-trips |

---

## 📊 Measured results

Reproduce with `python examples/04_meta_learning.py` and `examples/03_circle_world_planning.py`:

**Adaptation speed (SAI core metric), CircleWorld, ~40K parameters**

| Holdout task | before meta-training | after Reptile |
|---|---|---|
| ω = 0.3 (below training range) | never reaches threshold, AUC 1.205 | **5 steps**, AUC 0.035 |
| ω = 2.5 (above training range) | never, AUC 1.159 | **5 steps**, AUC 0.033 |
| ω = 1.2, 2.5× noise | never, AUC 1.248 | **1 step**, AUC 0.023 |

**Prediction (unseen phase)**

| Model | 1-step RMSE |
|---|---|
| Mamba-only, observation space | 0.0098 |
| JEPA, single timescale | 0.0062 |
| JEPA, multi-scale (3 layers) | **0.0039** |

---

## 🎬 Examples

| Script | Topic |
|---|---|
| `examples/01_ssm_basics.py` | SSM internals: scan equivalence, state layout, gradient reach, multi-timescale init |
| `examples/02_latent_predictor.py` | train a JEPA world model on a damped oscillator + LeJEPA health check |
| `examples/03_circle_world_planning.py` | the CircleWorld benchmark and latent planning (Theorem 4) |
| `examples/04_meta_learning.py` | Reptile meta-learning and adaptation-speed evaluation |
| `examples/05_identifiability_monitors.py` | Theorems 1–4 on healthy vs. deliberately broken latents |
| `examples/06_anomaly_detection_streaming.py` | streaming anomaly detection with `O(1)` steps and MAD calibration |

---

## 🧬 Background

The design follows the LeJEPA line of work and the SAI thesis:

- **Mamba-3** (Lahoti et al.): exponential-trapezoidal discretisation, complex-valued
  state transitions with data-dependent RoPE, MIMO formulation, BCNorm.
- **JEPA** (LeCun): prediction happens in representation space; a decoder exists only
  as an auxiliary reconstruction path.
- **SIGReg / LeJEPA** (Balestriero & LeCun, 2025): collapse prevention by matching the
  empirical characteristic function of the embeddings to that of `N(0, I)` — one
  hyper-parameter, no stop-gradient, no teacher-student, no EMA schedule.
- **Identifiability theory** (Klindt, LeCun & Balestriero, 2026): the four conditions
  implemented in `ssm_latent.monitor`.
- **SAI** (Goldfeder, Wyder, LeCun & Shwartz-Ziv, 2026): the metric that matters is
  *adaptation speed*, which is what `ssm_latent.meta.eval` measures.

---

## 🧭 Design notes & shared semantics

`docs/DESIGN.md` documents the module layout, the numerical conventions that are shared
with the sibling Rust / PyTorch ports, and where this implementation deliberately diverges
(framework idioms and a few correctness choices).  Notable conventions include the
head-wise `theta` rotation, the `1/n_layers` residual scaling with a `±100` clamp, and the
LeakyReLU in the closed-loop imagination path.  The Reptile outer loop implements the
general `θ ← θ + β(θ′ − θ)` update (with `outer_lr=1.0` reducing to a hard copy of the
adapted weights).

---

## 📜 License

MIT — see [LICENSE](LICENSE).

## 📖 Citation

```bibtex
@software{ando2026ssmlatentjax,
  title  = {ssm-latent-jax: Mamba-3 x JEPA latent world models in JAX/Flax},
  author = {Ando, Yoshihiro},
  year   = {2026},
  url    = {https://github.com/yosh95/ssm-latent-jax}
}
```

If you build on the Rust or PyTorch ports instead, cite the corresponding repository
([`ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs) /
[`ssm-latent-py`](https://github.com/yosh95/ssm-latent-py)).

The algorithms themselves come from:

```bibtex
@article{balestriero2025lejepa,
  title={LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Balestriero, R. and LeCun, Y.}, journal={arXiv:2511.08544}, year={2025}}

@article{klindt2026identifiability,
  title={When Does LeJEPA Learn a World Model?},
  author={Klindt, D. and LeCun, Y. and Balestriero, R.}, journal={arXiv:2605.26379}, year={2026}}

@article{goldfeder2026sai,
  title={AI Must Embrace Specialization via Superhuman Adaptable Intelligence},
  author={Goldfeder, J. and Wyder, P. and LeCun, Y. and Shwartz-Ziv, R.},
  journal={arXiv:2602.23643v1}, year={2026}}
```

## 💬 Keywords

`jax` `flax` `mamba` `jepa` `ssm` `state-space-model` `world-model` `latent-prediction`
`sigreg` `lejepa` `meta-learning` `reptile` `sai` `robotics` `physical-ai`
