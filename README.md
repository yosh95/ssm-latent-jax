# SSM Latent World Model — Python (JAX) ⚡

### Mamba-3 × JEPA latent world models for research, verification and prototyping

[![CI](https://github.com/yosh95/ssm-latent-py/actions/workflows/ci.yml/badge.svg)](https://github.com/yosh95/ssm-latent-py/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-%2BFlax%20NNX-9cf.svg)](https://github.com/google/jax)

A **JAX / Flax NNX port of [`ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs)** — the
Rust (Burn) implementation of a Mamba-3 selective state-space model fused with a
JEPA (Joint-Embedding Predictive Architecture) latent world model.

> **Why a Python port?** Rust is excellent for deployment, but compile times and
> the size of intermediate artefacts make it a poor fit for *iterating* on research
> ideas. This package keeps the exact numerical semantics of the Rust reference
> (including its quirks) while making the codebase explorable in a notebook, hackable
> in one line, and verifiable with `pytest`.

---

## 📦 What is inside

| Module | What it does | Rust counterpart |
|---|---|---|
| `ssm_latent.ssm` | Mamba-3 selective SSM: exponential-trapezoidal recurrence, complex-valued rotation, MIMO, BCNorm, causal conv; parallel scan **and** `O(1)` streaming step | `src/ssm.rs` |
| `ssm_latent.latent` | JEPA latent predictors, recon/stability/curvature losses, SIGReg (LeJEPA) objective, MLP encoder | `src/latent.rs` |
| `ssm_latent.monitor` | LeJEPA identifiability monitors: Theorems 1–4 (Gaussianity, stationarity, isotropic exploration, planning consistency) | `src/latent.rs` |
| `ssm_latent.predictor` | Pure-Mamba predictors, closed-loop "imagination" (`d_model` and `h`-based), scheduled h-sampling training | `src/predictor.rs` |
| `ssm_latent.multimodal` | Vision (CNN) + sensor + action fusion | `src/multimodal.rs` |
| `ssm_latent.meta` | Task distributions, adaptation strategies, Reptile meta-learning, adaptation-speed evaluation (the SAI core metric) | `src/meta/*` |
| `ssm_latent.data` | CircleWorld, SineWave, damped oscillator, Lissajous, chaotic circle, tiny-stories corpus | `src/meta/tasks.rs`, demos |

---

## 🚀 Install

```bash
git clone https://github.com/yosh95/ssm-latent-py
cd ssm-latent-py
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # CPU JAX; use ".[cuda]" for CUDA 12
```

or, without installing:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

---

## ⚡ Quickstart

```python
import jax, jax.numpy as jnp
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

The central invariant of the reference implementation is preserved and tested:

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
| `test_config.py` | 15 | every validation rule of the Rust `SsmConfig` (parity on error messages) |
| `test_ssm.py` | 34 | parallel ≡ step ≡ `lax.scan`; conv causality; state shapes; damping/rotation init; JIT + grad + vmap composition |
| `test_latent.py` | 34 | loss decomposition, `stop_gradient` placement, VICReg/SIGReg properties, frozen projection buffers, checkpoint round-trip |
| `test_monitor.py` | 28 | Theorems 1–4: rotation invariance, Procrustes, kurtosis, exploration risk levels, planning consistency |
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

## 📄 Porting notes and known differences

This is a faithful port, not a re-design. `docs/PARITY.md` documents the module-by-module
mapping and, importantly, the few places where the Python version deliberately differs
(notably: the Rust `ReptileMetaLearner` collapses the outer-loop interpolation into a hard
copy; here the proper `θ ← θ + β(θ′ − θ)` is implemented, with `outer_lr=1.0` recovering the
Rust behaviour). Documented quirks that are *reproduced*, not fixed, include the head-wise
shared `theta` rotation, the `1/n_layers` residual scaling with a `±100` clamp, and the
LeakyReLU in the closed-loop imagination path.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

## 📖 Citation

```bibtex
@software{ando2026ssmlatentpy,
  title  = {SSM Latent World Model (Python/JAX): Mamba-3 x JEPA},
  author = {Ando, Yoshihiro},
  year   = {2026},
  url    = {https://github.com/yosh95/ssm-latent-py}
}
```

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
