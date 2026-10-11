# Rust ↔ Python parity notes

Reference implementation: [`yosh95/ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs)
(crate `ssm-latent-model`, Burn 0.21). This document is the porting log: what matches,
what is intentionally different, and what was deliberately left out.

## Module map

| Rust | Python | Notes |
|---|---|---|
| `src/ssm.rs::SsmConfig` | `config.SsmConfig` | validation rules and error strings match 1:1 |
| `src/ssm.rs::MultiScaleSsmConfig` | `config.MultiScaleSsmConfig` | adds `timescale_range()` helper |
| `src/ssm.rs::SsmBlock` | `ssm.SsmBlock` | `forward` ≡ parallel scan, `forward_step` ≡ `step` |
| `src/ssm.rs::parallel_scan` | `ssm.parallel_scan` | same doubling (offset *= 2) schedule; `mode="sequential"` added as a slow reference |
| `src/ssm.rs::rotate_state` | `SsmBlock.step` (inline) | complex rotation on the hidden state |
| `src/latent.rs::LatentPredictor` | `latent.LatentPredictor` | `stability_projections` is a frozen `Buffer`, as in Rust |
| `src/latent.rs::MultiScaleLatentPredictor` | `latent.MultiScaleLatentPredictor` | `action_dim=0` ⇒ fusion is `d_model → d_model` |
| `src/latent.rs::{stability_loss, sigreg_loss, lejepa_loss, curvature_loss*}` | `latent.*` | same formulas, including `stability_loss_running` |
| `src/latent.rs` monitoring block | `monitor.*` | `compute_identifiability_r2`, `procrustes_alignment`, `gennorm_identifiability_score`, `linear_latent_plan`, `plan_path_cost`, `compute_exploration_quality`, `check_stationarity`, `check_planning_consistency`, `health_check` |
| `src/predictor.rs::MambaPredictor` | `predictor.MambaPredictor` | `loss()` compares `pred[:, :-1]` with `target[:, 1:]` |
| `src/predictor.rs::MultiScaleMambaPredictor` | `predictor.MultiScaleMambaPredictor` | `forward`, `forward_with_action`, `step_imagine`, `step_imagine_h`, `forward_with_h_sampling` |
| `src/multimodal.rs` | `multimodal.*` | channels-first public API, channels-last internally (flux conv convention) |
| `src/meta/tasks.rs` | `meta.tasks` | `CircleWorldTask/Distribution`, `SineWaveTask/Distribution`, `CompositeDistribution` |
| `src/meta/adapt.rs` | `meta.adapt` | `inner_loop`, `adapt`, `AdaptationStrategy`, `LossWeights` |
| `src/meta/eval.rs` | `meta.eval` | `AdaptationSpeedReport`, `evaluate_adaptation_speed` |
| `src/meta/train.rs` | `meta.train` | `ReptileMetaLearner`, `run_meta_pipeline` |
| `src/preprocess.rs` | *not ported* | ONNX/tokenizer text embedding is out of scope for the numerical core |
| `tiny-stories-jepa-demo` | *not ported* | language modelling is out of scope |
| `circle-world-demo` (terminal animation) | `examples/03_circle_world_planning.py` | same task, quantitative output instead of ASCII frames |
| `game-playing-wasm` | *not ported* | WASM is a strength of the Rust build, not of this one |

## Intentionally reproduced quirks

These look like bugs but are part of the reference semantics; changing them would
invalidate the comparison.

1. **Head-wise shared `theta`.** In `dt_proj`/`lambda_proj`/`theta_proj` (`theta_proj`
   outputs `n_heads * d_state/2`), the Rust code reinterprets the `[B, T, n_heads]`
   activation as `[B, T, n_heads, 1, 1]`. The Python port keeps `theta` head-wise
   (`[B, T, H, d_state/2]`) so the rotation angle stays head-local.
2. **`step_imagine` uses LeakyReLU**, not `output_proj`, to produce the observation
   that closes the d_model loop — i.e. the emitted prediction is not the decoder output.
3. **Multi-scale residual scaling.** `out = clip(y * (1/n_layers) + residual, ±100)`.
   Layer 0 owns the only causal convolution; the slow layers rely on the residual path.
4. **`step` uses `a_im` only.** The state decay uses the *real* exponential while the
   imaginary part is folded into the rotation angle; `a_re` is also used to scale the
   `beta` (previous-input) term.
5. **`stability_loss` normalises projections as `w / (‖w‖ + 1e-6)`**, adding the epsilon
   *before* the division, whereas a defensive implementation would clamp the norm.
6. **`gennorm_identifiability_score`** uses the Rust branch structure for the target
   kurtosis, including its `alpha <= 0.5` branch, which is unreachable.
7. **`curvature_loss_with_dt`** clamps `dt` from below with `1e-6` rather than raising.

## Deliberate differences

| Difference | Rationale |
|---|---|
| **RNG.** JAX `jax.random` (keyed) instead of `rand::StdRng`. | Determinism per key rather than per seed; test fixtures pin keys. |
| **Parameter init.** flax initialisers (LeCun normal for `Linear`), then the same *uniform overrides* for `a_re ∈ [-1, -0.1]` / `a_im ∈ [0, 2π]` and per-layer `a_re` ranges. | Exact bit-parity with Burn's `Tensor::random` is not achievable across frameworks; distributional parity is preserved. |
| **`theta_proj` gets a bias**, `dt_proj` does not (its bias is the `-3.1` constant). | Matches the Rust construction, where `dt_proj.bias` is *overwritten* after `LinearConfig::init`. |
| **Checkpointing.** `flax.serialization` + msgpack via `save()`/`load()` instead of Burn's `BinFileRecorder`. | Same ergonomics (one binary file, whole state incl. buffers), different format. |
| **Reptile outer loop.** `θ ← θ + β(θ′ − θ)` instead of the Rust `interpolate_weights`, which ignores its arguments and returns the adapted weights. | The Rust helper is a stub (`// Simplified: return adapted weights directly`). `outer_lr=1.0` reproduces the Rust behaviour exactly, so both extremes are testable. |
| **`adaptation_curve` evaluates on the query set at every step.** | The SAI metric is about generalisation speed; the Rust `inner_loop` curve is the support loss. |
| **No `Dropout` in `MlpEncoder`.** | Burn's dropout is a no-op at inference and would break test determinism; the parameter exists for API parity. |
| **`MultiScaleMambaPredictor.step_imagine` validates its input width.** | It is fed a `d_model`-wide SSM output, not the decoded prediction; the check turns a silent numerical drift into an exception. |

## Not yet ported

- `ort` feature (ONNX text embeddings) — `src/preprocess.rs`.
- MAML as an alternative outer loop (listed as "in progress" in the Rust roadmap too).
- The WASM/web demos.
