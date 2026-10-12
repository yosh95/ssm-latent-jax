# Design notes & shared semantics

`ssm-latent-jax` is the primary reference implementation of these algorithms,
and the source of truth for the shared numerical conventions.  The project began
as a Rust world model ([`yosh95/ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs),
crate `ssm-latent-model`, built on Burn); the research loop moved to Python / JAX
because Rust's compile times and build-artefact size made the edit → run → measure
cycle of algorithm work too slow.  Rust remains a first-class **deployment target**;
a minimal PyTorch sibling exists for cross-language comparison:

- Rust / Burn — [`yosh95/ssm-latent-rs`](https://github.com/yosh95/ssm-latent-rs) (crate `ssm-latent-model`) — deployment target; the original implementation
- PyTorch — [`yosh95/ssm-latent-py`](https://github.com/yosh95/ssm-latent-py) — minimal re-implementation for comparison

All three share the same numerical conventions so results stay comparable.  This
document records those shared semantics: the module map, the conventions that are
load-bearing for numerical agreement, and where this implementation deliberately
diverges (framework idioms, correctness fixes, and out-of-scope features).

## Module map

| Rust / PyTorch | `ssm_latent` | Notes |
|---|---|---|
| `src/ssm.rs::SsmConfig` | `config.SsmConfig` | validation rules and error strings match 1:1 |
| `src/ssm.rs::MultiScaleSsmConfig` | `config.MultiScaleSsmConfig` | adds `timescale_range()` helper |
| `src/ssm.rs::SsmBlock` | `ssm.SsmBlock` | `forward` ≡ parallel scan, `forward_step` ≡ `step` |
| `src/ssm.rs::parallel_scan` | `ssm.parallel_scan` | same doubling (offset *= 2) schedule; `mode="sequential"` added as a slow reference |
| `src/ssm.rs::rotate_state` | `SsmBlock.step` (inline) | complex rotation on the hidden state |
| `src/latent.rs::LatentPredictor` | `latent.LatentPredictor` | `stability_projections` is a frozen `Buffer` (not a trainable `Param`) |
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
| `src/preprocess.rs` (Rust) | *not implemented* | ONNX/tokenizer text embedding is out of scope for the numerical core |
| tiny-stories demo | *dataset only* | `data.TinyStories` provides the tokeniser; language modelling is out of scope |
| `circle-world-demo` (terminal animation) | `examples/03_circle_world_planning.py` | same task, quantitative output instead of ASCII frames |
| `game-playing-wasm` (Rust) | *not implemented* | browser/WASM target is out of scope for this package |

## Shared numerical conventions

These conventions are load-bearing for agreement across the Rust / PyTorch / JAX
implementations.  Some of them look like odd choices; they are documented here so
that a change is a deliberate, reviewable decision rather than an accident.

> **On scan error.**  The parallel scan and the streaming step agree to
> float32 precision in the operating regime (``|exp(dt·a_re)| <= 1``, i.e. damped
> transitions): relative error stays ~1e-7 even at ``T = 4096``.  For *expansive*
> transitions the absolute error grows with the sequence length and the gain — as it
> must, since the sequential reference itself grows — so compare scans relatively,
> not with a fixed absolute tolerance.

1. **Head-wise shared `theta`.** `dt_proj` and `theta_proj` each emit `n_heads`
   values, so the rotation angle is defined per head.  Here `theta` is shaped
   `[B, T, H, d_state/2]` so the rotation stays head-local.
2. **`step_imagine` uses LeakyReLU**, not `output_proj`, to produce the observation
   that closes the d_model loop — i.e. the emitted prediction is not the decoder output.
3. **Multi-scale residual scaling.** `out = clip(y * (1/n_layers) + residual, ±100)`.
   Layer 0 owns the only causal convolution; the slow layers rely on the residual path.
4. **`step` uses `a_im` only.** The state decay uses the *real* exponential while the
   imaginary part is folded into the rotation angle; `a_re` is also used to scale the
   `beta` (previous-input) term.
5. **`stability_loss` normalises projections as `w / (‖w‖ + 1e-6)`**, adding the epsilon
   *before* the division, whereas a defensive implementation would clamp the norm.
6. **`gennorm_identifiability_score`** uses a piecewise-linear interpolation for the
   target kurtosis of a generalised normal; one branch (`alpha <= 0.5` relative to the
   neighbouring range) is unreachable and kept only for structural stability.
7. **`curvature_loss_with_dt`** clamps `dt` from below with `1e-6` rather than raising.

## Deliberate differences (framework & correctness)

| Difference | Rationale |
|---|---|
| **RNG.** JAX `jax.random` (keyed) instead of `rand::StdRng`. | Determinism per key rather than per seed; test fixtures pin keys. |
| **Parameter init.** flax initialisers (LeCun normal for `Linear`), then *uniform overrides* for `a_re ∈ [-1, -0.1]` / `a_im ∈ [0, 2π]` and the per-layer `a_re` ranges. | Bit-for-bit equality with the Rust `Tensor::random` is not achievable across frameworks; the *distributions* are shared. |
| **`theta_proj` gets a bias**, `dt_proj` does not (its bias is the `-3.1` constant). | `dt_proj`'s bias is set to the constant `-3.1` so `softplus(dt) ≈ 0.045` at init; `theta_proj` keeps a learned bias. |
| **Checkpointing.** `flax.serialization` + msgpack via `save()`/`load()`. | One binary file holding the whole state (params *and* buffers); portable across the Python stack. |
| **Reptile outer loop.** `θ ← θ + β(θ′ − θ)` with a configurable `outer_lr`. | The general interpolation; `outer_lr=1.0` reduces to a hard copy of the adapted weights. |
| **`adaptation_curve` evaluates on the query set at every step.** | The SAI metric is about *generalisation* speed, so the curve is the held-out loss, not the support loss. |
| **No `Dropout` in `MlpEncoder`.** | Dropout is a no-op at inference and would break test determinism; the flag is kept only for API stability. |
| **`MultiScaleMambaPredictor.step_imagine` validates its input width.** | It is fed a `d_model`-wide SSM output, not the decoded prediction; the check turns a silent numerical drift into a clear exception. |

## Known pitfalls (and how they are handled)

A few sharp edges that a reader (or a future refactor) should not have to rediscover:

- **Single-step sequences.** With ``T = 1`` there is no ``(t, t+1)`` pair, so the
  next-step prediction term of every loss is *vacuous*.  It is computed through
  :func:`ssm_latent.utils.mean_squared_error_or_zero`, which returns ``0`` rather
  than letting ``(...).mean()`` produce ``NaN`` (``0/0``) and poison the loss.
  This matters for streaming / single-frame inference, not just for tidiness.
- **No ``dropout`` knob.** Dropout is a no-op at inference and would break test
  determinism, so it is deliberately **not** offered on the configs (see
  :class:`ssm_latent.config.MultiScaleSsmConfig`).  There is no silently-ignored
  field to trip over.
- **Actions are never silently dropped.** An action-conditioned predictor
  (``action_dim > 0``) raises on ``forward(obs, None)`` and on ``step(z, None, …)``
  instead of quietly falling back to the unconditioned path.
- **The inner adaptation loop is deterministic** (it has no ``seed``); its only
  randomness is the task data itself.

## Out of scope / not implemented here

- ONNX text embeddings (`src/preprocess.rs` in the Rust port).
- MAML as an alternative outer loop (an open item in the shared roadmap).
- The WASM/web demos.
