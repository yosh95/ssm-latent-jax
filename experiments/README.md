# experiments/

Scratchpad for **unpublished, exploratory** work.  Nothing here is part of the
published `ssm_latent` package, and nothing here is run by CI: it exists so that
the numbers behind a draft (and the negative results that never make it into a
paper) can be regenerated exactly.

Each script is standalone and writes its table to stdout:

```bash
python experiments/01_adaptation_vs_imagination.py
python experiments/02_h_sampling_remedy.py
python experiments/03_robustness_sweep.py          # ~11 min; --force to recompute
python experiments/05_loose_loop.py
python experiments/06_llm_driver.py                # needs LLM_CLI_API_URL / _MODEL / _API_KEY
python experiments/07_llm_agent.py                 # same env; runs the LLM in a loop
python experiments/08_k_robustness.py
```

Raw numbers are cached under `experiments/results/` so a re-run reuses them unless
`--force` is passed.

| Script | Question | Preliminary finding (single/low-seed, synthetic) |
|---|---|---|
| `01_adaptation_vs_imagination.py` | Are "fast adaptation", "one-step accuracy" and "closed-loop imagination" the same thing? | Adaptation speed and one-step RMSE track each other almost perfectly (r=+0.98 over 11 tasks) but **neither predicts closed-loop imagination** (r=-0.62). After meta-training, imagination beats the persistence baseline in 6/7 in-range tasks but only 1/4 out-of-range ones - even though adaptation is *fast* at omega=0.3 (5 steps, AUC 0.035). |
| `02_h_sampling_remedy.py` | Does scheduled `h`-sampling (`forward_with_h_sampling`), the "remedy" named in `examples/03`, actually reduce closed-loop drift? | **No.** Over 3 seeds, raising `h_sampling_prob` 0.0 -> 0.6 made both one-step (0.27 -> 0.50) and closed-loop (0.46 -> 0.68) RMSE *worse*. |
| `03_robustness_sweep.py` | Harden 01 into a pre-declared protocol: 2 sizes x 3 seeds x 9 omegas, with bootstrap CIs. | Holds up. r(adapt AUC, one-step) = +0.93 [+0.89, +0.96], r(adapt AUC, imagination) = -0.59 [-0.71, -0.48] (n=54). Imagination beats persistence in 27/30 in-range cells vs 4/24 out-of-range; larger models do *worse* out-of-range, not better. |
| `04_objective_ablation.py` | Is the breakdown a property of the objective (JEPA/SIGReg) or of closing the loop in general? | **Not run yet** (implemented + smoke-tested). First half of (B). |
| `05_loose_loop.py` | Can a *scripted* driver + a few adaptation steps make a pretrained model usable (closed loop) on a new target? | In-range: usable at k=0. Out-of-range omega=0.25: rescued by k=1-3 (both seeds), drifts back at k>=5. omega=3.0: seed-dependent. Weak support for the loose loop. |
| `06_llm_driver.py` | A frozen LLM picks a training curriculum in one shot. | **No better than the obvious human heuristic.** default -0.268, wide -0.252, LLM one-shot -0.289. The LLM reinvents "include the extremes". |
| `07_llm_agent.py` | The LLM *runs* trials via a `run_trial` tool and revises. | On seed 0 it found the `k` lever and beat `wide` (-0.151 vs -0.252). **Did not survive exp08.** |
| `08_k_robustness.py` | Is exp07's `k=10` finding robust across seeds? | **No.**  k=10>k=5 in 2/3 seeds; beating `wide` in 1/3. All cells negative. exp07 was a single-seed artefact. |

See [`FINDINGS.md`](FINDINGS.md) for the narrative, the caveats, and what it implies for
the next step.

## How to read these

These are **leads, not results**.  The models are tiny, the tasks synthetic, and
the training short; the point is to find structure worth chasing, then confirm it
with proper seeds, baselines and a pre-declared protocol.  Treat every table as
"this is what one afternoon of compute said".

## Conventions

- Everything is CPU, float32, and deterministic per seed (the generators are
  deterministic and the RNG is keyed).
- `common.py` holds the shared metrics; importing it puts `src/` on `sys.path`,
  so a fresh checkout runs without an editable install.
