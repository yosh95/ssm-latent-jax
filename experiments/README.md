# experiments/

Scratchpad for **unpublished, exploratory** work.  Nothing here is part of the
published `ssm_latent` package, and nothing here is run by CI: it exists so that
the numbers behind a draft (and the negative results that never make it into a
paper) can be regenerated exactly.

Each script is standalone and writes its table to stdout:

```bash
python experiments/01_adaptation_vs_imagination.py
python experiments/02_h_sampling_remedy.py
```

| Script | Question | Preliminary finding (single/low-seed, synthetic) |
|---|---|---|
| `01_adaptation_vs_imagination.py` | Are "fast adaptation", "one-step accuracy" and "closed-loop imagination" the same thing? | Adaptation speed and one-step RMSE track each other almost perfectly (r=+0.98 over 11 tasks) but **neither predicts closed-loop imagination** (r=-0.62). After meta-training, imagination beats the persistence baseline in 6/7 in-range tasks but only 1/4 out-of-range ones - even though adaptation is *fast* at omega=0.3 (5 steps, AUC 0.035). |
| `02_h_sampling_remedy.py` | Does scheduled `h`-sampling (`forward_with_h_sampling`), the "remedy" named in `examples/03`, actually reduce closed-loop drift? | **No.** Over 3 seeds, raising `h_sampling_prob` 0.0 -> 0.6 made both one-step (0.27 -> 0.50) and closed-loop (0.46 -> 0.68) RMSE *worse*. |

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
