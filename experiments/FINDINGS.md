# Findings log

A running record of what each experiment in this directory actually showed, the
numbers behind it, and - just as important - the caveats.  Nothing here is a
publication; it is a lab notebook so we do not have to re-derive (or mis-remember)
a result.

Guiding question of the whole line of work:

> Can a small, cheap **world model** be steered by a frozen **LLM** (the "human with
> a car" picture) so that the *pair* reaches tasks neither could reach alone - and
> does that require touching any LLM weights?  (No: the coupling is intentionally
> loose - text and scalars only.)

---

## exp01 - adaptation speed vs. imagination (single seed)

**Question.** The SAI metric is *adaptation speed*.  Is it the same thing as one-step
accuracy, and does either predict whether the model can *imagine* (roll out in closed
loop)?

**Setup.** One ~40K model, 11 angular velocities; Reptile meta-training; closed-loop
imagination vs. a "persistence" (nothing-changes) baseline.

**Result.** Adaptation speed and one-step RMSE move together (r = +0.98), but neither
predicts closed-loop imagination (r = -0.62).  After meta-training imagination beats
persistence in 6/7 in-range tasks but only 1/4 out-of-range ones - even where
adaptation is fast (omega=0.3: 5 steps, AUC 0.035).

**Caveat.** Single seed, single model size.  Superseded by exp03.

## exp02 - does h-sampling cure closed-loop drift?

**Question.** `examples/03` calls `forward_with_h_sampling` "the remedy" for drift.
Does it work?

**Result.** **No.**  Over 3 seeds, raising `h_sampling_prob` 0.0 -> 0.6 made one-step
(0.27 -> 0.50) and closed-loop (0.46 -> 0.68) RMSE *worse*.  `examples/03` now says so.

## exp03 - robustness sweep (pre-declared protocol)

**Question.** Does exp01 hold up across sizes and seeds, with uncertainty reported?

**Setup.** 2 sizes x 3 seeds x 9 omegas = 54 cells; bootstrap CIs.

**Result.** Holds up.  r(adapt AUC, one-step) = **+0.93 [+0.89, +0.96]**;
r(adapt AUC, imagination gain) = **-0.59 [-0.71, -0.48]**.  Imagination beats
persistence in **27/30 in-range** cells vs **4/24 out-of-range**.  Counter to
intuition, the *larger* model does **worse** out of range (omega=3.2 gain -0.15 at
d_model=32 vs -0.03 at d_model=16).

**Conclusion.** "Adapts fast / predicts well one step" is a bad proxy for "can be
used to imagine".  The breakdown is concentrated out of the training range.

## exp04 - objective ablation (INCOMPLETE)

**Question.** Is the closed-loop breakdown a property of the *objective* (JEPA/SIGReg)
or of closing the loop in general?

**Status.** Implemented and smoke-tested, but **not yet run at full scale** (the run
was paused before it finished).  It compares an obs-space MSE predictor against the
JEPA latent model under VICReg vs. SIGReg regularisation.  This is the natural next
piece of (B).

## exp05 - the loose loop (frozen model + a few adaptation steps on a demo)

**Question.** With a *scripted* driver, can a few adaptation steps on a short demo make
a pretrained model usable (closed loop) on a new target, including out of range?

**Result.**
- In-range targets: usable already at k=0 (the meta-learned prior suffices); extra
  adaptation is neutral-to-slightly-harmful.
- Out-of-range omega=0.25: k=1-3 adaptation *rescues* it (**both seeds positive**),
  but k>=5 drifts back.
- Out-of-range omega=3.0: **seed-dependent** (one seed improves, one does not).

**Reading.** A few steps can carry the model a little beyond its prior - "the driver
reaches a place it could not" - but the effect is fragile and over-adaptation hurts.
This is the first (weak) support for the "human + car" picture.

## exp06 - the LLM chooses a curriculum (one shot)

**Question.** Give a frozen LLM the targets and ask for a training curriculum.  Does it
beat the obvious human heuristic ("widen the range")?

**Result.** **No.**  default meanOOD -0.268, **wide -0.252**, LLM one-shot -0.289.
The LLM independently converges on "include the extremes" - i.e. *the same idea as a
human* - and all curricula stay negative out of range.

**Reading.** A single answer is not "driving".  The value of loose coupling, if any,
must come from *operating* the model (trying, observing, revising), not from one guess.

## exp07 - the LLM *runs the experiments* (hypothesis -> trial -> revise)

**Question.** Give the LLM a `run_trial` tool and a budget of trials.  Does searching
beat answering once, and beat the scripted baseline?

**Result (seed 0).** The LLM ran 10 trials, discovered the adaptation budget `k` as a
lever, raised it 5 -> 10, and reached meanOOD **-0.151** - beating wide (-0.252) by
+0.10.

**Caveat.**  One seed, one trial per setting.  Directly re-tested by exp08.

## exp08 - is exp07 robust? (pre-declared)

**Question.** Does the LLM's `k=10` finding hold across seeds?

**Result.** **NOT robust.**  k=10 over k=5 holds in **2/3** seeds; beating `wide`
holds in **1/3** seeds (seed1 and seed2: wide wins).  All 9 cells negative.  The best
single cell overall was actually `wide` at k=5, seed2 (**-0.040**).

**Secondary observation.** The LLM's `k=10` setting was *tight* across seeds
(-0.151 / -0.138 / -0.140) while `wide` was scattered (-0.252 .. -0.040) - i.e. the
LLM may have found a *stable* setting rather than an *accurate* one.  Unverified.

**Meta-lesson.** exp07's positive claim was a single-seed artefact.  Recorded here on
purpose: this is why exp08 existed.

## exp09 - do the *mechanism* levers (objective, adaptation) fix the breakdown?

**Question.** Settings search (curriculum, budget) hit a ceiling, so change the two
things that could actually be the cause: the objective (one-step vs. closed-loop
"rollout" training) and the adaptation mechanism (gradient steps vs. a weight-free
state warm-up).  2x2 grid, 2 seeds.

**Result.** **No lever helps.**  Against the baseline (oneshot + weight, mean OOD
-0.160): oneshot+warmup -0.278, rollout+weight **-0.438** (and it *breaks in-range too*:
-0.04, -0.18), rollout+warmup -0.284.  Usable cells: **0/8**.

**Reading.** Training for the thing we measure (imagination) makes out-of-range
*worse*, not better - fitting the closed loop on the in-range data buys overfitting,
not generalisation.  The breakdown is not a loss-design problem.

## exp10 - is the breakdown the model's fault, or the baseline's?  (diagnosis)

**Question.** At omega=0.25 the point barely moves, so "repeat the last frame" should
be nearly correct.  Is "the model loses to persistence" even a meaningful failure?

**Result.**  Two things are true at once:
- **The baseline is very strong at low omega.**  persistence RMSE 0.32 and *linear*
  extrapolation (continue the last velocity) **0.11** at omega=0.25, vs the model's
  0.70.  So that headline overstates the failure: the task is nearly straight-line,
  and even a trivial extrapolator wins.
- **The model has a real defect.**  Its latent rollout error (0.79) is **2.1x** the
  true latent's travel (0.38) at omega=0.25 - the model *cannot hold still*: it drifts
  more than the task moves (err/travel = 0.46 in-range, 2.11 at omega=0.25).

**Reading.** Part of the "OOD collapse" is an evaluation artefact (a too-strong
baseline on an easy task); part is a genuine failure to stop.  Any comparison should
report both the persistence *and* the linear baseline, and a "hold-still" error.

---

## Where this leaves the "LLM drives a world model" picture

| Claim | Status |
|---|---|
| A few adaptation steps can push a pretrained model out of its prior (loose loop works at all) | **Weak yes** (exp05, omega=0.25, both seeds) |
| Tuning the *curriculum* or *budget* fixes out-of-range imagination | **No** (exp06, exp08, seed-robust) |
| A frozen LLM, given a tool, can search better than a scripted heuristic | **Unproven** - the one positive result did not reproduce (exp07 vs exp08) |
| Changing the objective / adaptation mechanism fixes it | **No** (exp09 - rollout training made it worse; 0/8 usable cells) |
| The breakdown is (partly) an evaluation artefact | **Yes** (exp10 - a linear extrapolation reaches 0.11 at omega=0.25 vs the model's 0.70) |
| The model has a genuine defect | **Yes** (exp10 - it drifts 2.1x more than the task moves at omega=0.25; it cannot hold still) |
| A learned world model beats "just write the equations" on writable dynamics | **No** (exp11 - code is 10-40x more accurate on both a non-chaotic and a chaotic system) |
| Hiding the parameters makes code lose | **No** (exp12 - a fitted integrator is still 7-20x more accurate; not knowing the *form*, not the parameters, is what would stop code) |

## exp11 - on a task with no closed form, does a learned world model earn its keep?

**Question.** Every generator so far was closed-form, where an *exact* Python tool
predicts to ~1e-6 - so a learned model could never win.  Move to systems that are only
numerically integrable (damped driven pendulum; chaotic double pendulum) and race three
forecasters: persistence, a learned JEPA/SSM world model, and "code" (the *true*
equations integrated from a state estimated from observations alone).

**Result.** Code wins everywhere, and by a lot:
- driven pendulum: code 0.019 / 0.033 / 0.054 at horizons 5/10/20, vs learned
  0.271 / 0.538 / 0.728 and persistence 0.299 / 0.537 / 0.785.
- double pendulum: code 0.017 / 0.048 / 0.054, vs learned 0.741 / 0.882 / 1.195 and
  persistence 0.266 / 0.686 / 1.423.

The learned model is roughly at persistence at best (driven pendulum), and *worse* than
persistence at short horizons on the chaotic system.  Code is 10-40x more accurate, and
the gap does **not** close with horizon - even chaos stays predictable for ~20 steps.

**Reading.** If the dynamics are *writable as equations*, writing them beats learning
them - always, and cheaply.  There is no "learned world model" niche here.  This
quantifies the intuition that motivated the whole detour: an agent that can write Python
already has a better "car" for these tasks.

**Where a learned model could still matter.**  Only when the observation does *not*
contain enough to write the equations: hidden state, unknown parameters, partial
observability.  That is what exp12 sets out to construct.

## exp12 - where code stops winning: hidden parameters + partial observability?

**Question.** exp11 said a learned model only matters when the observation cannot yield
the equations.  So build that: a damped driven pendulum whose *(b, amp, drive)* vary per
episode and are never observed (only `(sin theta, cos theta)` is), and race a learned
world model against the *strongest* code baseline - "code+sysid", which fits the hidden
parameters by gradient descent and then integrates the true equations.

**Result.** The prediction was **wrong**.  code+sysid wins at every band width
(`spread` 0.0 / 0.5 / 1.0): at horizon 10 it scores 0.031 / 0.032 / 0.042, versus the
learned model's 0.535 / 0.466 / 0.465 and persistence's 0.577 / 0.574 / 0.566.

**Reading.** Hiding the *parameters* does not stop code: fitting three numbers from a
32-step context is easy, and the fitted integrator stays 7-20x more accurate.  So
"unknown parameters" is not "cannot write the equations".  The only thing that would
stop code is not knowing the *functional form* - which none of exp11/exp12 tested.

**The consistent conclusion of the whole sweep.**  Across closed-form (pre-exp11),
numerically-integrable, chaotic, and hidden-parameter/partial-observation dynamics, code
always wins, usually by 1-2 orders of magnitude.  The learned world model sits at roughly
the persistence baseline.  A learned model is only worth driving when the *shape* of the
dynamics is unknown - a case that has not yet been constructed here.

## Where this leaves the search for a fix

Neither the *settings* (exp06, exp08) nor the *mechanism* (exp09) moves the needle, and
exp10 shows the low-omega failure is half evaluation artefact, half a real "won't hold
still" defect.  So the next lever is not another knob on the training loop - it is
either

  * the **task / evaluation design** (report a linear-extrapolation baseline and a
    hold-still error; low omega is close to trivial), or
  * the **architecture / data** (the model cannot represent "stop", which may be an
    initialisation or timescale problem - cf. exp03, where the *bigger* d_model did
    worse out of range).

exp04 (objective ablation at full scale) is still the clean way to rule out the
objective once and for all; it is implemented and smoke-tested but not yet run.

## Reproduce

Every number above is regenerable: each experiment caches its raw results under
`experiments/results/*.json` and re-runs with `--force`.  exp04 has not been run to
completion yet; everything else has.


---

# Overall status (as of exp12)

## The one thing the sweep established

Across every task class we tried, **a forecaster that knows the functional form of the
dynamics beats a forecaster that has to learn it from data**:

| task class | best code | learned WM | code advantage | experiment |
|---|---|---|---|---|
| closed-form circle/sine | ~1e-6 (analytic) | 0.5-1.1 | ~10^6x | pre-exp11, exp10 |
| non-chaotic, no closed form (driven pendulum) | 0.02-0.05 | 0.27-0.73 | 10-40x | exp11 |
| chaotic (double pendulum) | 0.02-0.05 | 0.74-1.20 | 10-40x | exp11 |
| hidden params + partial observation | 0.02-0.10 | 0.30-0.70 | 7-20x | exp12 |

The learned model rarely beats the trivial persistence baseline by much, and never
approaches code.  "Unknown parameters" is *not* enough to stop code - fitting them is
easy; only an unknown *functional form* would.

## The other things learned along the way

- **Adaptation speed is not a proxy for usability.**  Adaptation AUC and one-step RMSE are
  nearly the same axis (r=+0.93, exp03); neither predicts closed-loop imagination
  (r=-0.59).  A model can adapt in 5 steps and still fail to imagine.
- **Low-omega "collapse" is half artefact, half real defect.**  The baseline is very
  strong there (linear extrapolation 0.11, exp10), *and* the model cannot hold still
  (latent drifts 2.1x the task's motion).
- **No setting fixes it.**  Curriculum (exp06, exp08, seed-robust), budget (exp08), the
  training objective, and the adaptation mechanism (exp09) all fail.  0 usable cells.
- **"A frozen LLM drives a world model" was not demonstrated.**  The one positive result
  (exp07) did not reproduce across seeds (exp08).  The LLM, given a tool, searched no
  better than a scripted heuristic on this task.
- **The meta-observation.**  The most capable "car" in this whole exercise was the Python
  interpreter the assisting model was using: it diagnosed the failures exactly, and
  predicted the physics to 1e-6.  That is the same picture the project set out to build -
  an intelligence extended by an external tool - already working, for these tasks.

## What this means for the vision (LLM drives a learned world model)

The vision is not falsified, but its **domain is much narrower than assumed**.  For any
system whose equations an agent can *write* (even with unknown parameters), writing them
wins, and a learned world model adds nothing worth driving.  The learned world model earns
its keep only when the functional form is unavailable - contact/friction/plasticity,
unknown constitutive laws, biology/markets, or the closed loop of a learned policy.  That
is the regime any future experiment should target; everywhere else, the honest baseline is
"write the simulator".

---

> The world-model search was **paused** on 2026-10-12.  For *why* - the argument that
> the field may be conflating intelligence with life (self-closure, not compute) - see
> [`REFLECTION.md`](REFLECTION.md).
