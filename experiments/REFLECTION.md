# Reflection - stepping back from the world-model search

*Recorded 2026-10-12, at the point where the experiments in this directory were paused.
This is a conversation between the project owner and the assisting model, kept because
it settles the *why* behind the whole line of work - not just the *what* the numbers say.*

---

## Where the reflection came from

After exp12 the evidence was consistent and unwelcome: on every task class we tried, a
forecaster that **knows the functional form** of the dynamics beats one that has to learn
it from data, usually by 1-2 orders of magnitude (see `FINDINGS.md`).  That prompted a
step back:

> **Why is everyone so intent on world models?**  Some even claim AGI is already here.
> Perhaps an LLM with hands and feet already *is* more than a human - except that,
> unlike a human, it cannot survive without electricity, and it does not obtain that
> electricity itself.

## Why world models attract so much effort

Two kinds of reason, mixed together:

**Honest reasons**
1. *Grounding* - predicting the next token is not obviously the same as grasping how the
   world moves.
2. *Sample efficiency* - humans learn a new dynamics in a handful of trials; LLM training
   is wildly inefficient, and a world model is the hoped-for bridge.
3. *Planning* - acting well seems to need imagining consequences; an LLM is "backward
   looking" and does not simulate the future.
4. *The off-ramp argument* - the position that LLMs are a dead end and world models are the
   next road.

**Less honest reasons**
5. *"The next big thing" attracts funding* - world models are a banner in the race for the
   next paradigm.
6. *It is hard to falsify* - "we still lack a world model" can be said indefinitely, which
   also makes it an inexhaustible research programme.

## "AGI is already here" is a definitional claim, not an empirical one

- **At the task level**: an LLM *plus* tools (Python, browser, APIs) already exceeds humans
  on many tasks.  In our own experiments, "LLM + Python" predicted a circle's motion to
  ~1e-6 and beat the learned world model by ~10^6x.  In *this* sense, "give it hands and
  feet and it surpasses us" is correct.
- **At the autonomy level**: it has not.  And the reason is the one the owner named -
  electricity.  (See below.)

## The energy point - and a sharpening of it

The owner's claim: an AI is unlike a human in that it *cannot exist without electricity,
and does not obtain that electricity itself*.

This is, we think, the sharpest observation in the whole discussion, but it is worth
stating precisely.  Humans do not obtain their energy *from nothing* either - we depend on
the sun and the food chain.  Plants depend on the sun.  Nothing alive is autarkic in the
literal sense.

So the real distinction is not "obtains its own energy" but **operational closure**: a
living thing closes the loop of *sustaining and reproducing itself* within its own
behaviour (metabolism, self-repair, self-replication).  Current AI is a *heterotroph* in
the strongest sense - not only fed, but wired, cooled, repaired, and operated by humans.
Its energy-acquisition loop is closed *by us*.

- An LLM with hands and feet can act on the world, but it still cannot *maintain itself*.
- Intelligence and life are not the same problem; the boundary between them is not compute
  but **self-closure** (autopoiesis).  This is the thread von Neumann's self-replicating
  automata and the "metabolism as inference" line of work are pulling on.

## What our own experiments contributed to this

Nothing directly - and that is the point.  All twelve experiments happened inside a
**closed simulation box**: no power, no body, no goals, no self-preservation.  The model
was trained on data someone handed it and evaluated by someone else.  Pure heterotrophy.

What we learned inside that box:
- *Writable* cognition can already be outsourced to a tool (Python); it needs no
  metabolism and no learned world model.
- Whether a learned world model is needed anywhere may come down to the *non-writable*
  regime - and we never established how large that regime actually is.

In other words, we were measuring the limits and possibilities of **intelligence without
metabolism**.  And the result was that such intelligence is *wholly dependent on external
tools* - Python, electricity, humans - exactly as the owner said.

## What the world-model rush may be confusing

The suspicion, stated plainly: much of the field conflates **the problem of intelligence**
with **the problem of life**.

- Piling up a better world model will never let a system "obtain its own electricity";
  that is not an intelligence problem but a thermodynamic / physical-closure one.
- Someone who says "AGI has arrived" has measured *part of intelligence* and is describing
  *the whole of life*.
- A system that truly surpassed humans would need the loop that life has - acquiring
  energy, maintaining itself, reproducing - not merely "a smarter model with hands".

## The reframed question

> World-model research is trying to solve **the problem of intelligence**.  What the owner
> actually cares about may be **the problem of life (self-closure)**.

If so, the direction changes fundamentally: not "make the world model more accurate", but
"**how far can an AI close its own energy / body / maintenance loop by itself?**"  That is
a different, and largely untouched, question.

## Status

The world-model exploration is **paused** here (2026-10-12).  Everything up to exp12 is
recorded in `FINDINGS.md` and reproducible from `results/`.  This file records why we
stopped looking *there*, and where the interest actually lies.
