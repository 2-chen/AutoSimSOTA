---
name: where-successes-come-from
description: Ask what could produce a successful trajectory here, not whether the repository ships an expert. The three sources are not interchangeable.
scope: general
confidence: methodological — a way to read a benchmark, not a measurement on one
evidence: |
  Found the hard way. On LIBERO the first reading reported that the benchmark could not
  produce trajectories without a person, because every collection script in the tree
  drives the robot by keyboard or SpaceNav. That reading was wrong. LIBERO's environment
  re-samples object placements on every reset unless `deterministic_reset` is set, and its
  step function returns `done = self._check_success()` — so the environment already
  labels any rollout it is given. Rolling out a policy and keeping the episodes the
  benchmark itself marks successful produces training data with no expert and no human.
  The repository was searched for a script that solves the task; the question was whether
  successes can be produced, and the two are not the same question.
---

# Where successes come from

Training data is a set of successful trajectories. Before concluding that a benchmark
cannot supply any, work out which of these it has — they cost different things and fail in
different ways.

**An authored expert.** A script that drives the environment to the goal: a planner, an
action bank, a scripted controller. It produces successes from zero, on demand, and its
failures are informative because it is trying. This is the strongest source and the rarest
— it exists because somebody wrote one per task, which is a lot of somebody's time.

**A human.** Teleoperation produces excellent trajectories and cannot be scaled, scheduled,
or run while you sleep. It is a real source; treat "someone can drive this" as different
from "this can produce data", because only one of them is available to a loop.

**The policy's own rollouts, filtered by the benchmark's own success predicate.** If the
environment randomises its initial state at reset and exposes a predicate that says whether
the task was completed, then running any policy gives trajectories that the benchmark
itself labels. Keep the successes. No expert, no human.

The third is easy to miss, and it is not free. Look for it in the environment's reset path
and its step function: does a reset draw a new initial state, and does a step report
success? If both are true, the source exists. It is worth knowing that the yield scales
with the current policy's success rate — at zero success it produces exactly nothing, so
it is a bootstrap rather than a supply. That is the real distinction from an expert, and it
is a distinction about *when* you can use it, not about whether it works.

## How to apply

When a benchmark's data supply is in question, answer these in order, and say which one you
are answering:

1. Is there a script that reaches the goal on its own?
2. Can a person drive it?
3. Does the environment randomise at reset and report success per step?

If only the third is true, the benchmark can produce data and its yield depends on the
policy you already have. That is a conditional yes, and reporting it as a no discards the
most common way a simulation benchmark can be researched.

If the answer depends on the policy rather than on the repository — which it does whenever
the third source is the only one — then reading the repository cannot settle it. Say what
is unresolved and what would resolve it: run the current policy for a batch of episodes and
count the successes. A measurement settles in minutes what an argument cannot settle at all.

## What it does not mean

It is not a reason to claim a benchmark can supply data when you have not checked that the
environment randomises and reports success. Environments with fixed initial states and no
per-step success signal exist, and for those the third source is genuinely absent.

It is also not a reason to skip asking whether the collected data would be *admissible*.
A benchmark that supplies no expert may also publish rules about what its submitted
policies may be trained on, and "I produced this myself" is a different answer from "the
benchmark authorises this".
