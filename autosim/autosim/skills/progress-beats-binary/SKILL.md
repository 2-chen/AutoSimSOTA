---
name: progress-beats-binary
description: A graded progress score separates policies with far fewer rollouts than a pass/fail verdict.
scope: general
confidence: high — a statistical result, benchmark-independent, validated at scale
evidence: |
  N-SCORE / "Beyond Binary Success" (RSS 2026, arXiv:2603.13616): a sequential
  anytime-valid test over paired progress scores reduced evaluation burden by roughly 70%
  versus batch testing (>50% better than STEP) on simulation, ~45% on hardware. On
  RoboArena it recovered a full ranking of four policies at α=0.05 using 1,420 of ~2,560
  evaluations. The paper states partial-credit metrics "consistently separate competing
  policies faster than binary success".

  RoboArena (arXiv:2506.18123) builds its protocol on a continuous 0-100 progress score
  alongside pairwise preference, rather than success alone.
---

# Progress beats binary

Two policies that both fail a task can have failed by very different amounts, and a
pass/fail verdict throws that information away. Keeping it makes every rollout more
informative.

## How to apply

Where the task admits a graded measure — how far through a sequence it got, how close to
the goal state, how many of N sub-goals completed — record it alongside the binary
outcome. Compare on both. The graded score answers "is this policy better" with fewer
rollouts; the binary score answers "does it work", which is what a reader ultimately
wants and what the benchmark's own metric reports.

This matters most when rollouts are expensive and effects are small, which is the usual
situation.

## What it does not mean

It does not replace the benchmark's own success criterion. The headline number must stay
the native one — a graded score is an *additional* instrument for ranking under noise,
not a substitute metric. And a progress score is only worth having if it is measurable
without privileged access the policy itself could not have; a score computed from
simulator internals is a diagnostic, not an evaluation.
