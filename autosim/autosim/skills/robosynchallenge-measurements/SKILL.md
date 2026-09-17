---
name: robosynchallenge-measurements
description: Concrete numbers measured on RoboSynChallenge. Scoped to this benchmark — do not carry them to another one.
scope: benchmark:RoboSynChallenge
confidence: measured — paired per-episode seeds, n=100 per arm, official evaluation entry point
evidence: |
  All figures from controlled arms run through policy/act/eval.sh with 100 episodes,
  the random setting and paired seeds, on one machine. The machine does not reproduce
  the organiser's absolute published rates (the same released checkpoint scored 51-56%
  locally across four master seeds against 37% published, z=6.2), so these numbers are
  valid for comparing arms against each other and not as absolute claims.
---

# RoboSynChallenge: measured numbers

Scoped to this benchmark. These are the effects of specific levers as measured here.

## Sampling mass on locally collected data

The largest single lever found. 62 local episodes at 0.5 mass: **51% → 63-67%**.
The same data at natural mass (3.1%): **no change at all** (58/100 vs 58/100, p=1.0000).
See `weight-the-small-set` for why, and note this benchmark's original datasets are
1000 episodes per task, which is what makes a few dozen episodes negligible at natural
frequency.

## Which collection profile

Weak. Failure-analysis-chosen profile 67%; two independently drawn legal profiles 64%
and 63%; p=0.55 / 0.34 against the analyst's choice, 0.73 pooled. See
`make-your-prior-earn-it`. The legal targeted profiles here vary genuinely different
things — `composite_hard` randomises robot pose and camera intrinsics, `targeted_clutter`
randomises neither — and the difference did not show up in the score.

## The mixture sampler

Not neutral. Original data alone with the stratified phase sampler and no new data:
**51% → 58%** (~7 of the 16 points otherwise credited to adding data). See
`hold-one-variable`. The sampler's phase weights are partly cancelled by its horizon
weighting, so the net shift in the sampled distribution is smaller than the weights
suggest — which is why the size of this effect was worth measuring rather than deriving.

## Collection yield

Roughly **44-67%** of attempts admitted, varying by profile. See
`budget-attempts-not-episodes`.

## Available levers

Ten tasks, each with `random` and `clear` settings; the ranking suite freezes `random`.
Profiles available per task depend on the event families its config declares, so a task
with fewer randomised families has fewer legal profiles. Only `click_bell` currently has
a validated policy-correction adapter, so `policy_correction` is a legal collection mode
there and nowhere else.

## What is not comparable

Absolute rates are not comparable across machines or against the published table. The
per-task published baselines (click_bell 37%, water_pouring 72%, drawer_open_place 29%)
were measured on the organiser's stack; this machine's released-checkpoint numbers run
well above them and the drawer figure is additionally a modified-physics run. Use them
for orientation only.
