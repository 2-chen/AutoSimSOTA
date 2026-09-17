---
name: weight-the-small-set
description: A small in-domain set carries weight only if the sampler is told to lean on it; volume alone does nothing.
scope: general
confidence: measured, paired, two benchmarks — but the exact mass that works is benchmark-specific
evidence: |
  Measured on RoboSynChallenge click_bell, n=100 per arm with paired per-episode seeds.
  Adding 62 locally collected episodes to a 1000-episode original dataset and giving them
  0.5 sampling mass moved success from 51% to 63-67% (exact paired p from 0.0024 to
  0.0001). The same kind of data left at its natural mass — 32 episodes at 3.1% — moved
  it by exactly nothing: 58/100 against 58/100, p=1.0000.
---

# Weight the small set

Your own collected data is a handful of episodes next to a large original dataset. In
frame terms it is a rounding error. It only affects training if the sampler is told to
pick it disproportionately often.

## How to apply

Treat "how much did you collect" and "how often does training see it" as two separate
decisions. Collecting without up-weighting is close to collecting nothing, and it is an
easy mistake to make because the data *is* in the mixture and the run does look
different.

When you up-weight, say what mass you chose and why. When you do not, say why you expect
the data to matter at its natural frequency — usually you should not.

## What it does not mean

The mass is not free. Up-weighting a small set to a large fraction means training spends
most of its samples on a few episodes and sees the original distribution rarely. The
right mass depends on how far the original distribution is from what the evaluation
actually presents, which is why the useful value is something to measure rather than
assume.
