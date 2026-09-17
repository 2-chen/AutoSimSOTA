---
name: diversity-before-volume
description: Generalisation scales with the number of distinct environments and objects, not with the number of demonstrations.
scope: general
confidence: high — a controlled ablation plus a dedicated scaling study across two independent groups
evidence: |
  RoboCasa365 pretraining ablation (ICLR 2026, arXiv:2603.04356): Human50 34.7 average vs
  Human300 40.0, with the gain concentrated on Composite-Unseen (23.8 -> 32.3) rather than
  in-distribution. Scene-count scaling reported in the same work: 29.6 -> 39.6 -> 44.7.

  Lin et al., "Data Scaling Laws in Imitation Learning" (ICLR 2025, arXiv:2410.18647),
  40k+ demos and 15k+ real rollouts: generalisation follows a power law in the number of
  training environments and objects, and once demonstrations-per-environment pass a
  threshold "additional demonstrations have minimal effect". Their recipe — 32
  environments x 1 object x 50 demos — reached ~90% on novel environments with unseen
  objects, collected by four people in an afternoon.
---

# Diversity before volume

When you have budget for more data, the first question is not "how much" but "how many
distinct situations".

## How to apply

Count the distinct scenes/objects/configurations your data covers before you count the
episodes. If two demonstrations come from the same initial configuration with a
different seed, you have one situation observed twice, not two situations. Adding more
of those saturates fast.

This changes what a collection round should ask for. A targeted profile that varies
appearance, camera and pose is purchasing diversity; a profile that varies nothing but
the seed is purchasing volume, and volume is the axis that saturates.

## What it does not mean

It is not a licence to ignore per-configuration demonstration count — Lin et al. found a
threshold below which more demos *do* help. And "diverse" is not automatically "useful":
RoboCasa365 found that adding a large batch of lower-quality generated trajectories to a
curated corpus made generalisation worse, not better (see `adding-more-can-subtract`).
Diversity of *situations* is what scales; volume of *items* is what saturates.
