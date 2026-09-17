---
name: adding-more-can-subtract
description: Adding lower-quality data on top of a curated corpus can make generalisation worse, not better.
scope: general
confidence: medium-high — a clean controlled table plus several consistent results in the same direction, all in the "add-to-an-already-strong-base" regime
evidence: |
  RoboCasa365 (arXiv:2603.04356), §4.4, verbatim: "Compared to training on all pretraining
  data (Human300 + MG60), we find that training on just the human data (Human300) yields
  better downstream learning results." Averages: Human300 40.0 vs Human300+MG60 35.9 —
  and the loss is concentrated on Composite-Unseen, 32.3 -> 22.7. The synthetic data hurt
  generalisation, not in-distribution fit.

  Consistent results: arXiv:2606.20999 finds MimicGen additions give no improvement to
  inductive generalisation on RoboCasa; DataMIL (arXiv:2505.09603) finds "selecting data
  naively may actually harm downstream performance"; LIBERO (arXiv:2306.03310) reports
  that "naive supervised pretraining can have a negative impact"; Ambient Diffusion
  Policy (arXiv:2606.12365) finds co-training plateaus as suboptimal data is scaled.

  A sign-change worth knowing: the earlier RoboCasa (RSS 2024) reported the opposite —
  MimicGen data beating human-only substantially. Both can hold: synthetic data wins when
  the human corpus is small, and stops winning once it is large and diverse.
---

# Adding more can subtract

More data is not a monotone improvement. Past some point, adding a lower-quality block to
a good corpus reduces what the model learns, and it shows up worst on the cases you care
about most — the unseen ones.

## How to apply

Treat "include this block of data" as a claim that needs a control, exactly like a
parameter change. The control is the same training without the block. Where the block is
synthetic or machine-generated, expect it to help when the curated corpus is thin and to
stop helping once it is not.

If you do include it, the evidence says to reweight or filter rather than dump it in —
see `quality-needs-reweighting`. Mixing ratio is a real variable, and pushing it toward
the larger but weaker source degrades results.

## What it does not mean

It is not an argument against synthetic data in general, and the effect is
scale-conditional: the sign flips depending on how good the base corpus already is. It is
an argument against assuming that a larger mixture is a better one.
