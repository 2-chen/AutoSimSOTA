---
name: control-same-budget
description: Judge an intervention against a control that spent the same total budget, never against the released checkpoint.
scope: general
confidence: methodological — this is a comparison-validity requirement, not a measurement
evidence: |
  A candidate that trains across several rounds has received several rounds of training
  budget. Measured on click_bell: pure continuation training on the original data for the
  same number of steps moved the released checkpoint from 53% to 51% — i.e. extra
  training alone did nothing — while the locally-augmented arm reached 63-67%. Without
  the same-steps control, that gap would have looked like an effect of the data when part
  of it was budget the control never got.
---

# Control at the same budget

The released checkpoint is not your baseline. It has had less training than your
candidate the moment you train for a single step.

## How to apply

Every time you claim an improvement, name the control that spent the same compute,
the same steps, the same evaluation protocol, and differed only in the thing you are
claiming credit for. If your run chains training across rounds, the control must chain
too, or must be a single run of the *total* steps.

The cheapest correct control is usually continuation training on the original data for
the same total steps, evaluated on the same frozen bank.

## What it does not mean

It does not mean more training never helps — it means you cannot tell whether it helped
unless something else held the budget fixed. It also does not excuse you from reporting
the released-checkpoint number; that number is what a reader will compare against, it
just is not evidence about your method.
