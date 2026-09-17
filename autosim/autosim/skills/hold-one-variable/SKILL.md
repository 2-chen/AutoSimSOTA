---
name: hold-one-variable
description: The harness configuration is a variable too; changing it while changing data credits the wrong thing.
scope: general
confidence: measured — the confound was real and large
evidence: |
  Measured on click_bell, n=100 per arm. Every arm that added local data also enabled a
  mixture manifest, and enabling that manifest switches the temporal sampler as well. A
  control with the original data and the new sampler but no new data at all scored 58%
  against 51% for the same data with the default sampler — roughly 7 of the 16 points
  that had been credited to adding data belonged to the sampling scheme.
---

# Hold one variable

Your experiment has more knobs than the one you are thinking about. The harness that
runs the training is one of them.

## How to apply

Before crediting a change, list what else moved when you made it. A pipeline option
that gets switched on because of your change — a sampler, an augmentation default, a
data-loading path, a normalisation recompute — is part of the treatment whether you
meant it to be or not.

The test is mechanical: what would you have to set differently to run your change with
the old harness behaviour? If that is not expressible, the harness is entangled with
your variable.

## What it does not mean

It does not mean the harness is off-limits. It means that if you are going to get credit
for a data-side change, you need an arm where the harness is held fixed, and if the
harness is itself worth improving, that is a separate claim deserving its own control.
