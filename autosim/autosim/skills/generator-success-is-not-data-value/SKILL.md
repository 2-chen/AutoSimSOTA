---
name: generator-success-is-not-data-value
description: How often your collection pipeline succeeds is not a proxy for how useful the data it produced will be.
scope: general
confidence: high — stated by the authors of the generator against their own intuition, with two concrete reversals
evidence: |
  MimicGen (arXiv:2310.17596), verbatim: "It is tempting to think that data generation
  success rate and trained agent performance are correlated, but we found that this is
  not necessarily true." Their examples: a task with 29.5% generation success produced an
  82.0% agent; a task with 8.2% generation success produced 76.0%.

  The same work documents that generation is badly uneven across seeds: in one generated
  dataset "over 850 of the 1000 episodes come from just 3 source demonstrations", with one
  seed yielding >170 episodes and another <10 under uniform attempt budgets. The
  mechanism is that the generator only filters on task success, so a dataset can be
  simultaneously large and heavily biased toward whatever the generator finds easy.
---

# Generator success is not data value

An easy-to-produce episode and a valuable episode are different things, and your
collection pipeline's yield rate measures the first.

## How to apply

Do not use collection yield as a quality signal for the dataset, and do not discard a
low-yield collection on the assumption that it was a failure. A profile that admits few
trajectories may be the one reaching situations the policy cannot handle — which is
exactly what you went looking for. The low-yield end is often where the value is.

Symmetrically, a collection that admitted nearly everything is not thereby good. It may
have re-sampled the cases the pipeline already handles.

What *is* worth watching is the **distribution of the yield across seeds/sources**. If a
handful of sources produce most of the accepted episodes, the dataset is narrow no matter
how large it looks — high total volume, low effective diversity.

## What it does not mean

It does not mean yield is uninformative. A zero-yield round still means no data, and a
sudden drop in yield may indicate something broke. It means yield is a fact about the
generator, not a verdict on the dataset.
