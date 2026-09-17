---
name: quality-needs-reweighting
description: A dataset of mixed quality is not usable as-is by plain behaviour cloning; the quality has to be handled, not just included.
scope: general
confidence: high — a controlled robomimic study plus two independent replication and curation results
evidence: |
  robomimic (arXiv:2108.03298, and the study page): offline RL (CQL, IRIS) beats behaviour
  cloning *only* on Machine-Generated data; on Proficient-Human and Mixed-Human data
  history-conditioned BC wins. The authors read this as BC being bounded by its data. A
  replication on robomimic's combined mixture concluded that "removing the low-quality
  data allows for expert performance, as in original robomimic" — the failure was caused
  by the low-quality portion, not by data volume.

  Curation results in the same direction: SAL filtering reached +16% on Transport using 50
  of 300 demonstrations; TED reached +20% on real Push Block with half the data.

  And human data specifically has a documented pathology: it is non-Markovian (teleop
  device plus action history), which is why offline RL methods that work on
  machine-generated data scored ~20-70% on human data where BC variants exceeded 96%.
---

# Quality needs reweighting

Mixed-quality data is a different problem from same-quality data, not a bigger version of
it. Concatenating it and training as usual lets the weakest portion set the ceiling.

## How to apply

When a corpus has more than one quality tier, decide *per tier* what happens to it:
excluded, down-weighted, filtered by some criterion, or fed to a method that can use it
(a return-conditioned or offline-RL objective). "It is in the mixture" is not a decision.

The diagnostics are cheap. If a small curated subset outperforms the full mixture, you
have the robomimic result. If dropping a tier improves the number, the tier was net
negative — see `adding-more-can-subtract`.

Watch for the non-Markovian case as well: data collected with a teleoperation device
carries action history the observation does not contain, and methods that assume the
Markov property will underperform on it while looking fine on machine-generated data.

## What it does not mean

It does not mean low-quality data is useless — it means it cannot be treated as
interchangeable with high-quality data at equal weight. The offline-RL result is that
there *are* methods that extract value from it; plain BC is not one of them.
