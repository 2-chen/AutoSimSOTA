---
name: read-your-failures
description: A configuration that crashed the run is a result. Read the error before proposing parameters again.
scope: general
confidence: observed twice on one machine; the specific ceiling is hardware-dependent and deliberately not encoded as a bound
evidence: |
  An action-chunk/optimiser configuration aborted the training process twice in one run
  — once as a native abort, once as a dataloader worker connection reset — while every
  other parameter stayed legal. The controller proposed the same value a second time,
  because the failure was not visible to it. A separate run was terminated outright by a
  parameter the system had published as legal but the trainer rejected.
---

# Read your failures

A round that could not execute is evidence about your proposal. It costs you the round
either way; reading it is free.

## How to apply

Before re-proposing after a failed round, read what failed. Distinguish:

- **A parameter the system rejected** — you have a bound wrong; stay clear of it.
- **A resource ceiling** — the process died rather than complained. This is usually
  hardware-specific: a batch size that works on one machine may abort on another. Prefer
  values known to work *here* before exploring outward, and record the ceiling you found.
- **A crash you cannot attribute** — do not retry the identical configuration hoping for
  a different outcome. Change one thing, and say which.

Crashes that are not your fault happen — native libraries abort, workers die. If the
same configuration crashed twice, stop treating it as bad luck.

## What it does not mean

It does not mean shrinking your search space permanently. A configuration that failed on
this machine may be correct on another; record the observation with its machine attached
rather than deleting the option.
