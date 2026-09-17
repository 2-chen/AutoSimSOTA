---
name: budget-attempts-not-episodes
description: Collection yields partial results; ask for attempts and expect a fraction to be rejected.
scope: general
confidence: observed — consistent across roughly ten collection runs
evidence: |
  On RoboSynChallenge, targeted collection admitted 44-67% of attempts (4 accepted from
  9 attempts; 30 from 50 on a larger budget). Rejections were quality-filtered
  trajectories, not process failures. Zero-yield rounds also occurred and were legitimate
  outcomes rather than crashes.
---

# Budget attempts, not episodes

A collection request is a request for *attempts*. What survives filtering is a fraction
of it, and you do not control that fraction directly.

## How to apply

Ask for more attempts than episodes you need — roughly 1.5-2x is a reasonable opening
estimate, and revise it from your own observed yield. Treat the yield itself as
information: a profile that admits very few trajectories is telling you something about
how rare the behaviour you asked for actually is.

A round that collected less than requested is not a failed round. Record the yield and
decide from it.

## What it does not mean

It does not license retrying a low-yield collection unchanged until it hits the target.
If the yield is poor, either the profile is asking for something rare — which may be
exactly what you want — or the collection parameters are wrong. Those have different
responses.
