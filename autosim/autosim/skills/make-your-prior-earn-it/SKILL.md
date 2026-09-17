---
name: make-your-prior-earn-it
description: A reasoned choice must be shown to beat an arbitrary legal one before you credit the reasoning.
scope: general
confidence: methodological — this is an epistemics rule, not a measurement
evidence: |
  Measured on RoboSynChallenge click_bell, n=100 per arm, paired seeds: a collection
  profile chosen from failure analysis scored 67%, and two independently drawn legal
  profiles scored 64% and 63%. Against the analyst's choice the exact paired p-values
  were 0.55 and 0.34; pooled, 0.73 with the random draws numerically higher. The
  reasoned profile was genuinely different — telemetry confirmed it randomised robot
  pose across all 14 dimensions and camera intrinsics, while the drawn profile varied
  neither — and it still bought nothing.
---

# Make your prior earn it

Your reasoning produced a choice. Before treating the *reasoning* as the thing that
worked, check that the choice beat an arbitrary legal alternative at matched cost.

## How to apply

When you are about to credit a specific decision — which distribution to sample, which
failure mode to target, which hyperparameter to move — ask what an arbitrary legal
choice would have scored. If you cannot answer, you do not yet know that your reasoning
did any work. You may be measuring "we did something in-domain" and attributing it to
"we did the *right* thing in-domain".

This does not mean the reasoning was wasted. A reasoned choice and a random one often
cost the same to execute, so there is no reason to prefer the random one. It means the
*credit* belongs to the coarser mechanism until you can separate them.

## What it does not mean

It is not an argument for choosing arbitrarily. It is an argument for not claiming a
mechanism you have not isolated. If you have the budget, spend it on isolating the
mechanism rather than on a third variation of the same reasoned choice.
