---
name: more-of-the-same-saturates
description: Adding more data of the same distribution stops paying long before you run out of budget.
scope: general
confidence: high — stated by the authors of two separate data generators about their own systems, plus an independent scaling study
evidence: |
  MimicGen (arXiv:2310.17596), about its own generated data: "There is a large jump in
  performance from 200 to 1000, but not much from 1000 to 5000", with the figure caption
  reading "there are diminishing returns".

  DemoGen (RSS 2025) reports "performance saturation as more synthetic demos are added",
  attributed to visual mismatch — an independent system reaching the same conclusion on
  different tasks.

  Lin et al. (arXiv:2410.18647): generalisation saturates in demonstrations-per-environment.

  "The Curse of Precision" (arXiv:2607.23108, ManiSkill3 + Diffusion Policy): for a fixed
  target success rate the required data scales super-exponentially as task precision
  approaches the system's mechanical limit — so on high-precision tasks the saturating
  axis arrives much earlier.
---

# More of the same saturates

The first few hundred episodes of a new distribution move the number a lot. The next few
thousand move it very little.

## How to apply

Before asking for a larger collection, ask what the previous one bought. If round N
added data and round N+1 added more of the same and the number barely moved, the
distribution is saturated — more of it is the wrong purchase. The productive moves are
then to change *what* you collect (see `diversity-before-volume`), change *how it is
weighted*, or accept the plateau and report it.

Precision matters here: a task whose success hinges on millimetre clearance saturates far
earlier than a coarse one, so the same episode budget buys much less.

## What it does not mean

It does not mean additional data is worthless — it means equal-distribution additional
data is. It also does not tell you where the plateau is for *your* task; the numbers above
are other systems' tasks. Treat the plateau as something to detect from your own curve
rather than assume at a particular episode count.
