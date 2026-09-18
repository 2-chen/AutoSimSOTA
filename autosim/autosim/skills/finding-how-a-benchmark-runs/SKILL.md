---
name: finding-how-a-benchmark-runs
description: Locate the entry points that train, evaluate and collect, from a repository you have not seen, without guessing from filenames.
scope: general
confidence: methodological — a way to read a repository, not a measurement on one
evidence: |
  Measured on RoboSynChallenge, asking for four entry points from a survey alone and then
  from a survey plus chosen file contents. From the survey alone, two of four were right:
  the evaluator was found, and the trainer was not — the repository ships six policy
  implementations and the wrong one was named. With the files read, three of four matched
  the entry points the system's hand-written backend uses, and the fourth was not wrong but
  higher level: `launch/run_task.sh`, which the repository's own README documents and which
  calls `scripts/run_env.py` — the very script the hand-written backend names.
---

# Finding how a benchmark runs

You are looking for the commands that train a policy, evaluate one, and produce new
trajectories. What makes this hard is not scarcity — a benchmark of any size has several
files that could plausibly be any of the three — but telling the one that runs from the ones
that merely mention running.

## Read the documentation before the filenames

The strongest evidence is a repository documenting its own invocation. A README that shows
`./launch/run_task.sh click_bell clear 2_1 --max_episodes 100` has told you the entry point,
its arguments and the order they go in — more than any amount of reading source will, and
it cannot be wrong about itself. Look there first, then in the script headers, which
frequently carry a usage line for the same reason.

A filename is the weakest evidence there is. `train_pytorch.py` is a trainer; so is
`train.py` in three other directories; which one this benchmark uses is not a fact about
names.

## When several implementations exist, ask what the repository ships

A benchmark that supports six policy families will contain six trainers, and nothing in the
source says which is *the* one. What settles it is what the repository actually publishes: a
suite of checkpoints named for one family is the benchmark telling you which family its own
results use. The same holds for datasets — a shipped collection in one format says which
loader is real.

Read what is present before deciding what is meant.

## A wrapper is a finding, not an error

A script that calls the real entry point is a legitimate answer, and often a better one —
it may chain stages, supply defaults the bare script needs, and be the invocation the
benchmark's authors actually intend. But say which level you are naming. A wrapper that
runs collection and then converts the result is one answer to two stages, and a caller who
does not know that will convert twice.

## Third-party code inside a repository is not that repository's

Projects vendor other projects: a checkout can contain an entire copy of a different
benchmark, with its own collector and its own trainer, under a directory named after a
policy family or a dependency. Those files are inside the tree and are not this
repository's entry points. When a path contains a segment naming another project, treat it
as evidence about that project.

## Prefer the specific file to the directory

`policy/act/scripts/train.py` is an answer. `policy/` is not, and neither is
`scripts/`. If you can only name a directory, you have found where to look rather than what
to run, and the honest answer says so.

## Say when a stage is absent

A benchmark that cannot produce new trajectories without a person has no collection entry
point, and reporting that is a finding rather than a gap. The distinction worth keeping: a
stage with no entry point because the benchmark does not have that capability is different
from a stage whose entry point you could not find. Say which one you mean, and say what
would settle it — see `where-successes-come-from` for the second case.
