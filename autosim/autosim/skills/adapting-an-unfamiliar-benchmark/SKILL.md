---
name: adapting-an-unfamiliar-benchmark
description: How to read a repository you have never seen and write down what it can do, so the result is checkable rather than plausible.
scope: general
confidence: methodological — a way to read a repository, not a measurement on one
evidence: |
  Distilled from reading several benchmarks, where the same failure kept appearing: an
  answer that was reasonable, unsupported, and only caught because something downstream
  checked it. A contract was written from the code that produced the data instead of from
  the data, and claimed a state dimension of one. A task list was typed out from a suite
  name where a glob over the task files would have been counted. A remark was written into
  a path field, so the existence check failed while looking like it should pass. Each was
  a good-faith reading and each was wrong, and none of them was wrong in a way that made
  the answer look doubtful.
---

# Adapting an unfamiliar benchmark

You are reading a repository to write down facts about it that another part of the system
will act on. Two things make this different from ordinary code reading. Everything you
assert may be checked against the filesystem, so an unsupported claim is a liability rather
than a guess. And the facts you write down become the shape of every experiment that
follows, so an error here is not corrected by a later stage -- it is inherited by one.

## Where to look for what

**A task's numbers are in its data, not in the code that produced it.** Observation and
action dimensions, camera names and resolutions, episode length and frame rate are
properties of the recorded trajectories. Read one. Code tells you the shape of the
author's intent; a recording tells you what actually happened, and when they disagree the
recording is what a policy will face.

**A task list should be countable.** If the repository holds one file per task, give a glob
over those files and let the system count the matches. A list you type cannot be checked,
and a suite name is not a task list -- suites contain tasks. Reserve a typed list for tasks
that genuinely exist nowhere but in code, and say in your evidence that you looked.

**A field named `path` holds a path.** Not a path with an explanation after it. A sentence
in a path field fails every existence check while looking like it ought to pass, which
turns a correct answer into a reported missing file. Explanations go in the field meant for
them.

**Prefer deriving over restating.** If the configuration already names every entity,
reading them costs one traversal and a table costs an edit every time a task is added. A
hand-kept table is a copy that will be wrong later; a derivation is right until the config
changes, and then it is right again.

## Ownership of anything outside the checkout

A file next to the checkout is not evidence about the checkout. A sibling project's
checkpoints sit on the same disk, have plausible names, and pass every cheap check there
is: the path is real, the file opens, the size is believable.

Ownership is decided by two questions, in this order. Does the repository itself name that
location -- a download script, a config key, a default path? Or does the artifact record
provenance about itself that resolves inside the checkout -- the task definition it was
generated from, the release tag it belongs to? If neither holds, the answer is that
ownership is not established, and saying so is correct even when the file is obviously the
right one to a human looking at it.

The same reasoning cuts the other way. A benchmark that downloads its data to a directory
of the user's choosing is not denying that the data is its own; the repository not pinning
a location is a fact about the repository, not about the data.

## Answering when the facts do not settle it

`unsupported` and `unknown` are answers, not failures, and choosing between them is worth
a moment. `unsupported` says the benchmark cannot do this -- a property of the benchmark.
`unknown` says the facts you were given do not settle it -- a property of your reading.
They lead somewhere different: the first closes a line of work, the second names what would
open it. When you write `unknown`, say what would settle it. That sentence is the most
useful thing you can produce, because it turns a question into a measurement.

Prefer `unknown` when the answer depends on something you cannot see from here. Whether a
benchmark can produce training data is the clearest case: that depends on the policies and
environments involved, not only on the text of the repository (see
`where-successes-come-from`).

## What a good answer looks like

Every claim traceable to something you read, and the trace written down. Claims about
behaviour marked as claims until something runs. Paths that resolve. A task list that can
be counted. And where you were unsure, the uncertainty stated plainly rather than smoothed
over -- an honest unknown costs one experiment, and a confident guess costs a run.
