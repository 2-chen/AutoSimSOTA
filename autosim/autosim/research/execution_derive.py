"""Find how a benchmark runs, so the system does not have to be told.

The execution layer was one hand-written backend per benchmark: a file naming RoboSyn's
trainer, its evaluator, its collector and its converter, and a second file naming another
benchmark's. That is the same shape the adapters had before they were derived, and it has
the same cost -- a benchmark nobody has onboarded cannot run at all.

This asks the same two questions the rest of the system asks, in the same order. A survey
reports what is there; the model chooses what to read; the model answers with entry points
and invocations; and every path it names is checked against the filesystem. What it cannot
do is check that an entry point is the *right* one -- a repository with six policy families
has six trainers and all six exist. Settling that needs the stage to actually run, which is
the declared-to-verified ladder's job and not a static check's.

The separation matters for reading the output: `entrypoint` is a claim about this
repository, `available` is a claim about the benchmark's capability, and neither is
evidence until something runs.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .declarative_backend import artifact_pattern_problem, invocation_staging
from . import failure_memory, monitor
from .common import (atomic_json, bounded_run, digest, inspection_argv, inspection_mountpoint,
                     isolated_argv, now, object_digest, read_json, redact,
                     sanitize_model_text,
                     run_local_environment)
from .skills import skills_reference
from .survey import peek_many, summarise_for_model, survey


#: The stages a research loop needs, in the system's own words. A benchmark that cannot do
#: one of these says so; it is not a gap for the system to paper over. Which stages exist is
#: the same question `declaration.CAPABILITIES` asks, asked again here at the level of *how*
#: rather than *whether* -- a benchmark can declare it trains and still be unclear where.
STAGES: dict[str, str] = {
    "prepare_data": "turn the benchmark's shipped demonstrations into the form its trainer "
                    "reads. Not every benchmark has this step; one whose data is already in "
                    "that form does not.",
    "train": "fit a policy using the repository's native method (offline demonstrations, "
             "online simulator interaction, or both) and write a checkpoint",
    "evaluate": "score a policy on the benchmark's own initial states and its own success "
                "predicate",
    "collect": "produce new trajectories. If the benchmark cannot produce any without a "
               "person, report that here rather than naming a script that needs one.",
}

# Keep command verification and an otherwise-unspecified first training measurement on the
# same bounded workload. A zero default can make a perfectly verified trainer silently
# produce only an initialization checkpoint when the research loop later calls it.
TRAINING_VERIFICATION_STEPS = 1024

# A command derivation may execute simulator/trainer code while it verifies a draft. Keep
# retries bounded per controller action; a harder adaptation can be resumed with new
# evidence, but must not silently become dozens of GPU runs inside one "derive" action.
ARGV_DERIVATION_ATTEMPTS = 3
ARGV_DERIVATION_ROUNDS = 4

SYSTEM = (
    "You are reading a benchmark repository to find how each stage of a research loop is "
    "invoked. For every stage listed, name the entry point and how it is called: the file to "
    "run, the arguments that matter, and what file appears under the output directory when "
    "it succeeds -- a path or glob, not a description of one, because that is what will be "
    "looked for. If a stage "
    "has no entry point in this repository, say so and say why -- a missing stage is a "
    "finding, not a failure, and a stage you could not find is a different finding from a "
    "stage the benchmark does not have.\n\n"
    "Actively investigate native data acquisition even when existing demonstrations make "
    "it optional for the first baseline. Trace the documented wrapper to its actual action "
    "provider, configuration, required resources, saved episodes, converter and trainer "
    "loader. Preserve a source-supported collection route as an optimization option. Explain "
    "missing assets or human/policy prerequisites separately from absence of an entrypoint; "
    "an available source path still requires a bounded native collection-to-loader probe.\n\n"
    "A human-only collector does not imply that all autonomous training-data routes are "
    "impossible. Separately report simulator step/reset/success APIs, compatible policy "
    "loading, legal training-scene controls and whether a training-side success-filtered "
    "rollout wrapper could be implemented. Such a possible wrapper is NOT an available "
    "stage until it exists and its producer/consumer connection is source-supported; "
    "do not invent expert labels, confuse evaluation dumps with training data, or claim "
    "DAgger without an expert that can relabel learner-visited states.\n\n"
    "Name only files you have seen in the survey, the excerpts or the file contents, and "
    "give paths relative to the repository root. A file belonging to a different project "
    "that happens to be vendored inside this one is not this repository's entry point.\n\n"
    "Every invocation needs values the caller will supply, and some need values only this "
    "benchmark knows -- which policy implementation, which configuration, which dataset "
    "identifier. Declare those in `parameters`: one entry per value the invocation needs that "
    "is not a path the caller passes in, with the value you found and the evidence for it. A "
    "parameter whose value you inferred rather than read must say what you inferred it from.\n\n"
    "Choose a coherent path for the selected task and assets actually available: a "
    "demonstration trainer is not runnable when its required dataset is absent, while a "
    "native online trainer may be viable without demonstrations. Do not require every "
    "asset mentioned in a repository if this research path does not consume it. When two "
    "native train→score paths are both valid for the selected task, make the first path the "
    "one with fewer unavailable external prerequisites and a bounded short verification "
    "run; a slower alternate family can be recorded separately, but do not mix its data "
    "conversion with another family's trainer. This priority is for first measurement, "
    "not a claim that the simpler method is best.\n\n"
    "Stage availability is per implementation, not per repository-wide family: the missing "
    "dataset for an offline imitation trainer does not make a separate online simulator "
    "trainer unavailable. Classify a stage as `train` when its training branch performs "
    "policy updates and writes a policy artifact, even if that branch shares a file with an "
    "evaluation mode. `collect` means producing trajectories/data through an independent "
    "producer invocation. A trajectory, video, or HDF5 file emitted only as a side effect "
    "of evaluating a policy belongs to that evaluation stage; it does not establish a "
    "separate `collect` stage. Require source evidence that a claimed collector can run "
    "independently and produces data usable by a downstream consumer, rather than inferring "
    "collection from a recorded artifact extension or a shared entrypoint. An evaluation "
    "command that writes rollout videos or HDF5 trajectories is not a policy trainer and "
    "cannot satisfy an evaluator's checkpoint input. Do not mark training unavailable until each "
    "source-supported native training family has been checked independently.\n\n"
    "The optional `research_context` supplies a caller-selected task and already surveyed "
    "assets. Use it to rank native paths, but verify any task or asset claim against the "
    "repository; do not invent an entry point just to satisfy the context.\n\n"
    "If `research_context.rejected_handoffs` names a trainer/evaluator pairing whose "
    "checkpoint save/load formats did not match, inspect a different native family or "
    "report the score path unavailable. Never join unrelated trainer and evaluator "
    "families merely because both use `.pt` files.\n\n"
    "A parameter's `value` is the literal the command substitutes -- one token, no "
    "explanation. If you cannot settle on a single literal, that is a finding: say so in "
    "`why` and mark the stage unavailable rather than writing a description into the value.\n\n"
    "An invocation is not only its arguments. Say where the command must run "
    "(`working_directory`, usually the repository root) and what its environment must "
    "contain (`environment`, a map of variable to value): a package whose top level has no "
    "`__init__.py` cannot be imported from outside its parent, and a renderer has to be told "
    "to run headless. Use {repo} for the checkout's path.\n\n"
    "The four named stages are compatibility capability questions, not a mandatory order. "
    "When this repository genuinely needs another discrete runnable stage (for example a "
    "rollout, policy update, export, or generated-data step), add it under `stages` with a "
    "short identifier plus `role`, `why_required`, and `evidence`. Do not add a stage that "
    "the repository does not actually provide. "
    "If the repository exposes a real video/GIF recording command, add the optional "
    "`record_demo` stage with its actual invocation and artifact path. Do not invent a "
    "recording flag; absence of this capability is a finding. An extra stage, including "
    "`record_demo`, must have `available: true`, `role`, `why_required`, and `evidence`; "
    "if these cannot be justified, omit that extra stage entirely. "
    "If ordering matters, add an optional "
    "`execution_graph` with `nodes`: each has `id`, `role`, `depends_on` and optional "
    "`bindings` mapping input names such as `dataset` or `checkpoint` to an upstream node. "
    "Set `score_target` to the id of the evaluate-role node whose metric is primary. "
    "Every graph node id must exactly match an available stage name. If the graph has an "
    "evaluate-role node, `score_target` is mandatory. Omit the graph when those facts "
    "cannot be established. Only declare an edge supported by repository evidence; do "
    "not infer a fixed prepare/train/evaluate order.\n\n"
    "Keep `reasoning` to at most 120 words and keep parameter evidence concise; the stage "
    "objects, not a long narrative, are the required result. Return every named stage, even "
    "when unavailable, and do not omit a key because it is optional. Return one JSON object "
    "of the form {\"stages\": {\"<stage>\": {\"available\": true or "
    "false, \"entrypoint\": \"<path>\", \"invocation\": \"<how it is called>\", "
    "\"working_directory\": \"<where it runs, e.g. {repo}>\", "
    "\"environment\": {\"<VAR>\": \"<value>\"}, \"artifact\": "
    "\"<a path or glob, relative to the output directory, that exists only on success>\", "
    "\"level\": \"entry point, or the path this wraps\", "
    "\"parameters\": [{\"name\": \"<as the invocation spells it>\", \"value\": \"<the value "
    "for this repository>\", \"evidence\": \"<where that value comes from>\"}], "
    "\"evidence\": \"<required for available collect: source for independent producer and "
    "reusable data output>\", "
    "\"why\": \"<when unavailable>\"}}, \"reasoning\": \"<how you found them>\"}."
)

TRIAGE_SYSTEM = (
    "You are deciding what to read. A benchmark repository has been surveyed; the survey "
    "reports file names, sizes, and the signals found in each source, but not their "
    "contents. Return one JSON object {\"files_to_read\": [\"<path>\", ...]} naming up to "
    "twelve repository-relative files that would show how each of these stages is invoked: "
    "prepare_data, train, evaluate, collect. Prefer a file that runs a stage over one that "
    "merely mentions it, prefer this repository's own code over third-party code vendored "
    "inside it, and prefer documentation that shows an invocation. Do not ask for a file "
    "only to confirm its name. Use `research_context` to include the native trainer and "
    "evaluator for the selected task and available assets. When no demonstration dataset "
    "is present, prioritize reading at least one native online simulator-training "
    "script and its invocation documentation if this repository has one; an offline "
    "trainer alone cannot establish a runnable research path."
)


#: What a stage function may read. The system's vocabulary, not the benchmark's: a stage is
#: told what it is being asked to do, and it says how this particular benchmark expresses
#: that. A benchmark needing something outside this list cannot be driven, which is a real
#: limit and is reported as one rather than papered over with a free-form command string.
INPUT_VOCABULARY: dict[str, str] = {
    "python": "the interpreter to run with",
    "repo": "absolute path to the benchmark checkout",
    "task": "the task name this stage is for",
    "setting": "which of the benchmark's own configurations to run under, when it names "
               "them (a randomisation mode, a scene variant); a value the benchmark "
               "defines, which the caller selects",
    "dataset": "absolute path to training data, when the stage consumes it",
    "checkpoint": "absolute path to a policy checkpoint, when the stage consumes one",
    "output": "absolute directory the stage should write into",
    "steps": "explicit optimizer-update budget, as an int; verification supplies its bounded "
             "probe budget, formal research must choose its own learning work. Do not silently "
             "reuse a verification cap or treat native epochs as updates",
    "episodes": "target number of completed evaluation episodes, as an int; this is a "
                "count of episodes, not a number of environment steps",
    "seed": "the seed, as an int",
    "device": "the device this stage should run on, spelled the way the framework does: "
              "`cuda`, `cpu`, `cuda:1`. Not an index -- a benchmark doing "
              "`torch.device(cfg.device)` raises TypeError on the integer 0",
    "device_index": "the ordinal of the device the caller selected, as an int, for a "
                    "benchmark whose own flag takes an index (`--device_id 0`)",
    #: The channel a research proposal travels down. A proposal names settings the
    #: benchmark's own declared space contains -- a loss scale, an epoch count, a policy
    #: family -- and each benchmark spells those differently: one as a command-line flag,
    #: one as a config override, one as a field in a file it writes. The function is where
    #: that spelling lives, so the layer that decides *what* to vary never learns it.
    "settings": "a map of setting name to value, from the benchmark's declared space; "
                "express each in whatever way this benchmark's invocation takes them, and "
                "ignore the ones the stage does not use",
    "extra": "a dict of anything the caller was told to pass through; may be empty",
}


def blank_inputs() -> dict[str, Any]:
    """Every key in the vocabulary, with a neutral value.

    The vocabulary is fixed and known, so a caller that supplies part of it is asking the
    generated function for a `KeyError` it cannot act on -- and the error surfaces inside a
    function nobody can read, naming a key that is part of the system's own vocabulary. A
    stage that reads `i["setting"]` is reading a name the system defined; not finding it says
    nothing about the benchmark.
    """
    inputs: dict[str, Any] = {}
    for key in INPUT_VOCABULARY:
        if key in ("settings", "extra"):
            inputs[key] = {}
        elif key in ("steps", "episodes", "seed", "device_index"):
            inputs[key] = 0
        else:
            inputs[key] = ""
    return inputs


ARGV_SYSTEM = (
    "You write one pure Python function that returns the command line for one stage of a "
    "research loop on a benchmark you have been shown.\n\n"
    "The function takes one argument `i`, a dict whose keys are exactly the ones listed in "
    "`inputs_available`, and returns a list of strings: the argv, starting with the program "
    "to run. Build it from the entry point and the invocation you were given.\n\n"
    "Constraints, enforced by a validator that rejects the draft outright:\n"
    "* exactly one function, named for the stage, taking one positional argument\n"
    "* no imports, no decorators, no nested functions, no recursion, no exception handling\n"
    "* no attribute access except the methods named below; subscripts and comprehensions "
    "otherwise\n"
    "* the only methods you may call are keys, values, items, get, copy, index, count, "
    "append, extend, insert, remove, pop, sort, reverse, update, join, split, splitlines, "
    "strip, lstrip, rstrip, replace, startswith, endswith, lower, upper, title, capitalize, "
    "zfill, isdigit, add, union, intersection, difference\n"
    "* the only callable names are int, float, str, len, sum, min, max, range, enumerate, "
    "zip, abs, list, dict, tuple, set, sorted, bool, all, any\n"
    "* no f-strings and no `str.format`; build strings with + and str()\n"
    "* read an index that is a variable with `d[k]`, and a fixed key with `d[\"k\"]`\n"
    "* build every path from `i[\"repo\"]` -- `i[\"repo\"] + \"/<subdir>/<entry>.py\"` -- "
    "never a path relative to the working directory\n\n"
    "The last of those is not style. Where a stage runs is a variable this system revises, so "
    "a relative path means one file from the checkout and a different one from whatever "
    "subdirectory the stage happens to be running in -- and a checkout with a package named "
    "after its own subdirectory has two files with the same relative path. A round went to "
    "that: the working directory had been revised and the argv had not, and the program "
    "reported a file that was not there, twice, from the same command. An absolute path built "
    "from `i[\"repo\"]` cannot be wrong this way.\n\n"
    "Every element of the returned list must be a string.\n\n"
    "There are two channels and they behave differently. Getting them the wrong way round is "
    "the most common way this function fails, so read this part twice.\n\n"
    "**Values already settled for this repository** are given under "
    "`values_already_settled_for_this_repository`. They are final. Write each one into the "
    "argv as a literal, exactly as given -- `\"policy=bc_transformer_policy\"`. They are NOT "
    "in `i`, so reading them from `i` raises KeyError and wastes the attempt.\n\n"
    "**`i[\"settings\"]` is a map of research variables**, and it is sparse: any subset of "
    "the space's names may be there or absent on any given call. Apply it by appending, so "
    "that the value in the map replaces the repository's own default rather than the "
    "caller having to supply every setting at once:\n\n"
    "    argv = [i[\"python\"], \"-m\", \"pkg.train\", \"benchmark_name=<the task this runs>\"]\n"
    "    for key in sorted(i[\"settings\"]):\n"
    "        argv = argv + [key + \"=\" + str(i[\"settings\"][key])]\n"
    "    return argv\n\n"
    "Do not also hard-code a CLI option in the base argv when that same option can arrive "
    "through `i[\"settings\"]` or `i[\"extra\"]`: append-only overrides would then put two "
    "different values for one option in the command. Keep the repository's actual defaults "
    "implicit where possible, and make each research variable enter the argv exactly once. "
    "Before returning, check the complete argv for repeated long-option names (treat `-` and "
    "`_` as the same spelling); a fixed value plus a dynamic override is an ambiguous command, "
    "not a last-value-wins policy.\n\n"
    "Never index `i[\"settings\"]` by a fixed name -- `i[\"settings\"][\"train.n_epochs\"]` "
    "raises KeyError on every call that does not happen to vary that setting, which is most "
    "of them. Iterate it and append; a setting that is absent then simply uses the default "
    "you already wrote. The syntax of the appended override is whatever this benchmark takes: "
    "`name=value` for a config system, `--name` then the value for a flag parser.\n\n"
    "For a training stage, the verifier supplies `i[\"steps\"]` as a bounded positive "
    "execution budget. Bind it to the repository's native total-work option in the function "
    "source. If the native unit is an epoch rather than a step, derive a conservative "
    "steps-to-epochs conversion from the repository's dataset/batch semantics; a 1024-step "
    "request must not silently become 1024 full epochs. The native work limit must rise "
    "when the caller's budget doubles, but it need not change for a one-step increment. "
    "For direct step controls, e.g. `argv = argv + [\"--total_steps=\" + "
    "str(i[\"steps\"])]`. For compatibility, the runner substitutes only an exact `{steps}` "
    "or `i['steps']` token when it is the complete argv value or the complete value of "
    "`key=value`; it does not evaluate Python text, prose, or partial templates. A full-run "
    "total-work value shown in the repository is NOT a settled repository parameter: do not "
    "repeat it under `parameters` as a fixed value. `parameters` holds only repository values; "
    "caller budgets are read from `i`. The verifier checks that "
    "the emitted native option changes when its caller budget changes. Choose batch/parallelism "
    "settings that permit at least one actual optimizer update within that budget. A fresh "
    "initialization checkpoint "
    "after zero iterations is not a verified training command. A hard-coded full training "
    "example is not a safe smoke test. "
    "Do not append `i[\"device\"]`, `i[\"seed\"]`, or `i[\"output\"]` unless the "
    "entry point demonstrably accepts the corresponding option.\n\n"
    "For an evaluation stage, `i[\"episodes\"]` is a target count of completed episodes, "
    "not a step count. Read the evaluator and task horizon in source: do not pass this value "
    "directly to an option such as `num_eval_steps` unless the source proves that option is "
    "measured in episodes. Choose the native horizon/parallelism needed to complete at least "
    "one episode (and the requested count when the interface permits), and treat an explicit "
    "zero-completed-episode output as a failed verification, not a valid score.\n\n"
    "Every key in `inputs_available` is always present, so `if \"checkpoint\" in i` is always "
    "true and passes an empty value. Test the value, not the key: `if i[\"checkpoint\"]:`\n\n"
    "**`i[\"python\"]` is an interpreter, and it is the program only when the entry point is "
    "Python.** The first element of the argv is what gets executed. Writing "
    "`[i[\"python\"], i[\"repo\"] + \"/path/to/eval.sh\", ...]` asks the Python interpreter to "
    "parse a shell script, and the program answers `SyntaxError: invalid syntax` on the "
    "script's second line -- `set -euo pipefail` -- which reads as a broken script rather than "
    "a wrong invocation. A `.sh` entry point is run with `bash`; a `.py` entry point with "
    "`i[\"python\"]`; an entry point that is executable on its own can be run directly. This "
    "is not a stylistic choice: when the first element is the wrong program, the failure is a "
    "syntax error from an interpreter reading a script as its own language, and no working "
    "directory, environment variable or parameter changes what the first element means. A "
    "stage lost its whole round budget to it before this was said plainly.\n\n"
    "**A path the caller supplies goes into the slot for it, and the repository's own name "
    "for that thing is the fallback.** The two are different spellings of one argument: the "
    "caller hands over an absolute path to a checkpoint it just trained, and the repository's "
    "convention is a name resolved under its own `checkpoints/` directory. Write the caller's "
    "value when there is one and the repository's when there is not:\n\n"
    "    ckpt = i[\"checkpoint\"] or \"<the repository's own name for it>\"\n\n"
    "Writing only the repository's name makes a policy this system trained impossible to "
    "evaluate -- the stage runs, loads whatever that name resolves to, and reports on the "
    "wrong weights, or reports that the directory does not exist. An evaluator was handed a "
    "checkpoint by the caller and ignored it for sixty-five rounds, because the derived "
    "`ckpt_name` parameter and the caller's `checkpoint` input never met. The same applies to "
    "`i[\"dataset\"]` and to any other path in the vocabulary.\n\n"
    "A name in an override must be a key the program's configuration declares, and a name "
    "that describes the value is not one. A key that reads like a sensible description of "
    "what you want is the guess, not the answer: the key is written in the program's own "
    "source and config, under whatever word its author chose. Read the keys the program "
    "declares and use one of them; a config system refuses an invented key outright, and the "
    "refusal names the key you guessed rather than the one you wanted.\n\n"
    "A flag name must come from the program, not from the shape of the invocation you were "
    "handed. Argparse and its relatives accept any unambiguous prefix of a real flag, so an "
    "invented `--task` is silently matched to `--task_id` and the program then complains "
    "about the *value* -- which sends you looking for a value to change when the flag name "
    "was wrong. When the program prints a usage line, treat every flag in it as the list of "
    "what exists and use only names that appear there.\n\n"
    "Where the benchmark's overrides are parsed by a config grammar -- hydra, omegaconf, "
    "`name=value` on the command line -- an unquoted value is not a string, it is a token "
    "that grammar has to lex, and a path is often not lexable: `output_dir=/tmp/a b` is "
    "refused at the equals sign, and so is any path containing a non-ASCII character. Quote "
    "the value of every `name=value` override whose value is a path, whether it came from `i` "
    "or was written into the function as a settled literal -- `key + '=\"' + value + '\"'` "
    "and `'key=\"<a path>\"'` alike. Quote only the value, never the key, and leave a "
    "value that is already quoted alone. Restricting this to values from `i` cost a round: "
    "the literal was left bare and the lexer refused it.\n\n"
    "`i[\"output\"]` is the directory the caller wants results in, and a benchmark may have "
    "no way to be told it. Many do not have a config key for it at all: the directory is "
    "computed from other config values and the program's own working directory, and "
    "`output_dir=` is refused by the config system because no such key exists -- correctly. "
    "Do not invent a key for it. If the benchmark declares no way to "
    "choose where it writes, leave the output directory out of the command and say so in the "
    "reasoning; the caller looks for the artifacts where the benchmark put them, which is "
    "usually relative to where it ran. A guessed key costs a round and teaches nothing.\n\n"
    "A dataset that holds an open file handle cannot cross a process boundary, so a program "
    "whose data is in HDF5 or a similar format fails the moment its loader starts a worker: "
    "`h5py objects cannot be pickled`, raised from inside multiprocessing. The command is "
    "right and a value is missing -- the loader has to run in-process. Find the setting that "
    "names the worker count and pass zero for it, and pass it for *every* loader that reads "
    "that data: training and evaluation usually have separate ones, and fixing the first "
    "moves the failure to the second rather than removing it. Both had to be passed before "
    "a single epoch could finish.\n\n"
    "If the command was run and refused, you may correct the values under "
    "`values_already_settled_for_this_repository`: some failures are the function's fault and "
    "some are a value's, and a program saying it cannot find something is telling you which. "
    "Write the corrected value into the argv as well as returning it under `parameters`, "
    "since the argv is what runs. Return the whole set, not just the changed entry.\n\n"
    "If the latest real native failure requires environment/staging/invocation repair "
    "rather than another argv, you may immediately return only "
    "{\"invocation_handoff\":{\"reason\":\"evidence-based explanation\","
    "\"native_evidence_ref\":\"exact latest supplied native evidence reference\"}}. "
    "This yields to diagnosis/revision without replaying a command. It is allowed only "
    "after a failed native verification in this generation, not before any command ran. "
    "A handoff is a repair request, never successful verification.\n\n"
    "Return only {\"source\": \"<the function>\", \"reasoning\": \"<one or two "
    "sentences, including anything the vocabulary could not express>\", "
    "\"shape_was_accepted\": true or false, "
    "\"parameters\": {\"<name>\": \"<corrected value>\"} (optional)}.\n\n"
    "`shape_was_accepted` is about the last run, when there was one: did the program get "
    "past reading its arguments and fail at something else, or did it refuse the arguments "
    "themselves? A usage message, an unrecognized flag, a missing required argument -- those "
    "are a refusal. A traceback from inside the program's own work, a missing module, a file "
    "it could not open -- it read the arguments and went on. Answer **null** when the command "
    "has not been run yet. This is what decides whether the next attempt keeps the shape and "
    "varies the values, or writes a different shape altogether, so a wrong answer costs the "
    "attempts that follow it. A parsed-but-failed command is not immutable: if fixing the "
    "failure requires changing the executable, wrapper or command shape, return the new "
    "source and explain that replacement in reasoning, grounded in the supplied native "
    "failure and source. The executor records the old/new identities and requires a fresh "
    "native verification; it will not silently discard your replacement. Preserve the "
    "task, checkpoint input and evaluation protocol."
)


#: Suffixes that mean "this is a directory of benchmark data" rather than a directory of
#: code. A program whose data is absent from its own checkout will say so by naming a path
#: it could not open, and the question that leaves is where that data actually is -- which
#: is answerable by looking, and is not answerable by guessing.
DATA_SUFFIXES = (".hdf5", ".h5", ".hdf", ".npy", ".npz", ".pt", ".pth", ".pkl", ".pddl",
                 ".bddl", ".lerobot", ".tar", ".parquet")


def data_near(repo: Path, *, levels: int = 2, limit: int = 500) -> dict[str, Any]:
    """Directories near the checkout that hold data files, and what those files are called.

    LIBERO is why this exists. Its demonstration loader joins `{folder}/{problem}/{task}.hdf5`
    with `folder` defaulting into its own checkout, where no dataset was ever shipped; the
    data is in a sibling directory of the checkout. Nothing in the repository says so, so a
    derivation reading the repository cannot reach it -- but the answer is on the disk, and
    a system that only asks the model and never looks cannot get there at all.

    Reported as a survey, not as an answer: names, counts and one example each, so the model
    decides which directory is the one the failing path was pointing at. It costs one
    `os.walk` of bounded depth and it is only run when a stage has already failed.
    """
    found: dict[str, Any] = {}
    root = Path(repo).resolve()
    # Never the filesystem root: a checkout under a shallow path (`/tmp/x`) has `/` among its
    # parents, and walking `/` is both enormous and full of files this process may not stat.
    # That is not hypothetical -- it raised `PermissionError` on `/etc/xrdp/key.pem` from
    # inside a *diagnostic*, which ended the run it was there to help.
    parents = [parent for parent in root.parents if len(parent.parts) > 1][:levels]
    # The immediate children that hold data, not the first three names found: a directory
    # holding a whole benchmark under `libero/libero_10/` reports three `*.hdf5` names that
    # say nothing about which benchmark they belong to, and a reader shown only those
    # concludes -- correctly, from what it was shown -- that it is something else.
    structure: dict[str, list[str]] = {}
    roots_children: list[Path] = []
    for parent in parents:
        try:
            children = sorted(p for p in parent.iterdir() if p.is_dir())
        except OSError:
            continue
        for child in children:
            if child == root or child.name.startswith("."):
                continue
            count, examples = 0, []
            # `os.walk` with an `onerror` that ignores: a directory this process cannot read
            # is a directory it has nothing to say about, not a reason to stop looking.
            for directory, _, names in os.walk(child, onerror=lambda _: None):
                if count >= limit:
                    break
                for name in sorted(names):
                    if Path(name).suffix not in DATA_SUFFIXES:
                        continue
                    count += 1
                    relative = Path(directory, name).relative_to(child)
                    if len(relative.parts) > 1:
                        # The immediate child that holds the data. This is what tells one
                        # benchmark's directory from another's, and the file names alone do
                        # not: `datasets/` reported three `demo_v15.hdf5` and read as a
                        # robomimic directory, when it holds every suite LIBERO ships.
                        holder = relative.parts[0]
                        if holder not in structure.setdefault(str(child), []):
                            structure[str(child)].append(holder)
                    if len(examples) < 3:
                        examples.append(str(Path(directory, name).relative_to(parent)))
                    if count >= limit:
                        break
            if count:
                found[str(child)] = {
                    "files": count,
                    "subdirectories_holding_data": structure.get(str(child), [])[:8],
                    "examples": examples}
                roots_children.append(child)
    if found:
        # The one fact the survey was missing, and the one that matters: what the program
        # asked for is often a single file with a name it has already told us.
        found["_how_to_use_this"] = (
            "`files_the_program_could_not_open` is empty when nothing failed to open. When "
            "it is not, that is the answer -- a file with the missing name exists at that "
            "path, and the parameter to redirect the program at it is what is wanted.")
    return found


#: A path the program said it could not open, as it appears in an error: quoted, and with a
#: data-ish suffix so that a quoted word in prose is not mistaken for one.
_MISSING_PATH = re.compile(
    r"""['"]([^'"\n]*\.(?:hdf5|h5|npy|npz|pt|pth|pkl|bddl|pddl|json|csv|parquet|lerobot|tar))['"]""",
    re.IGNORECASE)


def files_the_program_could_not_open(failure: str, *, roots: list[Path],
                                     limit: int = 3) -> dict[str, list[str]]:
    """Find files with the name of the one the program said it could not open.

    This is what a person does with the error, and it is a much stronger signal than any
    survey: the program has already said exactly which file it wanted. LIBERO's trainer
    printed `unable to open file: name = '.../LIVING_ROOM_SCENE2_..._demo.hdf5'`, and a
    survey that reported the first three file names in each nearby directory led the reviser
    to conclude -- from those three names, correctly -- that the directory held a different
    benchmark's data, when the file it was asking about was two levels further down.

    So: take the name the error carries, and look for that name.
    """
    wanted: list[str] = []
    for match in _MISSING_PATH.finditer(failure):
        name = Path(match.group(1)).name
        if name and name not in wanted:
            wanted.append(name)
    found: dict[str, list[str]] = {}
    for name in wanted[:limit]:
        hits: list[str] = []
        for root in roots:
            for directory, _, names in os.walk(root, onerror=lambda _: None):
                if name in names:
                    hits.append(str(Path(directory, name)))
                    break
        if hits:
            # Two roots can reach the same file -- a parent and the directory itself -- and
            # a list that shows the answer twice reads as two copies of the data.
            found[name] = list(dict.fromkeys(hits))[:4]
    return found


def missing_roots(repo: Path) -> list[Path]:
    """Where to look for a file the program named: the checkout and its neighbours."""
    root = Path(repo).resolve()
    parents = [parent for parent in root.parents if len(parent.parts) > 1][:2]
    roots = [root]
    for parent in parents:
        try:
            roots.extend(child for child in sorted(parent.iterdir())
                         if child.is_dir() and not child.name.startswith(".")
                         and child != root)
        except OSError:
            continue
    return roots


#: A key in a configuration a program printed about itself: `'folder': null,` from a Python
#: dict dump, or `folder: null` from YAML. The second form also matches prose ("note: this"),
#: so it is only consulted when the output already looks like a configuration.
_PRINTED_KEY = re.compile(r"""^\s*['"]?([A-Za-z_][A-Za-z0-9_.]*?)['"]?\s*:""")


#: A key whose printed value is empty: `'folder': null`, `folder: none`. A configuration
#: that prints its own keys *and* which of them are unset is telling the caller which knobs
#: are its to turn, and that is the question a failing override is asking.
_UNSET = re.compile(
    r"""^\s*[{[]?\s*['"]?([A-Za-z_][A-Za-z0-9_.]*?)['"]?\s*:\s*"""
    r"""(?:null|None|''|""|~)\s*,?\s*$""")


def unset_keys_the_program_printed(output: str, *, limit: int = 60) -> list[str]:
    """The keys the program printed as having no value.

    LIBERO's own config says `folder: null # use default path` -- the key that was wanted, and
    the fact that it is the caller's to set, in one line. A list of a hundred key names does
    not say that; the model reads `folder` as opaque and reaches for `data.dataset_path`
    because the name describes the value. Being unset is what makes a key a candidate.
    """
    found: list[str] = []
    for line in output.splitlines():
        match = _UNSET.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.endswith(("Error", "Exception", "Warning")) or len(name) > 40:
            continue
        if name not in found:
            found.append(name)
    return found[:limit]
#: A program indexing a table by a name it was handed: `TASK_CONFIGS[ckpt_setting]`.


#: How many things the reviser may look at before it has to answer. Bounded because each is a
#: round trip and a stage's budget is finite; generous because the alternative is guessing,
#: and guessing is what this replaced.
INSPECTIONS_PER_REVISION = 8

#: How much of each answer to keep. A file's first twenty thousand characters hold every
#: configuration table this has needed; a directory listing is capped because a checkout can
#: hold fifty thousand files and what is wanted is the names at one level.
FILE_BYTES, LISTING_ENTRIES, COMMAND_BYTES = 20_000, 200, 6_000

#: Commands with ordinary write semantics are refused before the read-only sandbox is built.
#: The OS namespace, not this vocabulary, is the security boundary.
_WRITES = frozenset({"rm", "mv", "cp", "dd", "truncate", "chmod", "chown", "ln", "mkdir",
                     "rmdir", "touch", "tee", "git", "pip", "conda", "apt", "sudo", "make"})

_INSPECTION_PRIVATE_PARTS = frozenset({
    "data", "dataset", "datasets", "demo", "demos", "demonstration", "demonstrations",
    "trajectory", "trajectories", "rollout", "rollouts", "recording", "recordings",
    "video", "videos", "checkpoint", "checkpoints", "weight", "weights", "secret",
    "secrets", "credentials",
})
_INSPECTION_PRIVATE_SUFFIXES = frozenset({
    ".pt", ".pth", ".ckpt", ".safetensors", ".pkl", ".pickle", ".h5", ".hdf5",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".npy", ".npz", ".parquet",
    ".tfrecord", ".key", ".pem",
})
_INSPECTION_ENV = frozenset({
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "MUJOCO_GL", "PYOPENGL_PLATFORM",
    "EGL_PLATFORM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "LD_LIBRARY_PATH",
})


def _private_inspection_path(path: Path, *, repo: Path) -> bool:
    """Keep task assets, model weights, media, and credential files out of LLM context."""
    try:
        relative = path.resolve(strict=False).relative_to(repo.resolve(strict=True))
    except (OSError, ValueError):
        return True
    parts = [part.lower() for part in relative.parts]
    if any(part in _INSPECTION_PRIVATE_PARTS for part in parts):
        return True
    name = path.name.lower()
    return (path.suffix.lower() in _INSPECTION_PRIVATE_SUFFIXES or name.startswith(".env") or
            any(marker in name for marker in ("credential", "secret", "token")))


def _inspection_environment(environment: dict[str, str], *, repo: Path) -> dict[str, str]:
    """Pass only non-secret diagnostic settings into the networkless inspection sandbox."""
    checkout = repo.resolve(strict=True)
    mountpoint = inspection_mountpoint(checkout)
    path_roots = (Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"),
                  Path("/lib64"), Path("/opt"), Path("/nix/store"))

    def resolve_path(value: str) -> Path:
        candidate = Path(value or ".").expanduser()
        if not candidate.is_absolute():
            candidate = checkout / candidate
        return candidate.resolve(strict=False)

    def visible_path(value: str) -> bool:
        resolved = resolve_path(value)
        return resolved.is_relative_to(checkout) or any(
            resolved == root or resolved.is_relative_to(root) for root in path_roots)

    paths: list[str] = []
    for raw in environment.get("PATH", "").split(os.pathsep):
        if not raw or not visible_path(raw):
            continue
        resolved = resolve_path(raw)
        mapped = str(mountpoint) + str(resolved.relative_to(checkout)) \
            if resolved.is_relative_to(checkout) else str(resolved)
        if mapped not in paths:
            paths.append(mapped)
    for relative in (".venv/bin", "venv/bin"):
        candidate = checkout / relative
        if candidate.is_dir():
            mapped = str(mountpoint / relative)
            if mapped not in paths:
                paths.insert(0, mapped)
    result = {
        "PATH": os.pathsep.join(paths or ["/usr/local/bin", "/usr/bin", "/bin"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for key in _INSPECTION_ENV:
        value = environment.get(key)
        if value is None or "\x00" in value or "\n" in value or "\r" in value:
            continue
        if key == "LD_LIBRARY_PATH":
            safe = [item for item in value.split(os.pathsep) if item and visible_path(item)]
            if safe:
                result[key] = os.pathsep.join(
                    str(mountpoint) + str(resolve_path(item).relative_to(checkout))
                    if resolve_path(item).is_relative_to(checkout) else str(resolve_path(item))
                    for item in safe)
        else:
            result[key] = value
    pythonpath = environment.get("PYTHONPATH", "")
    safe_pythonpath = []
    for item in pythonpath.split(os.pathsep):
        if item and visible_path(item):
            resolved = resolve_path(item)
            safe_pythonpath.append(str(mountpoint) + str(resolved.relative_to(checkout))
                                   if resolved.is_relative_to(checkout) else str(resolved))
    if safe_pythonpath:
        result["PYTHONPATH"] = os.pathsep.join(safe_pythonpath)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM", "EGL_PLATFORM"):
        # Explicit stage values above already win; these entries are retained only when
        # they came through the allowlist and contain no control characters.
        if key in result and not re.fullmatch(r"[A-Za-z0-9_.+-]{1,80}", result[key]):
            result.pop(key, None)
    return result


def _inspection_command(command: str, *, repo: Path) -> list[str]:
    """Parse one diagnostic argv; never invoke a shell or accept shell metacharacters."""
    if len(command) > 2048 or re.search(r"[;&|<>`$\n\r]", command):
        raise ValueError("inspection accepts one argv only; shell syntax is not allowed")
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid inspection argv: {exc}") from exc
    if not argv or len(argv) > 32:
        raise ValueError("inspection command must contain 1–32 arguments")
    executable = Path(argv[0]).name.lower()
    readonly_exception = (
        (executable == "conda" and argv[1:] in (["env", "list"], ["info"])) or
        (executable == "git" and (
            (argv[1:2] == ["status"] and all(
                arg in {"--short", "--branch", "-sb"} for arg in argv[2:])) or
            (argv[1:2] == ["rev-parse"] and argv[2:] in (
                ["HEAD"], ["--show-toplevel"], ["--abbrev-ref", "HEAD"])) or
            (argv[1:2] == ["diff"] and argv[2:] in (["--stat"], ["--name-only"]))
        )) or
        (executable == "pip" and argv[1:2] in (["list"], ["show"]))
    )
    if executable in _WRITES and not readonly_exception:
        raise ValueError(f"{executable!r} changes things; an inspection only looks")
    if executable in {"sh", "bash", "dash", "zsh", "ksh", "fish", "command", "eval"}:
        raise ValueError("shell interpreters and builtins are not inspection commands")
    if executable in {"find", "xargs", "tree"} or any(arg in {"-R", "-r", "--recursive"}
                                                for arg in argv[1:]):
        raise ValueError("recursive inspection commands are not allowed")
    if executable in {"rg", "grep", "egrep", "fgrep", "ag"} and len(argv) < 3:
        raise ValueError("search commands must name an explicit checkout path")

    python_like = executable in {"python", "python3", "pypy", "pypy3"} or \
        executable.startswith("python3.")
    if python_like:
        if argv[1:] in (["--version"], ["-V"]):
            return argv
        if argv[1:] in (["--help"], ["-h"]):
            return argv
        if "-c" in argv or "-m" in argv or not any(arg in {"--help", "-h", "--version"}
                                                    for arg in argv[1:]):
            raise ValueError("Python inspection is limited to version or a repository script's help")
        scripts = [Path(arg) for arg in argv[1:] if not arg.startswith("-")]
        if len(scripts) != 1 or not scripts[0].suffix == ".py":
            raise ValueError("Python help must name exactly one .py script in the checkout")

    checkout = repo.resolve(strict=True)
    python_script = None
    if python_like:
        python_script = next((Path(arg) for arg in argv[1:] if not arg.startswith("-")), None)
    for index, raw in enumerate(argv):
        value = raw.split("=", 1)[1] if raw.startswith("-") and "=" in raw else raw
        candidate = Path(value).expanduser()
        path_like = (candidate.is_absolute() or "/" in value or value.startswith(".") or
                     (not raw.startswith("-") and (checkout / candidate).exists()) or
                     (python_script is not None and candidate == python_script))
        if not path_like:
            continue
        if not candidate.is_absolute():
            candidate = checkout / candidate
        resolved = candidate.resolve(strict=False)
        if index == 0 and any(resolved == root or resolved.is_relative_to(root)
                              for root in (Path("/usr"), Path("/bin"), Path("/sbin"),
                                           Path("/lib"), Path("/lib64"), Path("/opt"),
                                           Path("/nix/store"))):
            continue
        if not resolved.is_relative_to(checkout):
            raise ValueError("inspection paths must stay inside the checkout")
        if _private_inspection_path(resolved, repo=checkout):
            raise ValueError("inspection of datasets, checkpoints, media, or secrets is refused")
    return argv


def inspect(request: dict[str, Any], *, repo: Path, directory: Path,
            environment: dict[str, str], timeout: int = 90) -> dict[str, Any]:
    """One thing the reviser asked to see, answered from the checkout or sandbox.

    Three shapes, because there are three questions a failed command makes you ask:

    * `{"look_at": "<path>"}` -- what does that source/config file say. A config table's keys, a script's
      argument list, where a path is built from.
    * `{"look_at_dir": "<path>"}` -- what source/config files are in that checkout directory.
    * `{"run": "<command>"}` -- ask the machine with one argv, never a shell. It runs in a
      read-only, networkless bubblewrap namespace; host home, temp files, devices, and common
      data/checkpoint/media paths are not exposed. Missing isolation is an explicit refusal.

    Paths resolve against the checkout and stage directory, but must remain within the
    checkout. Sensitive asset/weight/media paths are not returned to the external diagnostician.

    Why this exists at all: the system used to hand the reviser a *fixed* set of facts, each
    one added by hand after a particular benchmark needed it -- the environment names on this
    machine, the keys of a table beside the entry point, where a bundled binary lives. Every
    one of those was a substitute for looking, and none of them generalises: the next
    repository finds its facts in a file nobody wrote a helper for.
    """
    # `{repo}` is substituted wherever it appears, in a path or in a command. Every field of
    # the payload a reviser is shown spells the checkout that way -- `working_directory` is
    # `{repo}/policy/ACT` -- so it writes the same thing back when it asks to look at
    # something, and a literal `{repo}` in a path resolves to a directory that does not
    # exist. It asked, was told the file was not there, asked again with the same string, and
    # spent its budget: the placeholder is part of the vocabulary and nothing here honoured
    # it.
    def substitute(value: str) -> str:
        return value.replace("{repo}", str(Path(repo).resolve(strict=True)))

    for key, handler in (("look_at", _look_at), ("look_at_dir", _look_at_dir)):
        wanted = request.get(key)
        if isinstance(wanted, str) and wanted.strip():
            try:
                path = _resolve_inspection(substitute(wanted.strip()), repo=repo,
                                           directory=directory)
            except (OSError, ValueError) as exc:
                return {"error": str(exc)}
            return handler(path, repo=repo)
    command = request.get("run")
    if isinstance(command, str) and command.strip():
        return _run_inspection(substitute(command.strip()), repo=repo, directory=directory,
                               environment=environment, timeout=timeout)
    return {"error": "an inspection is one of {\"look_at\": path}, {\"look_at_dir\": path}, "
                     "or {\"run\": command}"}


def _resolve_inspection(wanted: str, *, repo: Path, directory: Path) -> Path:
    """Where a reviser's path points, trying the two places it can mean.

    A repository-relative path is what the model has been shown, so it is tried first. The
    stage's own directory is second because that is where `train.sh`'s neighbours are, and a
    reviser that writes `TASK_CONFIGS.json` means the one beside the entry point.
    """
    checkout = Path(repo).resolve(strict=True)
    stage = Path(directory).resolve(strict=False)
    if not stage.is_relative_to(checkout):
        raise ValueError("inspection working directory must stay inside the checkout")
    candidate = Path(wanted)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = None
        for root in (checkout, stage):
            option = root / candidate
            if option.exists() or option.is_symlink():
                resolved = option.resolve(strict=False)
                break
        if resolved is None:
            resolved = checkout / candidate
        resolved = resolved.resolve(strict=False)
    if not resolved.is_relative_to(checkout):
        raise ValueError("inspection paths must stay inside the checkout")
    if _private_inspection_path(resolved, repo=checkout):
        raise ValueError("inspection of datasets, checkpoints, media, or secrets is refused")
    return resolved


def _look_at(path: Path, *, repo: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if not path.is_file():
            return {"path": _shown(path, repo), "error": "inspection target is not a regular file"}
        with path.open("rb") as stream:
            raw = stream.read(FILE_BYTES + 1)
        truncated = size > FILE_BYTES or len(raw) > FILE_BYTES
        raw = raw[:FILE_BYTES]
        if b"\x00" in raw:
            return {"path": _shown(path, repo), "bytes": size,
                    "error": "inspection target is binary; content withheld"}
        text = raw.decode("utf-8")
        truncated = False
    except UnicodeError:
        return {"path": _shown(path, repo), "error": "inspection target is not valid UTF-8; "
                "content withheld"}
    except OSError as exc:
        return {"path": _shown(path, repo), "error": f"{type(exc).__name__}: {exc}"}
    return {"path": _shown(path, repo), "bytes": size,
            "lines_in_preview": text.count("\n") + (1 if text else 0),
            "content": sanitize_model_text(text, local_roots=(repo,)),
            **({"truncated": True} if truncated else {})}


def _look_at_dir(path: Path, *, repo: Path) -> dict[str, Any]:
    try:
        entries = []
        with os.scandir(path) as stream:
            for one in stream:
                candidate = Path(one.path)
                if _private_inspection_path(candidate, repo=repo):
                    continue
                suffix = "/" if one.is_dir(follow_symlinks=False) else \
                    ("@" if one.is_symlink() else "")
                entries.append((one.is_file(follow_symlinks=False), one.name + suffix))
        entries.sort(key=lambda item: (item[0], item[1]))
    except OSError as exc:
        return {"path": _shown(path, repo), "error": f"{type(exc).__name__}: {exc}"}
    return {"path": _shown(path, repo),
            "entries": [name for _, name in entries[:LISTING_ENTRIES]],
            **({"truncated": True} if len(entries) > LISTING_ENTRIES else {})}


def _run_inspection(command: str, *, repo: Path, directory: Path,
                    environment: dict[str, str], timeout: int) -> dict[str, Any]:
    where = Path(directory).resolve(strict=False)
    try:
        if not where.is_dir() or not where.is_relative_to(Path(repo).resolve(strict=True)):
            raise ValueError("inspection working directory must stay inside the checkout")
        argv = _inspection_command(command, repo=repo)
        safe_environment = _inspection_environment(environment, repo=repo)
        isolated = inspection_argv(argv, repo=repo, directory=where,
                                   environment=safe_environment)
        done = bounded_run(isolated, shell=False, cwd=Path(repo), timeout=timeout,
                           env=safe_environment)
    except subprocess.TimeoutExpired:
        return {"command": command, "error": f"it did not finish within {timeout}s"}
    except (OSError, ValueError) as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}
    said = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0 and any(line.startswith("bwrap:") for line in said.splitlines()):
        return {"command": command,
                "error": "read-only inspection sandbox is unavailable; the requested command "
                         "was not run: " + said[-COMMAND_BYTES:]}
    return {"command": command, "cwd": _shown(where, repo), "returncode": done.returncode,
            "said": said[:COMMAND_BYTES],
            **({"truncated": True} if len(said) > COMMAND_BYTES else {})}


def _shown(path: Path, repo: Path) -> str:
    """A path as the reviser wrote it, when it is inside the checkout."""
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


#: A shell script, by extension or by shebang. The first element of an argv is what gets
#: executed, and a Python interpreter given one of these parses it as Python.
_SHELL = ("sh", "bash", "zsh", "dash", "ksh")


def argv_problems(argv: Any) -> list[str]:
    """Commands that cannot work, as messages. Empty means runnable, not correct.

    A static check, before the process is started, because the alternative is what happened:
    the draft is executed, the shell script fails with `SyntaxError: invalid syntax` on its
    second line, and the failure is classified as an environment problem -- so the loop
    revised the working directory and the environment, three times, for a command whose
    first element was wrong. RoboTwin's `evaluate` and `train` both lost their whole budget
    to this, and the instruction not to do it is already in the prompt and was not followed.

    Telling a model what not to write is weaker than refusing to run what it wrote.
    """
    if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv):
        return []
    duplicate = conflicting_option_problem(argv)
    if duplicate:
        return [duplicate]
    program = Path(argv[0]).name.lower()
    if not program.startswith("python"):
        return []
    if len(argv) < 2:
        return []
    target = Path(argv[1])
    named = target.suffix.lower() in (".sh", ".bash")
    if not named and target.is_file():
        try:
            with target.open("rb") as handle:
                first = handle.readline(200).decode("utf-8", "replace")
        except OSError:
            first = ""
        named = any(first.lstrip().startswith(f"#!") and f"/{one}" in first
                    for one in _SHELL)
    if not named:
        return []
    return [f"the first element is {Path(argv[0]).name!r}, a Python interpreter, and the "
            f"second is {target.name!r}, a shell script. Python will parse it as Python and "
            f"fail on its first lines. The program is `bash`: the argv must start with "
            f"\"bash\", then the script's path, then its arguments."]


def coalesce_dynamic_overrides(argv: list[str], overrides: dict[str, Any]) -> list[str]:
    """Make a supplied sparse override effective when a generated argv also has a default.

    Stage functions are generated per repository and may spell a native default literally,
    then append a research setting. Leaving both in argv delegates experiment semantics to
    whichever CLI parser happens to be used (first-wins, last-wins, or error). For a known
    dynamic key, keep its value exactly once and preserve the command's observed spelling and
    assignment style. Unmatched keys are left alone: only the generated function can establish
    how a benchmark accepts a previously unseen option.
    """
    if not isinstance(argv, list) or not isinstance(overrides, dict):
        return list(argv) if isinstance(argv, list) else argv

    def normalize(name: str) -> str:
        return str(name).lstrip("-").split("=", 1)[0].lower().replace("_", "-")

    result = list(argv)
    for key, value in overrides.items():
        target = normalize(str(key))
        if target in {"checkpoint", "dataset", "output", "python", "repo", "task",
                      "task-id", "env-id", "benchmark", "benchmark-name"}:
            # These identify the frozen task, its inputs, or the caller's exact artifact;
            # they are not knobs for a sparse research override.
            continue
        matches: list[tuple[int, int, str, str]] = []
        index = 1
        while index < len(result):
            token = result[index]
            if not token.startswith("-") and "=" not in token:
                index += 1
                continue
            name = token.split("=", 1)[0]
            if normalize(name) != target:
                index += 1
                continue
            if "=" in token:
                matches.append((index, index + 1, token.split("=", 1)[0], "equals"))
                index += 1
            elif (token.startswith("--") and index + 1 < len(result) and
                  not result[index + 1].startswith("--")):
                matches.append((index, index + 2, token, "separate"))
                index += 2
            else:
                # A bare boolean switch has parser-specific true/false spelling. Do not
                # guess whether this repository wants --flag, --no-flag, or --flag=false.
                index += 1
        if not matches:
            continue

        _, _, prefix, style = matches[-1]
        replacement = ([prefix + "=" + str(value)] if style == "equals" else
                       [prefix, str(value)])
        for start, end, _, _ in reversed(matches):
            del result[start:end]
        result.extend(replacement)
    return result


def bind_step_budget_placeholders(argv: list[str], steps: Any) -> list[str]:
    """Bind only complete, explicit steps markers to the caller's bounded budget.

    Model-generated invocation/parameter values sometimes preserve `{steps}` or the
    literal text `i['steps']` instead of building the argv element from the function's
    input. Treating those strings as executable Python would be unsafe; leaving them
    unchanged makes the verifier reject an otherwise clear binding. This deliberately
    narrow adapter substitutes only a whole argv value (or the value side of `key=value`)
    when it exactly matches a known marker. It does not interpret expressions, prose, or
    arbitrary templates, and static numeric values remain subject to the normal budget
    variation check.
    """
    if not isinstance(argv, list):
        return argv
    markers = {"{steps}", "${steps}", "i['steps']", 'i["steps"]'}
    replacement = str(steps)
    result: list[str] = []
    for part in argv:
        if not isinstance(part, str):
            result.append(part)
            continue
        if part in markers:
            result.append(replacement)
            continue
        if "=" in part:
            option, value = part.rsplit("=", 1)
            if value in markers:
                result.append(option + "=" + replacement)
                continue
        result.append(part)
    return result


def conflicting_option_problem(argv: list[str]) -> str:
    """Refuse conflicting repeated long options, regardless of benchmark CLI spelling.

    An append-only settings map once emitted both a bounded baseline step count and a
    research proposal's full-run step count. Native CLIs disagree on first/last-wins;
    neither is a trustworthy experiment. Repeated equal values are harmless, while
    different values need one explicit effective option before the command is run.
    """
    seen: dict[str, str | None] = {}
    for index, part in enumerate(argv):
        if not part.startswith("--") or part == "--":
            continue
        body = part[2:]
        if "=" in body:
            name, value = body.split("=", 1)
        else:
            name = body
            value = (argv[index + 1] if index + 1 < len(argv) and
                     not argv[index + 1].startswith("--") else None)
        name = name.lower().replace("_", "-")
        if name in seen and seen[name] != value:
            return (f"option --{name} has conflicting values {seen[name]!r} and "
                    f"{value!r}; emit exactly one effective value. This is a command-builder "
                    "conflict, not a benchmark usage error: the base argv and a dynamic "
                    "settings/extra override both supplied this option. Remove the fixed "
                    "copy from the base argv and let the sparse dynamic override supply it "
                    "once; do not rely on first-wins or last-wins behavior.")
        seen[name] = value
    return ""
def checkpoint_for_verification(report: dict[str, Any], *,
                                declaration: dict[str, Any] | None = None) -> dict[str, Any]:
    """A checkpoint on this machine, so a stage that consumes one can be verified at all.

    **The verifier was never given one.** `inputs_for_verify` sets `dataset` and leaves
    `checkpoint` at the empty string `blank_inputs` starts it with, so a generated function
    that correctly reads `i["checkpoint"]` builds a command with an empty argument, the
    program says it cannot find the checkpoint, and the loop revises the one thing that was
    right. RoboTwin's evaluator reached exactly there: sixty-five rounds, ending on a
    correctly-built `bash eval.sh ... ` whose checkpoint argument it had been forced to bake
    in as a name, because the caller's slot was empty.

    Offer the exact discovered artifact, not a guessed parent directory. Some native
    evaluators load a file while others load a bundle directory; the command derivation can
    adapt a real file to its repository's convention, but it cannot recover a filename that
    was discarded before it saw the input.
    """
    declared = ((declaration or {}).get("assets") or {}).get("checkpoint") or {}
    path = str(declared.get("path") or "").strip()
    if path and Path(path).exists():
        return {"path": str(Path(path)),
                "from": "the declaration",
                "note": "the declaration named this checkpoint"}
    for row in (report.get("model_artifacts") or []):
        candidate = Path(str(row.get("path") or ""))
        if not candidate.exists():
            continue
        directory = candidate.parent if candidate.is_file() else candidate
        if any(directory.glob("*.ckpt")) or any(directory.glob("*.pt")) \
                or any(directory.glob("*.safetensors")):
            return {"path": str(candidate), "from": "the survey",
                    "note": f"found under {row.get('found_under')}"}
    return {"path": "", "from": "",
            "note": "no checkpoint exists on this machine, so a stage that consumes one "
                    "cannot be verified here. That is a finding about the benchmark's "
                    "assets rather than about the command."}


def keys_the_program_printed(output: str, *, limit: int = 120) -> list[str]:
    """The configuration keys a program listed about itself, when it listed any.

    A config-driven program prints what it composed before it does anything, and that print is
    its own definitive answer to "which keys exist" -- the question every failed override is
    asking. LIBERO prints a hundred lines of it, so quoting the dump into a prompt-sized
    excerpt is hopeless, and taking the first few lines is worse than useless: the key that
    was wanted (`folder`) is forty lines down. The names, though, are small, and they are
    what the model is guessing at. Confirmed on LIBERO: rounds went to `data_root`, then
    `data.dataset_path`, then `+data.dataset_path`, none of which exist, while `folder` was
    printed by the program every single time.
    """
    keys: list[str] = []
    for line in output.splitlines():
        match = _PRINTED_KEY.match(line)
        if not match:
            continue
        name = match.group(1)
        # A URL, a Windows drive letter and a sentence's first clause all look like this.
        if name in ("http", "https", "file", "note", "warning", "error", "usage") or len(name) > 40:
            continue
        # `TypeError: ...` in a traceback is a line with a colon in it, and reading it as a
        # configuration key put an exception name at the head of the list handed to the model.
        if name.endswith(("Error", "Exception", "Warning")):
            continue
        if name not in keys:
            keys.append(name)
    return keys[:limit] if len(keys) >= 4 else []


def parameter_keys(name: str) -> list[str]:
    """The keys a parameter is reachable by, however the invocation spells it.

    A parameter is *named* the way the command line spells it -- `--max_episodes` -- and a
    generated function reads it by whatever key it chose, usually the bare word. Requiring
    those to match exactly turns a naming convention into a failure, and the failure is
    reported as a KeyError in a function nobody can see.
    """
    bare = str(name).lstrip("-").replace("-", "_")
    keys = [str(name), bare]
    if bare != str(name).lstrip("-"):
        keys.append(str(name).lstrip("-"))
    return list(dict.fromkeys(keys))


def expand_parameters(parameters: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, value in (parameters or {}).items():
        for key in parameter_keys(name):
            out[key] = value
    return out


def error_excerpt(output: str, *, limit: int = 1200) -> str:
    """The part of a program's output that says what went wrong.

    Not the tail, and not only the head either. A program that prints a version banner on
    import and then refuses its arguments has its cause *above* output that looks more
    recent, so the first marker is where to start reading. But a nested traceback -- anything
    raised under a framework that re-raises, hydra, click, pytest -- carries its actual
    exception at the very *end*, after frames of library internals, and a reader given the
    first fourteen lines learns only that something was re-raised. The model said so:
    "the truncated traceback shows hydra's run_and_report re-raising an exception ... the
    visible error text does not identify" the cause. Both ends, with the gap marked.
    """
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    markers = ("error:", "usage:", "Traceback (most recent call last)", "Error:", "Exception:",
               "cannot", "No such file", "not found", "unrecognized", "expected one argument")
    hits = [index for index, line in enumerate(lines)
            if any(marker in line for marker in markers)
            or line.lstrip().startswith(("File \"", "raise "))]
    if not hits:
        return "\n".join(lines[-8:])[:limit]
    first = max(0, hits[0] - 1)
    # A config-driven program prints the configuration it composed before it does anything,
    # and that print is its own definitive statement of which keys exist -- the thing a
    # failed override is asking about. Starting at the first marker throws it away: LIBERO's
    # output is a hundred lines of config and then a traceback, so the answer to "what key
    # sets the dataset directory" was in the part being discarded.
    preamble = lines[:6] if first > 6 else []
    head = preamble + (["  ..."] if preamble else []) + lines[first:first + 14]
    if len(lines) <= len(head) + 4:
        return "\n".join(head)[:limit]
    tail = lines[-8:]
    # Budgeted between the two ends rather than applied to their join. Cutting the joined
    # text from the right removes the exception line -- which is the same mistake as
    # returning only the head, made one level down, and it survived the first fix because
    # the two ends together are longer than the limit on any real traceback.
    budget = max(1, limit // 2)
    return "\n".join(head)[:budget] + "\n  ...\n" + "\n".join(tail)[-budget:]


def argv_function_name(stage: str) -> str:
    return f"stage_argv_{stage}"


def is_total_work_control(name: str) -> bool:
    """Whether an option names total training work rather than rollout topology/horizon.

    Deliberately excludes controls such as `num_envs`, `num_steps` and
    `max_episode_steps`: those shape a rollout or an episode, not the run's total budget.
    The caller's bounded `steps` value must replace a genuine total-work option.
    """
    key = str(name).lstrip("-").split("=", 1)[0].lower().replace("-", "_")
    if "." in key:
        namespace, key = key.rsplit(".", 1)
        if namespace.split(".")[-1] not in {"train", "training", "trainer"}:
            return False
    unit = r"(?:steps?|timesteps?|iterations?|updates?|epochs?|frames?|samples?|episodes?)"
    return (re.fullmatch(rf"(?:total|train|training|max)_{unit}", key) is not None or
            key in {"steps", "timesteps", "iterations", "updates", "epochs", "episodes",
                    "num_iterations", "num_updates", "num_epochs", "num_episodes",
                    "n_steps", "n_iters", "n_updates", "n_epochs"})


def step_budget_problem(argv: list[str], altered: list[str], *, steps: int) -> str:
    """Check that a changed argv actually changes a native work-limit control.

    This is a conservative structural check, not proof of framework semantics. In
    particular, changing `num_envs` while leaving `total_timesteps` at the README's
    full-run value must not satisfy the small verification budget.
    """
    def controls(parts: list[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for index, part in enumerate(parts):
            token = str(part)
            if token.startswith("--"):
                body = token[2:]
                if "=" in body:
                    key, value = body.split("=", 1)
                elif index + 1 < len(parts) and not str(parts[index + 1]).startswith("-"):
                    key, value = body, str(parts[index + 1])
                else:
                    continue
            elif "=" in token and not token.startswith("-"):
                key, value = token.split("=", 1)
            else:
                continue
            key = key.lower().replace("-", "_")
            if any(word in key for word in
                   ("step", "timestep", "iteration", "update", "epoch", "frame", "sample")):
                found.setdefault(key, []).append(value)
        return found

    first, second = controls(argv), controls(altered)
    authoritative = {key for key in first if is_total_work_control(key)}
    considered = authoritative or set(first)
    changed = [key for key in considered if first.get(key) != second.get(key)]
    if not changed:
        return ("training argv changed without changing its native total-work limit; "
                f"work-limit controls were {first}; bind i['steps'] to total steps, "
                "iterations, updates or epochs, not parallelism/batch size")
    if any(len(first[key]) != 1 or len(second.get(key, [])) != 1 for key in changed):
        return "training argv repeats a work-limit option, so its effective budget is ambiguous"
    for key in changed:
        if key.endswith(("_per_env", "_per_epoch")):
            return f"{key} is a per-unit setting, not a total training budget"
        if is_total_work_control(key):
            try:
                amount = int(first[key][0].replace("_", ""))
                larger_budget_amount = int(second[key][0].replace("_", ""))
            except ValueError:
                continue
            if larger_budget_amount <= amount:
                return (f"native work limit {key} must increase when the caller's "
                        "training budget doubles")
            if amount > max(10_000, steps * 10):
                return (f"native work limit {key}={amount} exceeds the bounded "
                        f"verification request of {steps} steps by too much")
    return ""


def _argv_integer_option(argv: list[str], names: set[str]) -> int | None:
    """Read one explicit numeric long option from a generated command."""
    found: list[int] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if not token.startswith("--"):
            index += 1
            continue
        body = token[2:]
        if "=" in body:
            name, value = body.split("=", 1)
        elif index + 1 < len(argv) and not str(argv[index + 1]).startswith("-"):
            name, value = body, str(argv[index + 1])
            index += 1
        else:
            index += 1
            continue
        normalized = name.lower().replace("-", "_")
        if normalized in names:
            try:
                found.append(int(value.replace("_", "")))
            except ValueError:
                return None
        index += 1
    return found[0] if len(found) == 1 else None


def bounded_training_batch_guidance(native: dict[str, float], *, steps: int,
                                    argv: list[str] | None = None) -> str:
    """Suggest a feasible small rollout only when native counters prove the relation.

    Some vectorized trainers print their actual rollout size. If their own counters prove
    `batch_size == num_envs * num_steps`, this gives the controller a concrete, auditable
    way to fit one update inside a small smoke budget without guessing benchmark flags.
    """
    try:
        envs = int(native["num_envs"])
        batch = int(native["batch_size"])
        budget = int(steps)
    except (KeyError, TypeError, ValueError, OverflowError):
        return ""
    try:
        horizon = int(native.get("num_steps") or 0)
    except (TypeError, ValueError, OverflowError):
        horizon = 0
    if horizon < 1 and envs > 0 and batch % envs == 0:
        # Some trainers print batch_size and num_envs but omit num_steps from their run
        # banner. Infer the quotient only when an explicit command option agrees with the
        # observed batch, rather than assuming a rollout formula from benchmark identity.
        declared_horizon = _argv_integer_option(
            argv or [], {"num_steps", "n_steps", "rollout_steps"})
        if declared_horizon == batch // envs:
            horizon = declared_horizon
    if horizon < 1:
        return ""
    if min(envs, horizon, batch, budget) < 1 or envs * horizon != batch or batch <= budget:
        return ""
    max_envs = budget // horizon
    max_horizon = budget // envs
    return (f"The program's own counters prove batch_size = num_envs × num_steps "
            f"({envs} × {horizon} = {batch}), which is larger than the requested "
            f"{budget}-step smoke budget. To obtain at least one update, keep its native "
            f"total-work limit at the caller's budget and choose a documented rollout "
            f"configuration with num_envs ≤ {max_envs} at num_steps={horizon}, or "
            f"num_steps ≤ {max_horizon} at num_envs={envs}; do not increase the total "
            "work budget or change unrelated task-horizon settings.")


_WITHHELD_ARTIFACT_MARKERS = (
    "[WITHHELD_ASSET_PATH]", "[WITHHELD_ARTIFACT]", "[LOCAL_RESOURCE]",
    "[DECLARED_LOCAL_ARTIFACT]", "[LOCAL_PATH]", "[EXTERNAL_PATH]",
)


def preserve_local_artifact(row: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """Do not let a redacted prompt placeholder overwrite the local artifact declaration."""
    result = dict(change)
    proposed = str(result.get("artifact") or "")
    if any(marker in proposed for marker in _WITHHELD_ARTIFACT_MARKERS):
        if "artifact" in row:
            result["artifact"] = row["artifact"]
        else:
            result.pop("artifact", None)
    return result


def native_failure_feedback(outcome: dict[str, Any]) -> str:
    """Keep each native stream visible; a verbose stdout must not bury stderr."""
    reason = error_excerpt(str(outcome.get("error") or ""))
    diagnostics = outcome.get("native_diagnostics")
    if not isinstance(diagnostics, dict):
        return reason
    return reason + "\n\nNative execution diagnostics (not a score):\n" + json.dumps({
        "returncode": diagnostics.get("returncode"),
        "native_returncode": diagnostics.get("native_returncode"),
        "cleanup": diagnostics.get("cleanup", {}),
        "stderr_tail": str(diagnostics.get("stderr_tail") or "")[-4000:],
        "stdout_tail": str(diagnostics.get("stdout_tail") or "")[-2000:],
        "native_evidence_ref": outcome.get("native_evidence_ref", ""),
    }, ensure_ascii=False)


def verification_retry_key(argv: list[str], inputs: dict[str, Any]) -> tuple[str, ...]:
    """Compare a native command independently of its disposable output directory.

    Only a complete argument value equal to/under the caller's output slot is
    normalized. Input paths, seeds, other numbers and opaque shell code are kept.
    """
    output = str(inputs.get('output') or '').rstrip('/')
    key = []
    for part in argv:
        token = str(part)
        prefix, separator, value = token.partition('=')
        if not separator:
            value = token
        if output and (value == output or value.startswith(output + '/')):
            value = '{verification_output}' + value[len(output):]
            token = prefix + '=' + value if separator else value
        key.append(token)
    return tuple(key)


def caller_bound_work_options(function, inputs: dict[str, Any], argv: list[str]) -> set[str]:
    """Observe the Agent's binding instead of guessing native budget flag names.

    Only repository defaults are masked by these names. Explicit settings/extra
    still pass through normal protocol checks. Uncontrolled positional/shell
    values are not reinterpreted or rewritten here.
    """
    def option_values(parts):
        values = {}
        if not isinstance(parts, list):
            return values
        for index, token in enumerate(parts[1:], 1):
            if not isinstance(token, str):
                continue
            if '=' in token:
                name, value = token.split('=', 1)
            elif token.startswith('--') and index + 1 < len(parts):
                name, value = token, parts[index + 1]
                if not isinstance(value, str) or value.startswith('--'):
                    continue
            else:
                continue
            name = name.lstrip('-').lower().replace('_', '-')
            values.setdefault(name, set()).add(value)
        return values

    original = option_values(argv)
    bound = set()
    for slot in ('steps', 'episodes'):
        value = inputs.get(slot)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            continue
        # A validated pure function has no filesystem/process side effects.
        # If it cannot accept the alternate input, do not invent a binding.
        try:
            changed = option_values(function({**inputs, slot: max(2, value * 2)}))
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError):
            continue
        bound.update(name for name, current in original.items()
                     if len(current) == 1 and len(changed.get(name, ())) == 1
                     and current != changed[name])
    return bound


def generate_argv(client: Any, stage: str, *, entrypoint: str, invocation: str,
                  repository_files: list[str],
                  declared_parameters: dict[str, str] | None = None,
                  verify: Any = None, inputs_for_verify: dict[str, Any] | None = None,
                  attempts: int = ARGV_DERIVATION_ATTEMPTS, hint: str = "",
                  verification_context: dict[str, Any] | None = None,
                  verification_inputs_factory: Any = None,
                  require_step_control: bool = False,
                  require_evaluation_progress: bool = False,
                  agent_timeout_seconds: float = 180
                  ) -> tuple[str | None, dict[str, str], list[dict[str, Any]]]:
    """Write the function that turns the system's inputs into this benchmark's command.

    Generated rather than templated because invocations differ in kind rather than in
    spelling: one benchmark takes flags, another a config file and a name, a third a shell
    wrapper with positional arguments. A template language general enough to cover them
    would be a programming language, and generating the function is the smaller thing.
    """
    from .patch_validation import checked_function
    name = argv_function_name(stage)
    inputs_for_verify = dict(inputs_for_verify or {})
    # Whether the program got past its own arguments, as the model last reported it. Null
    # until a command has been run.
    accepted = False
    current_parameters = dict(declared_parameters or {})
    if require_step_control:
        # Repository examples commonly declare a full-run `--total_timesteps=2_000_000`.
        # Treating that as a final repository parameter overrode the function's dynamic
        # caller budget before verification, making the budget-binding check impossible.
        current_parameters = {key: value for key, value in current_parameters.items()
                              if not is_total_work_control(key)}
    def request(parameters: dict[str, str]) -> str:
        payload: dict[str, Any] = {
        "stage": stage,
        "function_name": name,
        "entrypoint": entrypoint,
        "invocation_as_written_in_the_repository": invocation,
        "inputs_available": INPUT_VOCABULARY,
        "verification_inputs": inputs_for_verify,
        "native_resource_context": verification_context or {},
        "values_already_settled_for_this_repository": parameters,
        "files_in_the_repository": repository_files[:200],
        "note": "The values under `values_already_settled_for_this_repository` are known and "
                "final. Write them into the argv as literals. They are NOT in the `i` dict "
                "and must not be read from it; the only keys `i` has are the ones in "
                "`inputs_available`. Return the argv only.",
    }
        # Only when there is one. A field that is present and empty reads as "the last round
        # established nothing", which is a different statement from "there was no last round".
        if hint:
            payload["what_the_last_round_established"] = hint
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    log: list[dict[str, Any]] = []
    # A generated function can return exactly the same argv on every retry. Give an
    # identical command one retry for transient failures, but do not spend the rest of
    # the stage budget executing it unchanged. The cache is local to this generation
    # call: a revised environment or invocation starts a fresh verification series.
    failed_argv: dict[tuple[str, ...], tuple[int, dict[str, Any]]] = {}
    #: The most recent command the program accepted the shape of. Held across attempts so a
    #: later regression cannot undo it.
    anchored: str | None = None
    anchored_evidence_ref = ''
    for repair in range(attempts):
        if verification_inputs_factory is not None:
            inputs_for_verify.update(verification_inputs_factory())
        # Rebuilt from the current values on every attempt. Built once, it carried the
        # original set forever: a draft would establish that the dataset lives at a
        # particular path, the next draft would not be told, and would invent a different
        # name for it -- which is what forty rounds of the same four guesses looks like.
        user = request(current_parameters)
        if repair == 0:
            current = user
        else:
            failed = log[-1]["error"]
            runtime_failure = "The arguments were accepted" in failed
            # The program printed its configuration before it refused the override, and that
            # print names every key it has. Quoted it is a hundred lines and does not fit;
            # by name it is one line, and it answers the question this loop keeps guessing
            # at. LIBERO printed `folder` on every run while successive drafts reached for
            # `data_root`, `data.dataset_path` and `data.data_folder`, none of which exist.
            said_output = str(log[-1].get("said") or "")
            printed = keys_the_program_printed(said_output)
            unset = unset_keys_the_program_printed(said_output)
            current = user + json.dumps({
                "rejected": failed,
                **({"the_program_printed_these_config_keys": printed,
                    "and_these_are_the_only_keys_it_has":
                        "use one of them, spelled exactly as printed; if none of them is the "
                        "thing the caller is trying to set, drop the override and say so"}
                   if printed else {}),
                **({"keys_it_printed_as_having_no_value": unset,
                    "start_here": "a key the program itself printed as unset is one it expects "
                                  "the caller to set; the key you want is more likely in this "
                                  "list than in the one above"}
                   if unset else {}),
                "instruction": (
                    # The retry instruction has to ask for the field that can fix it. Saying
                    # only "return the corrected function" left the model regenerating a
                    # command whose shape was already accepted, with nothing to change.
                    f"Return the corrected object. Keep `source` for pure value fixes, and "
                    f"add to `parameters` every value "
                    f"the program said it was missing, each with the value for this "
                    f"repository and the evidence for it. If a different executable/wrapper "
                    f"or command shape is necessary, return changed source with an explicit "
                    f"replacement rationale in reasoning; parsed arguments are not proof "
                    f"this failed invocation must be replayed."
                    if runtime_failure else
                    f"Return the corrected object. The program refused the arguments, so "
                    f"build the argv to match its usage line exactly. You may also correct "
                    f"`parameters`.")},
                ensure_ascii=False)
        content, metadata = client.chat_with_metadata(
            sanitize_model_text(ARGV_SYSTEM),
            sanitize_model_text(current), max_tokens=3000,
            timeout=agent_timeout_seconds, thinking="disabled")
        verification_called = False
        outcome = {}
        source, argv, text, revised = None, None, None, None
        source_replacement = {}
        try:
            payload = _object(content)
            if 'invocation_handoff' in payload:
                handoff = payload['invocation_handoff']
                last = log[-1] if log else {}
                ref = last.get('native_evidence_ref')
                if (not isinstance(handoff, dict) or not ref or
                        handoff.get('native_evidence_ref') != ref or
                        not isinstance(handoff.get('reason'), str) or
                        not handoff['reason'].strip()):
                    raise ValueError('invocation_handoff needs the exact latest failed '
                        'native evidence reference and a nonempty reason; no command ran')
                log.append({**last, 'attempt': repair + 1,
                    'status': 'Agent requested invocation repair',
                    'handoff_reason': redact(handoff['reason'])[:2000],
                    'response_sha256': object_digest(content)})
                return None, current_parameters, log
            source = str(payload["source"])
            if anchored is not None and repair > 0 and source != anchored:
                reason = payload.get('reasoning')
                if not isinstance(reason,str) or not reason.strip():
                    raise ValueError('changing a previously parsed failed source requires '
                        'an explicit reasoning rationale grounded in the supplied failure; '
                        'no replacement was run, and the old source was not silently replayed')
                source_replacement = {'why':redact(reason.strip())[:2000],
                    'prior_source_sha256':object_digest(anchored),
                    'replacement_source_sha256':object_digest(source),
                    'prior_native_evidence_ref':anchored_evidence_ref,
                    'authority':'agent_rationale_requires_new_native_verification'}
                anchored = None
            try:
                checked_function(source, name)
            except (ValueError, SyntaxError) as exc:
                from .contract_codegen import _offending_call
                raise ValueError(f"{exc}{_offending_call(source)}") from None
            revised = {**current_parameters,
                       **{str(k): str(v) for k, v in (payload.get("parameters") or {}).items()}}
            accepted = payload.get("shape_was_accepted") is True and not source_replacement
            if verify is not None:
                # The command is run before it is accepted. Nothing static decides whether
                # a flag exists on a program: the generated argv passed every check and was
                # refused by the program itself, printing the usage line that says so.
                #
                # The revised parameters go to the verifier, not the original ones, because
                # some failures are a value's fault rather than the function's -- a program
                # that cannot load something is naming the value it was handed.
                # The verifier is handed the *command*, built here from the revised
                # parameters, not the source. Handing it the source leaves the caller to
                # work out how the parameters reach the argv -- and a function that
                # hardcodes a value instead of reading it from `i` then looks like one that
                # read it, which is precisely the difference this loop exists to find.
                try:
                    command_inputs = {**blank_inputs(), **expand_parameters(revised),
                                      **inputs_for_verify}
                    # Settled parameters may have been corrected after a value-level
                    # failure while the accepted CLI shape remains anchored. Apply their
                    # current values to the generated command; stage-specific research
                    # settings and extra values take precedence over those repository values.
                    function = checked_function(source, name)
                    raw_argv = function(command_inputs)
                    bound_options = caller_bound_work_options(function, command_inputs, raw_argv)
                    dynamic_overrides: dict[str, Any] = {
                        key: value for key, value in expand_parameters(revised).items()
                        if str(key).lstrip('-').split('=', 1)[0].lower().replace('_', '-')
                        not in bound_options}
                    for input_name in ("settings", "extra"):
                        rows = command_inputs.get(input_name)
                        if isinstance(rows, dict):
                            dynamic_overrides.update(rows)
                    argv = coalesce_dynamic_overrides(
                        raw_argv, dynamic_overrides)
                    if require_step_control:
                        argv = bind_step_budget_placeholders(
                            argv, command_inputs.get("steps"))
                        altered_inputs = {
                            **command_inputs,
                            "steps": max(2, int(inputs_for_verify.get("steps") or 1) * 2)}
                        altered = coalesce_dynamic_overrides(
                            checked_function(source, name)(altered_inputs),
                            dynamic_overrides)
                        altered = bind_step_budget_placeholders(
                            altered, altered_inputs.get("steps"))
                        if altered == argv:
                            literal_budget = any(
                                marker in str(part).lower() for part in argv for marker in
                                ("{steps}", "i['steps']", 'i["steps"]', "${steps}"))
                            if literal_budget:
                                raise ValueError(
                                    "training argv contains a literal steps placeholder; "
                                    "`{steps}` and `i['steps']` are not substitutions in "
                                    "invocation/parameters. Build the option in the function "
                                    "source from `str(i['steps'])` so changing the caller's "
                                    "budget changes the native total-work argument")
                            raise ValueError("training argv ignores the caller's steps "
                                             "verification budget; bind i['steps'] to the "
                                             "native total-step/iteration option")
                        problem = step_budget_problem(
                            argv, altered, steps=int(inputs_for_verify.get("steps") or 1))
                        if problem:
                            raise ValueError(problem)
                    # Refused before the process starts. A command that cannot work is a
                    # rejected draft with a message the generator can act on, and running it
                    # instead turns a wrong first element into a SyntaxError from inside a
                    # shell script -- which reads as an environment problem and gets revised
                    # as one.
                    faults = argv_problems(argv)
                    if faults:
                        raise ValueError("; ".join(faults))
                    checkpoint = str(inputs_for_verify.get("checkpoint") or "")
                    if (stage == "evaluate" and checkpoint and
                            "checkpoint" in invocation.lower() and
                            not any(checkpoint in str(part) for part in argv)):
                        raise ValueError("the evaluation argv ignored the caller's "
                                         "checkpoint path; the exact frozen policy path "
                                         "must be passed instead of a default or template")
                except Exception as exc:
                    raise ValueError(f"{name} could not build a command: "
                                     f"{type(exc).__name__}: {exc}") from None
                argv_key = verification_retry_key(argv, inputs_for_verify)
                prior = failed_argv.get(argv_key)
                if prior and prior[0] >= 2:
                    outcome = {**prior[1], "error": str(prior[1].get("error") or "")
                               + "\nExact argv already failed twice without a change "
                                 "apart from a disposable verification output directory; "
                                 "revise the command or its parameters."}
                else:
                    verification_called = True
                    outcome = verify(argv)
                    if outcome.get("ok"):
                        failed_argv.pop(argv_key, None)
                    else:
                        failed_argv[argv_key] = ((prior[0] if prior else 0) + 1,
                                                 dict(outcome))
                if not outcome.get("ok"):
                    text = str(outcome.get("error") or "")
                    # Whether the program got past its arguments is the model's answer, not
                    # a reading of its output. `shape_was_accepted` is a field on the reply
                    # that produced this command, and it is what decides whether the next
                    # attempt keeps the shape and varies the values.
                    #
                    # It used to be inferred here from seventeen substrings -- `usage:`,
                    # `unrecognized arguments`, `no such file or directory: '/home` -- and
                    # the last of those fired only under this machine's home directory, so
                    # the same failure was an invocation problem here and a value problem
                    # elsewhere. The verdict that followed was appended to the failure text
                    # and read by the model as though it were part of what the program said.
                    # The model's shape verdict cannot anchor a draft rejected by our own
                    # preflight: the benchmark has not seen that argv. Otherwise a sparse
                    # research override colliding with a baked-in default is rejected before
                    # execution, yet every repair is forced to repeat the same conflict.
                    if (accepted and verification_called and anchored is None
                            and outcome.get("launch_verified") is not False):
                        anchored = source
                        anchored_evidence_ref = str(outcome.get('native_evidence_ref') or '')
                    # The model is handed the failure and the output, unedited, and decides
                    # what it means. It is the same reader that has to act on the answer.
                    raise ValueError("the command did not run. The program said:\n"
                                     + native_failure_feedback(outcome))
                current_parameters = revised
            log.append({"stage": stage, "attempt": repair + 1, "status": "accepted",
                        "source_replacement":source_replacement,
                        "reasoning": payload.get("reasoning"),
                        "parameters": current_parameters,
                        "native_evidence_ref": outcome.get("native_evidence_ref", ""),
                        "training_progress": outcome.get("training_progress", {}),
                        "verification_output": outcome.get("verification_output", ""),
                        "said": redact(str(outcome.get("error") or ""))[-12000:]
                        if verify is not None else "",
                        **({"verified_artifact": outcome["verified_artifact"]}
                           if verify is not None and
                           isinstance(outcome.get("verified_artifact"), dict) else {}),
                        "response_sha256": object_digest(content)})
            return source, current_parameters, log
        except (ValueError, KeyError, SyntaxError, json.JSONDecodeError) as exc:
            # Read through `locals()`: the failure this handler exists for can happen before
            # the earlier lines of the `try` have run -- a response that is not JSON at all
            # raises on the first statement -- and naming `source` directly then raises
            # `UnboundLocalError` *inside the handler*, which escapes and ends the run. The
            # failure path has to be at least as robust as the success path, because it is
            # the one that runs when something has already gone wrong.
            proposed = locals().get("source")
            attempted = locals().get("argv")
            output = locals().get("text")
            # What the draft said about the values survives its rejection. Without this the
            # next attempt starts from the original set, so every round re-derives every fact
            # from nothing: LIBERO's dataset directory was established in one round, absent
            # from the next round's command entirely, and re-derived in the round after that.
            # A value the program accepted is a fact no matter what happened to the command
            # that carried it.
            settled = locals().get("revised")
            if isinstance(settled, dict):
                current_parameters.update(settled)
            log.append({"stage": stage, "attempt": repair + 1, "status": "rejected",
                        "source_replacement":source_replacement,
                        # What it proposed, kept because a refusal that leaves no record of
                        # the draft leaves nothing to diagnose it from. This is what the
                        # attempt actually wrote, not a summary of it.
                        "source": proposed if isinstance(proposed, str) else content,
                        "failure_kind": outcome.get("failure_kind", ""),
                        "native_evidence_ref": outcome.get("native_evidence_ref", ""),
                        "native_diagnostics": outcome.get("native_diagnostics", {}),
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        # The program's whole output, unexcerpted. The message above holds
                        # only what fits in a prompt; the caller needs the rest, because what
                        # a config-driven program printed about itself is a hundred lines
                        # long and the key that was wanted is forty lines down.
                        "said": output[:40000] if isinstance(output, str) else "",
                        # What the command looked like when it was refused, so a caller that
                        # has to change the invocation rather than the command can see it.
                        "argv": attempted if isinstance(attempted, list) else [],
                        "response_sha256": object_digest(content)})
        current = user + json.dumps({"rejected": log[-1]["error"]}, ensure_ascii=False)
    return None, current_parameters, log


DIAGNOSE_SYSTEM = (
    "A stage of a research loop will not run. You are given the stage as it was derived, the "
    "command that was built from it, and what the program said. **Say what is wrong.** You do "
    "not propose a fix -- that is a separate question asked separately -- and a finding that "
    "names the wrong thing sends the next step somewhere useless.\n\n"
    "**Look before you answer.** Almost every failure names something -- a setting, an "
    "environment, a file, a program -- and the machine has the answer, one of three requests "
    "away. Ask, and you will be shown what you asked for; then ask again, or answer:\n\n"
    "    {\"look_at\": \"<a file>\"}                 what that file says\n"
    "    {\"look_at_dir\": \"<a directory>\"}        what is in it\n"
    "    {\"run\": \"<a command that only looks>\"}  ask the machine\n\n"
    "Paths may be relative to the checkout or to where the stage runs, but must stay inside "
    "the checkout. Do not ask to inspect data, demonstrations, trajectories, videos, weights, "
    "checkpoints, secrets, or credential files. `run` accepts one argv only (not shell syntax) "
    "and executes in a read-only, networkless namespace with a sanitized environment; host "
    "home files, devices and external paths are unavailable. Use bounded queries such as "
    "`conda env list`, `which ffmpeg`, the program's own `--help`, or `git status`. If the "
    "sandbox is unavailable, treat that as an environment boundary and do not retry it with "
    "shell syntax or another route. This is why you are given no pre-computed list of facts: "
    "the system "
    "does not know which convention this repository follows, and a list of the conventions it "
    "happens to know about is worse than none, because it reads as complete.\n\n"
    "**If you find yourself writing \"I would need to see ...\", that is the request -- make "
    "it.** An answer that says what you would have to look at is worth less than the look "
    "itself, and it costs one turn. A diagnostician here answered with \"need to look at the "
    "entrypoint and the config file to find the right key\" while the key was in a file it "
    "could have opened: the sentence was correct, its instinct was right, and it stopped one "
    "step short of the thing it had already identified.\n\n"
    "Five shapes are worth recognising, because each looks like something else and has an "
    "answer that is one look away.\n\n"
    "**A name the program looked up and did not find.** It says `KeyError`, `EnvironmentName"
    "NotFound`, `No such file or directory` on a name, or a registry that lists what it does "
    "have. The name it was given is a guess and the thing that holds the real names is a file "
    "or a command: the config the entry point loads, `conda env list`, a directory listing. A "
    "name that *describes* the run -- the task, the policy, the checkpoint -- is not a name "
    "the program ships.\n\n"
    "**A program the command spawned and could not find.** An executable name with nothing on "
    "the search path. Look inside the environment the stage runs in before concluding it is "
    "absent: packages that bundle a binary ship it under a name of their own, so the directory "
    "alone will not do if the filename differs.\n\n"
    "**Something the code called that is not there.** An `AttributeError` on a method, an "
    "`ImportError` on a module, a config key nothing defines. Two readings and they are "
    "different findings: the version installed is not the version the code was written "
    "against, or the code is being driven in a way its authors did not intend and never "
    "exercised. Reading the source that raises it is how the two are told apart.\n\n"
    "**A staging step that has already run.** `as_derived.staging` is a list of commands this "
    "stage runs before itself, made to satisfy something the repository assumes and does not "
    "have. It may be the fault, and if it is, say so and say what about it: a guard that "
    "compares against a filename form the data does not use, a destination named the way the "
    "system would name it rather than the way the loader opens it, a link made from the wrong "
    "directory. That is a finding about the repository and it is answered by correcting the "
    "script -- which is a thing the next step can do, and only if you say which part of it is "
    "wrong.\n\n"
    "**A path the program built that does not exist.** It may be built from its own directory, "
    "from a config value, or from a relative path whose base is wherever it happens to run. "
    "Look at what it computed and at what is on the disk beside the checkout: a directory of "
    "the same kind of files, under a name that does not quite match, is the ordinary "
    "arrangement and not an absence.\n\n"
    "**A program that was never reached.** The failure is about an import, a module, a display "
    "or a shell, and the program's own arguments were never read. This is a finding about how "
    "the command is invoked and not about what it was asked.\n\n"
    "One search-path mistake is worth naming because it looks like the fix and is the opposite "
    "of it. A directory with no `__init__.py` on the search path is a namespace package, and a "
    "*regular* package of the same name found through another entry replaces it -- so adding a "
    "package's own directory beside its parent changes which module a dotted import resolves "
    "to. Measured: with a checkout and its own same-named subdirectory both on the path, "
    "`import <pkg>.<pkg>` resolved the outer name to the inner package, whose own `__path__` "
    "has no such name in it, and the program reported that a module it ships does not exist. The "
    "checkout alone is right; the checkout plus its package directory is wrong.\n\n"
    "`what_this_failure_has_looked_like_before` is what this same failure has needed at other "
    "repositories, and `already_tried_and_did_not_work` is what was tried here and did not "
    "work. Read both: the first is the strongest evidence you have and the second is not worth "
    "proposing again. `how_the_search_is_going` is the shape of the attempts so far, which you "
    "cannot see from inside one of them.\n\n"
    "**Some findings are about the repository and not about the command.** A script that "
    "computes a path whose depth cannot be satisfied from where it must run; a loader that "
    "opens a filename the checkpoint on disk does not have; a config table a previous step "
    "generates. AutoSOTA calls the answer protocol-preserving repository repair: the agent may "
    "synthesise missing glue, repair file-system assumptions or reconstruct non-core scripts, "
    "**provided the evaluation protocol, the dataset split and the target setting stay "
    "unchanged**. Say so as a finding -- `{refers_to: \"the repository\", what_would_have_to_"
    "change: \"...\"}` -- and a later step may make it true. What that step may not do is "
    "change what is being measured: an evaluator's own files, the contents of the data, the "
    "definition of success.\n\n"
    "Return one JSON object: {\"finding\": \"<what is wrong, in terms that say what would "
    "have to change>\", \"evidence\": \"<what you looked at that shows it>\", "
    "\"about\": \"the command\" or \"the repository\", "
    "\"changes_that_cannot_fix_it\": \"<a change the next step might otherwise propose, and "
    "why it would not work -- empty when you know of none>\", \"no_change_can_fix_this\": "
    "\"<null, or why no change to where it runs or in what environment can make it run>\"}"
)

REVISE_SYSTEM = (
    "You are given a stage's invocation, and a finding about why the command built from it "
    "does not run. **Return the invocation change the finding calls for.**\n\n"
    "An invocation is five things. `working_directory` is where it runs. `environment` is what "
    "its process has. `stdin` is what is fed to it when it asks a question. `invocation` is "
    "how it is called. `parameters` are the values only this benchmark knows. A finding that "
    "names a search path, a shell, a display or a directory is about the first two; one that "
    "names a key, a task or a checkpoint is about the last.\n\n"
    "**Use {repo} for the checkout in every path.** It is substituted before the command runs, "
    "and how the field is written decides whether it resolves.\n\n"
    "The selected stage interpreter is separately exposed as {verified_interpreter}. "
    "Use this exact handle in environment values or invocation fields when a native wrapper "
    "chooses its own Python. The executor resolves it to the already selected interpreter; "
    "do not prefix it with {repo} or guess an environments directory under the checkout. "
    "The handle is an executable identity, not proof every consumer import will succeed.\n\n"
    "**An environment value may refer to one that is already set, and should when it is an "
    "addition.** `\"PATH\": \"{repo}/env/bin:${PATH}\"` puts a directory in front of the "
    "existing path; `\"PATH\": \"{repo}/env/bin\"` replaces it, and every executable the "
    "system relies on -- the shell itself among them -- stops being findable. A finding about "
    "a missing program is usually this mistake and not a missing program. The same holds for "
    "`PYTHONPATH`, `LD_LIBRARY_PATH` and any variable a program appends to.\n\n"
    "**Change as little as you can, and enough to fix this.** Do not add a variable that "
    "narrows what the interpreter can see -- hiding a site-packages directory, disabling an "
    "index -- as a precaution: the environment already runs other stages, and a package that "
    "is reachable today stops being reachable the moment a door is closed. Every addition has "
    "to be the thing the finding asked for. Do not set `PYTHONPATH` to the checkout *and* the "
    "package directory both: see the note about namespace packages in what you were given.\n\n"
    "**Give the whole environment the stage needs, not only the variable you are adding.** The "
    "fields merge over what is already known, so a field left out keeps its previous value and "
    "a field given replaces it entire.\n\n"
    "**The staging this stage already has is in `as_derived`.** Return the whole list when "
    "you change it -- a field given replaces the value it had -- and read it first, because a "
    "finding that says the existing staging is wrong is a correction to *that* script and not "
    "a reason to write a different one from scratch.\n\n"
    "**A finding whose `about` is \"the repository\" is answered with `staging`.** That field "
    "is a list of shell commands run in the stage's directory, with the stage's environment, "
    "before the stage itself -- for making true something the repository assumes and does not "
    "have. A symlink where a script computes a path, a copy under the filename a loader opens, "
    "a step that generates the table a later stage reads. Two rules and both are absolute: "
    "**the commands must be safe to run twice** (they run before every attempt, and before "
    "every round of the research loop), and **they must not change what is measured** -- not "
    "the evaluator's own files, not the contents of the data, not the definition of success. "
    "A staging command that would alter an evaluation is caught when that evaluation runs, "
    "because the files it depends on are frozen and checked; what saves the round is not "
    "doing it.\n\n"
    "**Take a change back when it made things worse.** You are shown the changes already tried "
    "and what each was for; removing one is a revision like any other.\n\n"
    "**`no_change_can_fix_this` is the last resort, not the summary.** Set it only when the "
    "finding really is outside what these five fields can express. If what you are about to "
    "write there is a change to where it runs, in what environment, what is fed to it, or how "
    "it is called, **write it in that field instead**. A reviser here diagnosed exactly the "
    "right thing -- \"the environment I set prepends the venv's bin, which does not contain "
    "`bash`, and the standard PATH entries must stay behind it\" -- and returned it as an "
    "obstacle. The obstacle field holds a `PATH`, and setting it was a change the caller would "
    "have applied the moment it was asked for.\n\n"
    "Return one JSON object: {\"working_directory\": \"<where it runs>\", "
    "\"environment\": {\"<VAR\": \"<value>\"}, \"stdin\": \"<what to feed it when it "
    "asks, if it asks>\", \"interpreter\": \"<a python this stage needs instead of the "
    "one the run is using, when it needs its own>\", "
    "\"invocation\": \"<how it is called, if that was also wrong>\", "
    "\"artifact\": \"<actual output-relative or cwd-relative path/glob, if the "
    "declared artifact did not appear>\", "
    "\"parameters\": {\"<name>\": \"<value>\"}, \"why\": \"<what in the finding this "
    "answers>\", \"not_an_invocation_problem\": \"<null unless nothing about where or in "
    "what environment it runs can fix this; then state the real obstacle>\"}"
)


def make_runnable(client: Any, stage: str, row: dict[str, Any], *, repo: Path,
                  repository_files: list[str], inputs_for_verify: dict[str, Any],
                  rounds: int = ARGV_DERIVATION_ROUNDS,
                  attempts: int = ARGV_DERIVATION_ATTEMPTS,
                  base_environment: dict[str, str] | None = None,
                  on_event: Any = None,
                  on_revision: Any = None,
                  require_step_control: bool = False,
                  require_evaluation_progress: bool = False,
                  verification_timeout: int = 600,
                  verification_window: Any = None,
                  verification_inputs_guard: Any = None,
                  unique_verification_outputs: bool = False,
                  remaining_seconds: Any = None,
                  agent_timeout_seconds: float = 180
                  ) -> tuple[str | None, dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    """Generate a stage's command, and revise the invocation when arguments are not the cause.

    The argv is a function of the invocation, so a failure the invocation caused cannot be
    repaired by generating the argv again -- the loop would spin, which is what it did, five
    attempts at an import error each regenerating a command whose arguments were never the
    issue. When the command cannot be made to run, the invocation is what gets revised and
    the command is generated once more from it.

    Returns the accepted source, the parameters as they ended up, the whole log, and the
    stage row as it now stands. All four, because a caller that runs the command has to run it
    the way the verifier did: values it does not supply raise `KeyError` inside a function it
    cannot see, and -- the failure this return value was added for -- a *revised* environment
    or working directory is what made the command work in the first place. The driver kept the
    source and dropped the row, so every stage that verified ran afterwards without the
    `PYTHONPATH` that the verification had established, and failed at its first import.
    """
    from .declarative_backend import (invocation_directory, invocation_environment,
                                      invocation_stdin)
    # The interpreter this stage was verified with, recorded on the stage rather than left
    # to the caller. A run has one interpreter and a benchmark's stages may not: an entry
    # point that starts a policy server under one environment and a simulation under another
    # is an ordinary shape, and "which python" is a property of the command. A revision that
    # names its own replaces this, and the check below refuses one that names a path which
    # is not there.
    row = {**row, "interpreter": str(row.get("interpreter")
                                     or inputs_for_verify.get("python") or "")}
    verify_output = str(inputs_for_verify.get("output") or "")
    from .stage_verification import resource_context, missing_checkout_paths, run_native, recent_attempts
    resources = resource_context(repo)
    if verify_output:
        resources['previous_native_verifications'] = [
            attempt for attempt in recent_attempts(Path(verify_output).parent)
            if attempt.get('stage') == stage][:3]
        resources['history_instruction'] = (
            'These are sealed historical trials, not current readiness or scores. Use their '
            'diagnostics to avoid replaying unchanged failures; inspect the wrapper/loader '
            'source and use the caller input slots for actual paths. A new native receipt '
            'is required to prove any repair worked.')
    def next_verification_inputs():
        # Do not create the output itself: native trainers often require it absent.
        path = Path(verify_output).parent/'verification_outputs'/uuid.uuid4().hex
        path.parent.mkdir(parents=True, exist_ok=True)
        inputs_for_verify['output'] = str(path)
        return dict(inputs_for_verify)
    def local_environment(env: dict[str, str]) -> dict[str, str]:
        return (run_local_environment(Path(verify_output).parent, env)
                if verify_output else env)
    # Held in a dict rather than closed over: a revision replaces the directory and the
    # environment, and the verifier has to see the new ones. A default argument captures its
    # value at definition time, so the closure would keep testing the invocation that had
    # already been revised -- which is what it did.
    # The machine's decision about which device this runs on goes under anything the stage
    # declares, and above the ambient environment. A command verified on one device and run
    # on another is the same failure as one verified with a `PYTHONPATH` and run without it:
    # the verification establishes something the execution then does not have.
    current = {"directory": invocation_directory(row, repo=repo, default=repo),
               "environment": local_environment({
                   **os.environ, **(base_environment or {}),
                   **invocation_environment(row, repo=repo,
                                            base={**os.environ,
                                                  **(base_environment or {})})}),
               "stdin": invocation_stdin(row)}
    artifact_failures: dict[str, dict[str, Any]] = {}
    native_postconditions: dict[str, Any] = {}

    def allowed_timeout(requested: float) -> float:
        if remaining_seconds is None:
            return requested
        left = float(remaining_seconds())
        if left <= 0:
            raise TimeoutError("run wall-clock budget exhausted during verification")
        return min(requested, left)

    def _stage() -> str:
        """Make true whatever the repository assumes and does not have, before running.

        Run before the command and not once at setup, because the verification is what decides
        whether the command works and a command verified against an unstaged repository has
        been verified against something that will not be there when the loop runs it. Run
        before every attempt for the same reason: the commands are required to be safe to run
        twice, and one that is not will say so the first time it is.
        """
        for command in invocation_staging(row):
            try:
                done = bounded_run(isolated_argv(["sh", "-c", command],
                                                 output=Path(verify_output).parent
                                                 if verify_output else repo, repo=repo,
                                                 native_environment=current['environment']
                                                 if verify_output else None,
                                                 require_pid_namespace=True),
                                   timeout=allowed_timeout(600),
                                   cwd=Path(current["directory"]),
                                   env=current["environment"])
            except (subprocess.TimeoutExpired, OSError, ValueError, TimeoutError) as exc:
                return f"{type(exc).__name__}: {exc}"
            if done.returncode != 0:
                return ((done.stdout or "") + (done.stderr or ""))[-2000:]
        return ""

    def verify(argv):
        # A command can exit successfully yet fail an artifact/progress contract.
        # Keep its actual execution evidence on ALL those branches, not just on
        # nonzero exits. Otherwise Fix sees an artifact refusal with no native ID
        # and the Agent cannot hand it off without rerunning the whole evaluator.
        native_postconditions.clear()
        outcome = verify_command(argv)
        return {**native_postconditions, **outcome}

    def verify_command(argv):
        """Run it. The program is the only authority on what it accepts."""
        progress: dict[str, Any] = {}
        pattern_problem = artifact_pattern_problem(str(row.get("artifact") or ""))
        if pattern_problem:
            return {"ok": False, "error": f"invalid declared artifact glob: {pattern_problem}; "
                                             "correct the artifact pattern before rerunning"}
        cache_key = object_digest({"argv": argv, "directory": str(current["directory"]),
                                   "environment": current["environment"],
                                   "artifact": row.get("artifact"),
                                   "staging": invocation_staging(row)})
        if cache_key in artifact_failures:
            return artifact_failures[cache_key]
        staged = _stage()
        if staged and staged.strip():
            return {"ok": False, "error": f"staging did not succeed:\n{staged}"}
        missing = missing_checkout_paths(argv, repo=repo,
                    output=Path(str(inputs_for_verify['output'])) if verify_output else None)
        if missing:
            return {"ok": False, "launch_verified":False,
                "failure_kind":"resource_path_missing", "error":
                f"local checkout paths do not exist in native execution: {missing}. "
                "No command was launched. Use the explicitly bound resource targets and "
                "their inspected child directories in native_resource_context; do not "
                "fall back to the original repository's unbound path or an implicit download. "
                "For an intentionally generated path, declare source-backed staging first."}
        if verification_inputs_guard is not None:
            try:
                verification_inputs_guard()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                return {'ok':False, 'launch_verified':False,
                        'failure_kind':'producer_identity_changed',
                        'error':f'Frozen producer input failed verification: {type(exc).__name__}: {exc}'}
        try:
            started = time.time()
            if verification_window is not None and verify_output:
                done = run_native([str(a) for a in argv], repo=repo,
                    output=Path(verify_output).parent, cwd=Path(current["directory"]),
                    env=current["environment"], stdin=current["stdin"],
                    timeout=allowed_timeout(verification_timeout), window=verification_window,
                    stage=stage, verification_output=str(inputs_for_verify.get('output') or ''))
            else:
                done = bounded_run(isolated_argv([str(a) for a in argv],
                                             output=Path(verify_output).parent
                                             if verify_output else repo, repo=repo,
                                             native_environment=current['environment']
                                             if verify_output else None,
                                             require_pid_namespace=True),
                               timeout=allowed_timeout(verification_timeout),
                               cwd=Path(current["directory"]),
                               env=current["environment"], input=current["stdin"])
        except subprocess.TimeoutExpired as exc:
            partial = (str(exc.stderr or "") + "\n" + str(exc.output or ""))[-6000:]
            return {"ok": False, "launch_verified": True,
                    "error": f"launch worked but the command did not finish within "
                             f"{verification_timeout}s; "
                             "evaluation and artifact postconditions remain unverified. "
                             f"Native partial output (not discarded):\n{partial}", "said":partial,
                    "native_evidence_ref":getattr(exc, "evidence_ref", "")}
        except (OSError, ValueError, TimeoutError) as exc:
            # A draft can name a command that cannot be executed at all -- drop the
            # interpreter and the argv starts with the script, which is not executable. That
            # is a rejected draft like any other, and letting it escape ends the whole run:
            # a `PermissionError` from `subprocess` killed a derivation that was one round
            # from finishing. Nothing a draft contains may be able to do that.
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    'native_evidence_ref':getattr(exc, 'evidence_ref', '')}
        native_postconditions.update(
            native_evidence_ref=getattr(done, 'evidence_ref', ''),
            verification_output=str(inputs_for_verify.get('output') or ''),
            launch_verified=True,
            native_diagnostics={'returncode':done.returncode,
                'native_returncode':getattr(done, 'native_returncode', done.returncode),
                'cleanup':getattr(done, 'native_cleanup', {}),
                'stderr_tail':str(done.stderr or '')[-4000:],
                'stdout_tail':str(done.stdout or '')[-2000:]})
        if done.returncode == 0 and stage == "train" and require_step_control:
            from .readings import named_numbers, training_progress
            native_output = (done.stdout or "") + "\n" + (done.stderr or "")
            native = named_numbers(native_output)
            progress = training_progress(native_output)
            if (progress['status'] == 'unknown' and row.get('artifact')
                    and getattr(client, 'supports_native_progress_audit', False)
                    and getattr(done, 'evidence_ref', None)):
                from .declarative_backend import DeclarativeBackend
                output_root = Path(str(inputs_for_verify['output']))
                artifact = DeclarativeBackend(repo=repo, answer={'stages':{stage:row}},
                                               sources={}).check_artifact(stage, output_root, since=started)
                if artifact.get('matched') and not artifact.get('invalid_pattern'):
                    from .training_progress_audit import audit
                    progress = audit(client, repo=repo, root=Path(verify_output).parent,
                        output=output_root, argv=argv, since=started,
                        native_evidence_ref=done.evidence_ref)
            if progress["status"] != "observed":
                sizing = {key: native[key] for key in
                          ("batch_size", "minibatch_size", "num_envs", "num_minibatches",
                           "num_steps", "num_iterations", "num_updates")
                          if key in native}
                budget_hint = bounded_training_batch_guidance(
                    native, steps=int(inputs_for_verify.get("steps") or 1), argv=argv)
                return {"ok": False,
                        "native_evidence_ref":getattr(done, 'evidence_ref', ''),
                        "training_progress":progress,
                        "verification_output":str(inputs_for_verify.get('output') or ''),
                        "failure_kind":"training_progress_unverified", "error":
                        f"trainer exited zero but positive training progress was not "
                        f"verified ({progress}); native sizing counters: {sizing}; "
                        f"requested total-step budget: {inputs_for_verify.get('steps')}; "
                        "adjust the native batch, parallelism and minibatch count so this "
                        "bounded verification performs at least one update and emits "
                        "native progress evidence IF sizing proves zero work. Unknown "
                        "progress is not proof training failed: inspect the Supervisor audit "
                        "reason and sealed native evidence before repeating training; " +
                        ((budget_hint + " ") if budget_hint else "") +
                        "log tail: " + native_output[-500:]}
        if done.returncode == 0 and require_evaluation_progress:
            from .readings import evaluation_progress
            native_output = (done.stdout or "") + "\n" + (done.stderr or "")
            progress = evaluation_progress(native_output)
            if progress["status"] == "zero":
                return {"ok": False, "error":
                        "The arguments were accepted, but the native evaluator explicitly "
                        "reported zero completed episodes ("
                        f"{progress.get('evidence')}). Re-derive its horizon/parallelism from "
                        "source so verification completes at least one episode; do not treat "
                        "the requested episode count as completed work."}
        verified_artifact: dict[str, Any] | None = None
        if done.returncode == 0 and row.get("artifact") and inputs_for_verify.get("output"):
            from .declarative_backend import DeclarativeBackend
            backend = DeclarativeBackend(repo=repo, answer={"stages": {stage: row}},
                                         sources={})
            evidence = backend.check_artifact(
                stage, Path(str(inputs_for_verify["output"])), since=started)
            if evidence.get("invalid_pattern"):
                return {"ok": False, "error": f"invalid declared artifact glob: "
                                                f"{evidence.get('why')}; correct the pattern"}
            if not evidence.get("matched"):
                evidence = backend.artifact_beside(
                    stage, Path(current["directory"]), since=started)
            if evidence.get("invalid_pattern"):
                return {"ok": False, "error": f"invalid declared artifact glob: "
                                                f"{evidence.get('why')}; correct the pattern"}
            if not evidence.get("matched"):
                pattern = str(row["artifact"])
                prefix = pattern.split("*", 1)[0].split("?", 1)[0].split("[", 1)[0]
                nearby = Path(current["directory"]) / (
                    Path(prefix) if prefix.endswith("/") else Path(prefix).parent
                    if prefix else Path("."))
                fresh: list[str] = []
                if nearby.is_dir() and nearby.is_relative_to(repo):
                    for index, path in enumerate(nearby.rglob("*")):
                        if index >= 5000 or len(fresh) >= 6:
                            break
                        try:
                            if path.is_file() and path.stat().st_mtime >= started:
                                fresh.append(str(path.relative_to(Path(current["directory"]))))
                        except (OSError, ValueError):
                            continue
                failure = {**native_postconditions, "ok": False, "error":
                           f"stage exited zero but fresh declared artifact "
                           f"{pattern!r} did not appear under the output or working "
                           f"directory; fresh nearby files: {fresh}; correct the artifact "
                           "pattern or the command"}
                artifact_failures[cache_key] = failure
                return failure
            if evidence.get("matched"):
                evidence_root = (Path(current["directory"]) if
                                 evidence.get("found_beside_the_command") else
                                 Path(str(inputs_for_verify["output"])))
                working_root = Path(current["directory"]).resolve()
                output_root = (Path(str(inputs_for_verify['output'])).resolve()
                               if inputs_for_verify.get('output') else None)
                approved_roots = [repo.resolve(), Path(verify_output).parent.resolve()
                                  if verify_output else repo.resolve()]
                raw_checkpoint = str(inputs_for_verify.get("checkpoint") or "")
                policy_parent: Path | None = None
                if raw_checkpoint:
                    try:
                        checkpoint_path = Path(raw_checkpoint).expanduser().resolve(strict=True)
                        parent = checkpoint_path.parent.resolve(strict=True)
                        policy_root = checkpoint_path if checkpoint_path.is_dir() else parent
                        if (checkpoint_path.exists() and
                                any(policy_root.is_relative_to(root) for root in approved_roots)):
                            policy_parent = policy_root
                    except OSError:
                        pass

                examples = (evidence.get("candidate_paths") or
                            evidence.get("examples") or [])
                declared_path = ""
                if evidence.get("matched") == 1 and examples:
                    raw = Path(str(examples[0]))
                    candidate = raw if raw.is_absolute() else evidence_root / raw
                    if not candidate.is_symlink():
                        try:
                            resolved = candidate.resolve(strict=True)
                            if (resolved.is_file() and any(
                                    resolved.is_relative_to(root)
                                    for root in approved_roots)):
                                declared_path = str(resolved)
                        except OSError:
                            pass

                # A declared video/checkpoint glob can match many files while a single
                # structured episode sidecar beside those outputs carries the native score.
                # Keep those candidates even when the declared artifact itself is not unique;
                # the metric binder must see the schema before preferring an aggregate log.
                sidecar_dirs: dict[Path, str] = {}
                for value in examples[:12]:
                    raw = Path(str(value))
                    candidate = raw if raw.is_absolute() else evidence_root / raw
                    try:
                        resolved = candidate.resolve(strict=True)
                        if not resolved.is_file() or candidate.is_symlink():
                            continue
                        if policy_parent and resolved.is_relative_to(policy_parent):
                            root_kind = "policy_parent"
                        elif output_root and resolved.is_relative_to(output_root):
                            root_kind = "output"
                        elif resolved.is_relative_to(working_root):
                            root_kind = "working_directory"
                        else:
                            continue
                        sidecar_dirs[resolved.parent] = root_kind
                    except OSError:
                        continue

                structured: list[dict[str, Any]] = []
                seen_structured: set[Path] = set()
                for directory, root_kind in sorted(sidecar_dirs.items(),
                                                   key=lambda item: str(item[0])):
                    try:
                        adjacent = sorted(directory.iterdir(), key=lambda path: path.name)
                    except OSError:
                        adjacent = []
                    for path in adjacent:
                        if len(structured) >= 12:
                            break
                        if path.suffix.lower() not in {".json", ".csv"} or path.is_symlink():
                            continue
                        try:
                            resolved = path.resolve(strict=True)
                            stat = resolved.stat()
                            if (resolved in seen_structured or not resolved.is_file() or
                                    stat.st_mtime < started or
                                    not any(resolved.is_relative_to(root)
                                            for root in approved_roots)):
                                continue
                            if root_kind == "policy_parent":
                                if policy_parent is None or not resolved.is_relative_to(policy_parent):
                                    continue
                                relative = resolved.relative_to(policy_parent)
                            elif root_kind == "output":
                                if output_root is None or not resolved.is_relative_to(output_root):
                                    continue
                                relative = resolved.relative_to(output_root)
                            else:
                                if not resolved.is_relative_to(working_root):
                                    continue
                                relative = resolved.relative_to(working_root)
                            structured.append({
                                "path": str(resolved), "root": root_kind,
                                "relative_path": relative.as_posix(),
                                "suffix": resolved.suffix.lower(),
                                "size_bytes": stat.st_size,
                                "mtime": stat.st_mtime,
                                "sha256": digest(resolved),
                            })
                            seen_structured.add(resolved)
                        except (OSError, ValueError):
                            continue
                verified_artifact = {
                    "attempt_started": started,
                    "pattern": evidence.get("pattern"),
                    "matched": evidence.get("matched"),
                    "declared_path": declared_path,
                    "working_directory": str(working_root),
                    "output_directory": str(output_root) if output_root else "",
                    "policy_parent": str(policy_parent) if policy_parent else "",
                    "found_beside_the_command": bool(
                        evidence.get("found_beside_the_command")),
                    "structured_candidates": structured,
                }
        return {"ok": done.returncode == 0,
                "native_evidence_ref":getattr(done, "evidence_ref", ""),
                "native_diagnostics": native_postconditions['native_diagnostics'],
                "training_progress":progress if stage == "train" and require_step_control else {},
                "verification_output":str(inputs_for_verify.get("output") or ""),
                "error": (done.stderr or "") + "\n" + (done.stdout or ""),
                **({"verified_artifact": verified_artifact}
                   if verified_artifact is not None else {})}

    source, parameters, log = None, {}, []
    hint, attempted = "", []
    # What went wrong before, and what fixed it. Read once and consulted on every round:
    # a failure that matches something already solved is answered from memory before the
    # model is asked to reason about it, which is AutoSOTA's "retrieval before repair" and
    # the ordering is the point -- the model's turn is worth spending on what memory does
    # not cover.
    memory = failure_memory.FailureMemory()
    #: The last revision, and the failure it was made against, so that the next round can
    #: say whether it worked. A remedy is only worth storing with its outcome attached.
    last_change: dict[str, Any] = {}
    last_failure = ""
    #: What the search has done, for the observer. Each round's change and the state it left
    #: behind -- the state and not the round, because a round that rewrote the invocation into
    #: a different spelling of the same command has moved nothing.
    trace: list[dict[str, Any]] = []
    #: How many rounds in a row answered "the command is not the problem". One is a finding
    #: worth acting on -- the obstacle becomes the next draft's guidance. Three is the loop
    #: being told the same thing three times, and the thing to question is the requirement.
    #:
    #: The exit used to be `said in hint`, which compares the *wording*: RoboTwin's training
    #: stage was told thirty-seven times that the dataset its script expects is not on this
    #: machine, in thirty-seven different sentences, and each one reset the test. A loop that
    #: only stops when a model repeats itself verbatim does not stop.
    obstacles_in_a_row = 0
    obstacle = ""
    last_failure_signature = ""
    same_failure_count = 0
    def printed_keys() -> list[str]:
        """The config keys the program listed about itself in its most recent output.

        This is the fact the generator keeps guessing at. It is discovered during a revision,
        but the thing that writes the command never sees a revision -- so unless it is handed
        over, every round re-guesses the same names from what the value means. LIBERO's output
        names `folder` on every run, and rounds went to `data_root`, `data.dataset_path`,
        `+data.dataset_path` and `data.dataset_root` regardless.
        """
        said = str((log[-1].get("said") if log else "") or "")
        return keys_the_program_printed(said)

    for _ in range(rounds):
        # Nothing a round does may end the loop. Four separate crashes came out of this one
        # function -- a draft that dropped the interpreter, a survey that walked into a file
        # it could not read, a handler that named a variable the failure had prevented from
        # being bound, and a model that answered with a paragraph -- and each one discarded
        # the rounds that had already succeeded along with the rounds that had not started.
        # A round that raises is a round that produced nothing, and the budget bounds it.
        try:
            if verify_output:
                resources['previous_native_verifications'] = [
                    attempt for attempt in recent_attempts(Path(verify_output).parent)
                    if attempt.get('stage') == stage][:3]
            resources['stage_postconditions'] = {
                'declared_artifact':row.get('artifact'),
                'instruction':'Verify the fresh source-backed artifact under the caller output '
                    'or working directory. The output slot is a root, not the repository default '
                    'directory name. Inspect native writers and prior receipt output roots; '
                    'an artifact mismatch after rc=0 is not proof the evaluator never ran. '
                    'Request invocation repair when the declaration needs correction rather '
                    'than blindly rerunning identical evaluations. For command verification '
                    'prefer the caller episodes target as a small smoke test; a repository '
                    'full evaluation example is not a required verification work budget. '
                    'Do not silently freeze the full example episode count for later formal '
                    'evaluations: read the caller episode slot and native source semantics.'}
            source, parameters, log = generate_argv(
                client, stage, entrypoint=str(row.get("entrypoint")),
                invocation=str(row.get("invocation")),
                repository_files=repository_files,
                declared_parameters={p["name"]: p["value"] for p in (row.get("parameters") or [])
                                     if isinstance(p, dict) and "name" in p},
                verify=verify, inputs_for_verify=inputs_for_verify, attempts=attempts,
                verification_context=resources,
                verification_inputs_factory=(next_verification_inputs
                    if unique_verification_outputs and verify_output else None),
                hint=hint, require_step_control=require_step_control,
                require_evaluation_progress=require_evaluation_progress,
                agent_timeout_seconds=agent_timeout_seconds)
        except Exception as exc:                                       # noqa: BLE001
            log = [{"stage": stage, "attempt": 0, "status": "the round raised",
                    "error": redact(f"{type(exc).__name__}: {exc}")[:600]}]
            if on_event:
                on_event(stage, log)
            if isinstance(exc, TimeoutError) and "wall-clock budget exhausted" in str(exc):
                break
            continue
        if on_event:
            on_event(stage, log)
        if source is not None:
            if last_change:
                memory.record(last_failure, last_change, outcome="accepted", repo=repo.name)
                # And if that was the repository where the pairing stopped being an anecdote,
                # write it down as a method. Guarded: a method library that cannot be written
                # is a library that learns nothing new, and it is not a reason to discard the
                # command that just verified.
                try:
                    failure_memory.distil()
                except Exception:                                    # noqa: BLE001
                    pass
            break
        # The full output when there is one: the excerpt is what the prompt holds, and the
        # keys a program printed about itself live outside it.
        failure = str(log[-1].get("said") or log[-1].get("error") or "")
        failure_signature = failure_memory.signature(failure)
        if failure_signature and failure_signature == last_failure_signature:
            same_failure_count += 1
        else:
            same_failure_count = 1 if failure_signature else 0
        last_failure_signature = failure_signature
        if same_failure_count >= 3:
            obstacle = ("the same stage-verification failure recurred across three adapted "
                        "drafts; stopping this bounded derivation rather than spending more "
                        f"simulator runs on an unchanged failure class ({failure_signature})")
            trace.append({"change": {}, "state": row, "failure": failure,
                          "looked": int(log[-1].get("looked") or 0)})
            if on_event:
                on_event(stage, [{"status": "same failure repeated; stopping bounded "
                                            "derivation", "error": redact(obstacle)[:800]}])
            break
        # The same failure coming back after a revision means the revision did not fix it.
        if last_change and failure_memory.signature(failure) == failure_memory.signature(
                last_failure):
            memory.record(failure, last_change, outcome="refuted", repo=repo.name)
        # What is wrong, and then what to do about it. Two questions, two prompts -- they were
        # one, and it had grown to carry eleven concerns and stopped answering either well.
        # Both are read-only over the record and both are guarded, for the same reason: a
        # memory that cannot be read and a monitor that cannot count are worse than no memory
        # and no monitor, and neither is a reason to discard a round.
        try:
            recalled = memory.recall(failure)
            picture = monitor.observe(trace)
        except Exception:                                            # noqa: BLE001
            recalled, picture = {}, {}
        found = diagnose(client, stage, row, repo=repo, argv=_last_argv(log), failure=failure,
                         on_event=on_event, recalled=recalled,
                         searched=monitor.guidance(picture), attempted=attempted,
                         native_diagnostics=log[-1].get('native_diagnostics'),
                         agent_timeout_seconds=agent_timeout_seconds)
        if found.get("finding") and on_event:
            # The finding, in the record. It is the answer to "why did the loop do that" for
            # this round, and until now the only trace of it was in one response body.
            on_event(stage, [{"status": "diagnosed", "error": redact(found["finding"])[:600],
                              "evidence": redact(found.get("evidence") or "")[:300]}])
        inspections = int(found.get("looked") or 0)
        about = str(found.get("about") or "the command")
        change = revise_invocation(client, stage, row, repo=repo,
                                   argv=_last_argv(log), failure=failure,
                                   finding=str(found.get("finding") or ""), about=about,
                                   python=str(current.get("environment", {}).get("python", "")),
                                   on_event=on_event, attempted=attempted,
                                   recalled=recalled,
                                   how_the_search_is_going=monitor.guidance(picture),
                                   native_diagnostics=log[-1].get('native_diagnostics'),
                                   agent_timeout_seconds=agent_timeout_seconds)
        if on_event:
            on_event(stage, [{"status": "revising the invocation",
                              "error": redact(json.dumps(change, ensure_ascii=False))[:1200]}])
        # A finding the diagnosis called "about the repository" is not a finding that stops
        # the loop -- it is what `staging` is for, and the reviser was told so. Treating it as
        # terminal is what the loop did: a diagnosis said "the real key names are in
        # TASK_CONFIGS.json" -- naming the file it had just read -- and was filed as
        # `no_change_can_fix_this`, so the stage ended with the answer in the record and
        # nothing acting on it.
        repository_repair = about == "the repository"
        if change is None or change.get("obstacle") or (
                found.get("no_change_can_fix_this") and not repository_repair):
            # The answer is "the command is wrong, not where it runs" -- which is a finding
            # about the *command*, and the thing that writes commands is still in this loop.
            # Ending here threw that away: the model said "the dataset path is redirected via
            # the `data.root` parameter" and nothing was ever told to write that parameter.
            said = str((change or {}).get("obstacle") or found.get("no_change_can_fix_this")
                       or "no revision offered")
            obstacles_in_a_row += 1
            obstacle = said
            trace.append({"change": {}, "state": row, "failure": failure,
                          "looked": inspections})
            if on_event:
                on_event(stage, [{"status": "not an invocation problem, so the command is "
                                            "regenerated with that as guidance",
                                  "error": redact(said)[:600]}])
            # "The command is not the problem" three times is the answer, not an instruction
            # to try again. Reported with the obstacle attached, because a derivation that
            # stops without saying why is the shape of failure this module exists to avoid.
            if not said or said in hint or obstacles_in_a_row >= 3:
                break
            # The keys the program printed are the answer to what the generator is guessing
            # at, and the generator never sees a revision. Handed over, not hoped for.
            keys = printed_keys()
            hint = said + ("\n\nThe program printed the keys its configuration has: "
                           + ", ".join(keys) + ". Use one of those, exactly as spelled."
                           if keys else "")
            if on_event:
                on_event(stage, [{"status": "guidance for the next draft",
                                  "error": redact(hint)[:1400]}])
            continue
        # A revision returns parameters as a name -> value map; the derivation carries them
        # as a list of records. Merged in the derivation's shape so the next round reads
        # what it expects.
        merged = {k: v for k, v in change.items() if k not in {"why", "parameters"}}
        merged = preserve_local_artifact(row, merged)
        named = str(merged.get("interpreter") or "").strip()
        if named and not Path(named).is_file():
            # A stage whose interpreter does not exist cannot run, and recording it would
            # make the next attempt fail for a reason the record already knew.
            merged.pop("interpreter", None)
        if isinstance(change.get("parameters"), dict):
            existing = {p["name"]: p for p in (row.get("parameters") or [])
                        if isinstance(p, dict) and "name" in p}
            for name, value in change["parameters"].items():
                existing[str(name)] = {**(existing.get(str(name)) or {}), "name": str(name),
                                       "value": value,
                                       "evidence": "revised after the command was run"}
            merged["parameters"] = list(existing.values())
        last_change, last_failure = merged, failure
        trace.append({"change": merged, "state": {**row, **merged}, "failure": failure,
                      "looked": inspections})
        if all(row.get(k) == v for k, v in merged.items()):
            # The reviser looked at the failure and concluded the invocation is already
            # right -- which is a statement about the *command*, and the thing that writes
            # commands is still in this loop. LIBERO reached exactly here: the invocation it
            # had was the correct one, the generated argv appended an override hydra refuses,
            # and the reasoning that said so ("stop passing output_dir") was thrown away
            # because the round it arrived in changed nothing about the invocation. Retrying
            # the same command is pointless, so the reasoning goes back as guidance instead --
            # once, since a hint that produced no change must not be repeated.
            said = str(change.get("why") or "").strip()
            obstacles_in_a_row += 1
            obstacle = said
            trace.append({"change": {}, "state": row, "failure": failure,
                          "looked": inspections})
            if not said or said in hint or obstacles_in_a_row >= 3:
                if on_event:
                    on_event(stage, [{"status": "the reviser repeated itself, so the loop "
                                                "stops rather than spin",
                                      "error": redact(json.dumps(change, ensure_ascii=False))[:400]}])
                break
            keys = printed_keys()
            hint = said + ("\n\nThe program printed the keys its configuration has: "
                           + ", ".join(keys) + ". Use one of those, exactly as spelled."
                           if keys else "")
            if on_event:
                on_event(stage, [{"status": "the invocation is already right, so the command "
                                            "is regenerated with guidance",
                                  "error": redact(hint)[:1400]}])
            continue
        obstacles_in_a_row = 0
        obstacle = ""
        attempted.append({"changed": merged, "because": change.get("why")})
        row = {**row, **merged}
        if on_revision:
            on_revision(stage, dict(row), str(change.get('why') or
                found.get('finding') or 'Agent revised the failed invocation'),
                str(log[-1].get('native_evidence_ref') or ''))
        current["directory"] = invocation_directory(row, repo=repo, default=repo)
        current["environment"] = local_environment({
            **os.environ, **(base_environment or {}),
            **invocation_environment(row, repo=repo,
                                     base={**os.environ, **(base_environment or {})})})
        current["stdin"] = invocation_stdin(row)
    if source is None and obstacle:
        # Said out loud rather than left as an empty log: `no runnable command` with no
        # reason is the shape of failure this module exists to avoid, and the obstacle is a
        # finding about the benchmark -- RoboTwin's training script wants a dataset layout
        # this machine does not have, which is worth knowing and is not a command's fault.
        log = list(log) + [{"stage": stage, "attempt": 0, "status": "no command can fix this",
                            "error": redact(obstacle)[:800]}]
    return source, parameters, log, row


def verify_kept(source: str, stage: str, row: dict[str, Any], *, repo: Path,
                inputs: dict[str, Any], base_environment: dict[str, str] | None = None,
                timeout: int = 600) -> tuple[bool, str]:
    """Run a command that was verified before, to see whether it still is.

    A kept command is a proven artifact; a changed device is a different question about it.
    Re-*deriving* throws the proof away and gambles on a generation that succeeds perhaps one
    time in twelve -- which is what happened: a command that had run a twelve-hour
    measurement was discarded because the machine it was verified on had been recorded as
    unpinned, and the derivation spent forty rounds failing to reproduce it.

    Re-*verifying* is deterministic, costs one run, and answers the actual question.
    """
    from .declarative_backend import (invocation_directory, invocation_environment,
                                      invocation_stdin)
    from .patch_validation import checked_function
    function = checked_function(source, argv_function_name(stage))
    argv = function({**blank_inputs(), **inputs})
    environment = {**os.environ, **(base_environment or {}),
                   **invocation_environment(row, repo=repo,
                                            base={**os.environ, **(base_environment or {})})}
    fed = invocation_stdin(row)
    try:
        done = bounded_run(isolated_argv([str(a) for a in argv], output=repo, repo=repo,
                                         require_pid_namespace=True), timeout=timeout,
                           cwd=invocation_directory(row, repo=repo, default=repo),
                           env=environment, input=fed)
    except subprocess.TimeoutExpired:
        return False, "launch worked but the command timed out before verification completed"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    said = (done.stderr or "") + "\n" + (done.stdout or "")
    return done.returncode == 0, said


def _last_argv(log: list[dict[str, Any]]) -> list[str]:
    """What the command looked like when it was refused, for the revision to look at."""
    for entry in reversed(log):
        if isinstance(entry.get("argv"), list) and entry["argv"]:
            return [str(a) for a in entry["argv"]]
    return []


_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|COOKIE)"
    r"[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
_PRIVATE_FILE_TOKEN = re.compile(
    r"(?i)(?:[^\s,;\"']*/)?[^\s,;\"']+"
    r"\.(?:pt|pth|ckpt|safetensors|pkl|pickle|mp4|mkv|avi|mov|webm|key|pem)"
    r"(?=[\s,;\"')\]}:]|$)"
)
_ABSOLUTE_PATH_TOKEN = re.compile(r"(?<![\w}])/(?:[^\s,;\"'<>|&]+)")
_PROMPT_SECRET_KEY = re.compile(
    r"(?:^|_)(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIALS?|AUTH(?:ORIZATION)?|COOKIE)(?:_|$)",
    re.IGNORECASE)


def _safe_diagnostic_text(value: str, *, repo: Path) -> str:
    """Remove credentials and private artifact/host paths before an external model sees text."""
    checkout = Path(repo).resolve(strict=False)
    home = Path.home().resolve(strict=False)
    text = redact(value).replace(str(checkout), "{repo}")
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)

    def safe_path(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ").]}:":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        resolved = Path(raw).expanduser().resolve(strict=False)
        if resolved.is_relative_to(checkout):
            relative = resolved.relative_to(checkout)
            sensitive_parts = {"checkpoint", "checkpoints", "weight", "weights", "video",
                               "videos", "recording", "recordings", "secret", "secrets",
                               "credentials"}
            if (any(part.lower() in sensitive_parts for part in relative.parts) or
                    resolved.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors",
                                                ".pkl", ".pickle", ".mp4", ".mkv", ".avi",
                                                ".mov", ".webm", ".key", ".pem"}):
                return "[WITHHELD_ASSET_PATH]" + trailing
            return ("{repo}" if not relative.parts else
                    "{repo}/" + relative.as_posix()) + trailing
        if resolved.is_relative_to(home):
            return "{host-home}/[REDACTED]" + trailing
        allowed_system = (Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"),
                          Path("/lib64"), Path("/opt"), Path("/nix/store"))
        if any(resolved == root or resolved.is_relative_to(root) for root in allowed_system):
            return str(resolved) + trailing
        if (any(part.lower() in _INSPECTION_PRIVATE_PARTS for part in resolved.parts) or
                resolved.suffix.lower() in _INSPECTION_PRIVATE_SUFFIXES):
            return "[WITHHELD_ASSET_PATH]" + trailing
        return "[EXTERNAL_PATH]" + trailing

    text = _ABSOLUTE_PATH_TOKEN.sub(safe_path, text)
    return _PRIVATE_FILE_TOKEN.sub("[WITHHELD_ARTIFACT]", text)


def _safe_diagnostic_value(value: Any, *, repo: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_diagnostic_value(item, repo=repo)
                for key, item in value.items()
                if not _PROMPT_SECRET_KEY.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_safe_diagnostic_value(item, repo=repo) for item in value]
    if isinstance(value, str):
        return _safe_diagnostic_text(value, repo=repo)
    return value


def _told_about(row: dict[str, Any], stage: str, argv: list[str], failure: str, *,
                repo: Path, attempted: list[dict[str, Any]] | None,
                recalled: dict[str, Any] | None, searched: str,
                may_look: bool = False,
                native_diagnostics: dict[str, Any] | None = None) -> str:
    """What both halves of a revision are shown: the stage, the command, the failure.

    Shared rather than written twice, because the two are asked about the same thing and a
    payload that drifted between them would mean the diagnostician and the revisor disagreeing
    about what happened.
    """
    payload = {
        "stage": stage,
        "as_derived": {k: row.get(k) for k in
                       # `staging` is in this list and was not, and that is why the
                       # correction a diagnosis asked for could not be made. RoboTwin's
                       # training stage reached a finding that read "the staging script itself
                       # must be corrected to discover by content and to name the destination
                       # files the way the loader opens them" -- and the reviser was never
                       # shown the staging script, so the one thing the finding named was the
                       # one thing it could not see. A field a stage can carry is a field its
                       # correction has to start from.
                       ("entrypoint", "invocation", "working_directory", "environment",
                        "artifact", "parameters", "staging")},
        "selected_stage_interpreter": {
            "execution_ref": "{verified_interpreter}" if row.get('interpreter') else None,
            "authority": "already selected stage executable, not an import/rollout verdict",
            "instruction": "Use the execution_ref literally if the wrapper requires a Python "
                           "override. It resolves outside/inside checkout exactly as selected; "
                           "never guess {repo}/environments or reinterpret it as checkout-relative."},
        "command_built": argv,
        "program_said": error_excerpt(failure),
        "native_execution_diagnostics": native_diagnostics or {},
        # The value, not a description of it. This said "the checkout's absolute path" -- a
        # sentence that tells a reader what the placeholder means and not what it is, so every
        # absolute path it wrote was a combination it had to guess at. One guessed
        # `.../AutoSimSOTA` for `{repo}` and asked for `.../AutoSimSOTA/RoboTwin/RoboTwin/
        # scripts/eval_policy_xpolicylab.py`, one directory too deep; seven inspections went
        # into that, each returning "no such file" -- a true answer to a question that was
        # wrong about where the checkout is.
        "placeholders": {"repo": {"value": "{repo}",
                                  "means": "the checkout, and the root that every "
                                           "repository-relative path in this payload is "
                                           "relative to"}},
        # Only for the one that can. Offering an inspection to a reviser that has none is a
        # trap: it will ask, and the asking is not a revision.
        **({"what_you_may_look_at": {
            "look_at": "a non-sensitive source/config file inside the checkout",
            "look_at_dir": "a checkout directory (private asset directories are withheld)",
            "run": "one argv in a read-only, networkless namespace with sanitized environment; "
                   "no shell syntax or host paths"}} if may_look else {}),
        "data_directories_near_the_checkout": data_near(repo),
        "files_the_program_could_not_open": files_the_program_could_not_open(
            failure, roots=missing_roots(repo)),
        "configuration_keys_the_program_printed": keys_the_program_printed(failure),
        "what_this_failure_has_looked_like_before": recalled or {},
        "how_the_search_is_going": searched or "nothing recorded yet, so nothing to say",
        "changes_already_tried_and_what_they_were_for": attempted or [],
    }
    return json.dumps(_safe_diagnostic_value(payload, repo=repo),
                      ensure_ascii=False, default=str)


def _where_it_runs(row: dict[str, Any], *, repo: Path) -> tuple[Path, dict[str, str]]:
    """Where an inspection runs, and with what. The stage's own directory and environment,
    because the question is about *this* stage: `which ffmpeg` is only meaningful in the
    environment the stage will run in, and a config file is only that file when the working
    directory is the one that holds the entry point."""
    from .declarative_backend import invocation_directory, invocation_environment
    directory = invocation_directory(row, repo=repo, default=repo)
    environment = {**os.environ, **invocation_environment(row, repo=repo, base=dict(os.environ))}
    return directory, environment


def diagnose(client: Any, stage: str, row: dict[str, Any], *, repo: Path, argv: list[str],
             failure: str, attempts: int = 2, on_event: Any = None,
             recalled: dict[str, Any] | None = None, searched: str = "",
             attempted: list[dict[str, Any]] | None = None,
             native_diagnostics: dict[str, Any] | None = None,
             agent_timeout_seconds: float = 180) -> dict[str, Any]:
    """What is wrong, from the failure and from whatever the machine can be asked.

    A separate question from what to do about it, and separate here because they were one
    prompt and it stopped working. That prompt had grown to nine thousand characters carrying
    eleven distinct concerns -- the inspection protocol, the five fields, the obstacle rules,
    the memory, the monitor, three named failure classes, several warnings -- and its answers
    degraded from "the entrypoint is a bash script and the command runs it with Python" to
    "Let me look at what actually failed". AutoSOTA's whole agent architecture exists for that
    observed failure and gives it a name: **monolithic LLM prompt collapse under long-horizon
    execution**. One role per prompt is the fix, and the roles were already there in the
    question being asked.

    Returns what it found, or an empty dict when it could not say. **Never raises**: a
    diagnosis that dies fails exactly when the loop is already in trouble.
    """
    directory, environment = _where_it_runs(row, repo=repo)
    payload = _told_about(row, stage, argv, failure, repo=repo, attempted=attempted,
                          recalled=recalled, searched=searched, may_look=True,
                          native_diagnostics=native_diagnostics)
    looked: list[dict[str, Any]] = []
    judged = 0
    while judged < attempts:
        try:
            content, _ = client.chat_with_metadata(
                sanitize_model_text(DIAGNOSE_SYSTEM),
                sanitize_model_text(payload, local_roots=(repo,)),
                max_tokens=2000, timeout=agent_timeout_seconds, thinking="disabled")
        except Exception:                                            # noqa: BLE001
            return {}
        try:
            asked = _objects(content)
            requests = [one for one in asked
                        if any(key in one for key in ("look_at", "look_at_dir", "run"))]
            if requests:
                # Every one of them, not only the first. A diagnostician asking to see a
                # directory and then the file in it wrote both, in one reply, separated by a
                # blank line -- and taking the first object alone ran half of what it asked
                # for and then reported that it had answered with nothing.
                if len(looked) + len(requests) > INSPECTIONS_PER_REVISION:
                    raise ValueError(
                        f"{INSPECTIONS_PER_REVISION} inspections is the whole budget for one "
                        f"diagnosis and you asked for {len(requests)} more; answer with what "
                        f"you have.")
                for request in requests:
                    seen = inspect(request, repo=repo, directory=directory,
                                   environment=environment)
                    looked.append(seen)
                    # Said out loud, in the event stream. What a diagnosis looked at is part
                    # of why it concluded what it concluded, and a run whose record shows only
                    # the conclusion cannot be read back -- the same fault as a decision
                    # without its grounds.
                    if on_event:
                        on_event(stage, [{"status": "looked",
                                          "asked": json.dumps(request, ensure_ascii=False)[:200],
                                          "found": json.dumps(
                                              {k: v for k, v in seen.items() if k != "content"},
                                              ensure_ascii=False)[:400],
                                          "error": seen.get("error", "")}])
                    payload = payload + "\n\n### WHAT YOU ASKED TO SEE\n" + json.dumps(
                        seen, ensure_ascii=False, default=str)
                continue
            judged += 1
            value = _object(content)
            finding = str(value.get("finding") or "").strip()
            if not finding:
                raise ValueError("a diagnosis needs a `finding`: what is wrong")
            return {"finding": finding,
                    "evidence": str(value.get("evidence") or "").strip(),
                    # How many times the diagnosis stopped to look. The observer reads this
                    # to tell a run that is circling from one that is not looking at
                    # anything -- and until now nothing produced it, so that signal fired on
                    # every round regardless. A check with no producer is a check that
                    # reports whatever the reader wants to hear.
                    "looked": len(looked),
                    # Whether the finding is about the command or about the repository. The
                    # second is not a reason to stop: it is what repository repair is for, and
                    # a loop that has no way to say so concludes "no command can fix this" and
                    # ends -- which is what every one of RoboTwin's stages did, each with a
                    # correct diagnosis and no way to act on it.
                    "about": str(value.get("about") or "the command").strip().lower(),
                    "no_change_can_fix_this": (str(value.get("no_change_can_fix_this") or "")
                                               .strip() or None),
                    "changes_that_cannot_fix_it": (str(value.get("changes_that_cannot_fix_it")
                                                       or "").strip())}
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            judged += 1
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(f"{type(exc).__name__}: {exc}")[:400],
                 "instruction": "Return a finding, or ask to look at something."},
                ensure_ascii=False)
    return {}


def revise_invocation(client: Any, stage: str, row: dict[str, Any], *, repo: Path,
                      argv: list[str], failure: str, finding: str = "", attempts: int = 2,
                      python: str = "", on_event: Any = None,
                      recalled: dict[str, Any] | None = None,
                      how_the_search_is_going: str = "",
                      about: str = "the command",
                      attempted: list[dict[str, Any]] | None = None,
                      native_diagnostics: dict[str, Any] | None = None,
                      agent_timeout_seconds: float = 180
                      ) -> dict[str, Any] | None:
    """The invocation change a finding calls for.

    Given the diagnosis and nothing to look at: what to look at was the previous question's
    business, and a reviser that can still investigate is a reviser that will, at the moment
    it should be answering.
    """
    payload = _told_about(row, stage, argv, failure, repo=repo, attempted=attempted,
                          recalled=recalled, searched=how_the_search_is_going,
                          native_diagnostics=native_diagnostics)
    if finding:
        payload = payload + "\n\n### WHAT IS WRONG\n" + finding
    if about == "the repository":
        payload = payload + (
            "\n\n### AND IT IS ABOUT THE REPOSITORY, NOT THE COMMAND\n"
            "The diagnosis says the fault is a property of the repository: a path it computes "
            "and does not have, a name it opens and does not hold, a step that generates what "
            "a later stage reads. No value in the invocation fields below fixes that. Answer "
            "with `staging`: the list of shell commands that make it true, run in the stage's "
            "directory before it. They must be safe to run twice and they must not change "
            "what is measured.")
    judged = 0
    while judged < attempts:
        try:
            content, _ = client.chat_with_metadata(
                sanitize_model_text(REVISE_SYSTEM),
                sanitize_model_text(payload, local_roots=(repo,)),
                max_tokens=2000, timeout=agent_timeout_seconds, thinking="disabled")
        except Exception:                                            # noqa: BLE001
            # A client that cannot answer is a round that produced nothing. It is not a
            # reason to end the stage: this call sits on the loop's hot path, and the claim
            # that "a round that raises is a round that produced nothing" was only true of
            # the half of the round that happened to be inside a `try`.
            return None
        try:
            value = _object(content)
            # `str(None).strip()` is "None", which is truthy -- so a model that answered the
            # question with a null reported an obstacle of null, and the caller read the
            # falsy value as "no obstacle, proceed" and retried the same failing thing.
            said = value.get("not_an_invocation_problem")
            if isinstance(said, str) and said.strip():
                return {"obstacle": said.strip()}
            if not isinstance(value.get("environment") or {}, dict):
                raise ValueError("environment must be a map of variable to value")
            # The prompt offers `invocation` and `parameters` as "if that was also wrong", so
            # a null means *unchanged* -- and merging it anyway set the row's invocation to
            # None, after which the next draft was built from the string "None". A field the
            # model declined to answer is not a field it answered with nothing. An empty
            # string is how a model spells "unchanged" the second time it is asked.
            # `environment`, by contrast, is a map and an empty one is a real answer.
            value = {k: v for k, v in value.items()
                     if v is not None and not (k != "environment" and isinstance(v, str)
                                               and not v.strip())}
            # A list of records is the shape the derivation itself carries parameters in, and
            # a model that has just been shown that shape answers in it. Accepting only the
            # map shape made the change look like it carried no parameters at all.
            if isinstance(value.get("parameters"), list):
                value["parameters"] = {
                    str(entry.get("name")): entry.get("value")
                    for entry in value["parameters"]
                    if isinstance(entry, dict) and entry.get("name")}
            if not str(value.get("working_directory", "")).strip():
                raise ValueError("working_directory is required")
            if "staging" in value:
                commands = value["staging"]
                if isinstance(commands, str):
                    commands = [commands]
                if not isinstance(commands, list) or not all(
                        isinstance(one, str) and one.strip() for one in commands):
                    raise ValueError("staging must be a list of shell commands, each a "
                                     "non-empty string; it is run before the stage and the "
                                     "stage does not run if it fails")
                value["staging"] = [one.strip() for one in commands]
            # The sentinel is the answer to a question, not a field of the invocation. Left
            # in, the caller merges it into the stage and the next payload carries a key the
            # prompt does not define.
            def resolve_interpreter_ref(item: Any) -> Any:
                if isinstance(item, dict):
                    return {key:resolve_interpreter_ref(part) for key,part in item.items()}
                if isinstance(item, list):
                    return [resolve_interpreter_ref(part) for part in item]
                if isinstance(item, str) and '{verified_interpreter}' in item:
                    interpreter = str(row.get('interpreter') or '')
                    if not interpreter:
                        raise ValueError('no selected stage interpreter backs this execution_ref')
                    if '{repo}/{verified_interpreter}' in item:
                        raise ValueError('verified_interpreter is not a checkout-relative path')
                    return item.replace('{verified_interpreter}',interpreter)
                return item
            return resolve_interpreter_ref({k: v for k, v in value.items()
                                            if k != "not_an_invocation_problem"})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            # A refused answer has been judged. Counting it only on the accepted path made
            # the budget unbounded: a reviser that kept asking to look past its inspection
            # allowance was refused on every turn and the refusal never cost it anything --
            # the loop spun until the suite's timeout, which is how this was found.
            judged += 1
            payload = payload + "\n\n### YOUR PREVIOUS ANSWER WAS REFUSED\n" + json.dumps(
                {"error": redact(f"{type(exc).__name__}: {exc}")[:400],
                 "instruction": "Answer with the invocation fields."}, ensure_ascii=False)
    return None


def problems_in(value: dict[str, Any], repo: Path) -> list[str]:
    """Faults in a derived answer, as messages. Empty means well-formed, not correct.

    Two questions, both answerable without running anything: is the answer complete enough
    to act on, and do the paths it names exist? Neither settles whether the entry point is
    the right one, and the report does not pretend otherwise.
    """
    if not isinstance(value, dict):
        return ["answer must be one JSON object"]
    stages = value.get("stages")
    if not isinstance(stages, dict):
        return ["stages must be an object"]
    problems: list[str] = []
    extras = sorted(set(stages) - set(STAGES))
    for name in extras:
        row = stages[name]
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name) or
                not isinstance(row, dict) or row.get("available") is not True or
                any(not str(row.get(key) or "").strip()
                    for key in ("role", "why_required", "evidence"))):
            problems.append(f"stages has no such entry: {name!r}; extra stages need a "
                            "safe name, availability, role, why_required, and evidence")
    for name in (*STAGES, *extras):
        row = stages.get(name)
        if not isinstance(row, dict):
            problems.append(f"stages.{name} must be answered, including as unavailable")
            continue
        if not isinstance(row.get("available"), bool):
            problems.append(f"stages.{name}.available must be true or false")
            continue
        if not row["available"]:
            if not str(row.get("why", "")).strip():
                problems.append(f"stages.{name} is reported unavailable and must say why")
            continue
        entrypoint = str(row.get("entrypoint", "")).strip()
        if not entrypoint:
            problems.append(f"stages.{name} claims to be available and names no entry point")
            continue
        target = repo / entrypoint
        if not target.exists():
            problems.append(f"stages.{name}.entrypoint does not exist: {entrypoint}")
        elif target.is_dir():
            # A directory is where to look, not what to run. Reported rather than rejected:
            # it is a weaker answer, and the difference is worth keeping visible.
            problems.append(f"stages.{name}.entrypoint is a directory, not a file to run: "
                            f"{entrypoint}")
        for key in ("invocation",):
            if not str(row.get(key, "")).strip():
                problems.append(f"stages.{name}.{key} is required when available")
        if name == "collect" and not str(row.get("evidence") or "").strip():
            problems.append("stages.collect.evidence is required when available; cite source "
                            "showing an independent producer mode and reusable data output")
        workdir = str(row.get("working_directory", "") or "").strip()
        if workdir and "{" not in workdir and not (repo / workdir).exists():
            # A declared directory has to be one, so it is checked rather than trusted.
            problems.append(f"stages.{name}.working_directory does not exist: {workdir}")
        environment = row.get("environment")
        if environment is not None and not isinstance(environment, dict):
            problems.append(f"stages.{name}.environment must be a map of variable to value")
        artifact = str(row.get("artifact", "")).strip()
        if not artifact:
            problems.append(f"stages.{name}.artifact is required when available")
        elif " " in artifact:
            # A sentence cannot be looked for. The artifact is what the execution check
            # waits for, so it has to be a path or a glob relative to the stage's output,
            # not a description of one.
            problems.append(f"stages.{name}.artifact must be a path or glob relative to the "
                            f"output directory, not a sentence: {artifact[:60]!r}")
        elif (pattern_problem := artifact_pattern_problem(artifact)):
            problems.append(f"stages.{name}.artifact has invalid glob syntax: {pattern_problem}")
        parameters = row.get("parameters")
        if parameters is not None and not isinstance(parameters, list):
            problems.append(f"stages.{name}.parameters must be a list when present")
            continue
        for index, parameter in enumerate(parameters or []):
            if not isinstance(parameter, dict):
                problems.append(f"stages.{name}.parameters[{index}] must be an object")
                continue
            for key in ("name", "value", "evidence"):
                if not str(parameter.get(key, "")).strip():
                    # A value without evidence is the guess this field exists to expose: the
                    # disambiguation between several implementations is exactly a claim a
                    # repository can be asked about, and the answer shown to be checked.
                    problems.append(f"stages.{name}.parameters[{index}].{key} is required")
            parameter_value = str(parameter.get("value", ""))
            if " " in parameter_value.strip():
                # A sentence is a description of several values, not a value, and it is how
                # an unsure answer looks: the derivation wrote
                # "pi0 (directory policy/pi0), also pi0.5 (policy/pi05)" into a parameter a
                # command has to substitute, and the generated function then tried to parse
                # it rather than use it. Saying a value is unsettled is a legitimate answer
                # and a different field.
                problems.append(
                    f"stages.{name}.parameters[{index}].value is a description, not a value: "
                    f"{parameter_value[:60]!r}. Give the literal the command needs, or report the "
                    f"stage unavailable with a why")
    collected = stages.get("collect")
    evaluated = stages.get("evaluate")
    if (isinstance(collected, dict) and collected.get("available") is True and
            isinstance(evaluated, dict) and evaluated.get("available") is True):
        def stage_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
            entrypoint = Path(str(row.get("entrypoint") or "")).as_posix()
            invocation = re.sub(r"\s+", " ",
                                str(row.get("invocation") or "").strip())
            workdir = str(row.get("working_directory") or "{repo}").strip()
            workdir = "{repo}" if workdir in {"", ".", "./"} else Path(workdir).as_posix()
            environment = row.get("environment")
            environment = environment if isinstance(environment, dict) else {}
            return (entrypoint, invocation, workdir,
                    json.dumps(environment, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False))

        if stage_identity(collected) == stage_identity(evaluated):
            problems.append("stages.collect duplicates the evaluate command and context; "
                            "evaluation-time trajectory/video/HDF5 recording is an output "
                            "of evaluate, not an independent collection stage. Mark collect "
                            "unavailable unless source evidence supports a distinct producer")
    graph_doc = value.get("execution_graph")
    if graph_doc is not None:
        from .execution_graph import ExecutionGraph
        try:
            graph = ExecutionGraph(graph_doc)
        except ValueError as exc:
            problems.append(f"execution_graph is invalid: {exc}")
        else:
            for name in graph.nodes:
                if name not in stages or not isinstance(stages[name], dict) or not (
                        stages[name].get("available")):
                    problems.append(f"execution_graph node {name} has no available stage")
            score_target = graph_doc.get("score_target")
            if score_target is None and any(node.role == "evaluate"
                                             for node in graph.nodes.values()):
                problems.append("execution_graph.score_target is required for an "
                                "evaluate-role node")
            if score_target is not None and (
                    not isinstance(score_target, str) or score_target not in graph.nodes or
                    graph.nodes[score_target].role != "evaluate"):
                problems.append("execution_graph.score_target must name an evaluate node")
    return problems


def _objects(content: str) -> list[dict[str, Any]]:
    """Every JSON object in a reply, in the order they appear.

    From the first `{` to the *last* `}` is what this used to do, and it is right only when
    the reply holds exactly one object. A model that answers with two -- which one did, asking
    to look at a directory and then at the file inside it -- gets `{a}\\n\\n{b}`, which is not
    JSON, and the failure reads as a model that answered with nothing. Balanced scanning, so
    the reply's shape is whatever the model's shape is; the objects are separated by prose as
    often as by newlines.

    Nothing about this is specific to the reviser: every caller of `_object` in this module
    was one two-object reply away from the same silent nothing.
    """
    text = content.strip()
    for opener in ("```json", "```"):
        if text.startswith(opener):
            text = text[len(opener):]
    out: list[dict[str, Any]] = []
    depth, start, in_string, escaped = 0, -1, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    value = json.loads(text[start:index + 1])
                except json.JSONDecodeError:
                    start = -1
                    continue
                if isinstance(value, dict):
                    out.append(value)
                start = -1
    return out


def _object(content: str) -> dict[str, Any]:
    found = _objects(content)
    if not found:
        raise ValueError("no JSON object in the response")
    return found[0]


def _discard_malformed_optional_stages(value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Do not lose valid core stages solely because an optional stage is malformed.

    An optional stage is never made runnable by this recovery. It is discarded with an
    audit entry, and only when no graph node binds to it. The remaining full answer still
    passes ``problems_in`` before use.
    """
    stages = value.get("stages")
    if not isinstance(stages, dict):
        return value, []
    graph = value.get("execution_graph")
    bound: set[str] = set()
    if isinstance(graph, dict):
        bound.add(str(graph.get("score_target") or ""))
        for node in graph.get("nodes") or []:
            if isinstance(node, dict):
                bound.add(str(node.get("id") or ""))
                bound.update(str(name) for name in node.get("depends_on") or [])
                bound.update(str(name) for name in (node.get("bindings") or {}).values())
    discard: list[str] = []
    for name, row in stages.items():
        if name in STAGES or name in bound:
            continue
        if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", name) or
                not isinstance(row, dict) or row.get("available") is not True or
                any(not str(row.get(key) or "").strip()
                    for key in ("role", "why_required", "evidence", "entrypoint"))):
            discard.append(name)
    if not discard:
        return value, []
    return {**value, "stages": {name: row for name, row in stages.items()
                                 if name not in discard}}, discard


def triage(report: dict[str, Any], client: Any, *, limit: int = 12,
           context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Which files to read. A reader that has seen only a file listing is guessing."""
    coding_runtime = bool(getattr(client, "supports_main_agent", False))
    methods = skills_reference(
        query="repository discovery benchmark workflow native train evaluate collect "
              "entrypoint task assets source evidence",
        include_index=not coding_runtime,
        max_selected=2 if coding_runtime else 3)
    user = json.dumps({"stages_to_find": STAGES,
                       "survey": summarise_for_model(report),
                       "research_context": context or {}, "method_library": methods},
                      ensure_ascii=False)
    content, metadata = client.chat_with_metadata(
        sanitize_model_text(TRIAGE_SYSTEM),
        sanitize_model_text(user, local_roots=(Path(report["repo"]),)),
        max_tokens=1200, timeout=120, thinking="disabled")
    try:
        chosen = [str(p) for p in (_object(content).get("files_to_read") or [])][:limit]
    except (ValueError, AttributeError):
        chosen = []
    return {"files_to_read": chosen, "provider": metadata,
            "method_selection": methods["selection"]}


def derive(repo: Path, *, client: Any, report: dict[str, Any] | None = None,
           read_limit: int = 12, peek_lines: int = 70,
           attempts: int = 5,
           context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Survey, choose what to read, answer with entry points, and check what can be checked.

    The loop is the one the rest of the system uses, and for the same reason: a fault the
    filesystem can see is worth a retry, and the retry is worth a specific message. It
    cannot fix a wrong-but-real entry point, and does not claim to.
    """
    repo = Path(repo).expanduser().resolve()
    report = report or survey(repo)
    triage_result = triage(report, client, limit=read_limit, context=context)
    files = peek_many(triage_result["files_to_read"], repo=repo,
                      lines=peek_lines, limit=read_limit)
    for row in files:
        row["requested_as"] = row.get("path", "[withheld-path]")
    excerpts = [{k: row[k] for k in ("requested_as", "readable", "head") if k in row}
                for row in files]

    coding_runtime = bool(getattr(client, "supports_main_agent", False))
    methods = skills_reference(
        query="native stage train evaluate collect invocation entrypoint artifact source "
              "task contract command failure runtime validation",
        include_index=not coding_runtime,
        max_selected=2 if coding_runtime else 3)
    payload = json.dumps({"stages_to_find": STAGES,
                          "survey": summarise_for_model(report),
                          "file_contents": excerpts,
                          "research_context": context or {},
                          "method_library": methods}, ensure_ascii=False)
    log: list[dict[str, Any]] = []
    for repair in range(attempts):
        current = payload if repair == 0 else payload + json.dumps({
            "rejected": log[-1]["error"],
            "instruction": "Return the complete corrected object. Name only files that "
                           "exist in this repository. Keep the response compact; the previous "
                           "response was rejected, and its finish reason/usage are recorded.",
            "previous_response": log[-1].get("response_metadata", {})}, ensure_ascii=False)
        content, metadata = client.chat_with_metadata(
            sanitize_model_text(SYSTEM),
            sanitize_model_text(current, local_roots=(repo,)), max_tokens=8000,
                                                      timeout=240, thinking="disabled")
        try:
            value = _object(content)
            value, discarded_optional = _discard_malformed_optional_stages(value)
            faults = problems_in(value, repo)
            if faults:
                raise ValueError("; ".join(faults[:6]))
            log.append({"attempt": repair + 1, "status": "accepted",
                        "method_selection": methods["selection"],
                        "reasoning": value.get("reasoning"),
                        "discarded_malformed_optional_stages": discarded_optional,
                        "response_metadata": metadata,
                        "response_sha256": object_digest(content)})
            return {"stages": value["stages"], "reasoning": value.get("reasoning"),
                    "read": triage_result["files_to_read"], "attempts": log,
                    # Carried on the answer, not only in the prose around it. Every path is
                    # checked and no entry point is: a reader who takes this for settled has
                    # been told otherwise here, in the object they are reading.
                    "evidence": "static_only",
                    "method_selection": methods["selection"],
                    "triage_method_selection": triage_result.get("method_selection", []),
                    "unverified": "no stage has been executed; an entry point that exists "
                                  "and is not the right one passes every check here",
                    "provider": metadata}
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            log.append({"attempt": repair + 1, "status": "rejected",
                        "method_selection": methods["selection"],
                        "error": redact(f"{type(exc).__name__}: {exc}"),
                        "response_metadata": metadata,
                        "response_sha256": object_digest(content)})
    return {"stages": None, "reasoning": None, "read": triage_result["files_to_read"],
            "method_selection": methods["selection"],
            "triage_method_selection": triage_result.get("method_selection", []),
            "attempts": log, "evidence": "static_only", "provider": None}


def run(repo: Path, *, client: Any, output: Path,
        context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Derive and record, so a failure is legible rather than a stack trace."""
    result = derive(repo, client=client, context=context)
    output = Path(output)
    atomic_json(output / "execution.json",
                {"schema_version": 1, "created_at": now(), "repo": str(repo), **result})
    return result
