"""Verify (or falsify) the device-alignment model on a real multi-GPU node.

The mechanism is already known from the engine's own symbols -- ``DX_PickPhysicalDevice``
matches the configured index's **NVML UUID** against each Vulkan candidate's ``deviceUUID``,
while ``select_default_renderer`` reads the same integer in **CUDA** space.  So this probe
does not search for a mechanism; it runs a ladder in which every arm is a *prediction*:

  C  negative control   CVD=i, gpu_id=0        must reproduce the historical semaphore abort
  D  cheap diagnostic   CVD=i, gpu_id=i, auto  must print "Failed to query GPU name for device i"
  B  preferred          CVD=i, gpu_id=i        one real episode, only card i grows
  A  identity fallback  CVD=all, gpu_id=i      same, but torch sees every card
  concurrent            winning form on two    two *different* ready jobs at the same time

C and D are falsification points: if C does not abort, or if D resolves the GPU name
successfully, the model is missing something and the winning arm cannot be trusted.  A, by
contrast, is the fallback production form and is expected to pass as well -- it is only
rejected for being dangerous under concurrency (any ``cuda``/``cuda:0`` default lands on
card 0).  The 2026-09-13 ladder measured that caveat more precisely: A completes a real
episode on every card, and the torch/warp default device is aligned to the leased card, but
each arm still leaves ~525 MiB on card 0.  Passing arms are folded into a probe receipt
keyed by node+driver+device set; a multi-device run without a matching receipt refuses to
start (unknown capability is never assumed available).

    python -m autosim.research.device_probe --workspace <platform_root> --output <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .accounting import UtilizationSampler
from .common import atomic_json, digest, now
from .devices import (SimDeviceSelection, discover, probe_receipt_key, run_text, select,
                      write_probe_receipt)
from .registry import load_task
from .runtime import Runtime, startup_census, startup_receipt

DEFAULT_VULKAN_PROBE = Path("/data/AutoResearch/AutoSimSOTA/probe_vulkan_devices.py")
VULKAN_COUNT = re.compile(r"vulkan_physical_device_count=(\d+)")
VULKAN_UUID = re.compile(r"uuid=(GPU-[0-9a-fA-F-]+|\S+)")
OWN_GROWTH_MIB = 1024          # the device that ran the episode must grow at least this much
OTHER_CEILING_MIB = 512        # every other allocated device must stay under this

TORCH_SNIPPET = (
    "import json, torch\n"
    "rows = [{'i': i, 'name': torch.cuda.get_device_name(i),\n"
    "         'uuid': str(torch.cuda.get_device_properties(i).uuid)}\n"
    "        for i in range(torch.cuda.device_count())]\n"
    "print(json.dumps({'count': torch.cuda.device_count(), 'rows': rows}))\n")
RENDERER_SNIPPET = (
    "import sys; from embodichain.lab.sim.utility.render_utils import select_default_renderer;"
    " print('RESULT_MARKER', select_default_renderer(int(sys.argv[1])))")
NODE_ISOLATION_SNIPPET = """
import json, subprocess, torch
listed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                        capture_output=True, text=True).stdout.strip()
rows = [{"i": i, "uuid": str(torch.cuda.get_device_properties(i).uuid)}
        for i in range(torch.cuda.device_count())]
print("RESULT_MARKER", json.dumps({"nvidia_smi": listed,
                                   "torch_count": torch.cuda.device_count(),
                                   "torch_rows": rows}))
"""


def run_merged(command: Sequence[str], *, env: Mapping[str, str], timeout: int) -> dict:
    """Run a diagnostic with **stderr merged**, because the engine logs there.

    ``embodichain.utils.logger`` is a plain ``logging`` logger with no ``stream=``, so every
    ``log_info``/``log_warning`` goes to stderr.  ``devices.run_text`` keeps stdout only,
    which is right for ``nvidia-smi`` and wrong here: it made a diagnostic whose entire
    prediction is a warning line look like a falsification of the model.
    """
    try:
        completed = subprocess.run(list(command), capture_output=True, text=True,
                                   env=dict(env), timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"returncode": None, "text": f"__error__: {type(exc).__name__}: {exc}"}
    text = (completed.stdout or "") + (completed.stderr or "")
    return {"returncode": completed.returncode, "text": text.strip()}


def _growth(peak: dict, baseline: dict, uuid: str) -> dict:
    row = peak.get(uuid) or {}
    base = (baseline.get(uuid) or {}).get("peak_memory_mib") or 0
    top = row.get("peak_memory_mib")
    return {"peak_memory_mib": top, "baseline_memory_mib": base,
            "growth_mib": None if top is None else top - base,
            "mean_utilization_pct": row.get("mean_utilization_pct"),
            "samples": row.get("samples", 0),
            "processes": row.get("processes") or {}}


# The engine names this failure itself, so the ladder reads its words rather than guessing
# from a return code: ``OptixDevice.cpp`` does ``cuDeviceGet(&m_cudaDevice, m_ordinal)``.
# "Invalid device ID: 3. Available devices: 0-0" is the whole story of ``pinned_index`` on a
# node where the engine index is not a CUDA ordinal.
ABORT_MARKERS = {
    "invalid_device_ordinal": ("CUDA_ERROR_INVALID_DEVICE", "invalid device ordinal",
                               "Invalid device ID"),
}
TOUCHED_MIB = 512       # a card that moved this much before the abort really ran something


def classify_abort(*, output: Path, growths: Mapping[str, dict]) -> dict:
    """Name *why* a dead arm died, from its own log and from what moved before it did.

    Every ``rc=-6`` arm fails inside ``World.__init__``, so the crash stack cannot separate
    the two failure classes and a ladder reading only return codes will call a deterministic
    ordinal abort "the historical semaphore abort", or the reverse and write a real
    addressing failure off as flakiness. The distinction that matters is whether *a card was
    already running the engine* when the process died.
    """
    text = ""
    for candidate in (output / "process/stdout.log", output / "process/stderr.log"):
        if candidate.is_file():
            text += candidate.read_text(encoding="utf-8", errors="replace")
    touched = sorted((uuid for uuid, row in growths.items()
                      if (row.get("growth_mib") or 0) >= TOUCHED_MIB),
                     key=lambda uuid: -(growths[uuid].get("growth_mib") or 0))
    for kind, markers in ABORT_MARKERS.items():
        for marker in markers:
            if marker in text:
                line = next((row.strip() for row in text.splitlines() if marker in row), marker)
                return {"kind": kind, "evidence": line[:200],
                        "cards_touched_before_abort": touched}
    return {"kind": ("aborted_after_a_card_started" if touched
                     else "aborted_before_any_card_moved"),
            "evidence": (f"{len(touched)} card(s) had already started the engine when the "
                         f"process died before the first reset" if touched else
                         "no card had started the engine when the process died before the "
                         "first reset"),
            "cards_touched_before_abort": touched}


def environment_measurements(python: Path, base_env: dict, devices: list[dict], *,
                             vulkan_probe: Path, loader_select: str | None) -> dict:
    """P0: free measurements -- no engine, no episode, seconds not minutes.

    The question they answer is fixed: under ``CUDA_VISIBLE_DEVICES=i``, which physical
    cards do torch and Vulkan each see, and does the engine's own renderer query resolve the
    index we hand it?  ``VK_LOADER_DEVICE_SELECT`` is recorded as a *observation only*: it
    prioritises rather than filters, and four identical 5090s share one PCI id, so it cannot
    disambiguate them (it is not a candidate mechanism).
    """
    def measure(overrides: dict) -> dict:
        env = {**base_env, **overrides}
        row: dict = {}
        text = run_text([str(python), "-c", TORCH_SNIPPET], env=env, timeout=180)
        try:
            row["torch"] = json.loads(text)
        except (ValueError, TypeError):
            row["torch"] = {"error": text.strip()[:200]}
        # Merged streams here too: the prediction this measurement exists to test is a
        # ``log_warning``, and ``run_text`` would reduce the whole row to the bare marker --
        # the first probe recorded exactly that, "RESULT_MARKER hybrid" identically in all
        # six environments, which is what a blind measurement looks like.
        renderer = run_merged([str(python), "-c", RENDERER_SNIPPET,
                               str(devices[-1]["index"])], env=env, timeout=180)
        row["renderer_auto"] = renderer["text"][-400:]
        row["renderer_auto_returncode"] = renderer["returncode"]
        if vulkan_probe.is_file():
            vk = run_text([str(python), str(vulkan_probe)], env=env, timeout=180)
            counts = VULKAN_COUNT.findall(vk)
            row["vulkan"] = {"physical_devices": int(counts[0]) if counts else None,
                             "uuids": VULKAN_UUID.findall(vk)[:8],
                             "error": vk.strip()[:200] if vk.strip().startswith("__error__")
                             else None}
        return row

    all_indices = ",".join(str(g["index"]) for g in devices)
    table = {"shared": {"outer_default": measure({}), "cvd_all": measure(
        {"CUDA_VISIBLE_DEVICES": all_indices})}, "by_index": {}}
    if loader_select:
        table["shared"]["loader_select"] = measure(
            {"CUDA_VISIBLE_DEVICES": all_indices, "VK_LOADER_DEVICE_SELECT": loader_select})
    for device in devices:
        index = str(device["index"])
        table["by_index"][index] = measure({"CUDA_VISIBLE_DEVICES": index})
    return table


def node_isolation_measurement(python: Path, base_env: Mapping[str, str], devices: Sequence[dict],
                               *, timeout: int = 240, dri: Path = Path("/dev/dri")) -> dict:
    """Can this container hide the other cards' device nodes -- and does the survivor renumber?

    Every ``identity`` arm on a card other than 0 leaves ~525 MiB on card 0 (measured
    2026-09-13), and it is not torch's default device and not warp's: both are aligned to the
    leased card and the receipt shows it.  Whatever resolves "device 0" for itself, the
    mechanism that neutralises it *without having to name it first* is to make card 0 **be**
    the leased card: ``unshare --mount`` with the other ``/dev/nvidiaN`` nodes bound to
    ``/dev/null``, which is what ``devices.select(..., "node_isolated")`` already models.
    Whether that is available is a property of this container's privileges, and it costs
    seconds, so it is measured rather than assumed.

    Two things fail independently, so both are reported: the namespace (``unshare`` needs
    ``CAP_SYS_ADMIN``; ``--user`` is tried as the fallback spelling) and the renumbering (a
    node that disappears may keep its own minor, in which case the survivor is still index 3
    and hiding nodes is not by itself the answer).  ``/dev/dri`` is listed because Vulkan
    enumerates render nodes rather than ``nvidia*`` nodes: if this proves out, hiding the
    card from the engine's Vulkan side needs those too, which is a follow-up this
    measurement does not claim.

    Feasibility only, never verification: nothing here writes a receipt and no arm is called
    passed on the strength of it.
    """
    target = devices[-1]
    hidden = [f"/dev/nvidia{device['index']}" for device in devices
              if device["index"] != target["index"]]
    script = ("for node in " + " ".join(hidden) + '; do mount --bind /dev/null "$node" || '
              "echo BIND_FAILED $node; done; exec " + str(python) + " -c "
              + shlex.quote(NODE_ISOLATION_SNIPPET))
    measurement: dict = {"target_index": target["index"], "target_uuid": target["uuid"],
                         "hidden_nodes": hidden, "runs": {}}
    variants = (("namespace_only", [], {}),
                ("with_cuda_visible_0", [], {"CUDA_VISIBLE_DEVICES": "0"}),
                ("user_namespace", ["--user", "--map-root-user"], {}))
    for label, prefix, overrides in variants:
        captured = run_merged(["unshare", *prefix, "--mount", "--propagation", "private",
                               "sh", "-c", script],
                              env={**base_env, **overrides}, timeout=timeout)
        text = captured["text"]
        found = re.search(r"RESULT_MARKER\s+(\{.*\})", text)
        row: dict = {"returncode": captured["returncode"]}
        if found:
            try:
                row.update(json.loads(found.group(1)))
            except ValueError:
                row["parse_error"] = found.group(1)[:160]
        row["bind_failures"] = re.findall(r"BIND_FAILED (\S+)", text)
        visible = [line for line in str(row.get("nvidia_smi") or "").splitlines() if line.strip()]
        row["visible_devices"] = len(visible)
        row["survivor_index"] = visible[0].split(",")[0].strip() if len(visible) == 1 else None
        row["feasible"] = bool(row["returncode"] == 0 and not row["bind_failures"]
                               and row["visible_devices"] == 1 and row["survivor_index"] == "0"
                               and row.get("torch_count") == 1)
        if not row["feasible"]:
            row["output"] = text[-300:]
        measurement["runs"][label] = row
    measurement["feasible"] = all(row.get("feasible") for row in measurement["runs"].values())
    try:
        measurement["dri_nodes"] = sorted(node.name for node in dri.iterdir())
    except OSError as exc:
        measurement["dri_nodes"] = f"unreadable: {exc.strerror or exc}"
    return measurement


def judge(arm: dict, *, own_uuid: str, allocation: list[str], baseline: dict, peak: dict,
          idle_uuids: list[str] | None = None) -> dict:
    """The leased device must have run it -- and no device that should be idle may move.

    ``idle_uuids`` defaults to "every allocated device except this one", which is the right
    expectation for a sequential arm.  A concurrent arm must instead name the devices its
    sibling legitimately occupies, or it would fail on its own sibling's memory.
    """
    growths = {uuid: _growth(peak, baseline, uuid) for uuid in allocation}
    own = growths.get(own_uuid, {})
    idle = [uuid for uuid in (allocation if idle_uuids is None else idle_uuids)
            if uuid != own_uuid]
    checks = {
        "completed": arm.get("status") == "completed",
        "real_simulation": arm.get("execution_mode") == "real_simulation",
        "one_episode": arm.get("episode_count") == 1,
        "no_worker_failure": not arm.get("worker_failure"),
        "own_device_grew": (own.get("growth_mib") or 0) >= OWN_GROWTH_MIB,
        "others_stayed_idle": all((growths.get(uuid, {}).get("growth_mib") or 0) < OTHER_CEILING_MIB
                                  for uuid in idle),
    }
    return {"checks": checks, "passed": all(checks.values()), "growth_mib": growths,
            "own_device_uuid": own_uuid, "idle_devices": idle,
            "foreign_holders": foreign_holders(growths, idle)}


def foreign_holders(growths: Mapping[str, dict], idle: Sequence[str]) -> list[dict]:
    """Name what moved on a card that was supposed to stay idle.

    Memory on a device nobody leased is either a library defaulting to device 0 -- bounded,
    idle, and the same process that ran the arm -- or somebody else's work.  ``others stayed
    idle`` cannot tell them apart, and a receipt that only reports the megabytes leaves the
    reader with a number and no decision.  So the growth, the *command lines* that held it,
    and whether the holder was the arm's own process are all recorded.
    """
    rows = []
    for uuid in idle:
        row = growths.get(uuid) or {}
        if (row.get("growth_mib") or 0) < OTHER_CEILING_MIB:
            continue
        rows.append({"device_uuid": uuid, "growth_mib": row.get("growth_mib"),
                     "mean_utilization_pct": row.get("mean_utilization_pct"),
                     "holders": [{"pid": pid, **detail}
                                 for pid, detail in (row.get("processes") or {}).items()]})
    return rows


def reproduced_native_abort(arm: dict) -> bool:
    """Did the negative control fail *the way the model says*, not merely fail?

    The first probe answered this with ``status == "failed"`` and counted an argparse error
    (rc=2, 1.8 s, no ``startup.json`` at all) as "reproduced the historical abort" -- a
    falsification that was really a typo in our own command line.  The claim under test is
    specific: a *native* signal before the first simulated reset.  Only the arm's own process
    receipt can witness that, so the receipt is what this reads, and a missing one (the
    launcher died before ``Popen``) is not evidence of an abort either.
    """
    receipt = arm.get("startup_receipt") or {}
    return (arm.get("status") == "failed"
            and receipt.get("status") == "failed"
            and receipt.get("returncode") in {-6, -11}
            and receipt.get("startup_phase") in {"environment_constructing", "environment_ready"}
            and not receipt.get("initializations_recorded"))


def run_arm(*, name: str, runtime: Runtime, spec, checkpoint: Path, output: Path,
            master_seed: int, smoke_timeout: int, selection: SimDeviceSelection,
            sampler: UtilizationSampler | None) -> dict:
    """One real smoke episode under one device selection, with memory watched per card.

    ``sampler`` is passed in for the concurrent arm so a single watch covers both jobs;
    otherwise this arm gets its own, and its first reading is the arm's baseline.
    """
    local = sampler or UtilizationSampler(interval_seconds=2.0)
    if sampler is None:
        local._sample_once()
    baseline = local.summary()
    bound = runtime.for_job(selection, job=name, output=output)
    arm = {"name": name, "selection": selection.as_dict(), "started_at": now()}
    started = time.time()
    try:
        if sampler is None:
            with local:
                data = bound.evaluate(spec, checkpoint, output, episodes=1,
                                      master_seed=master_seed, purpose="smoke",
                                      startup_attempts=1, smoke_timeout=smoke_timeout)
        else:
            data = bound.evaluate(spec, checkpoint, output, episodes=1, master_seed=master_seed,
                                  purpose="smoke", startup_attempts=1, smoke_timeout=smoke_timeout)
        arm.update(status="completed", execution_mode=data.get("execution_mode"),
                   episode_count=len(data.get("episodes") or []), summary=data.get("summary"))
    except Exception as exc:                      # noqa: BLE001 -- the receipt is the result
        arm.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:400],
                   startup_receipt=startup_receipt(output))
    arm["elapsed_seconds"] = round(time.time() - started, 1)
    failure = output / "worker_failure.json"
    arm["worker_failure"] = (json.loads(failure.read_text(encoding="utf-8"))
                             if failure.is_file() else None)
    arm["memory"] = {uuid: _growth(local.summary(), baseline, uuid) for uuid in local.summary()}
    # Per-card memory of this arm's own process at each phase it reached: the number says a
    # card leaked, the timeline says which phase did it.
    arm["startup_census"] = startup_census(output)
    # Which process ran this arm, so a foreign holder on another card can be compared against
    # it.  A context the arm's *own* process left on card 0 and a stranger's process is the
    # difference between "this library defaults to device 0" and "another job is sharing my
    # card", and the two need opposite responses.
    arm["own_pid"] = (startup_receipt(output) or {}).get("pid")
    if arm.get("status") != "completed":
        arm["abort"] = classify_abort(output=output, growths=arm["memory"])
    arm["_baseline"], arm["_peak"] = baseline, local.summary()
    return arm


def negative_control_holds(arm: dict | None) -> bool | None:
    """The negative control must die *the way the model says*, not merely die.

    ``reproduced_native_abort`` accepts any pre-reset native abort, and the second probe on
    this pool showed why that is too generous: three ``pinned_index`` arms died with the same
    return code, the same startup phase and the same crash stack as the control, but at
    ``cuDeviceGet`` -- a different event entirely, with the opposite meaning.  What separates
    them is whether a card had started the engine: the split-spaces abort happens *after* the
    engine comes up on two different cards, the ordinal abort never reaches a card at all.
    """
    if arm is None:
        return None
    if not reproduced_native_abort(arm):
        return False
    return (arm.get("abort") or {}).get("kind") != "aborted_before_any_card_moved"


def probe_verdict(*, arms: Sequence[dict], allocation: Sequence[dict],
                  requested_mode: str) -> dict:
    """Which addressing mode this node verified, on which devices, and why not the other.

    Two claims are being decided, and they are not symmetric.  The **preferred** mode
    (``pinned_index``: one visible card, engine index = physical index) is what a plan would
    use by default; the **fallback** (``identity``: every card visible, engine index = CUDA
    ordinal = physical index) is what remains when the preferred mode turns out to be a
    special case.  Whichever mode verified *every allocated device* wins: a capability that
    holds on one card and aborts on the others is not a capability, and demanding the
    preferred mode specifically would report a usable node as unusable.  A mode that failed
    for a nameable reason is recorded as a limitation; one that failed for no nameable reason
    is a discrepancy, because then the model is missing something.
    """
    by_name = {arm.get("name"): arm for arm in arms}
    negative, diagnostic = by_name.get("C"), by_name.get("D")
    preferred = [a for a in arms if str(a.get("name", "")).startswith("B_")]
    fallback = [a for a in arms
                if a.get("name") == "A" or str(a.get("name", "")).startswith("A_")]
    concurrent = [a for a in arms if str(a.get("name", "")).startswith("concurrent")]

    def per_device(rows: Sequence[dict]) -> dict:
        return {str(a["device_index"]): {"passed": a["judged"]["passed"], "name": a["name"],
                                         "reason": [k for k, v in a["judged"]["checks"].items()
                                                    if not v],
                                         **({"abort": a["abort"]} if a.get("abort") else {}),
                                         **({"own_pid": a["own_pid"]} if a.get("own_pid") else {}),
                                         **({"foreign_holders": a["judged"]["foreign_holders"]}
                                            if a.get("judged", {}).get("foreign_holders") else {})}
                for a in rows}

    preferred_ok = bool(preferred) and all(a["judged"]["passed"] for a in preferred)
    fallback_ok = bool(fallback) and all(a["judged"]["passed"] for a in fallback)
    if preferred_ok:
        mode, winner, winning = requested_mode, "preferred", preferred
    elif fallback_ok:
        mode, winner, winning = "identity", "fallback", fallback
    else:
        mode, winner, winning = None, None, []
    verified = sorted({a["judged"]["own_device_uuid"] for a in winning if a["judged"]["passed"]})
    verified_all = bool(mode) and set(verified) == {g["uuid"] for g in allocation}
    concurrent_ok = (len(concurrent) == 2
                     and all(a.get("judged", {}).get("passed") for a in concurrent))
    negative_ok = negative_control_holds(negative)

    limitations, discrepancies = [], []
    failed_preferred = [a for a in preferred if not a["judged"]["passed"]]
    if failed_preferred:
        kinds = {str((a.get("abort") or {}).get("kind") or "unclassified")
                 for a in failed_preferred}
        if kinds == {"invalid_device_ordinal"}:
            limitations.append(
                f"{requested_mode} is only valid on physical index 0 on this node: the engine "
                f"hands the configured index to CUDA/OptiX as an *ordinal*, so with a single "
                f"visible card every index above 0 dies at cuDeviceGet (OptixDevice.cpp) with "
                f"CUDA_ERROR_INVALID_DEVICE before any card moves; identity has no such limit "
                f"because nothing is renumbered")
        else:
            limitations.append(f"{requested_mode} failed with cause(s) "
                               f"{', '.join(sorted(kinds))}")
            discrepancies.append("preferred_mode_failed_without_a_named_cause")
    if negative is not None and negative_ok is not True:
        discrepancies.append("negative_control_did_not_reproduce_the_abort"
                             if not reproduced_native_abort(negative)
                             else "negative_control_aborted_before_reaching_a_card")
    if diagnostic is not None and diagnostic["prediction_met"] is not True:
        discrepancies.append("diagnostic_prediction_not_met")
    if mode is None:
        discrepancies.append("no_addressing_mode_verified_every_device")
    elif not verified_all:
        discrepancies.append("the_winning_mode_did_not_verify_every_allocated_device")
    if not concurrent_ok:
        discrepancies.append("two_devices_never_ran_an_episode_at_the_same_time")
    # Memory on a card nobody leased is reported as a named fact rather than a failed check
    # with no witness.  It is not a capability verdict by itself: the same names repeated
    # across every arm on the same card is a property of the library, not of one arm.
    residual = [{"arm": a["name"], "device_index": a.get("device_index"),
                 "own_pid": a.get("own_pid"), **holder}
                for a in arms for holder in (a.get("judged", {}).get("foreign_holders") or [])]

    return {
        "winner": winner, "selection_mode": mode,
        "negative_control_reproduced_abort": negative_ok,
        "negative_control_evidence": (negative or {}).get("abort"),
        "diagnostic_prediction_met": None if diagnostic is None else diagnostic["prediction_met"],
        "devices_verified_under_preferred_mode": per_device(preferred),
        "devices_verified_under_winning_mode": per_device(winning),
        "fallback_mode_verified": fallback_ok if fallback else None,
        "concurrent_two_devices_verified": concurrent_ok,
        "cross_card_residual": residual,
        "mode_limitations": limitations,
        "model_discrepancies": discrepancies,
        "verified_devices": verified,
        "verified_every_allocated_device": verified_all,
        "passed": bool(mode and verified_all and concurrent_ok
                       and (diagnostic is None or diagnostic["prediction_met"] is not False)
                       and (negative is None or negative_ok is True)),
    }


def main() -> int:                                # noqa: C901 -- a linear ladder reads better flat
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path,
                        default=Path("/data/AutoResearch/AutoSimSOTA/AutoSimSOTA"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="click_bell")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gpus", default="auto", help="container indices, e.g. 0,1,2,3")
    parser.add_argument("--mode", default="pinned_index", choices=["pinned_index", "identity"])
    parser.add_argument("--smoke-timeout", type=int, default=900,
                        help="per arm; DexSim needs ~360 s before the first reset")
    parser.add_argument("--master-seed", type=int, default=2000000001)
    parser.add_argument("--arms", default="C,D,B,A,concurrent")
    parser.add_argument("--worker-spec", default=os.environ.get("SCO_WORKER_SPEC"))
    parser.add_argument("--image", default=os.environ.get("AUTOSIM_IMAGE"))
    parser.add_argument("--vulkan-probe", type=Path, default=DEFAULT_VULKAN_PROBE)
    parser.add_argument("--skip-capability-write", action="store_true",
                        help="measure only; do not publish a usable-device receipt")
    args = parser.parse_args()

    platform_root = args.workspace.resolve()
    bench = platform_root / "RoboSynChallenge"
    checkpoint = args.checkpoint or bench / "checkpoints" / f"ACT_sim_{args.task}"
    args.output.mkdir(parents=True, exist_ok=True)
    report = discover(disk_path=platform_root)
    allocation = [g for g in report["gpus"] if g["index"] in report["allowed"]]
    if args.gpus not in ("auto", None, ""):
        wanted = {int(part) for part in str(args.gpus).split(",")}
        outside = wanted - {g["index"] for g in allocation}
        if outside:
            raise SystemExit(f"--gpus {args.gpus} outside the allowed range "
                             f"{[g['index'] for g in allocation]}")
        allocation = [g for g in allocation if g["index"] in wanted]
    if not allocation:
        raise SystemExit(f"no usable device: nvidia-smi reported {len(report['gpus'])} "
                         f"device(s); outer filter={report['outer'].get('reason')}")
    receipt: dict = {
        "schema_version": 1, "kind": "autosim_device_probe", "probe_started_at": now(),
        "host": report["host"], "outer": report["outer"],
        "allocation": [{k: g.get(k) for k in ("index", "uuid", "model", "driver_version",
                                              "compute_capability", "memory_total_mib")}
                       for g in allocation],
        "visible_index_to_uuid": report["visible_index_to_uuid"],
        "cuda_order_mismatch": report["cuda_order_mismatch"],
        "cuda_index_to_uuid": report["cuda_index_to_uuid"],
        "errors": report["errors"], "cpu": report["cpu"], "memory": report["memory"],
        "disk": report["disk"],
        "resource_list": [f"index {g['index']} uuid {g['uuid']} {g['model']}" for g in allocation],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": (digest(checkpoint / "model.safetensors")
                              if (checkpoint / "model.safetensors").is_file() else None),
        "task": args.task, "requested_mode": args.mode, "arms": [],
    }
    base_runtime = Runtime(platform_root, args.output / "runtime", repo_path=bench,
                           eval_repo_path=bench)
    # Measurements of the *environment* must not inherit this adapter's own device override,
    # or "outer default" would silently mean "whatever Runtime.gpu happens to be".
    base_env = {k: v for k, v in base_runtime.environment(bench).items()
                if k != "CUDA_VISIBLE_DEVICES"}
    print(f"[P0] host={report['host']} allowed={report['allowed']} "
          f"reason={report['outer'].get('reason')}", flush=True)

    def pci_id(index: int) -> str | None:
        text = run_text(["nvidia-smi", "--query-gpu=index,pci.device_id",
                         "--format=csv,noheader"], timeout=60)
        for line in text.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 2 and parts[0] == str(index):
                return parts[1].replace("0x", "").lower()
        return None

    loader = pci_id(allocation[-1]["index"])
    receipt["environments"] = environment_measurements(
        base_runtime.python, base_env, allocation, vulkan_probe=args.vulkan_probe,
        loader_select=f"0x10de:0x{loader}" if loader else None)
    print(json.dumps(receipt["environments"], indent=1)[:5000], flush=True)
    # Free, and it decides which repair is even available: if this container can hide the
    # other cards' device nodes, a mode exists that makes "device 0" mean the leased card for
    # every subsystem at once -- including whichever one the 525 MiB on card 0 belongs to.
    receipt["node_isolation"] = node_isolation_measurement(base_runtime.python, base_env,
                                                           allocation)
    print(f"[P0-node] feasible={receipt['node_isolation']['feasible']} "
          f"{json.dumps(receipt['node_isolation']['runs'])[:900]}", flush=True)

    spec = load_task(bench, args.task)
    primary = allocation[-1]
    wanted = [name.strip() for name in args.arms.split(",") if name.strip()]

    def selection_for(device: dict, style: str) -> SimDeviceSelection:
        if style == "negative_control":
            return replace(select(device, "pinned_index"), mode="negative_control",
                           vulkan_gpu_id=0, torch_index=0)
        if style == "identity":
            return replace(select(device, "identity"),
                           cuda_visible=",".join(str(g["index"]) for g in allocation))
        return select(device, args.mode)

    def record(result: dict, device: dict, baseline: dict, peak: dict,
               idle_uuids: list[str] | None = None) -> dict:
        result["judged"] = judge(result, own_uuid=device["uuid"],
                                 allocation=[g["uuid"] for g in allocation], baseline=baseline,
                                 peak=peak, idle_uuids=idle_uuids)
        result["device_index"] = device["index"]
        receipt["arms"].append(result)
        print(f"[{result['name']}] index={device['index']} passed={result['judged']['passed']} "
              f"{json.dumps(result['judged']['checks'])}", flush=True)
        return result

    # One arm per allocated device under the preferred mode: a receipt may only claim a
    # device is usable if that device itself ran an episode ("unknown capability is never
    # assumed available").
    ladder = [("C", primary, "negative_control"), ("D", primary, "diagnostic")]
    ladder += [(f"B_{g['index']}", g, args.mode) for g in allocation] if "B" in wanted else []
    for name, device, style in ladder:
        if style == "diagnostic":
            # Not a smoke run: renderer="auto" is refused by the evaluator on purpose, and
            # the prediction resolves in seconds. Run the engine's own selection function.
            captured = run_merged([str(base_runtime.python), "-c", RENDERER_SNIPPET,
                                   str(device["index"])],
                                  env={**base_env, "CUDA_VISIBLE_DEVICES": str(device["index"])},
                                  timeout=180)
            text = captured["text"]
            predicted = f"Failed to query GPU name for device {device['index']}"
            resolved = re.search(r"RESULT_MARKER\s+(\S+)", text)
            receipt["arms"].append({
                "name": name, "style": style, "device_index": device["index"],
                "cuda_visible": str(device["index"]), "gpu_id": device["index"],
                "returncode": captured["returncode"], "resolved_renderer": resolved.group(1) if resolved else None,
                "output": text[-1200:], "predicted": predicted,
                "prediction_met": predicted in text, "status": "measured"})
            print(f"[D] {text[-400:]}", flush=True)
            continue
        result = run_arm(name=name, runtime=base_runtime, spec=spec, checkpoint=checkpoint,
                         output=args.output / f"arm_{name}", master_seed=args.master_seed,
                         smoke_timeout=args.smoke_timeout,
                         selection=selection_for(device, style), sampler=None)
        record(result, device, result.pop("_baseline"), result.pop("_peak"))

    preferred_arms = [a for a in receipt["arms"] if str(a.get("name", "")).startswith("B_")]
    preferred_ok = bool(preferred_arms) and all(a["judged"]["passed"] for a in preferred_arms)
    # The fallback is measured on one device while the preferred mode already covers every
    # device (its failure would still end the probe with a decision), and on *every* allocated
    # device when it does not: "identity works here" is a claim about each card, and a mode
    # verified on one card is exactly the over-claim a receipt exists to prevent.
    if "A" in wanted:
        targets = [primary] if (preferred_ok or not preferred_arms) else list(allocation)
        for device in targets:
            name = f"A_{device['index']}"
            result = run_arm(name=name, runtime=base_runtime, spec=spec, checkpoint=checkpoint,
                             output=args.output / f"arm_{name}", master_seed=args.master_seed,
                             smoke_timeout=args.smoke_timeout,
                             selection=selection_for(device, "identity"), sampler=None)
            record(result, device, result.pop("_baseline"), result.pop("_peak"))

    # Two devices at the same time, in the mode the plan would actually use -- running the
    # concurrency arm under a mode already known to abort would measure the abort twice and
    # call it a concurrency result (the second probe did exactly that).
    concurrency_mode = args.mode if preferred_ok else "identity"
    if "concurrent" in wanted:
        if len(allocation) < 2:
            receipt["arms"].append({"name": "concurrent", "status": "skipped",
                                    "reason": "fewer than two allocated devices"})
        else:
            first, second = allocation[-1], allocation[-2]
            sampler = UtilizationSampler(interval_seconds=2.0)
            sampler._sample_once()
            baseline = sampler.summary()
            results: dict[str, dict] = {}
            lock = threading.Lock()

            def one(label: str, device: dict) -> None:
                result = run_arm(name=label, runtime=base_runtime, spec=spec,
                                 checkpoint=checkpoint, output=args.output / f"arm_{label}",
                                 master_seed=args.master_seed, smoke_timeout=args.smoke_timeout,
                                 selection=selection_for(device, concurrency_mode), sampler=sampler)
                with lock:
                    results[label] = result

            threads = [threading.Thread(target=one, args=(label, device))
                       for label, device in (("concurrent_a", first), ("concurrent_b", second))]
            # The sampler only samples inside ``with``: it is entered here, around the whole
            # overlap window, because ``run_arm`` deliberately skips its own sampling when a
            # shared sampler is passed in.  Without this the "peak" after the join is the
            # baseline again and both concurrent arms report zero growth on every card --
            # which reads as "two devices never ran an episode at the same time" when the
            # episodes did run and only the measurement was missing.  That is exactly what
            # the 2026-09-13 ladder recorded, so this enter/exit pair is load-bearing.
            with sampler:
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            peak = sampler.summary()
            uninvolved = [g["uuid"] for g in allocation if g not in (first, second)]
            for label, device in (("concurrent_a", first), ("concurrent_b", second)):
                result = results[label]
                result.pop("_baseline"), result.pop("_peak")
                result["overlap"] = {"devices": [first["uuid"], second["uuid"]]}
                record(result, device, baseline, peak, idle_uuids=uninvolved)
            receipt["concurrent"] = {"devices": [first["uuid"], second["uuid"]],
                                     "mode": concurrency_mode,
                                     "longest_arm_seconds": max(
                                         r["elapsed_seconds"] for r in results.values())}

    # The two diagnostic arms answer different questions, so they gate differently.  D tests a
    # mechanism this code *depends on* -- that the engine's index and torch's index are
    # different spaces under ``CUDA_VISIBLE_DEVICES=i`` -- so a missed prediction falsifies the
    # addressing the plan is built on.  C tests a *bug*: if the mismatch no longer aborts, the
    # capability measured by the B/A arms is unaffected, but the causal story is unverified and
    # saying nothing about it would let a receipt imply a mechanism nobody observed.
    verdict = probe_verdict(arms=receipt["arms"], allocation=allocation,
                            requested_mode=args.mode)
    mode, verified = verdict["selection_mode"], verdict["verified_devices"]
    receipt["verdict"] = verdict
    receipt["passed"] = verdict["passed"]
    # Only devices that actually ran an episode *under the winning mode* are ever called
    # usable, and the mode is part of the receipt: a capability measured under one addressing
    # scheme says nothing about another.
    receipt["mode"] = mode
    receipt["verified_devices"] = sorted(verified)
    receipt["devices"] = {g["uuid"]: ("verified_sim" if g["uuid"] in verified else "unknown")
                          for g in allocation}
    receipt["probe_finished_at"] = now()
    key = probe_receipt_key(host=report["host"], driver=allocation[0].get("driver_version"),
                            image=args.image, worker_spec=args.worker_spec,
                            uuids=[device["uuid"] for device in allocation])
    receipt["probe_receipt_key"] = key
    atomic_json(args.output / "device_probe.json", receipt)
    if receipt["passed"] and not args.skip_capability_write:
        write_probe_receipt(platform_root, key, receipt)
    print("\n=== device probe verdict ===")
    print(json.dumps(verdict, indent=1))
    print(f"passed={receipt['passed']} mode={mode} receipt_key={key}")
    if verdict["model_discrepancies"]:
        print(f"MODEL DISCREPANCIES (reported, not explained away): "
              f"{verdict['model_discrepancies']}")
    for row in verdict["cross_card_residual"]:
        names = "; ".join(f"pid {h.get('pid')} ({h.get('cmd') or h.get('note') or '?'}, "
                          f"parent {h.get('parent')})" for h in row.get("holders") or [])
        print(f"CROSS-CARD RESIDUAL: arm {row['arm']} (own pid {row.get('own_pid')}) left "
              f"{row.get('growth_mib')} MiB on device {row['device_index']} "
              f"[{row['device_uuid']}] at {row.get('mean_utilization_pct')}% utilization, "
              f"held by {names or 'no process nvidia-smi could name'}")
    for note in verdict.get("mode_limitations") or []:
        print(f"MODE LIMITATION: {note}")
    print(f"receipt: {args.output / 'device_probe.json'}")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
