"""What the run's best number is the best *of*, which is the part a report leaves out.

A research loop measures candidates, keeps the highest, and writes the highest down. Every
individual measurement in that is recorded correctly and the number that comes out of the run
is not a measurement of anything: it is the maximum of however many draws were taken, and the
maximum of draws sits above the middle of them by an amount that grows with how many were
taken and how noisy they were. Thirty candidates on forty episodes will produce an incumbent
better than the baseline roughly always, with no improvement present at all.

Nothing here can undo that -- the selection has already happened by the time a number is
reported, and no statistical adjustment recovers the sample the loop looked at. What this
module does is make the count visible and attach it to the number, so that "0.58 against a
0.41 baseline" arrives as "0.58, the best of nine arms measured at this protocol, against a
0.41 baseline". A reader given that can decide what it is worth. A reader given the bare pair
cannot, and the run has then produced an artifact whose honesty depends on nobody asking.

**Two counts, not one.** *Attempted* is how many arms the loop started; *scored* is how many
produced a number. The distance between them is the run's failure rate, and a loop that
reports the best of four while silently starting nineteen has reported something about the
four that is false of the nineteen.

**The comparison is against the baseline's own reading, at the same protocol.** That is the
only comparison the record is entitled to make, because it is the only other arm that was
measured on the same episodes. An arm measured under a different protocol hash was measured
against a different benchmark and is not pooled with these.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_json, now, object_digest, read_json
from .significance import compare_proportions, proportion

#: Where a run keeps this. Beside `comparison_protocol.json`, and mutable where that is
#: frozen: the protocol says what a measurement *is*, and this says how many were taken.
RECORD = "selection_record.json"

#: Settings are a request, not an episode receipt. Preserve them for protocol comparison,
#: but never infer a statistical denominator from them.
COUNT_KEYS = ("episodes", "n_episodes", "num_episodes", "eval_episodes", "n_eval_episodes",
              "num_eval_episodes", "trials", "n_trials", "rollouts", "n_rollouts",
              "eval_trials", "test_episodes", "n_test_episodes")


def _episodes(settings: dict[str, Any]) -> int | None:
    for key in COUNT_KEYS:
        value = settings.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _load(root: Path) -> dict[str, Any]:
    path = Path(root) / RECORD
    try:
        record = read_json(path)
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _arms_from_files(root: Path) -> list[dict[str, Any]]:
    """Arms recovered from the measurement files, for runs the record does not cover.

    Every arm `_measure` takes writes exactly one `measurements/<label>.json` before it
    returns, so those files are the ground truth that an arm happened. The record is the
    better counter -- it is append-only, so it survives a resumed run re-measuring a label --
    but a run that predates the record, or whose record was lost, still took its arms and
    their files are still here.

    Reading only the record made an eight-arm run report `attempted: 0`, and understating the
    search is the flattering direction: it is the count that decides whether the best number
    is a measurement or a maximum of draws.
    """
    directory = Path(root) / "measurements"
    if not directory.is_dir():
        return []
    arms: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            row = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if row.get("confirmation") is True:
            continue
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        native = row.get("metric_reading") if isinstance(row.get("metric_reading"), dict) else {}
        value = row.get("metric_value")
        arms.append({"label": str(row.get("label") or path.stem), "at": "",
                     "started": True, "scored": isinstance(value, (int, float)),
                     "metric_value": value if isinstance(value, (int, float)) else None,
                     "metric_utility": (row.get("metric_utility")
                                        if isinstance(row.get("metric_utility"), (int, float))
                                        else None),
                     "baseline": str(row.get("label") or path.stem) == "baseline",
                     "episodes": native.get("episodes_completed"),
                     "successes": native.get("successes"),
                     "metric_kind": "binary" if (row.get("metric") or {}).get("name") ==
                     "success_rate" else "unknown",
                     # Kept, because whether the arms were measured the same way is a question
                     # about their settings and not about the file's contents otherwise.
                     "settings": settings,
                     "from": "measurement file"})
    return arms


def arms_of(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Every arm the run took, and which sources the answer came from.

    Recorded arms win on a label they mention, because the record may hold several entries
    for one label where the file holds only the last. Labels the record does not mention are
    recovered from their files, so losing the record costs a run its exact counts and not the
    existence of the arms.
    """
    root = Path(root)
    recorded = [arm for arm in (_load(root).get("arms") or []) if isinstance(arm, dict)]
    sources = ["selection record"] if recorded else []
    known = {str(arm.get("label") or "") for arm in recorded}
    recovered = [arm for arm in _arms_from_files(root) if str(arm.get("label") or "") not in known]
    if recovered:
        sources.append("measurement files")
    for arm in recovered:
        arm.pop("from", None)
    return recorded + recovered, sources


def note(root: Path, *, protocol_sha256: str, label: str, started: bool,
         metric_value: float | None, metric_utility: float | None,
         baseline: bool = False, settings: dict[str, Any] | None = None,
         metric_kind: str = "binary", actual_episodes: int | None = None,
         successes: int | None = None) -> dict[str, Any]:
    """Add one arm to the run's selection record.

    Called for every measurement the loop takes, successful or not. An arm that failed is
    still a draw the loop took: it cost the same wall clock, its failure is what the loop
    learned from, and leaving it out of the count is how a search of twenty comes to be
    reported as a search of four.
    """
    root = Path(root)
    record = _load(root)
    arms = list(record.get("arms") or [])
    arms.append({"label": label, "at": now(), "started": bool(started),
                 "scored": metric_value is not None,
                 "metric_value": metric_value, "metric_utility": metric_utility,
                 "baseline": bool(baseline),
                 "episodes": (actual_episodes if isinstance(actual_episodes, int) and
                              not isinstance(actual_episodes, bool) and actual_episodes > 0
                              else None),
                 "successes": (successes if isinstance(successes, int) and
                               not isinstance(successes, bool) else None),
                 "metric_kind": metric_kind,
                 # The settings the arm was measured at, because whether these arms were
                 # measured the same way is a question about this field and nothing else.
                 # Recorded here it can be read back; left out, the check comparing them is
                 # comparing empty dictionaries and always finds them identical.
                 "settings": dict(settings or {})})
    atomic_json(root / RECORD, {"schema_version": 1,
                                "protocol_sha256": protocol_sha256,
                                "arms": arms})
    return summary(root)


def summary(root: Path, *, alpha: float = 0.05) -> dict[str, Any]:
    """The record, reduced to what a reader of the run's result needs.

    Nothing here decides that anything improved. It reports the maximum and how many draws it
    was the maximum of, and -- when the baseline is one of the arms and both carry a
    denominator -- whether the two rates are distinguishable at all.
    """
    root = Path(root)
    arms, sources = arms_of(root)
    scored = [arm for arm in arms if arm.get("scored") and
              isinstance(arm.get("metric_utility"), (int, float))]
    result: dict[str, Any] = {
        "protocol_sha256": _load(root).get("protocol_sha256") or "",
        "attempted": len(arms), "scored": len(scored),
        "failed": len(arms) - len(scored),
        "counted_from": sources,
        "best": None, "baseline": None, "comparison": None,
        "identical_reading": False, "arms": [{key: arm.get(key) for key in
                                              ("label", "scored", "metric_value",
                                               "metric_utility", "baseline", "episodes",
                                               "successes", "metric_kind")}
                                             for arm in arms]}

    def strongest(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        # `metric_utility` is already direction-normalised by the metric contract, so the
        # maximum is the best arm whichever way the native metric points.
        #
        # Ties go to the baseline, and that is stated rather than left to the order the arms
        # were appended in. An arm reading exactly the same as the configuration the run
        # started from has not improved on it, and calling that arm "best" would put a gain
        # into the run's own summary that its own numbers do not contain.
        return max(rows, key=lambda arm: (arm["metric_utility"], bool(arm.get("baseline")))) \
            if rows else None

    # Whether these arms were measured under one protocol or under whatever each was run
    # with. A best-of-N means the N were measured the same way; without the frozen protocol
    # that is an assumption about the run rather than a fact from it, and the settings are
    # recorded per arm so it can be checked instead of assumed.
    digests = {object_digest(arm.get("settings") or {}) for arm in arms}
    result["one_protocol"] = len(digests) <= 1 and bool(result["protocol_sha256"])
    result["distinct_settings"] = len(digests)
    result["best"] = strongest(scored)
    baseline = [arm for arm in scored if arm.get("baseline")]
    result["baseline"] = strongest(baseline)

    if not scored:
        result["verdict"] = "nothing_measured"
        result["reading"] = ("No arm produced a number, so the run has no result to report. "
                             "Its findings are the failures, which are recorded per arm.")
        return result

    count = len(scored)
    result["best_is_the_maximum_of"] = count
    best = result["best"]
    comparable = count == 1 or result["one_protocol"]
    result["reading"] = (
        f"{best.get('metric_value')} is the best of {count} arm"
        f"{'s' if count != 1 else ''} "
        # "at this protocol" is a claim, and it is withheld where the arms' settings disagree
        # or none was frozen -- the sentence is not allowed to assert the comparability that
        # the next clause is about to deny.
        + ("measured in this run" if comparable else "taken in this run")
        + (f" ({len(arms) - count} further arms were started and produced no number)"
           if len(arms) > count else "")
        + ". " + ("One arm was measured, so it is a measurement rather than a maximum."
                  if count == 1 else
                  "It is a maximum of draws and not a measurement of the candidate: a run "
                  "that measures many arms and reports the best reports the highest noise "
                  "among them as well as the best candidate."))
    if not comparable and count > 1:
        # `best of N` presumes the N were measured the same way. Where no protocol was frozen,
        # or the arms' own settings disagree, that is an assumption about the run rather than
        # something read out of it -- and the arms are not comparable, so calling one of them
        # the best of the others is a comparison the run never fixed.
        result["reading"] += (
            " These arms were not all measured under one frozen protocol"
            + (f" ({result['distinct_settings']} distinct setting sets)" if
               result["distinct_settings"] > 1 else
               " (no protocol was frozen for this run)")
            + ", so they are not comparable and the count is a count of arms taken rather "
              "than of comparable draws.")

    # A maximum of draws whose members all read the same is worth saying out loud, in both
    # directions: it is evidence the benchmark cannot tell the arms apart, and it is what a
    # loop that is re-measuring one unchanged policy looks like.
    utilities = {round(float(arm["metric_utility"]), 12) for arm in scored}
    result["identical_reading"] = len(utilities) == 1 and count > 1
    if result["identical_reading"]:
        result["reading"] += (" Every arm scored identically, which this measurement cannot "
                              "distinguish from the arms being the same.")

    if result["baseline"] and len(scored) > 1:
        left, right = result["baseline"], best
        if right.get("label") != left.get("label"):
            n_left, n_right = left.get("episodes"), right.get("episodes")
            values = [left.get("metric_value"), right.get("metric_value")]
            if (left.get("metric_kind") == "binary" and right.get("metric_kind") == "binary"
                    and all(isinstance(one, (int, float)) and 0 <= one <= 1 for one in values)
                    and isinstance(n_left, int) and isinstance(n_right, int)
                    and isinstance(left.get("successes"), int)
                    and isinstance(right.get("successes"), int)
                    and 0 <= left["successes"] <= n_left
                    and 0 <= right["successes"] <= n_right
                    and abs(left["metric_value"] - left["successes"] / n_left) < 1e-9
                    and abs(right["metric_value"] - right["successes"] / n_right) < 1e-9):
                comparison = compare_proportions(
                    left["successes"], n_left, right["successes"], n_right, alpha=alpha)
                result["comparison"] = {
                    **comparison.as_dict(),
                    "counts_reconstructed": False,
                    "counts_from": "native completed episode receipts"}
            else:
                result["comparison"] = {
                    "verdict": "not_established",
                    "why": ("native completed episode counts and matching binary success "
                            "totals are unavailable, so no interval is established"),
                    "limitation": "two aggregate numbers without verified episode totals"}
    if result["comparison"] is None:
        # Not a fault. A run that measured one arm has nothing to compare it against, and a
        # baseline measured in an earlier run under an earlier protocol is not this run's
        # baseline. Saying which of those it is matters more than filling the field.
        result["comparison"] = {
            "verdict": "not_established",
            "why": ("no baseline arm was measured in this run at this protocol"
                    if not result["baseline"] else
                    "the best arm is the baseline, so there is nothing to compare"),
            "limitation": "nothing in this record establishes a difference"}
    return result


def rate_of(arm: dict[str, Any] | None) -> dict[str, Any] | None:
    """One arm's binary rate with an interval, when native counts are verified."""
    if not arm or not isinstance(arm.get("metric_value"), (int, float)):
        return None
    count = arm.get("episodes")
    successes = arm.get("successes")
    if (arm.get("metric_kind") != "binary" or not isinstance(count, int) or count <= 0
            or not isinstance(successes, int) or not 0 <= successes <= count
            or not 0 <= arm["metric_value"] <= 1
            or abs(arm["metric_value"] - successes / count) >= 1e-9):
        return None
    return proportion(successes, count)


def fingerprint(root: Path) -> str:
    """The record's bytes, so a document can say which version of it it quoted."""
    record = _load(Path(root))
    return object_digest(record) if record else ""
