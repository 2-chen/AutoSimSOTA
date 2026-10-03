"""The run, written down so that a person can read it.

A run leaves a tree full of records and none of them is a document. The answer to "what did it
do, and why" is spread across `events.json`, `measurements/*.json`, `decisions.json`, the
exchanges, the observations and the recipe, each written for a program to read back, each
individually unreadable as an account of what happened.

This assembles them into `RUN.md`, and it is built around one rule:

**Every section says whether it was computed or written.** The timeline, the numbers, the
artifact index, the unresolved list and the reproduction steps are derived from the run's own
files: they cannot be wrong about the run without the run's files being wrong. The summary is
written by a model, and a model can be wrong. If the two look alike on the page, a reader has
no way to tell which sentences can be checked -- and a document whose checkable and uncheckable
halves are indistinguishable is worth less than `ls`, because at least `ls` does not mislead.

So the written section is bracketed, attributed, time-stamped, and *checked*: every number in
it is looked for in the computed record, and the ones that cannot be found are listed as such.
The check establishes that a number appears in the record and not that it is true -- those are
different claims, and saying which one was established is the whole point.

Two rules carried over from the observer next door, for the same reasons. **It never raises**:
a document generator that dies on a malformed record fails exactly when the run has gone wrong
and the document is most wanted. **It never decides**: it reports what the record says, and a
section it could not assemble says so rather than being left out -- "nothing to report" and
"nothing was looked at" must not read the same.
"""

from __future__ import annotations

import json
import fcntl
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from contextlib import contextmanager

from . import media_manifest, mermaid
from .claims import check_claims, known_numbers, write_summary
from .selection import summary as selection_summary
from .common import atomic_json, atomic_text, digest, now, object_digest, read_json, redact
from .research_state import ResearchStateError, verify_event_rows

#: The two kinds of statement this document makes, and they are not degrees of each other.
COMPUTED = "computed"
WRITTEN = "written"

#: Where a generator looks for a file the run did not write into its own directory -- the
#: recipe, the plan, the derivation -- before giving up. The run root sits inside the
#: provisioning directory that holds them, under a name this module does not need to know.
NEARBY_DEPTH = 3

#: How many files of one kind the artifact index lists before it summarises the rest. High
#: enough that a small run is fully enumerated and low enough that a large one stays readable.
LISTED_PER_KIND = 12


@dataclass
class Section:
    """One part of the document, with where it came from and how far it can be trusted."""

    title: str
    provenance: str                  # COMPUTED | WRITTEN
    sources: list[str] = field(default_factory=list)
    body: str = ""
    notes: list[str] = field(default_factory=list)


# -- reading the run's own files -----------------------------------------------------------

def _load(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def nearby(root: Path, name: str) -> Path | None:
    """A file the run's directory may not hold but its surroundings do.

    The recipe and the plan belong to the provisioning step, which is one or two levels above
    the run. Looking upward by a fixed depth rather than by a path this module is told keeps
    it working for a runner that nests differently -- and the search is reported, so a reader
    knows which copy was used.
    """
    here = Path(root).resolve()
    for level in range(NEARBY_DEPTH + 1):
        candidate = here / name
        if candidate.is_file():
            return candidate
        if here.parent == here:
            break
        here = here.parent
    return None


#: The numbers in a full-pipeline round's development summary that are worth a column. The
#: rest are in the file; a table with thirty columns is a table nobody reads.
_ROUND_READINGS = ("episode_count", "success_count", "average_action_steps",
                   "average_inference_time_per_episode_seconds")


def read_rounds(root: Path) -> list[dict[str, Any]]:
    """The full pipeline's rounds, where the proposal and its result are two separate files.

    This pipeline keeps what the controller proposed in `round_N/proposal.json` and what came
    of it in `round_N/round_result.json`, written minutes apart by different code. Joining them
    here is a reconstruction and the document says so: the pair is not a decision record, and
    reading one as the other would hide that the system never wrote down the reasoning at the
    moment it decided -- which is the thing this whole layer exists to fix.
    """
    out: list[dict[str, Any]] = []
    for directory in sorted((Path(root) / "rounds").glob("round_*"),
                            key=lambda one: _round_number(one.name)):
        proposal_path, result_path = directory / "proposal.json", directory / "round_result.json"
        proposal_raw, result = _load(proposal_path), _load(result_path)
        if not isinstance(result, dict):
            result = {}
        proposal = proposal_raw.get("proposal") if isinstance(proposal_raw, dict) else None
        proposal = proposal if isinstance(proposal, dict) else {}
        summary = result.get("development_summary")
        summary = summary if isinstance(summary, dict) else {}
        out.append({
            "round": _round_number(directory.name),
            "at": result.get("completed_at") or (proposal_raw or {}).get("created_at"),
            "hypothesis": str(proposal.get("hypothesis") or ""),
            "varied": {k: v for k, v in (proposal.get("training") or {}).items()
                       if not isinstance(v, dict)} if isinstance(proposal.get("training"), dict)
                      else {},
            "status": str(result.get("status") or ("proposed" if proposal_raw else "no record")),
            "error": result.get("error") or result.get("reason"),
            "success_rate": summary.get("success_rate"),
            "readings": {k: summary[k] for k in _ROUND_READINGS if k in summary},
            "proposal_path": str(proposal_path) if proposal_path.is_file() else "",
            "result_path": str(result_path) if result_path.is_file() else "",
            "sha256": (proposal_raw or {}).get("response_sha256"),
        })
    return out


def read_scout(root: Path) -> dict[str, Any] | None:
    """The onboarding runs, which are a third shape and were the majority of this machine's work.

    A scout run derives a benchmark's declaration by asking a model to read the repository and
    then checking the answer against the checkout. Its records are a different set of files --
    `draft_<stage>_<attempt>.json` for each exchange, `verification.json` for the verdict,
    `declaration.json` for what survived -- and none of them were in any index, so thirty-eight
    attempts to onboard two benchmarks were invisible to everything that reads runs.

    Every draft is kept, including the rejected ones, and that is the most informative thing
    here: a draft that was refused after verification is the record of a claim that reading
    could not settle.
    """
    root = Path(root)
    drafts = sorted(root.glob("draft_*.json"))
    if not drafts and not (root / "verification.json").is_file() \
            and not (root / "declaration.json").is_file():
        return None
    rows: list[dict[str, Any]] = []
    for path in drafts:
        record = _load(path) or {}
        stage = str(record.get("stage") or path.stem)
        attempt = _round_number(str(record.get("attempt") or path.stem))
        provider = record.get("provider") or {}
        rows.append({
            "stage": stage, "attempt": attempt,
            "model": str(provider.get("provider_model") or provider.get("requested_model") or ""),
            "prompt_chars": record.get("prompt_chars"), "response_chars": record.get("response_chars"),
            "path": str(path), "sha256": object_digest(str(record.get("content") or "")),
        })
    verification = _load(root / "verification.json")
    declaration = _load(root / "declaration.json")
    onboarding = _load(root / "onboarding.json")
    failed = [check for check in (verification or {}).get("checks", [])
              if isinstance(check, dict) and not check.get("passed")]
    return {
        "drafts": rows,
        "stages": sorted({row["stage"] for row in rows}),
        "attempts": max((row["attempt"] for row in rows), default=0),
        "verification": verification if isinstance(verification, dict) else None,
        "declaration": declaration if isinstance(declaration, dict) else None,
        "onboarding": onboarding if isinstance(onboarding, dict) else None,
        "failed_checks": failed,
        "tasks": (verification or {}).get("task_count") if isinstance(verification, dict) else None,
    }


def _round_number(name: str) -> int:
    digits = "".join(one for one in name if one.isdigit())
    return int(digits) if digits else 0


def read_processes(root: Path, limit: int = 60) -> list[dict[str, Any]]:
    """Every command this run is recorded as having executed, from the records of running them.

    The full pipeline does not write an event stream, but it does write one of these beside
    each process it starts: the argv, the working directory, when it began, how long it took
    and what it returned. That is a timeline and a reproduction recipe in the same file, and it
    was there the whole time -- nothing had read it back out.
    """
    out: list[dict[str, Any]] = []
    root = Path(root)
    for path in sorted(root.rglob("process.json")):
        if path.parent.name != "process":
            continue
        record = _load(path)
        if not isinstance(record, dict) or not record.get("command"):
            continue
        try:
            label = str(path.parent.parent.relative_to(root))
        except ValueError:
            label = str(path.parent.parent)
        out.append({"label": label, "at": record.get("started_at"),
                    "finished_at": record.get("finished_at"), "command": record["command"],
                    "cwd": record.get("cwd"), "returncode": record.get("returncode"),
                    "seconds": record.get("elapsed_seconds"), "status": record.get("status"),
                    "path": str(path)})
        if len(out) >= limit:
            break
    # By the clock, because this is a timeline. Sorted by path -- which is how they are found --
    # the selection evaluation appears above the round it selects from.
    return sorted(out, key=lambda row: str(row.get("at") or ""))


def gather(root: Path) -> dict[str, Any]:
    """Everything the run left, read once, with what could not be read named.

    Nothing here is interpreted. A missing file is an entry in `unreadable` rather than an
    absent key, because every section below has to be able to say whether it is empty because
    there was nothing or because there was no material.
    """
    root = Path(root)
    source: dict[str, Any] = {"root": str(root), "unreadable": []}

    def nearby_record(name: str) -> Path | None:
        """Find a run-wide record beside this phase's nested research directory."""
        here = root
        for _ in range(4):
            candidate = here / name
            if candidate.is_file():
                return candidate
            if here.parent == here:
                break
            here = here.parent
        return None

    def relative_ref(path: Path) -> str:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")

    def want(key: str, path: Path | None) -> Any:
        # Three states, kept apart. A file this run never wrote is not a fault -- a run that
        # predates a feature has no file for it, and calling that "unreadable" would report
        # every old run as damaged. A file that is there and cannot be parsed is a fault. The
        # difference matters because only the second one is worth acting on.
        if path is None or not path.is_file():
            source["unreadable"].append(f"{key}：这次运行没有写这个文件")
            return None
        value = _load(path)
        if value is None:
            source["unreadable"].append(f"{key}：文件在，但读不出来（{path}）")
        return value

    source["events"] = want("events", root / "events.json") or {}
    shared_events_path = nearby_record("run_events.json")
    source["run_events_path"] = relative_ref(shared_events_path) if shared_events_path else ""
    source["run_events"] = (want("run_events", shared_events_path) or {}
                             if shared_events_path else {})
    if shared_events_path and source["run_events"]:
        try:
            if source["run_events"].get("schema_version") == 2:
                # Human-facing tail projection is bounded. Full history verification is
                # the state store's job, not a synchronous per-message report task.
                journal = shared_events_path.parent / "run_events.jsonl"
                rows = []
                if journal.exists():
                    with journal.open("rb") as stream:
                        size = journal.stat().st_size
                        stream.seek(max(0, size - 1024 * 1024))
                        if size > 1024 * 1024:
                            stream.readline()
                        data = stream.read()
                    if data and not data.endswith(b"\n"):
                        raise ResearchStateError("journal tail has an incomplete append")
                    rows = [json.loads(line) for line in data.splitlines()][-100:]
                if rows:
                    verify_event_rows(rows, start_sequence=rows[0]["sequence"],
                                      previous_hash=rows[0]["previous_hash"])
                source["run_events"]["rows"] = rows
                source["run_events"]["projection_scope"] = "bounded_verified_tail"
            event_rows = source["run_events"].get("rows")
            if not isinstance(event_rows, list) or not all(
                    isinstance(row, dict) for row in event_rows):
                raise ResearchStateError("run event log rows are invalid")
            if source["run_events"].get("schema_version") != 2:
                verify_event_rows(event_rows)
        except (ResearchStateError, TypeError, ValueError, KeyError, OSError) as exc:
            source["unreadable"].append(
                f"run_events：共享事件链校验失败（{type(exc).__name__}: {exc}）")
    source["decisions"] = want("decisions", root / "decisions.json") or {}
    source["observations"] = want("observations", root / "observations.json") or {}
    source["report"] = want("research_report", root / "research_report.json")
    protocol_path = root / "comparison_protocol.json"
    source["comparison_protocol"] = (_load(protocol_path) if protocol_path.is_file()
                                     else None)
    if protocol_path.is_file() and source["comparison_protocol"] is None:
        source["unreadable"].append(f"comparison_protocol：文件在，但读不出来（{protocol_path}）")
    source["measurements"] = [
        {"path": str(path), "record": record}
        for path in sorted((root / "measurements").glob("*.json"))
        for record in [_load(path)]
        if isinstance(record, dict)
    ]
    source["rounds"] = read_rounds(root)
    source["processes"] = read_processes(root)
    source["scout"] = read_scout(root)
    state_path = nearby_record("run_state.json")
    source["state_path"] = relative_ref(state_path) if state_path else ""
    source["state"] = want("run_state", state_path) if state_path else None
    # Computed, not read. This key used to be `selection.json`, which nothing in the package
    # has written since the second loop was removed -- so every run's document carried a line
    # saying the file was missing, and the two sections that consume this key branched on
    # fields (`excluded_candidates`, `qualifies_for_final`) that could never be present. A
    # reader was being shown, in the "unresolved" section, a rule about final qualification
    # that no run had ever been judged by.
    #
    # What the key holds now is derived from the measurement files themselves, which is the
    # one place an arm is definitely recorded.
    source["selection"] = selection_summary(root)
    # What the run could have done, what it was not allowed to do, and how far it got. Three
    # documents the loop writes for itself and the record did not read -- so a run that spent
    # every round reaching a conclusion and then stopped had its reasoning in the run root and
    # none of it in the account.
    for name in ("ideas.json", "rubric.json", "audits.json", "snapshots/snapshots.json"):
        key = name.removesuffix(".json").replace("/", "_")
        path = root / name
        source[key] = want(key, path)
        source[f"{key}_path"] = str(path) if path.is_file() else ""
    source["exchanges"] = [
        {"path": str(path), "record": record}
        for path in sorted((root / "exchanges").glob("*.json"))
        for record in [_load(path)]
        if isinstance(record, dict)
    ]
    for name in ("recipe.json", "plan.json", "derived_stages.json",
                 "environment.json", "seed.json", "budget.json"):
        key = name.removesuffix(".json")
        path = nearby(root, name)
        source[key] = want(key, path)
        source[f"{key}_path"] = str(path) if path else ""
    source["survey"] = survey(root)
    return source


@dataclass
class Bucket:
    """Files of one kind: how many, how big, and a few of them by name.

    Counting and keeping a fixed number of examples, rather than collecting every path and
    slicing afterwards, because the largest run root here holds twenty-eight thousand files
    and the index is a document, not a manifest. Collecting them all to then show twelve is a
    manifest nobody reads paid for at every stage boundary.
    """

    what: str
    caveat: str
    count: int = 0
    bytes: int = 0
    #: (relative path, its own size). The size travels with the path because the column it
    #: lands in says "size", and printing the group's mean there instead -- which this did --
    #: puts a number in a per-file column that is not any file's size.
    examples: list[tuple[str, int]] = field(default_factory=list)
    #: "recording" for a bucket of episodes. A flag rather than a test on the description:
    #: matching Chinese substrings to decide what a file is put the evaluator's recordings in
    #: the wrong bucket, because its description says "录下的回合" and not "录像".
    kind: str = "file"

    def see(self, relative: str, size: int, *, examples: int) -> None:
        self.count += 1
        self.bytes += size
        if len(self.examples) < examples:
            self.examples.append((relative, size))


def survey(root: Path, *, examples: int = LISTED_PER_KIND) -> dict[tuple[str, str], Bucket]:
    """Every file under the run root, bucketed by what it is and what it can be evidence of.

    Walks rather than recursing into a list, so that a run root holding a full repository copy
    plus a dataset costs one pass and a bounded amount of memory.
    """
    out: dict[tuple[str, str], Bucket] = {}
    root = Path(root)
    media_events = _load(root / "media_events.json") or {}
    demo_paths = {str(item.get("path")) for event in media_events.get("rows", [])
                  if isinstance(event, dict) and event.get("status") == "captured"
                  for item in (event.get("media") or [])
                  if isinstance(item, dict) and item.get("path")}
    for directory, subdirectories, names in os.walk(root, followlinks=False):
        # Symlinked directories are not descended into: this system creates them to point a
        # non-ASCII path at an ASCII one, and following them would list the same files twice
        # under two names and call it two pieces of evidence.
        subdirectories[:] = [one for one in subdirectories
                             if not Path(directory, one).is_symlink()]
        for name in names:
            if name.endswith((".tmp", ".lock")):
                continue
            path = Path(directory, name)
            if path.is_symlink():
                # A media link may point outside this run. Neither the document nor its
                # manifest may present those bytes as an owned experiment artifact.
                continue
            try:
                size = path.stat().st_size
                relative = str(path.relative_to(root))
            except OSError:
                continue
            if name in _IS_THE_DOCUMENT or relative == "media/manifest.json":
                # The document does not index itself. Not modesty: its own size changes when
                # it is regenerated, so listing it would make two readings of the same
                # unchanged run differ by the length of the account, and there is no fixed
                # point. It is the account, not evidence about the run.
                continue
            what, caveat = role_of(relative)
            bucket_kind = "file"
            if relative in demo_paths and path.suffix.lower() in media_manifest.MEDIA_SUFFIXES:
                what = "原生录制阶段回执报告的媒体文件"
                caveat = "阶段回执报告了该路径；当前字节须与媒体清单中的 SHA-256 匹配后，"
                caveat += "才能核验归属于该尝试；哈希延期或不匹配时只作待验证文件；"
                caveat += "不证明 episode 结果或代表性"
                bucket_kind = "recording"
            elif what.startswith(_RECORDING_ROLE) and (recording := _recording_kind(relative)):
                what, caveat = recording
                bucket_kind = "recording"
            out.setdefault((what, caveat), Bucket(what, caveat, kind=bucket_kind)).see(
                relative, size, examples=examples)
    return out


# -- what each kind of file is for ---------------------------------------------------------

#: What a file is, and what it does not establish. The second half is the point: an index that
#: lists artifacts and stops has told a reader that evidence exists without telling them what
#: it can bear. Ordered by specificity, because `measurements/round_1.json` matches more than
#: one entry and the more particular description is the true one.
#:
#: In Chinese, like the rest of the document. The code around it is in English and the document
#: is not: it is written to be read by a person, and every other report in this repository that
#: a person reads is in Chinese.
_ROLES: tuple[tuple[str, str, str], ...] = (
    ("run_events.json", "准备与研究共用的追加式事件流，含完整性哈希链",
     "链可检测意外改写，不证明记录内容本身为真；仍须对照对应 attempt 回执和原始产物"),
    ("run_state.json", "从共享事件流投影出的当前阶段快照和恢复入口",
     "活动动作的结果可能未知；恢复前须对照回执、输出与仍存活进程"),
    ("decisions.json", "这次运行做过的每个决定、依据、以及结果",
     "结果是事后回填的；一条仍标着 open 的决定说明没有人回头看它导致了什么，"
     "而不是说它什么都没导致"),
    ("events.json", "系统按顺序做了什么，以及它当时给的理由",
     "是时序，不是因果：它记的是系统自己的动作，而一个从未执行的步骤在这里不留任何痕迹"),
    ("observations.json", "记录里与记录其余部分对不上的地方",
     "是**跑过的**那几项检查，不是应该跑的那几项；一个没人想到要看的矛盾不在这里"),
    ("research_report.json", "循环自己对每一轮的判定，包括那些没有产生数字的轮次",
     "是跑这个循环的那段代码写的，所以它报告的是循环自己的看法 —— 要对着 `measurements/` 读"),
    ("comparison_protocol.json", "本轮冻结的评测设置、主指标与评测节点身份",
     "只覆盖通用键和明确声明的协议键；它不证明原生程序没有隐式读取其他设置，"
     "也不等于独立复评"),
    ("graph_runs/", "显式执行图每次尝试的节点顺序、绑定与实际回执",
     "拓扑完成不等于指标有效；要与同次 `measurements/` 及原生日志一起核对"),
    ("measurements/", "一次测量：命令、它打印的东西、以及从中读出的数字",
     "它证明了命令跑过、以及在它跑的时候说了什么。它不证明这个数字会重复，"
     "也不证明与另一组的比较是公平的"),
    ("exchanges/", "控制器被喂了什么、它答了什么，逐字保存",
     "产出提案的那个推理过程。它不证明该推理成立，"
     "而一个看起来合理的提案也不是它起了作用的证据"),
    ("telemetry.jsonl", "评测过程中逐步记录下来的位姿",
     "唯一能离屏重建场景的源；它是轨迹，不是对轨迹的评分"),
    ("recipe.json", "把本机配置成这样的那些命令，以及当时已经存在的东西",
     "它证明的是环境在**这台机器上**是这么建起来的，不是同一批命令在别处也成立"),
    ("derived_stages.json", "每个阶段的命令是怎么得出来的、依据是什么",
     "是推导过程，不是它所依据的那份声明"),
    ("plan.json", "在什么都还没测之前，系统打算做什么",
     "是意图。它若与时间线不一致，时间线才是发生过的"),
    ("rubric.json", "这次运行的目标被拆成的问题，以及每一问走到哪一步",
     "分数衡量的是**走了多远**，不是结论有多好。一个拿到 0.4 的运行可能只是在"
     "测不出数字的地方多走了几步，这不等于它比 0.3 的那个更接近答案"),
    ("ideas.json", "这次运行考虑过的所有做法，包括没被选中的",
     "它是考虑过什么，不是做到了什么。一条从未被选中的想法在这里看起来"
     "和一条跑失败的一样安静"),
    ("audits.json", "每条想法在跑之前对着红线判定的结果",
     "判定依据的是想法**自己怎么描述自己**：一条描述得含糊到看不出会动评测的改动，"
     "在这里也会是 cleared"),
    ("snapshots/", "改过哪些文件、改前的样子、以及目前最好的那个状态",
     "快照只覆盖这次运行**自己动过**的文件，不是整份 checkout；"
     "`_best` 指向的是记录里最好的状态，不证明它在别处也最好"),
    ("RUN.md", "本文件",
     "对上面这些文件的第二手叙述；它与它们不一致时，以它们为准"),
)


#: Artifacts matched by their own name wherever they sit, because these belong beside the
#: records they were computed from rather than in one place at the top of the run.
_BY_NAME: tuple[tuple[str, str, str], ...] = (
    ("trajectory.svg", "评测过程中记录下来的位姿画成的轨迹图",
     "**它是轨迹，不是回放**：没有东西被渲染，没有相机被模拟。它显示东西在哪里，"
     "不显示看起来什么样"),
    ("trajectory.json", "从位姿记录里算出来的数字：局数、物体、路程、终点距离",
     "它和同目录的 `trajectory.svg` 是同一批输入的两种输出；输入没变时它们不重算"),
    ("evaluation_metrics.json", "一次评测的结果：每一局的种子与成败、动作步数、推理耗时",
     "它是一次评测的**结论**。同一份种子上换一个评测器、换一批回合，"
     "这个数字可以完全不同 —— 它是这一次的结果，不是这个策略的水平"),
    ("evaluation_request.json", "这次评测**被要求**了什么：检查点、种子、局数、要不要录像",
     "被要求了什么，不是得到了什么。它与 `protocol.json` 分开存放，"
     "因为后者要被各分片逐一比对"),
    ("protocol.json", "这次评测自己声明的身份：冻结的文件、检查点哈希、种子、局数",
     "是这次运行的自我声明。它与别的运行能不能比，要看两份协议的差，不在这份文件里"),
    ("shard_plan.json", "种子库怎么切给各设备的，切之前就冻住了",
     "切法本身。它不证明每一片都跑完了 —— 那要看合起来的那份结果"),
)


def role_of(relative: str) -> tuple[str, str]:
    """What this file is for, and what it cannot bear. Unknown files say so."""
    name = relative.rsplit("/", 1)[-1]
    for candidate, what, caveat in _BY_NAME:
        if name == candidate:
            return what, caveat
    for prefix, what, caveat in _ROLES:
        if relative == prefix or relative.startswith(prefix):
            return what, caveat
    if relative.endswith(".log"):
        return ("某阶段按原样打印出来的输出",
                "保留下来的那一段。日志是程序选择要打印的东西，"
                "而一个什么都没打印的程序在这里与一个正常工作的程序无法区分")
    if relative.endswith((".pth", ".pt", ".ckpt", ".safetensors", ".bin")):
        return ("这次运行的某个阶段产出的学习产物",
                "只证明文件存在。它能做什么是关于它的一个主张，"
                "而这个主张来自测量，不来自这份清单")
    # Scene material before media, because a photograph of a table is an asset and calling it
    # a recording of the run would be a plain misreading of the file. Every simulator ships a
    # directory of these, under one of a handful of names, and the extension cannot tell them
    # apart from a screenshot.
    if any(part in relative.lower() for part in
           ("asset", "texture", "mesh", "material", "urdf", "scene/", "assets/")):
        return ("场景素材 —— 这个 benchmark 用来搭出场景的贴图、网格或模型",
                "它属于被仿真的对象，不属于仿真结果；它不会因为这次运行成功或失败而不同")
    if relative.endswith((".mp4", ".gif", ".webm", ".mov", ".avi")):
        return (_RECORDING_ROLE, "它显示的是**当时**发生了什么。看的人可以据此形成一个判断，"
                                 "但录像不构成一次测量 —— 除非里面有明确标注的数字")
    if relative.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
        return ("一张图或一份可打印的材料", "它的内容是什么，要看它本身；"
                                            "这份清单只知道它在这里")
    if relative.endswith((".jsonl", ".sqlite", ".db")):
        return ("一份过程记录，按行或按库追加", "它记下了记录的当时；"
                                                "它不是对这些记录的解释")
    return ("这次运行的一个产物", "这里对它的用途一无所知，因此不应该对它做任何解读")


# -- the computed sections -----------------------------------------------------------------

def _clock(value: Any) -> str:
    text = str(value or "")
    return text[11:19] if len(text) > 19 and text[10] == "T" else text


def timeline(source: dict[str, Any]) -> Section:
    """What happened, in the order it happened, from the timestamps the system wrote."""
    rows = [row for row in (source.get("events") or {}).get("rows", [])
            if isinstance(row, dict)]
    shared_rows = []
    for row in (source.get("run_events") or {}).get("rows", []):
        if not isinstance(row, dict):
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        action = details.get("action") if isinstance(details.get("action"), dict) else {}
        shared_rows.append({
            "at": row.get("at"),
            "event": f"{row.get('phase') or 'run'}:{row.get('event') or '?'}",
            "stage": (details.get("stage") or details.get("step") or action.get("step") or
                      details.get("controller_event") or details.get("local_event_sequence") or ""),
            "status": row.get("status"),
            "why": details.get("why") or details.get("reason") or "",
            "sequence": row.get("sequence"),
            "source": "run_events.json",
        })
    def sequence_number(row: dict[str, Any]) -> int:
        try:
            return int(row.get("sequence") or 0)
        except (TypeError, ValueError):
            return 0

    rows = sorted([*rows, *shared_rows],
                  key=lambda row: (str(row.get("at") or ""), sequence_number(row)))
    body: list[str] = []
    if rows:
        body += ["| 时刻 | 事件 | 内容 | 结果 |", "| --- | --- | --- | --- |"]
        for row in rows:
            event = str(row.get("event") or "?")
            detail = str(row.get("stage") or row.get("round") or row.get("subject") or "")
            outcome = (row.get("why") or row.get("error") or row.get("detail") or
                       row.get("status") or "")
            if row.get("seconds") is not None:
                outcome = f"{outcome} ({row['seconds']}s)".strip()
            if row.get("returncode") is not None:
                outcome = f"{outcome} rc={row['returncode']}".strip()
            body.append(f"| {_clock(row.get('at'))} | {event} | {detail} | {outcome} |")
    elif source.get("rounds"):
        # The other pipeline keeps no event stream; its rounds are its timeline.
        body += ["| 时刻 | 轮次 | 状态 | 成功率 |", "| --- | ---: | --- | ---: |"]
        for row in source["rounds"]:
            rate = row.get("success_rate")
            body.append(f"| {_clock(row.get('at'))} | 第 {row['round']} 轮 | {row['status']} | "
                        f"{'—' if rate is None else f'{rate:.3f}'} |")
    elif source.get("scout"):
        # The onboarding runs keep no clock per step. What they have is the order the drafts
        # were made in, which is the order the work happened in.
        scout = source["scout"]
        body += ["| 顺序 | 阶段 | 第几次 | 模型 | 提问字数 | 回答字数 |",
                 "| ---: | --- | ---: | --- | ---: | ---: |"]
        for index, row in enumerate(scout["drafts"], start=1):
            body.append(f"| {index} | {row['stage']} | {row['attempt']} | {row['model'] or '—'} | "
                        f"{row['prompt_chars'] or '—'} | {row['response_chars'] or '—'} |")
        for key, label in (("onboarding", "onboarding.json"), ("declaration", "declaration.json")):
            record = scout.get(key) or {}
            if record.get("created_at"):
                body.append(f"\n{label} 写于 `{record['created_at']}`。")
    else:
        body.append("_没有事件记录。_")
    # Drawn under the table rather than instead of it: the table carries the exact numbers and
    # the chart carries the shape, and a reader who wants to check one against the other needs
    # both. The chart is skipped rather than approximated when nothing has a usable timestamp.
    bars = [{"at": row.get("at"), "seconds": row.get("seconds"),
             "label": row.get("stage") or row.get("event"), "section": row.get("event")}
            for row in rows]
    bars += [{"at": row.get("at"), "seconds": row.get("seconds"), "label": row.get("label"),
              "section": "commands"} for row in (source.get("processes") or [])]
    bars += [{"at": row.get("at"), "seconds": None, "label": f"round {row['round']}",
              "section": "rounds"} for row in (source.get("rounds") or []) if row.get("at")]
    chart = mermaid.gantt(bars, title="运行时间线")
    if chart:
        body += ["", "### 甘特图", "",
                 "_每个条的位置和长度都来自记录里的时间戳。没有可用时间戳的条目**不画** —— "
                 "一个画错位置的条是一句看起来像测量结果的假话。_", "", chart]
    processes = source.get("processes") or []
    if processes:
        body += ["", "### 记录下来的命令", "",
                 "| 开始 | 位置 | 命令 | 结果 |", "| --- | --- | --- | --- |"]
        for row in processes:
            shown = _one_line(" ".join(str(one) for one in row["command"]), 220)
            outcome = str(row.get("status") or "?")
            if row.get("returncode") is not None:
                outcome += f" rc={row['returncode']}"
            if row.get("seconds") is not None:
                outcome += f" ({row['seconds']}s)"
            body.append(f"| {_clock(row.get('at'))} | `{row['label']}` | `{shown}` | {outcome} |")
    state = source.get("state")
    if isinstance(state, dict):
        if state.get("status"):
            action = state.get("current_action")
            if not isinstance(action, dict):
                phase_state = (state.get("phases") or {}).get(state.get("phase"), {})
                action = phase_state.get("current_action") if isinstance(phase_state, dict) else None
            step = action.get("step") if isinstance(action, dict) else None
            phase = f"，阶段 `{state['phase']}`" if state.get("phase") else ""
            active = f"，当前动作 `{step}`" if step else ""
            body.append(f"\n运行状态：**{state['status']}**{phase}{active}")
        if state.get("error"):
            body.append(f"\n运行报的错：`{_one_line(state['error'], 400)}`")
    for measured in source.get("measurements") or []:
        record = measured["record"]
        argv = record.get("argv") or []
        if argv:
            body.append(f"\n**{record.get('label', '?')}** 运行的命令：\n\n```\n"
                        f"{' '.join(str(one) for one in argv)}\n```")
    sources = ["events.json", "measurements/*.json"]
    if source.get("run_events_path"):
        sources.append(str(source["run_events_path"]))
    if source.get("state_path"):
        sources.append(str(source["state_path"]))
    return Section("时间线", COMPUTED, sources, "\n".join(body))


def _readings_worth_showing(record: dict[str, Any]) -> dict[str, Any]:
    """The readings of a measurement, each named with the stage that printed it.

    `readings.named_numbers` deliberately does not filter -- which parts of what a program
    said matter is the caller's question. This is the caller answering it, and the answer
    used to be the wrong question: whenever `success_rate` was absent the whole column was
    hidden, on the theory that a run with no number printed only crash noise.

    The counterexample is the run that got furthest. A trainer that ran for three hours and
    printed its loss every epoch, followed by an evaluator that crashed, has `success_rate:
    None` and entirely real readings -- and every one of them was hidden, with the docstring's
    explanation ("these came from failed output") printed as though it had been established.

    **What tells a reading from crash noise is not another field. It is which stage printed
    it.** A number from a stage that ran is a reading; a number from a stage that did not run
    is the wreckage of its traceback. The measurement records both, per stage, so the document
    can say which it is showing instead of deciding whether to show anything.
    """
    by_stage = record.get("readings_by_stage")
    if isinstance(by_stage, dict):
        out: dict[str, Any] = {}
        for stage, values in by_stage.items():
            for name, value in (values or {}).items():
                out[f"{stage}.{name}"] = value
        return out
    return dict(record.get("readings") or {})


def numbers(source: dict[str, Any]) -> Section:
    """The numbers, with what they were measured under. Nothing here is inferred."""
    body: list[str] = []
    # Before the table of measurements: how many arms the run took, and therefore whether the
    # number at the top of it is a measurement or the highest of several draws. Stated here
    # rather than left to the reader because the table shows every arm as a row of equal
    # standing, and the run's own conclusion picks one of them -- which is a selection, and a
    # selection is a thing the document has to admit to rather than one a reader has to infer
    # from counting rows.
    confirmation = (source.get("report") or {}).get("confirmation")
    if isinstance(confirmation, dict) and confirmation.get("status"):
        # The one number in the run that the search did not get to pick. A run whose held-out
        # set was declared and never measured has left it on the table, and that is worth
        # saying plainly -- the alternative is a reader assuming the best number was checked.
        state = confirmation["status"]
        if state == "taken":
            body += [f"**未参与搜索的确认测量**：在 "
                     f"`{'/'.join(f'{k}={v}' for k, v in sorted((confirmation.get('held_out') or {}).items()))}`"
                     f" 上测得候选值 {confirmation.get('metric_value')}。这是预先保留的"
                     f"确认流程，不是搜索轮次中的最佳值。", ""]
            comparison = confirmation.get("comparison") or {}
            if confirmation.get("baseline_metric_value") is not None:
                body += [f"同协议留出基线：{confirmation.get('baseline_metric_value')}；"
                         f"改善结论：{comparison.get('verdict') or 'not_established'}。"
                         f"{comparison.get('why') or ''}", ""]
            statistical = comparison.get("statistical_evidence") or {}
            if statistical:
                body += [f"配对统计诊断：{statistical.get('verdict')}，"
                         f"n={statistical.get('n_candidate')}，"
                         f"区间={statistical.get('interval')}；"
                         "这不是未经审计的原生状态哈希可自动支持的 L4 结论。", ""]
        elif state == "available_not_taken":
            body += [f"**声明了未参与搜索的设置却没有测**：{confirmation.get('why')}。"
                     f"所以上面的最佳值仍然是搜索自己那些点里的最大值。", ""]
        elif state == "not_declared":
            body += [f"**没有未参与搜索的测量**：{confirmation.get('why')}。", ""]
        elif state == "attempted_and_failed":
            body += [f"**未参与搜索的确认测量失败了**：{confirmation.get('why')}。"
                     + ("评测已接触留出状态，不能重试或继续搜索。" if not confirmation.get(
                         "retryable", False) else
                        "评测尚未启动，修复启动条件后可重试，但不能继续搜索。"), ""]
        else:
            body += [f"**确认测量不可用**：{confirmation.get('why')}", ""]
    selection = source.get("selection")
    if isinstance(selection, dict) and selection.get("attempted"):
        best = selection.get("best") or {}
        count = selection.get("best_is_the_maximum_of") or 0
        scored, taken = selection.get("scored", 0), selection.get("attempted", 0)
        if count == 0:
            line = (f"这次运行取过 {taken} 个点，**没有一个产生数字** —— "
                    f"所以它没有可报的结果，它的收获是那些失败本身。")
        elif count == 1:
            line = (f"{best.get('metric_value')} 是这次运行唯一测出数字的点"
                    f"（共取过 {taken} 个），所以它是一次测量，不是若干次里的最大值。")
        else:
            line = (f"{best.get('metric_value')} 是 **{count} 个有数字的点里的最大值**"
                    f"（共取过 {taken} 个，其中 {taken - scored} 个没有产生数字）。"
                    f"挑最大值这个动作本身就会把噪声里最高的那个留下来，所以这个数字"
                    f"高于「候选真的更好」能解释的部分。")
        if not selection.get("one_protocol") and count > 1:
            line += ("这些点也不都在同一个冻结协议下测过"
                     + (f"（{selection['distinct_settings']} 套设置）"
                        if selection.get("distinct_settings", 0) > 1 else "")
                     + "，所以它们之间不可比。")
        body += [f"**这次搜索取了多少个点**：{line}", ""]
        comparison = selection.get("comparison") or {}
        if comparison.get("verdict") in ("better", "worse", "no_difference"):
            body += [f"**与基线的比较**：{comparison.get('why')}"
                     "。这是这一次比较的结论，不是对 benchmark 测试集的结论。", ""]
        elif comparison.get("verdict") == "not_established" and best.get("label") != "baseline":
            body += [f"**与基线的比较尚未确立**：{comparison.get('why')}。", ""]
    protocol = source.get("comparison_protocol")
    if isinstance(protocol, dict):
        settings = protocol.get("settings") or {}
        body += ["**本次比较冻结的条件**："
                 + (", ".join(f"`{key}`={redact(str(value))}" for key, value in
                              sorted(settings.items())) if isinstance(settings, dict) and settings
                    else "没有显式评测设置键"),
                 "协议仅覆盖通用键和已声明键；相同 seed 不自动证明相同初始状态。", ""]
    rows = source.get("measurements") or []
    if rows:
        body += ["| 测量 | 主指标 | 命令行里的数字 | 变化的设置 |", "| --- | ---: | --- | --- |"]
        for measured in rows:
            record = measured["record"]
            metric = record.get("metric") or {}
            rate = record.get("metric_value", record.get("success_rate"))
            metric_name = str(metric.get("name") or "success_rate")
            metric_unit = str(metric.get("unit") or "")
            readings = _readings_worth_showing(record)
            varied = {k: v for k, v in (record.get("varied") or {}).items()}
            body.append(
                f"| {record.get('label', '?')} | "
                f"{'—' if rate is None else f'{metric_name}={rate:.3f} {metric_unit}'.strip()} | "
                f"{', '.join(f'{k}={v}' for k, v in sorted(readings.items())) or '—'} | "
                f"{', '.join(f'{k}={v}' for k, v in sorted(varied.items())) or '—'} |")
        if any((measured["record"].get("readings_by_stage") or {}).get("evaluate")
               and measured["record"].get("metric_value", measured["record"].get("success_rate"))
               is None for measured in rows):
            body += ["", "有测量**没有**有效主指标，它的读数来自各阶段自己打印的东西。"
                         "一个跑了三小时的训练器报的 `loss` 是读数；评测崩了之后从 traceback 里"
                         "抓到的行号不是。两者现在按**打印它的阶段**分开列，读者自己判断 —— "
                         "把它们混在一栏里，是一次报错看起来像一次测量的原因。", ""]
    elif source.get("rounds"):
        body += ["| 轮次 | 成功率 | 其它读数 | 变化的设置 |", "| ---: | ---: | --- | --- |"]
        for row in source["rounds"]:
            rate = row.get("success_rate")
            body.append(
                f"| {row['round']} | {'—' if rate is None else f'{rate:.3f}'} | "
                f"{', '.join(f'{k}={v}' for k, v in sorted((row.get('readings') or {}).items())) or '—'} | "
                f"{', '.join(f'{k}={v}' for k, v in sorted((row.get('varied') or {}).items())) or '—'} |")
    else:
        body.append("_没有测量记录。_")
    rounds = (source.get("report") or {}).get("rounds") if source.get("report") else None
    if rounds:
        body += ["", "### 循环自己的说法", ""]
        for row in rounds:
            if not isinstance(row, dict):
                continue
            line = (f"- 第 {row.get('round', '?')} 轮："
                    f"**{row.get('status') or row.get('label') or '?'}**")
            value = row.get("metric_value", row.get("success_rate"))
            if value is not None:
                line += f"，{row.get('metric_name') or 'success_rate'} {value:.3f}"
            if row.get("why_not"):
                line += f" —— {row['why_not']}"
            body.append(line)
    return Section("数字", COMPUTED, ["measurements/*.json", "research_report.json"],
                   "\n".join(body))


def decision_section(source: dict[str, Any]) -> Section:
    """Each decision, its grounds, and what came of it -- open ones stated as open."""
    rows = [row for row in (source.get("decisions") or {}).get("rows", [])
            if isinstance(row, dict)]
    body: list[str] = []
    notes: list[str] = []
    sources = ["decisions.json"]
    opened = [row for row in rows if row.get("kind") == "decision"]
    if opened:
        body += ["| 谁 | 决定 | 为什么 | 结果 |", "| --- | --- | --- | --- |"]
        for row in opened:
            outcome = row.get("outcome") or {}
            if outcome.get("state") == "open":
                what = "**未回填**（没有人回头看它导致了什么）"
            else:
                what = _flatten(outcome.get("what"))
            body.append(f"| {row.get('by', '?')}：{row.get('agent') or '—'} | "
                        f"{_one_line(row.get('activity'))} | {_one_line(row.get('why'))} | {what} |")
    elif source.get("rounds"):
        # Not a decision record and not presented as one. The grounds and the consequence were
        # written to two different files by two different pieces of code, minutes apart, and
        # joining them is this document's work rather than the system's.
        sources = ["rounds/round_*/proposal.json", "rounds/round_*/round_result.json"]
        body += ["| 谁 | 决定 | 为什么 | 结果 |", "| --- | --- | --- | --- |"]
        for row in source["rounds"]:
            rate = row.get("success_rate")
            what = f"{row['status']}"
            if rate is not None:
                what += f"，成功率 {rate:.3f}"
            if row.get("error"):
                what += f"：{_one_line(row['error'])}"
            elif row.get("sha256"):
                what += f"（响应 sha256 `{str(row['sha256'])[:12]}…`）"
            body.append(f"| model：控制器 | 第 {row['round']} 轮：{_one_line(row.get('hypothesis'), 120)} | "
                        f"{_one_line(row.get('hypothesis'), 200)} | {what} |")
        notes.append("这里的「决定」与「结果」来自两个分开写的文件，本文档把它们并起来，"
                     "而不是系统在做出决定的那一刻就记下了它们。两者的区别在于："
                     "并起来只能看到**提案说了什么**，看不到当时**为什么选它而不是别的**。")
    elif source.get("scout"):
        # The scout keeps every draft, including the ones verification refused, and the refusals
        # are the informative half: a claim that reading could not settle, corrected once.
        scout = source["scout"]
        sources = ["draft_*.json", "verification.json"]
        # The header says what the table holds and not what it does not. The other branches
        # can print a "why" column because a proposal carries a hypothesis; a scout draft does
        # not, and filling that column with "see the file" would be claiming a reason was
        # recorded when the record holds a question and an answer.
        body += ["| 谁 | 做了什么 | 素材在哪 | 结果 |", "| --- | --- | --- | --- |"]
        verified = scout.get("verification") or {}
        state = str(verified.get("state") or "未验证")
        broken = len(scout["failed_checks"])
        verdict = f"{state}，{broken} 项检查未通过" if broken else state
        for row in scout["drafts"]:
            name = Path(row["path"]).name
            body.append(f"| model：{row['model'] or '—'} | 起草 `{row['stage']}`"
                        f"（第 {row['attempt']} 次，问了 {row['prompt_chars'] or '?'} 字，"
                        f"答了 {row['response_chars'] or '?'} 字） | "
                        f"`{name}` 保存了完整的提问与回答 | 对整份声明的校验结论：**{verdict}** |")
        notes.append("**这次运行没有单独记录「为什么」。** 它记录的是问了什么、答了什么、"
                     "以及校验放行了没有 —— 推理在 `draft_*.json` 的响应正文里，只能读，"
                     "不能被这一节代读。读一个模型在事后替它总结的理由，和读它当时的回答，"
                     "是两件事。")
    else:
        body.append("_没有决策记录。_")
    extra = [row for row in rows if row.get("kind") == "outcome_only"]
    if extra:
        body.append(f"\n另有 {len(extra)} 条结果找不到对应的决定 —— 有调用方关闭了一个它从未记录的决定，"
                    f"这本身是运行的一个发现。")
    graph = mermaid.provenance(rows)
    if graph:
        body += ["", "### 溯源图", "",
                 "_只画记录里已有的边（决定引用了什么、产出了什么）。没有依据的决定就没有入边 —— "
                 "那个空缺是发现，不是需要补上的格式问题。_", "", graph]
    return Section("决策与结果", COMPUTED, sources, "\n".join(body), notes)


def artifact_index(source: dict[str, Any]) -> Section:
    """Every file the run left, what it is for, and what it cannot establish."""
    buckets = list((source.get("survey") or {}).values())
    body: list[str] = []
    if not buckets:
        body.append("_这个目录里没有文件。_")
    else:
        # Grouped by what the file is for, and a group is listed file by file until it stops
        # being worth reading. Collapsing a group to its first entry -- which this did -- hides
        # exactly the files a reader does not already know about: four hundred unclassified
        # artifacts came out as a single row saying there were four hundred of them.
        body += ["| 文件 | 大小 | 它是什么 | 它不能证明什么 |", "| --- | ---: | --- | --- |"]
        for bucket in sorted(buckets, key=lambda one: -one.count):
            # Sorted, because the walk returns files in the order the filesystem holds them and
            # a document that shows a different twelve on each reading cannot be referred to.
            for relative, size in sorted(bucket.examples):
                body.append(f"| `{relative}` | {_bytes(size)} | {bucket.what} | {bucket.caveat} |")
            if bucket.count > len(bucket.examples):
                body.append(f"| _以上是其中 {len(bucket.examples)} 个（按路径排序的取样，"
                            f"不是全部）；同类共 **{bucket.count}** 个、"
                            f"{_megabytes(bucket.bytes)}_ | | {bucket.what} | {bucket.caveat} |")
    for note in source.get("unreadable") or []:
        body.append(f"\n- 读不到：{note}")
    return Section("产物与证据", COMPUTED, ["the run directory", "recipe.json", "plan.json"],
                   "\n".join(body))


#: The role a video file gets before it is classified further by where it sits.
_RECORDING_ROLE = "一段录像"

#: The documents and their own cache, excluded from the index they produce. `RUN.html` is here
#: for the same reason as `RUN.md` and one more: it is generated *from* the survey, so listing
#: it would make each regeneration longer than the last by the size of the previous page.
_IS_THE_DOCUMENT = ("RUN.md", "RUN.html", "summary.json")

#: What a recording is, by where it sits. Ordered by specificity, because the last entry is a
#: catch-all and the specific ones must win. The distinction that matters most is the first two:
#: a recording made while collecting is *training data*, and a recording made while evaluating
#: is *evidence about a policy*. They look identical on disk, and an index that lists them
#: together has told a reader nothing about what either one supports.
_RECORDINGS: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("videos/chunk-", "videos\\chunk-"),
     "采集过程中录下的回合，按相机分目录、按回合分文件 —— 它们是训练数据本身",
     "它们证明了策略当时看到了什么，不证明训练出来的策略学到了什么。"
     "一个成功回合和一千个失败回合在这里长得一样"),
    (("eval_result/",),
     "benchmark 自己的评测器录下的回合",
     "它显示**那一个**回合的过程。一个成功的录像不建立成功率，"
     "而挑选出来的成功录像连过程是否典型也不建立"),
    (("native_probe_data/", "probe_data/"),
     "探针运行的录像，用来确认原生评测链路能跑通",
     "它证明链路能启动并跑完一个回合，不证明任何关于策略的事"),
    (("videos/",),
     "按 `videos/` 目录约定存放的录像",
     "它们由哪一段代码写下的，从路径上看不出来；在弄清之前不要把它们当作某一次评测的证据"),
)


def _recording_kind(relative: str) -> tuple[str, str] | None:
    for needles, what, caveat in _RECORDINGS:
        if any(needle in relative for needle in needles):
            return what, caveat
    return None


def recordings(source: dict[str, Any], listed: int = 8) -> Section:
    """The recordings the run left, grouped by what they are evidence of.

    A number says whether a policy succeeded; an episode says how. More than a thousand of
    these were already on this machine before anything listed them, so finding a demo of a
    particular failure meant knowing which component had written it and where.
    """
    buckets = [bucket for bucket in (source.get("survey") or {}).values()
               if bucket.kind == "recording"]
    body: list[str] = []
    manifest = _load(Path(source["root"]) / "media" / "manifest.json") or {}
    active = [row for row in manifest.get("recordings", []) if isinstance(row, dict) and
              row.get("trigger") and row.get("hash_status") == "checked" and
              row.get("media_identity_status") == "verified_sha256_match"]
    deferred = [row for row in manifest.get("recordings", []) if isinstance(row, dict) and
                row.get("trigger") and row not in active]
    if active or deferred:
        def inline(value: Any) -> str:
            if value is None or value == "":
                return "未记录"
            return (redact(str(value)).replace("`", "'").replace("\n", " ")
                    .replace("\r", " ")[:180])

        def sha(value: Any) -> str:
            text = str(value or "")
            return f"`{text}`" if re.fullmatch(r"[0-9a-f]{64}", text) else "未记录"

        def attempt_link(value: Any) -> str:
            text = str(value or "")
            if not re.fullmatch(r"[0-9a-f]{32}", text):
                return inline(value)
            return f"[`{text}`](<attempts/{text}/receipt.json>)"

        if active:
            body += ["### 主动采集的真实 demo", "",
                     "以下录像由已推导的原生录制阶段执行；下面分别列出录制尝试、产生测量的评测尝试、"
                     "协议/策略/媒体哈希和采集耗时。episode 只有原生回执明确提供时才会填写，"
                     "缺失身份不会被推断。", ""]
            for row in active[:listed]:
                episode = (inline(row.get("episode_id"))
                           if row.get("episode_id") is not None else
                           "未报告（原生录制器未提供 episode ID）")
                seconds = row.get("capture_seconds")
                capture_cost = (f"{float(seconds):.3f}s"
                                if isinstance(seconds, (int, float)) and
                                not isinstance(seconds, bool) else "未记录")
                body += [f"- **{inline(row.get('trigger'))}**；测量："
                         f"{inline(row.get('measurement_label'))}；任务："
                         f"{inline(row.get('task'))}；seed：{inline(row.get('seed'))}；"
                         f"episode：{episode}。",
                         f"  - 录制 attempt：{attempt_link(row.get('attempt_id'))}；评测 attempt："
                         f"{attempt_link(row.get('evaluation_attempt_id'))}；"
                         f"采集耗时：{capture_cost}。",
                         f"  - 评测回执 SHA-256：{sha(row.get('evaluation_receipt_sha256'))}；"
                         f"协议 SHA-256：{sha(row.get('comparison_protocol_sha256'))}；"
                         f"策略 SHA-256：{sha(row.get('policy_sha256'))}；"
                         f"媒体 SHA-256：{sha(row.get('sha256'))}。",
                         f"  - [打开真实运行录像](<{row['path']}>)。"]
            body.append("")
        if deferred:
            body += ["### 主动媒体待验证（未纳入已验证 demo）", "",
                     "原生阶段报告了这些媒体，但文件哈希被大小上限延后或当前字节与回执不一致；"
                     "这里只保留待核对入口，不将其作为已验证策略/episode 证据。", ""]
            for row in deferred[:listed]:
                body.append(
                    f"- **{inline(row.get('trigger'))}**；文件：`{inline(row.get('path'))}`；"
                    f"任务：{inline(row.get('task'))}；seed：{inline(row.get('seed'))}；"
                    f"哈希状态：`{inline(row.get('hash_status'))}` / "
                    f"`{inline(row.get('media_identity_status'))}`；来源 attempt："
                    f"{attempt_link(row.get('reported_by_attempt_id'))}；"
                    f"[查看媒体](<{row.get('path')}>).")
            body.append("")
    if not buckets:
        body.append("_这次运行没有留下录像。_\n\n"
                    "可能是没有录，也可能是录了而这份文档没有找到 —— 这两件事在这里分不开，"
                    "而它们不是一回事。下一步需调查原生评测器是否支持录制；不能假设一个通用开关。")
    else:
        total = sum(bucket.count for bucket in buckets)
        body += [f"共 **{total}** 个录像文件。**先说清楚它们不是一类东西**：采集时录下的是训练数据，"
                 "评测时录下的是关于策略的证据，而这两者证明的不是一回事。", ""]
        for bucket in sorted(buckets, key=lambda one: -one.count):
            body += [f"### {bucket.what}", "",
                     f"**{bucket.count}** 个，共 {_megabytes(bucket.bytes)}。", ""]
            for path, _ in bucket.examples[:listed]:
                if Path(path).suffix.lower() == ".gif":
                    body.append(f"![真实运行录制：{Path(path).name}](<{path}>)")
                else:
                    body.append(f"- [打开真实运行录像](<{path}>) — `{path}`")
            if bucket.count > len(bucket.examples[:listed]):
                body.append(f"- ……其余 {bucket.count - len(bucket.examples[:listed])} 个同类，"
                            f"未逐个列出")
            body += ["", f"> **它不能证明什么**：{bucket.caveat}", ""]
    trajectories = [path for bucket in (source.get("survey") or {}).values()
                    for path, _ in bucket.examples if path.endswith("trajectory.svg")]
    if trajectories:
        body += ["### 轨迹图（不是场景回放）", "",
                 f"![从真实遥测生成的轨迹](<{trajectories[0]}>)", "",
                 "这张图仅表示记录下来的位置变化，不代表成功率或视觉仿真画面。"]
    return Section("录像与 demo", COMPUTED, ["the run directory"], "\n".join(body))


def _megabytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}GB"


def _bytes(size: int) -> str:
    """One file's size, exactly. A file's size is knowable, so it is stated and not rounded --
    rounding belongs on the group total, where it saves a reader from counting digits."""
    return f"{size}B"


def trajectory_section(root: Path, source: dict[str, Any]) -> Section:
    """What the recorded poses say, in numbers, and where the drawings are.

    Markdown cannot carry a drawing, so this section is the index and the caption and the
    `RUN.html` beside it holds the picture. Saying which one was drawn, and how many were not,
    is the difference between a document that points at its evidence and one that gestures.
    """
    from . import trajectory
    directories = trajectory.directories_under(root)
    lines: list[str] = []
    if not directories:
        lines.append("_这次运行没有留下逐步位姿记录。_\n\n"
                     "评测会在 `telemetry.jsonl` 里逐步记下每个物体的位姿，"
                     "而这次运行里没有这样的文件 —— 所以没有任何东西可以从姿态上回看。")
        return Section("轨迹", COMPUTED, ["telemetry.jsonl"], "\n".join(lines))
    described: list[tuple[Path, dict[str, Any]]] = []
    for directory in directories:
        detail = trajectory.readings(directory)
        if detail:
            described.append((directory, detail))
    if not described:
        lines.append(f"有 {len(directories)} 个 `telemetry.jsonl`，但都没有可读的行。")
        return Section("轨迹", COMPUTED, ["telemetry.jsonl"], "\n".join(lines))
    total_rows = sum(detail["rows"] for _, detail in described)
    lines += [f"**{len(described)} 次评测记下了逐步位姿，共 {total_rows} 行。**", "",
              "| 评测 | 局数 | 记录的物体 | 位姿行数 | 能关联到每局结果的 |",
              "| --- | ---: | --- | ---: | ---: |"]
    for directory, detail in described:
        try:
            name = str(directory.relative_to(root))
        except ValueError:
            name = str(directory)
        joined = detail["joinable_outcomes"]
        lines.append(f"| `{name}` | {detail['episodes']} | "
                     f"{'、'.join(detail['entities'])} | {detail['rows']} | "
                     f"{joined}/{detail['episodes']}"
                     f"{'（这是**画得出来**的那种：每条路径按它那一局的成败着色）' if joined else '（没有每局结果可关联，只能画成同一种颜色）'} |")
    drawn = next(((directory, detail) for directory, detail in
                  sorted(described, key=lambda one: -one[1]["rows"])
                  if detail.get("by_outcome")), None)
    if drawn is not None:
        _, detail = drawn
        measured = detail["by_outcome"]
        primary = measured["primary"]
        lines += ["", f"### 按成败分开看（{primary} 是动得最多的那个物体）", ""]
        rows = []
        for side, label in (("success", "成功的局"), ("failure", "失败的局")):
            if f"{side}_median_travel" in measured:
                rows.append((label, measured[f"{side}_episodes"],
                             measured[f"{side}_median_travel"],
                             measured.get(f"{side}_median_final_separation")))
        if rows:
            lines += ["| | 局数 | 路程中位数 | 最后两者距离中位数 |", "| --- | ---: | ---: | ---: |"]
            for label, count, travel, apart in rows:
                lines.append(f"| {label} | {count} | {travel} | "
                             f"{'—' if apart is None else apart} |")
            if len(rows) == 2 and rows[1][2]:
                ratio = rows[0][2] / rows[1][2]
                lines.append(f"\n成功的局里，{primary} 走的路程是失败局的 **{ratio:.2f} 倍**。")
            lines.append("\n这是从位姿记录里算出来的，不是从成功率里推出来的 —— "
                         "两个数字来自同一批种子，所以可以直接比。")
    lines += ["", f"**图在 `RUN.html` 里**（markdown 放不下图）。画的是记录里位姿跨度最大的两个轴，"
                  f"起始位置画成空心圈，每条路径按它那一局的成败着色。", "",
              "> **这是轨迹，不是回放。** 位姿被投影到平面上画成路径，没有任何东西被渲染出来，"
              "也没有相机被模拟。它显示东西**在哪里**，不显示**看起来什么样**，"
              "更不显示策略当时**看到了什么**。当成渲染图来读是这张图会主动诱导的错误，"
              "所以这里把话说白。"]
    return Section("轨迹", COMPUTED, ["telemetry.jsonl", "evaluation_metrics.json"],
                   "\n".join(lines))


def unresolved(source: dict[str, Any]) -> Section:
    """What the record says is not settled. The section that makes the rest accountable."""
    body: list[str] = []
    observations = [row for row in (source.get("observations") or {}).get("rows", [])
                    if isinstance(row, dict)]
    for row in observations:
        body.append(f"- **[{row.get('kind', '?')}] {row.get('subject', '?')}** —— "
                    f"{row.get('detail', '')}")
    open_decisions = [row for row in (source.get("decisions") or {}).get("rows", [])
                      if isinstance(row, dict) and row.get("kind") == "decision"
                      and (row.get("outcome") or {}).get("state") == "open"]
    for row in open_decisions:
        body.append(f"- **决定没有结果** —— {_one_line(row.get('activity'))}")
    for measured in source.get("measurements") or []:
        record = measured["record"]
        if record.get("success_rate") is None:
            because = f"：{record['why']}" if record.get("why") else ""
            body.append(f"- **{record.get('label', '?')} 没有产生数字** —— "
                        f"{record.get('where', '?')} 阶段没有跑完{because}")
    for row in (source.get("report") or {}).get("rounds", []) if source.get("report") else []:
        if isinstance(row, dict) and row.get("status") == "nothing was measured":
            body.append(f"- **第 {row.get('round', '?')} 轮什么也没测到** —— "
                        f"{row.get('why_not', '')}")
    for row in source.get("rounds") or []:
        if row.get("status") == "completed":
            continue
        body.append(f"- **第 {row['round']} 轮没有完成**（{row['status']}）—— "
                    f"{_one_line(row.get('error') or '结果文件里没有写原因')}")
    scout = source.get("scout")
    if scout:
        onboarding = scout.get("onboarding") or {}
        for fault in (onboarding.get("structural_faults") or []) if isinstance(
                onboarding.get("structural_faults"), list) else []:
            body.append(f"- **结构缺陷** —— {_one_line(fault, 300)}")
        if isinstance(onboarding.get("readiness"), dict):
            for key, value in sorted(onboarding["readiness"].items()):
                if value not in (None, "", [], {}, True):
                    body.append(f"- **未就绪：{key}** —— {_one_line(value, 200)}")
        if isinstance(onboarding.get("structural_faults"), str) and onboarding["structural_faults"]:
            body.append(f"- **结构缺陷** —— {_one_line(onboarding['structural_faults'], 300)}")
        for check in scout["failed_checks"]:
            body.append(f"- **校验未通过：`{check.get('check')}` 于 "
                        f"`{check.get('subject')}`**"
                        f"{' —— ' + _one_line(check.get('why'), 200) if check.get('why') else ''}")
        verified = scout.get("verification") or {}
        for failure in (verified.get("identity_failures") or [])[:8]:
            body.append(f"- **身份对不上** —— {_one_line(failure, 240)}")
        if not scout.get("drafts"):
            body.append("- **这次运行没有留下任何草稿** —— 所以没有可查的推理")
    selection = source.get("selection")
    if isinstance(selection, dict) and selection.get("reading"):
        # The things the run did not settle, as the selection record knows them. These replace
        # three branches on fields nothing writes -- a rule about final qualification that was
        # printed in this section and had never been applied to any run.
        failed = [arm for arm in (selection.get("arms") or []) if not arm.get("scored")]
        if failed:
            body.append(f"- **{len(failed)} 条手臂没有产生数字** —— "
                        + "、".join(str(arm.get("label")) for arm in failed[:12])
                        + "。它们跑了、花了时间，只是没有数字出来；这次运行的最佳值是"
                          "其余手臂里的最大值，不是全部候选里的。")
        if selection.get("best_is_the_maximum_of", 0) > 1:
            body.append(f"- **最佳值是从 {selection['best_is_the_maximum_of']} 个点里挑的最大值**"
                        " —— 挑选本身会把噪声里最高的那个留下来，所以这个数字高于"
                        "「候选真的更好」能解释的部分。")
        comparison = selection.get("comparison") or {}
        if comparison.get("verdict") == "no_difference":
            body.append(f"- **与基线的差别没有被确立** —— {comparison.get('why')}。"
                        "这不是「没有提升」，是这次测量分不出来。")
        elif comparison.get("verdict") == "not_established" and (
                selection.get("best") or {}).get("label") != "baseline" and selection.get("baseline"):
            body.append(f"- **与基线的差别没有被确立** —— {comparison.get('why')}。"
                        "这不是「没有提升」，而是缺少可核验的比较证据。")
        elif comparison.get("verdict") == "better":
            body.append(f"- 与基线相比：{comparison.get('why')} —— "
                        "这是这一次比较的结论，不是对测试集的结论。")
        if not selection.get("one_protocol") and selection.get("best_is_the_maximum_of", 0) > 1:
            body += ["- **这些手臂不都在同一个冻结协议下测过** —— 所以它们之间不可比，"
                     "上面的计数是取了多少个点，不是多少次可比的抽样。"]
    if not body:
        body.append("_记录里没有未解决的事。这不是说没有 —— 是说记录里没有。_")
    return Section("未解决", COMPUTED, ["observations.json", "decisions.json",
                                        "measurements/*.json", "research_report.json"],
                   "\n".join(body))


def reproduce(source: dict[str, Any]) -> Section:
    """What someone would need to run this again, taken from the records and not remembered."""
    body: list[str] = []
    if source.get("recipe_path"):
        recipe = source.get("recipe") or {}
        created = (recipe.get("created_at") or recipe.get("at") or "") if isinstance(recipe, dict) else ""
        body.append(f"- 环境配方：`{source['recipe_path']}`"
                    f"{f'（{created}）' if created else ''}")
    if source.get("plan_path"):
        body.append(f"- 计划：`{source['plan_path']}`")
    if source.get("derived_stages_path"):
        body.append(f"- 各阶段命令的来由：`{source['derived_stages_path']}`")
    for measured in source.get("measurements") or []:
        record = measured["record"]
        argv = record.get("argv") or []
        if not argv:
            continue
        on = f" —— {record['device']}" if record.get("device") else ""
        body += ["", f"**{record.get('label', '?')}**{on}", ""]
        if record.get("device_why"):
            body.append(f"设备是这样选的：{record['device_why']}\n")
        body += ["```", " ".join(str(one) for one in argv), "```"]
        settings = record.get("settings") or {}
        if settings:
            body.append(f"设置：`{json.dumps(settings, ensure_ascii=False, sort_keys=True)}`")
    for row in source.get("processes") or []:
        body += ["", f"**{row['label']}**", ""]
        if row.get("cwd"):
            body.append(f"工作目录：`{row['cwd']}`\n")
        body += ["```", " ".join(str(one) for one in row["command"]), "```",
                 f"（{row.get('status') or '?'}，返回码 {row.get('returncode')}，"
                 f"{row.get('seconds')}s，记录在 `{row['path']}`）"]
    if not body:
        body.append("_这份文档在这些记录里没有找到可复现的命令。_\n\n"
                    "命令可能记在本文档没有读的地方，也可能是这个运行根本没有记 —— "
                    "这两种情况在这里分不开，而它们不是一回事。")
    body.append("")
    body.append("以上是记录下来的东西。**它没有说这些命令在别处也能跑通** —— 那是另一回事，"
                "而把它当成同一回事正是这类文档最容易犯的错。")
    return Section("复现方法", COMPUTED,
                   ["recipe.json", "plan.json", "derived_stages.json", "measurements/*.json"],
                   "\n".join(body))


def _material(source: dict[str, Any]) -> str:
    """The computed sections as text, which is what the model is shown when it writes the
    summary. Everything the summary may state a number about has to be in here, because the
    check afterwards looks for those numbers in the record and a number the model could not
    have read is a number it invented."""
    return "\n\n".join(f"## {section.title}\n{section.body}"
                       for section in (timeline(source), numbers(source),
                                       decision_section(source), unresolved(source)))


# -- the written section, and the check on it ------------------------------------------------
#
# The summary is a model's prose about the records, and it is the only part of this document
# that can be wrong. The check on its numbers, and what that check does and does not establish,
# live in `claims.py`; this assembles it into a section. It sits above the computed ones in the
# document, and apart from them here, for the same reason.

def written_section(source: dict[str, Any], summary: str) -> Section:
    """The summary, with the check on its numbers printed underneath it."""
    check = check_claims(summary, known_numbers(source,
                                                and_what_was_shown=_material(source)))
    notes = ["本节由模型写出，不是从记录算出来的。"]
    if check["claims"]:
        notes.append(f"其中 {check['traced']}/{check['claims']} 个数字能在记录里找到出处。")
    if check["untraced"]:
        notes.append("**在记录里找不到出处的数字**：" +
                     "、".join(f"`{one}`" for one in check["untraced"]) +
                     " —— 它们可能算错了、可能来自记录之外，读者不应把它们当作依据。")
    notes.append("查得到出处，只说明这个数字出现在记录里，不说明它被用对了。"
                 "一个用真数字搭起来的错误结论，会通过上面这项检查。")
    return Section("摘要", WRITTEN, ["decisions.json", "exchanges/*.json",
                                     "measurements/*.json"], summary, notes)


#: In the order a reader wants them: what happened, what was decided and why, what the numbers
#: were, what can be watched, what is here to check, what is not settled, and how to run it
#: again. The written summary goes above all of them. `trajectory_section` takes the run root
#: as well as the gathered source, so it is wrapped rather than listed directly.
def objective_section(source: dict[str, Any]) -> Section:
    """How far the run got, and what it was allowed to do about it.

    Placed before the numbers and not after them, because on a run that never produced a
    number it is the only section with anything in it. A reader arriving at a report whose
    every number is `null` has, until now, been told nothing but that -- when the run's own
    records hold which rung it reached, which ideas it considered, and which of them were
    refused before they ran.
    """
    rubric = source.get("rubric")
    ideas = source.get("ideas")
    audits = source.get("audits")
    if not any(isinstance(one, dict) and one for one in (rubric, ideas, audits)):
        return Section("做到哪一步", COMPUTED, [], "_这次运行没有留下评分表、想法库或审计记录。_")
    body: list[str] = []
    sources = ["rubric.json", "ideas.json", "audits.json"]
    if isinstance(rubric, dict) and rubric.get("checks"):
        body += [f"**完成度 {float(rubric.get('score') or 0.0):.0%}**", ""]
        body += ["| 问题 | 权重 | 状态 | 依据 |", "| --- | ---: | --- | --- |"]
        for one in rubric["checks"]:
            body.append(f"| **{_one_line(one.get('question'), 80)}** | "
                        f"{float(one.get('weight') or 0):.0f} | {one.get('state')} | "
                        f"{_one_line(one.get('because'), 100)} |")
            for two in one.get("children") or []:
                # The mark says who answered. A question the run's own code read off a record
                # and one the run was asked are different kinds of evidence, and the value
                # alone does not tell a reader which they are holding.
                mark = "" if not two.get("by") else " *(模型作答)*"
                body.append(f"| └ {_one_line(two.get('question'), 80)}{mark} | "
                            f"{float(two.get('weight') or 0):.0f} | {two.get('state')} | "
                            f"{_one_line(two.get('because'), 100)} |")
        body += ["", "「not reached」与「no」不是一回事：前者是还没走到，"
                     "后者是走到了、没通过。", ""]
    if isinstance(ideas, dict) and isinstance(ideas.get("ideas"), list) and ideas["ideas"]:
        # Rendered by the library's own function rather than here, so the document and
        # `ideas.py` cannot come to disagree about what a library looks like -- and so the
        # ideas that were never chosen are in it, which is the half a reader cannot guess.
        try:
            from . import ideas as idea_library
            body += [idea_library.render(
                idea_library.IdeaLibrary(Path(source["root"]) / "ideas.json")), ""]
        except Exception as exc:                                 # noqa: BLE001
            body += [f"_想法库在，但渲染不出来：{type(exc).__name__}_", ""]
    if isinstance(audits, dict) and audits.get("audits"):
        rows = audits["audits"]
        body += ["| 想法 | 判定 | 红线 | 会改的文件 | 理由 |", "| --- | --- | --- | --- | --- |"]
        for one in rows:
            body.append(f"| {_one_line(one.get('idea'), 60)} | {one.get('verdict')} | "
                        f"{one.get('line') or '—'} | {_one_line(one.get('file'), 40) or '—'} | "
                        f"{_one_line(one.get('because'), 120)} |")
        refused = sum(1 for one in rows if one.get("verdict") != "cleared")
        body += ["", f"其中 {refused} 条在跑之前就被挡下 —— 它们没有跑过，"
                     f"所以它们的失败不计在这次运行头上。", ""]
    return Section("做到哪一步", COMPUTED, sources, "\n".join(body).rstrip())


COMPUTED_SECTIONS = (timeline, objective_section, decision_section, numbers, recordings,
                     lambda source: trajectory_section(Path(source["root"]), source),
                     artifact_index, unresolved, reproduce)


# -- rendering -----------------------------------------------------------------------------

def _one_line(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _flatten(value: Any, limit: int = 200) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        return "; ".join(f"{key}={_one_line(one, 60)}" for key, one in sorted(value.items())
                         if one not in (None, "", [], {}))[:limit] or "—"
    return _one_line(value, limit)


_LABEL = {COMPUTED: "算出来的", WRITTEN: "写出来的"}
_MARK = {COMPUTED: "`computed`", WRITTEN: "`written`"}


def render(source: dict[str, Any], sections: Iterable[Section], *, title: str = "") -> str:
    """The document. Every section carries what it is and where it came from."""
    lines = [f"# {title or '运行记录'}", ""]
    lines.append(f"运行目录：`{source.get('root', '')}`")
    unreadable = source.get("unreadable") or []
    lines.append(f"这份文档从运行自己的记录装配而成；有 {len(unreadable)} 项它读不到，"
                 f"逐条列在「产物与证据」一节。" if unreadable else "记录齐全。")
    lines += ["", "> 本文件由 `run_record.py` 生成，可重复生成。每一节都标明它是**算出来的**"
                  "（来自运行自己的记录，不会与记录矛盾）还是**写出来的**（模型写的，"
                  "可能错）。写出来的那节里，每个数字都在记录里找过出处。", "", "---", ""]
    for section in sections:
        sources = "、".join(f"`{one}`" for one in section.sources)
        lines += [f"## {section.title}", "",
                  f"**{_LABEL[section.provenance]}** {_MARK[section.provenance]}"
                  f"{' · 来源：' + sources if sources else ''}", "", section.body, ""]
        for note in section.notes:
            lines.append(f"> {note}")
        lines += ["", "---", ""]
    return "\n".join(lines).rstrip() + "\n"


# -- the entry point -----------------------------------------------------------------------

def generate(root: Path, *, client: Any = None, summary: str | None = None,
             refresh_summary: bool = False, title: str = "") -> Path:
    """Build `RUN.md` from the run's records. Never raises; says what it could not do.

    The written half is cached in `summary.json` and reused, so regenerating the document --
    which happens on every stage -- does not make a model call each time, and so the document
    does not quietly change between two readings of the same run. A new summary is asked for
    when the caller passes a client and either there is none cached or `refresh_summary` is
    set, which is what a stage boundary does.
    """
    root = Path(root)
    destination = root / "RUN.md"
    try:
        source = gather(root)
        try:
            from . import media_manifest
            media_manifest.write(root, source.get("survey") or {})
        except Exception as exc:                                   # noqa: BLE001
            atomic_json(root / "media_manifest_error.json", {
                "at": now(), "error": f"{type(exc).__name__}: {exc}"})
        cached = _load(root / "summary.json") or {}
        text = summary if summary is not None else str(cached.get("text") or "")
        record: dict[str, Any] = {"text": text, "at": cached.get("at"),
                                  "model": cached.get("model"), "checked": None}
        if client is not None and (refresh_summary or not text):
            try:
                record = {"text": write_summary(_material(source), client), "at": now(),
                          "model": str(getattr(client, "model", "") or ""), "checked": None}
                if summary is not None:            # an explicit summary wins over a fresh one
                    record["text"] = summary
            except Exception as exc:               # noqa: BLE001
                record["why_not"] = f"{type(exc).__name__}: {exc}"[:400]
        if summary is not None:
            record["text"] = summary
        sections: list[Section] = []
        if record["text"]:
            sections.append(written_section(source, record["text"]))
            record["checked"] = check_claims(
                record["text"],
                known_numbers(source, and_what_was_shown=_material(source)))
        else:
            sections.append(Section(
                "摘要", WRITTEN, [], "_这一节没有被写出来。_",
                [f"原因：{record.get('why_not')}" if record.get("why_not") else
                 "没有可用的模型客户端，所以写出来的这一节不存在。"
                 "这不等于没有值得说的东西 —— 它等于没人说过。"]))
        sections += [builder(source) for builder in COMPUTED_SECTIONS]
        atomic_json(root / "summary.json", record)
        try:
            from . import report_page
            report_page.write(root)
        except Exception:                                        # noqa: BLE001
            pass
        publish_document(destination, render(source, sections, title=title))
        # A prepared run has one top-level human entry point. Keep its research card in step
        # with child measurements and media while retaining the child page as full detail.
        if root.parent.name == "research":
            run_root = root.parent.parent
            if (run_root / "RUN.md").is_file():
                refresh_research_projection(run_root, root.name)
    except Exception as exc:                       # noqa: BLE001
        # A document generator that dies is worse than one that writes a short document: it
        # dies exactly when the run has gone wrong and the document is most wanted.
        try:
            atomic_json(root / "report_renderer_error.json", {
                "at": now(), "renderer": "markdown", "error": f"{type(exc).__name__}: {exc}"})
            if not destination.is_file():
                atomic_text(destination,
                    f"# {title or '运行记录'}\n\n这份记录没能生成：{type(exc).__name__}: {exc}\n\n"
                    f"运行目录是 `{root}`；记录本身还在那里。\n")
        except OSError:
            pass
    return destination


_LIVE_START = "<!-- AUTOSIM_LIVE_START -->"
_LIVE_END = "<!-- AUTOSIM_LIVE_END -->"


@contextmanager
def _report_lock(destination: Path):
    """Serialize whole-document and heartbeat writes to one Markdown file."""
    lock = destination.with_name(destination.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def publish_document(destination: Path, content: str) -> Path:
    """Atomically publish a full report under the same lock as its live strip."""
    destination = Path(destination)
    with _report_lock(destination):
        # A background Recorder can finish between the main publisher's render and lock.
        # Rehydrate its newest fact snapshot/narrative inside the publication transaction.
        from .scheduling import policy
        if policy(destination.parent).get("async_recorder"):
            from . import recorder
            if recorder.START in content and recorder.END in content:
                snapshot = recorder.read(destination.parent, "report/presentation.json")
                if snapshot:
                    narrative = recorder.read(destination.parent, "report/narrative.json")
                    before, rest = content.split(recorder.START, 1)
                    _, after = rest.split(recorder.END, 1)
                    content = before + recorder.render(destination.parent, snapshot, narrative) + after
        atomic_text(destination, content)
        _publish_presentation_html(destination, content)
    return destination


def _publish_presentation_html(destination: Path, content: str) -> None:
    if "<!-- AUTOSIM_PRESENTATION_START -->" not in content:
        return
    try:
        from .report_page import write_live_page
        write_live_page(destination.with_suffix(".html"), content)
    except Exception as exc:
        atomic_json(destination.parent / "report" / "html_error.json", {
            "at": now(), "error": type(exc).__name__})


def _safe_cell(value: Any, *, limit: int = 180) -> str:
    return " ".join(str(value if value is not None else "—").split())[:limit].replace("|", "\\|")


def _markdown_text(value: Any, *, limit: int = 180) -> str:
    """Escape untrusted telemetry labels before they enter the local Markdown report."""
    import html
    escaped = html.escape(_safe_cell(value, limit=limit), quote=False)
    for marker in ("\\", "`", "*", "_", "[", "]", "(", ")", "!", "#"):
        escaped = escaped.replace(marker, "\\" + marker)
    return escaped


def build_report_view(root: Path, run_id: str, *, status: str = "",
                      current_action: str = "") -> dict[str, Any]:
    """Build the deterministic local view and record the revisions it projects."""
    root = Path(root)
    research = root / "research" / run_id
    measurements: list[dict[str, Any]] = []
    receipt_paths: list[Path] = []
    measurement_paths = sorted((research / "measurements").glob("*.json")) \
        if (research / "measurements").is_dir() else []
    for path in measurement_paths:
        record = _load(path)
        if not isinstance(record, dict):
            continue
        metric = record.get("metric_value")
        valid_value = (record.get("ok") is True and isinstance(metric, (int, float)) and
                       not isinstance(metric, bool) and math.isfinite(metric))
        reading = record.get("metric_reading") or {}
        evaluation = record.get("evaluate") or {}
        metric_contract = record.get("metric") or {}
        # The presentation consumes recorded facts, not numbers invented by narration.
        from . import recorder
        stages_cost = {}
        protocol_hash = None
        for stage in ("collect", "prepare_data", "train", "evaluate"):
            stage_row = record.get(stage) or {}
            attempt_id = stage_row.get("attempt_id") if isinstance(stage_row, dict) else None
            if isinstance(attempt_id, str) and re.fullmatch(r"[0-9a-f]{32}", attempt_id):
                receipt = recorder.read(root, f"research/{run_id}/attempts/{attempt_id}/receipt.json")
                receipt_paths.append(root / f"research/{run_id}/attempts/{attempt_id}/receipt.json")
                if receipt.get("attempt_id") == attempt_id:
                    if recorder.finite(receipt.get("seconds")):
                        stages_cost[stage] = receipt["seconds"]
                    if stage == "evaluate":
                        claimed = receipt.get("comparison_protocol_sha256")
                        if isinstance(claimed, str) and re.fullmatch(r"[0-9a-f]{64}", claimed):
                            protocol_hash = claimed
        dataset = record.get("dataset") or {}
        dataset_summary = {key: dataset[key] for key in ("samples", "episodes", "version", "sha256")
                           if isinstance(dataset, dict) and key in dataset}
        measurements.append({
            "label": str(record.get("label") or path.stem),
            "status": str(record.get("status") or
                          ("measured" if record.get("ok") is True else "unscored")),
            "metric_name": str(metric_contract.get("name") or record.get("metric_name") or ""),
            "metric_value": float(metric) if valid_value else None,
            "samples": reading.get("episodes_completed", reading.get("samples")) if isinstance(reading, dict) else None,
            "attempt_id": (evaluation.get("attempt_id")
                           if isinstance(evaluation, dict) else None),
            "measurement_ref": f"research/{run_id}/measurements/{path.name}",
            "source_sha256": digest(path),
            "confirmation": record.get("confirmation") is True,
            "metric_direction": metric_contract.get("direction"),
            "metric_unit": metric_contract.get("unit"),
            "protocol_sha256": protocol_hash,
            "changes": record.get("varied") or {},
            "data_summary": dataset_summary,
            "stage_seconds": stages_cost,
            "disposition": record.get("verdict") or record.get("why_not"),
        })

    telemetry_attempts: list[dict[str, Any]] = []
    telemetry_files: list[Path] = []
    telemetry_chart_files: list[Path] = []
    telemetry_dir = root / "report" / "telemetry"
    attempts_dir = research / "attempts"
    if telemetry_dir.is_dir() and not telemetry_dir.is_symlink():
        for path in sorted(telemetry_dir.glob("*.json")):
            attempt_id = path.stem
            if not re.fullmatch(r"[0-9a-f]{32}", attempt_id) or path.is_symlink():
                continue
            attempt_dir = attempts_dir / attempt_id
            receipt_path = attempt_dir / "receipt.json"
            if (attempt_dir.is_symlink() or receipt_path.is_symlink() or
                    not receipt_path.is_file()):
                continue
            telemetry = _load(path)
            receipt = _load(receipt_path)
            if (not isinstance(telemetry, dict) or not isinstance(receipt, dict) or
                    telemetry.get("attempt_id") != attempt_id or
                    str(receipt.get("stage") or receipt.get("node_id") or "") != "train"):
                continue
            chart_path = telemetry_dir / f"{attempt_id}.svg"
            chart_ref = (f"report/telemetry/{attempt_id}.svg"
                         if chart_path.is_file() and not chart_path.is_symlink() else None)
            observer_status = str(telemetry.get("status") or "unknown")
            stage_status = str(receipt.get("status") or "unknown")
            receipt_is_terminal = stage_status not in {"running", "unknown"}
            stale_observer = receipt_is_terminal and observer_status != stage_status
            effective_status = stage_status if receipt_is_terminal else observer_status
            metrics = []
            for series in telemetry.get("series") or []:
                if not isinstance(series, dict):
                    continue
                samples = series.get("samples") or []
                last_sample = samples[-1] if samples and isinstance(samples[-1], dict) else {}
                metrics.append({
                    "name": str(series.get("metric_name") or "unknown"),
                    "unit": series.get("value_unit"),
                    "samples": int(series.get("samples_seen") or len(samples)),
                    "last_step": last_sample.get("native_step"),
                    "verification": last_sample.get("verification"),
                })
            metrics = metrics[:12]
            telemetry_attempts.append({
                "attempt_id": attempt_id,
                "stage_status": stage_status,
                "status": effective_status,
                "stale_observer_snapshot": stale_observer,
                "sample_count": int(telemetry.get("sample_count") or 0),
                "updated_at": telemetry.get("updated_at"),
                "metrics": metrics,
                "chart_ref": chart_ref,
                "data_ref": f"report/telemetry/{attempt_id}.json",
                "receipt_ref": f"research/{run_id}/attempts/{attempt_id}/receipt.json",
                "errors": [str(code)[:100] for code in (telemetry.get("errors") or [])[:8]],
                "source_aliases": [str(alias)[:100]
                                   for alias in (telemetry.get("source_aliases") or [])[:16]],
            })
            telemetry_files.append(path)
            if chart_ref:
                telemetry_chart_files.append(chart_path)
            telemetry_files.append(receipt_path)
    known_telemetry_attempts = {row["attempt_id"] for row in telemetry_attempts}
    if attempts_dir.is_dir() and not attempts_dir.is_symlink():
        for receipt_path in sorted(attempts_dir.glob("*/receipt.json")):
            attempt_id = receipt_path.parent.name
            if (attempt_id in known_telemetry_attempts or
                    not re.fullmatch(r"[0-9a-f]{32}", attempt_id) or
                    receipt_path.parent.is_symlink() or receipt_path.is_symlink()):
                continue
            receipt = _load(receipt_path)
            summary = receipt.get("telemetry") if isinstance(receipt, dict) else None
            if (not isinstance(receipt, dict) or
                    str(receipt.get("stage") or receipt.get("node_id") or "") != "train" or
                    not isinstance(summary, dict)):
                continue
            stage_status = str(receipt.get("status") or "unknown")
            observer_status = str(summary.get("status") or "unknown")
            receipt_is_terminal = stage_status not in {"running", "unknown"}
            stale_observer = receipt_is_terminal and observer_status != stage_status

            def safe_report_ref(value: Any) -> str | None:
                relative = Path(str(value or ""))
                if not value or relative.is_absolute() or ".." in relative.parts:
                    return None
                candidate = root / relative
                try:
                    return (relative.as_posix() if candidate.is_file() and
                            not candidate.is_symlink() and
                            candidate.resolve().is_relative_to(root.resolve()) else None)
                except OSError:
                    return None

            chart_ref = safe_report_ref(summary.get("chart_ref"))
            data_ref = safe_report_ref(summary.get("data_ref"))
            telemetry_attempts.append({
                "attempt_id": attempt_id,
                "stage_status": stage_status,
                "status": stage_status if receipt_is_terminal else observer_status,
                "stale_observer_snapshot": stale_observer,
                "sample_count": int(summary.get("sample_count") or 0),
                "updated_at": receipt.get("finished_at") or receipt.get("started_at"),
                "metrics": [{"name": str(name)[:120], "unit": None,
                             "samples": int(summary.get("sample_count") or 0),
                             "last_step": None, "verification": "native_log_observation"}
                            for name in (summary.get("metrics") or [])[:12]],
                "chart_ref": chart_ref,
                "data_ref": data_ref,
                "receipt_ref": f"research/{run_id}/attempts/{attempt_id}/receipt.json",
                "errors": [str(code)[:100] for code in (summary.get("errors") or [])[:8]],
                "source_aliases": [],
            })
            telemetry_files.append(receipt_path)
            if chart_ref:
                telemetry_chart_files.append(root / chart_ref)
    telemetry_attempts.sort(key=lambda row: (str(row.get("updated_at") or ""),
                                              row["attempt_id"]))

    manifest = _load(research / "media" / "manifest.json") or {}
    recordings = manifest.get("recordings") or []
    verified_media = []
    pending_media = 0
    for row in recordings:
        if not isinstance(row, dict) or not row.get("path"):
            continue
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            pending_media += 1
            continue
        is_verified = (row.get("media_identity_status") == "verified_sha256_match" and
                       row.get("hash_status") == "checked")
        if not is_verified:
            pending_media += 1
            continue
        preview_ref = None
        preview = row.get("preview") or {}
        if (isinstance(preview, dict) and preview.get("status") == "decoded_frame" and
                preview.get("parent_sha256") == row.get("sha256")):
            preview_relative = Path(str(preview.get("path") or ""))
            if (not preview_relative.is_absolute() and ".." not in preview_relative.parts and
                    preview_relative.parts[:2] == ("media", "previews")):
                preview_path = research / preview_relative
                try:
                    if (preview_path.is_file() and not preview_path.is_symlink() and
                            preview_path.resolve().is_relative_to(research.resolve()) and
                            preview_path.stat().st_size <= 2 * 1024**2 and
                            digest(preview_path) == preview.get("sha256")):
                        preview_ref = f"research/{run_id}/{preview_relative.as_posix()}"
                except OSError:
                    pass
        verified_media.append({
            "path": f"research/{run_id}/{relative.as_posix()}",
            "preview_ref": preview_ref,
            "trigger": row.get("trigger") or row.get("classification") or "recording",
            "task": row.get("task"), "seed": row.get("seed"),
            "episode_id": row.get("episode_id"),
            "measurement_label": row.get("measurement_label"),
            "media_identity_status": row.get("media_identity_status"),
            "attempt_id": row.get("attempt_id"),
        })

    report = _load(research / "research_report.json") or {}
    session = _load(research / "controller_session.json") or {}
    budget = _load(root / "budget.json") or {}
    live = _load(research / "running.json") or {}
    phase_state = {}
    state = _load(root / "run_state.json") or {}
    if isinstance(state.get("phases"), dict):
        phase_state = state["phases"]
    observed_action = str(live.get("stage") or "")
    if not observed_action and isinstance(state.get("current_action"), dict):
        observed_action = str(state["current_action"].get("step") or "")
    if not observed_action:
        research_phase = phase_state.get("research", {})
        preparation_phase = phase_state.get("preparation", {})
        action = (research_phase.get("current_action") or
                  preparation_phase.get("current_action") or {})
        if isinstance(action, dict):
            observed_action = str(action.get("step") or "")
    resolved_status = (status or str(live.get("status") or state.get("status") or
                                    report.get("run_status") or session.get("status") or
                                    "unknown"))
    # A terminal owner has reconciled the action. A leftover child running.json is historical
    # evidence, not a reason to keep presenting that stage as active.
    if resolved_status == "running":
        current_action = observed_action or current_action
    elif status:
        current_action = current_action or ""
    else:
        current_action = ""

    from .baseline_reference import view as reference_view
    from .experiment_view import view as experiment_view
    from .data_versions import catalog
    reference = reference_view(root)
    lifecycle = experiment_view(root,run_id)
    versions = catalog(root)
    job_sources=[root/row['request_ref'] for row in lifecycle['experiments']]
    version_sources=[root/'data_versions'/(row['id']+'.json') for row in versions]
    candidates = [root / "baseline_reference.json", *sorted((root/'baseline_references').glob('*.json')),
                  root / "run_state.json", root / "run_events.json", root / "budget.json",
                  root / f"preparation_{run_id}.json", research / "research_report.json",
                  research / "controller_session.json", research / "media" / "manifest.json",
                  *measurement_paths, *receipt_paths, *telemetry_files, *telemetry_chart_files,
                  *job_sources,*[p.with_name('result.json') for p in job_sources],*version_sources]
    revisions: dict[str, str | None] = {}
    for path in candidates:
        try:
            relative = path.relative_to(root).as_posix()
            revisions[relative] = digest(path) if path.is_file() and not path.is_symlink() else None
        except (OSError, ValueError):
            revisions[str(path)] = "unreadable"
    before = dict(revisions)
    for path in candidates:
        try:
            relative = path.relative_to(root).as_posix()
            after = digest(path) if path.is_file() and not path.is_symlink() else None
            if before.get(relative) != after:
                revisions[relative] = "changed_during_view_build"
        except (OSError, ValueError):
            revisions[str(path)] = "unreadable"
    current_measurements = (sorted((research / "measurements").glob("*.json"))
                            if (research / "measurements").is_dir() else [])
    if current_measurements != measurement_paths:
        revisions[f"research/{run_id}/measurements/*"] = "changed_during_view_build"
    stable = not any(value in {"changed_during_view_build", "unreadable"}
                     for value in revisions.values())
    revision_material = {"run_id": run_id, "sources": revisions}
    view = {
        "schema_version": 1,
        "run_id": run_id,
        "view_revision": object_digest(revision_material),
        "generated_at": now(),
        "source_consistency": "stable" if stable else "changed_during_view_build",
        "source_revision_vector": revisions,
        "status": resolved_status,
        "current_action": current_action or None,
        "budget": budget,
        "verified_level": report.get("verified_level"),
        "measurements": measurements,
        "baseline_reference": reference,
        "experiment_lifecycle": lifecycle,
        "data_versions": versions,
        "telemetry_attempts": telemetry_attempts[-12:],
        "verified_media": verified_media[:12],
        "unverified_media_count": pending_media,
        "limitations": ["media is linked only when its producer receipt hash is verified",
                        "episode attribution is shown only when the native record supplies it"],
    }
    atomic_json(root / "report" / "view.json", view)
    return view


def _research_overview(root: Path, run_id: str, *, status: str = "",
                       current_action: str = "") -> list[str]:
    """Render a compact research view from the versioned local projection."""
    research = Path(root) / "research" / run_id
    view = build_report_view(Path(root), run_id, status=status,
                             current_action=current_action)
    if not research.is_dir():
        return ["## Research", "", "No experiment record yet.",
                f"Run view: `{str(view.get('view_revision') or '')[:12]}` · "
                f"status `{_safe_cell(view.get('status'))}` · "
                "[View evidence](report/view.json)."]
    lines = ["## Research", "",
             f"研究状态：`{_safe_cell(view.get('status'))}`；"
             f"view `{str(view.get('view_revision') or '')[:12]}`；"
             f"来源一致性：`{_safe_cell(view.get('source_consistency'))}`。",
             f"[Detailed experiment record](research/{run_id}/RUN.md) · "
             f"[Research report](research/{run_id}/research_report.json) · "
             f"[View evidence](report/view.json)", ""]
    rows = view.get("measurements") or []
    if rows:
        lines += ["### Measurements", "",
                  "| Label | Outcome | Metric | Samples | Attempt |",
                  "| --- | --- | ---: | ---: | --- |"]
        for row in rows:
            value = row.get("metric_value")
            metric = f"{value:.6g}" if isinstance(value, (int, float)) else "—"
            lines.append(f"| [{_safe_cell(row.get('label'))}]({row.get('measurement_ref')}) | "
                         f"{_safe_cell(row.get('status'))} | "
                         f"{_safe_cell(row.get('metric_name'))}: {metric} | "
                         f"{_safe_cell(row.get('samples'))} | `{_safe_cell(row.get('attempt_id'))}` |")
        lines.append("")
    else:
        lines += ["No measurement has been recorded yet.", ""]
    telemetry_rows = view.get("telemetry_attempts") or []
    lines += ["### Live training telemetry (observation, not official score)", ""]
    if telemetry_rows:
        for row in telemetry_rows[-3:]:
            metric_summary = ", ".join(
                f"{_markdown_text(metric.get('name'), limit=80)}"
                f" ({metric.get('samples', 0)} points"
                f"{', step ' + str(metric.get('last_step')) if metric.get('last_step') is not None else ''})"
                for metric in row.get("metrics") or [])
            if not metric_summary:
                metric_summary = "waiting for native named scalars"
            suffix = ("; observer snapshot differs from terminal stage receipt (receipt wins)"
                      if row.get("stale_observer_snapshot") else "")
            lines.append(f"- Train attempt `{row['attempt_id']}` · "
                         f"stage `{_safe_cell(row.get('stage_status'))}` · "
                         f"telemetry `{_safe_cell(row.get('status'))}` · "
                         f"{metric_summary}{suffix} · "
                         f"[data]({row['data_ref']}) · "
                         f"[receipt]({row['receipt_ref']})")
            if row.get("chart_ref"):
                lines.append(f"\n![Native training observations for attempt "
                             f"{row['attempt_id']}](<{row['chart_ref']}>)\n")
            if row.get("errors"):
                lines.append("  Telemetry warnings: " + ", ".join(
                    f"`{_safe_cell(error, limit=80)}`" for error in row["errors"]))
        lines.append("Training telemetry is observational; official measurements still come "
                     "from the verified native evaluator.")
    else:
        lines.append("No training telemetry recorded yet.")
    lines.append("")
    lines += ["### Real simulation media", ""]
    verified = view.get("verified_media") or []
    if verified:
        for row in verified[:3]:
            context = []
            for key, label in (("task", "task"), ("seed", "seed"),
                               ("episode_id", "episode"),
                               ("measurement_label", "measurement")):
                if row.get(key) is not None:
                    context.append(f"{label}={_safe_cell(row[key], limit=60)}")
            details = " · " + ", ".join(context) if context else ""
            lines.append(f"- **{_safe_cell(row.get('trigger'))}**{details} · "
                         f"[Open source-verified run recording](<{row.get('path')}>).")
            if row.get("preview_ref"):
                lines.append(f"\n![从上述真实仿真录像解码的帧](<{row['preview_ref']}>)\n")
    else:
        lines.append("No source-verified run recording is available yet.")
    if view.get("unverified_media_count"):
        lines.append(f"{view['unverified_media_count']} discovered recording(s) remain unverified "
                     "and are not shown as evidence.")
    lines += ["", f"[Media manifest](research/{run_id}/media/manifest.json)"]
    return lines


_RESEARCH_START = "<!-- AUTOSIM_RESEARCH_START -->"
_RESEARCH_END = "<!-- AUTOSIM_RESEARCH_END -->"


def refresh_research_projection(root: Path, run_id: str) -> Path:
    """Refresh the bounded research section in the top-level run document."""
    root = Path(root)
    destination = root / "RUN.md"
    with _report_lock(destination):
        try:
            original = destination.read_text(encoding="utf-8")
        except OSError:
            return destination
        if _RESEARCH_START not in original or _RESEARCH_END not in original:
            return destination
        before, rest = original.split(_RESEARCH_START, 1)
        _, after = rest.split(_RESEARCH_END, 1)
        projection = "\n".join(_research_overview(root, run_id))
        atomic_text(destination, before + _RESEARCH_START + "\n" + projection +
                    "\n" + _RESEARCH_END + after)
    # No model call from telemetry refresh; the last narration retains its own revision.
    try:
        from . import recorder
        view = _load(root / "report" / "view.json") or {}
        if view:
            presentation = recorder.refresh(root, view)
            with _report_lock(destination):
                content = destination.read_text(encoding="utf-8")
                if recorder.START in content and recorder.END in content:
                    prefix, remainder = content.split(recorder.START, 1)
                    _, suffix = remainder.split(recorder.END, 1)
                    atomic_text(destination, prefix + presentation + suffix)
                    _publish_presentation_html(destination, prefix + presentation + suffix)
    except Exception as exc:  # report refresh must not interrupt training
        atomic_json(root / "report" / "presentation_error.json", {"at": now(), "error": type(exc).__name__})
    return destination


def refresh_live_status(root: Path, *, destination: Path | None = None,
                        live_root: Path | None = None, fallback_status: str = "",
                        fallback_current: str = "") -> Path:
    """Update the live strip without rescanning a possibly huge run tree."""
    root = Path(root)
    destination = Path(destination) if destination is not None else root / "RUN.md"
    live_root = Path(live_root) if live_root is not None else root
    live = _load(live_root / "running.json") or {}
    if not live and fallback_status in {"", "running"}:
        state = _load(root / "run_state.json") or {}
        action = state.get("current_action") or {}
        if (state.get("status") == "running" and isinstance(action, dict)
                and action.get("step")):
            from datetime import datetime
            try:
                value = action.get("started_at") or action.get("at")
                if value is None:
                    parent = action.get("parent_action") or {}
                    value = parent.get("started_at") or parent.get("at")
                started = float(value) if isinstance(value, (int, float)) else datetime.fromisoformat(str(value)).timestamp()
                if math.isfinite(started):
                    live = {"stage": action["step"], "started_at": started}
            except (ValueError, TypeError, OverflowError):
                pass
    budget = _load(destination.parent / "budget.json") or _load(root / "budget.json") or {}
    elapsed = max(0, round(time.time() - float(live.get("started_at") or time.time())))
    remaining = budget.get("remaining_wall_seconds")
    if isinstance(budget.get("started_epoch"), (int, float)) and isinstance(
            budget.get("wall_seconds"), (int, float)):
        remaining = max(0.0, float(budget["wall_seconds"]) -
                        (time.time() - float(budget["started_epoch"])))
    budget_line = (f"墙钟预算剩余约 {float(remaining):.0f} 秒。"
                   if isinstance(remaining, (int, float)) else "墙钟预算未记录。")
    terminal_fallback = bool(fallback_status and fallback_status != "running")
    current = (str(fallback_current or "unknown") if terminal_fallback else
               str(live.get("stage") or fallback_current or "unknown"))[:120]
    status = (str(fallback_status) if terminal_fallback else
              "running" if live else str(fallback_status or "unknown"))
    health_path = destination.parent / "report" / "health.json"
    previous_health = _load(health_path) or {}
    previous_at = previous_health.get("last_successful_refresh_epoch")
    if (not fallback_status and previous_health.get("status") == "running" and
            isinstance(previous_at, (int, float)) and time.time() - previous_at > 90):
        status = "stale: process state requires reconciliation"
        current = str(previous_health.get("current_action") or "unknown")[:120]
    localized_status = ("运行中" if status == "running" else
                        "状态待核对" if status.startswith("stale:") else status)
    strip = (f"{_LIVE_START}\n"
             f"Updated: {now()} · Status: {status} · 状态：{localized_status} · "
             f"Current action: `{current}`; stage elapsed about {elapsed}s "
             f"（约 {elapsed} 秒）。{budget_line}\n"
             f"{_LIVE_END}")
    with _report_lock(destination):
        try:
            original = destination.read_text(encoding="utf-8")
        except FileNotFoundError:
            original = "# Research run\n\n"
        if _LIVE_START in original and _LIVE_END in original:
            beginning, rest = original.split(_LIVE_START, 1)
            _, ending = rest.split(_LIVE_END, 1)
            updated = beginning + strip + ending
        else:
            first, sep, rest = original.partition("\n")
            updated = (first + f"\n\n{strip}\n" + (rest if sep else ""))
        # This panel must advance during a long blocking native operation, not
        # only when the Scheduler's entire derivation finally returns. Bound the
        # evidence scan rate; it neither invokes Recorder nor claims liveness.
        panel_at = previous_health.get('native_panel_refresh_epoch', 0)
        if not isinstance(panel_at, (int, float)):
            panel_at = 0
        if time.time() - panel_at >= 15:
            try:
                from .recorder import refresh_native_panel
                updated = refresh_native_panel(updated, root)
                panel_at = time.time()
            except (OSError, ValueError, TypeError, KeyError):
                pass  # presentation failures must not interrupt execution
        atomic_text(destination, updated)
        _publish_presentation_html(destination, updated)
        health_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(health_path, {"schema_version": 1, "destination": str(destination),
                                  "last_successful_refresh": now(),
                                  "last_successful_refresh_epoch": time.time(),
                                  "native_panel_refresh_epoch": panel_at,
                                  "status": status, "current_action": current})
    return destination
