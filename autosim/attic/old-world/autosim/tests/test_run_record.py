"""The run's document, and the property that makes it worth reading.

Every claim in this file is about one thing: whether a reader can tell which sentences of the
document can be checked against the record and which cannot. A generated document that mixes
the two is worse than no document, because it lends a model's prose the authority of a file
listing.
"""

import json
from pathlib import Path

import pytest

from autosim.research import run_index as ri
from autosim.research import run_record as rr


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _run(tmp_path: Path, **kwargs) -> Path:
    """The usual case: one run, at `<tmp_path>/run`."""
    return _run_at(tmp_path / "run", **kwargs)


def _run_at(root: Path, *, stage: str = "train", returncode: int = 0,
            rate=None, readings=None, said: str = "") -> Path:
    """A run at a path the caller chooses, for the tests that need two of them side by side."""
    _write(root / "events.json", {"rows": [
        {"at": "2026-09-20T02:37:09+00:00", "event": "stage", "stage": stage,
         "returncode": returncode, "seconds": 1053.0, "device": "cuda:0",
         "device_why": "card 0, 31532 MiB free"}]})
    _write(root / "measurements" / "baseline.json", {
        "label": "baseline", "ok": returncode == 0, "success_rate": rate,
        "readings": readings or {}, "where": stage if rate is None else None,
        "argv": ["python", "train.py", "steps=200"], "device": "cuda:0",
        "device_why": "card 0, 31532 MiB free", "said": said,
        "settings": {"train.n_epochs": 1}})
    return root


# -- the rule the whole module is built around -------------------------------------------------

def test_every_section_says_whether_it_was_computed_or_written(tmp_path):
    """The design decision everything else follows from. A model's summary that looks like a
    file listing is a summary a reader will cite as though it were one."""
    root = _run(tmp_path, rate=0.65)
    rr.generate(root, summary="这一轮成功了。")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "**写出来的**" in document and "**算出来的**" in document
    heading = document.index("## 摘要")
    written = document.index("**写出来的**", heading)
    assert heading < written < document.index("## 时间线")
    # And the computed sections say the other thing, so the two are never mistaken for each
    # other by a reader skimming the page.
    for title in ("## 时间线", "## 决策与结果", "## 数字", "## 产物与证据", "## 未解决", "## 复现方法"):
        at = document.index(title)
        assert "**算出来的**" in document[at:at + 200], title


def test_a_section_with_nothing_in_it_says_which_kind_of_nothing(tmp_path):
    """"nothing to report" and "nothing was looked at" must not read the same. Every section
    here is either empty because the run left nothing or empty because the file is missing."""
    root = _run(tmp_path, rate=0.65)
    rr.generate(root, summary="")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "这一节没有被写出来" in document
    assert "没有可用的模型客户端" in document
    assert "没有决策记录" in document


# -- the check on the written half -------------------------------------------------------------

def test_a_number_the_record_does_not_contain_is_reported_as_unsourced(tmp_path):
    root = _run(tmp_path, rate=0.6523, readings={"train_loss": 5.36})
    rr.generate(root, summary="成功率 0.65，训练损失 5.36，另外提升了 47.2%。")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "`47.2%`" in document and "找不到出处" in document
    # The two that are sourced are not listed as unsourced.
    untraced = document[document.index("找不到出处的数字"):]
    untraced = untraced[:untraced.index("\n")]
    assert "5.36" not in untraced


def test_a_rounded_number_counts_as_sourced_and_a_mis_scaled_one_does_not():
    """`0.65` about `0.6523` is a summary rounding a number it read. `116.7%` is not a rounding
    of `1.2`, and treating the percentage's precision as one decimal -- which this did -- makes
    it one, and quietly passes a fabricated figure."""
    pool = [0.3, 0.6523, 1.2]
    assert rr.check_claims("成功率 0.65。", pool)["untraced"] == []
    assert rr.check_claims("成功率 0.30。", pool)["untraced"] == []
    assert rr.check_claims("提升了 116.7%。", pool)["untraced"] == ["116.7%"]


def test_numbers_hiding_inside_the_record_s_text_are_found(tmp_path):
    """The free-memory figure lives in `device_why` as prose and the losses live in `said`. A
    check that reads only the JSON structure calls every one of those fabricated, and a check
    that cries wolf is a check nobody reads."""
    root = _run(tmp_path, rate=0.5, said="[info] Epoch: 0 | train loss: 5.36")
    source = rr.gather(root)
    result = rr.check_claims("当时 31532 MiB 可用，损失 5.36。", rr.known_numbers(source))
    assert result["untraced"] == []


def test_the_check_reports_what_it_established_and_not_more(tmp_path):
    """It establishes that a number appears in the record. A wrong conclusion assembled out of
    real numbers passes it, and the document has to say so rather than imply otherwise."""
    root = _run(tmp_path, rate=0.5)
    rr.generate(root, summary="成功率 0.50。")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "不说明它被用对了" in document
    assert "会通过上面这项检查" in document


def test_identifiers_are_not_mistaken_for_claims():
    """`cuda:0` and `LIBERO_10` carry digits and are not numbers anybody asserted. The strip
    pattern that removes them must not also remove bare numbers -- when it did, the check
    reported nothing at all about a summary that invented every figure in it."""
    result = rr.check_claims("跑在 cuda:0 上，用 LIBERO_10，成功率 0.65。", [0.3])
    assert result["claims"] == 1 and result["untraced"] == ["0.65"]


# -- reading the two kinds of run this system produces -----------------------------------------

def test_a_run_that_stopped_before_measuring_anything_still_produces_a_document(tmp_path):
    """The case that most needs writing down: a run that ends with no number at all. It used to
    be the case with the least to go on and the most to explain."""
    root = _run(tmp_path, stage="train", returncode=-9, rate=None)
    rr.generate(root, summary="训练被杀死了。")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "没有产生数字" in document and "train 阶段没有跑完" in document
    assert "rc=-9" in document


def test_the_other_pipeline_s_rounds_are_read_from_the_two_files_that_hold_them(tmp_path):
    """The full pipeline writes the proposal and the result minutes apart, from different code.
    Joining them is this document's work, and it says so rather than presenting the join as
    something the system recorded."""
    root = tmp_path / "run"
    _write(root / "rounds" / "round_1" / "proposal.json", {
        "created_at": "2026-09-16T11:41:57+00:00", "response_sha256": "a" * 64,
        "proposal": {"hypothesis": "the failures are all incomplete_after_motion",
                     "training": {"steps": 200}}})
    _write(root / "rounds" / "round_1" / "round_result.json", {
        "round": 1, "status": "completed", "completed_at": "2026-09-16T11:44:55+00:00",
        "development_summary": {"success_rate": 1.0, "episode_count": 4}})
    _write(root / "rounds" / "round_2" / "round_result.json", {
        "round": 2, "status": "failed", "error": "TypeError: NoneType is not callable"})
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "incomplete_after_motion" in document and "1.000" in document
    assert "两个分开写的文件" in document
    assert "第 2 轮没有完成" in document and "NoneType" in document


def test_commands_are_read_back_from_the_records_of_running_them(tmp_path):
    """The full pipeline keeps no event stream, but it writes argv, cwd, duration and return
    code beside every process it starts. That is the timeline and the reproduction recipe, and
    nothing had read it back out."""
    root = tmp_path / "run"
    _write(root / "evaluations" / "official" / "process" / "process.json", {
        "command": ["python", "-m", "autosim.research.evaluation", "--episodes", "4"],
        "cwd": "/tmp/clean", "status": "completed", "returncode": 0,
        "elapsed_seconds": 84.2, "started_at": "2026-09-16T11:40:22+00:00"})
    _write(root / "evaluations" / "round_1" / "process" / "process.json", {
        "command": ["python", "-m", "autosim.research.collection_worker"],
        "status": "completed", "returncode": 0, "elapsed_seconds": 40.2,
        "started_at": "2026-09-16T11:39:00+00:00"})
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "记录下来的命令" in document and "--episodes 4" in document
    assert "/tmp/clean" in document
    # Ordered by the clock, not by the path they were found under.
    assert document.index("collection_worker") < document.index("--episodes 4")


# -- the artifact index, which is where a reader checks the rest -------------------------------

def test_scene_assets_are_not_described_as_recordings():
    """A photograph of a table in `assets/` is something the benchmark renders, not something
    the run did. The extension cannot tell them apart; the path can."""
    what, _ = rr.role_of("optimized_repo/assets/Basket/kago2_u1_v1_diffuse.jpg")
    assert "场景素材" in what
    what, _ = rr.role_of("rollouts/episode_0.mp4")
    assert "录像" in what


def test_the_index_does_not_hide_a_large_group_behind_one_row(tmp_path):
    """Collapsing a group to its first entry hides exactly the files a reader does not already
    know about. Four hundred unclassified artifacts came out as one row saying there were four
    hundred of them. The count is exact; the names are a sample, and the document says which."""
    root = _run(tmp_path, rate=0.5)
    for index in range(20):
        (root / "rollouts").mkdir(exist_ok=True)
        (root / "rollouts" / f"trace_{index}.dat").write_text("x", encoding="utf-8")
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert document.count("rollouts/trace_") == rr.LISTED_PER_KIND
    assert "同类共 **20** 个" in document
    assert "按路径排序的取样" in document
    # And the same twelve on a second reading, because a sample that moves cannot be cited.
    rr.generate(root)
    assert (root / "RUN.md").read_text(encoding="utf-8") == document


def test_a_file_the_run_never_wrote_is_not_reported_as_a_fault(tmp_path):
    """A run that predates a feature has no file for it. Calling that "could not be read" would
    report every old run as damaged, and the difference is what tells a reader whether there is
    anything to act on."""
    root = _run(tmp_path, rate=0.5)
    source = rr.gather(root)
    assert not [one for one in source["unreadable"] if "读不出来" in one]
    assert [one for one in source["unreadable"] if "没有写这个文件" in one]
    (root / "decisions.json").write_text("{not json", encoding="utf-8")
    assert [one for one in rr.gather(root)["unreadable"] if "读不出来" in one]


# -- it must not fail when the run has already failed -------------------------------------------

def test_it_produces_a_document_even_when_the_records_are_unreadable(tmp_path):
    """A generator that dies on a malformed record dies exactly when the run has gone wrong and
    the document is most wanted."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "events.json").write_text("[[[", encoding="utf-8")
    assert rr.generate(root)
    assert "运行记录" in (root / "RUN.md").read_text(encoding="utf-8")


def test_regenerating_reuses_the_written_half(tmp_path):
    """Regeneration happens at every stage boundary. Calling the model each time would make the
    document change between two readings of the same run."""
    root = _run(tmp_path, rate=0.5)
    rr.generate(root, summary="第一版。")
    rr.generate(root)
    assert "第一版。" in (root / "RUN.md").read_text(encoding="utf-8")


def test_a_run_with_nothing_in_it_at_all_still_renders(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert rr.generate(root, summary="没有东西可写。")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "没有事件记录" in document and "没有测量记录" in document


def test_an_unrecognised_maker_is_refused_by_the_decision_record(tmp_path):
    """`by` is not decoration, and this is the module that renders it to a reader."""
    from autosim.research.decisions import Decision, Decisions
    with pytest.raises(ValueError):
        Decisions(tmp_path).record(Decision(activity="x", why="y", by="the system"))


def test_recordings_are_grouped_by_what_they_are_evidence_of(tmp_path):
    """A recording made while collecting is training data; one made while evaluating is
    evidence about a policy. They are the same file format on disk and they support opposite
    kinds of claim, so an index that lists them together has said nothing."""
    root = _run(tmp_path, rate=0.5)
    for relative in ("rounds/round_1/collection/targeted/data/x/videos/chunk-000/"
                     "observation.images.cam_high/episode_000003.mp4",
                     "eval_result/beat_block_hammer/ACT/demo_clean/episode0.mp4",
                     "optimized_repo/assets/table/wood.jpg",
                     "evaluations/round_1_candidate_development/videos/episode_0.mp4"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 录像与 demo"):document.index("## 产物与证据")]
    assert "训练数据本身" in section and "评测器录下" in section
    # The scene texture is not a recording and must not be counted as one.
    assert "wood.jpg" not in section
    assert "共 **3** 个录像文件" in section
    assert "证明的不是一回事" in section


def test_a_run_with_no_recordings_says_why_there_might_not_be_any(tmp_path):
    """"no videos were made" and "videos were made and this did not find them" are different,
    and the reader of a run that has no demos needs to know which one to act on."""
    root = _run(tmp_path, rate=0.5)
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 录像与 demo"):document.index("## 产物与证据")]
    assert "这次运行没有留下录像" in section and "resolution.video_log" in section


def test_a_client_that_raises_does_not_take_the_document_with_it(tmp_path):
    """The summary is written by a model, and a model call is the least reliable thing in this
    pipeline. Its failure is a missing section, not a missing document."""
    class Broken:
        model = "broken"

        def chat(self, *a, **k):
            raise TimeoutError("the provider did not answer")

    root = _run(tmp_path, rate=0.5)
    assert rr.generate(root, client=Broken(), refresh_summary=True)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "这一节没有被写出来" in document and "TimeoutError" in document
    # And the computed half is still there, which is the half that cannot be wrong.
    assert "## 时间线" in document and "1053.0s" in document


def test_the_full_pipeline_writes_the_document_without_failing_the_run(tmp_path):
    """`write_run_record` is called from this pipeline's exception handlers. A document
    generator that raises there would replace the run's real error with its own."""
    from autosim.research.repository_autoresearch import RepositoryAutoResearch

    class Any:
        def __getattr__(self, name):
            raise RuntimeError(f"the run is not initialized and {name} was reached")

    runner = Any.__new__(RepositoryAutoResearch)
    runner.run_root = tmp_path / "run"
    runner.run_root.mkdir()
    runner.config = type("C", (), {"run_id": "test_run"})()
    runner.state = {}
    runner.write_run_record(object())          # a client that is not one, on purpose
    assert (runner.run_root / "RUN.md").is_file()


# -- the third shape, and the index over all of them -------------------------------------------

def _scout(tmp_path: Path) -> Path:
    root = tmp_path / "scouting" / "libero_onboard_v1"
    root.mkdir(parents=True)
    _write(root / "draft_capabilities_1.json", {
        "stage": "capabilities", "attempt": 1, "prompt_chars": 53967, "response_chars": 8239,
        "provider": {"provider_model": "deepseek-flash"}, "content": "{\"capabilities\": {}}"})
    _write(root / "draft_identity_1.json", {
        "stage": "identity", "attempt": 1, "prompt_chars": 49179, "response_chars": 4200,
        "provider": {"provider_model": "deepseek-flash"}, "content": "{\"identity\": {}}"})
    _write(root / "verification.json", {
        "benchmark": "LIBERO", "state": "verified", "task_count": 130,
        "identity_failures": ["declared state_dim 8, the program says 4"],
        "checks": [{"check": "path_exists", "subject": "setup.py", "passed": True},
                   {"check": "path_exists", "subject": "assets/", "passed": False,
                    "why": "the directory is not in the checkout"}]})
    _write(root / "declaration.json", {"created_at": "2026-09-17T17:09:38+00:00",
                                       "declaration": {"benchmark": "LIBERO"}})
    _write(root / "onboarding.json", {"created_at": "2026-09-17T17:09:38+00:00",
                                      "readiness": {"training": "no entry point found"}})
    return root


def test_an_onboarding_run_produces_a_document_too(tmp_path):
    """A third of this machine's work was onboarding benchmarks, and none of it was in any
    index. Its records are a different set of files -- a draft per exchange, a verdict, a
    declaration -- and the same rule applies to them."""
    root = _scout(tmp_path)
    rr.generate(root, title="LIBERO onboarding")
    document = (root / "RUN.md").read_text(encoding="utf-8")
    assert "capabilities" in document and "deepseek-flash" in document
    assert "53967" in document
    assert "校验结论：**verified，1 项检查未通过**" in document
    assert "`assets/`" in document and "not in the checkout" in document
    assert "declared state_dim 8" in document


def test_it_does_not_claim_a_reason_the_record_does_not_hold(tmp_path):
    """"Why" is the question this whole layer exists to answer, and the answer here is that
    this shape never recorded one. A column headed 为什么 filled with "see the file" would be
    a reason-shaped blank, which is worse than an admitted gap -- it is the ADR-015 failure
    reproduced in the document meant to fix it."""
    root = _scout(tmp_path)
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 决策与结果"):document.index("## 数字")]
    assert "没有单独记录「为什么」" in section
    assert "| 为什么 |" not in section
    assert "| 做了什么 |" in section


def test_the_index_holds_every_shape_and_admits_the_directories_it_skipped(tmp_path):
    """Finding a run meant listing a directory and opening what looked promising. An index
    that quietly omits the directories which produced nothing makes the work look tidier
    than it was."""
    runs = tmp_path / "autoresearch_runs"
    _scout(runs)
    _run_at(runs / "Derived" / "a_derived_run", rate=0.65)
    _write(runs / "Research" / "a_research_run" / "run_state.json",
           {"status": "completed", "stage": "complete", "created_at": "2026-09-16T00:00:00+00:00"})
    (runs / "scouting" / "an_attempt_that_produced_nothing").mkdir(parents=True)

    document = ri.runs_index(runs)
    shapes = {row["run"]: row["shape"] for row in document["runs"]}
    assert shapes["libero_onboard_v1"] == "scouting"
    assert shapes["a_derived_run"] == "derived"
    assert shapes["a_research_run"] == "research"
    assert document["directories_with_no_records"] == ["scouting/an_attempt_that_produced_nothing"]
    assert document["with_unreadable_records"] == []


def test_a_run_that_measured_nothing_has_no_score_rather_than_a_zero(tmp_path):
    """Zero is a measurement -- the policy failed every episode. A run that measured nothing
    produced no number, and an index that prints 0.0 for it states something false."""
    runs = tmp_path / "autoresearch_runs"
    _run_at(runs / "Research" / "scored", rate=0.65)
    _run_at(runs / "Research" / "unscored", rate=None)
    rows = {row["run"]: row for row in ri.runs_index(runs)["runs"]}
    assert rows["scored"]["best_success_rate"] == 0.65
    assert rows["unscored"]["best_success_rate"] is None


def test_the_index_reports_a_fault_and_not_a_missing_file(tmp_path):
    """`gather` reports every file a run's shape does not have, which for one shape is all of
    another shape's records. That is true and useless in an index whose job is to point at
    what is worth opening, so the index keeps only what is actually wrong."""
    runs = tmp_path / "autoresearch_runs"
    root = _run_at(runs / "Research" / "damaged", rate=0.5)
    (root / "decisions.json").write_text("{not json", encoding="utf-8")
    row = next(r for r in ri.runs_index(runs)["runs"] if r["run"] == "damaged")
    assert row["unreadable"] and all("读不出来" in one for one in row["unreadable"])
