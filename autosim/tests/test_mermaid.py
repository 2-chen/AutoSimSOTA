"""The two diagrams, and the property that decides whether they are safe to draw.

A diagram that fails to parse renders as an error box and is harmless. A diagram that parses
and says something false renders as a picture with the authority of a chart, and that is the
failure this file is mostly about: the renderers must invent no edges and no bars, and every
string that reaches the output must be incapable of ending a Mermaid label early.

The invariants below were derived by running the real parser over hostile input -- `cuda:0`,
`train.n_epochs=1`, quoted prose, brackets, Chinese punctuation -- and reading what it choked
on. `tools/mermaid_parse.mjs` is that parser, kept so the check can be repeated; when
`MERMAID_PARSER` names it, the tests below also parse what they built, and skip that part when
it is absent rather than pretending the invariant is the same check.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from autosim.research import mermaid


def _parser() -> str | None:
    return os.environ.get("MERMAID_PARSER") or None


def _parses(block: str) -> None:
    """Parse with the real Mermaid when it is available, and otherwise say nothing.

    Not a silent skip: the invariants below are asserted either way. This is the stronger
    check, and it is only absent because Mermaid needs a node dependency tree that does not
    belong in this repository's test run.
    """
    parser = _parser()
    if not parser:
        return
    body = re.search(r"```mermaid\n(.*?)```", block, re.S)
    source = body.group(1) if body else block
    result = subprocess.run(["node", parser, source], capture_output=True, text=True,
                            timeout=180, check=False)
    assert "PARSE OK" in result.stdout, result.stdout + result.stderr


def _hostile_rows():
    return [
        {"at": "2026-09-20T02:37:09+00:00", "seconds": 1053.0,
         "label": "cuda:0 | train.n_epochs=1", "section": "stage"},
        {"at": "2026-09-20T03:00:00+00:00", "seconds": 1.0,
         "label": 'he said "stop", [then] {this}', "section": "stage"},
        {"at": "2026-09-20T04:00:00+00:00", "seconds": 90.0,
         "label": "提案：把 mass 从 0.5 调到 0.75", "section": "stage"},
        {"at": "2026-09-20T05:00:00+00:00", "seconds": 7200.0,
         "label": "a/b\\c<d>&e#f", "section": "stage"},
    ]


def test_no_label_can_end_the_syntax_it_sits_in():
    """Every character Mermaid reads as its own syntax, in every label the two renderers
    emit. The colon matters most because this system's data is full of them: `cuda:0`, and
    every `train.n_epochs=1` that reaches a task name."""
    block = mermaid.gantt(_hostile_rows(), title='运行 & <时间线> "x"')
    _parses(block)
    for line in block.splitlines():
        body = line.strip()
        if not body or body.startswith(("```", "gantt", "title", "dateFormat", "axisFormat",
                                        "tickInterval", "section")):
            continue
        # A task line is `label :start, duration`, so the separator is ` :` and everything
        # before it is the label. The colons further along the line belong to the timestamp,
        # which is this renderer's own output and not a value from the record.
        label, separator, rest = body.partition(" :")
        assert separator == " :", body
        assert not re.search(r'[\[\]{}()|<>#`"\'\\,%;:]', label), label
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}, \d+[sm]", rest), rest


def test_a_bar_is_placed_only_where_the_record_gave_a_time():
    """A Gantt bar's position *is* its claim. A row with no usable timestamp is left out
    rather than drawn at the origin, because a bar in the wrong place is a false statement
    that looks like a measurement."""
    rows = [{"at": "", "seconds": 10, "label": "no time"},
            {"at": "not a timestamp", "seconds": 10, "label": "unparseable"},
            {"at": "2026-09-20T02:00:00+00:00", "seconds": 10, "label": "real"}]
    block = mermaid.gantt(rows)
    assert "real" in block
    assert "no time" not in block and "unparseable" not in block


def test_a_run_with_no_timestamps_draws_nothing_rather_than_an_empty_chart():
    assert mermaid.gantt([{"at": "", "seconds": 1, "label": "x"}]) == ""
    assert mermaid.provenance([]) == ""


def test_the_provenance_graph_draws_only_the_edges_the_record_holds():
    """It is a provenance graph, so an edge that is not in the record must not appear. A
    decision with no recorded grounds has no incoming edge, and the graph says that by
    omission rather than by inventing a plausible source."""
    decisions = [
        {"kind": "decision", "by": "model", "activity": "vary the scale",
         "used": ["evidence:abc"], "produced": ["round_1/proposal.json"],
         "outcome": {"state": "known", "what": {"rate": 0.65}}},
        {"kind": "decision", "by": "tool", "activity": "no grounds recorded",
         "used": [], "produced": [], "outcome": {"state": "open"}},
        {"kind": "outcome_only", "activity": "not a decision"},
    ]
    block = mermaid.provenance(decisions)
    _parses(block)
    assert "evidence/abc" in block and "round_1/proposal.json" in block
    assert "结果未回填" in block
    assert "not a decision" not in block
    assert block.count("-->") == 3          # two from the first, one open marker from the second


def test_the_same_record_produces_the_same_graph():
    """Stable node ids, so two readings of an unchanged run can be compared. A diagram whose
    nodes move when nothing moved cannot be diffed against itself, which is most of what a
    diagram in a record is for."""
    decisions = [{"kind": "decision", "by": "model", "activity": "a", "used": ["e1"],
                  "produced": ["p1"], "outcome": {"state": "open"}}]
    assert mermaid.provenance(decisions) == mermaid.provenance(decisions)


def test_two_renders_of_one_run_produce_the_same_document(tmp_path):
    """The end-to-end form of the same property, over a document that contains both diagrams."""
    from autosim.research import run_record as rr
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    (root / "events.json").write_text(
        '{"rows": [{"at": "2026-09-20T02:37:09+00:00", "event": "stage", "stage": "cuda:0",'
        ' "seconds": 1053.0}]}', encoding="utf-8")
    (root / "decisions.json").write_text(
        '{"rows": [{"id": "d1", "kind": "decision", "by": "model", "activity": "vary x",'
        ' "used": ["e"], "produced": [], "outcome": {"state": "open"}}]}', encoding="utf-8")
    rr.generate(root, summary="摘要。")
    first = (root / "RUN.md").read_text(encoding="utf-8")
    rr.generate(root)
    assert (root / "RUN.md").read_text(encoding="utf-8") == first
    assert "```mermaid" in first


def test_a_duration_is_drawn_in_a_unit_the_chart_can_show():
    """Sub-minute work is drawn in seconds and anything longer in minutes: a bar rounded to
    `0m` is a bar that is not there."""
    assert mermaid._duration(0) == "1s"
    assert mermaid._duration(45.2) == "45s"
    assert mermaid._duration(600) == "10m"
    assert mermaid._duration("not a number") == "1s"


@pytest.mark.parametrize("value,expected", [
    ("cuda:0", "cuda/0"),
    ('he said "stop"', "he said stop"),
    ("a[1] {b} (c)", "a 1 b c"),
    ("x | y", "x / y"),
    ("train.n_epochs=1", "train.n_epochs=1"),
])
def test_escaping_substitutes_rather_than_deletes(value, expected):
    """Substitution, not removal, so two different values do not collapse into one label:
    `cuda:0` and `cuda 0` are different devices and a diagram printing both as the same word
    is worse than one printing neither."""
    assert mermaid._safe(value) == expected
