"""The single-file page, and the two properties that make it worth sending to somebody.

**It must depend on nothing.** No script tag, no stylesheet link, no font, no image, no URL.
That is not a stylistic preference: the constraint is what rules out Mermaid here, and a page
that needs a script to draw its chart draws nothing in a mail client. Every assertion about
self-containment below is a check that the file still opens on a machine that has never heard
of this repository.

**It must be well-formed.** A browser given a malformed SVG renders nothing at all and says
nothing at all -- the failure is a blank space, not an error. So every chart is parsed as XML
here, with labels carrying the characters that break it: `<`, `>`, `&`, quotes, brackets.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from autosim.research import report_page as rp


def _hostile_rows():
    return [
        {"at": "2026-09-20T02:37:09+00:00", "seconds": 10,
         "label": 'a<b>&"c" [d] {e}', "section": "s"},
        {"at": "2026-09-20T03:00:00+00:00", "seconds": 900,
         "label": "中文 标签 · train.n_epochs=1", "section": "s"},
    ]


def _hostile_decisions():
    return [
        {"kind": "decision", "by": "model", "activity": 'vary <a> & "b"',
         "used": ['evidence:<x>&"y"', "e2"], "produced": ['p"1"', "p<2>"],
         "outcome": {"state": "open"}},
        {"kind": "decision", "by": "tool", "activity": "no grounds recorded",
         "used": [], "produced": [], "outcome": {"state": "known"}},
    ]


def test_the_page_asks_nothing_of_the_network():
    """A single file, openable from a filesystem, forwardable by mail. Checked by looking for
    the constructs that would make it neither."""
    html = rp.page({"root": "/tmp/run", "unreadable": []}, [], title="run")
    assert "<script" not in html
    assert not re.search(r'\ssrc\s*=', html)
    assert not re.search(r'<link\b', html)
    assert not re.search(r'https?://', html)
    assert not re.search(r'@import', html)
    assert html.startswith("<!DOCTYPE html>")


def test_every_chart_is_well_formed_xml():
    """A malformed SVG renders as nothing, silently. The labels carry `<`, `>`, `&` and quotes
    because this system's own records do -- an activity string is model prose."""
    charts = {"gantt": rp.gantt_svg(_hostile_rows(), title='t<b>&"x"'),
              "provenance": rp.provenance_svg(_hostile_decisions()),
              "curve": rp.curve_svg([("r<1>", 0.5), ("r2", None), ("r3&x", 0.8)])}
    for name, svg in charts.items():
        assert svg, name
        ET.fromstring(svg)                       # raises if malformed
        assert "<b>" not in svg and "<x>" not in svg, name


def test_a_hostile_title_and_root_do_not_escape_into_the_markup():
    html = rp.page({"root": "/x<y>", "unreadable": []}, [], title='<hostile> & "title"')
    assert "<hostile>" not in html
    assert "&lt;hostile&gt;" in html


def test_a_round_with_no_number_is_absent_from_the_curve_rather_than_zero():
    """Zero is a measurement -- the policy failed every episode -- and a missing number is not
    a measurement at all. Plotting the second as the first is the commonest chart lie."""
    svg = rp.curve_svg([("round 1", 0.0), ("round 2", None), ("round 3", 0.8)])
    assert "0.000" in svg and "0.800" in svg
    assert "round 2" not in svg
    assert rp.curve_svg([("round 1", None)]) == ""


def test_a_bar_with_no_start_time_is_not_drawn():
    """A bar's position is its claim. One placed by a guess is a false statement that reads
    as a measurement."""
    svg = rp.gantt_svg([{"at": "", "seconds": 5, "label": "no time"},
                        {"at": "2026-09-20T02:00:00+00:00", "seconds": 5, "label": "real"}])
    assert "real" in svg and "no time" not in svg
    assert rp.gantt_svg([{"at": "nonsense", "seconds": 5, "label": "x"}]) == ""


def test_the_provenance_chart_draws_only_the_edges_the_record_holds():
    svg = rp.provenance_svg(_hostile_decisions())
    assert 'evidence' in svg and "no grounds recorded" in svg
    # Two sources and two products, so four edges, and nothing for the decision with none.
    assert svg.count("<line") == 4
    assert rp.provenance_svg([{"kind": "decision", "used": [], "produced": []}]) == ""


def test_a_chart_label_that_was_truncated_says_so():
    assert rp._clip("a" * 100).endswith("…")
    assert rp._clip("short") == "short"


# -- the document as a whole ---------------------------------------------------------------

def _run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    (root / "events.json").write_text(
        '{"rows": [{"at": "2026-09-20T02:37:09+00:00", "event": "stage", "stage": "train",'
        ' "returncode": -9, "seconds": 1053.0}]}', encoding="utf-8")
    (root / "measurements" / "baseline.json").write_text(
        '{"label": "baseline", "success_rate": null, "where": "train", "readings": {},'
        ' "argv": ["python", "train.py"], "settings": {}}', encoding="utf-8")
    (root / "decisions.json").write_text(
        '{"rows": [{"id": "d1", "kind": "decision", "by": "model", "activity": "vary x",'
        ' "used": ["evidence:abc"], "produced": [], "outcome": {"state": "open"}}]}',
        encoding="utf-8")
    return root


def test_the_page_carries_the_same_sections_and_the_same_provenance_labels(tmp_path):
    """It is the same document in another format, so a reader of either must be able to tell
    which half is checkable. A page that dropped the labels would be a different document."""
    root = _run(tmp_path)
    rp.write(root)
    html = (root / "RUN.html").read_text(encoding="utf-8")
    for title in ("摘要", "时间线", "决策与结果", "数字", "产物与证据", "未解决", "复现方法"):
        assert f"<h2>{title}</h2>" in html, title
    assert "算出来的" in html and "写出来的" in html
    assert html.count("<svg") == 2          # the timeline and the decisions both have records


def test_it_writes_a_page_saying_what_failed_rather_than_not_writing_one(tmp_path, monkeypatch):
    """A view that fails to generate is a view nobody can look at, which is the worst moment
    for it to be missing. A malformed record is not enough to trigger this -- `gather` reports
    what it could not read rather than raising -- so the failure is forced here."""
    from autosim.research import run_record

    def broken(root):
        raise RuntimeError("the run root is on a filesystem that went away")

    monkeypatch.setattr(run_record, "gather", broken)
    root = tmp_path / "run"
    root.mkdir()
    assert rp.write(root)
    page = (root / "RUN.html").read_text(encoding="utf-8")
    assert "没能生成" in page and "filesystem that went away" in page


def test_a_malformed_record_still_produces_a_readable_page(tmp_path):
    """The other half: bad JSON is reported in the document, not thrown out of it. The
    unreadable list already says which file, so the page renders everything else."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "events.json").write_text("[[[not json", encoding="utf-8")
    rp.write(root)
    page = (root / "RUN.html").read_text(encoding="utf-8")
    assert "<h2>时间线</h2>" in page and "events" in page


def test_regenerating_the_page_gives_the_same_bytes(tmp_path):
    """Same property as the markdown: a page that differs between two readings of an unchanged
    run cannot be diffed against itself."""
    root = _run(tmp_path)
    rp.write(root)
    first = (root / "RUN.html").read_text(encoding="utf-8")
    rp.write(root)
    assert (root / "RUN.html").read_text(encoding="utf-8") == first


def test_the_converter_is_optional_and_its_absence_is_stated(monkeypatch):
    """`markdown` is the one dependency this module takes and it is not required. When it is
    absent the body is escaped into a `pre` -- degraded formatting, not a lost section, and
    never a page that fails to open."""
    body, converted = rp.to_html("# 标题\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    assert converted and "<table>" in body
    import builtins
    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name == "markdown":
            raise ImportError("no markdown here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without)
    body, converted = rp.to_html("# 标题\n\n<not a tag>")
    assert not converted
    assert body.startswith("<pre>") and "&lt;not a tag&gt;" in body
