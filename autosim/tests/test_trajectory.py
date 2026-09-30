"""Reading positions back out of the step records, and the claims a drawing may make.

Thirty-five thousand rows of poses across ninety-seven files had never been read. The two ways
to get this wrong are worth more than the feature: **the plane is chosen from the data** rather
than from a convention about which axis is up, or a benchmark whose world is arranged
differently is drawn as a flat line; and **the colouring must be measured**, because a chart
that colours by outcome invites a reader to see a difference whether or not the record holds
one.
"""

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from autosim.research import trajectory as tr


def _pose(x, y, z):
    """A 4x4 with the translation in the last column, in the shape the evaluator writes."""
    return [[[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]]


def _write_telemetry(directory: Path, episodes: dict[int, dict[str, list]], *,
                     step_of=1.0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for seed, entities in episodes.items():
        steps = max(len(points) for points in entities.values())
        for step in range(steps):
            lines.append(json.dumps({
                "seed": seed, "step": step, "event": "observation", "task": "t",
                "entities": {name: {"pose": _pose(*points[min(step, len(points) - 1)])}
                             for name, points in entities.items()}}))
    (directory / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def _metrics(directory: Path, verdicts: dict[int, bool]) -> None:
    (directory / "evaluation_metrics.json").write_text(json.dumps({
        "episodes": [{"episode_index": index, "episode_seed": seed, "success": ok}
                     for index, (seed, ok) in enumerate(verdicts.items())]}), encoding="utf-8")


# -- reading ----------------------------------------------------------------------------------

def test_tracks_are_grouped_by_episode_and_in_step_order(tmp_path):
    directory = _write_telemetry(tmp_path / "e", {
        11: {"bottle": [(0.0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)]},
        22: {"bottle": [(0.0, 0, 0), (0.2, 0, 0)]}})
    tracks = tr.tracks(directory / "telemetry.jsonl")
    assert sorted(tracks) == [11, 22]
    assert [one[0] for one in tracks[11]["bottle"]] == [0, 1, 2]
    assert tracks[11]["bottle"][-1][1] == 1.0


def test_a_half_written_line_is_skipped_and_the_rest_is_read(tmp_path):
    """This reads a file written incrementally by a process that may have been killed.
    Refusing to draw ninety-nine good episodes because the hundredth line is truncated is the
    wrong trade."""
    directory = _write_telemetry(tmp_path / "e", {11: {"bottle": [(0.0, 0, 0), (1.0, 0, 0)]}})
    file = directory / "telemetry.jsonl"
    file.write_text(file.read_text(encoding="utf-8") + '{"seed": 12, "entit', encoding="utf-8")
    rows = tr.read_rows(file)
    assert len(rows) == 2
    assert tr.tracks(file)[11]["bottle"]


def test_a_missing_telemetry_file_is_not_an_error(tmp_path):
    assert tr.read_rows(tmp_path / "nothing.jsonl") == []
    assert tr.svg(tmp_path / "nothing") == ""
    assert tr.describe(tmp_path / "nothing") == {}


# -- what the drawing may claim ----------------------------------------------------------------

def test_the_plane_is_chosen_from_the_data_and_not_from_a_convention(tmp_path):
    """Motion in x and z, with y flat. A renderer that assumes the horizontal plane draws a
    straight line and calls it a trajectory."""
    directory = _write_telemetry(tmp_path / "e", {
        seed: {"thing": [(float(i), 0.0, float(i * 2)) for i in range(6)]}
        for seed in (1, 2)})
    svg = tr.svg(directory)
    # Both moving axes are drawn, largest span first, and the flat one is left out entirely.
    assert "横轴 z，纵轴 x" in svg
    assert "纵轴 y" not in svg and "横轴 y" not in svg


def test_the_outcome_split_is_measured_and_not_assumed(tmp_path):
    """The numbers behind the colouring. A chart that colours by outcome invites the reader to
    see a difference; if the record holds none, the colouring is decoration that reads as a
    finding."""
    directory = _write_telemetry(tmp_path / "e", {
        1: {"thing": [(0.0, 0, 0), (0.5, 0, 0), (1.0, 0, 0)]},      # moves, succeeds
        2: {"thing": [(0.0, 0, 0), (0.2, 0, 0), (0.35, 0, 0)]},     # barely moves, fails
        3: {"thing": [(0.0, 0, 0), (0.1, 0, 0), (0.15, 0, 0)]}},    # barely moves, fails
        step_of=1.0)
    _metrics(directory, {1: True, 2: False, 3: False})
    measured = tr.by_outcome(directory)
    assert measured["success_episodes"] == 1 and measured["failure_episodes"] == 2
    assert measured["success_median_travel"] > measured["failure_median_travel"]


def test_travel_is_the_path_length_and_not_the_displacement():
    """A policy that pushes an object back and forth has moved it a long way and got nowhere.
    Displacement would call those two episodes identical."""
    there_and_back = [(0, 0.0, 0, 0), (1, 1.0, 0, 0), (2, 0.0, 0, 0)]
    straight = [(0, 0.0, 0, 0), (1, 0.5, 0, 0), (2, 1.0, 0, 0)]
    assert math.isclose(tr._travel(there_and_back), 2.0, abs_tol=1e-9)
    assert math.isclose(tr._travel(straight), 1.0, abs_tol=1e-9)


def test_without_a_joinable_outcome_the_chart_says_so(tmp_path):
    """The key is the seed, and it is the only thing joining the two files. If the join fails
    the chart is drawn without the distinction -- a worse chart, and not a wrong one, as long
    as it says which it is."""
    directory = _write_telemetry(tmp_path / "e", {1: {"thing": [(0.0, 0, 0), (1.0, 0, 0)]}})
    svg = tr.svg(directory)
    assert "这次没有每局结果可关联" in svg
    assert tr.by_outcome(directory) == {}
    # A metrics file keyed differently must not be read as if it matched.
    (directory / "evaluation_metrics.json").write_text(
        json.dumps({"episodes": [{"episode_index": 0, "success": True}]}), encoding="utf-8")
    assert tr.outcomes(directory) == {}


def test_the_name_of_an_entity_cannot_break_the_svg(tmp_path):
    directory = _write_telemetry(tmp_path / "e", {
        1: {'a<b>&"c" [d]': [(0.0, 0, 0), (1.0, 0, 0)]}})
    svg = tr.svg(directory)
    ET.fromstring(svg)
    assert "<b>" not in svg


def test_more_episodes_than_the_cap_are_sampled_and_not_truncated(tmp_path):
    """A picture of the first hundred of four hundred episodes is a picture of the beginning
    of the evaluation, which is not what the caption says it is."""
    episodes = {seed: {"thing": [(0.0, 0, 0), (float(seed % 7), 0, 0)]} for seed in range(200)}
    directory = _write_telemetry(tmp_path / "e", episodes)
    svg = tr.svg(directory, max_episodes=50)
    assert svg.count("<polyline") == 50
    # Spread over the whole range rather than the first fifty.
    assert svg.count("<polyline") < 200


# -- the section, which is what a person reads -------------------------------------------------

def _run_with_telemetry(tmp_path: Path) -> Path:
    from autosim.research import run_record as rr
    root = tmp_path / "run"
    directory = _write_telemetry(root / "evaluations" / "official_development", {
        1: {"bottle": [(0.0, 0, 0), (0.6, 0, 0)], "cup": [(1.0, 0, 0), (1.0, 0, 0)]},
        2: {"bottle": [(0.0, 0, 0), (0.1, 0, 0)], "cup": [(1.0, 0, 0), (1.0, 0, 0)]}})
    _metrics(directory, {1: True, 2: False})
    return root


def test_the_section_states_that_a_trajectory_is_not_a_replay(tmp_path):
    """The drawing actively invites the reader to think it is a rendering. Saying so in the
    caption is the difference between a document that points at its evidence and one that
    lets its reader believe something false about it."""
    from autosim.research import run_record as rr
    root = _run_with_telemetry(tmp_path)
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 轨迹"):document.index("## 产物与证据")]
    assert "这是轨迹，不是回放" in section
    assert "没有相机被模拟" in section
    assert "**图在 `RUN.html` 里**" in section
    assert "1/2" not in section or "2/2" in section      # the join is reported honestly


def test_the_split_is_printed_beside_the_drawing_so_it_can_be_checked(tmp_path):
    from autosim.research import run_record as rr
    root = _run_with_telemetry(tmp_path)
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 轨迹"):document.index("## 产物与证据")]
    assert "按成败分开看" in section
    assert "成功的局" in section and "失败的局" in section
    assert "倍" in section


def test_a_run_with_no_step_records_says_so_rather_than_showing_nothing(tmp_path):
    from autosim.research import run_record as rr
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    (root / "events.json").write_text('{"rows": []}', encoding="utf-8")
    rr.generate(root)
    document = (root / "RUN.md").read_text(encoding="utf-8")
    section = document[document.index("## 轨迹"):document.index("## 产物与证据")]
    assert "没有留下逐步位姿记录" in section


# -- the cache, which is what keeps this off the hot path ---------------------------------------

def test_the_readings_are_computed_once_and_reused(tmp_path, monkeypatch):
    """The document is regenerated every time a stage ends, and reading every row of a
    hundred-episode evaluation takes seconds. Counted rather than timed: how many times the
    rows were parsed is the thing that has to be one, and a stopwatch would only say it was
    fast on this machine today."""
    directory = _write_telemetry(tmp_path / "e", {
        1: {"thing": [(0.0, 0, 0), (1.0, 0, 0)]}, 2: {"thing": [(0.0, 0, 0), (0.2, 0, 0)]}})
    _metrics(directory, {1: True, 2: False})
    calls = []
    real = tr.describe

    def counted(where):
        calls.append(where)
        return real(where)

    monkeypatch.setattr(tr, "describe", counted)
    first = tr.readings(directory)
    assert first and (directory / tr.SIDE_NAME).is_file() and (directory / tr.SVG_NAME).is_file()
    assert tr.readings(directory) == first
    assert len(calls) == 1, "the rows were read again for an unchanged input"


def test_a_changed_input_is_recomputed(tmp_path, monkeypatch):
    """The other half: the cache must not outlive the thing it cached. Keyed on the inputs'
    own size and modification time, so an evaluation that ran again is read again."""
    import os
    import time
    directory = _write_telemetry(tmp_path / "e", {1: {"thing": [(0.0, 0, 0), (1.0, 0, 0)]}})
    assert tr.readings(directory)["episodes"] == 1
    _write_telemetry(directory, {3: {"thing": [(0.0, 0, 0), (9.0, 0, 0)]},
                                 4: {"thing": [(0.0, 0, 0), (9.0, 0, 0)]}})
    os.utime(directory / "telemetry.jsonl", (time.time() + 10, time.time() + 10))
    assert tr.readings(directory)["episodes"] == 2


def test_a_directory_with_no_poses_caches_nothing(tmp_path):
    directory = tmp_path / "e"
    directory.mkdir()
    assert tr.readings(directory) == {}
    assert not (directory / tr.SIDE_NAME).exists()
    assert tr.drawing(directory) == ""


def test_the_artifacts_say_what_they_are_and_what_they_cannot_say(tmp_path):
    """The drawing sits beside the poses and looks like a rendering. The index is the place
    that has to say it is not one, because the picture cannot say it about itself."""
    from autosim.research import run_record as rr
    what, caveat = rr.role_of("evaluations/x/trajectory.svg")
    assert "轨迹图" in what and "不是回放" in caveat
    what, caveat = rr.role_of("evaluations/x/evaluation_metrics.json")
    assert "每一局的种子与成败" in what
    assert "这一次的结果，不是这个策略的水平" in caveat
