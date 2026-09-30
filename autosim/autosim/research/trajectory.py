"""What moved where, from the positions the evaluator recorded as it went.

`telemetry.jsonl` holds a 4x4 pose per entity per step per episode -- ninety-seven files of it
on this machine, thirty-five thousand rows -- and nothing had ever read one back. It is the
only record this system keeps from which a scene can be rebuilt off-screen, and for a simulator
benchmark that is the most on-point thing it has: a number says whether the policy succeeded,
and an episode says how.

**What this draws is a trajectory, not a replay.** The recorded poses are projected onto the
plane and drawn as paths; nothing is rendered, and no camera is simulated. So it shows where
things were and not what anything looked like, and the captions say so. Reading the output as a
rendering would be a mistake the picture actively invites, which is the reason to be blunt
about it in the text rather than only in this docstring.

The outcome join is the part that makes it worth drawing: `evaluation_metrics.json` keys each
episode by the same seed the telemetry rows carry, so every path can be coloured by whether
that episode succeeded. Paths that end near the target in the failures and paths that end on it
in the successes are two different findings, and neither is visible in the summary rate.

Entities are chosen by what the record shows rather than by what this module expects: every
entity that appears with a pose over the episode is drawn, in its own panel. Which object
matters is a question about the task, and this module does not know any tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

#: Rows read per file. A file is one evaluation; the largest here is thirty-five hundred rows
#: for a hundred episodes, so this bounds a pathological case without touching a real one.
ROW_LIMIT = 200_000

WIDTH, HEIGHT, PAD = 460, 380, 44
SUCCESS, FAILURE = "#2f7f4f", "#b04a4a"


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The telemetry file as rows, skipping anything that is not one.

    A malformed line is skipped rather than raised on: this reads a file written incrementally
    by a process that may have been killed, and refusing to draw ninety-nine good episodes
    because the hundredth line was half-written is the wrong trade.
    """
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and isinstance(row.get("entities"), dict):
            rows.append(row)
        if len(rows) >= ROW_LIMIT:
            break
    return rows


def _position(pose: Any) -> tuple[float, float, float] | None:
    """The translation out of a 4x4, given as a list of rows or a list of one such matrix."""
    matrix = pose
    while isinstance(matrix, list) and len(matrix) == 1 and isinstance(matrix[0], list):
        matrix = matrix[0]
    if not isinstance(matrix, list) or len(matrix) < 3:
        return None
    try:
        return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))
    except (IndexError, TypeError, ValueError):
        return None


def tracks(path: Path) -> dict[int, dict[str, list[tuple[int, float, float, float]]]]:
    """Per episode, per entity, the positions in step order."""
    out: dict[int, dict[str, list[tuple[int, float, float, float]]]] = {}
    for row in read_rows(path):
        seed = row.get("seed")
        if not isinstance(seed, int):
            continue
        step = row.get("step")
        step = step if isinstance(step, int) else len(out.get(seed, {}))
        episode = out.setdefault(seed, {})
        for name, value in row["entities"].items():
            point = _position((value or {}).get("pose"))
            if point is None:
                continue
            episode.setdefault(str(name), []).append((step, *point))
    for episode in out.values():
        for points in episode.values():
            points.sort(key=lambda one: one[0])
    return out


def outcomes(directory: Path) -> dict[int, bool]:
    """Which episodes succeeded, keyed by the seed the telemetry rows also carry.

    The key is the only thing that joins the two files. If a metrics file keys its episodes
    differently, this returns nothing and the chart is drawn without the distinction -- which
    is a worse chart and not a wrong one.
    """
    found: dict[int, bool] = {}
    metrics = Path(directory) / "evaluation_metrics.json"
    try:
        document = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return found
    for episode in document.get("episodes") or []:
        if isinstance(episode, dict) and isinstance(episode.get("episode_seed"), int):
            found[episode["episode_seed"]] = bool(episode.get("success"))
    return found


def _panel(name: str, episodes: dict[int, dict[str, list]], verdicts: dict[int, bool],
           plane: tuple[int, int]) -> str:
    """One entity, every episode's path, in the plane the record moved in most."""
    axis_a, axis_b = plane
    points = [(one[1 + axis_a], one[1 + axis_b])
              for episode in episodes.values() for one in episode.get(name, [])]
    if not points:
        return ""
    xs = [one[0] for one in points]
    ys = [one[1] for one in points]
    span_x = max(max(xs) - min(xs), 1e-6)
    span_y = max(max(ys) - min(ys), 1e-6)
    inner_w, inner_h = WIDTH - 2 * PAD, HEIGHT - 2 * PAD
    # Equal scale on both axes, because a trajectory whose shape depends on the aspect ratio
    # of the box it was drawn in is a picture of the box and not of the motion.
    scale = min(inner_w / span_x, inner_h / span_y)
    offset_x = PAD + (inner_w - span_x * scale) / 2 - min(xs) * scale
    offset_y = PAD + (inner_h - span_y * scale) / 2 - min(ys) * scale

    def place_at(x: float, y: float) -> tuple[float, float]:
        return x * scale + offset_x, (HEIGHT - PAD) - (y - min(ys)) * scale

    with_outcome = any(seed in verdicts for seed in episodes)
    parts = [f'<text x="0" y="12" font-size="12" font-weight="600">{_esc(name)}'
             f'{" （绿=成功，红=失败）" if with_outcome else " （这次没有每局结果可关联）"}'
             f'</text>',
             f'<rect x="{PAD}" y="{PAD}" width="{inner_w}" height="{inner_h}" fill="#fcfcfd" '
             f'stroke="#e4e4e8"/>']
    for seed, episode in sorted(episodes.items()):
        series = episode.get(name) or []
        if len(series) < 2:
            continue
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in
                        (place_at(one[1 + axis_a], one[1 + axis_b]) for one in series))
        if seed in verdicts:
            colour = SUCCESS if verdicts[seed] else FAILURE
            opacity = "0.75"
        else:
            colour, opacity = "#5577aa", "0.5"
        parts.append(f'<polyline points="{path}" fill="none" stroke="{colour}" '
                     f'stroke-width="1.3" opacity="{opacity}"/>')
        start = place_at(series[0][1 + axis_a], series[0][1 + axis_b])
        parts.append(f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="2.4" fill="none" '
                     f'stroke="{colour}" stroke-width="1.2" opacity="{opacity}"/>')
    parts.append(f'<text x="{PAD}" y="{HEIGHT - 10}" font-size="10" fill="#777">'
                 f'横轴 {_AXES[axis_a]}，纵轴 {_AXES[axis_b]}；起始位置画成空心圈</text>')
    return "".join(parts)


_AXES = ("x", "y", "z")


def _esc(value: Any) -> str:
    import html
    return html.escape(str(value if value is not None else ""), quote=True)


def svg(directory: Path, *, max_entities: int = 3, max_episodes: int = 120) -> str:
    """Every entity the evaluation recorded a pose for, as paths in the plane it moved in.

    The plane is chosen from the data: the two axes with the largest spread across all
    episodes. A bottle carried across a table moves in the horizontal plane exactly as often
    as it is lifted, and picking the plane by the record rather than by a convention about
    which axis is "up" is what keeps this working on a benchmark whose world is arranged
    differently from the last one.
    """
    directory = Path(directory)
    file = directory / "telemetry.jsonl"
    if not file.is_file():
        return ""
    episodes = tracks(file)
    if not episodes:
        return ""
    if len(episodes) > max_episodes:
        # Sampled evenly rather than truncated, so the picture is of the whole evaluation and
        # not of its first hundred episodes. The caption says how many were left out.
        keep = sorted(episodes)[:: max(1, len(episodes) // max_episodes)][:max_episodes]
        episodes = {seed: episodes[seed] for seed in keep}
    names: dict[str, float] = {}
    for episode in episodes.values():
        for name, points in episode.items():
            xs = [one[1] for one in points]
            ys = [one[2] for one in points]
            zs = [one[3] for one in points]
            names[name] = names.get(name, 0.0) + max(
                max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    ordered = [name for name, _ in sorted(names.items(), key=lambda kv: -kv[1])][:max_entities]
    # Which two axes to draw, from the record rather than from a convention about which one is
    # "up". A benchmark whose world is arranged differently would otherwise be drawn as a line.
    spans = [0.0, 0.0, 0.0]
    for episode in episodes.values():
        for name in ordered:
            for index in range(3):
                values = [one[1 + index] for one in episode.get(name, [])]
                if values:
                    spans[index] = max(spans[index], max(values) - min(values))
    board = sorted(range(3), key=lambda index: -spans[index])
    plane = (board[0], board[1])
    verdicts = outcomes(directory)

    panels = [name for name in ordered
              if any(episode.get(name) for episode in episodes.values())]
    width = WIDTH * max(1, len(panels))
    parts = [f'<svg viewBox="0 0 {width} {HEIGHT}" width="{width}" height="{HEIGHT}" '
             f'role="img" aria-label="轨迹俯视图">',
             f'<rect width="{width}" height="{HEIGHT}" fill="#fff"/>']
    for index, name in enumerate(panels):
        parts.append(f'<g transform="translate({index * WIDTH},0)">'
                     f'{_panel(name, episodes, verdicts, plane)}</g>')
    parts.append("</svg>")
    return "".join(parts)


def _travel(points: Iterable[tuple]) -> float:
    """How far the entity went, summed over its steps -- the path length, not the displacement.

    A policy that pushes an object back and forth has moved it a long way and got nowhere, and
    the two numbers differ; this is the one that says how much was attempted.
    """
    series = list(points)
    total = 0.0
    for before, after in zip(series, series[1:]):
        total += sum((after[index] - before[index]) ** 2 for index in (1, 2, 3)) ** 0.5
    return total


def by_outcome(directory: Path) -> dict[str, Any]:
    """How far the things that move went, split by whether the episode succeeded.

    The numbers behind the colouring. A chart that colours by outcome invites the reader to
    see a difference, and if the record does not hold one then the colouring is decoration
    that reads as a finding -- so the difference is measured here and printed beside the
    drawing, where it can be checked instead of believed.
    """
    directory = Path(directory)
    episodes = tracks(directory / "telemetry.jsonl")
    verdicts = outcomes(directory)
    if not episodes or not verdicts:
        return {}
    moved: dict[str, float] = {}
    for episode in episodes.values():
        for name, points in episode.items():
            moved[name] = moved.get(name, 0.0) + _travel(points)
    if not moved:
        return {}
    primary = max(moved, key=lambda name: moved[name])
    travel = {"success": [], "failure": []}
    apart = {"success": [], "failure": []}
    for seed, episode in episodes.items():
        if seed not in verdicts or not episode.get(primary):
            continue
        side = "success" if verdicts[seed] else "failure"
        travel[side].append(_travel(episode[primary]))
        others = [name for name in episode if name != primary]
        if others:
            other = max(others, key=lambda name: moved.get(name, 0.0))
            here, there = episode[primary][-1], episode[other][-1]
            apart[side].append(sum((here[i] - there[i]) ** 2 for i in (1, 2, 3)) ** 0.5)
    out: dict[str, Any] = {"primary": primary}
    for side in ("success", "failure"):
        if travel[side]:
            out[f"{side}_episodes"] = len(travel[side])
            out[f"{side}_median_travel"] = round(median(travel[side]), 4)
        if apart[side]:
            out[f"{side}_median_final_separation"] = round(median(apart[side]), 4)
    return out


def describe(directory: Path) -> dict[str, Any]:
    """What the telemetry says, in numbers, for the section that has no room for a drawing."""
    directory = Path(directory)
    file = directory / "telemetry.jsonl"
    if not file.is_file():
        return {}
    episodes = tracks(file)
    if not episodes:
        return {}
    names: dict[str, int] = {}
    for episode in episodes.values():
        for name, points in episode.items():
            names[name] = names.get(name, 0) + len(points)
    verdicts = outcomes(directory)
    return {"file": str(file), "episodes": len(episodes), "entities": sorted(names),
            "rows": sum(names.values()),
            "joinable_outcomes": len([seed for seed in episodes if seed in verdicts]),
            "by_outcome": by_outcome(directory)}


def directories_under(root: Path) -> list[Path]:
    """Every directory beneath `root` that holds a telemetry file, in path order."""
    root = Path(root)
    return sorted({path.parent for path in root.rglob("telemetry.jsonl")})


#: The drawing and what was computed from the poses, written beside the records they came from.
#: The names are deliberately the plain ones: a person who finds a directory of poses should
#: find the picture of them next to it without knowing this module exists.
SVG_NAME = "trajectory.svg"
SIDE_NAME = "trajectory.json"


def _freshness(directory: Path) -> tuple[float, int]:
    """What the drawing depends on, as a value: the inputs' newest mtime and total size.

    Not a hash of the contents. The question is only whether the drawing is out of date, and
    this answers it by reading two `stat` results instead of re-reading twenty-five thousand
    rows -- which matters because the document that embeds the drawing is regenerated every
    time a stage ends.
    """
    newest, total = 0.0, 0
    for name in ("telemetry.jsonl", "evaluation_metrics.json"):
        path = directory / name
        try:
            info = path.stat()
        except OSError:
            continue
        newest = max(newest, info.st_mtime)
        total += info.st_size
    return newest, total


def readings(directory: Path) -> dict[str, Any]:
    """What the poses say, computed once and reused until the poses change.

    Both the drawing and the section that describes it come from reading every row of the
    telemetry. That is seconds per file on a hundred-episode evaluation, and the document is
    regenerated at every stage boundary -- so the answer is kept beside the records it was
    computed from, keyed by the inputs' own size and modification time. The key is not a hash:
    the only question is whether this is out of date, and `stat` answers it.
    """
    directory = Path(directory)
    if not (directory / "telemetry.jsonl").is_file():
        return {}
    stamp = _freshness(directory)
    sidecar = _load_json(directory / SIDE_NAME)
    if isinstance(sidecar, dict) and tuple(sidecar.get("inputs") or ()) == stamp:
        return sidecar.get("description") or {}
    description = describe(directory)
    drawing = svg(directory) if description else ""
    if drawing:
        (directory / SVG_NAME).write_text(drawing, encoding="utf-8")
    _write_json(directory / SIDE_NAME, {"inputs": list(stamp), "description": description,
                                        "drawn_bytes": len(drawing)})
    return description


def drawing(directory: Path) -> str:
    """The drawing, from beside the records if it is current."""
    directory = Path(directory)
    readings(directory)                     # refreshes the drawing when its inputs changed
    path = directory / SVG_NAME
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    try:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
