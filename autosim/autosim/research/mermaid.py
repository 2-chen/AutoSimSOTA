"""Two diagrams, rendered from the run's own records into text a markdown reader draws.

A Gantt chart of what ran and for how long, and a provenance graph of what was decided on what
and what it produced. Both are Mermaid, which means zero dependencies and a diagram that renders
wherever the document does -- and it means something else that matters more: **a diagram that is
wrong renders as a picture that looks authoritative.** A broken one renders as an error box and
is therefore safe; a plausible one built from a misread record is not.

So these are pure functions of the records, they invent no edges, and where an edge or a bar
would have to be guessed they are left out and the text says what was left out. A node whose
source is missing from the record is drawn as missing rather than dropped -- a decision that
cites evidence nobody can find is a finding, not a formatting problem.

**Escaping is the whole difficulty of this module.** Mermaid parses its own syntax out of the
labels, and this system's labels are paths, argv and prose in two languages. A colon in a task
name ends the name; a bracket ends a node; a quote ends a label. Every string that reaches the
output goes through `_safe`, and a test asserts the result contains nothing that would break the
parse -- because the failure mode is not an exception, it is a diagram.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Iterable

#: Mermaid reads these out of a label and ends part of its own syntax on them. The colon is
#: listed first because it is the one this system's own data contains constantly -- every
#: `cuda:0`, every `train.n_epochs=1` in a task name.
_UNSAFE = ("\n", "\r", '"', "'", ":", ";", ",", "[", "]", "{", "}", "(", ")",
           "|", "<", ">", "&", "#", "`", "%", "\\")


def _safe(text: Any, limit: int = 60) -> str:
    """A string that cannot end a Mermaid label early, or change its meaning.

    Substitution rather than removal, so that two different values do not become one label:
    `cuda:0` and `cuda 0` are different devices and a diagram that prints both as the same
    word is worse than one that prints neither.
    """
    out = str(text or "").replace(":", "/").replace(",", " ").replace("|", "/")
    for one in _UNSAFE:
        out = out.replace(one, " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out[:limit] + ("…" if len(out) > limit else "")


def _moment(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _duration(seconds: Any) -> str:
    """Mermaid wants a unit. Sub-minute work is drawn as seconds; anything else rounds up to
    the next minute, because a bar that cannot be seen is not a bar."""
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "1s"
    if value < 60:
        return f"{max(1, int(round(value)))}s"
    return f"{max(1, int(round(value / 60.0)))}m"


def gantt(rows: Iterable[dict[str, Any]], *, title: str = "运行时间线",
          sections: Iterable[str] = ()) -> str:
    """When each thing ran and how long it took, from the timestamps the system wrote.

    Rows without a usable start time are skipped rather than placed arbitrarily: a Gantt bar's
    position *is* its claim, and a bar drawn at the wrong time is a false statement that looks
    like a measurement.
    """
    placed: list[tuple[datetime, str, float, str]] = []
    for row in rows:
        started = _moment(row.get("at") or row.get("started_at"))
        if started is None:
            continue
        try:
            seconds = float(row.get("seconds") or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        label = _safe(row.get("label") or row.get("stage") or row.get("event") or "?", 48)
        group = _safe(row.get("section") or row.get("event") or "run", 24)
        placed.append((started, label, seconds, group))
    if not placed:
        return ""
    placed.sort(key=lambda one: one[0])
    # Mermaid's Gantt axis is one clock for the whole chart: the section order is the order the
    # bars are written, and a bar whose section is repeated further down is drawn where it
    # belongs rather than where it is written. So the sections are emitted in first-start order.
    order: list[str] = []
    for _, _, _, group in placed:
        if group not in order:
            order.append(group)
    lines = ["```mermaid", "gantt", f"    title {_safe(title, 60)}",
             "    dateFormat YYYY-MM-DDTHH:mm:ss", "    axisFormat %H:%M", "    tickInterval 1h"]
    for group in order:
        lines.append(f"    section {group}")
        for started, label, seconds, own in placed:
            if own != group:
                continue
            # Normalised to UTC before formatting: the axis is one clock, and records written
            # in different offsets would otherwise be drawn in the order they were read rather
            # than the order they happened.
            stamp = started.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
            lines.append(f"    {label} :{stamp}, {_duration(seconds)}")
    lines.append("```")
    return "\n".join(lines)


def provenance(decisions: Iterable[dict[str, Any]], *, title: str = "决策溯源",
               limit: int = 30) -> str:
    """What each decision was made on, and what it produced.

    The graph is built only from the edges the record holds -- `used` and `produced` on each
    decision. Nothing is inferred: a decision whose evidence was never written down has no
    incoming edge, and that absence is the point rather than a gap to fill in.
    """
    rows = [row for row in decisions if isinstance(row, dict)
            and row.get("kind") == "decision"][-limit:]
    if not rows:
        return ""
    lines = ["```mermaid", "flowchart LR"]
    declared: dict[str, str] = {}
    edges: list[tuple[str, str]] = []

    def node(key: str, label: str, shape: str = "rect") -> str:
        # The id comes from the key rather than from the position, so two renderings of an
        # unchanged run produce the same graph. A diagram whose nodes move when nothing moved
        # cannot be compared with itself, which is most of what a diagram is for.
        name = "n" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        if name not in declared:
            if shape == "round":
                lines.append(f'    {name}(["{label}"])')
            else:
                lines.append(f'    {name}["{label}"]')
            declared[name] = key
        return name

    for index, row in enumerate(rows):
        decision = node(f"d{index}:{row.get('activity', '')}",
                        f"{row.get('by', '?')}: {_safe(row.get('activity'), 44)}")
        for used in row.get("used") or []:
            edges.append((node(f"u{used}", _safe(used, 40), "round"), decision))
        for produced in row.get("produced") or []:
            edges.append((decision, node(f"p{produced}", _safe(produced, 40))))
        outcome = row.get("outcome") or {}
        if outcome.get("state") == "open":
            edges.append((decision, node("open", "结果未回填", "round")))
    for left, right in edges:
        lines.append(f"    {left} --> {right}")
    lines.append("```")
    return "\n".join(lines)
