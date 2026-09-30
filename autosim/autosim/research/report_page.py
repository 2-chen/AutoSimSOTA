"""The run as one file a person can open, forward, and read without a network.

`RUN.md` is the record; this is the same record as a single HTML file with no external
requests: no stylesheet link, no script tag, no font, no image. It opens from a filesystem, it
survives being emailed, and it renders the same in five years because it depends on nothing
that can move.

That constraint is what rules out Mermaid here, and the diagrams are redrawn as inline SVG
instead. Mermaid is JavaScript, and a document that needs a script to show its chart is a
document that shows nothing in a mail client. The SVG is generated from the same records the
markdown diagrams are, by the same rule: **a chart that is wrong looks authoritative in a way
a broken one does not**, so nothing is drawn that the record does not support, and a series
with no usable values is left out rather than plotted at the origin.

One dependency is accepted and only one: `markdown`, to turn the section bodies into HTML. When
it is absent the sections are shown verbatim in a `<pre>` and the page says so at the top --
degraded formatting is a smaller problem than a page that silently loses its structure, and
much smaller than a page that fails to open.

The legacy detailed page draws homogeneous per-round metrics without a plotting dependency.
The live top-level page instead mirrors the same Markdown publication and adjacent plots.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

#: Chart geometry. Fixed rather than responsive: a document that reflows with the window is a
#: document whose charts cannot be described in words to someone who is not looking at it.
WIDTH, ROW, LEFT, RIGHT, TOP = 940, 24, 260, 24, 34

#: Colours by position in a palette, so two charts of the same run use the same ones and a
#: reader who has seen one has seen the other.
PALETTE = ("#2f6f9f", "#4f9f6f", "#9f6f2f", "#7f4f9f", "#9f4f4f", "#4f7f9f", "#6f9f4f")

_CSS = """
:root { color-scheme: light dark; }
body { margin: 0 auto; max-width: 1040px; padding: 28px 20px 80px;
       font: 15px/1.65 -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
       color: #1c1c1e; background: #fff; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 34px 0 6px; padding-bottom: 4px; border-bottom: 1px solid #ddd; }
h3 { font-size: 16px; margin: 20px 0 4px; }
code { font: 13px/1.5 ui-monospace, "SF Mono", Menlo, monospace;
       background: #f4f4f6; padding: 1px 4px; border-radius: 3px; }
pre { background: #f7f7f9; padding: 10px 12px; border-radius: 5px; overflow-x: auto;
      border-left: 3px solid #ccc; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13.5px; }
th, td { border: 1px solid #e2e2e4; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f6f6f8; font-weight: 600; }
blockquote { margin: 8px 0; padding: 6px 12px; border-left: 3px solid #c9c9cf;
             color: #555; background: #fafafb; }
.log { color: #666; font-size: 13.5px; margin: -2px 0 10px; }
.prov { font-size: 12.5px; font-weight: 600; padding: 2px 7px; border-radius: 9px;
        margin-right: 6px; letter-spacing: .02em; }
.computed { background: #e6f0e8; color: #1f5b32; }
.written { background: #fdf0e0; color: #7a4a12; }
.lead { color: #555; font-size: 13.5px; margin: 2px 0 0; }
.warn { border: 1px solid #e0c08a; background: #fdf8ee; padding: 10px 12px;
        border-radius: 5px; margin: 14px 0; }
ul { margin: 6px 0; padding-left: 22px; }
svg { max-width: 100%; height: auto; display: block; margin: 10px 0; }
.svgcaption { color: #666; font-size: 13px; margin: 0 0 14px; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _clip(value: Any, limit: int = 46) -> str:
    """A label that fits a box. Truncation is visible (`…`) rather than silent, because a
    chart label that has quietly lost its qualifier is a different claim from the one it was
    drawn from."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


_cdatastyle = re.compile(r"^```(\w*)\n(.*?)^```$", re.S | re.M)


def to_html(body: str) -> tuple[str, bool]:
    """The section body as HTML, and whether a converter was available to do it.

    The markdown library is the one dependency this module takes. It is optional: when it is
    missing the body is escaped into a `pre` and the caller is told, so the page says its
    formatting was degraded rather than quietly presenting a different structure.
    """
    try:
        import markdown
    except ImportError:
        return f"<pre>{_esc(body)}</pre>", False
    return markdown.markdown(body, extensions=("tables", "fenced_code", "sane_lists")), True


# -- charts, as SVG ------------------------------------------------------------------------

def _moment(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _bars(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        started = _moment(row.get("at") or row.get("started_at"))
        if started is None:
            continue
        try:
            seconds = max(0.0, float(row.get("seconds") or 0))
        except (TypeError, ValueError):
            seconds = 0.0
        out.append({"at": started, "seconds": seconds,
                    "label": _clip(row.get("label") or row.get("stage") or "?", 38),
                    "section": str(row.get("section") or "run")})
    return sorted(out, key=lambda one: one["at"])


def gantt_svg(rows: Iterable[dict[str, Any]], *, title: str = "") -> str:
    """Each recorded command as a bar, positioned by its own timestamps.

    Nothing is drawn for a row without a start time. A bar's position is its claim, and one
    placed by a guess is a false statement that reads as a measurement.
    """
    bars = _bars(rows)
    if not bars:
        return ""
    spans = []
    for bar in bars:
        spans.append((bar["at"], 0.0))
        spans.append((bar["at"], bar["seconds"]))
    origin = min(one[0] for one in spans)
    span = max((start - origin).total_seconds() + seconds for start, seconds in spans) or 1.0
    # A bar shorter than the eye can find is drawn at a floor, and the axis carries the real
    # numbers, so the floor cannot be mistaken for the duration.
    inner = WIDTH - LEFT - RIGHT
    height = TOP + ROW * len(bars) + 26

    parts = [f'<svg viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" '
             f'role="img" aria-label="{_esc(title or "timeline")}">',
             f'<rect width="{WIDTH}" height="{height}" fill="#fff"/>']
    if title:
        parts.append(f'<text x="0" y="14" font-size="13" font-weight="600">'
                     f'{_esc(title)}</text>')
    for index, (start, seconds) in enumerate(spans):
        if index % 2:
            continue
        x = LEFT + (start - origin).total_seconds() / span * inner
        parts.append(f'<line x1="{x:.1f}" y1="{TOP - 6}" x2="{x:.1f}" '
                     f'y2="{height - 20}" stroke="#eee"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" font-size="10" fill="#888" '
                     f'text-anchor="middle">{start.strftime("%H:%M")}</text>')
    for index, bar in enumerate(bars):
        y = TOP + index * ROW
        x = LEFT + (bar["at"] - origin).total_seconds() / span * inner
        width = max(2.0, bar["seconds"] / span * inner)
        colour = PALETTE[index % len(PALETTE)]
        parts.append(f'<text x="{LEFT - 8}" y="{y + 13}" font-size="11.5" fill="#333" '
                     f'text-anchor="end">{_esc(bar["label"])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y + 3}" width="{width:.1f}" height="{ROW - 8}" '
                     f'rx="2" fill="{colour}" opacity="0.85"><title>'
                     f'{_esc(bar["label"])} — {bar["at"].isoformat()} '
                     f'({bar["seconds"]:.0f}s)</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def provenance_svg(decisions: Iterable[dict[str, Any]]) -> str:
    """What each decision was made on and what it produced, laid out in three columns.

    Only the edges the record holds. A decision with no recorded grounds has no incoming edge,
    and the space where one would be is left empty -- that absence is the finding.
    """
    rows = [row for row in decisions if isinstance(row, dict) and row.get("kind") == "decision"]
    if not rows:
        return ""
    used, produced = [], []
    for row in rows:
        for one in row.get("used") or []:
            if one not in used:
                used.append(one)
        for one in row.get("produced") or []:
            if one not in produced:
                produced.append(one)
    if not used and not produced:
        return ""

    box_w, box_h, gap = 224, 26, 8
    width = WIDTH
    # Three columns: what a decision was made on, the decisions, what they produced. Laid out
    # in that order because that is the direction the record's own relations point -- `used`,
    # then the act, then `produced`.
    columns = [("依据", [(f"used:{one}", one) for one in used], 0),
               ("决定", [(f"decision:{index}", _decision_label(row))
                         for index, row in enumerate(rows)], 1),
               ("产物", [(f"produced:{one}", one) for one in produced], 2)]
    lanes = max(len(items) for _, items, _ in columns)
    height = 26 + lanes * (box_h + gap)
    xs = (0, (width - box_w) / 2, width - box_w)
    spot: dict[str, tuple[float, float]] = {}
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-label="provenance 决策溯源图">',
             f'<rect width="{width}" height="{height}" fill="#fff"/>']
    for heading, items, column in columns:
        parts.append(f'<text x="{xs[column]}" y="12" font-size="12" font-weight="600" '
                     f'fill="#555">{_esc(heading)}</text>')
        for index, (key, _) in enumerate(items):
            spot[key] = (xs[column], 26 + index * (box_h + gap))

    for index, row in enumerate(rows):
        _, y = spot[f"decision:{index}"]
        mid = y + box_h / 2
        for one in row.get("used") or []:
            x, sy = spot.get(f"used:{one}", (0.0, mid))
            # From the right edge of the source box to the left edge of the decision box, so
            # the line visibly attaches to both rather than passing under them.
            parts.append(f'<line x1="{x + box_w:.1f}" y1="{sy + box_h / 2:.1f}" '
                         f'x2="{xs[1]:.1f}" y2="{mid:.1f}" stroke="#c8c8cc" '
                         f'stroke-width="1.2"/>')
        for one in row.get("produced") or []:
            ex, ey = spot.get(f"produced:{one}", (0.0, mid))
            parts.append(f'<line x1="{xs[1] + box_w:.1f}" y1="{mid:.1f}" '
                         f'x2="{ex:.1f}" y2="{ey + box_h / 2:.1f}" stroke="#c8c8cc" '
                         f'stroke-width="1.2"/>')

    for heading, items, column in columns:
        fill = ("#f2f6f9", "#eef4ee", "#f8f4ec")[column]
        stroke = ("#9db8cc", "#9dc0a6", "#ccb98d")[column]
        for key, label in items:
            x, y = spot[key]
            shown = label.split(":", 1)[-1] if label.startswith(("evidence:", "round")) else label
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_w}" height="{box_h}" '
                         f'rx="4" fill="{fill}" stroke="{stroke}">'
                         f'<title>{_esc(label)}</title></rect>')
            parts.append(f'<text x="{x + 8:.1f}" y="{y + 17:.1f}" font-size="11.5">'
                         f'{_esc(_clip(shown, 34))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _decision_label(row: dict[str, Any]) -> str:
    """A decision box, labelled with who decided and what they decided -- not with the
    outcome, which is the other half of the record and has its own place in the document."""
    return f"{row.get('by', '?')}：{_clip(row.get('activity'), 30)}"


def curve_svg(points: Iterable[tuple[str, float | None]], *,
              title: str = "开发指标（点为实测，线为运行顺序）") -> str:
    """One homogeneous metric in recorded order, including negative/minimized metrics.

    Only points that have a number. A round that produced none is left out of the line and
    named in the caption instead of being plotted at zero -- zero is a measurement and a
    missing number is not.
    """
    rated = [(label, float(value)) for label, value in points if value is not None]
    if len(rated) < 1:
        return ""
    width, height = WIDTH, 200
    left, right, top, bottom = 46, 16, 26, 34
    inner_w, inner_h = width - left - right, height - top - bottom
    count = max(len(rated), 2)
    top_value = max(0.0, max(value for _, value in rated))
    bottom_value = min(0.0, min(value for _, value in rated))
    span = top_value - bottom_value or 1.0

    def place(index: int, value: float) -> tuple[float, float]:
        x = left + inner_w * index / (count - 1)
        y = top + inner_h * (1 - (value - bottom_value) / span)
        return x, y

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-label="{_esc(title)}">',
             f'<rect width="{width}" height="{height}" fill="#fff"/>',
             f'<text x="0" y="13" font-size="13" font-weight="600">{_esc(title)}</text>']
    for step in range(5):
        value = bottom_value + span * step / 4
        _, y = place(0, value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
                     f'stroke="#eee"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="10" fill="#888" '
                     f'text-anchor="end">{value:.2f}</text>')
    best_line = []
    for index, (_, value) in enumerate(rated):
        best_line.append(place(index, value))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in best_line)
    parts.append(f'<polyline points="{line}" fill="none" stroke="#2f6f9f" '
                 f'stroke-width="2.2"/>')
    for index, (label, value) in enumerate(rated):
        x, y = place(index, value)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2f6f9f">'
                     f'<title>{_esc(label)} — {value:.3f}</title></circle>')
        parts.append(f'<text x="{x:.1f}" y="{height - 16}" font-size="10.5" fill="#666" '
                     f'text-anchor="middle">{_esc(_clip(label, 18))}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" font-size="10.5" fill="#2f6f9f" '
                     f'text-anchor="middle">{value:.3f}</text>')
    parts.append("</svg>")
    return "".join(parts)


# -- the page ------------------------------------------------------------------------------

def trajectory_for(root: Path) -> str:
    """The one evaluation worth drawing, and no more.

    A run can hold dozens of telemetry files and drawing all of them would make a page nobody
    can open. The one drawn is the one with the most recorded poses -- the most evidence about
    what happened -- and the markdown section beside this lists every file it did not draw, so
    the choice is visible rather than silent.
    """
    from . import trajectory
    # Drawn into the evaluation's own directory and reused until its inputs change. Reading
    # `describe` for every directory would parse twenty-five thousand rows to pick one of
    # them, on every regeneration of the document, to answer a question a `stat` answers.
    best, best_rows = "", 0
    for directory in trajectory.directories_under(Path(root)):
        try:
            rows = (directory / "telemetry.jsonl").stat().st_size
        except OSError:
            continue
        if rows > best_rows:
            best, best_rows = str(directory), rows
    return trajectory.drawing(Path(best)) if best else ""


def page(source: dict[str, Any], sections: Iterable[Any], *, title: str = "",
         curve: str = "", gantt: str = "", provenance: str = "",
         trajectory: str = "") -> str:
    """The whole document, one string, no external reference of any kind."""
    body: list[str] = []
    for section in sections:
        css = "computed" if section.provenance == "computed" else "written"
        label = "算出来的" if css == "computed" else "写出来的"
        inner, converted = to_html(section.body)
        if not converted and "<pre" not in inner:
            inner = f"<pre>{_esc(section.body)}</pre>"
        sources = "、".join(f"<code>{_esc(one)}</code>" for one in section.sources)
        body.append(f'<h2>{_esc(section.title)}</h2>'
                    f'<p class="lead"><span class="prov {css}">{label}</span>'
                    f'{"来源：" + sources if sources else ""}</p>{inner}')
        for note in section.notes:
            body.append(f'<blockquote>{_esc(note)}</blockquote>')
        if section.title.startswith("时间线") and gantt:
            body.append(gantt)
            body.append('<p class="svgcaption">每个条的位置和长度都来自记录里的时间戳；'
                        '没有可用时间戳的条目没有画。</p>')
        if section.title.startswith("决策") and provenance:
            body.append(provenance)
            body.append('<p class="svgcaption">只画记录里已有的边。没有依据的决定就没有入边，'
                        '那个空缺是发现。</p>')
        if section.title.startswith("轨迹") and trajectory:
            body.append(trajectory)
            body.append('<p class="svgcaption">位姿在跨度最大的两个轴上的投影，起始位置是空心圈，'
                        '每条路径按它那一局的成败着色。**这是轨迹，不是渲染** —— '
                        '没有任何东西被渲染出来，也没有相机被模拟。</p>')
        if section.title.startswith("数字") and curve:
            body.append(curve)
            body.append('<p class="svgcaption">只有产生了数字的轮次在线上；'
                        '没产生数字的轮次不画成一个 0。</p>')
    unreadable = source.get("unreadable") or []
    banner = ""
    if unreadable:
        banner = (f'<div class="warn">这份文档从运行自己的记录装配而成，'
                  f'有 {len(unreadable)} 项它读不到 —— 逐条列在「产物与证据」一节。</div>')
    return ("<!DOCTYPE html>\n"
            '<html lang="zh"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_esc(title or '运行记录')}</title>"
            f"<style>{_CSS}</style></head><body>"
            f"<h1>{_esc(title or '运行记录')}</h1>"
            f'<p class="lead">运行目录 <code>{_esc(source.get("root", ""))}</code> · '
            f"由 <code>report_page.py</code> 生成，单文件，不依赖网络</p>"
            f"{banner}" + "".join(body) + "</body></html>\n")


def write(root: Path) -> Path:
    """Build `RUN.html` beside `RUN.md`. Never raises; a broken page is a file that says so."""
    root = Path(root)
    destination = root / "RUN.html"
    try:
        from . import run_record
        source = run_record.gather(root)
        sections = [run_record.written_section(source, _summary_text(root))]
        sections += [builder(source) for builder in run_record.COMPUTED_SECTIONS]
        text = page(source, sections, title=root.name,
                    gantt=gantt_svg([{"at": row.get("at"), "seconds": row.get("seconds"),
                                      "label": row.get("stage") or row.get("event"),
                                      "section": row.get("event")}
                                     for row in (source.get("events") or {}).get("rows", [])]
                                    + [{"at": row.get("at"), "seconds": row.get("seconds"),
                                        "label": row.get("label"), "section": "commands"}
                                       for row in (source.get("processes") or [])]),
                    provenance=provenance_svg((source.get("decisions") or {}).get("rows", [])),
                    curve=curve_svg(_points(source)),
                    trajectory=trajectory_for(root))
        from .common import atomic_text
        atomic_text(destination, text)
    except Exception as exc:                                     # noqa: BLE001
        try:
            from .common import atomic_json, atomic_text, now
            atomic_json(root / "page_renderer_error.json", {
                "at": now(), "renderer": "html", "error": f"{type(exc).__name__}: {exc}"})
            if not destination.is_file():
                atomic_text(destination,
                    f"<!DOCTYPE html><meta charset='utf-8'><h1>{_esc(root.name)}</h1>"
                    f"<p>这份视图没能生成：{_esc(type(exc).__name__)}: {_esc(exc)}</p>"
                    f"<p>记录本身还在 <code>{_esc(root)}</code>。</p>")
        except OSError:
            pass
    return destination


def write_live_page(destination: Path, markdown_text: str) -> None:
    """Top-level HTML is the exact MD publication, not an independent data rescan.

    Only local links/media and passive formatting survive conversion. This page uses
    adjacent artifacts, unlike the older self-contained detailed research page.
    """
    from html.parser import HTMLParser
    from urllib.parse import urlsplit, unquote
    from .common import atomic_text

    class Passive(HTMLParser):
        allowed = {"h1", "h2", "h3", "h4", "p", "br", "hr", "strong", "em", "code", "pre",
                   "details", "summary", "ul", "ol", "li", "blockquote", "table", "thead",
                   "tbody", "tr", "th", "td", "a", "img"}

        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.parts = []

        def handle_starttag(self, tag, attrs):
            if tag not in self.allowed:
                return
            safe = []
            for key, value in attrs:
                if key not in {"href", "src", "alt", "title"} or value is None:
                    continue
                if key in {"href", "src"}:
                    decoded = unquote(value)
                    parsed = urlsplit(decoded)
                    if (parsed.scheme or parsed.netloc or decoded.startswith(("/", "\\")) or
                            ".." in Path(parsed.path).parts or any(ord(c) < 32 for c in decoded)):
                        continue
                safe.append(f' {key}="{_esc(value)}"')
            self.parts.append('<' + tag + ''.join(safe) + '>')

        def handle_endtag(self, tag):
            if tag in self.allowed:
                self.parts.append('</' + tag + '>')

        def handle_data(self, data):
            self.parts.append(_esc(data))

    converted, _ = to_html(markdown_text)
    parser = Passive()
    parser.feed(converted)
    parser.close()
    atomic_text(destination, '<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>研究运行记录</title><style>' + _CSS +
                'img{max-width:100%;height:auto}details{margin-top:24px}</style><body>' +
                ''.join(parser.parts) + '</body></html>')


def _summary_text(root: Path) -> str:
    try:
        import json
        return str(json.loads((Path(root) / "summary.json").read_text(encoding="utf-8"))
                   .get("text") or "")
    except (OSError, ValueError):
        return ""


def _points(source: dict[str, Any]) -> list[tuple[str, float | None]]:
    """The rounds that have a number, in order, from whichever shape this run is."""
    out: list[tuple[str, float | None]] = []
    groups = set()
    for measured in source.get("measurements") or []:
        record = measured["record"]
        if record.get("confirmation") or record.get("ok") is False:
            continue
        metric = record.get("metric") or {}
        groups.add((metric.get("name") or "success_rate", metric.get("unit"),
                    metric.get("direction"), record.get("comparison_protocol_sha256")))
        value = record.get("metric_value", record.get("success_rate"))
        from .recorder import finite
        out.append((str(record.get("label") or "?"), value if finite(value) else None))
    if len(groups) > 1:
        return []  # heterogeneous metrics must never share a single unexplained curve
    if out:
        return out
    for row in source.get("rounds") or []:
        out.append((f"round {row['round']}", row.get("success_rate")))
    return out
