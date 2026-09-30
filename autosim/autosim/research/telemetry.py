"""Incremental, observational telemetry from trainer logs and TensorBoard event files.

This module never chooses the benchmark's score. It records named native scalars as
training-time observations and draws separate small multiples, so loss, throughput and
in-training evaluation are not silently merged into the formal evaluator result.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .common import atomic_json, atomic_text, read_json
from .readings import named_numbers

_STEP_NAMES = ("global_step", "optimizer_step", "update_step", "step", "iteration", "epoch")
_EVENT_DISCOVERY_SECONDS = 30.0
_MAX_SEARCH_DIRECTORIES = 2_048
_MAX_EVENT_FILES = 32
_MAX_EVENT_FILE_BYTES = 64 * 1024 * 1024
_MAX_STDOUT_BYTES_PER_POLL = 2 * 1024 * 1024
_MAX_PARTIAL_LINE_BYTES = 64 * 1024
_MAX_SERIES = 12
_MAX_POINTS_PER_SERIES = 800


@dataclass(frozen=True)
class MetricBinding:
    name: str
    unit: str | None = None


@dataclass(frozen=True)
class TelemetrySpec:
    """Optional task-package aliases/units and bounded native event search roots."""

    metrics: dict[str, MetricBinding] = field(default_factory=dict)
    event_roots: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Any) -> "TelemetrySpec":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("telemetry_spec must be an object")
        version = value.get("schema_version", 1)
        if version != 1:
            raise ValueError("telemetry_spec.schema_version must be 1")
        raw_metrics = value.get("metrics") or {}
        if not isinstance(raw_metrics, dict):
            raise ValueError("telemetry_spec.metrics must map native names to bindings")
        metrics: dict[str, MetricBinding] = {}
        target_units: dict[str, str | None] = {}
        for source, raw in raw_metrics.items():
            if (not isinstance(source, str) or not source.strip() or len(source) > 160 or
                    not isinstance(raw, dict)):
                raise ValueError("each telemetry metric needs a native name and object binding")
            name = raw.get("name", source)
            unit = raw.get("unit")
            if not isinstance(name, str) or not name.strip() or len(name) > 160:
                raise ValueError(f"telemetry metric {source!r} has an invalid display name")
            if unit is not None and (not isinstance(unit, str) or len(unit) > 80):
                raise ValueError(f"telemetry metric {source!r} has an invalid unit")
            name, unit = name.strip(), unit.strip() if isinstance(unit, str) else None
            if name in target_units and target_units[name] != unit:
                raise ValueError(f"telemetry metric aliases for {name!r} disagree on units")
            target_units[name] = unit
            metrics[source] = MetricBinding(name=name, unit=unit)
        roots = value.get("event_roots") or []
        if not isinstance(roots, list) or len(roots) > 16:
            raise ValueError("telemetry_spec.event_roots must be a list of at most 16 paths")
        normalized_roots = []
        for root in roots:
            if not isinstance(root, str) or not root.strip() or len(root) > 240:
                raise ValueError("telemetry event roots must be non-empty relative paths")
            relative = Path(root)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("telemetry event roots cannot be absolute or traverse parents")
            normalized_roots.append(relative.as_posix())
        return cls(metrics=metrics, event_roots=tuple(normalized_roots))


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(float(value)))


def _chart_priority(name: str) -> tuple[int, str]:
    """Keep decision-relevant signals visible when a trainer emits many scalars."""
    normalized = str(name).casefold()
    if "success_once" in normalized:
        priority = 0
    elif "success_at_end" in normalized or "success_rate" in normalized:
        priority = 1
    elif "eval" in normalized and ("return" in normalized or "reward" in normalized):
        priority = 2
    elif "loss" in normalized:
        priority = 3
    elif normalized.rsplit("/", 1)[-1] in {"sps", "fps", "throughput"}:
        priority = 4
    elif "learning_rate" in normalized or normalized.endswith("/lr"):
        priority = 5
    elif "return" in normalized or "reward" in normalized:
        priority = 6
    else:
        priority = 10
    return priority, normalized


def _source_alias(path: Path, roots: tuple[Path, ...]) -> str:
    for index, root in enumerate(roots):
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            material = f"{index}:{relative}"
            return "native-log:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
        except (OSError, ValueError):
            continue
    return "native-log:" + hashlib.sha256(str(path.name).encode("utf-8")).hexdigest()[:12]


def _render_svg(series: dict[str, dict[str, Any]], *, status: str) -> str:
    """Render scalar series independently; different units never share a y-axis."""
    sampled = [(name, row) for name, row in series.items() if row.get("samples")]
    visible = sorted(sampled, key=lambda item: _chart_priority(item[0]))[:_MAX_SERIES]
    width = 900
    panel_height = 122
    height = max(170, 60 + panel_height * max(1, len(visible)))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
             f'width="{width}" height="{height}" role="img" '
             'aria-label="Live native training telemetry">',
             '<rect width="100%" height="100%" fill="#fff"/>',
             '<text x="16" y="23" font-size="16" font-weight="600" '
             'font-family="sans-serif">Native training telemetry — observational, not score</text>',
             f'<text x="16" y="42" font-size="11" fill="#666" '
             f'font-family="sans-serif">Status: {html.escape(status)}</text>']
    if not visible:
        parts.append('<text x="24" y="90" font-size="13" fill="#666" '
                     'font-family="sans-serif">Waiting for named trainer scalars; absence is not zero.</text>')
    elif len(sampled) > len(visible):
        parts.append(f'<text x="16" y="54" font-size="10" fill="#666" '
                     f'font-family="sans-serif">Showing {len(visible)} of {len(sampled)} '
                     'metrics by priority; full series remain in the JSON snapshot.</text>')
    colors = ("#286e9b", "#b85633", "#568342", "#8958a1", "#ba8a21", "#3b8490",
              "#a34b72", "#58657d", "#6e7933", "#9b6742", "#387d59", "#855a78")
    for index, (name, row) in enumerate(visible):
        top = 58 + index * panel_height
        left, right, graph_top, graph_bottom = 72, width - 20, top + 18, top + 91
        samples = [sample for sample in row["samples"]
                   if _finite(sample.get("value"))]
        if not samples:
            continue
        values = [float(sample["value"]) for sample in samples]
        low, high = min(values), max(values)
        if math.isclose(low, high):
            padding = max(abs(low) * 0.05, 0.5)
            low, high = low - padding, high + padding
        steps = [sample.get("native_step") for sample in samples]
        numeric_steps = [float(value) for value in steps if _finite(value)]
        use_steps = len(numeric_steps) == len(samples)
        x_values = numeric_steps if use_steps else list(range(len(samples)))
        x_low, x_high = min(x_values), max(x_values)
        if math.isclose(x_low, x_high):
            x_high = x_low + 1

        def point(sample_index: int) -> tuple[float, float]:
            x = left + (right - left) * (x_values[sample_index] - x_low) / (x_high - x_low)
            y = graph_bottom - (graph_bottom - graph_top) * (values[sample_index] - low) / (high - low)
            return x, y

        color = colors[index % len(colors)]
        points = [point(position) for position in range(len(samples))]
        unit = str(row.get("value_unit") or "").strip()
        escaped_name = html.escape(f"{name} ({unit})" if unit else str(name))
        latest = values[-1]
        parts.extend([
            f'<text x="16" y="{top + 13}" font-size="12" font-weight="600" '
            f'font-family="sans-serif">{escaped_name}</text>',
            f'<text x="{right}" y="{top + 13}" text-anchor="end" font-size="11" '
            f'fill="#555" font-family="sans-serif">n={int(row.get("samples_seen", len(samples)))} · '
            f'latest={latest:.5g}</text>',
            f'<line x1="{left}" y1="{graph_bottom}" x2="{right}" y2="{graph_bottom}" '
            'stroke="#d5dbe1"/>',
            f'<text x="{left - 7}" y="{graph_top + 4}" text-anchor="end" font-size="9" '
            f'fill="#777" font-family="sans-serif">{high:.4g}</text>',
            f'<text x="{left - 7}" y="{graph_bottom + 3}" text-anchor="end" font-size="9" '
            f'fill="#777" font-family="sans-serif">{low:.4g}</text>',
        ])
        if len(points) > 1:
            polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" '
                         'stroke-width="2"/>')
        for sample_index, (x, y) in enumerate(points):
            sample = samples[sample_index]
            title = (f"step={sample.get('native_step')} value={values[sample_index]:.7g} "
                     f"source={sample.get('source_ref', 'native')}")
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{color}">'
                         f'<title>{html.escape(title)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


class LiveTelemetryWriter:
    """Poll one active training attempt and atomically publish a local chart and receipt."""

    def __init__(self, *, output: Path, run_root: Path, repo: Path,
                 working_directory: Path, stage_directory: Path, log_path: Path,
                 stage: str, attempt_id: str, started_epoch: float,
                 telemetry_spec: dict[str, Any] | None = None):
        if not re.fullmatch(r"[0-9a-f]{32}", str(attempt_id)):
            raise ValueError("telemetry attempt id must be a 32-character hex identifier")
        self.output = Path(output).resolve()
        self.run_root = Path(run_root).resolve()
        self.repo = Path(repo).resolve()
        self.working_directory = Path(working_directory).resolve()
        self.stage_directory = Path(stage_directory).resolve()
        self.log_path = Path(log_path).resolve()
        self.stage = str(stage)
        self.attempt_id = str(attempt_id)
        self.started_epoch = float(started_epoch)
        self.report_dir = self.output / "report" / "telemetry"
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.report_dir / f"{self.attempt_id}.json"
        self.chart_path = self.report_dir / f"{self.attempt_id}.svg"
        self.series: dict[str, dict[str, Any]] = {}
        self.errors: list[str] = []
        # Absolute source paths exist only in this process. Persisted cursors and report
        # projections use a stable hash alias so a local report cannot disclose host paths.
        self.source_aliases: set[str] = set()
        self._stdout_offset = 0
        self._stdout_buffer = ""
        self._stdout_segment = 0
        self._event_cursors: dict[str, dict[str, int]] = {}
        self._event_paths: dict[str, Path] = {}
        self._accumulators: dict[str, Any] = {}
        self._event_sizes: dict[str, int] = {}
        self._last_event_discovery = 0.0
        self._published_status = ""
        self._dirty = True
        self._lock = threading.RLock()
        try:
            self.spec = TelemetrySpec.from_value(telemetry_spec)
        except (TypeError, ValueError):
            self.spec = TelemetrySpec()
            self._note_error("invalid_telemetry_spec")
        self._load_existing()
        self._tensorboard_missing = False
        self._publish("in_progress_waiting_for_native_scalars")

    def _load_existing(self) -> None:
        try:
            prior = read_json(self.state_path)
        except (OSError, ValueError):
            return
        if (not isinstance(prior, dict) or prior.get("attempt_id") != self.attempt_id or
                prior.get("schema_version") != 1):
            return
        rows = prior.get("series") or []
        if isinstance(rows, list):
            self.series = {str(row.get("metric_name")): row for row in rows
                           if isinstance(row, dict) and row.get("metric_name")}
        cursor = prior.get("cursor") or {}
        if isinstance(cursor, dict):
            self._stdout_offset = int(cursor.get("stdout_offset") or 0)
            # Never persist an unfinished raw log line: it can contain data outside the
            # numeric observations this local report is designed to retain.
            self._stdout_buffer = ""
            self._stdout_segment = int(cursor.get("stdout_segment") or 0)
            events = cursor.get("event_cursors") or {}
            if isinstance(events, dict):
                self._event_cursors = {str(alias): {str(tag): int(count)
                                                    for tag, count in tags.items()}
                                       for alias, tags in events.items()
                                       if isinstance(tags, dict)}
            sizes = cursor.get("event_sizes") or {}
            if isinstance(sizes, dict):
                self._event_sizes = {str(alias): int(size) for alias, size in sizes.items()
                                     if isinstance(size, (int, float))}
            aliases = cursor.get("source_aliases") or {}
            if isinstance(aliases, list):
                self.source_aliases = {str(alias) for alias in aliases}

    def _append(self, metric_name: str, *, value: float, native_step: int | float | None,
                observed_at: float, source_ref: str, source_position: str,
                verification: str) -> bool:
        if not _finite(value):
            return False
        raw_metric_name = metric_name
        binding = self.spec.metrics.get(raw_metric_name)
        if binding is not None:
            metric_name = binding.name
            value_unit = binding.unit
        else:
            value_unit = None
        row = self.series.setdefault(metric_name, {
            "metric_name": metric_name, "value_unit": value_unit,
            "attempt_id": self.attempt_id,
            "samples_seen": 0, "samples": [],
        })
        sample = {"schema_version": 1, "attempt_id": self.attempt_id,
                  "stage": self.stage, "metric_name": metric_name,
                  "source_metric_name": raw_metric_name,
                  "value": float(value), "value_unit": row.get("value_unit"),
                  "native_step": native_step if _finite(native_step) else None,
                  "step_unit": None, "observed_at": float(observed_at),
                  "source_ref": str(source_ref), "source_position": str(source_position),
                  "parser_version": "autosim-telemetry-v1",
                  "verification": verification}
        samples = row["samples"]
        if samples and samples[-1].get("source_ref") == sample["source_ref"] and \
                samples[-1].get("source_position") == sample["source_position"] and \
                samples[-1].get("value") == sample["value"]:
            return False
        samples.append(sample)
        row["samples_seen"] = int(row.get("samples_seen", 0)) + 1
        if len(samples) > _MAX_POINTS_PER_SERIES:
            newest = samples[-1]
            reduced = samples[::2]
            if reduced and reduced[-1] is newest:
                reduced = reduced[:-1]
            row["samples"] = reduced[-(_MAX_POINTS_PER_SERIES - 1):] + [newest]
        return True

    def _consume_stdout(self) -> bool:
        try:
            if self.log_path.is_symlink() or not self.log_path.is_file():
                return False
            size = self.log_path.stat().st_size
            if size < self._stdout_offset:
                self._stdout_offset = 0
                self._stdout_buffer = ""
                self._stdout_segment += 1
            with self.log_path.open("rb") as stream:
                stream.seek(self._stdout_offset)
                data = stream.read(_MAX_STDOUT_BYTES_PER_POLL)
            if not data:
                return False
            start = self._stdout_offset
            self._stdout_offset += len(data)
            text = self._stdout_buffer + data.decode("utf-8", errors="replace")
            pieces = re.split(r"[\r\n]+", text)
            complete = text.endswith(("\r", "\n"))
            lines = pieces[:-1]
            self._stdout_buffer = "" if complete else pieces[-1]
            if len(self._stdout_buffer.encode("utf-8", errors="replace")) > \
                    _MAX_PARTIAL_LINE_BYTES:
                self._stdout_buffer = ""
                self._note_error("stdout_partial_line_limit")
            changed = False
            position = start
            for line in lines:
                position += len(line.encode("utf-8", errors="replace"))
                stripped = line.strip()
                # Configuration dumps and command echoes are not observations of training.
                if stripped and not stripped.startswith(("args.", "Namespace(", "command:")):
                    values = named_numbers(stripped)
                    native_step = next((values[name] for name in _STEP_NAMES
                                        if name in values), None)
                    for name, value in values.items():
                        if name in _STEP_NAMES or not _finite(value):
                            continue
                        changed |= self._append(
                            name, value=value, native_step=native_step,
                            observed_at=time.time(), source_ref="stage_stdout",
                            source_position=f"{self._stdout_segment}:{position}",
                            verification="native_named_log")
                position += 1
            return changed
        except OSError:
            self._note_error("stdout_log_unavailable")
            return False

    def _event_files(self, *, force: bool = False) -> list[tuple[str, Path]]:
        monotonic = time.monotonic()
        if (not force and self._last_event_discovery and
                monotonic - self._last_event_discovery < _EVENT_DISCOVERY_SECONDS):
            cached = []
            for alias, path in self._event_paths.items():
                try:
                    if (path.is_file() and not path.is_symlink() and
                            path.stat().st_size <= _MAX_EVENT_FILE_BYTES):
                        cached.append((alias, path))
                    elif path.exists():
                        self._note_error("event_file_size_limit")
                except OSError:
                    continue
            return sorted(cached)
        self._last_event_discovery = monotonic
        found: dict[str, Path] = {}
        if self.spec.event_roots:
            configured = [self.working_directory / relative
                          for relative in self.spec.event_roots]
        else:
            configured = [self.working_directory]
        roots = tuple(dict.fromkeys((*configured, self.stage_directory)))
        visited = 0
        for root in roots:
            if not root.is_dir() or root.is_symlink():
                continue
            for parent, directories, names in os.walk(root, followlinks=False):
                visited += 1
                if visited > _MAX_SEARCH_DIRECTORIES:
                    self._note_error("event_search_directory_limit")
                    self._event_paths.update(found)
                    return sorted(found.items())
                base = Path(parent)
                directories[:] = sorted(name for name in directories
                                         if name not in {".git", ".venv", "__pycache__",
                                                         ".pytest_cache"} and
                                         not (base / name).is_symlink())
                for name in names:
                    if not name.startswith("events.out.tfevents"):
                        continue
                    path = base / name
                    try:
                        if (path.is_symlink() or not path.is_file() or
                                path.stat().st_mtime < self.started_epoch - 2 or
                                path.stat().st_size > _MAX_EVENT_FILE_BYTES):
                            continue
                        resolved = path.resolve()
                        if not (resolved.is_relative_to(self.repo) or
                                resolved.is_relative_to(self.output)):
                            continue
                    except OSError:
                        continue
                    alias = _source_alias(resolved, (self.repo, self.output))
                    found[alias] = resolved
                    if len(found) >= _MAX_EVENT_FILES:
                        self._note_error("event_file_limit")
                        self._event_paths.update(found)
                        return sorted(found.items())
        self._event_paths.update(found)
        return sorted(found.items())

    def _consume_tensorboard(self) -> bool:
        event_files = self._event_files()
        if not event_files:
            return False
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            self._tensorboard_missing = True
            self._note_error("tensorboard_reader_unavailable")
            return False
        changed = False
        for alias, path in event_files:
            try:
                size = path.stat().st_size
                accumulator = self._accumulators.get(alias)
                if accumulator is None or size < self._event_sizes.get(alias, 0):
                    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
                    self._accumulators[alias] = accumulator
                    self._event_cursors[alias] = {}
                self._event_sizes[alias] = size
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                cursors = self._event_cursors.setdefault(alias, {})
                self.source_aliases.add(alias)
                for tag in tags:
                    events = accumulator.Scalars(tag)
                    previous = int(cursors.get(tag, 0))
                    if len(events) < previous:
                        previous = 0
                    for index, event in enumerate(events[previous:], start=previous):
                        if (not _finite(event.value) or
                                float(getattr(event, "wall_time", 0)) <
                                self.started_epoch - 2):
                            continue
                        changed |= self._append(
                            str(tag), value=float(event.value), native_step=int(event.step),
                            observed_at=float(event.wall_time), source_ref=alias,
                            source_position=f"{int(event.step)}:{index}",
                            verification="native_tensorboard_scalar")
                    cursors[tag] = len(events)
            except Exception as exc:  # telemetry must never stop the native experiment
                self._note_error(f"tensorboard_{type(exc).__name__}")
        return changed

    def _note_error(self, code: str) -> None:
        if code not in self.errors:
            self.errors.append(code)
            self._dirty = True
        self.errors = self.errors[-8:]

    def note_error(self, code: str) -> None:
        """Record an observer-side failure without allowing it to affect the stage."""
        self._note_error(str(code)[:100])

    @property
    def sample_count(self) -> int:
        return sum(int(row.get("samples_seen", 0)) for row in self.series.values())

    def _state(self, status: str) -> dict[str, Any]:
        series = [self.series[name] for name in sorted(self.series)]
        return {"schema_version": 1, "stage": self.stage, "attempt_id": self.attempt_id,
                "started_at": self.started_epoch, "updated_at": time.time(),
                "status": status, "series": series,
                "sample_count": sum(int(row.get("samples_seen", 0)) for row in series),
                "source_aliases": sorted(self.source_aliases),
                "errors": list(self.errors),
                "limitations": ["training telemetry is observational and is not the official score",
                                "metric units are unknown unless the benchmark declaration supplies them"],
                "cursor": {"stdout_offset": self._stdout_offset,
                           "stdout_segment": self._stdout_segment,
                           "event_cursors": self._event_cursors,
                           "event_sizes": self._event_sizes,
                           "source_aliases": sorted(self.source_aliases)}}

    def _publish(self, status: str) -> dict[str, Any]:
        state = self._state(status)
        atomic_text(self.chart_path, _render_svg(self.series, status=status))
        atomic_json(self.state_path, state)
        self._published_status = status
        self._dirty = False
        return state

    def poll(self) -> dict[str, Any]:
        """Read only newly appended stdout/event data and publish a deterministic view."""
        with self._lock:
            before_errors = tuple(self.errors)
            changed = self._consume_stdout()
            changed = self._consume_tensorboard() or changed
            if self._tensorboard_missing and self.series:
                status = "in_progress_stdout_fallback"
            elif self.series:
                status = "in_progress"
            elif self.errors:
                status = "in_progress_telemetry_unavailable"
            else:
                status = "in_progress_waiting_for_native_scalars"
            if (self._dirty or changed or status != self._published_status or
                    before_errors != tuple(self.errors)):
                self._dirty = True
                return self._publish(status)
            return self._state(status)

    def finalize(self, status: str) -> dict[str, Any]:
        """Capture the final buffered output, then mark telemetry independently of scoring."""
        with self._lock:
            self._consume_stdout()
            self._consume_tensorboard()
            final_status = (status if status in {"completed", "failed", "timed_out",
                                                 "interrupted"} else "unknown")
            return self._publish(final_status)
