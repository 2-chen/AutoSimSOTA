"""An explicit primary metric, separate from onboarding progress and log observations."""

from __future__ import annotations

import math
import re
import csv
import glob
from dataclasses import asdict, dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .common import digest, read_json
from .readings import success_reading


@dataclass(frozen=True)
class MetricSpec:
    name: str = "success_rate"
    direction: str = "maximize"
    unit: str = "fraction"
    minimum: float | None = None
    maximum: float | None = None
    source: str = "log"
    artifact_root: str = ""
    artifact_pattern: str = ""
    json_key: str = ""
    json_value_key: str = ""
    csv_column: str = ""
    aggregation: str = ""
    min_samples: int = 1
    episode_id_column: str = ""
    task_column: str = ""
    initial_state_hash_column: str = ""

    @classmethod
    def from_declaration(cls, declaration: dict[str, Any]) -> "MetricSpec":
        goal = declaration.get("research_goal") or {}
        contract = declaration.get("task_contract") or {}
        row = goal.get("primary_metric") or contract.get("primary_metric")
        if not isinstance(row, dict):
            return cls(minimum=0.0, maximum=1.0)
        name = str(row.get("name") or "").strip()
        direction = str(row.get("direction") or "").strip().lower()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_ .-]{0,60}", name):
            raise ValueError("primary metric needs a concrete printable name")
        if direction not in {"maximize", "minimize"}:
            raise ValueError("primary metric direction must be maximize or minimize")
        source = str(row.get("source") or "log").lower()
        if source not in {"log", "json", "csv"}:
            raise ValueError("primary metric source must be log, json or csv")
        artifact_root = str(row.get("artifact_root") or "")
        artifact_pattern = str(row.get("artifact_pattern") or "")
        if artifact_pattern:
            path = PurePosixPath(artifact_pattern.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or "\x00" in artifact_pattern:
                raise ValueError("metric artifact pattern must be a safe relative path")
            if source not in {"json", "csv"}:
                raise ValueError("metric artifact pattern requires a structured source")
            if artifact_root not in {"working_directory", "output", "policy_parent"}:
                raise ValueError("metric artifact root must be working_directory, output, or "
                                 "policy_parent")
        elif artifact_root:
            raise ValueError("metric artifact root requires an artifact pattern")
        minimum = row.get("min")
        maximum = row.get("max")
        if name == "success_rate":
            minimum = 0.0 if minimum is None else minimum
            maximum = 1.0 if maximum is None else maximum
        if minimum is not None and maximum is not None and float(minimum) > float(maximum):
            raise ValueError("primary metric min exceeds max")
        aggregation = str(row.get("aggregation") or "").lower()
        if aggregation not in {"", "mean", "sum", "last"}:
            raise ValueError("primary metric aggregation must be mean, sum or last")
        min_samples = row.get("min_samples", 1)
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("primary metric min_samples must be a positive integer")
        episode_id_column = str(row.get("episode_id_column") or "")
        task_column = str(row.get("task_column") or "")
        initial_state_hash_column = str(row.get("initial_state_hash_column") or "")
        json_value_key = str(row.get("json_value_key") or "")
        if ((episode_id_column or task_column or initial_state_hash_column) and
                source not in {"csv", "json"}):
            raise ValueError("episode identity fields require a structured metric source")
        if task_column and not episode_id_column:
            raise ValueError("task identity requires an episode identity column")
        if initial_state_hash_column and not episode_id_column:
            raise ValueError("initial-state identity requires an episode identity column")
        if source == "json" and episode_id_column:
            if not json_value_key:
                raise ValueError("JSON episode records require json_value_key")
            if not str(row.get("json_key") or ""):
                raise ValueError("JSON episode records require json_key for their record array")
        if source != "json" and json_value_key:
            raise ValueError("json_value_key is only valid for a JSON metric source")
        if source == "log" and min_samples > 1:
            raise ValueError("sample-count validation requires a structured result source")
        if source == "json" and min_samples > 1 and not episode_id_column:
            raise ValueError("JSON sample-count validation requires episode identity fields")
        selected_columns = [one for one in (str(row.get("csv_column") or name),
                                             episode_id_column, task_column,
                                             initial_state_hash_column) if one]
        if len(selected_columns) != len(set(selected_columns)):
            raise ValueError("CSV metric and identity columns must be distinct")
        return cls(name=name, direction=direction, unit=str(row.get("unit") or ""),
                   minimum=float(minimum) if minimum is not None else None,
                   maximum=float(maximum) if maximum is not None else None,
                   source=source, artifact_root=artifact_root,
                   artifact_pattern=artifact_pattern,
                   json_key=str(row.get("json_key") or ""),
                   json_value_key=json_value_key,
                   csv_column=str(row.get("csv_column") or ""),
                   aggregation=aggregation, min_samples=min_samples,
                   episode_id_column=episode_id_column, task_column=task_column,
                   initial_state_hash_column=initial_state_hash_column)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def utility(self, value: float) -> float:
        return value if self.direction == "maximize" else -value


    def read_log(self, said: str) -> dict[str, Any]:
        if self.name == "success_rate" and self.unit == "fraction":
            return success_reading(said)
        # Exact label, one line, final observation. `reward` must not match `best_reward`.
        label = re.compile(r"(?<!best )(?<!max )(?<![A-Za-z0-9_])" + re.escape(self.name) +
                           r"[ \t]*[:=][ \t]*(?P<number>[-+]?(?:\d+\.?\d*|\.\d+)"
                           r"(?:[eE][-+]?\d+)?)", re.IGNORECASE)
        found = [(line, match) for line in (said or "").splitlines()
                 for match in label.finditer(line)]
        if not found:
            return {"value": None, "read_from": "", "why_not": "metric label not found"}
        line, match = found[-1]
        value = float(match.group("number"))
        if not math.isfinite(value):
            return {"value": None, "read_from": line[:300], "why_not": "non-finite metric"}
        if (self.minimum is not None and value < self.minimum) or (
                self.maximum is not None and value > self.maximum):
            return {"value": None, "read_from": line[:300], "why_not": "metric out of range"}
        return {"value": value, "read_from": line[:300], "unit": self.unit,
                "name": self.name}

    def read(self, *, said: str, artifact: Path | None = None) -> dict[str, Any]:
        if self.source == "log":
            return self.read_log(said)
        if self.source == "csv":
            return self.read_csv(artifact)
        if artifact is None or not artifact.is_file() or artifact.suffix.lower() != ".json":
            return {"value": None, "read_from": "", "why_not": "JSON result artifact missing"}
        try:
            document: Any = read_json(artifact)
            key = self.json_key or self.name
            value = self._json_path(document, key)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return {"value": None, "read_from": str(artifact),
                    "why_not": f"JSON metric missing: {type(exc).__name__}"}
        if self.episode_id_column:
            return self._read_json_episodes(artifact, document, value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {"value": None, "read_from": str(artifact),
                    "why_not": "JSON metric is not a scalar number"}
        try:
            value = float(value)
        except (OverflowError, ValueError):
            return {"value": None, "read_from": str(artifact),
                    "why_not": "JSON metric cannot be represented as a finite float"}
        if not math.isfinite(value) or (
                self.minimum is not None and value < self.minimum) or (
                self.maximum is not None and value > self.maximum):
            return {"value": None, "read_from": str(artifact),
                    "why_not": "JSON metric is non-finite or out of range"}
        return {"value": value, "read_from": f"{artifact}#{key}",
                "unit": self.unit, "name": self.name}

    @staticmethod
    def _json_path(document: Any, path: str) -> Any:
        value = document
        for part in path.split("."):
            if isinstance(value, list):
                value = value[int(part)]
            else:
                value = value[part]
        return value

    def _read_json_episodes(self, artifact: Path, document: Any,
                            records: Any) -> dict[str, Any]:
        """Read an explicitly mapped JSON record array; field names come from the contract."""
        if not isinstance(records, list) or not records:
            return {"value": None, "read_from": str(artifact),
                    "why_not": "JSON episode record array missing or empty"}
        values: list[float] = []
        episode_keys: list[str] = []
        initial_state_hashes: list[str] = []
        seen: set[str] = set()
        try:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("episode record is not an object")
                raw_episode = self._json_path(record, self.episode_id_column)
                if raw_episode is None or str(raw_episode).strip() == "":
                    raise ValueError("empty native episode identity")
                try:
                    raw_task = self._json_path(record, self.task_column) if self.task_column else None
                except (KeyError, IndexError, TypeError, ValueError):
                    raw_task = self._json_path(document, self.task_column) if self.task_column else None
                task = str(raw_task).strip() if raw_task is not None else ""
                if self.task_column and not task:
                    raise ValueError("empty native task identity")
                identity = f"{task}\x1f{str(raw_episode).strip()}"
                if identity in seen:
                    raise ValueError("duplicate native task/episode identity")
                seen.add(identity)
                episode_keys.append(identity)
                raw = self._json_path(record, self.json_value_key)
                if isinstance(raw, bool):
                    number = float(raw)
                elif isinstance(raw, (int, float)):
                    number = float(raw)
                else:
                    raise ValueError("episode metric value is not a number or boolean")
                if not math.isfinite(number):
                    raise ValueError("non-finite JSON episode metric value")
                if ((self.minimum is not None and number < self.minimum) or
                        (self.maximum is not None and number > self.maximum)):
                    raise ValueError("JSON episode metric value out of range")
                if self.name == "success_rate" and number not in (0.0, 1.0):
                    raise ValueError("per-episode success value must be zero or one")
                values.append(number)
                if self.initial_state_hash_column:
                    state = str(self._json_path(record, self.initial_state_hash_column) or "")
                    if not re.fullmatch(r"[0-9a-fA-F]{64}", state):
                        raise ValueError("initial-state hash must be a SHA-256 digest")
                    initial_state_hashes.append(state.lower())
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            return {"value": None, "read_from": str(artifact),
                    "why_not": f"invalid JSON episode records: {type(exc).__name__}: {exc}"}
        if len(values) < self.min_samples:
            return {"value": None, "read_from": str(artifact),
                    "why_not": f"JSON has {len(values)} samples; needs {self.min_samples}"}
        if len(values) > 1 and not self.aggregation:
            return {"value": None, "read_from": str(artifact),
                    "why_not": "multiple JSON episode records require explicit aggregation"}
        if self.aggregation == "mean":
            value = math.fsum(values) / len(values)
        elif self.aggregation == "sum":
            value = math.fsum(values)
        else:
            value = values[-1]
        if not math.isfinite(value) or (
                self.minimum is not None and value < self.minimum) or (
                self.maximum is not None and value > self.maximum):
            return {"value": None, "read_from": str(artifact),
                    "why_not": "aggregated JSON metric is non-finite or out of range"}
        return {"value": value, "read_from": f"{artifact}#{self.json_key}",
                "unit": self.unit, "name": self.name,
                "samples": len(values), "aggregation": self.aggregation or "single",
                "episodes_completed": len(episode_keys), "episode_keys": episode_keys,
                "episode_values": values,
                **({"initial_state_hashes": initial_state_hashes}
                   if self.initial_state_hash_column else {})}

    def read_csv(self, artifact: Path | None) -> dict[str, Any]:
        """Read an explicit native result column; never guess across rows or columns."""
        if artifact is None or not artifact.is_file() or artifact.suffix.lower() != ".csv":
            return {"value": None, "read_from": "", "why_not": "CSV result artifact missing"}
        column = self.csv_column or self.name
        values: list[float] = []
        episode_keys: list[str] = []
        initial_state_hashes: list[str] = []
        seen_episodes: set[str] = set()
        try:
            with artifact.open(newline="", encoding="utf-8-sig") as stream:
                rows = csv.DictReader(stream)
                if not rows.fieldnames or rows.fieldnames.count(column) != 1:
                    return {"value": None, "read_from": str(artifact),
                            "why_not": "CSV metric column missing or duplicated"}
                for identity_column in (self.episode_id_column, self.task_column,
                                        self.initial_state_hash_column):
                    if identity_column and rows.fieldnames.count(identity_column) != 1:
                        return {"value": None, "read_from": str(artifact),
                                "why_not": f"CSV identity column missing or duplicated: {identity_column}"}
                for row in rows:
                    if self.episode_id_column:
                        episode = str(row.get(self.episode_id_column) or "").strip()
                        task = str(row.get(self.task_column) or "").strip() if self.task_column else ""
                        if not episode or (self.task_column and not task):
                            raise ValueError("empty native task or episode identity")
                        identity = f"{task}\x1f{episode}"
                        if identity in seen_episodes:
                            raise ValueError("duplicate native task/episode identity")
                        seen_episodes.add(identity)
                        episode_keys.append(identity)
                        if self.initial_state_hash_column:
                            state = str(row.get(self.initial_state_hash_column) or "").strip()
                            if not re.fullmatch(r"[0-9a-fA-F]{64}", state):
                                raise ValueError("initial-state hash must be a SHA-256 digest")
                            initial_state_hashes.append(state.lower())
                    raw = row.get(column)
                    if raw is None or not raw.strip():
                        raise ValueError("empty CSV metric cell")
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError("non-finite CSV metric cell")
                    if ((self.minimum is not None and value < self.minimum) or
                            (self.maximum is not None and value > self.maximum)):
                        raise ValueError("CSV metric cell out of range")
                    if self.episode_id_column and self.name == "success_rate" and value not in (0, 1):
                        raise ValueError("per-episode success value must be zero or one")
                    values.append(value)
        except (OSError, UnicodeError, csv.Error, ValueError, OverflowError) as exc:
            return {"value": None, "read_from": str(artifact),
                    "why_not": f"invalid CSV metric: {type(exc).__name__}: {exc}"}
        if len(values) < self.min_samples:
            return {"value": None, "read_from": str(artifact),
                    "why_not": f"CSV has {len(values)} samples; needs {self.min_samples}"}
        if len(values) > 1 and not self.aggregation:
            return {"value": None, "read_from": str(artifact),
                    "why_not": "multiple CSV rows require explicit aggregation"}
        if self.aggregation == "mean":
            value = math.fsum(values) / len(values)
        elif self.aggregation == "sum":
            value = math.fsum(values)
        else:
            value = values[-1]
        if not math.isfinite(value) or (
                self.minimum is not None and value < self.minimum) or (
                self.maximum is not None and value > self.maximum):
            return {"value": None, "read_from": str(artifact),
                    "why_not": "aggregated CSV metric is non-finite or out of range"}
        return {"value": value, "read_from": f"{artifact}#{column}",
                "unit": self.unit, "name": self.name, "samples": len(values),
                "aggregation": self.aggregation or "single",
                **({"episodes_completed": len(episode_keys),
                    "episode_keys": episode_keys,
                    "episode_values": values,
                    **({"initial_state_hashes": initial_state_hashes}
                       if self.initial_state_hash_column else {}),
                    "successes": int(sum(values)) if self.name == "success_rate" else None}
                   if self.episode_id_column else {})}


def resolve_metric_artifact(spec: MetricSpec, *, roots: dict[str, Path],
                            started_at: float, allowed_roots: list[Path]) -> dict[str, Any]:
    """Resolve one fresh, declared result file and bind its bytes to this score attempt."""
    if not spec.artifact_pattern:
        return {"status": "not_declared", "matched": 0}
    base = roots.get(spec.artifact_root)
    if base is None:
        return {"status": "missing_root", "matched": 0}
    base = Path(base).resolve()
    raw_pattern = spec.artifact_pattern.replace("\\", "/")
    relative = PurePosixPath(raw_pattern)
    if relative.is_absolute() or ".." in relative.parts:
        return {"status": "unsafe_pattern", "matched": 0}
    matches = sorted(Path(name) for name in glob.glob(str(base / raw_pattern), recursive=True))
    fresh: list[Path] = []
    for path in matches:
        try:
            if path.is_symlink():
                continue
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            if (not resolved.is_file() or stat.st_mtime < float(started_at) or
                    not any(resolved.is_relative_to(Path(root).resolve())
                            for root in allowed_roots)):
                continue
            fresh.append(resolved)
        except (OSError, TypeError, ValueError):
            continue
    evidence: dict[str, Any] = {
        "status": "matched" if len(fresh) == 1 else
                  "missing" if not fresh else "ambiguous",
        "pattern": spec.artifact_pattern, "root": spec.artifact_root,
        "matched": len(fresh), "candidate_paths": [str(path) for path in fresh[:8]],
        "freshness_checked": True,
    }
    if len(fresh) != 1:
        return evidence
    selected = fresh[0]
    stat = selected.stat()
    evidence.update(path=str(selected), sha256=digest(selected),
                    size_bytes=stat.st_size, mtime=stat.st_mtime)
    return evidence
