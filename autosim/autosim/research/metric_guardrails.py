"""Frozen, Agent-declared secondary objectives; never infer direction from names."""
import math
from pathlib import Path
from .metric_contract import MetricSpec


def specifications(declaration: dict):
    rows = (declaration.get("research_goal") or {}).get("guardrail_metrics", [])
    if not isinstance(rows, list) or len(rows) > 16:
        raise ValueError("guardrail_metrics must be a bounded list")
    result, names = [], set()
    for row in rows:
        if not isinstance(row, dict): raise ValueError("guardrail metric must be an object")
        spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": row}})
        tolerance = row.get("max_regression", 0)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("guardrail regression tolerance must be finite and nonnegative")
        if spec.name in names: raise ValueError("duplicate guardrail metric")
        names.add(spec.name)
        result.append({"metric": spec.as_dict(), "max_regression": float(tolerance)})
    return result


def read_values(specs, *, said: str, artifact: Path | None, primary_spec=None):
    readings = {}
    for row in specs:
        spec = MetricSpec(**row["metric"])
        if (spec.source != "log" and spec.artifact_pattern and primary_spec is not None and
                (spec.source, spec.artifact_root, spec.artifact_pattern) !=
                (primary_spec.source, primary_spec.artifact_root, primary_spec.artifact_pattern)):
            readings[spec.name] = {"value": None, "status": "unavailable",
                "reason": "secondary artifact differs from archived primary result; requires separate verified archive"}
            continue
        readings[spec.name] = spec.read(said=said, artifact=artifact)
    return readings


def compare(specs, current: dict, baseline: dict):
    checks = []
    for row in specs:
        spec = MetricSpec(**row["metric"])
        value, reference = current.get(spec.name, {}).get("value"), baseline.get(spec.name, {}).get("value")
        known = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (value, reference))
        loss = spec.utility(reference) - spec.utility(value) if known else None
        checks.append({"name": spec.name, "unit": spec.unit, "value": value, "baseline": reference,
                       "max_regression": row["max_regression"], "regression": loss,
                       "status": "unknown" if not known else "passed" if loss <= row["max_regression"] else "violated"})
    status = "violated" if any(c["status"] == "violated" for c in checks) else "unknown" if any(c["status"] == "unknown" for c in checks) else "passed"
    return {"status": status, "checks": checks}
