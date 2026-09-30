"""A versioned, inspectable projection of one benchmark onboarding attempt.

This is a projection of observed records, not a new source of truth. In particular,
`declared` capabilities remain claims until a receipt verifies them.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .common import atomic_json, bounded_run, digest, now, object_digest, read_json


_INPUTS = ("execution.json", "environment.json", "derived_stages.json")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        found = read_json(path)
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def _revision(repo: Path) -> str:
    try:
        done = bounded_run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                           cwd=repo, env=dict(os.environ), timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def build(*, repo: Path, output: Path, declaration: dict[str, Any],
          run_id: str = "derived") -> Path:
    """Write the bundle using only records already produced by this run."""
    repo, output = Path(repo).resolve(), Path(output).resolve()
    root = output / "benchmark_bundle"
    root.mkdir(parents=True, exist_ok=True)
    inputs = {name: {"path": f"../{name}", "sha256": digest(output / name)}
              for name in _INPUTS if (output / name).is_file()}
    execution = _read(output / "execution.json")
    environment = _read(output / "environment.json")
    derived = _read(output / "derived_stages.json")
    stages = execution.get("stages") or {}
    if not isinstance(stages, dict):
        stages = {}
    declared_graph = execution.get("execution_graph")
    graph_error = ""
    if declared_graph is not None:
        from .execution_graph import ExecutionGraph
        try:
            contract = ExecutionGraph(declared_graph)
            for name in contract.nodes:
                if name not in stages or not isinstance(stages[name], dict) or not (
                        stages[name].get("available")):
                    raise ValueError(f"graph node {name} has no available native stage")
        except ValueError as exc:
            graph_error = str(exc)
            contract = None
    else:
        contract = None
    nodes = []
    for name in sorted(set(stages) | set(derived)):
        row = {**(stages.get(name) or {}), **((derived.get(name) or {}).get("row") or {})}
        graph_node = contract.nodes.get(name) if contract else None
        nodes.append({"id": name, "role": name, "available": bool(
                          (derived.get(name) or {}).get("source")),
                      "entrypoint": row.get("entrypoint"),
                      "artifact_pattern": row.get("artifact"),
                      "working_directory": row.get("working_directory"),
                      "source_ref": "../derived_stages.json" if name in derived else None,
                      "verification": "command_recorded" if name in derived else "unverified",
                      **({"role": graph_node.role,
                          "depends_on": list(graph_node.depends_on),
                          "bindings": dict(graph_node.bindings)} if graph_node else {})})
    task_contract = declaration.get("task_contract") or {}
    telemetry_spec = declaration.get("telemetry_spec")
    if telemetry_spec is None and isinstance(task_contract, dict):
        telemetry_spec = task_contract.get("telemetry_spec")
    protocol = {"task_contract": task_contract,
                "telemetry_spec": telemetry_spec,
                "evaluation_entrypoint": (stages.get("evaluate") or {}).get("entrypoint"),
                "status": "claimed_not_frozen"}
    frozen = _read(output / "research" / run_id / "protocol_frozen.json")
    if frozen.get("hashes") is not None:
        protocol["frozen_file_hashes"] = frozen["hashes"]
        protocol["status"] = "files_frozen_for_local_run"
    comparison = _read(output / "research" / run_id / "comparison_protocol.json")
    if comparison:
        protocol["comparison"] = comparison
        protocol["status"] = "comparison_settings_frozen_for_local_run"
    atomic_json(root / "capabilities.json", {"claims": declaration.get("capabilities") or {},
                                            "warning": "declaration alone is not execution proof"})
    atomic_json(root / "assets.json", {"claims": declaration.get("assets") or {},
                                      "warning": "ownership and loadability require receipts"})
    atomic_json(root / "environments.json", {"record": environment,
                                            "source_ref": "../environment.json"})
    atomic_json(root / "execution_graph.json", {"schema_version": 1, "nodes": nodes,
                                                "score_target": declared_graph.get("score_target")
                                                if contract else None,
                                                "graph_status": "declared_valid" if contract else
                                                "invalid_declaration" if graph_error else
                                                "dependencies_unknown",
                                                **({"graph_error": graph_error} if graph_error else {}),
                                                "warning": "Dependency edges are claims until "
                                                           "a graph execution produces receipts"})
    atomic_json(root / "protocol.json", protocol)
    manifest = {"schema_version": 1, "created_at": now(), "repo": str(repo),
                "repo_revision": _revision(repo), "run_id": run_id,
                "input_records": inputs,
                "declaration_sha256": object_digest(declaration),
                "status": "partial" if not derived else "onboarded_commands_unconfirmed",
                "warning": "A bundle is an onboarding record, not a verified experiment."}
    manifest["fingerprint"] = object_digest({k: v for k, v in manifest.items()
                                               if k not in {"created_at", "fingerprint"}})
    atomic_json(root / "manifest.json", manifest)
    return root
