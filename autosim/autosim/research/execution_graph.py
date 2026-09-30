"""A small, role-neutral execution DAG over commands that were actually derived.

The graph defines ordering and artifact bindings, not success. The caller supplies the
executor that runs each node and the function that resolves its verified output. This makes
it usable for data preparation, online rollout/update, and source-policy evaluation without
assuming that every repository has a train.py or a checkpoint.
"""

from __future__ import annotations

import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,79}\Z")
RESERVED_BINDINGS = frozenset({"stage", "settings", "timeout", "_reconsidered",
                               "repo", "python", "output", "device", "device_index"})


@dataclass(frozen=True)
class Node:
    id: str
    role: str
    depends_on: tuple[str, ...]
    bindings: tuple[tuple[str, str], ...]
    resources: dict[str, Any] | None = None


class ExecutionGraph:
    """Validated DAG; never infer edges from names or ordering in a dictionary."""

    def __init__(self, document: dict[str, Any]):
        if not isinstance(document, dict) or not isinstance(document.get("nodes"), list):
            raise ValueError("execution graph needs a nodes list")
        raw_nodes = document["nodes"]
        if not raw_nodes or len(raw_nodes) > 64:
            raise ValueError("execution graph needs 1..64 nodes")
        nodes: dict[str, Node] = {}
        for raw in raw_nodes:
            if not isinstance(raw, dict):
                raise ValueError("each execution graph node must be an object")
            name = raw.get("id")
            if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
                raise ValueError(f"invalid execution graph node id: {name!r}")
            if name in nodes:
                raise ValueError(f"duplicate execution graph node id: {name}")
            role = raw.get("role")
            if not isinstance(role, str) or not role.strip() or len(role) > 120:
                raise ValueError(f"node {name} needs a short role")
            dependencies = raw.get("depends_on", [])
            if not isinstance(dependencies, list) or any(
                    not isinstance(dep, str) or not IDENTIFIER.fullmatch(dep)
                    for dep in dependencies):
                raise ValueError(f"node {name} has malformed dependencies")
            if len(set(dependencies)) != len(dependencies):
                raise ValueError(f"node {name} repeats a dependency")
            bindings = raw.get("bindings", {})
            if not isinstance(bindings, dict) or any(
                    not isinstance(key, str) or not IDENTIFIER.fullmatch(key) or
                    not isinstance(source, str) or not IDENTIFIER.fullmatch(source)
                    for key, source in bindings.items()):
                raise ValueError(f"node {name} has malformed artifact bindings")
            if set(bindings) & RESERVED_BINDINGS:
                raise ValueError(f"node {name} binds a reserved runtime input")
            if not set(bindings.values()).issubset(dependencies):
                raise ValueError(f"node {name} binds an artifact outside its dependencies")
            from .scheduling import resource_profile
            resources = resource_profile(raw["resources"]) if "resources" in raw else None
            nodes[name] = Node(name, role.strip(), tuple(dependencies),
                               tuple(sorted(bindings.items())), resources)
        for node in nodes.values():
            missing = set(node.depends_on) - set(nodes)
            if missing:
                raise ValueError(f"node {node.id} depends on missing nodes: {sorted(missing)}")
        self.nodes = nodes
        self._order = self._sort()

    def _sort(self) -> tuple[str, ...]:
        order: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"execution graph has a cycle through {name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.nodes[name].depends_on:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in self.nodes:
            visit(name)
        return tuple(order)

    def order_for(self, target: str) -> tuple[str, ...]:
        if target not in self.nodes:
            raise ValueError(f"no such execution graph target: {target}")
        needed: set[str] = set()

        def require(name: str) -> None:
            if name in needed:
                return
            needed.add(name)
            for dependency in self.nodes[name].depends_on:
                require(dependency)

        require(target)
        return tuple(name for name in self._order if name in needed)

    def execute(self, target: str, *, invoke: Callable[[str, dict[str, str]], dict[str, Any]],
                artifact_of: Callable[[dict[str, Any]], Path | None],
                on_step: Callable[[dict[str, Any]], None] | None = None,
                max_workers: int = 1
                ) -> dict[str, Any]:
        """Stop on a missing prerequisite or failed receipt; never reuse an old artifact."""
        if not isinstance(max_workers, int) or isinstance(max_workers, bool) or not 1 <= max_workers <= 4:
            raise ValueError("graph concurrency must be 1..4")
        if max_workers > 1:
            return self._parallel(target, invoke=invoke, artifact_of=artifact_of,
                                  on_step=on_step, max_workers=max_workers)
        records: list[dict[str, Any]] = []
        artifacts: dict[str, str] = {}
        for name in self.order_for(target):
            node = self.nodes[name]
            inputs: dict[str, str] = {}
            for key, source in node.bindings:
                if source not in artifacts:
                    result = {"target": target, "status": "blocked", "where": name,
                              "why": f"{source} produced no verified artifact for {key}",
                              "nodes": records}
                    if on_step:
                        on_step(result)
                    return result
                inputs[key] = artifacts[source]
            record = invoke(name, inputs)
            if not isinstance(record, dict):
                raise TypeError(f"node {name} did not return a receipt object")
            summary = {"id": name, "role": node.role,
                       "attempt_id": record.get("attempt_id"),
                       "status": record.get("status"),
                       "returncode": record.get("returncode"),
                       "inputs": inputs}
            records.append(summary)
            artifact_check = record.get("artifact") or {}
            if (record.get("status") != "completed" or
                    record.get("returncode") != 0 or not record.get("ran") or
                    (isinstance(artifact_check, dict) and artifact_check.get("checked") and
                     not artifact_check.get("matched"))):
                result = {"target": target, "status": "failed", "where": name,
                          "why": str(record.get("why") or
                                     "node did not complete with its artifact postcondition"),
                          "nodes": records}
                if on_step:
                    on_step(result)
                return result
            artifact = artifact_of(record)
            if artifact is not None:
                artifact = Path(artifact)
                if not artifact.exists() or artifact.is_symlink():
                    artifact = None
            if artifact is not None:
                artifacts[name] = str(artifact)
                summary["artifact"] = str(artifact)
            if on_step:
                on_step({"target": target, "status": "running", "nodes": records})
        result = {"target": target, "status": "completed", "nodes": records,
                  "artifacts": artifacts,
                  "warning": "Process completion is not a scored or confirmed experiment."}
        if on_step:
            on_step(result)
        return result

    def _parallel(self, target, *, invoke, artifact_of, on_step, max_workers):
        """Only independent ready nodes run concurrently. Journal/artifact callbacks serialize."""
        order = self.order_for(target)
        if any(self.nodes[name].resources is None for name in order):
            raise ValueError("parallel graph requires explicit resources for every node")
        completed, artifacts, summaries, pending = set(), {}, {}, set(order)
        failure = None
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="autosim-dag") as pool:
            running = {}
            while pending or running:
                for name in order:
                    if failure or len(running) >= max_workers:
                        break
                    node = self.nodes[name]
                    if name not in pending or not set(node.depends_on) <= completed:
                        continue
                    missing = [source for _, source in node.bindings if source not in artifacts]
                    if missing:
                        failure = {"status": "blocked", "where": name,
                                   "why": f"{missing[0]} produced no verified bound artifact"}
                        break
                    inputs = {key: artifacts[source] for key, source in node.bindings}
                    running[pool.submit(invoke, name, inputs)] = (name, inputs)
                    pending.remove(name)
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    name, inputs = running.pop(future)
                    try:
                        receipt = future.result()
                        if not isinstance(receipt, dict):
                            raise TypeError("node did not return a receipt object")
                    except Exception as exc:
                        receipt = {"status": "failed", "why": f"{type(exc).__name__}: {exc}"}
                    summary = {"id": name, "role": self.nodes[name].role,
                        "attempt_id": receipt.get("attempt_id"), "status": receipt.get("status"),
                        "returncode": receipt.get("returncode"), "inputs": inputs}
                    summaries[name] = summary
                    check = receipt.get("artifact") or {}
                    okay = (receipt.get("status") == "completed" and receipt.get("returncode") == 0
                        and receipt.get("ran") and not (isinstance(check, dict) and
                            check.get("checked") and not check.get("matched")))
                    if not okay:
                        failure = failure or {"status": "failed", "where": name,
                            "why": str(receipt.get("why") or "node receipt/postcondition failed")}
                    else:
                        artifact = artifact_of(receipt)
                        if artifact is not None and Path(artifact).exists() and not Path(artifact).is_symlink():
                            artifacts[name] = str(artifact)
                            summary["artifact"] = str(artifact)
                        completed.add(name)
                    if on_step:
                        on_step({"target": target, "status": "running", "nodes":
                                 [summaries[n] for n in order if n in summaries]})
                # Already running siblings finish and preserve evidence; no successors after failure.
                if failure and not running:
                    break
        result = {"target": target, "status": "completed", "nodes":
                  [summaries[n] for n in order if n in summaries], "artifacts": artifacts,
                  "warning": "Process completion is not a scored or confirmed experiment."}
        if failure:
            result.update(failure)
        if on_step:
            on_step(result)
        return result
