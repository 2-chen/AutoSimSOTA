"""Trusted, explicit artifact commits; existence alone never admits a cached result.

This store is owned by the supervisor, outside a repair candidate's writable view.
Only named files are hashed: environments and asset trees are never walked.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import atomic_json, digest, exclusive, now, object_digest, read_json


class ArtifactConflict(RuntimeError):
    """An existing logical output cannot satisfy the requested contract."""


def lifecycle_compatible(receipt: dict | None, old: str, new: str, scientific_hash: str) -> bool:
    """Validate a host-issued compatibility receipt, never an Agent's prose verdict."""
    return bool(receipt and receipt.get("passed") is True
                and receipt.get("scope") == "lifecycle_only"
                and receipt.get("from_revision") == old and receipt.get("to_revision") == new
                and receipt.get("scientific_contract_sha256") == scientific_hash
                and receipt.get("validation_receipt_sha256"))


class ArtifactStore:
    def __init__(self, root: Path, *, artifact_root: Path | None = None):
        self.root = Path(root).absolute()
        self.artifact_root = Path(artifact_root or self.root.parent).resolve()

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key or len(key) > 1024:
            raise ValueError("artifact key must be a nonempty bounded string")
        return self.root / (object_digest(key) + ".json")

    def _contract(self, inputs: Any, scientific_contract: Any) -> dict:
        return {"inputs_sha256": object_digest(inputs),
                "scientific_contract_sha256": object_digest(scientific_contract)}

    def _check(self, row: dict, inputs: Any, scientific_contract: Any) -> None:
        expected = self._contract(inputs, scientific_contract)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ArtifactConflict("artifact inputs or scientific contract changed")

    def prepare(self, key: str, *, inputs: Any, scientific_contract: Any,
                execution_revision: str, attempt_id: str) -> dict:
        if not execution_revision or not attempt_id:
            raise ValueError("producer revision and attempt are required")
        path = self._path(key)
        with exclusive(path.with_suffix(".lock")):
            if path.exists():
                row = read_json(path)
                self._check(row, inputs, scientific_contract)
                if row["execution_revision"] != execution_revision or row["attempt_id"] != attempt_id:
                    raise ArtifactConflict("existing attempt must be reconciled before replacement")
                return row
            row = {"schema_version": 1, "kind": "harness_artifact_commit", "key": key,
                   **self._contract(inputs, scientific_contract),
                   "execution_revision": execution_revision, "attempt_id": attempt_id,
                   "state": "prepared", "created_at": now(), "history": []}
            atomic_json(path, row)
            return row

    def transition(self, key: str, state: str) -> dict:
        path = self._path(key)
        allowed = {"prepared": {"running", "failed"}, "running": {"validating", "failed"},
                   "validating": {"failed"}, "failed": set(), "committed": set()}
        with exclusive(path.with_suffix(".lock")):
            row = read_json(path)
            if state == row["state"]:
                return row
            if state not in allowed.get(row["state"], set()):
                raise ArtifactConflict(f"invalid artifact transition: {row['state']} -> {state}")
            row["history"].append({"from": row["state"], "to": state, "at": now()})
            row["state"] = state
            atomic_json(path, row)
            return row

    def restart_uncommitted(self, key: str, *, execution_revision: str, attempt_id: str,
                            compatibility: dict, previous_terminated: bool) -> dict:
        """A host-verified lifecycle revision may replace a fully reaped attempt.

        The workflow additionally decides whether any scientific work has started;
        this primitive never authorizes replay of a partially scored episode.
        """
        path = self._path(key)
        with exclusive(path.with_suffix(".lock")):
            old = read_json(path)
            if old["state"] == "committed" or not previous_terminated or not attempt_id:
                raise ArtifactConflict("only an uncommitted, terminated attempt may be replaced")
            if not lifecycle_compatible(compatibility, old["execution_revision"], execution_revision,
                                        old["scientific_contract_sha256"]):
                raise ArtifactConflict("replacement lacks verified lifecycle compatibility")
            if attempt_id == old["attempt_id"]:
                raise ArtifactConflict("replacement needs a new attempt identity")
            history = list(old.get("prior_attempts", []))
            history.append({k: v for k, v in old.items() if k != "prior_attempts"})
            row = {k: old[k] for k in ("schema_version", "kind", "key", "inputs_sha256",
                                      "scientific_contract_sha256")}
            row.update(execution_revision=execution_revision, attempt_id=attempt_id,
                       state="prepared", created_at=now(), history=[], prior_attempts=history)
            atomic_json(path, row)
            return row

    def _outputs(self, outputs: Mapping[str, Path]) -> dict:
        if not outputs:
            raise ValueError("a commit requires explicit output files")
        result = {}
        for name, value in sorted(outputs.items()):
            path = Path(value).resolve()
            if not path.is_relative_to(self.artifact_root) or not path.is_file():
                raise ArtifactConflict("artifact output missing or outside artifact root")
            result[name] = {"path": str(path.relative_to(self.artifact_root)),
                            "bytes": path.stat().st_size, "sha256": digest(path)}
        return result

    def commit(self, key: str, *, outputs: Mapping[str, Path], transaction_id: str) -> dict:
        """Call only after the trusted phase-specific validator has passed.

        Reuse transaction_id for budget settlement; replaying this commit is harmless.
        Hashes prove identity/completeness of named files, not benchmark correctness.
        """
        if not transaction_id:
            raise ValueError("commit needs a settlement transaction id")
        path = self._path(key)
        with exclusive(path.with_suffix(".lock")):
            row = read_json(path)
            files = self._outputs(outputs)
            if row["state"] == "committed":
                if row["transaction_id"] != transaction_id or row["outputs"] != files:
                    raise ArtifactConflict("committed result cannot be replaced")
                return row
            if row["state"] != "validating":
                raise ArtifactConflict("only validated phase outputs can be committed")
            row["history"].append({"from": "validating", "to": "committed", "at": now()})
            row.update(state="committed", transaction_id=transaction_id, outputs=files,
                       committed_at=now())
            row["commit_sha256"] = object_digest(row)
            atomic_json(path, row)
            return row

    def reuse(self, key: str, *, inputs: Any, scientific_contract: Any,
              execution_revision: str, compatibility: dict | None = None) -> dict | None:
        path = self._path(key)
        with exclusive(path.with_suffix(".lock")):
            if not path.exists():
                return None
            row = read_json(path)
            self._check(row, inputs, scientific_contract)
            if row["state"] != "committed":
                return None
            recorded_hash = row.get("commit_sha256")
            if recorded_hash != object_digest({k: v for k, v in row.items() if k != "commit_sha256"}):
                raise ArtifactConflict("commit record integrity changed")
            if row["execution_revision"] != execution_revision and not lifecycle_compatible(
                    compatibility, row["execution_revision"], execution_revision,
                    row["scientific_contract_sha256"]):
                raise ArtifactConflict("producer revision changed without verified compatibility")
            files = self._outputs({name: self.artifact_root / item["path"]
                                   for name, item in row["outputs"].items()})
            if files != row["outputs"]:
                raise ArtifactConflict("committed artifact contents changed")
            return row
