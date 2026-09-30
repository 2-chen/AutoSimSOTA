"""Signed, immutable lifecycle-component selection for a fresh Python process.

The frozen repository and benchmark paths remain unchanged.  The trusted outer
host validates a pure component, publishes a signed sidecar and passes its exact
identity to the next supervisor.  The research package's first-import hook uses
that sidecar before any coordinator imports the component.  No hot reload exists.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Mapping

from .common import atomic_json, digest, exclusive, immutable_json, object_digest, read_json
from .repair_tools import membership_validation_spec, verify_activation_request
from .repair_validation import check_pure_component, run_contract


COMPONENT = "autosim.research.probe_barrier_contract"
TARGET = "autosim/autosim/research/probe_barrier_contract.py"
FUNCTION = "ready_members_match"
HOST_FILES = ("autosim/autosim/research/__init__.py",
              "autosim/autosim/research/execution_activation.py")
ENVIRONMENT_KEYS = ("AUTOSIM_EXECUTION_ACTIVATION_FILE", "AUTOSIM_EXECUTION_ACTIVATION_SHA256",
                    "AUTOSIM_EXECUTION_TRUST_ROOT", "AUTOSIM_EXECUTION_REVISION")
_ACTIVE_IDENTITY: dict | None = None


class ActivationRefused(RuntimeError):
    pass


def _safe_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ActivationRefused("source manifest path escaped its frozen root")
    return path.as_posix()


def _check_sources(root: Path, files: Mapping[str, str]) -> None:
    root = root.resolve()
    if not files:
        raise ActivationRefused("activation requires a nonempty frozen source manifest")
    for relative, expected in files.items():
        path = root / _safe_relative(relative)
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root) or digest(path) != expected:
            raise ActivationRefused("frozen execution source no longer matches its host manifest")


def _signing_key(root: Path, *, create: bool) -> bytes:
    path = root / ".host-signing-key"
    if create and not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(os.urandom(32))
                stream.flush()
                os.fsync(stream.fileno())
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ActivationRefused("host signing key is absent or not private")
    value = path.read_bytes()
    if len(value) != 32:
        raise ActivationRefused("invalid host signing key")
    return value


def _signature(body: dict, key: bytes) -> str:
    return hmac.new(key, json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), hashlib.sha256).hexdigest()


def _signed_body(path: Path, *, expected_sha256: str, trust_root: Path, require_registered: bool = True) -> dict:
    trust_root = trust_root.resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(trust_root / "revisions"):
        raise ActivationRefused("activation sidecar path or digest is not host-pinned")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ActivationRefused("activation sidecar path or digest is not host-pinned")
    envelope = json.loads(content)
    if not isinstance(envelope, dict) or set(envelope) != {"body", "signature"}:
        raise ActivationRefused("invalid activation envelope")
    body = envelope["body"]
    expected = _signature(body, _signing_key(trust_root, create=False))
    if not isinstance(envelope["signature"], str) or not hmac.compare_digest(expected, envelope["signature"]):
        raise ActivationRefused("activation signature does not match this trusted host registry")
    if body.get("schema_version") != 1 or body.get("component") != COMPONENT or body.get("function") != FUNCTION:
        raise ActivationRefused("activation requests an unregistered component")
    component = trust_root / _safe_relative(body["component_file"])
    if (component.is_symlink() or not component.resolve().is_relative_to(trust_root / "components") or
            digest(component) != body["component_sha256"]):
        raise ActivationRefused("activated component changed after host verification")
    check_pure_component(component.read_text(), FUNCTION)
    validation_path = trust_root / _safe_relative(body["validation_file"])
    if (validation_path.is_symlink() or not validation_path.resolve().is_relative_to(trust_root / "validations") or
            digest(validation_path) != body["validation_sha256"]):
        raise ActivationRefused("host component validation changed")
    validation = read_json(validation_path)
    if (validation.get("passed") is not True or validation.get("execution_revision") != body["execution_revision"] or
            validation.get("component_sha256") != body["component_sha256"] or
            validation.get("scientific_contract_sha256") != body["scientific_contract_sha256"]):
        raise ActivationRefused("host validation is inconsistent with activation")
    _check_sources(Path(body["source_root"]), body["source_files"])
    registry = read_json(trust_root / "registry.json")
    identity = {"source_root": body["source_root"], "source_files": body["source_files"],
                "scientific_contract_sha256": body["scientific_contract_sha256"],
                "base_execution_revision": body["base_execution_revision"]}
    row = registry.get("revisions", {}).get(body["execution_revision"], {})
    if (registry.get("identity") != identity or require_registered and (
            row.get("sha256") != expected_sha256 or Path(row.get("sidecar", "")).absolute() != path.absolute())):
        raise ActivationRefused("signed activation is not registered by this host")
    if min(body["deadline_epoch"], registry["deadline_epoch"]) <= time.time():
        raise ActivationRefused("original global activation budget exhausted")
    return body


class ExecutionActivation:
    """Outer-host registry. Preparing a revision does not launch or activate it."""

    def __init__(self, root: Path, *, source_root: Path, source_files: Mapping[str, str],
                 scientific_contract: dict, base_execution_revision: str,
                 deadline_epoch: float):
        if not re.fullmatch(r"[a-f0-9]{64}", base_execution_revision) or not math.isfinite(deadline_epoch):
            raise ValueError("activation requires an immutable base revision and finite global deadline")
        self.root, self.source_root = Path(root).absolute(), Path(source_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.files = dict(source_files)
        self.scientific_hash = object_digest(scientific_contract)
        self.base_revision = base_execution_revision
        _check_sources(self.source_root, self.files)
        if any(name not in self.files for name in (TARGET, *HOST_FILES)):
            raise ActivationRefused("frozen source manifest omits the lifecycle component or trusted import hook")
        identity = {"source_root": str(self.source_root), "source_files": self.files,
                    "scientific_contract_sha256": self.scientific_hash, "base_execution_revision": self.base_revision}
        with exclusive(self.root / "host.lock"):
            path = self.root / "registry.json"
            if path.exists():
                state = read_json(path)
                if state["identity"] != identity:
                    raise ActivationRefused("host activation registry is bound to another frozen execution or science")
                state["deadline_epoch"] = min(state["deadline_epoch"], deadline_epoch)
            else:
                state = {"schema_version": 1, "identity": identity, "deadline_epoch": deadline_epoch, "revisions": {}}
            _signing_key(self.root, create=True)
            atomic_json(path, state)
            # A crash after the signed sidecar but before its registry index is
            # recoverable from host-authenticated evidence without re-execution.
            for sidecar in sorted((self.root / "revisions").glob("*.json")):
                if sidecar.stem not in state["revisions"]:
                    body = _signed_body(sidecar, expected_sha256=digest(sidecar), trust_root=self.root,
                                        require_registered=False)
                    if body["execution_revision"] != sidecar.stem:
                        raise ActivationRefused("orphan activation filename is inconsistent")
                    state["revisions"][sidecar.stem] = {"sidecar": str(sidecar), "sha256": digest(sidecar), "mode": body["mode"]}
            atomic_json(path, state)

    @property
    def deadline_epoch(self) -> float:
        return float(read_json(self.root / "registry.json")["deadline_epoch"])

    def _ensure(self) -> None:
        if self.deadline_epoch <= time.time():
            raise ActivationRefused("original global activation budget exhausted")
        _check_sources(self.source_root, self.files)

    def _publish(self, *, revision: str, component: Path, validation: dict,
                 mode: str, provenance: dict) -> dict:
        self._ensure()
        component_hash = digest(component)
        source = self.root / "components" / component_hash / "probe_barrier_contract.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and digest(source) != component_hash:
            raise ActivationRefused("registered component store was modified")
        if not source.exists():
            shutil.copyfile(component, source)
        check_pure_component(source.read_text(), FUNCTION)
        validation = {**validation, "schema_version": 1, "passed": True,
                      "execution_revision": revision, "scientific_contract_sha256": self.scientific_hash,
                      "source_files": self.files, "component_sha256": component_hash,
                      "scope": "pure_lifecycle_component", "native_readmission_required": True}
        validation_file = self.root / "validations" / f"{object_digest(validation)}.json"
        immutable_json(validation_file, validation)
        body = {"schema_version": 1, "mode": mode, "execution_revision": revision,
                "base_execution_revision": self.base_revision, "scientific_contract_sha256": self.scientific_hash,
                "component": COMPONENT, "function": FUNCTION, "component_sha256": component_hash,
                "component_file": str(source.relative_to(self.root)), "source_root": str(self.source_root),
                "source_files": self.files, "validation_file": str(validation_file.relative_to(self.root)),
                "validation_sha256": digest(validation_file), "deadline_epoch": self.deadline_epoch,
                "provenance": provenance, "issued_epoch": time.time(),
                "issuer": {"pid": os.getpid(), "host": os.uname().nodename}}
        sidecar = self.root / "revisions" / f"{revision}.json"
        if sidecar.exists():
            existing = _signed_body(sidecar, expected_sha256=digest(sidecar), trust_root=self.root)
            for key in ("execution_revision", "component_sha256", "source_files", "validation_sha256", "provenance"):
                if existing[key] != body[key]:
                    raise ActivationRefused("execution revision was already registered with different evidence")
        else:
            atomic_json(sidecar, {"body": body, "signature": _signature(body, _signing_key(self.root, create=False))})
        state = read_json(self.root / "registry.json")
        state["revisions"][revision] = {"sidecar": str(sidecar), "sha256": digest(sidecar), "mode": mode}
        atomic_json(self.root / "registry.json", state)
        return self._plan(revision, from_revision=None)

    def _plan(self, revision: str, *, from_revision: str | None, rollback: bool = False) -> dict:
        state = read_json(self.root / "registry.json")
        row = state["revisions"].get(revision)
        if row is None:
            raise ActivationRefused("execution revision has no host validation")
        sidecar = Path(row["sidecar"])
        body = _signed_body(sidecar, expected_sha256=row["sha256"], trust_root=self.root)
        validation = read_json(self.root / body["validation_file"])
        compatibility = None
        if from_revision is not None and from_revision != revision:
            if from_revision not in state["revisions"]:
                raise ActivationRefused("compatibility requires a previously registered source revision")
            compatibility = {"passed": True, "scope": "lifecycle_only", "from_revision": from_revision,
                             "to_revision": revision, "scientific_contract_sha256": self.scientific_hash,
                             "validation_receipt_sha256": body["validation_sha256"],
                             "reason": "Only the registered pure membership predicate changes; frozen science/source remain bound."}
        return {"execution_revision": revision, "validation_receipt": validation,
                "compatibility": compatibility, "rollback": rollback,
                "activation_file": str(sidecar), "activation_sha256": row["sha256"],
                "env": dict(zip(ENVIRONMENT_KEYS, (str(sidecar), row["sha256"], str(self.root), revision))),
                "native_readmission_required": True,
                "status": "prepared_for_new_process"}

    def register_baseline(self) -> dict:
        """Validate and register the actual immutable fallback for later rollback."""
        with exclusive(self.root / "host.lock"):
            self._ensure()
            if self.base_revision in read_json(self.root / "registry.json")["revisions"]:
                return self._plan(self.base_revision, from_revision=None)
            verdict = run_contract(self.source_root, membership_validation_spec(TARGET))
            if not verdict["passed"]:
                raise ActivationRefused("frozen fallback membership component failed the trusted contract")
            return self._publish(revision=self.base_revision, component=self.source_root / TARGET,
                                 validation={"component_contract": verdict}, mode="verified_baseline",
                                 provenance={"api_used": False, "base_component_sha256": self.files[TARGET]})

    def prepare_activation(self, request: dict, *, from_revision: str | None = None,
                           promote_historical_control: bool = False) -> dict:
        """Validate and pin an API candidate; never modify files or loaded modules.

        Historical promotion explicitly keeps two ancestries: the real defective
        control that the API fixed, and today's already-safe execution base.  The
        latter must independently pass the same host contract, as must the API
        component.  This is component promotion, not a claim that the API fixed
        the already-safe manual source.
        """
        self.register_baseline()
        with exclusive(self.root / "host.lock"):
            self._ensure()
            verify_activation_request(request)
            # A previously verified publication is replayable without creating
            # a second receipt with new timing measurements. It does not renew
            # either deadline or any API/repair attempt.
            if set(request.get("candidate_hashes", {})) == {TARGET}:
                revision = object_digest({"base_execution_revision": self.base_revision,
                    "component": COMPONENT, "component_sha256": request["candidate_hashes"][TARGET],
                    "scientific_contract_sha256": self.scientific_hash,
                    "contract_sha256": membership_validation_spec(TARGET).identity()})
                state = read_json(self.root / "registry.json")
                if revision in state["revisions"]:
                    row = state["revisions"][revision]
                    body = _signed_body(Path(row["sidecar"]), expected_sha256=row["sha256"], trust_root=self.root)
                    provenance = body["provenance"]
                    if (provenance.get("repair_manifest_sha256") != request.get("manifest_sha256") or
                            provenance.get("repair_validation_sha256") != request.get("validation_sha256")):
                        raise ActivationRefused("registered revision has different repair provenance")
                    if provenance["relation"] == "historical_control_to_validated_component_promotion" and not promote_historical_control:
                        raise ActivationRefused("historical component promotion must remain explicit")
                    return self._plan(revision, from_revision=from_revision or self.base_revision)
            if set(request.get("candidate_hashes", {})) != {TARGET} or set(request.get("base_hashes", {})) != {TARGET}:
                raise ActivationRefused("activation is limited to the one registered lifecycle helper")
            if request.get("pure_components") != {TARGET: FUNCTION}:
                raise ActivationRefused("candidate lacks the mandatory pure-component gate")
            base_matches = request["base_hashes"][TARGET] == self.files[TARGET]
            if not base_matches and not promote_historical_control:
                raise ActivationRefused("repair base differs from actual frozen execution base")
            old_verdict = read_json(Path(request["validation_path"]))
            contract = membership_validation_spec(TARGET)
            if not (old_verdict.get("baseline_reproduced") is True and old_verdict.get("passed") is True and
                    len(old_verdict.get("baseline", [])) == 1 and len(old_verdict.get("candidate", [])) == 1 and
                    old_verdict["baseline"][0].get("contract_sha256") == contract.identity() and
                    old_verdict["candidate"][0].get("contract_sha256") == contract.identity() and
                    old_verdict["baseline"][0].get("target_sha256") == request["base_hashes"][TARGET] and
                    old_verdict["candidate"][0].get("target_sha256") == request["candidate_hashes"][TARGET]):
                raise ActivationRefused("historical repair evidence does not bind this trusted contract and both code identities")
            candidate_root = Path(request["candidate_root"])
            candidate = candidate_root / TARGET
            check_pure_component(candidate.read_text(), FUNCTION)
            # Re-execute on the host, not merely trust a candidate-supplied report.
            current_verdict = run_contract(self.source_root, contract)
            candidate_verdict = run_contract(candidate_root, contract)
            if not current_verdict["passed"] or not candidate_verdict["passed"]:
                raise ActivationRefused("actual fallback and candidate must both satisfy the trusted lifecycle contract")
            if (digest(candidate) != request["candidate_hashes"][TARGET] or
                    candidate_verdict["target_sha256"] != request["candidate_hashes"][TARGET]):
                raise ActivationRefused("candidate changed during host validation")
            revision = object_digest({"base_execution_revision": self.base_revision,
                "component": COMPONENT, "component_sha256": digest(candidate),
                "scientific_contract_sha256": self.scientific_hash, "contract_sha256": contract.identity()})
            provenance = {"repair_incident_id": request["incident_id"], "candidate_id": request["candidate_id"],
                "repair_manifest_sha256": request["manifest_sha256"],
                "repair_validation_sha256": request["validation_sha256"],
                "repair_base_component_sha256": request["base_hashes"][TARGET],
                "execution_base_component_sha256": self.files[TARGET],
                "relation": "historical_control_to_validated_component_promotion" if not base_matches else "same_base_component_repair",
                "claim": "The selected component was validated against its defect control and today's fallback; no URDF engine root-cause claim."}
            self._publish(revision=revision, component=candidate,
                          validation={"current_base_contract": current_verdict,
                                      "candidate_contract": candidate_verdict, "repair_control": old_verdict},
                          mode="validated_repair_component", provenance=provenance)
            return self._plan(revision, from_revision=from_revision or self.base_revision)

    def prepare_rollback(self, *, from_revision: str, target_revision: str | None = None) -> dict:
        with exclusive(self.root / "host.lock"):
            self._ensure()
            target = target_revision or self.base_revision
            state = read_json(self.root / "registry.json")
            if target not in state["revisions"] or from_revision not in state["revisions"]:
                raise ActivationRefused("rollback requires two already validated revisions")
            return self._plan(target, from_revision=from_revision, rollback=True)

    def current_context(self, revision: str, *, from_revision: str | None = None) -> dict:
        """Re-read one previously verified revision; caller names actual revision."""
        with exclusive(self.root / "host.lock"):
            self._ensure()
            return self._plan(revision, from_revision=from_revision)


def verify_prepared_activation(plan: dict) -> dict:
    """Call immediately before ContinuationSupervisor.run_once with plan env."""
    env = plan.get("env", {})
    if set(env) != set(ENVIRONMENT_KEYS):
        raise ActivationRefused("prepared activation lacks its complete immutable environment identity")
    body = _signed_body(Path(env[ENVIRONMENT_KEYS[0]]), expected_sha256=env[ENVIRONMENT_KEYS[1]],
                        trust_root=Path(env[ENVIRONMENT_KEYS[2]]))
    if (body["execution_revision"] != plan["execution_revision"] or
            env[ENVIRONMENT_KEYS[3]] != body["execution_revision"] or body["deadline_epoch"] <= time.time()):
        raise ActivationRefused("prepared activation revision or original deadline changed")
    if object_digest(plan["validation_receipt"]) != object_digest(read_json(Path(env[ENVIRONMENT_KEYS[2]]) / body["validation_file"])):
        raise ActivationRefused("caller validation differs from signed host evidence")
    return body


def install_activation_from_environment() -> dict | None:
    """Only called at research-package first import, before coordinator imports."""
    global _ACTIVE_IDENTITY
    values = tuple(os.environ.get(name) for name in ENVIRONMENT_KEYS)
    if not any(values):
        return None
    if not all(values):
        raise ActivationRefused("partial execution activation environment")
    package = sys.modules.get("autosim.research")
    if getattr(package, "_ACTIVATION_ENV_AT_IMPORT", None) != values:
        raise ActivationRefused("activation environment changed after research package import; start a new process")
    if _ACTIVE_IDENTITY is not None:
        if _ACTIVE_IDENTITY["activation_sha256"] != values[1]:
            raise ActivationRefused("cannot hot-reload an active lifecycle component")
        return dict(_ACTIVE_IDENTITY)
    if COMPONENT in sys.modules:
        raise ActivationRefused("lifecycle component was already imported; start a new process")
    sidecar, expected, trust, revision = values
    trust_root = Path(trust)
    body = _signed_body(Path(sidecar), expected_sha256=expected, trust_root=trust_root)
    if body["execution_revision"] != revision or body["deadline_epoch"] <= time.time():
        raise ActivationRefused("new process activation identity or deadline is invalid")
    if body["issuer"]["host"] == os.uname().nodename and body["issuer"]["pid"] == os.getpid():
        raise ActivationRefused("issuing host cannot activate a component in its existing process")
    component = trust_root / body["component_file"]
    source_root = Path(body["source_root"]).resolve()
    loaded_host_files = (Path(package.__file__).resolve(), Path(__file__).resolve())
    if any(path != source_root / relative or body["source_files"].get(relative) != digest(path)
           for path, relative in zip(loaded_host_files, HOST_FILES)):
        raise ActivationRefused("actual trusted import hook is not the pinned frozen execution source")
    source = component.read_bytes()
    if hashlib.sha256(source).hexdigest() != body["component_sha256"]:
        raise ActivationRefused("component changed immediately before first import")
    check_pure_component(source.decode(), FUNCTION)
    module = types.ModuleType(COMPONENT)
    module.__file__ = str(component)
    module.__package__ = "autosim.research"
    exec(compile(source, str(component), "exec"), module.__dict__)
    module.__activation_revision__ = revision
    module.__activation_sha256__ = body["component_sha256"]
    sys.modules[COMPONENT] = module
    setattr(package, "probe_barrier_contract", module)
    _ACTIVE_IDENTITY = {"execution_revision": revision, "base_execution_revision": body["base_execution_revision"],
                        "component": COMPONENT, "component_sha256": body["component_sha256"],
                        "activation_sha256": expected, "scientific_contract_sha256": body["scientific_contract_sha256"],
                        "loaded_pid": os.getpid(), "host": os.uname().nodename, "loaded_epoch": time.time()}
    filename = object_digest(_ACTIVE_IDENTITY) + ".json"
    atomic_json(trust_root / "load_receipts" / filename, _ACTIVE_IDENTITY)
    return dict(_ACTIVE_IDENTITY)


def active_execution_identity() -> dict | None:
    """Actual loaded identity; changing environment never silently changes code."""
    if _ACTIVE_IDENTITY is None:
        if os.environ.get(ENVIRONMENT_KEYS[0]):
            raise ActivationRefused("activation was set after import; a new Python process is required")
        return None
    if tuple(os.environ.get(name) for name in ENVIRONMENT_KEYS)[1] != _ACTIVE_IDENTITY["activation_sha256"]:
        raise ActivationRefused("activation environment changed in an already-running process")
    return dict(_ACTIVE_IDENTITY)
