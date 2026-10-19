"""Native-format-neutral episode selection; never invent a robot expert or labels.

Manifests describe actual committed files and observed conditions. The native writer /
read-only inspector produces them; Supervisor still reviews action semantics and the
training domain. Selection returns a reversible view, not a converter or delete command.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .common import digest, object_digest
from .experiment_bundle import artifact_identity


def finite_json(value, *, maximum=4096):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("expected bounded finite JSON") from exc
    if len(encoded.encode()) > maximum:
        raise ValueError("JSON value exceeds bound")
    return value


def controls(value):
    if not isinstance(value, dict) or not 1 <= len(value) <= 12:
        raise ValueError("collection controls require 1..12 native fields")
    for name, setting in value.items():
        if (not isinstance(name, str) or len(name) > 160 or
                not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) or ".." in name):
            raise ValueError("control name must be a scoped native field, not a path or command")
        finite_json(setting, maximum=2000)
    finite_json(value, maximum=6000)
    return dict(value)


def predicate(value):
    if (not isinstance(value, dict) or set(value) != {"field", "op", "value", "unit"}
            or not isinstance(value["field"], str) or not value["field"].strip()
            or len(value["field"]) > 160 or not isinstance(value["unit"], str)
            or not 1 <= len(value["unit"]) <= 80):
        raise ValueError("predicate requires an observed field, op, finite value and unit")
    op, v = value["op"], finite_json(value["value"])
    if op not in {"eq", "in", "between", "lt", "le", "gt", "ge"}:
        raise ValueError("unsupported declarative predicate; executable expressions are forbidden")
    numeric = lambda x: type(x) in (int, float) and math.isfinite(x)
    if op == "between" and (not isinstance(v, list) or len(v) != 2
                             or not all(numeric(x) for x in v) or v[0] > v[1]):
        raise ValueError("between requires ordered finite bounds")
    if op in {"lt", "le", "gt", "ge"} and not numeric(v):
        raise ValueError("ordered predicate requires a finite number")
    if op == "in" and (not isinstance(v, list) or not 1 <= len(v) <= 64):
        raise ValueError("in requires a bounded nonempty list")
    return dict(value)


def matches(condition, target):
    """None means unobserved/incompatible; never count it as a negative observation."""
    target = predicate(target)
    if (not isinstance(condition, dict) or set(condition) - {"value", "unit", "source_ref"}
            or condition.get("unit") != target["unit"] or "value" not in condition):
        return None
    x, v, op = condition["value"], target["value"], target["op"]
    finite_json(x)
    if op == "eq":
        return type(x) is type(v) and x == v
    if op == "in":
        return any(type(x) is type(item) and x == item for item in v)
    if type(x) not in (int, float) or not math.isfinite(x):
        return None
    return {"between": lambda: v[0] <= x <= v[1], "lt": lambda: x < v,
            "le": lambda: x <= v, "gt": lambda: x > v, "ge": lambda: x >= v}[op]()


def validate_manifest(value, *, root=None):
    if (not isinstance(value, dict) or set(value) != {"schema_version", "identity_field",
            "identity", "scope", "episodes"} or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or value["scope"] not in {"full", "sampled"}
            or value["identity_field"] not in {"sha256", "content_sha256"}
            or not isinstance(value["identity"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["identity"])
            or not isinstance(value["episodes"], list) or len(value["episodes"]) > 10000):
        raise ValueError("invalid bounded episode manifest")
    identifiers = set()
    file_identities = {}
    rows = []
    for row in value["episodes"]:
        required = {"id", "committed", "files", "format_valid", "source_family",
                    "source_instance", "split", "supervision", "conditions"}
        if (not isinstance(row, dict) or not required <= set(row) or set(row) - required - {'content_sha256'}
                or not isinstance(row["id"], str) or not 1 <= len(row["id"]) <= 100
                or row["id"] in identifiers or type(row["committed"]) is not bool
                or row["format_valid"] not in (True, False, None)
                or (row["format_valid"] is not None and type(row["format_valid"]) is not bool)):
            raise ValueError("invalid, duplicate or unsupported episode row")
        identifiers.add(row["id"])
        if row.get('content_sha256') is not None and not re.fullmatch(r'[0-9a-f]{64}', str(row['content_sha256'])):
            raise ValueError('invalid native episode-content identity')
        for key in ("source_family", "source_instance", "split", "supervision"):
            if row[key] is not None and (not isinstance(row[key], str) or not 1 <= len(row[key]) <= 160):
                raise ValueError("episode provenance must be bounded text or explicit unknown")
        if (not isinstance(row["files"], dict) or len(row["files"]) > 32
                or (row["committed"] and not row["files"])):
            raise ValueError("episode requires committed file identities")
        for name, sha in row["files"].items():
            if not isinstance(name, str):
                raise ValueError("episode file names must be text")
            rel = Path(name)
            if (not isinstance(name, str) or not name or len(name) > 512 or rel.is_absolute()
                    or ".." in rel.parts or "\\" in name or
                    not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("unsafe episode file identity")
            if root is not None:
                if name not in file_identities:
                    path = Path(root) / rel
                    if (any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file()
                            or not path.resolve().is_relative_to(Path(root).resolve())):
                        raise ValueError("committed episode file changed or escaped its root")
                    file_identities[name] = digest(path)
                # HDF5 and other native containers can hold many episodes. Hash a
                # shared file once per validation, not once per episode; final
                # dataset identity below still detects changes across the scan.
                if file_identities[name] != sha:
                    raise ValueError("committed episode file changed or escaped its root")
        if not isinstance(row["conditions"], dict) or len(row["conditions"]) > 32:
            raise ValueError("episode conditions require a bounded object")
        for field, observation in row["conditions"].items():
            if (not isinstance(field, str) or not field or len(field) > 160
                    or not isinstance(observation, dict)
                    or set(observation) != {"value", "unit", "source_ref"}
                    or any(not isinstance(observation[k], str) or not 1 <= len(observation[k]) <= 300
                           for k in ("unit", "source_ref"))):
                raise ValueError("conditions need native values, units and measurement source")
            finite_json(observation["value"])
        rows.append(row)
    if root is not None and artifact_identity(Path(root)) != (value["identity_field"], value["identity"]):
        raise ValueError("manifest dataset bytes changed")
    return {**value, "episodes": rows}


def selection(manifest, *, target=None, limit=None, deduplicate="bytes", require_split="train",
              require_supervision=None):
    """Reusable Q0-format / Qt view. Caller chooses criteria from native source evidence.

    Missing provenance stays missing. Native observations are not inferred from file names,
    success is not inferred from readable parquet, and different hashes do not prove new families.
    """
    value = validate_manifest(manifest)
    if target is not None:
        target = predicate(target)
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 10000):
        raise ValueError("selection limit must be a positive bounded episode count")
    if deduplicate not in {"bytes", "source_instance", "none"}:
        raise ValueError("unsupported deduplication unit")
    for name, criterion in (('require_split', require_split), ('require_supervision', require_supervision)):
        if criterion is not None and (not isinstance(criterion, str) or not 1 <= len(criterion) <= 160):
            raise ValueError(name + ' must be bounded native text or explicit unknown')
    selected, rejected, unknown, hashes, sources, families = [], {}, {}, set(), set(), set()
    conditions_observed, conditions_matched = 0, 0
    for row in value["episodes"]:
        why = None
        if not row["committed"] or row["format_valid"] is not True:
            why = "uncommitted_or_format_unverified"
        elif require_split is not None and row["split"] != require_split:
            why = "training_split_unverified"
        elif require_supervision is not None and row["supervision"] != require_supervision:
            why = "supervision_unverified"
        content = row.get('content_sha256')
        source = (row["source_family"], row["source_instance"])
        if why is None and deduplicate == "bytes":
            if content is None:
                why = "episode_content_identity_unverified"
            elif content in hashes:
                why = "duplicate_content"
        if why is None and deduplicate == "source_instance":
            if None in source:
                why = "source_instance_unverified"
            elif source in sources:
                why = "duplicate_source_instance"
        if why is None and target:
            match = matches(row["conditions"].get(target["field"]), target)
            if match is None:
                why = "target_condition_unobserved"
            else:
                conditions_observed += 1
                conditions_matched += int(match)
                if not match:
                    why = "outside_target"
        if why is None and limit is not None and len(selected) >= limit:
            why = "selection_limit"
        if why:
            rejected[why] = rejected.get(why, 0) + 1
            continue
        selected.append(row["id"])
        if content is not None:
            hashes.add(content)
        if None not in source:
            sources.add(source)
        if row["source_family"] is not None:
            families.add(row["source_family"])
        for key in ("source_family", "source_instance", "supervision"):
            if row[key] is None:
                unknown[key] = unknown.get(key, 0) + 1
    return {"schema_version": 1, "manifest_sha256": object_digest(value),
            "data_identity": {value["identity_field"]: value["identity"]},
            "scope": value["scope"], "selected_episode_ids": selected,
            "rejected": rejected, "unknown": unknown,
            "counts": {"inspected": len(value["episodes"]), "selected": len(selected),
                "known_source_families": len(families), "target_observed": conditions_observed,
                "target_matched": conditions_matched},
            "criteria": {"target": target, "limit": limit, "deduplicate": deduplicate,
                "require_split": require_split, "require_supervision": require_supervision},
            "preserves_episode_groups": True,
            "authority": "reversible native-format view; no label quality, consumption or score certification"}


def coverage_change(before, after, target):
    """Observed probe contrast, not causal proof that a control worked."""
    target = predicate(target)
    rows = []
    for value in (before, after):
        manifest = validate_manifest(value)
        observations = [matches(row['conditions'].get(target['field']), target)
                        for row in manifest['episodes'] if row['committed'] and row['format_valid'] is True]
        known = [v for v in observations if v is not None]
        rows.append({'manifest_sha256':object_digest(manifest), 'scope':manifest['scope'],
                     'observed':len(known), 'matched':sum(known),
                     'unknown':len(observations)-len(known),
                     'rate':sum(known)/len(known) if known else None})
    if any(row['rate'] is None or row['unknown'] for row in rows):
        status = 'needs_native_observation'
    elif rows[0]['rate'] == rows[1]['rate']:
        status = 'no_observed_coverage_change'
    else:
        status = 'observed_coverage_change'
    return {'status':status,'target':target,'before':rows[0],'after':rows[1],
        'authority':'actual recorded-condition contrast; source routing, seed/domain matching and expert semantics still need review'}


def sealed_coverage_change(output, after, request):
    """Compare to one prior sealed native manifest, not model-supplied observations."""
    from .data_lineage import _record
    from .data_health import PROFILE_GENERATORS
    from .evidence_store import read_attempt_evidence
    if (not isinstance(request, dict) or set(request) != {'evidence_id', 'target'}
            or not isinstance(request['evidence_id'], str)
            or not re.fullmatch(r'[0-9a-f]{32}', request['evidence_id'])):
        raise ValueError('coverage reference requires a native evidence ID and declarative target')
    output = Path(output)
    verified = read_attempt_evidence(output, request['evidence_id'], limit=1)
    path, row = _record(output, verified['receipt_ref'], max_bytes=2 * 1024**2)
    if (verified['status'] != 'completed' or verified['returncode'] != 0
            or row.get('data_profile_generator') not in PROFILE_GENERATORS
            or row.get('data_profile_status') or row.get('gpu_access') is not False
            or row.get('repository_source_stable') is not True
            or row.get('native_context_stable') is False
            or ('diagnostic_data_binding' in row and row.get('diagnostic_data_binding_stable') is not True)):
        raise ValueError('coverage reference must be a successful stable read-only native observation')
    result = coverage_change(row.get('data_profile', {}).get('episode_manifest'), after, request['target'])
    record, _ = _record(output, f'evidence/{request["evidence_id"]}.json')
    return {**result, 'reference': {'evidence_id': request['evidence_id'],
        'evidence_sha256': digest(record), 'receipt_ref': verified['receipt_ref'],
        'receipt_sha256': digest(path)}}
