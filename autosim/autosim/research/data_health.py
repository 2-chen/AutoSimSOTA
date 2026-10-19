"""Explicit format-level, read-only checks; never infer expert quality or score.

LeRobot v2 is a data format, not a benchmark branch. Agents choose whether its
metadata describes their producer, then still verify native conversion/loader
and source/label semantics. Heavy optional dependencies are imported on demand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

LEROBOT_PROFILE_GENERATOR = 'autosim_lerobot_v2_readonly_manifest_v2'
PROFILE_GENERATORS = frozenset({'autosim_lerobot_v2_readonly_manifest_v1',
                              LEROBOT_PROFILE_GENERATOR, 'autosim_native_source_manifest_v1'})


def lerobot_v2_profile(root: Path, *, max_episodes: int = 10000,
                       decode_videos: bool = True, include_manifest: bool = False,
                       manifest_fields: dict | None = None) -> dict:
    """Check committed episode rows and declared videos without modifying data.

    A bounded partial scan reports sampled, never full. Schema, labels, split
    legality, generation controls and downstream utility remain separate checks.
    """
    if type(max_episodes) is not int or not 1 <= max_episodes <= 10000:
        raise ValueError("max_episodes must be a bounded positive integer")
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("data root must be an explicit regular directory")
    root = root.resolve()
    fields = manifest_fields or {}
    if (not isinstance(fields, dict) or set(fields) -
            {"source_family", "source_instance", "split", "supervision", "conditions"}):
        raise ValueError("invalid native metadata field mapping")
    for key, value in fields.items():
        if key != "conditions" and (not isinstance(value, str) or not value or len(value) > 160):
            raise ValueError("provenance mapping must name a native episode metadata field")
    conditions = fields.get("conditions", {})
    if not isinstance(conditions, dict) or len(conditions) > 12:
        raise ValueError("condition mapping is not bounded")
    for name, spec in conditions.items():
        if (not isinstance(name, str) or not name or len(name) > 160
                or not isinstance(spec, dict) or set(spec) != {"metadata_field", "unit", "source_ref"}
                or any(not isinstance(v, str) or not 1 <= len(v) <= 300 for v in spec.values())):
            raise ValueError("condition mapping requires a native field, unit and measurement source")

    def path(relative):
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("dataset metadata path escapes its root")
        target = root / p
        if any((root / Path(*p.parts[:n])).is_symlink() for n in range(1, len(p.parts) + 1)):
            raise ValueError("dataset metadata path is a symlink")
        return target

    def metadata(relative):
        p = path(relative)
        if p.stat().st_size > 8 * 1024**2:
            raise ValueError("dataset metadata exceeds bounded size")
        return p.read_text(encoding="utf-8")

    info = json.loads(metadata("meta/info.json"))
    if not isinstance(info, dict) or info.get("codebase_version") != "v2.1":
        raise ValueError("this checker requires explicit LeRobot v2.1 metadata")
    episodes = [json.loads(line) for line in metadata("meta/episodes.jsonl").splitlines() if line.strip()]
    if len(episodes) > 100000:
        raise ValueError("episode metadata exceeds bounded population")
    import pyarrow.parquet as pq
    features = info.get("features") or {}
    cameras = [key for key, value in features.items() if value.get("dtype") == "video"]
    av = None
    if cameras and decode_videos:
        import av
    issues, checked, frames, videos, decoded = [], 0, 0, 0, 0
    indices = set()
    data_sha = set()
    import hashlib
    manifest_rows, manifest_bytes = [], 0
    action_groups = {}
    fps = info.get('fps')
    fps_valid = type(fps) in (int, float) and math.isfinite(fps) and fps > 0

    def shape_matches(value, shape, *, scalar=True):
        # LeRobot scalar columns declare [1]; matrix/vector columns retain axes.
        if not shape:
            return not isinstance(value, list)
        if scalar and shape == [1] and not isinstance(value, list):
            return True
        return (isinstance(value, list) and len(value) == shape[0] and
                all(shape_matches(v, shape[1:], scalar=False) for v in value))

    def finite_numbers(value):
        if isinstance(value, list):
            return all(finite_numbers(v) for v in value)
        return type(value) in (int, float) and math.isfinite(value)

    def nonfinite_leaf(value):
        if isinstance(value, list):
            return any(nonfinite_leaf(v) for v in value)
        return type(value) in (int, float) and not math.isfinite(value)

    def file_hash(p):
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
        return h.hexdigest()

    def issue(episode, reason):
        if len(issues) < 128:
            issues.append({"episode_index": episode, "reason": reason})

    if not episodes:
        issue(None, "empty committed episode population; not a usable training pool")

    for row in episodes[:max_episodes]:
        index, length = row.get("episode_index"), row.get("length")
        prior_issues, committed_files = len(issues), {}
        checked += 1
        if type(index) is not int or index < 0 or type(length) is not int or not 1 <= length <= 1000000:
            issue(index, "invalid episode index or length")
            continue
        if index in indices:
            issue(index, "duplicate episode index")
        indices.add(index)
        if 'fps' in info and not fps_valid:
            issue(index, 'declared frame rate must be finite and positive')
        chunks = info.get("chunks_size", 1000)
        if type(chunks) is not int or chunks <= 0:
            raise ValueError("invalid data chunk size")
        values = {"episode_index": index, "episode_chunk": index // chunks}
        try:
            data = path(info["data_path"].format(**values))
            if data.stat().st_size > 256 * 1024**2:
                raise ValueError("episode parquet exceeds bounded size")
            h = hashlib.sha256()
            with data.open("rb") as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    h.update(block)
            if h.hexdigest() in data_sha:
                issue(index, "duplicate parquet bytes; source-family dedup still required")
            data_sha.add(h.hexdigest())
            if include_manifest:
                committed_files[str(data.relative_to(root))] = h.hexdigest()
            native = pq.ParquetFile(data)
            if native.metadata.num_rows > 1000000:
                raise ValueError("episode row population exceeds bounded scan")
            frames += native.metadata.num_rows
            if native.metadata.num_rows != length:
                issue(index, "parquet row count differs from committed length")
            for field in features:
                if field not in cameras and field not in native.schema_arrow.names:
                    issue(index, "declared feature missing: " + field)
            previous_time, count, times = None, 0, []
            numeric_fields = {field: feature for field, feature in features.items()
                              if str(feature.get('dtype') or '').startswith(('float', 'int', 'uint'))}
            action_hash = hashlib.sha256() if 'action' in numeric_fields else None
            action_valid = action_hash is not None
            for batch in native.iter_batches(batch_size=4096):
                for values_row in batch.to_pylist():
                    if values_row.get("episode_index") != index or values_row.get("frame_index") != count:
                        issue(index, "episode/frame indices are not aligned")
                    time = values_row.get("timestamp")
                    if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or (previous_time is not None and time <= previous_time):
                        issue(index, "timestamps are invalid or non-increasing")
                    if (fps_valid and count and finite_numbers(times[0]) and finite_numbers(time) and finite_numbers(previous_time)
                            and not math.isclose(time, times[0] + count / fps, rel_tol=2e-7, abs_tol=1e-5)):
                        issue(index, 'timestamps differ from declared frame rate')
                    previous_time = time
                    times.append(time)
                    for field, feature in numeric_fields.items():
                        value = values_row.get(field)
                        if value is None:
                            issue(index, "null numeric feature: " + field)
                        elif not finite_numbers(value):
                            issue(index, 'nonfinite or nonnumeric numeric feature: ' + field)
                        shape = feature.get('shape')
                        if (isinstance(shape, list) and shape and
                                all(type(n) is int and n > 0 for n in shape) and
                                not shape_matches(value, shape)):
                            issue(index, 'numeric feature dimension differs: ' + field)
                    for field, value in values_row.items():
                        if nonfinite_leaf(value):
                            issue(index, "nonfinite numeric feature: " + field)
                    if action_hash is not None:
                        action = values_row.get('action')
                        if finite_numbers(action):
                            action_hash.update(json.dumps(action, separators=(',', ':'), allow_nan=False).encode())
                            action_hash.update(b'\n')
                        else:
                            action_valid = False
                    count += 1
            if action_valid and count == length:
                action_groups.setdefault(action_hash.hexdigest(), []).append(index)
            for camera in cameras:
                video = path(info["video_path"].format(**values, video_key=camera))
                if not video.is_file():
                    raise ValueError("declared video missing: " + camera)
                videos += 1
                if include_manifest:
                    committed_files[str(video.relative_to(root))] = file_hash(video)
                if av is not None:
                    with av.open(str(video)) as container:
                        if len(container.streams.video) != 1:
                            raise ValueError("video requires exactly one video stream")
                        stream = container.streams.video[0]
                        stream.thread_count = 1  # CPU diagnostics must not contend with the trainer.
                        if (fps_valid and stream.average_rate is not None and
                                not math.isclose(float(stream.average_rate), fps, rel_tol=1e-5)):
                            issue(index, 'video rate differs from metadata: ' + camera)
                        n = 0
                        for frame in container.decode(video=0):
                            feature = features[camera]
                            names, shape = feature.get("names") or [], feature.get("shape") or []
                            if isinstance(names, list) and isinstance(shape, list) and len(names) == len(shape):
                                dimensions = dict(zip(names, shape))
                                if (dimensions.get("height", frame.height) != frame.height or
                                        dimensions.get("width", frame.width) != frame.width):
                                    issue(index, "video dimensions differ from metadata: " + camera)
                            pts = frame.time
                            if (n < len(times) and finite_numbers(times[n]) and
                                    (pts is None or not math.isclose(pts, times[n], rel_tol=2e-7, abs_tol=1e-5))):
                                issue(index, 'video presentation timestamp differs from parquet: ' + camera)
                            n += 1
                            if n > length:
                                break
                        decoded += 1
                        if n != length:
                            issue(index, "decoded frame count differs from committed length: " + camera)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issue(index, str(exc)[:240])
        if include_manifest:
            native_conditions = {}
            for name, spec in conditions.items():
                observed = row.get(spec["metadata_field"])
                if observed is not None:
                    json.dumps(observed, allow_nan=False)
                    native_conditions[name] = {"value": observed, "unit": spec["unit"],
                                               "source_ref": spec["source_ref"]}
            def provenance(key):
                v = row.get(fields.get(key, ""))
                return str(v) if type(v) in (str, int) and 1 <= len(str(v)) <= 160 else None
            item = {"id": str(index), "committed": len(committed_files) == 1 + len(cameras), "files": committed_files,
                "format_valid": len(issues) == prior_issues and len(issues) < 128 and
                    (decode_videos or not cameras),
                **{key: provenance(key) for key in ("source_family", "source_instance", "split", "supervision")},
                "conditions": native_conditions}
            item['content_sha256'] = hashlib.sha256(json.dumps(
                sorted(committed_files.values()), separators=(',', ':')).encode()).hexdigest() if item['committed'] else None
            size = len(json.dumps(item, ensure_ascii=False).encode())
            if manifest_bytes + size <= 60000:
                manifest_rows.append(item)
                manifest_bytes += size
    full = checked == len(episodes)
    if full and info.get("total_episodes") != len(episodes):
        issue(None, "declared total episodes differs from metadata population")
    if full and info.get("total_frames") != frames:
        issue(None, "declared total frames differs from parquet population")
    result = {"schema_version": 1, "format": "lerobot_v2.1",
            "scope": "full" if full else "sampled", "population_episodes": len(episodes),
            "counts": {"episodes_checked": checked, "parquet_frames": frames,
                       "videos_present": videos, "videos_decoded": decoded},
            "issues": issues, "issues_truncated": len(issues) >= 128,
            "video_decode_scope": ("not_checked" if not decode_videos else
                "full" if full and decoded == checked * len(cameras) else
                "sampled" if not full and decoded == checked * len(cameras) else "incomplete"),
            "unchecked": ["expert labels and success", "action units/frame", "source-family independence",
                          "target coverage", "native loader consumption", "policy improvement"],
            "authority": "format health only; no quality, training or SOTA certification"}
    repeated = [group for group in action_groups.values() if len(group) > 1]
    result['diagnostics'] = {'action_sequences_checked': sum(map(len, action_groups.values())),
        'unique_exact_action_sequences': len(action_groups),
        'repeated_action_episodes': sum(len(group) - 1 for group in repeated),
        'repeated_action_groups_preview': [group[:16] for group in repeated[:8]],
        'scope': result['scope'],
        'declared_fps_checked': fps_valid,
        'authority': 'Exact action repetition only; different observations may remain useful. '
                     'Not automatic rejection, source-family deduplication or coverage proof.'}
    if include_manifest:
        all_files = sorted(p for p in root.rglob("*") if p.is_file())
        if (len(all_files) > 10000 or any(p.is_symlink() for p in root.rglob("*"))
                or sum(p.stat().st_size for p in all_files) > 64 * 1024**3):
            raise ValueError("manifest dataset scan exceeds bounds or contains symlinks")
        pairs = [(str(p.relative_to(root)), file_hash(p)) for p in all_files]
        identity = hashlib.sha256(json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result["episode_manifest"] = {"schema_version": 1, "identity_field": "sha256",
            "identity": identity, "scope": "full" if full and len(manifest_rows) == len(episodes) else "sampled",
            "episodes": manifest_rows}
        result["manifest_status"] = "complete" if result["episode_manifest"]["scope"] == "full" else "bounded_partial"
    return result


def inspector_source(workspace: Path, ref: str) -> dict:
    from .common import digest
    from .frozen_policy_trial import _safe
    if not isinstance(ref, str) or not ref.endswith('.py') or len(ref) > 512:
        raise ValueError('source manifest inspector must be an explicit checkout-relative Python source')
    path = _safe(Path(workspace), ref)
    if not path.is_file() or not 1 <= path.stat().st_size <= 16000:
        raise ValueError('native manifest inspector source exceeds the 16 KiB bound or is missing')
    return {'ref': ref, 'sha256': digest(path), 'code': path.read_text(encoding='utf-8')}


def native_profile_code(request: dict, *, workspace: Path | None = None) -> str:
    """Embed the trusted checker in the existing native read-only CPU tool.

    No new tool, arbitrary host mounts, environment installation or file writes.
    """
    import inspect
    if not isinstance(request, dict) or set(request) - {"format", "root", "max_episodes", "decode_videos",
                                                      "include_manifest", "manifest_fields", "selection", "producer", "coverage_reference", "inspector_ref"}:
        raise ValueError("invalid data profile request fields")
    if request.get("format") not in {"lerobot_v2.1", "source_manifest"}:
        raise ValueError("unsupported explicit data format; use source-backed native checks")
    relative = request.get("root")
    if (not isinstance(relative, str) or not relative or len(relative) > 512 or
            Path(relative).is_absolute() or ".." in Path(relative).parts or
            "\\" in relative or "\x00" in relative):
        raise ValueError("data profile root must be a checkout/resource-relative path")
    producer = request.get('producer')
    if (relative == '@producer') != (producer is not None):
        raise ValueError('root=@producer requires a sealed producer grant; do not invent host paths')
    if request['format'] == 'source_manifest':
        if workspace is None:
            raise ValueError('source_manifest needs the actual isolated inspector workspace')
        if any(k in request for k in ('max_episodes', 'decode_videos', 'manifest_fields')):
            raise ValueError('native inspector owns its explicit scan/decoder semantics; do not mix format-specific fields')
        inspector = inspector_source(workspace, request.get('inspector_ref'))
        if 'coverage_reference' in request:
            from .data_quality import predicate
            ref = request['coverage_reference']
            if (not isinstance(ref, dict) or set(ref) != {'evidence_id', 'target'}):
                raise ValueError('coverage reference needs evidence_id and target')
            predicate(ref['target'])
        if 'selection' in request and (not isinstance(request['selection'], dict) or set(request['selection']) -
                {'target', 'limit', 'deduplicate', 'require_split', 'require_supervision'}):
            raise ValueError('invalid declarative episode selection')
        data = "os.environ['AUTOSIM_DIAGNOSTIC_DATA']" if producer is not None else repr(relative)
        return ("import os, runpy\nfrom pathlib import Path\n"
                f"os.environ['AUTOSIM_DATA_ROOT'] = str(Path({data}).resolve())\n"
                f"runpy.run_path({inspector['ref']!r}, run_name='__main__')\n")
    if 'inspector_ref' in request:
        raise ValueError('inspector_ref belongs only to source_manifest')
    maximum = request.get("max_episodes", 100)
    decode = request.get("decode_videos", True)
    if type(request.get("include_manifest", False)) is not bool:
        raise ValueError("include_manifest must be a boolean")
    manifest = request.get("include_manifest", False) or "selection" in request or "coverage_reference" in request
    if type(maximum) is not int or not 1 <= maximum <= 10000 or type(decode) is not bool:
        raise ValueError("invalid data profile scan bounds")
    if type(manifest) is not bool:
        raise ValueError("include_manifest must be a boolean")
    fields = request.get("manifest_fields")
    if fields is not None and not isinstance(fields, dict):
        raise ValueError("manifest_fields must be an object")
    from .data_quality import finite_json
    finite_json(fields, maximum=2500)
    if 'coverage_reference' in request:
        ref = request['coverage_reference']
        from .data_quality import predicate
        import re
        if (not isinstance(ref, dict) or set(ref) != {'evidence_id', 'target'}
                or not isinstance(ref['evidence_id'], str) or not re.fullmatch(r'[0-9a-f]{32}', ref['evidence_id'])):
            raise ValueError('coverage_reference requires a sealed native evidence ID and target')
        predicate(ref['target'])
    if "selection" in request:
        if not isinstance(request["selection"], dict) or set(request["selection"]) - {
                "target", "limit", "deduplicate", "require_split", "require_supervision"}:
            raise ValueError("invalid declarative episode selection")
    root_expression = "Path(__import__('os').environ['AUTOSIM_DIAGNOSTIC_DATA'])" if producer is not None else f"Path({relative!r})"
    # Serialize the trusted checker without comments/docstrings, not a lossy
    # summary. Preserve execution semantics and the existing diagnostic bound.
    import ast
    tree = ast.parse(inspect.getsource(lerobot_v2_profile))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.body and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            node.body.pop(0)
    rendered = ast.unparse(tree)
    compact = '\n'.join(' ' * ((len(line) - len(line.lstrip(' '))) // 4) + line.lstrip(' ')
                        for line in rendered.splitlines())
    if ast.dump(ast.parse(compact)) != ast.dump(tree):
        raise ValueError('native data checker serialization changed execution semantics')
    code = ("import json, math\nfrom pathlib import Path\n" + compact +
            f"\nprint(json.dumps(lerobot_v2_profile({root_expression}, max_episodes={maximum}, "
            f"decode_videos={decode!r}, include_manifest={manifest!r}, manifest_fields={fields!r}), ensure_ascii=False))\n")
    if len(code.encode()) > 16000:
        raise ValueError('native data checker request exceeds the 16000-byte diagnostic bound')
    return code
