"""Opaque, reversible run-local path references, separate from display redaction."""
import re
from pathlib import Path

PARENT = re.compile(r"(?<![\w.])(?:\.\./)+[\w.@~+/-]+")
TOKEN = re.compile(r"\{run_path_\d+\}")


class ExecutionPaths:
    def __init__(self, repo: Path, output: Path):
        self.repo = repo.resolve()
        self.output = output.resolve()
        self.paths = {}

    def bind_run_reference(self, reference):
        """Turn a resource inventory reference into an executable, run-root-bound handle."""
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("resource reference must be run-relative")
        target = (self.output / relative).resolve()
        if not target.is_relative_to(self.output):
            raise ValueError("resource reference escaped the run")
        alias = next((key for key, value in self.paths.items() if value == target), None)
        if alias is None:
            alias = "{run_path_" + str(len(self.paths) + 1) + "}"
            self.paths[alias] = target
        return alias

    def bind_inventory(self, resources):
        """Enrich model-facing metadata, without changing the persisted inventory digest."""
        wheels = []
        for row in resources.get("local_wheels", []):
            wheels.append({**row, "cache_execution_ref": self.bind_run_reference(row["cache_ref"]),
                "destination_execution_ref": self.bind_run_reference(row["suggested_destination"])})
        return {**resources, "local_wheels": wheels,
            "execution_guidance": "Use *_execution_ref handles in commands; cache_ref and "
                "suggested_destination are metadata relative to RUN ROOT, never checkout."}

    def manifests(self, manifests):
        result = {}
        for name, content in sorted(manifests.items()):
            relative = Path(name)
            base = (self.repo / relative).parent
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("manifest must be checkout-relative")
            def encode(match):
                target = (base / match.group()).resolve()
                if not target.is_relative_to(self.output):
                    # Outside resources need explicit bindings, never arbitrary host access.
                    return "[UNBOUND_PATH]"
                alias = next((key for key, value in self.paths.items() if value == target), None)
                if alias is None:
                    alias = "{run_path_" + str(len(self.paths) + 1) + "}"
                    self.paths[alias] = target
                return alias
            result[name] = PARENT.sub(encode, content)
        return result

    def catalog(self):
        return [{"alias": key, "run_relative_target": str(value.relative_to(self.output)),
                 "readiness": "path mapping only; existence and native consumption unverified"}
                for key, value in self.paths.items()]

    def encode_text(self, text):
        """Project known native paths to references before generic display sanitization."""
        for alias, target in sorted(self.paths.items(), key=lambda item: len(str(item[1])), reverse=True):
            text = text.replace(str(target), alias)
        return text

    def decode(self, text):
        if any(marker in text for marker in ("[OUTSIDE_PATH]", "[LOCAL_PATH]", "[UNBOUND_PATH]", "[REDACTED]")):
            raise ValueError("display-redacted or unbound path is not an executable reference")
        def restore(match):
            if match.group() not in self.paths:
                raise ValueError("unknown execution path alias")
            target = self.paths[match.group()]
            suffix = re.match(r"[/\w.@~+-]*", text[match.end():]).group()
            # Recheck existing symlinks before adopting model operations.
            if (not target.resolve().is_relative_to(self.output) or
                    not (target / suffix.lstrip('/')).resolve().is_relative_to(self.output)):
                raise ValueError("execution path alias escaped the run")
            return str(target)
        return TOKEN.sub(restore, text)

    def operations(self, value):
        for field in ("commands", "probes", "retire_operations"):
            if isinstance(value.get(field), list):
                decoded = []
                for index, item in enumerate(value[field]):
                    try:
                        decoded.append(self.decode(item) if isinstance(item, str) else item)
                    except ValueError as exc:
                        raise ValueError(f"{field}[{index}]: {exc}; execution rejected without "
                                         "rewriting the operation") from exc
                value[field] = decoded
        for request in value.get("resource_requests") or []:
            if isinstance(request, dict) and isinstance(request.get("command"), str):
                request["command"] = self.decode(request["command"])
        for asset in value.get("assets") or []:
            if isinstance(asset, dict) and isinstance(asset.get("where"), str):
                asset["where"] = self.decode(asset["where"])
        return value
