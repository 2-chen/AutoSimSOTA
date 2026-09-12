"""Verify that RoboTwin and RoboSyn use compatible, isolated runtimes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from .common import atomic_json, digest, now


def inspect_runtime(python: Path, *, include_curobo: bool) -> dict[str, Any]:
    imports = "import torch, warp"
    values = {
        "python": "__import__('sys').executable",
        "torch_version": "torch.__version__",
        "torch_cuda_version": "torch.version.cuda",
        "torch_path": "torch.__file__",
        "warp_version": "warp.__version__",
        "warp_path": "warp.__file__",
    }
    if include_curobo:
        imports += ", curobo"
        values.update(curobo_version="curobo.__version__", curobo_path="curobo.__file__")
    expression = "{" + ",".join(f"{key!r}:{value}" for key, value in values.items()) + "}"
    code = f"{imports}; import json; print(json.dumps({expression}))"
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"runtime probe failed for {python}: {result.stderr.strip()}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def probe(project_root: Path) -> dict[str, Any]:
    project_root = project_root.absolute()
    main = inspect_runtime(project_root / ".venv/bin/python", include_curobo=False)
    robotwin = inspect_runtime(project_root / ".venv_robotwin/bin/python", include_curobo=True)
    compiled = sorted((project_root / "RoboTwin/envs/curobo").rglob("*.so"))
    checks = {
        "distinct_python_prefixes": main["python"] != robotwin["python"],
        "shared_pytorch_build": main["torch_version"] == robotwin["torch_version"] == "2.7.1+cu128",
        "main_warp_for_newton": main["warp_version"] == "1.13.0",
        "robotwin_warp_for_curobo": robotwin["warp_version"] == "1.12.0",
        "curobo_version": robotwin["curobo_version"] == "0.7.8",
        "compiled_curobo_extensions_present": bool(compiled),
    }
    lock_files = [project_root / "RoboTwin/autoresearch_env/critical-requirements.txt",
                  project_root / "RoboTwin/autoresearch_env/README.md"]
    return {
        "schema_version": 1,
        "kind": "robotwin_runtime_isolation_probe",
        "created_at": now(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "main_runtime": main,
        "robotwin_runtime": robotwin,
        "compiled_extensions": [{"path": str(path), "sha256": digest(path)} for path in compiled],
        "lock_files": {str(path): digest(path) for path in lock_files},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.project_root)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
