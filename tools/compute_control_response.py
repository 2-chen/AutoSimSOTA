#!/usr/bin/env python3
"""Use the connected host's API to answer one frozen SCO allocation request."""
import argparse
from pathlib import Path
from autosim.research.common import read_json, atomic_json
from autosim.research.compute_planner import ComputePlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--env-file",type=Path,required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    root = args.request.parent
    planner = ComputePlanner(root / "connected_control",controller="api",env_file=args.env_file)
    decision = planner.propose(request["plan"],telemetry={"stage":"allocated_component_validation"})
    atomic_json(root / "compute_control_response.json", {"request_digest":request["request_digest"],
                "decision":decision,"api_on_control_host":True})
    print({"mode":decision["mode"],"request_digest":request["request_digest"],"parameters":decision["parameters"]})
    return 0 if decision["mode"] == "api" else 2


if __name__ == "__main__":
    raise SystemExit(main())
