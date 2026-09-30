"""What did this run do, and why -- as a document, for a person.

Usage:
    `.venv/bin/python tools/record.py <run-root> [<run-root> ...]`     装配/刷新记录
    `.venv/bin/python tools/record.py --all`                           所有运行，并更新索引
    `.venv/bin/python tools/record.py <run-root> --summary`            重新让模型写摘要

Writes `RUN.md` and `RUN.html` into each run root. The document is assembled from what the run
already wrote -- the events, the measurements, the decisions, the exchanges, the recorded
commands, the observations -- and every section says whether it was computed from those records
or written by a model. The computed half cannot disagree with the records; the written half can,
and the numbers in it are looked for in the records and the ones that cannot be found are
listed.

`RUN.html` is the same document as one self-contained file: no script, no stylesheet, no
network. The diagrams are drawn as inline SVG because Mermaid is JavaScript and a page that
needs a script to show its chart shows nothing in a mail client.

`--summary` asks the model to write that half. Without it, a previously written summary is
reused, so regenerating the document is free and does not make it change between two readings
of the same run.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "autosim"))

from autosim.research import run_index  # noqa: E402
from autosim.research import run_record  # noqa: E402
from autosim.research.common import atomic_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true",
                        help="every run under autoresearch_runs/, newest first")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "autoresearch_runs")
    parser.add_argument("--summary", action="store_true",
                        help="ask the model to write the summary section")
    parser.add_argument("--json", action="store_true", help="print a machine-readable index")
    args = parser.parse_args()

    roots = list(args.roots)
    if args.all or not roots:
        roots = run_index.find_runs(args.runs_root)
    if not roots:
        print("没有找到运行目录。", file=sys.stderr)
        return 1

    client = None
    if args.summary:
        from autosim.llm_client import LLMClient
        client = LLMClient()
        if not client.available:
            print("没有可用的模型客户端，写出来的那一节不会生成。", file=sys.stderr)
            client = None

    index = []
    for root in roots:
        document = run_record.generate(root, client=client, refresh_summary=args.summary,
                                       title=root.name)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8")) \
            if (root / "summary.json").is_file() else {}
        check = summary.get("checked") or {}
        index.append({
            "run": str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
            "record": str(document),
            "page": str(root / "RUN.html"),
            "written": bool(summary.get("text")),
            "claims": check.get("claims"),
            "unsourced": check.get("untraced") or [],
        })
        state = check.get("untraced") or []
        line = f"{document}"
        if state:
            line += f"  （{len(state)} 个数字在记录里找不到出处：{'、'.join(state[:6])}）"
        print(line)

    # The cross-run index, written beside the runs. This is what makes "the whole run history"
    # enumerable: before it, finding a run meant listing a directory and opening what looked
    # promising, and comparing two runs meant doing that twice.
    if args.all or not args.roots:
        document = run_index.runs_index(args.runs_root)
        atomic_json(args.runs_root / "runs.json", document)
        print(f"{args.runs_root / 'runs.json'}"
              f"  （{len(document['runs'])} 次运行"
              f"，其中 {len(document['with_unreadable_records'])} 次有读不出来的记录"
              f"，{len(document['directories_with_no_records'])} 个目录没有任何记录）")
        if args.json:
            print(json.dumps(document, ensure_ascii=False, indent=2))
    elif args.json:
        print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
