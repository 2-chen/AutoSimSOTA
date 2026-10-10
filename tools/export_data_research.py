"""Export verified development evidence; does not run experiments or promote scores."""
import argparse
from pathlib import Path

from autosim.research.common import atomic_text, immutable_json
from autosim.research.paper_evidence import build, markdown


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('repository',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args(argv)
    report=build(args.run,args.repository)
    root=args.destination/report['snapshot_id']
    immutable_json(root/'analysis.json',report)
    atomic_text(root/'RESULTS.md',markdown(report))
    print(root/'RESULTS.md')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
