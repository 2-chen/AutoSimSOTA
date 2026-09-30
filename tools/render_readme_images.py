"""Render authored SVG assets into compatible README PNGs using CPU-only Rsvg.

Run with the system Python providing PyGObject and Rsvg 2.0. No run data or
benchmark screenshots are read, and no model API or GPU is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "assets",
    )
    args = parser.parse_args()
    try:
        import gi
        gi.require_version("Rsvg", "2.0")
        from gi.repository import Rsvg
    except (ImportError, ValueError) as exc:
        parser.error(f"system Python requires PyGObject and Rsvg 2.0: {exc}")
    for stem in ("hero", "architecture", "run-preview"):
        source = args.assets / f"{stem}.svg"
        target = args.assets / f"{stem}.png"
        if not source.is_file():
            parser.error(f"missing authored SVG: {source}")
        Rsvg.Handle.new_from_file(str(source)).get_pixbuf().savev(
            str(target), "png", [], [],
        )
        print(f"Rendered {target.name}")


if __name__ == "__main__":
    main()
