"""No module may define the same top-level name twice.

Python binds the last one, so the earlier definition is dead: every edit to it reads as a
fix and changes nothing. This is not hypothetical. `execution_derive` carried two copies of
`error_excerpt`, and a rewrite of the first one -- the fix for a nested traceback's cause
being cut off before the model could read it -- was silently discarded by the second, which
then sent the same truncated excerpt and got the same failing draft back.
"""

import ast
from collections import Counter
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "autosim"


def modules() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_package_is_where_the_test_thinks_it_is():
    assert modules(), f"no modules found under {PACKAGE}"


@pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
def test_no_module_defines_a_top_level_name_twice(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    repeated = {name: count for name, count in Counter(names).items() if count > 1}
    assert not repeated, (
        f"{path.relative_to(PACKAGE.parent)} defines {repeated} more than once; the later "
        f"definition wins and the earlier one is dead code")
