"""Run a generated task reader over a list of inputs, in a process of its own.

Separate from `contract_codegen` and free of imports, because it is executed as a script:
the code it runs has passed a syntax allow-list (`patch_validation.checked_function`), which
already guarantees it cannot import, open a file or reach the network. What the allow-list
cannot bound is time and memory, and a process of its own is what makes those boundable.

Reads a JSON request on stdin, writes a JSON list of results on stdout. A candidate that
raises is reported as having raised rather than crashing the runner -- a function that fails
on one task is evidence about the function, and the caller needs the other nine results.
"""

import json
import sys

FUNCTION = "task_contract"


def main() -> int:
    request = json.loads(sys.stdin.read())
    namespace: dict = {}
    exec(compile(request["source"], "<task-contract>", "exec"), namespace)
    function = namespace[FUNCTION]
    results = []
    for case in request["cases"]:
        try:
            produced = function(case)
            # Round-trip through JSON so the comparison is between documents, not between
            # Python objects that happen to serialize the same way.
            results.append({"returned": json.loads(json.dumps(produced))})
        except BaseException as exc:
            results.append({"raised": f"{type(exc).__name__}: {exc}"})
    sys.stdout.write(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
