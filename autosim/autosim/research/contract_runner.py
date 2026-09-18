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
    functions = request["functions"]
    results = []
    for case in request["cases"]:
        produced: dict = {}
        errors: dict = {}
        # Every function is called and every failure is attributed to the function that
        # raised. Stopping at the first failure and reporting one error would leave the
        # caller unable to tell *which* field was wrong -- and a caller that guesses blames
        # whichever function it happens to look at first, accepting a broken one.
        for name in functions:
            try:
                produced[name] = json.loads(json.dumps(namespace[name](case)))
            except BaseException as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        results.append({"returned": produced, "errors": errors})
    sys.stdout.write(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
