"""Trusted validation of pure numeric audit kernels in a bounded Python subset.

Only JSON inputs, arithmetic, local containers and explicit numeric builtins are available.
This is intentionally narrower than arbitrary Python patches. OS/library/loader patches
require a separately provisioned OS sandbox and are not executed by this validator.
"""
from __future__ import annotations
import ast
import builtins
import json
import sys
import time

BUILTINS = {name: getattr(builtins, name) for name in
            ("int", "float", "len", "sum", "min", "max", "range", "enumerate", "zip", "abs",
             "list", "dict", "tuple", "set", "sorted", "bool", "all", "any")}
NODES = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
         ast.AugAssign, ast.AnnAssign, ast.For, ast.If, ast.Expr, ast.Pass, ast.Break,
         ast.Continue, ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
         ast.Subscript, ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
         ast.IfExp, ast.Call, ast.keyword, ast.ListComp, ast.DictComp, ast.SetComp,
         ast.GeneratorExp, ast.comprehension, ast.Load, ast.Store, ast.operator,
         ast.unaryop, ast.boolop, ast.cmpop)


def checked_function(source: str, name="_padding_audit"):
    if len(source.encode()) > 24000:
        raise ValueError("kernel too large")
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != name:
        raise ValueError("exactly the registered function must be supplied")
    function = tree.body[0]
    if function.decorator_list or function.args.vararg or function.args.kwarg:
        raise ValueError("decorators and variadic arguments are not supported")
    if any(not isinstance(n, ast.Constant) or type(n.value) is not int or not 1 <= n.value <= 10000
           for n in [*function.args.defaults, *(n for n in function.args.kw_defaults if n is not None)]):
        raise ValueError("kernel defaults must be small positive integer literals")
    function.returns = None
    for arg in (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs):
        arg.annotation = None
    for node in ast.walk(tree):
        if not isinstance(node, NODES):
            raise ValueError(f"unsupported kernel syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ValueError("nested functions are not allowed")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id == name):
            raise ValueError("private identifiers and recursion are not allowed")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in BUILTINS):
            raise ValueError("only registered numeric builtins may be called")
    namespace = {"__builtins__": BUILTINS}
    exec(compile(ast.fix_missing_locations(tree), "<validated-audit-kernel>", "exec"), namespace)
    return namespace[name]


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    payload = json.load(sys.stdin)
    function = checked_function(payload["source"])
    outputs = []
    started = time.perf_counter()
    for item in payload["cases"]:
        lengths, chunk = item["lengths"], item["chunk_size"]
        if (not isinstance(lengths, list) or len(lengths) > 100000 or
                any(type(n) is not int or n < 0 or n > 10000000 for n in lengths) or
                type(chunk) is not int or not 1 <= chunk <= 10000):
            raise ValueError("invalid audit inputs")
        outputs.append(function(lengths, chunk))
    print(json.dumps({"outputs": outputs, "elapsed_seconds": time.perf_counter() - started}, allow_nan=False))


if __name__ == "__main__":
    main()
