"""Separate explicit formal optimizer work from bounded command verification.

No repository defaults, filename heuristics, or benchmark-specific learning schedules
are invented here. The Agent supplies formal work; the executor validates provenance.
"""
import ast


def consumes_steps(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError):
        return False  # Source compilation has its own mandatory validation.
    return any(isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
               and node.slice.value == 'steps' for node in ast.walk(tree)) or any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'get' and node.args and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 'steps' for node in ast.walk(tree))


def missing_formal_work(source: str, inputs: dict) -> str:
    if not consumes_steps(source):
        return ''  # The native producer may use its own epoch/config budget.
    value = inputs.get('steps')
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return ('formal training requires an explicit positive optimizer-update budget: '
                'choose settings.steps from source, measured cost and remaining total budget. '
                'For a new baseline use run_the_loop(training_settings={steps: ...}); '
                'for a candidate propose a declared training-axis idea or submit a native job '
                'with explicit settings. Verification work is never a formal default; '
                'native epochs are not interchangeable with optimizer updates.')
    return ''
