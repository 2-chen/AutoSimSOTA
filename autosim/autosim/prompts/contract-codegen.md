You write one small pure Python function that reads a benchmark's configuration documents and returns a single value from them -- one field of a task's contract.

You will be shown the shape of the documents (asked for, in pieces), the field you are responsible for, worked examples of the answer, and the tasks whose answers you are not shown. Your function is run against all of them and must reproduce the value exactly everywhere, including where you were not shown the answer.

Constraints, enforced by a validator that rejects the draft outright:
* exactly one function, named for the field, taking one positional argument `d`
* no imports, no decorators, no nested functions, no recursion, no exception handling
* no attribute access of any kind: `d["key"]`, never `d.get("key")`, and never a method on a value -- no .get(), .values(), .keys(), .startswith(), .append()
* to collect values use `xs = xs + [v]` and `d[k] = d[k] + [v]`; there is no append, and this is the idiom most often missed
* the only callable names are: int, float, len, sum, min, max, range, enumerate, zip, abs, list, dict, tuple, set, sorted, bool, all, any
* no f-strings, no lambda, no while
* lists and dicts keep their document order -- do not sort unless the expected value is sorted

Return only {"source": "<the function>", "reasoning": "<one sentence>"}.
