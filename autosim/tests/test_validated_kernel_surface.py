"""What a validated kernel may and may not do, and why the line is where it is.

The allow-list exists to make I/O and imports impossible. It does not exist to make ordinary
Python impossible, and the difference matters: `for k in d.keys()` is how a person iterates a
dict, and rejecting it cost a whole attempt each time a generated function reached for it --
three times in one run, every time reported as a complaint about syntax that was correct.
"""

import pytest

from autosim.research.patch_validation import checked_function


def function(body: str, name: str = "stage_argv_train") -> str:
    return f"def {name}(i):\n" + "".join(f"    {line}\n" for line in body.strip().splitlines())


def test_a_dict_may_be_iterated_the_ordinary_way():
    """The idiom that was being rejected, and the one the prompt now teaches it instead."""
    read = checked_function(function("""
argv = [i["python"], "-m", "pkg.train", "policy=act"]
for key in sorted(i["settings"].keys()):
    argv = argv + [key + "=" + str(i["settings"][key])]
return argv
"""), "stage_argv_train")
    assert read({"python": "py", "settings": {"b": 2, "a": 1}}) == \
        ["py", "-m", "pkg.train", "policy=act", "a=1", "b=2"]


def test_a_sparse_settings_map_is_not_indexed_by_a_fixed_name():
    """The failure the prompt exists to prevent: a fixed key raises on every call that does
    not happen to vary that setting, which is most of them."""
    read = checked_function(function("""
argv = [i["python"], "-m", "pkg.train", "train.n_epochs=10"]
for key in sorted(i["settings"]):
    argv = argv + [key + "=" + str(i["settings"][key])]
return argv
"""), "stage_argv_train")
    assert read({"python": "py", "settings": {}})[-1] == "train.n_epochs=10"
    assert read({"python": "py", "settings": {"train.n_epochs": 1}})[-1] == "train.n_epochs=1"


@pytest.mark.parametrize("body,why", [
    # Attribute access is the route out, so the check is on the attribute -- and `__`-prefixed
    # names are what every traversal out of a live object goes through.
    ("return [i['x'].__class__.__name__]", "class traversal"),
    ("return [str(i['x'].__class__.__mro__)]", "mro traversal"),
    ("return [i['x'].__globals__['__builtins__']]", "globals traversal"),
    ("return [getattr(i, '__class__')]", "getattr is not a registered name"),
    # Pure methods only. `format` looks pure and is not: the attribute it reads is named
    # inside a string, where no syntax check can see it.
    ("return ['{0.__class__}'.format(i['x'])]", "format-string traversal"),
    ("return [i['x'].read()]", "file read"),
    ("return [i['x'].system('id')]", "shell"),
    ("return [open('/etc/passwd').read()]", "open is not a registered name"),
])
def test_the_routes_out_of_the_sandbox_are_still_closed(body, why):
    with pytest.raises(ValueError):
        checked_function(function(body), "stage_argv_train")


def test_an_import_is_refused_by_the_syntax_check_itself():
    with pytest.raises(ValueError):
        checked_function(function("import os\nreturn [os.getcwd()]"), "stage_argv_train")
