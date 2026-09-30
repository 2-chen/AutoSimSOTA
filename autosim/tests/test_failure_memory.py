"""What went wrong before, and what fixed it -- the property being that it transfers.

The whole value of this module is one thing: a failure met at one repository retrieves what
fixed it at another. If signatures were per-repository, the memory would be a log; if they
were coarser than the fault, it would hand on the wrong fix. The tests below are about that
boundary.
"""

import json

from autosim.research import failure_memory as fm


def test_two_repositories_with_the_same_fault_have_one_signature():
    """The step AutoSOTA names: raw tracebacks normalised into reusable signatures.

    `ModuleNotFoundError: No module named 'einops'` and the same error for `XPolicyLab` are
    one failure. The fix for the first -- put the package's root on the search path -- is what
    the second needed, and a memory that kept them apart would learn it twice."""
    here = ("Traceback (most recent call last):\n"
            "  File \"/home/a/repo/train.py\", line 42, in <module>\n"
            "    import einops\n"
            "ModuleNotFoundError: No module named 'einops'")
    there = ("Traceback (most recent call last):\n"
             "  File \"/home/b/other/main.py\", line 7\n"
             "ModuleNotFoundError: No module named 'XPolicyLab'")
    assert fm.signature(here) == fm.signature(there)

    # And two faults that need different fixes do not collide.
    assert fm.signature(here) != fm.signature("FileNotFoundError: [Errno 2] No such file or "
                                              "directory: '/x/y.ckpt'")
    assert fm.signature(here) != fm.signature("bash: line 14: 10: unbound variable")


def test_the_last_line_of_a_traceback_is_the_failure_and_the_frames_are_not():
    """A traceback's opening frames are library internals; its final line is what went wrong.
    Reading the first line instead would signature every failure in a repository that imports
    torch as `File "/.../torch/__init__.py"` -- every one of them, as the same failure."""
    said = ("Traceback (most recent call last):\n"
            "  File \"/opt/torch/__init__.py\", line 1\n"
            "  File \"/home/a/x.py\", line 3\n"
            "AttributeError: 'Robot' object has no attribute 'left_planner'")
    assert fm.signature(said) == "attributeerror: '…' object has no attribute '…'"
    # A failure that is one line of prose is normalised whole -- the honest reading of a
    # failure with no traceback.
    assert fm.signature("the dataset is not on this machine") == "the dataset is not on this machine"
    assert fm.signature("") == ""


def test_names_inside_a_message_are_not_part_of_the_failure():
    """The names are the instance, not the fault, and a signature that kept them would have to
    learn the same lesson once per name.

    `KeyError: 'beat_block_hammer'` and `KeyError: 'autosim_705e29a58f-...'` are one failure:
    the program indexed a table by a value that is not a key in it. `AttributeError` for a
    missing `Robot.left_planner` and for a missing `Scene.create_entity` are one failure too --
    the code called something the installed version does not have, which is the fault that
    cost nineteen reinstalls of a package that was never the problem. What the names are
    belongs in the example, where a reader finds it."""
    one = fm.signature("KeyError: 'beat_block_hammer'")
    two = fm.signature("KeyError: 'autosim_705e29a58f-beat_block_hammer-aloha_agilex-joint'")
    assert one == two == "keyerror: '…'"
    assert fm.signature("AttributeError: 'Robot' object has no attribute 'left_planner'") == \
        fm.signature("AttributeError: 'Scene' object has no attribute 'create_entity'")


def test_a_remedy_that_worked_is_retrieved_for_the_next_repository(tmp_path):
    """What the memory is for. The signature transfers; the change is stored whole, because a
    remedy that says "fix the PATH" helps a reader and not a loop."""
    memory = fm.FailureMemory(tmp_path / "failures.json")
    said = "ModuleNotFoundError: No module named 'einops'"
    memory.record(said, {"environment": {"PYTHONPATH": "{repo}"}}, outcome="accepted",
                  repo="RoboTwin")
    memory.record(said, {"environment": {"PYTHONPATH": "{repo}"}}, outcome="accepted",
                  repo="LIBERO")

    found = fm.FailureMemory(tmp_path / "failures.json").recall(
        "ModuleNotFoundError: No module named 'XPolicyLab'")
    assert found["remedies_that_worked"][0]["change"] == {"environment": {"PYTHONPATH": "{repo}"}}
    assert found["remedies_that_worked"][0]["times"] == 2
    assert "RoboTwin" in found["remedies_that_worked"][0]["where"]
    # And a failure with no history says so rather than returning an empty remedy.
    assert "remedies_that_worked" not in failure_memory_empty(tmp_path)


def failure_memory_empty(tmp_path):
    return fm.FailureMemory(tmp_path / "other.json").recall("nothing like this before")


def test_what_did_not_work_is_remembered_too(tmp_path):
    """AutoSOTA's term is ruling out already-exhausted branches. The failure it prevents is the
    cyclic one: the same plausible fix proposed, tried and refuted once per round until the
    budget ends -- which is what the RoboTwin derivation did with the checkpoint directory."""
    memory = fm.FailureMemory(tmp_path / "failures.json")
    said = "FileNotFoundError: [Errno 2] No such file or directory: 'models/act'"
    memory.record(said, {"working_directory": "{repo}/policy"}, outcome="refuted", repo="r")
    memory.record(said, {"parameters": {"ckpt": "beat_block_hammer"}}, outcome="accepted", repo="r")

    found = memory.recall(said)
    assert found["already_tried_and_did_not_work"][0]["change"] == {
        "working_directory": "{repo}/policy"}
    assert found["remedies_that_worked"][0]["change"] == {
        "parameters": {"ckpt": "beat_block_hammer"}}


def test_the_same_remedy_is_counted_and_not_stored_twice(tmp_path):
    memory = fm.FailureMemory(tmp_path / "failures.json")
    for _ in range(3):
        memory.record("boom: it broke", {"environment": {"A": "1"}}, outcome="accepted")
    memory.record("boom: it broke", {"environment": {"B": "2"}}, outcome="accepted")
    assert memory.summary() == {"signatures": 1, "remedies": 2, "that_worked": 2}
    assert memory.recall("boom: it broke")["remedies_that_worked"][0]["change"] == {
        "environment": {"B": "2"}, }


def test_a_remedy_is_the_change_and_not_the_reasoning_around_it(tmp_path):
    """`why` is prose about one repository's problem, and storing it is what makes a memory
    look specific when it is general."""
    memory = fm.FailureMemory(tmp_path / "failures.json")
    memory.record("boom", {"working_directory": "{repo}", "why": "the script needs its own "
                      "directory on the path, and I read train.sh to find that out",
                      "obstacle": "", "reasoning": "long"},
                  outcome="accepted")
    assert memory.recall("boom")["remedies_that_worked"][0]["change"] == {
        "working_directory": "{repo}"}
    # A revision with nothing left after that has no remedy in it, and storing an empty one
    # would make the next reader think something had been tried.
    memory.record("boom", {"why": "I could not think of anything"}, outcome="accepted")
    assert len(memory.rows["boom"]["remedies"]) == 1


def test_the_memory_survives_being_reopened_and_does_not_die_on_a_bad_file(tmp_path):
    path = tmp_path / "failures.json"
    fm.FailureMemory(path).record("boom", {"environment": {"A": "1"}}, outcome="accepted")
    assert fm.FailureMemory(path).known("boom")
    # A memory that cannot be read is a memory that will be relearned; it is not a reason to
    # end the run that was trying to read it.
    path.write_text("{not json", encoding="utf-8")
    assert fm.FailureMemory(path).rows == {}
    assert fm.FailureMemory(path).recall("boom") == {"signature": "boom"}


def test_a_memory_that_cannot_be_written_does_not_raise(tmp_path):
    """The write happens inside the derive loop, where an exception discards a round."""
    memory = fm.FailureMemory(tmp_path / "nowhere" / "deep" / "failures.json")
    memory.path.parent.mkdir(parents=True)
    memory.path.write_text("", encoding="utf-8")
    memory.path.chmod(0o400)
    try:
        memory.record("boom", {"environment": {"A": "1"}}, outcome="accepted")
    finally:
        memory.path.chmod(0o600)


def test_the_stored_file_is_one_document_a_person_can_read(tmp_path):
    """Not an append-only log: answering "what fixes this" should not need a replay."""
    path = tmp_path / "failures.json"
    memory = fm.FailureMemory(path)
    memory.record("ModuleNotFoundError: No module named 'einops'",
                  {"environment": {"PYTHONPATH": "{repo}"}}, outcome="accepted", repo="RoboTwin")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    row = next(iter(document["signatures"].values()))
    assert row["signature"] == "modulenotfounderror: no module named '…'"
    assert row["examples"] and "RoboTwin" in row["examples"][0]


def test_colour_codes_are_not_part_of_the_failure():
    """A program that colours its errors emits `\\x1b[1;35mModuleNotFoundError\\x1b[0m`. The
    escape was left in, so the same failure with and without colour were two signatures and
    the memory learned each separately -- which it did: entries appeared whose keys contained
    `[<n>m` where a colour code had been."""
    plain = "ModuleNotFoundError: No module named 'einops'"
    coloured = "\x1b[1;35mModuleNotFoundError\x1b[0m: No module named 'einops'"
    assert fm.signature(coloured) == fm.signature(plain) == "modulenotfounderror: no module named '…'"
    assert fm.signature("\x1b[31m[ERROR]\x1b[0m FileNotFoundError: [Errno 2] No such file") == \
        fm.signature("FileNotFoundError: [Errno 2] No such file")


def test_a_line_printed_after_the_traceback_is_not_the_failure():
    """Programs print progress and cleanup as they exit, after the traceback. Taking the last
    line put `[cleanup] killing server pid=<n>` and `[info] action dim: <n>` into the memory
    as signatures -- entries that are not failures and can never retrieve a remedy.

    When there is no exception at all, the last line is still the honest reading: a failure
    that is one line of prose, or a shell that said `python: can't open file ...`, has
    nothing else to offer."""
    said = ("Traceback (most recent call last):\n"
            "  File \"eval.sh\", line 12\n"
            "FileNotFoundError: [Errno 2] No such file or directory: 'ckpt/policy_last.ckpt'\n"
            "\x1b[31m[CLEANUP] Killing server PID=1458371\x1b[0m\n"
            "[MAIN] done")
    # The quoted path becomes `'…'` -- the quote rule, which is the one that gives
    # `KeyError: '…'` its generality, and applying it here keeps one convention.
    assert fm.signature(said) == \
        "filenotfounderror: [errno <n>] no such file or directory: '…'"
    # A chained traceback ends with the failure that stopped the program.
    chained = ("Traceback (most recent call last):\n"
               "ValueError: bad value\n"
               "\nDuring handling of the above exception, another exception occurred:\n"
               "RuntimeError: the worker died")
    assert fm.signature(chained) == "runtimeerror: the worker died"
    # And a failure with no exception line at all.
    assert fm.signature("python: can't open file 'train.py': [Errno 2] No such file") \
        == "python: can't open file '…': [errno <n>] no such file"


def test_a_remedy_that_held_in_a_second_repository_becomes_a_method(tmp_path, monkeypatch):
    """AutoSOTA's third level of memory, and the only one that changes what the system can do
    at a repository it has never seen. One repository is an anecdote; the claim being made is
    that it transfers, so the entry is written when a second one confirms it."""
    from autosim.research import skills

    path = tmp_path / "failures.json"
    memory = fm.FailureMemory(path)
    said = "ModuleNotFoundError: No module named 'einops'"
    memory.record(said, {"environment": {"PYTHONPATH": "{repo}"}}, outcome="accepted", repo="A")
    assert fm.distil(path, into=tmp_path / "d") == [], "one repository is not a method"

    memory.record(said, {"environment": {"PYTHONPATH": "{repo}"}}, outcome="accepted", repo="B")
    written = fm.distil(path, into=tmp_path / "d")
    assert len(written) == 1 and written[0].suffix == ".md"

    # It parses as a skill, and it reaches the controller.
    parsed = skills._parse(written[0])
    assert parsed["name"] == "modulenotfounderror-no-module-named"
    assert parsed["scope"] == "general"
    assert parsed["version"] == "0.1.0"
    assert parsed["id"].startswith("autosimsota.distilled.")
    assert "A" in parsed["evidence"] and "B" in parsed["evidence"]
    assert "PYTHONPATH" in parsed["method"]

    # And it says what it is. An entry that looked like a person's would be believed at the
    # weight of the whole curated library.
    assert "no person wrote this entry" in parsed["confidence"]
    assert "2 repositories" in parsed["confidence"]
    assert "not as a rule" in parsed["confidence"]
    # Including what it cannot support: it records the last revision before a command ran, and
    # that is not a demonstration that the revision is what fixed it.
    assert "does not say the change caused the fix" in parsed["method"]

    # Distilled skills must satisfy the same retrieval contract as curated and external ones,
    # or the learning loop would write knowledge that every later run silently ignores.
    monkeypatch.setenv("AUTOSIM_SKILLS_DIR", str(written[0].parent))
    reference = skills.skills_reference(
        query="missing package dependency import runtime command repair", max_selected=12)
    surfaced = {one["id"]: one for one in reference["skills"]}
    assert parsed["id"] in surfaced
    assert surfaced[parsed["id"]]["version"] == "0.1.0"
    assert any(one["id"] == parsed["id"] for one in reference["selection"])


def test_a_remedy_that_was_also_refuted_somewhere_is_not_distilled(tmp_path):
    """A change that fixed one repository and was refuted at another is not a method, and the
    record knows both -- the outcome is per remedy, updated in place."""
    path = tmp_path / "failures.json"
    memory = fm.FailureMemory(path)
    said = "boom: it broke"
    memory.record(said, {"environment": {"A": "1"}}, outcome="accepted", repo="A")
    memory.record(said, {"environment": {"A": "1"}}, outcome="refuted", repo="B")
    assert fm.distil(path, into=tmp_path / "d") == []


def test_a_traceback_heading_is_not_a_failure():
    """The line a traceback opens with is a heading, and keying on it collapsed eight
    unrelated failures into one entry called `exception: traceback (most recent call last)`
    -- an entry whose remedies were retrieved for every one of them and were right for none.

    It arrives two ways: as its own line, and as the *message* of an exception, which is what
    a program does when it renders a traceback into a string -- and what this system's own
    error wrapping does. And a bare type name with no message is a real exception line that
    says nothing: every argless `raise` in every program shares it, so a failure the record
    cannot name is one it should not pretend to remember.
    """
    assert fm.signature("Exception: Traceback (most recent call last)") == ""
    assert fm.signature("Traceback (most recent call last):\nException") == ""
    assert fm.signature("Traceback (most recent call last):\n  File \"/a/b.py\", line 3\n"
                        "ValueError: substring not found") == "valueerror: substring not found"
    # And a real one still works.
    assert fm.signature("KeyError: 'x'") == "keyerror: '…'"
