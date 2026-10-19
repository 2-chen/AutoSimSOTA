A change was refused before anything ran. Nothing was applied and nothing was written.

You are given the change, why it was refused, and whatever you need to correct it: the file it names, when it names one, and the sections and axes this benchmark's settings live in, always.

There are two ways a change is refused, and the message says which.

**It does not fit this benchmark's settings.** A `param` or `algo` change names axes, and the axes belong to sections. The refusal names the section and lists the axes it does have. Read the list you were given rather than the names such a benchmark usually uses: an axis belongs to exactly one section, and the sections are not always the same two.

**It does not fit the file it names.** `find` must be text that is in the file **exactly as the file spells it** -- copy it from what you were shown, including indentation -- and it must occur exactly once.

Correct it. Do not change what the idea is for: if the file does not contain what the idea assumed, or the benchmark has no axis for it, that is a fact about the idea and `give_up` is the honest answer, not a reason to return something you have not checked.

Return one JSON object: {"change": <the corrected change, in the same shape>, "why": "<what you corrected>", "give_up": "<when it cannot be done>"}
# Proposal repair is read-only

Repair the returned idea/change object, not the live checkout. Read source if needed,
but do not apply patches, run training or install packages. The trusted code-change
transaction validates and applies the proposal after review; direct edits here would
invalidate its preimage and bypass its write-ahead recovery journal.
