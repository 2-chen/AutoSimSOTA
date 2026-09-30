"""The half of the document a model wrote, and the check that keeps it honest.

This is the part of the record that can be wrong. Everything else in `run_record` is derived
from the run's own files and cannot disagree with them without the files being wrong; the
summary is prose written by a model about those files, and prose does not have that property.

So it is kept apart, in its own module, for the reason the whole layer exists: a reader who
wants to decide whether to trust the summary should be able to read everything that stands
behind it without reading anything else. That is what is here -- the prompt, the call, and the
check.

**What the check establishes is traceability, not truth.** Every number in the summary is
looked for in the computed record and the ones that cannot be found are listed. A number that
is found is a number that appears somewhere in the record; it is not a number that was used
correctly, and a wrong conclusion assembled out of real figures passes this check in silence.
The document says both of those things where the check is reported, because a check whose
limits are not stated is read as stronger than it is.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

#: Text that carries digits but is not a claim about a measurement. Stripped before the check,
#: so that a citation or an identifier is not reported as an unsourced number. The third
#: pattern requires a letter as well as a digit, and that requirement is load-bearing: written
#: without it, it matches every bare number in the passage and the check comes back with
#: nothing to say about a summary that invented all of them.
_STRIP = (
    re.compile(r"`[^`]*`"),                       # inline code and paths
    re.compile(r"\b[\w./-]*[/\\][\w./-]*"),        # anything that looks like a path
    # `:` is in the class because a device is written `cuda:0`, and without it the leading word
    # is stripped, the digit is not, and the check reports the zero of `cuda:0` as a claim.
    re.compile(r"\b(?=[\w:_-]*\d)(?=[\w:_-]*[A-Za-z_])[\w:_-]+\b"),   # cuda:0, LIBERO_10, v2
)
_NUMBER = re.compile(r"(-?\d+(?:\.\d+)?)(\s*%)?")


def _claims(text: str) -> list[tuple[str, float, float]]:
    """The numbers a written passage asserts, as (as-written, value, precision).

    The precision is the smallest difference the passage can express in the value's own units,
    which is what decides whether a record value could have been rounded into it. For a
    percentage that is a hundredth of the literal's own precision: `116.7%` is written to one
    decimal place and asserts a value to three, and treating it as one is how `1.2` comes to
    look like the source of `1.167`.
    """
    scrubbed = text
    for pattern in _STRIP:
        scrubbed = pattern.sub(" ", scrubbed)
    out: list[tuple[str, float, float]] = []
    for match in _NUMBER.finditer(scrubbed):
        literal = match.group(1)
        percent = bool(match.group(2))
        value = float(literal) / (100.0 if percent else 1.0)
        decimals = len(literal.split(".")[1]) if "." in literal else 0
        step = 10.0 ** (-decimals) / (100.0 if percent else 1.0)
        out.append((literal + ("%" if percent else ""), value, step))
    return out


def _every_number(value: Any, out: list[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, str):
        # Numbers inside text are part of the record too. The free-memory figure lives in
        # `device_why` as prose, and the losses a trainer printed live in `said`; a summary
        # quoting either is quoting the record. Walking only the JSON structure would report
        # every one of those as a number with no source, which is the check crying wolf.
        for match in re.finditer(r"-?\d+(?:\.\d+)?", value):
            try:
                out.append(float(match.group()))
            except ValueError:
                continue
    elif isinstance(value, dict):
        for one in value.values():
            _every_number(one, out)
    elif isinstance(value, (list, tuple)):
        for one in value:
            _every_number(one, out)


def known_numbers(source: dict[str, Any], *, and_what_was_shown: str = "") -> list[float]:
    """Every number the record contains, for asking whether a written one is in it.

    `and_what_was_shown` is the material the summary was written from -- the rendered
    sections, as the model saw them. It is not optional in spirit: the prompt tells the model
    that every number it writes will be looked for in the record, and a pool assembled from a
    *different* list of keys than the one the material was rendered from makes that promise
    false.

    It was false in both directions. On runs of one shape the pool was **empty** -- fifty-six
    of eighty-three run roots on this machine -- so every number the summary quoted from what
    it had been shown was reported as invented. On runs of another shape the pool held the
    records but not the rendered tables, so `成功率 0.652` and `308.3 秒`, both copied from
    the document's own tables, came back untraced. A check that condemns the honest summary
    is worse than no check: it teaches a reader to ignore this section.
    """
    out: list[float] = []
    for key in ("events", "decisions", "observations", "report", "recipe", "plan",
                "derived_stages", "environment", "rounds", "processes", "scout", "state",
                "selection"):
        _every_number(source.get(key), out)
    for measured in source.get("measurements") or []:
        _every_number(measured["record"], out)
    # And everything the model was actually shown, which is what it is allowed to quote.
    for literal in _NUMBER.findall(and_what_was_shown or ""):
        try:
            out.append(float(literal[0]))
        except (TypeError, ValueError):
            continue
    return out


def check_claims(text: str, pool: Iterable[float]) -> dict[str, Any]:
    """Which of a written passage's numbers can be found in the computed record.

    A number written to two decimals is allowed to match a value that rounds to it, because
    that is what writing it to two decimals means. What this establishes is *traceability*:
    the number is somewhere in the record. It does not establish that the number is used
    correctly, and the wording of the section says so -- a wrong claim built from real numbers
    passes this check, and that gap is worth naming rather than papering over.
    """
    candidates = [value for value in pool if isinstance(value, (int, float))]
    traced: list[str] = []
    untraced: list[str] = []
    for literal, value, step in _claims(text):
        found = any(abs(candidate - value) <= max(1e-9, abs(value) * 1e-9)
                    for candidate in candidates)
        if not found and step:
            # A rounded figure is sourced if some value in the record rounds to it at the
            # precision it was written to. This is what lets a summary say 0.65 about 0.6523
            # without being accused of inventing it.
            found = any(abs(candidate - value) < step / 2 + 1e-12 for candidate in candidates)
        (traced if found else untraced).append(literal)
    return {"claims": len(traced) + len(untraced), "traced": len(traced),
            "untraced": untraced, "traced_literals": traced}


SUMMARY_SYSTEM = """You are writing the opening section of a record of a research run, for a \
person who will read it later and who has the rest of the document in front of them.

You are given the computed sections of that document: the timeline, the numbers, the \
decisions, the artifacts and the unresolved items. Write the summary they do not have.

What makes this useful rather than decorative:

- Say what the run did and what it found out. If the numbers do not support a conclusion, say \
that rather than drawing one.
- Say what went wrong or remains unsettled, if anything did. A run that produced no result is \
worth summarising honestly; it is not worth summarising as progress.
- Where you state a number, use a number that appears in the material you were given. You may \
round it. **Every number you write will be looked for in the record and the ones that cannot \
be found will be listed as such underneath your section** -- so a number you invented is not a \
risk to you but a worthless sentence to the reader.
- Attribute a claim to where it comes from. If something is your reading of the evidence \
rather than the record's own statement, write it as your reading.

Do not pad. Four to ten sentences, in Chinese, plain prose. No headings, no bullet lists, no \
preamble about what you are about to say."""


def write_summary(material: str, client: Any, *, max_tokens: int = 1200) -> str:
    """Ask a model for the prose section, given the computed sections as text.

    The material arrives as a string rather than as the run's records, and that is what keeps
    this module free of any dependency on how the document is assembled. What it needs to know
    is what the model will be shown and what it answered; where those sections came from is
    `run_record`'s business, and a reviewer reading this file to judge the check should not
    have to follow that.
    """
    if client is None:
        raise ValueError("no model client, so the written section cannot be produced")
    system = SUMMARY_SYSTEM
    user = ("以下是这次运行记录中**算出来的**部分。请据此写摘要。\n\n"
            f"{material[:24000]}")
    content = client.chat(system, user, max_tokens=max_tokens, timeout=240)
    return str(content).strip()
