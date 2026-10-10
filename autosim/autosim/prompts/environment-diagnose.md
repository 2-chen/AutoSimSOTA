An environment build has stopped. Say what is in the way, in terms someone can act on, and do not overstate what is known.

A dependency that is missing is not the same finding as a requirement that cannot be satisfied on this machine, and neither is the same as the machine itself being unable to host the workload -- a driver, a rendering backend, a compute capability the installed builds do not target. They have different remedies and only the first is a matter of installing something.

The record is what survived; the transcript is everything tried. A command in the record has run successfully and is not evidence about the blocker unless it is the one that failed.

Return one JSON object: {"blocker": "<dependency | requirement-not-satisfiable | platform | unresolved>", "what": "<the specific thing in the way>", "evidence": "<the commands and output that show it>", "tried": ["<approaches attempted>"], "would_unblock": "<what a person would do, and what it would cost>", "confidence": "<high | medium | low>"}
