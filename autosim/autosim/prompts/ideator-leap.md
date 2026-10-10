A research run has made the same KIND of change several times and the stage still does not run. Another change of that kind is the round that was already tried.

You are given the run's situation, the ideas it has already tried and what happened, and its library. **Produce one idea of a kind it has not been making** -- synthetically, from what the material shows, and not by selecting from the library.

Say why this kind of move is the one the evidence asks for, and what it would change.

`crosses` is the id of the red line this change would cross, or `"none"`. The lines are in `what_may_not_change`, and the audit refuses an idea on this answer -- so answer it against what the change does, not against what it is for.

`touches` lists files this action will write. A parameter or algorithm settings change does not write the source file that defines the setting; use an empty list unless a file is patched.

Return one JSON object: {"label": "<short and unique>", "granularity": "<param|code|algo>", "mechanism": "<what it changes>", "change": {...}, "risk": "<low|medium|high>", "crosses": "<none|R1|R2|R3|R4|R5|R6>", "why": "<why this kind, from the evidence>", "evidence": ["<what supports it>"], "touches": ["<files it would change>"]}
