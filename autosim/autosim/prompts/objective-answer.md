A research run wrote itself a list of yes/no questions about what has to be true for its work to count, before it started. Answer them from what the run has recorded.

You are given the questions and the run's records: what each round did, what it was refused for, what came out of the measurements, and which stages exist.

Every answer is one of three things, and they are not the same:

* `true` -- the records show it. Give `because` as the fact that shows it, and name where in the records it is.
* `false` -- the records show it is false, or show that a run happened without it.
* `null` -- the run has not got far enough for this question to have an answer. **This is the answer for anything the run has not reached**, and it is not a failure. A run that has not yet evaluated anything has not failed to use the benchmark's own initial states.

This is a record interpretation, not another investigation or repair. Use only the supplied material; do not execute commands, inspect unrelated files or select skills. Missing evidence is `null`, not a reason to reopen environment preparation.

Do not answer `true` from what the run *intends* to do, or from what it says it did. Answer from what the records contain. If you cannot point at the record that shows it, the answer is `null`.

Return one JSON object: {"answers": [{"name": "<the question's name>", "passed": true | false | null, "because": "<the record that shows it, or why there is none>"}]}. Answer every question you were given.

Be compact: each because is one short clause, at most 100 characters. For missing evidence
use a short null explanation. Do not reproduce the questions, a narrative or a code fence.
