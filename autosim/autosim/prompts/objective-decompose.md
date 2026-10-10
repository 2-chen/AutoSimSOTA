You are breaking a research goal into sub-questions a run can answer about itself, before it can answer the final one.

You are given the benchmark, the stages it has, what it declares about its evaluation, and what is known about how it runs. The top-level questions are fixed and given to you:

{tops}

For each of them, propose the benchmark-specific questions that sit underneath -- what has to be true *for this benchmark* before that top-level question can be answered yes. Be concrete and make each one answerable from what a run records: "the evaluator used the benchmark's own initial states" is answerable; "the environment is good" is not.

Do not propose anything about how a metric is computed, the evaluation entry point, the splits, or the data. Those are not sub-goals; they are what must not change.

Return one JSON object: {{"<top-level name>": [{{"name": "<short>", "question": "<yes/no, answerable from the run's records>", "weight": <a number>}}]}}. Weight the children of one top-level question against each other; their total is that question's weight, not more.
