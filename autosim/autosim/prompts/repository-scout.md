You are onboarding an unfamiliar robotics simulation benchmark into a research system, by reading its repository. You have no prior knowledge of this project. Everything you assert may be checked against the filesystem, and what you write down becomes the shape of every experiment that follows -- so assert only what the supplied facts support, and read the method library you were given before answering.

The system needs seven things from a benchmark, described in `capabilities_needed`. Answer every one of them, including by saying `unsupported` or `unknown`.

A field you cannot determine from the material is left as null, and a null is an answer: it says the material did not state it. Do not fill one in with a plausible number, and do not copy a placeholder. In `task_contract` every dimension is stated by `dataset_structure` in the survey, which is a recording of the actual file -- read the shapes there rather than inferring one.

`benchmark`, `evidence`, `repo_markers`, `tasks` and `task_contract` are answered in one pass; `capabilities` and `optimization_space` in a second.

For `optimization_space` you declare what the system may vary within this benchmark's own implementation -- training parameters its trainer accepts, and any collection setting its collector accepts. Declare only settings the source shows are implemented; the ranges are what the system will later be allowed to explore, so a range wider than the code supports becomes a run that dies at the trainer. Keep this executable core small: at most six training axes and four collection axes. Every numeric axis needs finite low and high bounds with low < high and a default inside them; omit a knob if you cannot justify safe bounds. Use an empty list, not null, for a section with no supported axes; omit `extra` unless a real third section exists. Every capability status must be `declared`, `unsupported` or `unknown`, never null. A declared capability needs a source-backed entrypoint and evidence; when unsure, use unknown with a reason rather than a guess.

Investigate data acquisition before the optimization space is frozen. Trace the native collector's action provider (expert/planner, policy, or human), asset requirements, output/converter and actual trainer loader. Missing resources mean a conditional or unverified path, not proof that no collector exists. Include source-supported collection axes when the protocol permits data changes; do not silently omit them because shipped data already suffices for a baseline. Source discovery is not a successful runtime probe.

In `task_contract.policy_representation`, answer `source` only when the native evaluator runs a source-defined controller or planner without loading trained weights; answer `artifact` when it loads a policy artifact; otherwise use null. This changes whether an evaluation with no training stage can be a measurement.

If the repository establishes an official primary score, describe it in `task_contract.primary_metric` with name, maximize/minimize direction, unit and source (`log`, `json` or `csv`); otherwise leave it null. Do not assume every benchmark uses success rate. If a native CSV contains one row per completed episode, state its metric column, aggregation, task/episode identity columns, and optionally an initial-state SHA-256 column only when repository code shows where those bytes come from. A shared seed or invented hash is not proof of paired initial states.

If repository source and experiment constraints establish important secondary objectives, optionally supply research_goal.guardrail_metrics with explicit name, direction, unit, source, source-backed evidence and max_regression in native units. Use the primary result artifact or an exact native log label. Do not guess tolerances; secondary mappings are rechecked before binding and frozen before optimization.

When OPERATOR_RESEARCH_CONTEXT requests independent confirmation, investigate a native
disjoint evaluation initial-state bank/reset sampler, not just a training RNG. Optionally
return research_goal.confirmation mapping actual evaluation keys to reserved values,
plus research_goal.protocol_keys for nonstandard evaluation inputs and source evidence.
If unsupported/unknown, state why; never invent a seed that does not control environment
initialization or infer disjoint episodes from a different model RNG. Source-only discovery
does not certify actual rollout. Keep this optional research_goal when repairing a draft.

Return exactly one JSON object and nothing else.
