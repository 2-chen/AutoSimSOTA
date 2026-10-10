# Evidence and diagnostic tools

Use read_evidence with sealed IDs for failure logs. inspect_workspace_resources
shows explicit resource bindings; empty checkout mount placeholders are not proof
of missing data. Search public sources only when relevant, then read the source:
search hits and network failures are not evidence of resource validity or absence.
Never put secrets, private paths or raw data into search queries. Binary acquisition
requires a reviewed native acquisition operation, not diagnostic tool downloads.

inspect_native_environment runs bounded read-only CPU code in the selected native
environment; it is not a rollout or a new score. To check an already measured local
development policy, explicitly provide policy_request with its exact label and
measurement_sha256 from frozen_policy_trials. The executor verifies the measurement
and archive and exposes ONLY that archive read-only at
os.environ['AUTOSIM_DIAGNOSTIC_POLICY']. Use that variable to load its actual config
or weights on CPU; never search guessed private output paths. State whether a check
used actual frozen weights or a synthetic/uninitialized fixture. The grant cannot
access the full run, other checkpoints, test data, GPU or downloads. A rejected grant
is evidence to inspect, not permission to guess a different path or recreate weights.

For installed SDK source quotations, source_request on inspect_native_environment
captures an executor-read file under a published dependency root (root_index 0 is
the interpreter prefix; 1..N are dependency_roots), without executing diagnostic
code. Use the returned native-source:ID as source_refs.ref and quote exact source.
Use read_evidence with its 32-hex evidence_id to page beyond the excerpt. Ordinary
printed diagnostic text is not installed-source identity. The source/context is
revalidated during review and use; changed packages require fresh capture/review.
