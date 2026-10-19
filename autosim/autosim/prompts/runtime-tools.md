# Evidence and diagnostic tools

Large successful tool fields and request evidence may be represented by
{context_paged:true, preview, evidence_id, read_instruction}. The preview is
incomplete, not proof that a dependency, source connection or artifact is absent.
Read the full JSON with read_evidence and bounded offsets when needed; string
blobs decode to the original text. read_evidence responses are never recursively
paged. Immediate failures, stderr, protocol, permission and budget facts remain
inline; raw evidence is retained. Paging does not authorize execution or new data.
Reuse source-backed findings only while their source/schema/input identities still
match. Do not repeatedly read unchanged full logs or ask a subagent to rediscover
an already verified interface. A cached finding is not a passing native verdict:
repair and scoring still require their original operation and identity checks.

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

Source namespaces are different, even when arguments have the same name:

* submit_research_task/research_task source_paths selects BENCHMARK checkout files only.
  A framework catalog entry such as research/data_versions.py is not a benchmark path.
* Read a public framework catalog file with inspect_native_environment(purpose=...,
  framework_source_request={path:"research/data_versions.py",query:"resolve",limit:100}).
  Allowed names come from harness_repair.source_catalog. This needs no snapshot copy,
  interpreter installation, native process or edit mode; it does NOT permit framework edits.
* ONLY propose_harness_repair(source_paths=...) selects framework-catalog paths for an
  isolated repair candidate; never reuse that scope list as a benchmark reader snapshot.
* source_request uses published native SDK root indexes, not either of those namespaces.

For checkout code/config citations (especially JSON paths hidden by privacy), call
inspect_native_environment with purpose and checkout_source_request={ref:
"relative/file.json:101-109"}. It captures actual source bytes without executing
code or needing simulation readiness. Copy the returned checkout-source:ID and
entire quote exactly into source_refs. Do not reconstruct [LOCAL_RESOURCE] paths.
For a sealed TRAIN producer in this run, checkout_source_request may instead add
producer_evidence_id and view="patch" with ref="relative/train.py", or view="source"
with ref="relative/train.py:101-160". This reads its verified candidate snapshot/delta
without restoring a checkout or guessing private run paths. Use it to recover the
COMPLETE previously working support patch, not just infer its write call from logs.
Historical output is producer-source:ID/evidence_id, NOT a checkout-source citation
for current protocol/capability approval. It does not assert the whole executed import
graph, current installation, consumption or score. Inspect current source separately,
then explicitly compose and review a new proposal; no auto-replay or history edits.

Plain checkout citations accept file, file:line, or file:start-end; quotes must be
verbatim inside that range. Source IDs prove bytes, not independent evaluation
conditions or scientific approval. If source_citation_invalid is returned, use
its citation_index/code to repair that input, not the environment or the metric.
Use read_evidence with its 32-hex evidence_id to page beyond the excerpt. Ordinary
printed diagnostic text is not installed-source identity. The source/context is
revalidated during review and use; changed packages require fresh capture/review.

If you do not know the dependency root indexes, call inspect_native_environment
with purpose and source_catalog=true first. The catalog returns numeric roots
and relative names, not a host filesystem grant. Init/Fix/Data and Scheduler retain
this bounded source/CPU inspection in read-only mode. Do not switch to edit mode,
invalidate verified commands, recreate an overlay or install an SDK merely to
read its source. A missing remembered tool is not a measured access refusal:
check current configured tools and CLI discovery; attempt the authorized tool
and report its sealed error ID if refused. A dependency-root capture is different
from builtin Read, whose checkout restriction is intentional.

Objective has SOURCE-ONLY native access: source_catalog, source_request or checkout_source_request,
never code, policy_request or data_profile_request. Read the supplied authenticated
confirmation_reservation for its original conditions; an action_receipts/<16hex> filename
is not a 32-hex evidence_id. A stale citation does not erase the original sealed intent.
Read-only reader tasks use owner-sealed public source IDs. Their CPU diagnostic context,
when authorized for that role, borrows only the selected interpreter/dependencies read-only;
the delegated source snapshot is mapped in place of the parent checkout. This is not access
to live parent source, accounting or GPU, and omitted source files remain unavailable.
