# Independent system validation

These tools do not modify the frozen `experiment_system` runtime or the running
RoboSyn experiment. They are additive diagnostics and bounded scheduling support.

- `fault_suite`: 2 × 2 receipt-recovery/retry ablation with real CPU subprocesses,
  identical validators and bounded synthetic faults. Not a robot-policy benchmark
  or a comparison against published systems. Always use a fresh output directory.
- `priority_guard`: holds only the pre-existing production continuation gate,
  never a GPU or active-job lock. Pins the queue PID plus Linux start time and
  scheduler source. Releases on terminal queue state, changed source, exited
  queue process, error, or immutable absolute deadline. Does not preempt current
  work. A transient service lifetime bounds it even if it stops reporting state.
- `asset_audit`: hashes all copied objects, embodiments, textures and installed
  cuRobo files; checks selected URDF/collision references stay in the clean native
  tree. Records local content, not upstream provenance or GPU compatibility.
  The existing frozen queue does not yet enforce this supplementary manifest.
- `integration_remediation`: version-2 successor to the immutable failed native
  integration queue.  It pins an isolated Warp 1.12 overlay for RoboTwin and
  preserves numerical policy failures as failed smoke-test episodes instead of
  misclassifying them as infrastructure crashes.  Its safe evaluator is explicitly
  non-ranking and never changes the official success predicate, seeds, or horizon.

Tests: `test_experiment_validation.py`, `test_fault_suite.py`, `test_asset_audit.py`,
`test_integration_remediation.py`.
Measured outputs and limitations are documented in
`report/autonomous_system_fault_comparison_20260906.md` at the workspace root.

The priority guard is not a general multi-user GPU scheduler. Other schedulers
must obey the existing locks, and priority is only valid for its bounded window.
Stopping its service releases the gate; old experiments are never killed by it.
