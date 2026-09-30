import json

from autosim.research.robotwin_evidence import analyze, paired


def make_evidence(tmp_path, name, outcomes):
    trace = tmp_path / f"{name}.jsonl"
    trace.write_text("\n".join(json.dumps({"status": "episode_counted", "seed": seed,
                                                    "success": success, "action_steps": 12})
                                      for seed, success in outcomes) + "\n")
    checkpoint = tmp_path / f"{name}.ckpt"
    checkpoint.write_bytes(name.encode())
    return analyze(trace, checkpoint, task="task", setting="randomized", max_actions=20,
                   purpose="selection", evaluation_seed=1)


def test_analyze_and_pair_native_traces(tmp_path):
    baseline = make_evidence(tmp_path, "base", [(10, False), (11, True), (12, False)])
    candidate = make_evidence(tmp_path, "candidate", [(10, True), (11, True), (12, False)])
    assert baseline["metrics"]["summary"]["success_count"] == 1
    result = paired(candidate, baseline)
    assert result["candidate_only_successes"] == 1
    assert result["baseline_only_successes"] == 0
    assert result["success_delta"] == 1 / 3


def test_skips_are_evidence_but_not_counted(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"status": "seed_skipped", "seed": 9,
                                 "reason": "expert_failed"}) + "\n" +
                     json.dumps({"status": "episode_counted", "seed": 10,
                                 "success": False, "action_steps": 20}) + "\n")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"model")
    result = analyze(trace, checkpoint, task="task", setting="randomized", max_actions=20,
                     purpose="development", evaluation_seed=0)
    assert result["metrics"]["summary"]["episode_count"] == 1
    assert result["skipped_seed_reasons"] == {"expert_failed": 1}
    assert result["failure_categories"]["timeout_or_step_limit"] == 1
