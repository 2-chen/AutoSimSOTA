"""Continuous and minimizing metrics do not inherit success-rate assumptions."""

import pytest

from autosim.research.metric_contract import MetricSpec, resolve_metric_artifact


def test_exact_continuous_metric_with_direction_and_range():
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "mean_reward", "direction": "maximize", "unit": "return",
        "min": -100, "max": 100}}})
    reading = spec.read_log("best_mean_reward: 90\nmean_reward: -3.25\n")
    assert reading["value"] == -3.25
    assert spec.utility(reading["value"]) == -3.25
    assert spec.read_log("mean_reward: 999")["value"] is None
    assert spec.read_log("mean_reward:\n5")["value"] is None


def test_minimizing_metric_uses_negative_utility():
    spec = MetricSpec.from_declaration({"task_contract": {"primary_metric": {
        "name": "collision_cost", "direction": "minimize", "unit": "contacts"}}})
    assert spec.read_log("collision_cost = 4")["value"] == 4
    assert spec.utility(4) > spec.utility(7)


def test_json_result_is_preferred_when_declared_and_requires_a_scalar(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "json",
        "json_key": "summary.mean_reward", "unit": "reward"}}})
    artifact = tmp_path / "result.json"
    artifact.write_text('{"summary":{"mean_reward":12.5}}', encoding="utf-8")
    assert spec.read(said="return: 999", artifact=artifact)["value"] == 12.5
    artifact.write_text('{"summary":{"mean_reward":[12.5]}}', encoding="utf-8")
    assert spec.read(said="return: 999", artifact=artifact)["value"] is None


def test_csv_metric_requires_declared_aggregation_and_sample_count(tmp_path):
    artifact = tmp_path / "scores.csv"
    artifact.write_text("task,return\na,1.0\nb,3.0\n", encoding="utf-8")
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "csv",
        "csv_column": "return", "aggregation": "mean", "min_samples": 2}}})
    reading = spec.read(said="return: 999", artifact=artifact)
    assert reading["value"] == 2.0
    assert reading["samples"] == 2
    no_aggregation = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "csv"}}})
    assert no_aggregation.read(said="", artifact=artifact)["value"] is None
    artifact.write_text("task,return\na,1.0\n", encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None


def test_csv_rejects_ambiguous_columns_and_non_finite_values(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "reward", "direction": "maximize", "source": "csv",
        "aggregation": "mean"}}})
    artifact = tmp_path / "scores.csv"
    artifact.write_text("reward,reward\n1,2\n", encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None
    artifact.write_text("reward\nNaN\n", encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None
    with pytest.raises(ValueError, match="min_samples"):
        MetricSpec.from_declaration({"research_goal": {"primary_metric": {
            "name": "reward", "direction": "maximize", "min_samples": 0}}})
    with pytest.raises(ValueError, match="episode identity"):
        MetricSpec.from_declaration({"research_goal": {"primary_metric": {
            "name": "reward", "direction": "maximize", "source": "json",
            "min_samples": 2}}})
    bounded = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "csv",
        "aggregation": "mean"}}})
    artifact.write_text("success_rate\n2\n-1\n", encoding="utf-8")
    assert bounded.read(said="", artifact=artifact)["value"] is None


def test_native_csv_episode_identities_supply_actual_success_totals(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "csv",
        "aggregation": "mean", "episode_id_column": "episode_id",
        "task_column": "task"}}})
    artifact = tmp_path / "episodes.csv"
    artifact.write_text("task,episode_id,success_rate\nlift,0,1\nlift,1,0\n",
                        encoding="utf-8")
    reading = spec.read(said="", artifact=artifact)
    assert reading["value"] == 0.5
    assert reading["episodes_completed"] == 2
    assert reading["successes"] == 1
    assert reading["episode_keys"] == ["lift\x1f0", "lift\x1f1"]
    artifact.write_text("task,episode_id,success_rate\nlift,0,1\nlift,0,0\n",
                        encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None


def test_native_csv_can_bind_episode_outcomes_to_initial_state_hashes(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "csv",
        "aggregation": "mean", "episode_id_column": "episode_id",
        "task_column": "task", "initial_state_hash_column": "initial_state_sha256"}}})
    artifact = tmp_path / "episodes.csv"
    artifact.write_text("task,episode_id,initial_state_sha256,success_rate\n"
                        f"lift,0,{'a' * 64},1\n"
                        f"lift,1,{'b' * 64},0\n", encoding="utf-8")
    reading = spec.read(said="", artifact=artifact)
    assert reading["episode_values"] == [1.0, 0.0]
    assert reading["initial_state_hashes"] == ["a" * 64, "b" * 64]
    artifact.write_text("task,episode_id,initial_state_sha256,success_rate\n"
                        "lift,0,not-a-hash,1\n", encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None


def test_native_json_episode_records_bind_task_ids_and_aggregate(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "eval_success_once_mean", "direction": "maximize", "unit": "fraction",
        "source": "json", "json_key": "episodes", "json_value_key": "success",
        "aggregation": "mean", "min_samples": 2,
        "episode_id_column": "episode_id", "task_column": "env_info.env_id"}}})
    artifact = tmp_path / "trajectory.json"
    artifact.write_text(
        '{"env_info":{"env_id":"PushCube-v1"},"episodes":['
        '{"episode_id":0,"episode_seed":12,"success":false},'
        '{"episode_id":1,"episode_seed":13,"success":true}]}', encoding="utf-8")

    reading = spec.read(said="eval_success_once_mean=0.5", artifact=artifact)
    assert reading["value"] == 0.5
    assert reading["episodes_completed"] == 2
    assert reading["episode_keys"] == ["PushCube-v1\x1f0", "PushCube-v1\x1f1"]
    assert reading["episode_values"] == [0.0, 1.0]


def test_json_episode_identity_rejects_duplicate_ids_or_unmapped_metric_fields(tmp_path):
    declaration = {"research_goal": {"primary_metric": {
        "name": "success", "direction": "maximize", "unit": "fraction",
        "source": "json", "json_key": "episodes", "json_value_key": "success",
        "aggregation": "mean", "episode_id_column": "episode_id",
        "task_column": "task"}}}
    spec = MetricSpec.from_declaration(declaration)
    artifact = tmp_path / "episodes.json"
    artifact.write_text('{"episodes":[{"task":"lift","episode_id":0,"success":1},'
                        '{"task":"lift","episode_id":0,"success":0}]}',
                        encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None
    artifact.write_text('{"episodes":[{"task":"lift","episode_id":1}]}',
                        encoding="utf-8")
    assert spec.read(said="", artifact=artifact)["value"] is None


def test_metric_artifact_pattern_is_safe_and_resolves_one_fresh_file(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "return", "direction": "maximize", "source": "json",
        "artifact_root": "output", "artifact_pattern": "scores/*.json",
        "json_key": "return"}}})
    root = tmp_path / "out"
    (root / "scores").mkdir(parents=True)
    first = root / "scores" / "one.json"
    first.write_text('{"return":1}', encoding="utf-8")
    result = resolve_metric_artifact(spec, roots={"output": root},
                                     started_at=first.stat().st_mtime - 1,
                                     allowed_roots=[root])
    assert result["status"] == "matched"
    assert result["sha256"]
    (root / "scores" / "two.json").write_text('{"return":2}', encoding="utf-8")
    ambiguous = resolve_metric_artifact(spec, roots={"output": root},
                                        started_at=first.stat().st_mtime - 1,
                                        allowed_roots=[root])
    assert ambiguous["status"] == "ambiguous"
    stale = resolve_metric_artifact(spec, roots={"output": root},
                                    started_at=first.stat().st_mtime + 1,
                                    allowed_roots=[root])
    assert stale["status"] == "missing"


def test_metric_artifact_can_be_bound_relative_to_the_evaluated_policy_parent(tmp_path):
    spec = MetricSpec.from_declaration({"research_goal": {"primary_metric": {
        "name": "success_rate", "direction": "maximize", "source": "json",
        "artifact_root": "policy_parent", "artifact_pattern": "test_videos/trajectory.json",
        "json_key": "episodes", "json_value_key": "success", "aggregation": "mean",
        "episode_id_column": "episode_id", "task_column": "env_info.env_id"}}})
    policy_parent = tmp_path / "experiments" / "baseline" / "attempt"
    sidecar = policy_parent / "test_videos" / "trajectory.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(
        '{"env_info":{"env_id":"PushCube-v1"},"episodes":'
        '[{"episode_id":0,"success":false},{"episode_id":1,"success":true}]}',
        encoding="utf-8")

    resolved = resolve_metric_artifact(
        spec, roots={"policy_parent": policy_parent},
        started_at=sidecar.stat().st_mtime - 1, allowed_roots=[tmp_path])

    assert resolved["status"] == "matched"
    assert resolved["path"] == str(sidecar.resolve())
    assert spec.read(said="eval_success_once_mean=0.5", artifact=sidecar)["value"] == 0.5


@pytest.mark.parametrize("pattern", ["../result.json", "/tmp/result.json"])
def test_metric_artifact_pattern_cannot_escape_its_declared_root(pattern):
    with pytest.raises(ValueError, match="safe relative path"):
        MetricSpec.from_declaration({"research_goal": {"primary_metric": {
            "name": "return", "direction": "maximize", "source": "json",
            "artifact_root": "output", "artifact_pattern": pattern,
            "json_key": "return"}}})
