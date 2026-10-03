import json

import pytest

from autosim.research.common import sanitize_model_text, sanitize_model_payload_text
from autosim.research.execution_paths import ExecutionPaths
from autosim.research.execution_progress import execution_progress


def test_public_witness_syntax_survives_repeated_projection_without_episode_values():
    from autosim.research.common import sanitize_model_payload
    from autosim.research.policy_consumption import identity_contract
    contract = identity_contract()
    projected = contract
    for _ in range(3):
        projected = sanitize_model_payload(projected)
    assert projected["rollout_fields"] == contract["rollout_fields"]
    assert projected["metric_fields"] == contract["metric_fields"]
    assert projected["instruction"] == contract["instruction"]
    assert sanitize_model_payload_text("completed episode and metric aggregation sites") == (
        "completed episode and metric aggregation sites")
    private = sanitize_model_payload({"episode_id": "private-sample", "episode_ids": ["private-42"],
                                      "note": "episode_id: private-42 episode_foo episode 42"})
    assert "private" not in json.dumps(private)
    assert "episode_foo" not in json.dumps(private)
    assert "episode 42" not in json.dumps(private)
    assert private["episode_id"]["values_omitted"] is True
    schema = sanitize_model_payload({"fields": {"episode_id": {"type": "integer"},
                                                "sha256": {"type": "string"}}})
    assert schema["fields"]["episode_id"] == {"type": "integer"}
    assert schema["fields"]["sha256"] == {"type": "string"}


@pytest.mark.parametrize('scrub', [sanitize_model_text, sanitize_model_payload_text])
def test_serialized_python_is_not_a_windows_path(scrub):
    code = 'try:\n    import toppra\nexcept BaseException as e:\n    print(str(e))\n    raise\n'
    original = {'commands': [code], 'reasoning': 'verify native imports'}
    wire = json.dumps(original)
    for _ in range(3):
        wire = scrub(wire)
    assert json.loads(wire) == original
    compile(json.loads(wire)['commands'][0], '<probe>', 'exec')


@pytest.mark.parametrize('scrub', [sanitize_model_text, sanitize_model_payload_text])
def test_json_still_hides_real_windows_paths_secrets_and_host_paths(scrub):
    original = {'code': 'except Exception as e:\n    raise\n',
                'private': ['C:\\Users\\operator\\secret.txt', '/home/operator/private',
                            'api_key=sk-abcdefghijklmnopqrstuvwxyz123456']}
    safe = json.loads(scrub(json.dumps(original)))
    assert safe['code'] == original['code']
    assert all('operator' not in item and 'abcdefghijklmnopqrstuvwxyz' not in item
               for item in safe['private'])


def test_path_rejection_identifies_exact_field_and_does_not_rewrite(tmp_path):
    paths = ExecutionPaths(tmp_path/'checkout', tmp_path)
    operations = {'commands': ['echo safe', 'python [LOCAL_PATH]']}
    with pytest.raises(ValueError, match=r'commands\[1\]'):
        paths.operations(operations)
    assert operations['commands'] == ['echo safe', 'python [LOCAL_PATH]']


def test_prose_and_new_failure_ids_are_not_capability_progress():
    steps = []
    for i in range(3):
        steps += [{'step': 'build_the_environment', 'outcome': 'repair proposal rejected',
                   'evidence_id': str(i)}, {'step': 'update_research_plan', 'outcome': 'recorded'}]
    assert execution_progress(steps)['needs_strategy_change']
    assert execution_progress(steps)['failure_count'] == 3


@pytest.mark.parametrize('success', [
    {'step': 'build_the_environment', 'outcome': 'checkpoint', 'evidence_id': 'new-native'},
    {'step': 'run_the_loop', 'outcome': 'completed', 'verification_level': 'L2'},
])
def test_native_progress_resets_warning(success):
    steps = [{'step': 'build_the_environment', 'outcome': 'checkpoint failure'}]*3
    assert not execution_progress(steps + [success])['needs_strategy_change']


def test_plans_or_live_jobs_alone_are_not_failed_progress():
    assert not execution_progress([{'step': 'wait_for_jobs', 'outcome': 'event'}]*8)[
        'needs_strategy_change']
