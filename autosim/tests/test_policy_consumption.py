import hashlib
import json

from autosim.research.policy_consumption import (verify_policy_consumption,
                                                 verify_rollout_evidence,
                                                 verify_metric_lineage)


def test_published_identity_recipe_matches_verifier_for_file_and_nested_directory(tmp_path):
    from autosim.research.policy_consumption import identity_contract
    from autosim.research.experiment_bundle import artifact_identity
    contract = identity_contract()
    namespace = {}
    exec(contract['identity_recipe_python'], namespace)
    directory = tmp_path/'policy'
    (directory/'nested').mkdir(parents=True)
    (directory/'nested'/'权重.bin').write_bytes(b'weights')
    (directory/'config.json').write_text('{}')
    for path in [directory, directory/'nested'/'权重.bin']:
        assert namespace['loaded_artifact_identity'](path) == artifact_identity(path)
    assert contract['rollout_fields'] == ['episode_id', 'policy_sha256']
    assert contract['metric_fields'] == ['policy_sha256', 'episode_ids', 'value']
    log = tmp_path/'native.log'
    field, value = namespace['loaded_artifact_identity'](directory)
    log.write_text(contract['loaded_marker'] + json.dumps({'path':str(directory),field:value})+'\n'
        +contract['rollout_marker']+json.dumps({'episode_id':'0','policy_sha256':value})+'\n'
        +contract['metric_marker']+json.dumps({'episode_ids':['0'],'policy_sha256':value,'value':0})+'\n')
    assert verify_policy_consumption(log, directory)['status'] == 'verified'
    assert verify_rollout_evidence(log, policy_sha256=value)['status'] == 'verified'
    assert verify_metric_lineage(log, policy_sha256=value, episode_ids=['0'], value=0)['status'] == 'verified'


def test_schema_audit_reports_wrong_directory_field_and_episode_keys_without_certification():
    from autosim.research.policy_consumption import audit_identity_schema
    text = 'AUTOSIM_POLICY_LOADED ' + json.dumps({'path':'/run/policy','content_sha256':'a'*64})+'\n'
    text += 'AUTOSIM_ROLLOUT_COMPLETED ' + json.dumps({'episode_index':'0','policy_sha256':'a'*64})+'\n'
    text += 'AUTOSIM_METRIC_REPORTED ' + json.dumps({'episode_index':['0'],'policy_sha256':'a'*64,'value':0})+'\n'
    audit = audit_identity_schema(text, policy_kind='directory')
    assert audit['status'] == 'incompatible' and len(audit['issues']) == 3
    assert 'sha256' in audit['issues'][0]
    assert 'episode_id string' in audit['issues'][1]
    assert 'episode_ids as' in audit['issues'][2]
    assert 'not consumption or scoring evidence' in audit['authority']
    good = ('AUTOSIM_POLICY_LOADED '+json.dumps({'path':'/run/policy','sha256':'a'*64})+'\n'
            +'AUTOSIM_ROLLOUT_COMPLETED '+json.dumps({'episode_id':'0','policy_sha256':'a'*64})+'\n'
            +'AUTOSIM_METRIC_REPORTED '+json.dumps({'episode_ids':['0'],'policy_sha256':'a'*64,'value':0})+'\n')
    assert audit_identity_schema(good, policy_kind='directory')['status'] == 'schema_observed'
    assert audit_identity_schema('native timing only')['status'] == 'unavailable'


def test_native_policy_load_event_binds_frozen_bytes(tmp_path):
    policy = tmp_path / "policy.pth"
    policy.write_bytes(b"candidate weights")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    log = tmp_path / "evaluation.log"
    log.write_text("AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": digest}) + "\nsuccess: 0.75\n",
        encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "verified"
    policy.write_bytes(b"different weights")
    assert verify_policy_consumption(log, policy)["status"] == "mismatch"


def test_missing_or_conflicting_policy_load_event_does_not_verify(tmp_path):
    policy = tmp_path / "policy.pth"
    policy.write_bytes(b"weights")
    log = tmp_path / "evaluation.log"
    log.write_text("success: 0.9\n", encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "unverified"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    marker = "AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": digest}) + "\n"
    log.write_text(marker * 2, encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "verified"
    conflicting = "AUTOSIM_POLICY_LOADED " + json.dumps({
        "path": str(policy), "content_sha256": "0" * 64}) + "\n"
    log.write_text(marker + conflicting, encoding="utf-8")
    assert verify_policy_consumption(log, policy)["status"] == "mismatch"


def test_rollout_witness_requires_same_policy_and_completed_episode(tmp_path):
    log = tmp_path / "evaluation.log"
    identity = "a" * 64
    log.write_text("success: 1.0\n", encoding="utf-8")
    assert verify_rollout_evidence(log, policy_sha256=identity)["status"] == "unverified"
    log.write_text("AUTOSIM_ROLLOUT_COMPLETED " + json.dumps({
        "episode_id": "task-0/seed-3", "policy_sha256": identity}) + "\n",
        encoding="utf-8")
    assert verify_rollout_evidence(log, policy_sha256=identity)["status"] == "verified"
    assert verify_rollout_evidence(log, policy_sha256="b" * 64)["status"] == "mismatch"


def test_native_metric_lineage_binds_policy_episode_set_and_value(tmp_path):
    log = tmp_path / "evaluation.log"
    identity = "a" * 64
    log.write_text("AUTOSIM_METRIC_REPORTED " + json.dumps({
        "policy_sha256": identity, "episode_ids": ["one", "two"], "value": 0.5}) + "\n",
        encoding="utf-8")
    good = verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["two", "one"], value=0.5)
    assert good["status"] == "verified"
    assert verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["one"], value=0.5)["status"] == "mismatch"
    assert verify_metric_lineage(log, policy_sha256=identity,
                                 episode_ids=["one", "two"], value=0.7)["status"] == "mismatch"
