import json

from autosim.research import failure_observation as observation
from autosim.research.common import atomic_text
from autosim.research.evidence_store import capture_attempt_evidence
from autosim.research.operation_failure import seal


def test_current_envelope_not_latest_successful_checkpoint_is_observed(tmp_path):
    ident, other = 'b'*32, 'e'*32
    failed = seal(tmp_path, operation='data_validation', exc=ValueError('malformed sealed counters'),
        context={'training_attempt_id':ident, 'measurement_label':'round_11_handoff_correction'})
    result = observation.current_failure(tmp_path, {**failed, 'training_attempt_id':other,
                                                   'latest_training_attempt_id':other})
    assert result['status'] == 'verified_sealed_failure'
    assert result['subject'] == {'training_attempt_id':ident, 'measurement_label':'round_11_handoff_correction'}
    assert result['failure_status'] == 'operation_contract_failed'
    assert 'malformed sealed counters' in result['excerpt']
    assert other not in json.dumps(result)


def test_unknown_subject_cannot_be_inferred_from_latest_checkpoint(tmp_path):
    result = observation.current_failure(tmp_path, {'training_attempt_id':'e'*32})
    assert result['status'] == 'unavailable'
    assert 'subject' not in result
    failed = seal(tmp_path, operation='caller_refusal', exc=ValueError('no subject'), context={})
    assert observation.current_failure(tmp_path, failed)['subject'] == {}


def test_changed_sealed_log_is_not_observer_evidence(tmp_path):
    failed = seal(tmp_path, operation='data_validation', exc=ValueError('original'),
                  context={'training_attempt_id':'b'*32})
    (tmp_path / failed['log_ref']).write_text('modified')
    result = observation.current_failure(tmp_path, failed)
    assert result['status'] == 'unavailable'
    assert 'subject' not in result


def test_native_failed_consumer_remains_separate_from_successful_producer(tmp_path):
    ident = 'a'*32
    log = tmp_path / 'native.log'
    atomic_text(log, 'native consumer traceback')
    failed = capture_attempt_evidence(tmp_path, attempt_id=ident, log=log,
        receipt_ref='native_receipt.json', status='failed', returncode=139,
        termination_reason='nonzero_exit')
    result = observation.current_failure(tmp_path, {**failed, 'training_attempt_id':'b'*32})
    assert result['subject'] == {'failed_native_attempt_id':ident}
    assert result['returncode'] == 139


def test_monitor_receives_the_current_sealed_subject(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from tests.test_prepare import preparation
    real = preparation(tmp_path)
    real.output.mkdir(parents=True)
    failed = seal(real.output, operation='data_validation', exc=ValueError('malformed'),
                  context={'training_attempt_id':'b'*32, 'measurement_label':'round_11'})
    class Client:
        def as_role(self, role):
            return nullcontext(self)
        def chat_with_metadata(self, system, user, **kwargs):
            payload = json.loads(user)
            assert payload['current_failure']['subject']['training_attempt_id'] == 'b'*32
            assert payload['current_failure']['evidence_id'] == failed['evidence_id']
            assert kwargs['decision_only'] and kwargs['resume_session'] is False
            return json.dumps({'progress':'uncertain', 'summary':'data evidence failed',
                               'guidance':'Inspect original evidence, not the latest model', 'evidence_refs':[]}), {}
    real.client = Client()
    monkeypatch.setattr(real, '_decision_facts', lambda: {'research_progress':{'latest_training_attempt_id':'e'*32}})
    result = real._monitor_failed_action(action_id='current', action=failed)
    assert result['status'] == 'assessed'


def test_combined_data_and_bundle_failure_reaches_fix_without_replaying_handoff(tmp_path):
    from types import SimpleNamespace
    from tests.test_prepare import preparation
    real=preparation(tmp_path);real.output.mkdir(parents=True)
    row={'label':'round_2','ok':False,'where':'policy_artifact',
         'data_consumption':{'status':'unverified','why':'actual version id is null'},
         'train':{'attempt_id':'b'*32},
         'artifact_selection':{'status':'native_boundary_review_required','why':'select full bundle'}}
    result=real._data_consumer_failure(SimpleNamespace(output=real.output,run_id=real.run_id),
        row,operation='candidate_training_data_validation')
    assert result['failure_category']=='training_data_consumption_unverified'
    assert result['related_policy_handoff']==row['artifact_selection']
    assert result['native_operation_status']=='not_started'
    observed=observation.current_failure(real.output,result)
    assert observed['subject']=={'training_attempt_id':'b'*32,'measurement_label':'round_2'}
    assert 'actual version id is null' in observed['excerpt'] and 'select full bundle' in observed['excerpt']
