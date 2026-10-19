"""Paid narration responds to material changes, not the age of identical facts."""
from copy import deepcopy

from autosim.research import recorder
from autosim.research.common import atomic_json, read_json
from tests.test_recorder import snapshot, Writer


def test_old_successful_prose_does_not_rebuy_for_elapsed_time_or_receipt_churn(tmp_path):
    held=snapshot(tmp_path);writer=Writer()
    original=recorder.narrate(tmp_path,held,client=writer)
    original['written_epoch']=0
    atomic_json(tmp_path/'report/narrative.json',original)
    routine=deepcopy(held);routine['event_revision']='new-receipt-same-facts'
    routine['actions']=[{'step':'inspect','outcome':'completed','finished_at':'later'}]
    assert recorder.narrate(tmp_path,routine,client=writer)==original
    assert len(writer.calls)==3
    assert 'live tables' in read_json(tmp_path/'report/recorder_deferred.json')['reason']
    routine['runtime_failure']={'category':'new_native_failure'}
    routine['event_revision']='new-error'
    recorder.narrate(tmp_path,routine,client=writer)
    assert len(writer.calls)==6


def test_old_failed_prose_waits_for_new_evidence_not_a_periodic_paid_retry(tmp_path):
    held=snapshot(tmp_path);writer=Writer('service_failure')
    recorder.narrate(tmp_path,held,client=writer)
    trigger=read_json(tmp_path/'report/recorder_trigger.json');trigger['attempted_epoch']=0
    atomic_json(tmp_path/'report/recorder_trigger.json',trigger)
    routine=deepcopy(held);routine['event_revision']='age-only'
    recorder.narrate(tmp_path,routine,client=writer)
    assert len(writer.calls)==1
    routine['runtime_recovery']={'status':'new_provider_recovery_evidence'}
    routine['event_revision']='provider-recovered'
    recorder.narrate(tmp_path,routine,client=writer)
    assert len(writer.calls)==2


def test_new_data_or_research_plan_is_a_material_narration_event(tmp_path):
    held=snapshot(tmp_path);writer=Writer()
    recorder.narrate(tmp_path,held,client=writer)
    changed=deepcopy(held);changed['event_revision']='new-data'
    changed['data_versions']=[{'id':'new-training-asset','bytes':10}]
    recorder.narrate(tmp_path,changed,client=writer)
    assert len(writer.calls)==5  # The existing skill choice is reused for data updates.
    changed['event_revision']='new-plan'
    changed['plan']={'next_question':'test a different causal explanation'}
    recorder.narrate(tmp_path,changed,client=writer)
    assert len(writer.calls)==7
