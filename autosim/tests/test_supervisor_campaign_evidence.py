import json

import pytest
from autosim.research.common import atomic_json

from autosim.research import supervisor as c


def fixture(root):
    repo=root/'repo';repo.mkdir()
    output=root/'output';research=output/'research/derived'
    atomic_json(research/'research_report.json',{'run_id':'derived','repo':str(repo),
        'run_status':'completed','rounds':[]})
    return repo,output,research


def test_packet_transports_full_recipe_and_sealed_invocation_not_only_summary(tmp_path,monkeypatch):
    from autosim.research import data_campaign,data_versions,evidence_store
    repo,output,research=fixture(tmp_path)
    attempt='a'*32
    atomic_json(research/'measurements/round_1.json',{'label':'round_1','ok':True,
        'metric_value':.2,'train':{'attempt_id':attempt},
        'data_consumption':{'status':'verified','exposure':[{
            'consumption_stage':'sample_read','samples_read':8064}]}})
    ref='research/derived/attempts/'+attempt+'/receipt.json'
    atomic_json(output/ref,{'attempt_id':attempt,'run_id':'derived','stage':'train',
        'status':'completed','returncode':0,'argv':['python','train.py','--steps','1000']})
    monkeypatch.setattr(c,'verify_measurement',lambda *a,**k:{'status':'consistent'})
    monkeypatch.setattr(evidence_store,'read_attempt_evidence',lambda *a,**k:
        {'receipt_ref':ref,'status':'completed','returncode':0})
    recipe='public shared initialization and recipe. '*30
    comparison={'arms':['random_quality','agent_targeted'],'seeds':[3],'training_recipe':recipe}
    monkeypatch.setattr(data_campaign,'view',lambda *a:[{'id':'data','schema_version':2,
        'comparison':{**comparison,'training_recipe':recipe[:400]},'observations':[]}])
    monkeypatch.setattr(data_campaign,'execution_protocols',lambda *a:[{'id':'data',
        'comparison':comparison,'plan_sha256':'b'*64}])
    monkeypatch.setattr(data_versions,'catalog',lambda *a:[])
    packet,refs=c.build_audit_packet(output=output,repo=repo,run_id='derived')
    campaign=packet['data_campaigns'][0]
    assert campaign['comparison']['training_recipe']==recipe
    assert campaign['protocol_projection']['status']=='full_frozen_protocol_available'
    proof=packet['measurements'][0]['training_execution']
    assert proof['status']=='sealed_development_train_invocation'
    assert '--steps 1000' in proof['invocation_display']
    assert proof['evidence_ref']=='training_execution:round_1'
    assert proof['evidence_ref'] in refs
    consumption=packet['measurements'][0]['data_consumption']
    assert consumption['status']=='verified'
    assert consumption['training_batch_witness']=='not_reported'
    assert consumption['reported_stages']==['sample_read']
    assert 'not evidence of effective configuration' in proof['authority']
    before=c.packet_digest(packet)
    campaign['comparison']['training_recipe']+=' changed'
    assert c.packet_digest(packet)!=before


def review_fixture():
    observations=[{'arm':arm,'seed':3,'measurement_label':label,'status':'consumed_and_measured'}
        for arm,label in [('random_quality','round_1'),('agent_targeted','round_2')]]
    campaign={'id':'data','schema_version':2,'comparison':{
        'arms':['random_quality','agent_targeted'],'seeds':[3]},'observations':observations,
        'protocol_projection':{'status':'full_frozen_protocol_available'}}
    measurements=[{'label':label,'receipt_verification':{'status':'consistent'},
        'training_execution':{'status':'sealed_development_train_invocation',
                              'evidence_ref':'training_execution:'+label}}
        for label in ['round_1','round_2']]
    refs=['data_campaign:data','measurement:round_1','measurement:round_2',
          'training_execution:round_1','training_execution:round_2']
    packet={'data_campaigns':[campaign],'measurements':measurements}
    answer={'verdict':'real','summary':'Real scores, comparison requires a separate judgment.',
        'findings':[],'evidence_refs':['measurement:round_1'],
        'data_campaign_reviews':[{'id':'data','verdict':'real','why':'Fixture review.',
                                  'evidence_refs':refs.copy()}]}
    return packet,answer,refs


def test_cited_complete_packet_can_be_judged_but_kernel_does_not_infer_effect():
    packet,answer,refs=review_fixture()
    result=c.validate_response(answer,allowed_refs=refs,packet=packet)
    assert result['data_campaign_reviews'][0]['verdict']=='real'
    assert result['data_campaign_reviews'][0]['why']=='Fixture review.'


@pytest.mark.parametrize('missing',['protocol','outcome','invocation','citation','empty_matrix'])
def test_real_metrics_and_registration_cannot_certify_incomplete_v2_comparison(missing):
    packet,answer,refs=review_fixture();campaign=packet['data_campaigns'][0]
    if missing=='protocol':campaign.pop('protocol_projection')
    if missing=='outcome':campaign['observations'].pop()
    if missing=='invocation':packet['measurements'][1]['training_execution']['status']='unavailable'
    if missing=='citation':answer['data_campaign_reviews'][0]['evidence_refs'].remove('training_execution:round_2')
    if missing=='empty_matrix':campaign['comparison']['seeds']=[]
    result=c.validate_response(answer,allowed_refs=refs,packet=packet)
    assert result['verdict']=='real'
    assert result['data_campaign_reviews'][0]['verdict']=='uncertain'
    assert 'registration and real metrics alone' in result['data_campaign_reviews'][0]['why']


def test_confirmation_measurements_do_not_project_training_invocations(tmp_path,monkeypatch):
    repo,output,research=fixture(tmp_path)
    atomic_json(research/'measurements/confirmation.json',{'label':'confirmation',
        'confirmation':True,'ok':True,'metric_value':.98765,'train':{'attempt_id':'f'*32}})
    def forbidden(*a,**k):raise AssertionError('no held-out train disclosure')
    monkeypatch.setattr(c,'_training_execution',forbidden)
    packet,refs=c.build_audit_packet(output=output,repo=repo,run_id='derived')
    assert not packet['measurements']
    assert '.98765' not in json.dumps(packet)
    assert not any(r.startswith('training_execution:') for r in refs)
