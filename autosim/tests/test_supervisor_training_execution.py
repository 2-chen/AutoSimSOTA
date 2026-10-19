import json

import pytest
from autosim.research.common import atomic_json
from autosim.research.supervisor import _safe_json

from autosim.research import supervisor as m


def fixture(root, argv=None):
    attempt='a'*32
    ref='research/derived/attempts/'+attempt+'/receipt.json'
    row={'run_id':'derived','attempt_id':attempt,'stage':'train','status':'completed',
        'returncode':0,'requested_seed':123,'argv':argv or ['python','train.py','--steps','1000']}
    atomic_json(root/ref,row)
    proof={'receipt_ref':ref,'status':'completed','returncode':0}
    return {'train':{'attempt_id':attempt}},ref,row,proof


def invoke(root, measurement, proof, label='round_1', run='derived'):
    return m._training_execution(root,run,label,measurement,read_json=_safe_json,
        read_evidence=lambda *args,**kwargs:proof)


def test_public_flags_and_requested_work_not_effective_initialization(tmp_path):
    measurement,ref,row,proof=fixture(tmp_path)
    value=invoke(tmp_path,measurement,proof)
    assert value['status']=='sealed_development_train_invocation'
    assert '--steps 1000' in value['invocation_display']
    assert value['display_is_executable'] is False
    assert 'not evidence of effective configuration' in value['authority']
    assert 'shared initial parameter bytes' in value['authority']
    assert value['requested_seed']==123
    assert 'source_namespace' not in value


@pytest.mark.parametrize('argv,secret',[
    (['python','train.py','--api-key','sk-test-do-not-send-1234567890'],'sk-test-do-not-send-1234567890'),
    (['python','train.py','--password','two secret words'],'two secret words'),
    (['python','train.py','--token=private-test-value'],'private-test-value'),
    (['python','train.py','--pretrained-policy','/home/private_owner/models/confidential/checkpoint'],'private_owner'),
    (['python','train.py','--dataset','/tmp/private-data'],'private-data'),
])
def test_display_never_exports_secret_or_private_path(tmp_path,argv,secret):
    measurement,ref,row,proof=fixture(tmp_path,argv)
    value=invoke(tmp_path,measurement,proof)
    assert value['status']=='sealed_development_train_invocation'
    assert secret not in json.dumps(value)


@pytest.mark.parametrize('changes',[{'stage':'evaluate'},{'status':'failed'},{'returncode':1},
    {'run_id':'other'},{'attempt_id':'b'*32},{'argv':['x']*257},{'argv':['a\x00b']},
    {'argv':['x'*25000]},{'argv':'python train.py'}])
def test_malformed_or_wrong_producer_refused(tmp_path,changes):
    measurement,ref,row,proof=fixture(tmp_path);row.update(changes);atomic_json(tmp_path/ref,row)
    assert invoke(tmp_path,measurement,proof)['status']=='unavailable'


def test_proof_from_other_run_refused(tmp_path):
    measurement,ref,row,proof=fixture(tmp_path);proof['receipt_ref']=ref.replace('derived','other')
    assert invoke(tmp_path,measurement,proof)['status']=='unavailable'


@pytest.mark.parametrize('label,measurement',[('confirmation',{}),('confirmation_1',{}),
    ('round_1',{'confirmation':True})])
def test_confirmation_never_reads_a_receipt(tmp_path,label,measurement):
    def forbidden(*args,**kwargs):raise AssertionError('held-out receipt must not be read')
    result=m._training_execution(tmp_path,'derived',label,measurement,read_json=forbidden,read_evidence=forbidden)
    assert result['status']=='unavailable'


def test_parent_symlink_refused(tmp_path):
    measurement,ref,row,proof=fixture(tmp_path)
    source=tmp_path/'research/derived/attempts';target=tmp_path/'other_attempts';source.rename(target);source.symlink_to(target)
    assert invoke(tmp_path,measurement,proof)['status']=='unavailable'


def test_historical_training_outcome_is_supported_without_inventing_train_id(tmp_path):
    measurement,ref,row,proof=fixture(tmp_path);measurement={'training_outcome':measurement['train']}
    assert invoke(tmp_path,measurement,proof)['status']=='sealed_development_train_invocation'
    assert invoke(tmp_path,{},proof)['status']=='unavailable'


@pytest.mark.parametrize('option',['--password','--config.api_key','--auth-token','--session-cookie',
    '--wandb-api-key','--access_token','API_KEY=','--credential=','--auth','--header'])
def test_credential_values_are_removed_before_shell_quoting(tmp_path,option):
    secret='not-public secret with spaces'
    argv=['python','train.py']+([option+secret] if option.endswith('=') else [option,secret])
    assert secret not in m._display_argv(argv,tmp_path)


def test_noncredential_token_budget_and_policy_option_names_preserved(tmp_path):
    display=m._display_argv(['python','train.py','--token-budget','2048','--pretrained-policy','/home/private/model'],tmp_path)
    assert '--token-budget 2048' in display and '--pretrained-policy' in display
    assert 'private' not in display


def test_changed_receipt_bound_by_original_recovery_refused(tmp_path):
    from autosim.research.common import digest
    measurement,ref,row,proof=fixture(tmp_path)
    measurement['handoff_correction']={'training_receipt_sha256':digest(tmp_path/ref)}
    assert invoke(tmp_path,measurement,proof)['status']=='sealed_development_train_invocation'
    row['argv']+=['--different-configuration'];atomic_json(tmp_path/ref,row)
    assert invoke(tmp_path,measurement,proof)['status']=='unavailable'


def test_receipt_changed_during_read_refused(tmp_path):
    measurement,ref,row,proof=fixture(tmp_path)
    def changing(path,**kwargs):
        value=_safe_json(path,**kwargs);row['argv']+=['--changed'];atomic_json(path,row);return value
    result=m._training_execution(tmp_path,'derived','round_1',measurement,read_json=changing,
        read_evidence=lambda *a,**k:proof)
    assert result['status']=='unavailable'
