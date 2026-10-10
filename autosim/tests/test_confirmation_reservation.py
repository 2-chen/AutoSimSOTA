"""Source-reviewed reservation repair, not proof of independent robot scores."""
import json
from contextlib import contextmanager

import pytest

from autosim.research import confirmation_reservation as reservation
from autosim.research.common import atomic_json, read_json
from tests.test_prepare import preparation


def inputs(tmp_path):
    repo = tmp_path/'repo'; repo.mkdir()
    (repo/'evaluate.py').write_text('def reset(seed):\n    return initial_bank[seed]\n')
    proposal = dict(confirmation={'seed': 101},protocol_keys=[],reason='Separate native training-admissible and held-out initial banks.',
        source_refs=[dict(ref='evaluate.py',quote='return initial_bank[seed]')])
    review = dict(native_evaluation_input_supported=True,disjoint_conditions_supported=True,
        preserves_policy_and_metric=True,source_connection_supported=True,
        why='Fixture explicitly indexes a separate initial bank.',citations=proposal['source_refs'])
    return repo,tmp_path/'out',proposal,review


def test_reviewed_reservation_restores_without_changing_development_or_primary_metric(tmp_path):
    repo,output,proposal,review = inputs(tmp_path)
    row = reservation.record(output,repo,'derived',proposal,review)
    declared = dict(benchmark='Toy',research_goal=dict(primary_metric={'name':'success'}))
    applied = reservation.apply(output,repo,'derived',declared)
    assert applied['research_goal']['confirmation'] == {'seed':101}
    assert applied['research_goal']['primary_metric'] == {'name':'success'}
    assert 'confirmation' not in declared['research_goal']
    assert (output/'confirmation_reservations'/(row['sha256']+'.json')).exists()


@pytest.mark.parametrize('boundary',['session','protocol','measurement','confirmation'])
def test_cannot_retrofit_after_search(tmp_path,boundary):
    repo,output,proposal,review = inputs(tmp_path)
    ref = {'session':'controller_session.json','protocol':'comparison_protocol.json',
        'measurement':'measurements/baseline.json','confirmation':'confirmation_attempts/a.json'}[boundary]
    atomic_json(output/'research/derived'/ref,{})
    with pytest.raises(ValueError,match='precede'):
        reservation.record(output,repo,'derived',proposal,review)


@pytest.mark.parametrize('change',['unsupported','wrong_quote','fake_citation','source_changed','tamper','symlink','conflict'])
def test_no_heuristic_seed_approval_or_silent_stale_source_reuse(tmp_path,change):
    repo,output,proposal,review = inputs(tmp_path)
    if change=='unsupported': review['disjoint_conditions_supported']=False
    elif change=='wrong_quote': proposal['source_refs'][0]['quote']='missing line'
    elif change=='fake_citation': review['citations']=[dict(ref='wrong.py',quote='fake')]
    if change in {'unsupported','wrong_quote','fake_citation'}:
        with pytest.raises(ValueError): reservation.record(output,repo,'derived',proposal,review)
        return
    reservation.record(output,repo,'derived',proposal,review)
    declared = {'benchmark':'Toy'}
    if change=='source_changed': (repo/'evaluate.py').write_text('def reset(seed): return one_constant_state')
    elif change=='tamper':
        row=read_json(output/'confirmation_reservation.json');row['confirmation']={'seed':0}
        atomic_json(output/'confirmation_reservation.json',row)
    elif change=='symlink':
        path=output/'confirmation_reservation.json';path.unlink();path.symlink_to(repo/'evaluate.py')
    else: declared['research_goal']={'confirmation':{'seed':7}}
    with pytest.raises((ValueError,OSError)):
        reservation.apply(output,repo,'derived',declared)


def test_main_can_repair_missing_reservation_without_resurvey_or_native_work(tmp_path):
    real = preparation(tmp_path)
    source = tmp_path/'evaluate.py';source.write_text('return initial_bank[seed]\n')
    citations=[dict(ref='evaluate.py',quote='return initial_bank[seed]')]
    answer=dict(native_evaluation_input_supported=True,disjoint_conditions_supported=True,
        preserves_policy_and_metric=True,source_connection_supported=True,citations=citations)
    real.client.choices=[answer]
    real.declaration={'benchmark':'Toy','research_goal':{'primary_metric':{'name':'success'}}}
    settings=dict(real.base_settings)
    result=real._step_configure_confirmation_protocol(confirmation={'seed':101},source_refs=citations,reason='source-reviewed split')
    assert result['outcome']=='registered'
    assert real.declaration['research_goal']['confirmation']=={'seed':101}
    assert real.base_settings==settings
    assert len(real.client.seen)==1
    assert not list((real.output/'research/derived').glob('measurements/*.json'))


def test_failed_review_returns_objective_handoff(tmp_path):
    real=preparation(tmp_path);(tmp_path/'evaluate.py').write_text('return constant_state\n')
    real.client.choices=[{'native_evaluation_input_supported':True,'disjoint_conditions_supported':False,
                         'citations':[dict(ref='evaluate.py',quote='return constant_state')]}]
    result=real._step_configure_confirmation_protocol(confirmation={'seed':101},
        source_refs=[dict(ref='evaluate.py',quote='return constant_state')],reason='test')
    assert result['outcome']=='not attempted' and result['repair_owner']=='objective'
    assert not (real.output/'confirmation_reservation.json').exists()


def test_missing_reservation_does_not_justify_resurvey(tmp_path):
    real=preparation(tmp_path)
    real.declaration={'benchmark':'Toy'};real.base_settings['_confirm']=True
    real.steps=[dict(step='declare',outcome='done'),dict(step='run_the_loop',outcome='not attempted')]
    assert not real._declaration_refresh_is_justified(research_progress={})
