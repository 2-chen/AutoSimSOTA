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


def test_intent_view_survives_stale_source_without_approving_it(tmp_path):
    repo, output, proposal, review = inputs(tmp_path)
    row = reservation.record(output, repo, 'derived', proposal, review)
    original = (output / 'confirmation_reservation.json').read_bytes()
    before = reservation.view(output, repo, 'derived')
    assert before['status'] == 'registered'
    assert before['confirmation'] == proposal['confirmation']
    assert before['source_refs'] == row['source_refs']
    (repo / 'evaluate.py').write_text('def reset(seed): return one_constant_state\n')
    held = reservation.view(output, repo, 'derived')
    assert held['status'] == 'sources_require_refresh'
    assert held['confirmation'] == proposal['confirmation']
    assert held['protocol_keys'] == proposal['protocol_keys']
    assert held['source_refs'] == row['source_refs']
    assert held['source_errors'][0]['index'] == 0
    assert held['presearch_revision_allowed'] is True
    with pytest.raises(ValueError):
        reservation.apply(output, repo, 'derived', {})
    assert (output / 'confirmation_reservation.json').read_bytes() == original
    atomic_json(output / 'research/derived/controller_session.json', {})
    assert reservation.view(output, repo, 'derived')['presearch_revision_allowed'] is False


@pytest.mark.parametrize('fault', ['pointer', 'archive', 'linked_pointer', 'wrong_owner'])
def test_intent_view_does_not_project_unauthenticated_parameters(tmp_path, fault):
    repo, output, proposal, review = inputs(tmp_path)
    row = reservation.record(output, repo, 'derived', proposal, review)
    run_id = 'derived'
    if fault == 'pointer':
        changed = read_json(output / 'confirmation_reservation.json')
        changed['confirmation'] = {'seed': 7}
        atomic_json(output / 'confirmation_reservation.json', changed)
    elif fault == 'archive':
        atomic_json(output / 'confirmation_reservations' / (row['sha256'] + '.json'), {})
    elif fault == 'linked_pointer':
        p = output / 'confirmation_reservation.json'
        p.unlink()
        p.symlink_to(output / 'confirmation_reservations' / (row['sha256'] + '.json'))
    else:
        run_id = 'other'
    held = reservation.view(output, repo, run_id)
    assert held['status'] == 'invalid'
    assert 'confirmation' not in held and 'source_refs' not in held


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


def test_reviewed_presearch_replacement_commits_memory_and_pointer_together(tmp_path):
    real=preparation(tmp_path)
    (tmp_path/'evaluate.py').write_text('return initial_bank[seed]\n')
    citations=[dict(ref='evaluate.py',quote='return initial_bank[seed]')]
    answer=dict(native_evaluation_input_supported=True,disjoint_conditions_supported=True,
        preserves_policy_and_metric=True,source_connection_supported=True,citations=citations)
    real.client.choices=[dict(answer),dict(answer)]
    original={'benchmark':'Toy','research_goal':{'confirmation':{'seed':7},'primary_metric':{'name':'success'}}}
    real.declaration=json.loads(json.dumps(original))
    first=real._step_configure_confirmation_protocol(confirmation={'seed':7},source_refs=citations,reason='initial')
    assert first['outcome']=='registered'
    before=(real.output/'confirmation_reservation.json').read_bytes()
    second=real._step_configure_confirmation_protocol(confirmation={'seed':101},source_refs=citations,reason='reviewed presearch revision')
    assert second['outcome']=='registered'
    assert real.confirmation_reservation_error==''
    assert real.declaration['research_goal']['confirmation']=={'seed':101}
    row=read_json(real.output/'confirmation_reservation.json')
    assert row['replaces_confirmation']=={'seed':7}
    assert (real.output/'confirmation_reservation.json').read_bytes()!=before
    assert len(list((real.output/'confirmation_reservations').glob('*.json')))==2
    # Cold resume from the prior declared source must recover the SAME approved
    # effective intent, not resurrect the old overlay or require another study.
    restored=reservation.apply(real.output,tmp_path,real.run_id,original)
    assert restored['research_goal']==real.declaration['research_goal']
    assert original['research_goal']['confirmation']=={'seed':7}
    with pytest.raises(ValueError,match='disagrees'):
        reservation.apply(real.output,tmp_path,real.run_id,
                          {'research_goal':{'confirmation':{'seed':9}}})


def test_failed_replacement_does_not_change_existing_intent(tmp_path):
    real=preparation(tmp_path)
    (tmp_path/'evaluate.py').write_text('return initial_bank[seed]\n')
    citations=[dict(ref='evaluate.py',quote='return initial_bank[seed]')]
    answer=dict(native_evaluation_input_supported=True,disjoint_conditions_supported=True,
        preserves_policy_and_metric=True,source_connection_supported=True,citations=citations)
    real.client.choices=[answer,{**answer,'disjoint_conditions_supported':False}]
    real.declaration={'research_goal':{'primary_metric':{'name':'success'}}}
    assert real._step_configure_confirmation_protocol(confirmation={'seed':7},source_refs=citations,reason='initial')['outcome']=='registered'
    before=(real.output/'confirmation_reservation.json').read_bytes()
    declaration=json.loads(json.dumps(real.declaration))
    result=real._step_configure_confirmation_protocol(confirmation={'seed':101},source_refs=citations,reason='rejected')
    assert result['outcome']=='not attempted'
    assert real.declaration==declaration
    assert (real.output/'confirmation_reservation.json').read_bytes()==before


def test_native_names_need_protocol_admission_before_paid_review(tmp_path):
    real=preparation(tmp_path)
    real.declaration={'research_goal':{}}
    result=real._step_configure_confirmation_protocol(confirmation={'native_partition':'unseen'},
        source_refs=[{'ref':'not_needed.py','quote':'not read'}],reason='native key')
    assert result['outcome']=='not attempted'
    assert 'protocol_keys' in result['because']
    assert not real.client.seen
    assert not (real.output/'confirmation_reservation.json').exists()
    reservation.validate({'native_partition':'unseen'},'declared custom input',['native_partition'],require_protocol_keys=True)


def test_later_search_cannot_reconfigure_reviewed_replacement(tmp_path):
    repo,output,proposal,review=inputs(tmp_path)
    proposal['replaces_confirmation']={'seed':7}
    reservation.record(output,repo,'derived',proposal,review)
    before=(output/'confirmation_reservation.json').read_bytes()
    atomic_json(output/'research/derived/controller_session.json',{'status':'paused'})
    with pytest.raises(ValueError,match='precede'):
        reservation.record(output,repo,'derived',{**proposal,'confirmation':{'seed':999}},review)
    assert (output/'confirmation_reservation.json').read_bytes()==before
    assert reservation.apply(output,repo,'derived',{'research_goal':{'confirmation':{'seed':7}}})['research_goal']['confirmation']=={'seed':101}


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


@pytest.mark.parametrize('fault', [None, 'quoted_change', 'protected_change', 'archive',
                                   'snapshot', 'base', 'protocol', 'presearch'])
def test_restore_frozen_reservation_intent_after_legal_policy_change(tmp_path, fault):
    from autosim.research.checkout_source_citations import capture, resolve
    from autosim.research.workspace_snapshot import create
    from autosim.research.common import digest
    source = tmp_path / 'original'; source.mkdir()
    (source / 'deploy.yml').write_text('seed: 0\naction_steps: 8\n')
    (source / 'evaluate.py').write_text('return initial_bank[seed]\n')
    output = tmp_path / 'run'; repo = output / 'checkout'
    create(source, repo)
    captured = capture(output, repo, {'ref': 'deploy.yml:1'})
    refs = [{'ref': captured['ref'], 'quote': captured['quote']}]
    proposal = dict(confirmation={'seed': 101}, protocol_keys=[], reason='fixture independent initial bank', source_refs=refs)
    review = dict(native_evaluation_input_supported=True, disjoint_conditions_supported=True,
                  preserves_policy_and_metric=True, source_connection_supported=True, citations=refs)
    row = reservation.record(output, repo, 'derived', proposal, review)
    root = output / 'research/derived'
    if fault != 'presearch':
        atomic_json(root / 'comparison_protocol.json', {'confirmation': {'seed': 101}})
        atomic_json(root / 'protocol_frozen.json', {'hashes': {str(repo / 'evaluate.py'): digest(repo / 'evaluate.py')}})
    (repo / 'deploy.yml').write_text('seed: 0\naction_steps: 16\n')
    if fault == 'quoted_change':
        (repo / 'deploy.yml').write_text('seed: 2\naction_steps: 16\n')
    elif fault == 'protected_change':
        (repo / 'evaluate.py').write_text('return constant_state\n')
    elif fault == 'archive':
        atomic_json(output / 'confirmation_reservations' / (row['sha256'] + '.json'), {})
    elif fault == 'snapshot':
        (output / 'checkout_source_citations' / (captured['ref'].split(':')[1] + '.source')).write_text('seed: 2\n')
    elif fault == 'base':
        manifest = read_json(output / 'workspace_snapshot.json'); manifest['source_tree_fingerprint'] = 'a' * 64
        atomic_json(output / 'workspace_snapshot.json', manifest)
    elif fault == 'protocol':
        atomic_json(root / 'comparison_protocol.json', {'confirmation': {'seed': 999}})
    original = (output / 'confirmation_reservation.json').read_bytes()
    # New proposals still cannot cite a changed full source file.
    with pytest.raises(ValueError):
        resolve(output, repo, captured['ref'], captured['quote'])
    if fault:
        with pytest.raises(ValueError):
            reservation.apply(output, repo, 'derived', {'benchmark': 'fixture'})
    else:
        restored = reservation.apply(output, repo, 'derived', {'benchmark': 'fixture'})
        assert restored['research_goal']['confirmation'] == {'seed': 101}
    assert (output / 'confirmation_reservation.json').read_bytes() == original
