import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from autosim.research import candidate_execution_review as admission
from autosim.research.common import atomic_json, read_json, digest
from autosim.research.ideas import Idea


@pytest.mark.parametrize('initializer', ['export', 'symlink', 'too_large'])
def test_candidate_python_package_glue_is_bounded_source_not_an_inferred_edge(tmp_path, initializer):
    repo = tmp_path / 'repo'
    package = repo / 'policies' / 'adapter'
    package.mkdir(parents=True)
    (package / 'deploy.py').write_text('def act():\n    return 1\n')
    init = package / '__init__.py'
    if initializer == 'symlink':
        outside = tmp_path / 'not-public.txt'
        outside.write_text('private material must not be read')
        init.symlink_to(outside)
    else:
        init.write_text('from .deploy import act\n' if initializer == 'export' else 'x'*65537)
    idea = Idea(label='change adapter', granularity='code', mechanism='change policy act',
        change={'file': 'policies/adapter/deploy.py', 'find': 'return 1', 'replace': 'return 2'})
    packet = admission.material(repo, {'evaluate': 'def evaluate(i):\n    return ["python", "eval.py"]'}, idea)
    rows = [row for row in packet['sources'] if row.get('path') == 'policies/adapter/__init__.py']
    if initializer == 'export':
        assert len(rows) == 1 and rows[0]['text'] == 'from .deploy import act\n'
        assert rows[0]['complete'] and rows[0]['sha256'] == digest(init)
        assert any(ref['source_id'] == rows[0]['id'] and
                   'hint' in ref['authority'] for ref in packet['native_references'])
    else:
        assert rows == [] and 'private material' not in json.dumps(packet)


def fixture(tmp_path):
    repo=tmp_path/'repo'
    repo.mkdir()
    (repo/'unused.py').write_text('def load():\n    return 1\n')
    (repo/'primary.py').write_text('def train():\n    return "primary"\n')
    idea=Idea(label='unused adapter control',granularity='code',risk='low',
              mechanism='diagnose the alternate adapter',why='no training intended',
              change={'file':'unused.py','find':'return 1','replace':'return 2'})
    stages={'train':'def stage(i):\n    return ["python", "primary.py"]\n'}
    return repo,idea,stages


def test_prospective_source_is_exact_readonly_and_not_prior_candidate_inheritance(tmp_path):
    import hashlib
    repo,idea,stages=fixture(tmp_path)
    path=repo/'unused.py'
    before='ROUTING = False\nFRAMING = False\n'
    path.write_text(before)
    def view(find,replace):
        idea.change={'file':'unused.py','find':find,'replace':replace}
        packet=admission.material(repo,stages,idea)
        return packet,next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    _,first=view('ROUTING = False','ROUTING = True')
    packet,second=view('FRAMING = False','FRAMING = True')
    assert first['text']=='ROUTING = True\nFRAMING = False\n'
    assert second['text']=='ROUTING = False\nFRAMING = True\n'
    assert second['complete'] and second['source_phase']=='prospective_postimage_not_executed'
    assert second['base_sha256']==digest(path)
    assert second['sha256']==hashlib.sha256(second['text'].encode()).hexdigest()
    assert packet['candidate_base_contract']['prior_candidate_effects_assumed'] is False
    _,composed=view(before,'ROUTING = True\nFRAMING = True\n')
    assert composed['text']=='ROUTING = True\nFRAMING = True\n'
    assert path.read_text()==before


def test_prospective_source_does_not_execute_candidate_code(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    marker=tmp_path/'DO_NOT_CREATE'
    idea.change['replace']=f"return __import__('pathlib').Path({str(marker)!r}).touch()"
    before=(repo/'unused.py').read_bytes()
    packet=admission.material(repo,stages,idea)
    after=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert 'touch()' in after['text'] and not marker.exists()
    assert (repo/'unused.py').read_bytes()==before


def test_candidate_review_retains_authenticated_negative_history_without_adopting_it(tmp_path,monkeypatch):
    from autosim.research import development_evidence
    from autosim.research.common import object_digest
    repo,idea,stages=fixture(tmp_path)
    selected_root=tmp_path/'research/custom_run'
    failure={'label':'round_2','status':'sealed_reader_events_invalid',
             'evidence_id':'a'*32,'log_sha256':'b'*64,'why':'Extra data: column 300'}
    calls=[]
    def history(root):
        calls.append(root)
        return [failure]
    monkeypatch.setattr(development_evidence,'reader_failure_diagnostics',history)
    packet=admission.packet_for(tmp_path,repo,stages,idea,research_root=selected_root)
    source=next(s for s in packet['sources'] if s['id']=='historical_reader_failure_0')
    assert calls==[selected_root] and 'Extra data: column 300' in source['text']
    assert 'not a verdict about this candidate' in source['authority']
    assert source['id'] not in packet['required_native_source_ids']
    assert 'metric_value' not in source['text'] and 'consumption' not in failure
    failure['why']='a different authenticated error'
    changed=admission.packet_for(tmp_path,repo,stages,idea,research_root=selected_root)
    assert object_digest(packet)!=object_digest(changed)


def test_candidate_review_cannot_read_another_run_outputs_history(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    with pytest.raises(ValueError,match='this run output'):
        admission.packet_for(tmp_path,repo,stages,idea,
                             research_root=tmp_path/'another_output/research/derived')


def test_prospective_multiple_hunks_share_one_current_file_and_cannot_guess(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    idea.change={'patches':[{'file':'unused.py','find':'def load():','replace':'def changed():'},
                           {'file':'unused.py','find':'return 1','replace':'return 2'}]}
    packet=admission.material(repo,stages,idea)
    after=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert after['text']=='def changed():\n    return 2\n'
    idea.change['patches'][1]['find']='missing source'
    packet=admission.material(repo,stages,idea)
    after=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert after['source_phase']=='prospective_postimage_unavailable'
    assert 'sha256' not in after and after['complete'] is False
    assert (repo/'unused.py').read_text()=='def load():\n    return 1\n'


def test_prospective_large_utf8_view_is_bounded_and_never_claims_completeness(tmp_path):
    import hashlib
    repo,idea,stages=fixture(tmp_path)
    before='# 中文注释\n'*12000+'def load():\n    return 1\n'
    (repo/'unused.py').write_text(before)
    packet=admission.material(repo,stages,idea)
    after=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert after['complete'] is False and len(after['text'].encode())<=12000
    assert 'return 2' in after['text']
    assert after['sha256']==hashlib.sha256(before.replace('return 1','return 2',1).encode()).hexdigest()
    assert len(json.dumps(packet,ensure_ascii=False).encode())<=192*1024


def test_prospective_source_matches_executor_newline_handling_without_editing(tmp_path):
    import hashlib
    repo,idea,stages=fixture(tmp_path)
    path=repo/'unused.py';before=b'def load():\r\n    return 1\r\n';path.write_bytes(before)
    packet=admission.material(repo,stages,idea)
    after=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert after['text']=='def load():\n    return 2\n'
    assert after['base_sha256']==hashlib.sha256(before).hexdigest()
    assert path.read_bytes()==before


def test_full_postimage_is_not_artificially_clipped_when_original_packet_ceiling_allows(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    before=('# leading context\n'*700+'FIRST = False\n'+
            '# middle context\n'*1500+'LAST = False\n'+'# trailing context\n'*700)
    path=repo/'unused.py';path.write_text(before)
    idea.change={'patches':[
        {'file':'unused.py','find':'FIRST = False','replace':'FIRST = True  # first update witness'},
        {'file':'unused.py','find':'LAST = False','replace':'LAST = True  # final routing repair'}]}
    packet=admission.material(repo,stages,idea)
    row=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    expected=before.replace('FIRST = False','FIRST = True  # first update witness').replace(
        'LAST = False','LAST = True  # final routing repair')
    assert row['complete'] is True and row['text']==expected
    assert row['source_phase']=='prospective_postimage_not_executed'
    assert len(json.dumps(packet,ensure_ascii=False).encode())<=192*1024
    assert path.read_text()==before
    assert any(s['id']=='workflow_train' for s in packet['sources'])


def test_full_delivery_uses_actual_json_bytes_and_preserves_original_ceiling_on_escape_heavy_source(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    before='# '+('\\'*60000)+'\ndef load():\n    return 1\n'
    path=repo/'unused.py';path.write_text(before)
    packet=admission.material(repo,stages,idea)
    row=next(s for s in packet['sources'] if s['id']=='candidate_effective_source_0')
    assert row['complete'] is False
    assert len(json.dumps(packet,ensure_ascii=False).encode())<=192*1024
    assert any(s['id']=='candidate_source_0' for s in packet['sources'])
    assert any(s['id']=='workflow_train' for s in packet['sources'])
    assert path.read_text()==before


def test_bounded_postimage_shows_entire_changes_before_unrelated_file_edges():
    before='# leading\n'*2000+'FIRST = 0\n'+'# middle\n'*3000+'LAST = 0\n'+'# trailing\n'*2000
    replacement='def batch_writer():\n'+''.join(f'    field_{n} = {n}\n' for n in range(250))
    patches=[{'find':'FIRST = 0','replace':replacement},
             {'find':'LAST = 0','replace':'LAST = 1  # install after real update'}]
    row=admission._prospective_source({'id':'candidate_source_0','path':'x.py','sha256':'base'},
        before,patches,limit=12000)
    assert row['complete'] is False and row['changed_hunks_complete'] is True
    assert replacement in row['text'] and patches[1]['replace'] in row['text']
    assert len(row['text'].encode())<=12000


def test_hunk_coordinates_follow_later_edits_before_and_inside_earlier_hunks():
    before='HEAD = 0\n'+'# unchanged\n'*3000+'BODY = 0\n'+'# tail\n'*2000
    patches=[{'find':'BODY = 0','replace':'def body():\n    inner = 0\n    return inner'},
             {'find':'HEAD = 0','replace':'HEAD = 1\n'+'# inserted prefix\n'*100},
             {'find':'inner = 0','replace':'inner = 1  # composed repair'}]
    row=admission._prospective_source({'id':'candidate_source_0','path':'x.py','sha256':'base'},
        before,patches,limit=12000)
    assert row['complete'] is False and row['changed_hunks_complete'] is True
    assert 'HEAD = 1' in row['text'] and 'inner = 1  # composed repair' in row['text']
    assert 'inner = 0' not in row['text']


def test_oversize_changed_hunk_is_honestly_partial_and_byte_bounded():
    before='FIRST = 0\n'
    row=admission._prospective_source({'id':'candidate_source_0','path':'x.py','sha256':'base'},
        before,[{'find':'FIRST = 0','replace':'# 中文注释\n'*3000}],limit=1024)
    assert row['complete'] is False and row['changed_hunks_complete'] is False
    assert len(row['text'].encode())<=1024 and 'omitted' in row['text']


def test_prompts_explain_current_preimage_and_independent_corrective_candidates():
    from autosim.research.prompt_library import load_prompt
    assert 'not automatically' in ' '.join(load_prompt('fix').split())
    assert 'not inherited' in load_prompt('candidate-execution-review') or 'not implicitly inherited' in load_prompt('candidate-execution-review')
    assert 'Failed/declined experiments may' in load_prompt('scheduler')


def test_quote_boundary_normalization_is_exact_readonly_and_audited(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    packet=admission.packet_for(tmp_path,repo,stages,idea)
    answer={'verdict':'uncertain','reason':'need native evidence','source_evidence':[
        {'source_id':'candidate_source_0','quote':' \treturn 1 \r\n'}]}
    original=json.dumps(answer,sort_keys=True)
    normalized,changes=admission.normalize_citation_boundaries(answer,packet)
    admission.validate(normalized,packet)
    assert normalized['source_evidence'][0]['quote']=='return 1'
    assert json.dumps(answer,sort_keys=True)==original
    assert len(changes)==1 and changes[0]['kind']=='outer_ascii_whitespace_only'
    assert changes[0]['original_quote_sha256']!=changes[0]['exact_quote_sha256']


@pytest.mark.parametrize('verdict', ['applicable', 'inapplicable', 'uncertain'])
def test_source_id_citations_bind_whole_delivered_text_and_material_without_inventing_quotes(
        tmp_path, verdict):
    from copy import deepcopy
    repo, idea, stages = fixture(tmp_path)
    packet = admission.packet_for(tmp_path, repo, stages, idea)
    answer = {'verdict':verdict, 'reason':'source-backed routing analysis',
        'source_evidence':[{'source_id':s['id']} for s in packet['sources']
            if s['id'] in {'workflow_train','candidate_change'} or s['id'].startswith('native_source_')]}
    original = deepcopy(answer)
    bound, notes = admission.bind_offered_source_ids(answer, packet)
    admission.validate(bound, packet)
    assert answer == original and bound['reason'] == answer['reason']
    assert bound['verdict'] == verdict and notes
    assert all('quote' not in c and c['citation_kind'] == 'offered_source'
               for c in bound['source_evidence'])
    feedback = admission.citation_feedback(bound, packet)
    assert feedback['citation_errors'] == []


@pytest.mark.parametrize('fault', ['unknown_id', 'malformed_id', 'duplicate_id',
    'source_text', 'other_material', 'digest', 'mixed_quote', 'old_contract'])
def test_source_id_transport_never_accepts_unknown_ambiguous_or_changed_source_material(tmp_path, fault):
    from copy import deepcopy
    repo, idea, stages = fixture(tmp_path)
    packet = admission.packet_for(tmp_path, repo, stages, idea)
    answer = {'verdict':'uncertain', 'reason':'needs probe',
              'source_evidence':[{'source_id':'candidate_source_0'}]}
    if fault in {'unknown_id','malformed_id','duplicate_id','old_contract'}:
        if fault == 'unknown_id': answer['source_evidence'][0]['source_id'] = 'unoffered'
        elif fault == 'malformed_id': answer['source_evidence'][0]['source_id'] = []
        elif fault == 'duplicate_id':
            packet['sources'].append(deepcopy(next(s for s in packet['sources'] if s['id']=='candidate_source_0')))
        else: packet['review_contract_version'] = 11
        with pytest.raises(ValueError, match='source-ID'):
            admission.bind_offered_source_ids(answer, packet)
        return
    bound, _ = admission.bind_offered_source_ids(answer, packet)
    if fault == 'source_text':
        next(s for s in packet['sources'] if s['id']=='candidate_source_0')['text'] += '\n# changed'
    elif fault == 'other_material': packet['candidate']['label'] = 'different idea'
    elif fault == 'digest': bound['source_evidence'][0]['offered_text_sha256'] = '0'*64
    elif fault == 'mixed_quote': bound['source_evidence'][0]['quote'] = 'invented words'
    with pytest.raises(ValueError, match='exact material/text binding'):
        admission.validate(bound, packet)


def test_id_reference_still_requires_all_native_training_categories(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    packet = admission.packet_for(tmp_path, repo, stages, idea)
    answer = {'verdict':'applicable', 'reason':'insufficient categories',
              'source_evidence':[{'source_id':'candidate_source_0'}]}
    bound, _ = admission.bind_offered_source_ids(answer, packet)
    with pytest.raises(ValueError, match='both workflow and intervention'):
        admission.validate(bound, packet)


def test_real_review_id_citations_preserve_raw_reply_cache_and_verified_evidence(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    class SourceClient(Client):
        def chat_with_metadata(self, *args, **kwargs):
            text, metadata = super().chat_with_metadata(*args, **kwargs)
            answer = json.loads(text)
            answer['source_evidence'] = [{'source_id':c['source_id']} for c in answer['source_evidence']]
            return json.dumps(answer), metadata
    client = SourceClient('inapplicable')
    result = admission.review(tmp_path, repo, stages, idea, client)
    assert result['status']=='reviewed' and client.calls==1
    assert result['answer']['verdict']=='inapplicable' and result['responses'][0]['source_id_bindings']
    raw = json.loads(result['responses'][0]['text'])
    assert all(set(c)=={'source_id'} for c in raw['source_evidence'])
    assert admission.review(tmp_path, repo, stages, idea, client)==result and client.calls==1
    assert admission.read_review_evidence(tmp_path, result['evidence_id'])['status']=='reviewed'


def test_malformed_id_feedback_can_recover_once_without_unhashable_id_crash(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    class MalformedClient(Client):
        def chat_with_metadata(self, *args, **kwargs):
            text, metadata = super().chat_with_metadata(*args, **kwargs)
            answer = json.loads(text)
            if self.calls==1: answer['source_evidence']=[{'source_id':[]}]
            else: answer['source_evidence']=[{'source_id':c['source_id']} for c in answer['source_evidence']]
            return json.dumps(answer), metadata
    client = MalformedClient('applicable')
    result = admission.review(tmp_path, repo, stages, idea, client)
    assert client.calls==2 and result['status']=='reviewed'
    assert 'validation_error' in result['responses'][0]


@pytest.mark.parametrize('quote',['return 3 ','return\t1 ','ret urn 1 ',
    'def load():\nreturn 1','\u00a0return 1\u00a0','```return 1```',
    'return 1\n# invented text','  ',' '*900+'return 1'])
def test_quote_boundary_normalization_never_repairs_semantics_or_delivery_limits(tmp_path,quote):
    repo,idea,stages=fixture(tmp_path)
    packet=admission.packet_for(tmp_path,repo,stages,idea)
    answer={'verdict':'uncertain','reason':'need native evidence','source_evidence':[
        {'source_id':'candidate_source_0','quote':quote}]}
    normalized,changes=admission.normalize_citation_boundaries(answer,packet)
    assert normalized==answer and changes==[]
    with pytest.raises(ValueError,match='exact substring'):
        admission.validate(normalized,packet)


def test_review_keeps_raw_provider_reply_when_only_quote_boundaries_are_canonicalized(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    class BoundaryClient(Client):
        def chat_with_metadata(self,*args,**kwargs):
            content,metadata=super().chat_with_metadata(*args,**kwargs)
            response=json.loads(content)
            for citation in response['source_evidence']:
                citation['quote']+=' \t'
            return json.dumps(response),metadata
    row=admission.review(tmp_path,repo,stages,idea,BoundaryClient('applicable'))
    assert row['status']=='reviewed'
    raw=json.loads(row['responses'][0]['text'])
    assert row['responses'][0]['citation_boundary_normalizations']
    assert all(c['quote'].endswith(' \t') for c in raw['source_evidence'])
    assert all(not c['quote'].endswith(' \t') for c in row['answer']['source_evidence'])
    admission.validate(row['answer'],admission.packet_for(tmp_path,repo,stages,idea))


@pytest.mark.parametrize('identity',[{},[],None])
def test_quote_boundary_normalizer_does_not_crash_on_malformed_source_ids(tmp_path,identity):
    repo,idea,stages=fixture(tmp_path)
    packet=admission.packet_for(tmp_path,repo,stages,idea)
    answer={'verdict':'uncertain','reason':'need native evidence','source_evidence':[
        {'source_id':identity,'quote':' return 1 '}]}
    normalized,changes=admission.normalize_citation_boundaries(answer,packet)
    assert normalized==answer and changes==[]
    with pytest.raises(ValueError,match='exact substring'):
        admission.validate(normalized,packet)


def test_writer_repair_review_requests_worker_lifetime_and_framing_evidence_not_print_throttling():
    from autosim.research.prompt_library import load_prompt
    prompt=load_prompt('candidate-execution-review')
    assert 'constructor-time PID/instance ID' in prompt
    assert 'fork-aware/lazy worker identity' in prompt
    assert 'Throttling print frequency does not' in prompt
    assert 'multiworker CPU contract probe' in prompt
    assert 'not optimizer-completed training batches' in prompt


def test_review_receives_only_explicit_current_sealed_framework_contract(tmp_path):
    from autosim.research.harness_repair import read_source
    repo, idea, stages = fixture(tmp_path)
    contract = read_source(tmp_path, {'path':'research/data_versions.py',
                                     'query':'def loader_witnesses', 'limit':55})
    # Framework source IDs used to be silently discarded as non-native probes.
    identity = contract['evidence_id']
    assert admission.framework_observations(tmp_path, []) == []
    idea.evidence = ['evidence:'+identity]
    settings = {'dataset_root':'data-version:official_v1', 'steps':3}
    packet = admission.packet_for(tmp_path, repo, stages, idea,
                                 selected_training_settings=settings)
    rows = [s for s in packet['sources'] if s['id'].startswith('framework_contract_')]
    assert len(rows) == 1 and rows[0]['evidence_id'] == identity
    assert 'loader witness setting disagrees' in rows[0]['text']
    assert 'not native consumption' in rows[0]['authority']
    assert packet['selected_training_settings'] == settings
    selected = next(s for s in packet['sources'] if s['id']=='selected_training_inputs')
    assert '"setting": "dataset_root"' in selected['text']
    assert '"data_version_id": "official_v1"' in selected['text']
    assert list(packet).index('selected_training_settings') < list(packet).index('sources')
    assert 'not native execution' in selected['authority']
    assert rows[0]['id'] not in packet['required_native_source_ids']
    assert str(tmp_path) not in json.dumps(packet)
    assert admission.framework_observations(tmp_path/'another_run',[identity]) == []
    # Changing requested inputs changes admission identity; no prior verdict is reused.
    other = admission.packet_for(tmp_path, repo, stages, idea,
        selected_training_settings={**settings,'dataset_root':'data-version:other_v1'})
    from autosim.research.common import object_digest
    assert object_digest(packet) != object_digest(other)


def test_fourth_explicit_framework_source_is_not_silently_lost(tmp_path):
    from autosim.research.harness_repair import read_source
    repo,idea,stages = fixture(tmp_path)
    requests = [('research/derived_research.py','AUTOSIM_TRAIN_SELECTION_JSON'),
        ('research/declarative_backend.py','AUTOSIM_TRAIN_SELECTION_JSON'),
        ('research/derived_research.py','native_environment=surroundings'),
        ('research/data_versions.py','def native_selection_payload')]
    ids = [read_source(tmp_path,{'path':path,'query':query,'limit':20})['evidence_id']
           for path,query in requests]
    idea.evidence = ['evidence:'+ident for ident in ids]
    packet = admission.packet_for(tmp_path,repo,stages,idea,
        selected_training_settings={'dataset':'data-version:registered','steps':3})
    offered = [s for s in packet['sources'] if s['id'].startswith('framework_contract_')]
    assert [s['evidence_id'] for s in offered] == ids
    assert 'def native_selection_payload' in offered[-1]['text']
    assert all('not native consumption' in s['authority'] for s in offered)


def test_current_decision_can_supply_new_evidence_without_rewriting_immutable_idea(tmp_path):
    from autosim.research.harness_repair import read_source
    from autosim.research.common import object_digest
    repo,idea,stages=fixture(tmp_path)
    old=list(idea.evidence)
    original=admission.packet_for(tmp_path,repo,stages,idea)
    contract=read_source(tmp_path,{'path':'research/derived_research.py',
        'query':'AUTOSIM_TRAIN_SELECTION_JSON','limit':20})
    current=admission.packet_for(tmp_path,repo,stages,idea,
        additional_evidence_refs=['evidence:'+contract['evidence_id']])
    rows=[s for s in current['sources'] if s['id'].startswith('framework_contract_')]
    assert len(rows)==1 and rows[0]['evidence_id']==contract['evidence_id']
    assert object_digest(current)!=object_digest(original)
    assert idea.evidence==old
    # An invented assertion/unknown ID changes neither source evidence nor cache.
    unknown=admission.packet_for(tmp_path,repo,stages,idea,
        additional_evidence_refs=['the exporter is verified', 'evidence:'+'f'*32])
    assert unknown==original


def test_execution_boundary_forwards_only_main_selected_current_evidence_and_review_identity(tmp_path,monkeypatch):
    from autosim.research.prepare import Preparation
    repo,idea,stages=fixture(tmp_path)
    refs=['evidence:'+'a'*32]
    real=SimpleNamespace(output=tmp_path,repo=repo,stages=stages,
        client=SimpleNamespace(supports_candidate_execution_review=True),
        current_action={'research_decision':{'input_evidence':refs,
            'runtime_evidence_refs':['agent/processes/not-a-source.json']}})
    research=SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    seen={}
    def checking(*args,**kwargs):
        seen.update(kwargs)
        return {'answer':{'verdict':'uncertain','reason':'source still incomplete'},
            'material_sha256':'b'*64,'record_ref':'candidate_execution_reviews/'+'b'*64+'.json',
            'evidence_id':'execution-review:'+'b'*64}
    monkeypatch.setattr(admission,'review',checking)
    result=Preparation._review_candidate_execution(real,research,idea.label)
    assert seen['additional_evidence_refs']==refs
    assert result['evidence_id']=='execution-review:'+'b'*64
    assert result['independent_review']['role']=='supervisor'
    assert result['independent_review']['verdict']=='uncertain'


def test_requested_input_citation_is_not_native_proof_or_a_workflow_citation(tmp_path):
    repo,idea,stages = fixture(tmp_path)
    packet = admission.packet_for(tmp_path,repo,stages,idea,
        selected_training_settings={'dataset_root':'data-version:official_v1','steps':3})
    selected = next(s for s in packet['sources'] if s['id']=='selected_training_inputs')
    change = next(s for s in packet['sources'] if s['id']=='candidate_change')
    answer = {'verdict':'applicable','reason':'Requested keys alone do not prove a producer connection.',
        'source_evidence':[{'source_id':'selected_training_inputs','quote':'"setting": "dataset_root"'},
                           {'source_id':'candidate_change','quote':change['text']}]}
    with pytest.raises(ValueError,match='workflow and intervention'):
        admission.validate(answer,packet)
    assert 'not proof of training' in selected['text']


@pytest.mark.parametrize('corruption', ['log','source_hash','excerpt','bounds','record_symlink'])
def test_framework_contract_rejects_changed_or_unsafe_evidence(tmp_path, corruption):
    from autosim.research.harness_repair import read_source
    record = read_source(tmp_path, {'path':'research/data_versions.py','limit':20})
    identity = record['evidence_id']
    assert len(admission.framework_observations(tmp_path,[identity])) == 1
    path = tmp_path/'framework_source_reads'/f'{identity}.json'
    if corruption == 'log':
        (tmp_path/record['log_ref']).write_text('invented contract')
    elif corruption == 'record_symlink':
        moved = tmp_path/'other.json'
        path.rename(moved)
        path.symlink_to(moved)
    else:
        record = read_json(path)
        if corruption == 'source_hash': record['source_sha256'] = '0'*64
        elif corruption == 'excerpt': record['excerpt'] = 'invented contract'
        else: record['next_offset'] = 999999
        atomic_json(path,record)
    assert admission.framework_observations(tmp_path,[identity]) == []


@pytest.mark.parametrize('relative',['research/data_campaign.py',
    'research/candidate_execution_review.py','research/prompt_library.py'])
def test_framework_contract_excerpt_does_not_replace_native_routing_or_grant_edits(tmp_path,relative):
    from autosim.research import harness_repair
    record = harness_repair.read_source(tmp_path, {'path':relative,
                                                  'query':'def ', 'limit':15})
    assert record['candidate_edits_allowed'] is False
    assert record['live_activation_allowed'] is False
    repo, idea, stages = fixture(tmp_path)
    idea.evidence = ['evidence:'+record['evidence_id']]
    packet = admission.packet_for(tmp_path,repo,stages,idea)
    source = next(s for s in packet['sources'] if s['id'].startswith('framework_contract_'))
    change = next(s for s in packet['sources'] if s['id']=='candidate_change')
    answer = {'verdict':'applicable','reason':'A contract alone does not prove native routing.',
        'source_evidence':[
            {'source_id':'workflow_train','quote':stages['train']},
            {'source_id':'candidate_change','quote':change['text']},
            {'source_id':source['id'],'quote':source['text'][:200]}]}
    with pytest.raises(ValueError,match='actual offered native source'):
        admission.validate(answer,packet)


def test_preparation_forwards_requested_training_inputs_to_independent_review(tmp_path,monkeypatch):
    from autosim.research.prepare import Preparation
    repo,idea,stages = fixture(tmp_path)
    client = Client('applicable')
    real = Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=client)
    real.stages = stages
    research = SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    original = admission.review
    seen = []
    def check(*args,**kwargs):
        seen.append(kwargs.get('selected_training_settings'))
        return original(*args,**kwargs)
    monkeypatch.setattr(admission,'review',check)
    settings = {'dataset_root':'data-version:official_v1','steps':3}
    assert real._review_candidate_execution(research,idea.label,
        selected_training_settings=settings) is None
    assert seen == [settings]
    requests = list((real.output/'candidate_execution_reviews/requests').glob('*.json'))
    packet = read_json(requests[0])['payload']
    assert packet['selected_training_settings'] == settings
    assert not (real.output/'research/derived/measurements').exists()


@pytest.mark.parametrize('second_path', ['unused.py', './unused.py'])
def test_multi_hunk_same_file_has_one_citable_native_source(tmp_path, second_path):
    repo, idea, stages = fixture(tmp_path)
    stages = {'train':'def stage(i):\n    return ["python", "unused.py"]\n'}
    idea.change = {'patches':[
        {'file':'unused.py','find':'def load():','replace':'def load_v2():'},
        {'file':second_path,'find':'return 1','replace':'return 2'}]}
    packet = admission.material(repo, stages, idea)
    candidates = [s for s in packet['sources'] if s['id'].startswith('candidate_source_')]
    assert len(candidates) == 1 and candidates[0]['id'] == 'candidate_source_0'
    assert any(ref['source_id'] == 'candidate_source_0' for ref in packet['native_references'])
    change = next(s for s in packet['sources'] if s['id']=='candidate_change')
    assert 'load_v2' in change['text'] and 'return 2' in change['text']
    admission.validate({'verdict':'applicable','reason':'Both hunks affect the routed native producer.',
        'source_evidence':[
            {'source_id':'workflow_train','quote':stages['train']},
            {'source_id':'candidate_change','quote':change['text']},
            {'source_id':'candidate_source_0','quote':candidates[0]['text']} ]}, packet)
    public = admission.packet_for(tmp_path, repo, stages, idea)
    source = next(s for s in public['sources'] if s['id']=='candidate_source_0')
    assert source['path'] == '[LOCAL_PATH]' and source['repository_file'] == 'unused.py'
    assert next(r for r in public['native_references']
                if r['source_id']=='candidate_source_0')['repository_file'] == 'unused.py'
    assert str(tmp_path) not in json.dumps(public)


def test_citation_feedback_recommends_exact_matching_sources_but_does_not_accept_bad_id(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    packet = admission.material(repo, stages, idea)
    native = next(row for row in packet['sources'] if row['id'].startswith('native_source_'))
    answer = {'verdict': 'inapplicable', 'reason': 'The selected producer is different.',
        'source_evidence': [{'source_id': 'candidate_source_0', 'quote': native['text']}]}
    help = admission.citation_feedback(answer, packet)
    assert {'source_id': native['id'], 'quote': native['text']} in help['exact_quote_suggestions']
    assert {'workflow_train', 'candidate_change'} <= {s['source_id'] for s in help['exact_quote_suggestions']}
    assert any('workflow_' in error for error in help['citation_errors'])
    assert any('candidate_change' in error for error in help['citation_errors'])
    with pytest.raises(ValueError, match='exact substring'):
        admission.validate(answer, packet)


def test_citation_feedback_escape_help_is_advisory_and_ambiguous_matches_stay_explicit():
    packet = {'sources': [{'id': 'source_a', 'text': 'first\nsecond "literal"'},
                          {'id': 'source_b', 'text': 'first\nsecond "literal"'}]}
    answer = {'verdict': 'uncertain', 'source_evidence': [
        {'source_id': 'missing', 'quote': 'first\\nsecond \\"literal\\"'}]}
    help = admission.citation_feedback(answer, packet)
    assert {row['source_id'] for row in help['exact_quote_suggestions']} == {'source_a', 'source_b'}
    assert 'answer' not in help and 'verdict' not in help
    assert answer['source_evidence'][0]['source_id'] == 'missing'


def test_citation_feedback_is_bounded_and_does_not_invent_quotes():
    packet = {'sources': [{'id': f'source_{i}', 'text': 'literal source'} for i in range(20)]}
    help = admission.citation_feedback({'verdict': 'uncertain', 'source_evidence': [
        {'source_id': 'missing', 'quote': 'literal source'}]}, packet)
    assert len(help['exact_quote_suggestions']) == 8
    empty = admission.citation_feedback({'verdict': 'uncertain', 'source_evidence': [
        {'source_id': 'missing', 'quote': 'invented text'}]}, packet)
    assert empty['exact_quote_suggestions'] == []


def test_explicit_citation_choice_preserves_nested_bytes_and_raw_answer():
    from copy import deepcopy
    code = 'if key == "x":\n    work()'
    encoded = code
    for _ in range(2):
        encoded = json.dumps(encoded)[1:-1]
    packet = {'sources': [{'id': 'numbered_contract', 'text': '17: ' + encoded}]}
    bad = {'verdict': 'uncertain', 'reason': 'installed semantics unknown',
           'source_evidence': [{'source_id': 'numbered_contract', 'quote': code}]}
    feedback = admission.citation_feedback(bad, packet)
    choice = next(c for c in feedback['citation_choices'] if c['quote'] == encoded)
    answer = {**bad, 'source_evidence': [{'citation_choice': choice['citation_choice']}]}
    original = deepcopy(answer)
    resolved, notes = admission.expand_citation_choices(answer, packet, feedback['citation_choices'])
    admission.validate(resolved, packet)
    assert answer == original and bad['source_evidence'][0]['quote'] == code
    assert resolved['verdict'] == bad['verdict'] and resolved['reason'] == bad['reason']
    assert resolved['source_evidence'] == [{'source_id': 'numbered_contract', 'quote': encoded}]
    assert notes[0]['exact_quote_sha256'] == __import__('hashlib').sha256(encoded.encode()).hexdigest()
    assert len(feedback['citation_choices']) <= 8


@pytest.mark.parametrize('fault', ['unoffered', 'first_response', 'other_packet',
    'changed_quote', 'changed_source', 'mixed_fields', 'malformed_id'])
def test_citation_choices_cannot_invent_or_cross_packet_authority(fault):
    from copy import deepcopy
    packet = {'sources': [{'id': 'native', 'text': 'first\nsecond'}]}
    bad = {'verdict': 'uncertain', 'reason': 'need more evidence',
           'source_evidence': [{'source_id': 'native', 'quote': 'first\\nsecond'}]}
    choices = deepcopy(admission.citation_feedback(bad, packet)['citation_choices'])
    citation = {'citation_choice': choices[0]['citation_choice']}
    if fault == 'unoffered': citation['citation_choice'] = 'cq_' + 'f'*64
    elif fault == 'first_response': choices = []
    elif fault == 'other_packet': packet = {**packet, 'candidate': 'different proposal'}
    elif fault == 'changed_quote': choices[0]['quote'] = 'invented quote'
    elif fault == 'changed_source': choices[0]['source_id'] = 'missing'
    elif fault == 'mixed_fields': citation['quote'] = 'invented override'
    elif fault == 'malformed_id': citation['citation_choice'] = []
    with pytest.raises(ValueError, match='citation_choice'):
        admission.expand_citation_choices({**bad, 'source_evidence': [citation]}, packet, choices)


def test_choice_transport_does_not_satisfy_missing_scientific_source_categories():
    packet = {'sources': [{'id': 'optional', 'text': 'literal source'}]}
    bad = {'verdict': 'applicable', 'reason': 'needs required sources',
           'source_evidence': [{'source_id': 'missing', 'quote': 'literal source'}]}
    choices = admission.citation_feedback(bad, packet)['citation_choices']
    answer, _ = admission.expand_citation_choices({**bad, 'source_evidence': [
        {'citation_choice': choices[0]['citation_choice']}]}, packet, choices)
    with pytest.raises(ValueError, match='both workflow and intervention'):
        admission.validate(answer, packet)


def test_review_expands_only_explicit_second_reply_choices_and_seals_raw_response(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    class ChoiceClient(Client):
        def chat_with_metadata(self, system, user, **kwargs):
            packet = json.loads(user)
            text, metadata = super().chat_with_metadata(system, user, **kwargs)
            response = json.loads(text)
            if self.calls == 1:
                response['source_evidence'] = [
                    {'source_id': 'missing', 'quote': 'def train():'}]
            else:
                assert packet['citation_choices'] and 'format_feedback' in packet
                response['source_evidence'] = [
                    {'citation_choice': c['citation_choice']} for c in packet['citation_choices']]
            return json.dumps(response), metadata
    client = ChoiceClient('inapplicable')
    row = admission.review(tmp_path, repo, stages, idea, client)
    assert row['status'] == 'reviewed' and client.calls == 2
    assert row['answer']['verdict'] == 'inapplicable'
    assert row['responses'][1]['citation_choice_resolutions']
    raw = json.loads(row['responses'][1]['text'])
    assert all(set(c) == {'citation_choice'} for c in raw['source_evidence'])
    assert all(set(c) == {'source_id', 'quote'} for c in row['answer']['source_evidence'])
    admission.validate(row['answer'], admission.packet_for(tmp_path, repo, stages, idea))
    assert admission.review(tmp_path, repo, stages, idea, client) == row and client.calls == 2
    assert admission.read_review_evidence(tmp_path, row['evidence_id'])['status'] == 'reviewed'


@pytest.mark.parametrize('depth', [1, 2])
def test_citation_feedback_offers_encoded_native_code_without_rewriting_answer(depth):
    quote = 'if key.startswith("observation."):\n    values[key] = window'
    encoded = quote
    for _ in range(depth):
        encoded = json.dumps(encoded, ensure_ascii=False)[1:-1]
    packet = {'sources':[{'id':'diagnostic', 'text':'sealed prefix ' + encoded + ' suffix'}]}
    answer = {'verdict':'uncertain', 'reason':'source interpretation pending',
              'source_evidence':[{'source_id':'diagnostic', 'quote':quote}]}
    with pytest.raises(ValueError, match='exact substring'):
        admission.validate(answer, packet)
    feedback = admission.citation_feedback(answer, packet)
    assert {'source_id':'diagnostic', 'quote':encoded} in feedback['exact_quote_suggestions']
    assert answer['source_evidence'][0]['quote'] == quote
    assert 'verdict' not in feedback and 'answer' not in feedback
    corrected = {**answer, 'source_evidence':[{'source_id':'diagnostic', 'quote':encoded}]}
    admission.validate(corrected, packet)
    assert 'nested JSON' in feedback['citation_transport_hint']


def test_encoded_citation_suggestions_still_require_actual_offered_bytes():
    packet = {'sources':[{'id':'diagnostic', 'text':'unrelated sealed text'}]}
    feedback = admission.citation_feedback({'verdict':'uncertain', 'source_evidence':[
        {'source_id':'diagnostic','quote':'if key == "missing":\n    work()'}]}, packet)
    assert feedback['exact_quote_suggestions'] == []


def test_citation_feedback_corrects_leading_indent_only_via_exact_source_match():
    packet = {'sources':[{'id':'workflow_evaluate', 'text':
                         'def evaluator(inputs):\n    checkpoint = inputs["checkpoint"]'}]}
    answer = {'verdict':'uncertain', 'reason':'format correction pending', 'source_evidence':[
        {'source_id':'workflow_evaluate', 'quote':'        checkpoint = inputs["checkpoint"]'}]}
    with pytest.raises(ValueError, match='exact substring'):
        admission.validate(answer, packet)
    feedback = admission.citation_feedback(answer, packet)
    assert {'source_id':'workflow_evaluate', 'quote':'checkpoint = inputs["checkpoint"]'} \
        in feedback['exact_quote_suggestions']
    assert answer['source_evidence'][0]['quote'].startswith('        ')
    assert 'every listed invalid citation' in feedback['citation_transport_hint']


def test_training_feedback_names_native_ids_and_training_not_only_evaluation(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    stages['train'] = 'def train(i):\n    return ["python", "unused.py"]\n'
    stages['evaluate'] = 'def evaluate(i):\n    return ["python", "primary.py"]\n'
    packet = admission.material(repo, stages, idea)
    packet['execution_plan'] = {'stages':['train', 'evaluate'], 'will_train':True}
    packet['sources'].append({'id':'runtime_diagnostic_probe', 'text':'dataset = make_dataset(cfg)'})
    change = next(s for s in packet['sources'] if s['id']=='candidate_change')
    answer = {'verdict':'applicable', 'reason':'Native dataset consumes augmentation.',
        'source_evidence':[
            {'source_id':'candidate_change', 'quote':change['text']},
            {'source_id':'workflow_evaluate', 'quote':stages['evaluate']},
            {'source_id':'runtime_diagnostic_probe', 'quote':'dataset = make_dataset(cfg)'}]}
    with pytest.raises(ValueError, match='actual train workflow'):
        admission.validate(answer, packet)
    feedback = admission.citation_feedback(answer, packet)
    suggestions = {row['source_id']:row['quote'] for row in feedback['exact_quote_suggestions']}
    assert {'workflow_train', 'candidate_source_0'} <= suggestions.keys()
    assert 'candidate_source_0' in feedback['required_native_source_ids']
    assert any('supplemental' in e for e in feedback['citation_errors'])
    answer['source_evidence'].append({'source_id':'workflow_train', 'quote':stages['train']})
    with pytest.raises(ValueError, match='actual offered native source'):
        admission.validate(answer, packet)
    answer['source_evidence'].append({'source_id':'candidate_source_0',
                                     'quote':suggestions['candidate_source_0']})
    admission.validate(answer, packet)


class Client:
    calls=0
    role='scheduler'
    supports_candidate_execution_review=True
    def __init__(self,verdict='inapplicable'):
        self.verdict=verdict
    @contextmanager
    def as_role(self, role):
        previous=self.role
        self.role=role
        try: yield
        finally: self.role=previous
    def chat_with_metadata(self, system,user,**kwargs):
        assert self.role=='supervisor'
        assert kwargs['decision_only'] and kwargs['read_only'] and not kwargs['auto_skills']
        assert not kwargs['resume_session'] and not kwargs['include_research_context']
        self.calls+=1
        packet=json.loads(user)
        citations=[{'source_id':row['id'],'quote':row['text']} for row in packet['sources']
                   if row['id'] in {'workflow_train','candidate_change'} or row['id'].startswith('native_source_')]
        return json.dumps({'verdict':self.verdict,'reason':'source-backed mismatch',
                           'source_evidence':citations}),{}


def test_exact_source_review_is_cached_and_rehydrates_on_source_change(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    client=Client()
    first=admission.review(tmp_path,repo,stages,idea,client)
    assert first['answer']['verdict']=='inapplicable'
    assert client.role=='scheduler'
    assert admission.review(tmp_path,repo,stages,idea,client)==first and client.calls==1
    (repo/'unused.py').write_text('def load():\n    return 1\n# new evidence\n')
    second=admission.review(tmp_path,repo,stages,idea,client)
    assert second['material_sha256']!=first['material_sha256'] and client.calls==2


def test_feedback_contract_change_preserves_old_failure_and_bounds_new_retry(tmp_path, monkeypatch):
    repo, idea, stages = fixture(tmp_path)
    class UnavailableClient(Client):
        def chat_with_metadata(self, *args, **kwargs):
            self.calls += 1
            raise OSError('fixture unavailable')
    old = UnavailableClient()
    with pytest.raises(admission.ReviewUnavailable) as failure:
        admission.review(tmp_path, repo, stages, idea, old)
    path = tmp_path / failure.value.record_ref
    old_bytes = path.read_bytes()
    with pytest.raises(admission.ReviewUnavailable):
        admission.review(tmp_path, repo, stages, idea, old)
    assert old.calls == 1
    monkeypatch.setattr(admission, 'FORMAT_FEEDBACK_VERSION', admission.FORMAT_FEEDBACK_VERSION + 1)
    new = Client()
    result = admission.review(tmp_path, repo, stages, idea, new)
    assert result['status'] == 'reviewed' and new.calls == 1
    assert admission.review(tmp_path, repo, stages, idea, new) == result and new.calls == 1
    assert path.read_bytes() == old_bytes
    assert admission.read_review_evidence(tmp_path, failure.value.evidence_id)['status'] == 'unavailable'


def test_review_format_feedback_consolidates_quote_ids_and_missing_workflow_once(tmp_path):
    repo, idea, stages = fixture(tmp_path)
    class FormatClient(Client):
        def chat_with_metadata(self, system, user, **kwargs):
            packet = json.loads(user)
            text, metadata = super().chat_with_metadata(system, user, **kwargs)
            if self.calls == 1:
                answer = json.loads(text)
                answer['source_evidence'] = [row for row in answer['source_evidence']
                    if not row['source_id'].startswith('workflow_')]
                answer['source_evidence'][-1]['source_id'] = 'candidate_source_0'
                return json.dumps(answer), metadata
            assert packet['previous_answer']['verdict'] == 'inapplicable'
            assert any('workflow_' in error for error in packet['citation_errors'])
            assert packet['exact_quote_suggestions'][0]['source_id'] == 'native_source_0'
            return text, metadata
    client = FormatClient()
    result = admission.review(tmp_path, repo, stages, idea, client)
    assert client.calls == 2 and result['status'] == 'reviewed'
    assert result['answer']['verdict'] == 'inapplicable' and 'error' not in result
    assert 'validation_error' in result['responses'][0]
    assert admission.review(tmp_path, repo, stages, idea, client) == result
    assert client.calls == 2


@pytest.mark.parametrize('failure',[False,True])
def test_main_reads_success_and_failure_review_by_verified_namespace_id(tmp_path,failure):
    repo,idea,stages=fixture(tmp_path)
    client=Client('invented-status' if failure else 'inapplicable')
    if failure:
        with pytest.raises(admission.ReviewUnavailable) as caught:
            admission.review(tmp_path,repo,stages,idea,client)
        evidence_id=caught.value.evidence_id
    else:
        evidence_id=admission.review(tmp_path,repo,stages,idea,client)['evidence_id']
    chunks=[];offset=0
    while True:
        page=admission.read_review_evidence(tmp_path,evidence_id,offset=offset,limit=12000)
        chunks.append(page['text'])
        if page['next_offset']>=page['log_bytes']: break
        offset=page['next_offset']
    document=json.loads(''.join(chunks))
    assert document['record']['status']==('unavailable' if failure else 'reviewed')
    assert document['record']['responses'] and document['request']['payload']['candidate']['label']==idea.label
    assert 'source plausibility' in page['authority']
    from autosim.research.agent_runtime import _McpExecutor
    executor=object.__new__(_McpExecutor);executor.output=tmp_path;executor.workspace=repo
    assert executor.read_evidence({'evidence_id':evidence_id,'limit':500})['evidence_id']==evidence_id


@pytest.mark.parametrize('fault',['signature','request','record_symlink','ancestor_symlink','invalid_id','bad_offset','bad_limit'])
def test_review_evidence_read_rejects_tampering_and_unsafe_inputs(tmp_path,fault):
    repo,idea,stages=fixture(tmp_path)
    row=admission.review(tmp_path,repo,stages,idea,Client())
    path=tmp_path/row['record_ref'];request=tmp_path/row['request_ref']
    kwargs={};identity=row['evidence_id']
    if fault=='signature':
        value=read_json(path);value['authority']='native score';atomic_json(path,value)
    elif fault=='request': request.write_text(request.read_text()+' ')
    elif fault=='record_symlink':
        other=tmp_path/'other.json';path.rename(other);path.symlink_to(other)
    elif fault=='ancestor_symlink':
        directory=path.parent;other=tmp_path/'other';directory.rename(other);directory.symlink_to(other,target_is_directory=True)
    elif fault=='invalid_id': identity='execution-review:../'+path.stem
    elif fault=='bad_offset': kwargs['offset']=True
    elif fault=='bad_limit': kwargs['limit']=12001
    with pytest.raises(ValueError): admission.read_review_evidence(tmp_path,identity,**kwargs)


@pytest.mark.parametrize('verdict',['uncertain','invented-status'])
def test_main_candidate_catalog_surfaces_cached_review_and_failure(verdict,tmp_path):
    from autosim.research.prepare import Preparation
    from autosim.research.ideas import IdeaLibrary,CLEARED
    repo,idea,stages=fixture(tmp_path)
    client=Client(verdict)
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=client)
    real.stages=stages
    idea.status=CLEARED
    root=real.output/'research'/real.run_id
    IdeaLibrary(root/'ideas.json').add(idea)
    atomic_json(root/'controller_session.json',{'run_id':real.run_id,'repository':str(repo),
        'status':'paused','rounds':2,'next_round':1,'history':[{'metric_value':0.5}]})
    try: row=admission.review(real.output,repo,stages,idea,client)
    except admission.ReviewUnavailable as exc:
        row=read_json(real.output/exc.record_ref)
    options=real._research_options({'status':'paused','next_round':1},{})
    review=options['items'][0]['execution_review']
    assert review['evidence_id']==row['evidence_id']
    assert review['status']==row['status'] and review['because']
    calls=client.calls
    real._research_options({'status':'paused','next_round':1},{})
    assert client.calls==calls


def test_catalog_checks_native_observations_once_per_read_not_across_reads(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    from autosim.research.ideas import IdeaLibrary, CLEARED
    from autosim.research import frozen_policy_trial
    repo, idea, stages = fixture(tmp_path)
    real = Preparation(repo=repo, output=tmp_path/'run', scouting=tmp_path/'scout', client=Client('uncertain'))
    real.stages = stages
    root = real.output/'research'/real.run_id
    library = IdeaLibrary(root/'ideas.json')
    idea.status = CLEARED
    library.add(idea)
    library.add(Idea(label='second candidate', granularity='code', status=CLEARED,
        change={'file':'unused.py', 'find':'return 1', 'replace':'return 3'}))
    atomic_json(root/'controller_session.json', {'run_id':real.run_id, 'repository':str(repo),
        'status':'paused', 'rounds':3, 'next_round':1, 'history':[{'metric_value':.5}]})
    monkeypatch.setattr(frozen_policy_trial, 'offered', lambda _: [
        {'label':'baseline', 'measurement_sha256':'a'*64}])
    observed = []
    def diagnostics(output, checkout, **kwargs):
        observed.append((output, checkout))
        return [{'id':'runtime_diagnostic_probe', 'evidence_id':'probe',
                 'text':f'Current checked source {len(observed)}'}]
    monkeypatch.setattr(admission, 'native_observations', diagnostics)
    packets = []
    monkeypatch.setattr(admission, 'cached', lambda _, packet: packets.append(packet))
    real._research_options({'status':'paused', 'next_round':1}, {})
    assert len(observed) == 1 and len(packets) == 4  # Two candidates, two plans each.
    assert all(any(s['text']=='Current checked source 1' for s in p['sources']) for p in packets)
    packets.clear()
    real._research_options({'status':'paused', 'next_round':1}, {})
    assert len(observed) == 2 and len(packets) == 4
    assert all(any(s['text']=='Current checked source 2' for s in p['sources']) for p in packets)
    # A real execution review cannot inherit the catalog's previous evidence.
    admission.review(real.output, repo, stages, idea, real.client)
    assert len(observed) == 3


def test_inapplicable_or_malformed_review_does_not_admit_native_work(tmp_path):
    from autosim.research.prepare import Preparation
    repo,idea,stages=fixture(tmp_path)
    client=Client()
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=client)
    real.stages=stages
    research=SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    result=real._review_candidate_execution(research,idea.label)
    assert result['outcome']=='not attempted' and result['native_operation_status']=='not_started'
    assert not (real.output/'research/derived/attempts').exists()
    client.verdict='invented-status'
    stages['train']+='\n# new stage'
    result=real._review_candidate_execution(research,idea.label)
    assert result['outcome']=='not attempted'


def test_catalog_surfaces_unavailable_frozen_review_without_ordinary_review(tmp_path):
    from autosim.research.prepare import Preparation
    from autosim.research.ideas import IdeaLibrary, CLEARED
    repo, idea, stages = fixture(tmp_path)
    stages['evaluate'] = 'def stage(i):\n    return ["python", "unused.py"]\n'
    real = Preparation(repo=repo, output=tmp_path/'run', scouting=tmp_path/'scout', client=Client())
    real.stages = stages
    idea.status = CLEARED
    root = real.output/'research'/real.run_id
    IdeaLibrary(root/'ideas.json').add(idea)
    atomic_json(root/'controller_session.json', {'run_id':real.run_id, 'repository':str(repo),
        'status':'paused', 'rounds':2, 'next_round':1, 'history':[{'metric_value':.5}]})
    atomic_json(root/'snapshots/snapshots.json', {'best':'baseline'})
    parent = root/'measurements/baseline.json'
    atomic_json(parent, {'ok':True, 'policy_artifact':{'sha256':'a'*64}, 'train':{'attempt_id':'b'*32}})
    request = {'label':'baseline', 'measurement_sha256':digest(parent)}
    bad_client = Client('invented-status')
    with pytest.raises(admission.ReviewUnavailable) as failure:
        admission.review(real.output, repo, stages, idea, bad_client, reuse_policy=request)
    calls = bad_client.calls
    options = real._research_options({'status':'paused', 'next_round':1}, {})
    frozen = options['items'][0]['execution_review']['frozen_policy_reviews'][0]
    assert frozen['reuse_policy'] == request and frozen['status'] == 'unavailable'
    assert frozen['evidence_id'] == failure.value.evidence_id
    assert not frozen['retry_unchanged'] and 'Rewriting' in frozen['guidance']
    assert bad_client.calls == calls and real.client.calls == 0


@pytest.mark.parametrize('frozen_verdict', [None, 'uncertain', 'inapplicable'])
def test_catalog_separates_rejected_training_plan_from_explicit_frozen_plan(tmp_path, frozen_verdict):
    from autosim.research.prepare import Preparation
    from autosim.research.ideas import IdeaLibrary, CLEARED
    repo, idea, stages = fixture(tmp_path)
    stages['evaluate'] = 'def stage(i):\n    return ["python", "unused.py"]\n'
    class PlannedClient(Client):
        def chat_with_metadata(self, system, user, **kwargs):
            text, metadata = super().chat_with_metadata(system, user, **kwargs)
            answer = json.loads(text)
            for source in json.loads(user)['sources']:
                if source['id'] == 'workflow_evaluate':
                    answer['source_evidence'].append({'source_id': source['id'], 'quote': source['text']})
            return json.dumps(answer), metadata
    client = PlannedClient('inapplicable')
    real = Preparation(repo=repo, output=tmp_path/'run', scouting=tmp_path/'scout', client=client)
    real.stages = stages
    idea.status = CLEARED
    root = real.output/'research'/real.run_id
    IdeaLibrary(root/'ideas.json').add(idea)
    atomic_json(root/'controller_session.json', {'run_id': real.run_id, 'repository': str(repo),
        'status': 'paused', 'rounds': 2, 'next_round': 1, 'history': [{'metric_value': .5}]})
    # Offers are intentionally metadata-only; formal execution will verify all evidence.
    atomic_json(root/'snapshots/snapshots.json', {'best': 'baseline'})
    parent = root/'measurements/baseline.json'
    atomic_json(parent, {'ok': True, 'policy_artifact': {'sha256': 'a'*64},
                         'train': {'attempt_id': 'b'*32}})
    request = {'label': 'baseline', 'measurement_sha256': digest(parent)}
    ordinary = admission.review(real.output, repo, stages, idea, client)
    frozen = None
    if frozen_verdict is not None:
        frozen = admission.review(real.output, repo, stages, idea, PlannedClient(frozen_verdict), reuse_policy=request)
    calls = client.calls
    options = real._research_options({'status': 'paused', 'next_round': 1}, {})
    assert client.calls == calls  # Discovery must not trigger paid review or native work.
    assert options['frozen_policy_trials'][0]['label'] == 'baseline'
    if frozen_verdict == 'inapplicable':
        assert options['items'] == []
        rejected = options['excluded_incompatible'][0]
        assert rejected['label'] == idea.label
        assert rejected['evidence_id'] == frozen['evidence_id']
        assert rejected['execution_review']['evidence_id'] == ordinary['evidence_id']
        assert rejected['execution_review']['frozen_policy_review']['evidence_id'] == frozen['evidence_id']
    else:
        review = options['items'][0]['execution_review']
        assert review['scope'] == 'standard_training_and_evaluation'
        assert review['requires_reuse_policy'] == request
        assert review['evidence_id'] == ordinary['evidence_id']
        if frozen:
            assert review['frozen_policy_review']['verdict'] == frozen_verdict
            assert review['frozen_policy_review']['evidence_id'] == frozen['evidence_id']
    assert not (root/'attempts').exists()


def test_review_rejects_fake_citations_and_tampered_cache(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    packet=admission.packet_for(tmp_path,repo,stages,idea)
    with pytest.raises(ValueError):
        admission.validate({'verdict':'applicable','reason':'x',
                            'source_evidence':[{'source_id':'workflow_train','quote':'not present'}]},packet)
    row=admission.review(tmp_path,repo,stages,idea,Client())
    path=tmp_path/'candidate_execution_reviews'/f"{row['material_sha256']}.json"
    row['answer']['verdict']='applicable'
    atomic_json(path,row)
    assert admission.cached(tmp_path,packet) is None


@pytest.mark.parametrize('mode',['correct_once','always_invalid','transport'])
def test_review_seals_requests_replies_and_one_original_window_feedback(tmp_path,monkeypatch,mode):
    repo,idea,stages=fixture(tmp_path)
    clock=[100.0]
    timeouts=[]
    class DeliveryClient(Client):
        def chat_with_metadata(self,system,user,**kw):
            timeouts.append(kw['timeout'])
            if mode=='transport': raise OSError('fixture transport unavailable')
            content,metadata=super().chat_with_metadata(system,user,**kw)
            clock[0]+=30
            if len(timeouts)==1 or mode=='always_invalid':
                answer=json.loads(content)
                answer['source_evidence'][0]['quote']='invented quote'
                return json.dumps(answer),{'turn_id':'fixture-'+str(len(timeouts))}
            packet=json.loads(user)
            assert 'source_evidence[0]' in packet['format_feedback']
            return content,metadata
    monkeypatch.setattr(admission.time,'monotonic',lambda:clock[0])
    client=DeliveryClient()
    if mode=='correct_once':
        row=admission.review(tmp_path,repo,stages,idea,client)
        assert row['status']=='reviewed'
    else:
        with pytest.raises(admission.ReviewUnavailable) as failure:
            admission.review(tmp_path,repo,stages,idea,client)
        row=read_json(tmp_path/failure.value.record_ref)
        assert row['status']=='unavailable'
        count=len(timeouts)
        with pytest.raises(admission.ReviewUnavailable):
            admission.review(tmp_path,repo,stages,idea,client)
        assert len(timeouts)==count  # Unchanged failed delivery never pays again.
    assert timeouts==([90] if mode=='transport' else [90,60])
    request_path=tmp_path/row['request_ref']
    assert digest(request_path)==row['request_sha256']
    assert read_json(request_path)['payload']['candidate']['label']==idea.label
    assert len(row['responses'])==(0 if mode=='transport' else 2)
    if row['responses']:
        assert 'validation_error' in row['responses'][0]


def test_code_change_citations_use_literal_source_not_nested_json_escape(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    idea.change['find']='def load():\n    return 1'
    idea.change['replace']='def load():\n    return 2'
    packet=admission.packet_for(tmp_path,repo,stages,idea)
    source=next(r for r in packet['sources'] if r['id']=='candidate_change')
    assert idea.change['find'] in source['text']
    admission.validate({'verdict':'uncertain','reason':'more evidence needed',
                       'source_evidence':[{'source_id':'candidate_change','quote':idea.change['find']}]},packet)


def test_large_candidate_source_has_explicit_bounded_excerpts(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    (repo/'unused.py').write_text('# many lines\n'*10000+'def load():\n    return 1\n')
    packet=admission.material(repo,stages,idea)
    row=next(r for r in packet['sources'] if r['id']=='candidate_source_0')
    assert row['complete'] is False and 'return 1' in row['text']
    assert len(row['text'])<20000


def test_applicable_review_allows_dispatch_without_claiming_a_score(tmp_path):
    from autosim.research.prepare import Preparation
    repo,idea,stages=fixture(tmp_path)
    client=Client('applicable')
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=client)
    real.stages=stages
    research=SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    assert real._review_candidate_execution(research,idea.label) is None
    assert client.calls==1 and not (real.output/'research/derived/measurements').exists()


@pytest.mark.parametrize('verdict', ['uncertain', 'applicable'])
@pytest.mark.parametrize('with_current_evidence',[False,True])
@pytest.mark.parametrize('with_reader_history',[False,True])
def test_main_can_authorize_bounded_validation_without_changing_review_or_scoring(tmp_path, verdict, with_current_evidence, with_reader_history, monkeypatch):
    from autosim.research import candidate_validation_trial as trial
    from autosim.research import development_evidence
    repo, idea, stages = fixture(tmp_path)
    idea.status = 'cleared'
    settings = {'steps': 3}
    refs=[]
    if with_current_evidence:
        from autosim.research.harness_repair import read_source
        refs=['evidence:'+read_source(tmp_path,{'path':'research/data_versions.py',
            'query':'def native_selection_payload','limit':20})['evidence_id']]
    research_root=tmp_path/'research/derived' if with_reader_history else None
    if with_reader_history:
        monkeypatch.setattr(development_evidence,'reader_failure_diagnostics',lambda root:[{
            'label':'round_2','why':'Extra data at column 300','evidence_id':'c'*32}])
    row = admission.review(tmp_path, repo, stages, idea, Client(verdict),
                           selected_training_settings=settings,additional_evidence_refs=refs,
                           research_root=research_root)
    before = (tmp_path/row['record_ref']).read_bytes()
    research = SimpleNamespace(library=SimpleNamespace(get=lambda label:idea), sources=stages,
                               run_root=research_root)
    refusal = None if verdict == 'applicable' else {
        'failure_category':'candidate_execution_relevance', 'candidate_execution_review':row['answer']}
    kwargs = dict(output=tmp_path,repo=repo,stages=stages,idea_label=idea.label,
        settings=settings, training_settings=settings, decision_id='a'*16,
        request={'reason':'source path is plausible but needs a real reader observation',
                 'postcondition':'observe actual bytes and work before accepting a score',
                 'stage_window_seconds':45}, admission=refusal,additional_evidence_refs=refs)
    result = trial.authorize(research, **kwargs)
    assert result == trial.authorize(research, **kwargs)
    sealed = read_json(tmp_path/result['authorization_ref'])
    assert sealed['decision_id'] == 'a'*16 and sealed['review_verdict'] == verdict
    assert sealed['review_record_sha256'] == row['record_sha256']
    assert 'not reviewer approval' in sealed['authority']
    from autosim.research.agent_runtime import _McpExecutor
    executor=object.__new__(_McpExecutor);executor.output=tmp_path;executor.workspace=repo
    page=executor.read_evidence({'evidence_id':result['evidence_id'],'limit':12000})
    assert page['status']=='authorization_only' and page['returncode'] is None
    assert json.loads(page['text'])['decision_id']=='a'*16
    assert 'not reviewer approval' in page['text']
    assert before == (tmp_path/row['record_ref']).read_bytes()
    assert not (tmp_path/'research/derived/measurements').exists()
    assert (repo/'unused.py').read_text() == 'def load():\n    return 1\n'


@pytest.mark.parametrize('fault', ['inapplicable','unavailable','changed_settings','changed_source',
                                  'changed_verdict','bad_signature','symlink','not_audited'])
def test_bounded_validation_cannot_bypass_negative_missing_or_changed_reviews(tmp_path, fault):
    from autosim.research import candidate_validation_trial as trial
    repo, idea, stages = fixture(tmp_path)
    idea.status = 'cleared'
    settings = {'steps':3}
    row = admission.review(tmp_path,repo,stages,idea,Client('uncertain'),
                           selected_training_settings=settings)
    refusal = {'failure_category':'candidate_execution_relevance',
               'candidate_execution_review':row['answer']}
    if fault in {'inapplicable','changed_verdict'}:
        refusal['candidate_execution_review'] = {**row['answer'], 'verdict':'inapplicable'}
    elif fault == 'unavailable': refusal['failure_category'] = 'candidate_execution_review_unavailable'
    elif fault == 'changed_settings': settings = {'steps':4}
    elif fault == 'changed_source': (repo/'unused.py').write_text('def load():\n    return 3\n')
    elif fault == 'bad_signature':
        record = read_json(tmp_path/row['record_ref']);record['record_sha256']='0'*64
        atomic_json(tmp_path/row['record_ref'],record)
    elif fault == 'symlink':
        target = tmp_path/'other';target.mkdir()
        (tmp_path/'candidate_execution_reviews/validation_trials').symlink_to(target)
    elif fault == 'not_audited': idea.status = 'rejected'
    research = SimpleNamespace(library=SimpleNamespace(get=lambda label:idea), sources=stages)
    with pytest.raises(ValueError):
        trial.authorize(research,output=tmp_path,repo=repo,stages=stages,idea_label=idea.label,
            settings=settings,training_settings=settings,decision_id='a'*16,
            request={'reason':'test','postcondition':'real evidence','stage_window_seconds':10},
            admission=refusal)
    assert not list((tmp_path/'candidate_execution_reviews/validation_trials').glob('*.json'))


@pytest.mark.parametrize('fault', ['reason','postcondition','bool_window','nan','negative','extra',
                                  'no_decision','no_candidate','no_work','job','policy','protocol'])
def test_main_validation_request_requires_explicit_bounded_owned_work(fault):
    from autosim.research.candidate_validation_trial import validate_request
    request = {'reason':'test new connection','postcondition':'actual data and policy witnesses',
               'stage_window_seconds':30}
    kwargs = dict(decision_id='a'*16,idea_label='candidate',training_settings={'steps':3})
    if fault in {'reason','postcondition'}: request[fault] = ''
    elif fault == 'bool_window': request['stage_window_seconds'] = True
    elif fault == 'nan': request['stage_window_seconds'] = float('nan')
    elif fault == 'negative': request['stage_window_seconds'] = -1
    elif fault == 'extra': request['skip_score_checks'] = True
    elif fault == 'no_decision': kwargs['decision_id'] = ''
    elif fault == 'no_candidate': kwargs['idea_label'] = ''
    elif fault == 'no_work': kwargs['training_settings'] = {}
    elif fault == 'job': kwargs['job_id'] = 'b'*32
    elif fault == 'policy': kwargs['reuse_policy'] = {'label':'old'}
    elif fault == 'protocol': kwargs['evaluation_settings'] = {'episodes':1}
    with pytest.raises(ValueError): validate_request(request, **kwargs)


@pytest.mark.parametrize('fail', [False,True])
def test_main_validation_window_restores_limits_and_does_not_reset_budget(fail):
    from autosim.research.candidate_validation_trial import stage_window
    budget = SimpleNamespace(remaining=lambda:20, unchanged='original-start-and-usage')
    research = SimpleNamespace(STAGE_TIMEOUT_SECONDS=600, budget=budget)
    def run():
        with stage_window(research,{'stage_window_seconds':30}):
            assert research.STAGE_TIMEOUT_SECONDS == 20
            assert research.budget is budget
            if fail: raise RuntimeError('native failure remains a failure')
    if fail:
        with pytest.raises(RuntimeError): run()
    else: run()
    assert research.STAGE_TIMEOUT_SECONDS == 600
    assert budget.unchanged == 'original-start-and-usage'


def test_allowed_large_review_request_remains_readable_after_sealing(tmp_path, monkeypatch):
    repo,idea,stages = fixture(tmp_path)
    packet = admission.packet_for(tmp_path,repo,stages,idea)
    packet['sources'].append({'id':'bounded_framework_pages', 'text':'x'*(280*1024)})
    # Exercise the store's envelope, not source discovery or any approval.
    assert 256*1024 < len(json.dumps(packet).encode()) < admission.MAX_REVIEW_BYTES
    monkeypatch.setattr(admission,'packet_for',lambda *a,**k:packet)
    row = admission.review(tmp_path,repo,stages,idea,Client('uncertain'))
    page = admission.read_review_evidence(tmp_path,row['evidence_id'],limit=500)
    assert page['evidence_id'] == row['evidence_id'] and page['status'] == 'reviewed'
    assert page['log_bytes'] > 256*1024 and page['next_offset'] == 500


def test_uncertain_review_still_blocks_native_execution_by_default(tmp_path):
    from autosim.research.prepare import Preparation
    repo,idea,stages = fixture(tmp_path)
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=Client('uncertain'))
    real.stages=stages
    research=SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    result = real._review_candidate_execution(research,idea.label)
    assert result['candidate_execution_review']['verdict'] == 'uncertain'
    assert result['native_operation_status'] == 'not_started'
    assert not (real.output/'candidate_execution_reviews/validation_trials').exists()


def test_review_distinguishes_training_seed_from_frozen_evaluation(tmp_path):
    from autosim.research.candidate_settings import TRAINING_SEED
    repo,idea,stages=fixture(tmp_path)
    packet=admission.packet_for(tmp_path,repo,stages,idea,
        selected_training_settings={'seed':20261013,TRAINING_SEED:1000,'steps':3})
    row=next(x for x in packet['sources'] if x['id']=='selected_training_inputs')
    scopes=json.loads(row['text'])['seed_scopes']
    assert scopes['requested_train_seed']==1000 and scopes['evaluation_seed']==20261013
    assert 'not proof' in scopes['authority']


@pytest.mark.parametrize('verdict,explicit,expected_train_count,native_fail,authority_error', [
    ('uncertain',False,1,False,False), ('uncertain',True,2,False,False),
    ('uncertain',True,2,True,False),
    ('inapplicable',True,1,False,False), ('invalid-verdict',True,1,False,False),
    ('uncertain',True,1,False,True)])
@pytest.mark.parametrize('with_reader_history',[False,True])
def test_main_selected_uncertainty_through_prepare_and_real_native_workflow(
        tmp_path,monkeypatch,verdict,explicit,expected_train_count,native_fail,authority_error,with_reader_history):
    import sys
    from pathlib import Path
    from autosim.research.adapter_protocol import Axis,OptimizationSpace
    from autosim.research.compute_decision import ComputeDecision
    from autosim.research.declarative_backend import DeclarativeBackend
    from autosim.research.derived_research import DerivedResearch
    from autosim.research.prepare import Preparation
    repo=tmp_path/'repo';repo.mkdir()
    client=Client(verdict);client.supports_main_agent=True
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=client)
    real.declaration={'benchmark':'synthetic','task_contract':{'primary_metric':{
        'name':'success_rate','direction':'maximize','unit':'fraction','source':'log'}},
        'optimization_space':{'collection':[],'training':[{
            'name':'epochs','kind':'integer','description':'epochs','low':1,'high':10,'default':1}]}}
    real.interpreter=Path(sys.executable)
    native=('import pathlib,sys; p=pathlib.Path(sys.argv[1]); '
            '(p/"models").mkdir(parents=True,exist_ok=True); '
            '(p/"models"/"model.pth").write_text("ok"); print("succ: 0.50")')
    if native_fail:
        native += '; sys.exit(7) if int(sys.argv[2])==2 else None'
    real.stages={stage:(f'def stage_argv_{stage}(i):\n'
                       f'    return [i["python"],"-c",{native!r},i["output"],str(i["settings"].get("epochs",1))]\n')
                 for stage in ('train','evaluate')}
    real.execution={'stages':{stage:{'available':True} for stage in real.stages}}
    real.verified={stage:{'verified':True} for stage in real.stages}
    # This test isolates relevance admission; these pre-existing gates have
    # independent negative regressions in the broader Preparation suite.
    monkeypatch.setattr(real,'_metric_bound',lambda:True)
    monkeypatch.setattr(real,'_confirmation_admission_problem',lambda:None)
    from autosim.research import baseline_diagnosis
    monkeypatch.setattr(baseline_diagnosis,'view',lambda *a,**k:{'status':'not_required'})
    backend=DeclarativeBackend(repo=repo,answer={'stages':{
        stage:{'available':True,'entrypoint':'native.py','invocation':'python native.py',
               'artifact':'models/*.pth' if stage=='train' else ''} for stage in real.stages}},
        sources=real.stages,parameters={})
    engine=DerivedResearch(repo=repo,output=real.output,backend=backend,
        interpreter=Path(sys.executable),space=OptimizationSpace(training=(
            Axis('epochs','integer','epochs',low=1,high=10,default=1),)),
        client=None,stages=real.stages,run_id=real.run_id,
        compute=ComputeDecision(device='cpu',device_index=0,environment={},why='CPU regression'),
        benchmark='synthetic')
    engine.run(rounds=1,settings={},confirm=False,yield_after_action=True,
               main_controller_owns_selection=True)
    baseline=(engine.run_root/'measurements/baseline.json').read_bytes()
    idea=Idea(label='longer exposure',granularity='param',change={'epochs':2},status='cleared')
    engine.library.add(idea)
    engine.controller_decision_id='a'*16
    monkeypatch.setattr(real,'_research_controller',lambda:(engine,{}))
    if with_reader_history:
        from autosim.research import development_evidence
        monkeypatch.setattr(development_evidence,'reader_failure_diagnostics',lambda root:[{
            'label':'round_2','why':'Extra data at column 300','evidence_id':'c'*32}])
    if authority_error:
        from autosim.research import candidate_validation_trial
        def fail_authorization(*args,**kwargs):
            raise ValueError('synthetic authenticated reconstruction failure')
        monkeypatch.setattr(candidate_validation_trial,'authorize',fail_authorization)
    real.current_action={'decision_id':'a'*16}
    kwargs={'idea_label':idea.label,'training_settings':{'epochs':2}}
    if explicit:
        kwargs['execution_uncertainty']={'reason':'test plausible but unobserved native connection',
            'postcondition':'only a completed native measurement can provide a score',
            'stage_window_seconds':30}
    previous=engine.STAGE_TIMEOUT_SECONDS
    result=real._step_run_the_loop(**kwargs)
    receipts=[read_json(p) for p in (engine.run_root/'attempts').glob('*/receipt.json')]
    assert sum(r['node_id']=='train' for r in receipts)==expected_train_count, result
    assert (engine.run_root/'measurements/baseline.json').read_bytes()==baseline
    assert engine.STAGE_TIMEOUT_SECONDS==previous
    if expected_train_count==2:
        assert result['bounded_validation']['review_verdict']=='uncertain'
        assert result['bounded_validation']['authorization_ref'] in result['evidence_refs']
        assert result['attempt_ids'] and result['selected_idea_label']==idea.label
        if native_fail:
            measurement=read_json(engine.run_root/'measurements/round_1.json')
            assert not measurement['ok'] and measurement.get('metric_value') is None
            assert sum(r['node_id']=='evaluate' for r in receipts)==1
    else:
        assert result['native_operation_status']=='not_started'
        assert result['failure_category'] == ('candidate_validation_authority' if authority_error else
            'candidate_execution_review_unavailable' if verdict=='invalid-verdict' else
            'candidate_execution_relevance'), result
        assert not (real.output/'candidate_execution_reviews/validation_trials').exists()
        if authority_error:
            from autosim.research.evidence_store import read_attempt_evidence
            assert result['repair_owner']=='fix'
            assert 'synthetic authenticated reconstruction failure' in result['because']
            page=read_attempt_evidence(real.output,result['evidence_id'],limit=12000)
            assert page['status']=='operation_contract_failed'
            assert 'synthetic authenticated reconstruction failure' in page['text']
            assert 'execution-review:' in page['text']
            assert result['candidate_execution_review']['verdict']=='uncertain'
            record=next((real.output/'candidate_execution_reviews').glob('*.json'))
            assert read_json(record)['answer']['verdict']=='uncertain'


def test_detached_candidate_rejects_irrelevant_change_before_allocation(tmp_path, monkeypatch):
    from autosim.research.prepare import Preparation
    from autosim.research import candidate_jobs
    repo,idea,stages=fixture(tmp_path)
    real=Preparation(repo=repo,output=tmp_path/'run',scouting=tmp_path/'scout',client=Client())
    real.stages=stages
    research=SimpleNamespace(library=SimpleNamespace(get=lambda label:idea))
    monkeypatch.setattr(real,'_research_controller',lambda:(research,{}))
    def forbidden(*args,**kwargs):
        pytest.fail('irrelevant candidate must not allocate code or compute')
    monkeypatch.setattr(candidate_jobs,'allocation',forbidden)
    result=real._step_submit_native_job(stage='train',window_seconds=900,
                                      reason='diagnostic control',idea_label=idea.label)
    assert result['native_operation_status']=='not_started'
    assert result['failure_category']=='candidate_execution_relevance'


def test_malformed_citations_receive_one_bounded_format_retry(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    class Repair(Client):
        def chat_with_metadata(self,system,user,**kwargs):
            if not self.calls:
                self.calls+=1
                return json.dumps({'verdict':'applicable','reason':'reachable','source_evidence':[]}),{}
            assert 'both workflow and intervention' in json.loads(user)['format_feedback']
            return super().chat_with_metadata(system,user,**kwargs)
    client=Repair()
    row=admission.review(tmp_path,repo,stages,idea,client)
    assert row['answer']['verdict']=='inapplicable' and client.calls==2
    assert 'error' not in row


def test_standard_and_frozen_policy_plans_have_distinct_material_identity(tmp_path):
    repo,idea,stages=fixture(tmp_path)
    stages['evaluate']='def stage(i):\n    return ["python", "primary.py"]\n'
    normal=admission.packet_for(tmp_path,repo,stages,idea)
    request={'label':'baseline','measurement_sha256':'a'*64}
    frozen=admission.packet_for(tmp_path,repo,stages,idea,reuse_policy=request)
    assert normal['execution_plan']['will_train'] is True
    assert frozen['execution_plan']['stages']==['evaluate']
    assert normal['sources']==frozen['sources']
    from autosim.research.common import object_digest
    assert object_digest(normal)!=object_digest(frozen)


def test_requested_diagnostic_ids_follow_agent_citations_not_hash_substrings():
    older = Idea(label='old', evidence=['prior probe '+ 'a'*32])
    newer = Idea(label='new', evidence=['current sealed probe '+ 'b'*32,
        'measurement_sha256 '+ 'c'*64, 'private/../../path', {'not':'text'}])
    assert admission.requested_native_ids([older,newer]) == ['b'*32, 'a'*32]
    many = Idea(label='many', evidence=[' '.join(f'{i:032x}' for i in range(30))])
    assert len(admission.requested_native_ids([many])) == 12


def test_shared_pool_does_not_change_per_candidate_review_identity(tmp_path, monkeypatch):
    from autosim.research.common import object_digest
    repo, idea, stages = fixture(tmp_path)
    idea.evidence = ['Relevant installed probe '+ 'a'*32]
    def row(identity, recent):
        return {'id':'runtime_diagnostic_'+identity, 'evidence_id':identity,
                'text':'Public source '+identity, '_selection_recent':recent}
    # Another idea cites older b; it must not leak into this idea's formal packet.
    pool = [row('a'*32,False), row('b'*32,False)] + [row(f'{i:032x}',True) for i in (9,10,11)]
    monkeypatch.setattr(admission, 'native_observations', lambda *args,**kwargs:
                        admission.select_native_observations(pool,kwargs.get('requested_ids',())))
    fresh = admission.packet_for(tmp_path,repo,stages,idea)
    projected = admission.packet_for(tmp_path,repo,stages,idea,
        native_observation_snapshot={'pool':pool,'requested_ids':['b'*32,'a'*32]})
    assert object_digest(fresh) == object_digest(projected)
    assert all(s['id'] != 'runtime_diagnostic_'+'b'*32 for s in projected['sources'])
    assert '_selection_recent' not in json.dumps(projected)


@pytest.mark.parametrize('fault', ['none','context','tampered_log','gpu','symlink','changed_source'])
def test_explicit_older_native_evidence_survives_recency_only_when_still_valid(tmp_path, monkeypatch, fault):
    import os
    from autosim.research import native_context
    from autosim.research.diagnostic_source_identity import source_identity
    from autosim.research.evidence_store import capture_attempt_evidence
    monkeypatch.setattr(native_context,'load_context',lambda *args:{'identity':'current'})
    repo = tmp_path/'checkout'; repo.mkdir(); (repo/'train.py').write_text('N = 1\n')
    directory = tmp_path/'native_diagnostics'; directory.mkdir()
    for index in range(12):
        identity = f'{index:032x}'
        path = directory/f'{identity}.json'
        log = directory/f'{identity}.log'; log.write_text(f'Actual installed API evidence {index}\n')
        sealed = capture_attempt_evidence(tmp_path,attempt_id=identity,log=log,
            receipt_ref=f'native_diagnostics/{identity}.json',status='completed',returncode=0,
            termination_reason='cpu_native_diagnostic')
        receipt = {**sealed,'status':'completed','returncode':0,'context_identity':'current',
            'repository_source_identity':source_identity(repo),'repository_source_stable':True,
            'capability':'cpu_diagnostic_not_simulation_readiness','gpu_access':False}
        if index == 0 and fault == 'context': receipt['context_identity'] = 'old'
        if index == 0 and fault == 'gpu': receipt['gpu_access'] = True
        atomic_json(path,receipt)
        os.utime(path,(index+1,index+1))
        if index == 0 and fault == 'tampered_log': log.write_text('invented current contract')
        if index == 0 and fault == 'symlink':
            other = tmp_path/'old-receipt.json'; path.rename(other); path.symlink_to(other)
    if fault == 'changed_source': (repo/'train.py').write_text('N = 2\n')
    assert 'runtime_diagnostic_'+'0'*32 not in {r['id'] for r in admission.native_observations(tmp_path,repo)}
    selected = admission.native_observations(tmp_path,repo,requested_ids=['0'*32,'../unsafe'])
    ids = {row['id'] for row in selected}
    assert ('runtime_diagnostic_'+'0'*32 in ids) is (fault == 'none')
    if fault != 'changed_source':
        assert {f'runtime_diagnostic_{i:032x}' for i in (9,10,11)} <= ids
        assert len(ids) == (4 if fault == 'none' else 3)
    else:
        assert selected == []


@pytest.mark.parametrize('fault',['none','context','tampered_log','receipt_binding','gpu','symlink',
                                 'legacy_source', 'changed_source', 'unstable_source'])
def test_native_api_diagnostics_are_current_sealed_and_readonly(tmp_path,monkeypatch,fault):
    from autosim.research import native_context
    from autosim.research.evidence_store import capture_attempt_evidence
    monkeypatch.setattr(native_context,'load_context',lambda *args:{'identity':'context-current'})
    from autosim.research.diagnostic_source_identity import source_identity
    repo=tmp_path/'checkout';repo.mkdir();(repo/'policy.py').write_text('chunk = 8')
    directory=tmp_path/'native_diagnostics';directory.mkdir()
    identity='a'*32;log=directory/f'{identity}.log';log.write_text('actual API source\n'*1000)
    ref=f'native_diagnostics/{identity}.json'
    sealed=capture_attempt_evidence(tmp_path,attempt_id=identity,log=log,receipt_ref=ref,
                                    status='completed',returncode=0,termination_reason='cpu_native_diagnostic')
    receipt={**sealed,'status':'completed','returncode':0,'context_identity':'context-current',
             'repository_source_identity':source_identity(repo),'repository_source_stable':True,
             'capability':'cpu_diagnostic_not_simulation_readiness','gpu_access':False}
    if fault=='legacy_source': receipt.pop('repository_source_identity')
    if fault=='changed_source': (repo/'policy.py').write_text('chunk = 32')
    if fault=='unstable_source': receipt['repository_source_stable']=False
    if fault=='context': receipt['context_identity']='old'
    if fault=='gpu': receipt['gpu_access']=True
    if fault=='receipt_binding':
        other=tmp_path/'evidence'/f'{identity}.json'
        row=json.loads(other.read_text());row['receipt_ref']='other.json';atomic_json(other,row)
    atomic_json(tmp_path/ref,receipt)
    if fault=='tampered_log': log.write_text('altered')
    if fault=='symlink':
        directory.rename(tmp_path/'other');directory.symlink_to(tmp_path/'other',target_is_directory=True)
    rows=admission.native_observations(tmp_path,repo)
    if fault=='none':
        assert len(rows)==2 and rows[0]['complete'] is False and rows[1]['complete'] is False
        assert rows[1]['id']==f'runtime_diagnostic_{identity}_tail'
        assert rows[1]['offset']>=rows[0]['next_offset']
        assert rows[0]['context_identity']=='context-current'
        assert len(rows[0]['text'])<=8000 and 'not a score' in rows[0]['authority']
    else:
        assert rows==[]


def test_native_consumer_evidence_after_long_source_is_offered_as_separate_page(tmp_path,monkeypatch):
    from autosim.research import native_context
    from autosim.research.diagnostic_source_identity import source_identity
    from autosim.research.evidence_store import capture_attempt_evidence
    monkeypatch.setattr(native_context,'load_context',lambda *args:{'identity':'current'})
    repo=tmp_path/'repo';repo.mkdir();(repo/'policy.py').write_text('x=1')
    directory=tmp_path/'native_diagnostics';directory.mkdir()
    identity='f'*32;log=directory/f'{identity}.log'
    log.write_text('repository source\n'*1200+'ACTUAL_CONSUMER_QUEUE_KEY = observation.images\n')
    ref=f'native_diagnostics/{identity}.json'
    sealed=capture_attempt_evidence(tmp_path,attempt_id=identity,log=log,receipt_ref=ref,
        status='completed',returncode=0,termination_reason='cpu_native_diagnostic')
    atomic_json(tmp_path/ref,{**sealed,'status':'completed','returncode':0,'context_identity':'current',
        'capability':'cpu_diagnostic_not_simulation_readiness','gpu_access':False,
        'repository_source_identity':source_identity(repo),'repository_source_stable':True})
    rows=admission.native_observations(tmp_path,repo)
    assert len(rows)==2
    assert 'ACTUAL_CONSUMER_QUEUE_KEY' not in rows[0]['text']
    assert 'ACTUAL_CONSUMER_QUEUE_KEY' in rows[1]['text']
    assert all(len(row['text'].encode())<=8000 and row['complete'] is False for row in rows)
