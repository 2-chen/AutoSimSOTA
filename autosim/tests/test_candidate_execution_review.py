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
