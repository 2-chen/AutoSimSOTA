import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from autosim.research import training_progress_audit as pa, execution_derive as ed
from autosim.research.common import digest


def evidence(tmp_path):
    output = tmp_path/'output'
    output.mkdir()
    doc = output/'work.json'
    doc.write_text(json.dumps({'done':{'updates':17}, 'configuration':{'requested':1000}}))
    source = tmp_path/'producer.py'
    source.write_text('def train():\n    optimizer.step()\n    completed += 1\n    save({"updates": completed})\n')
    docs = pa.candidates(output, time.time()-10)
    code = [{'id':'source-0','path':source,'sha256':digest(source),'content':source.read_text()}]
    answer = {'verdict':'verified','counter':{'path':'work.json','field':['done','updates'],'value':17},
        'source_evidence':[{'source_id':'source-0','quote':'optimizer.step()\n    completed += 1'}],
        'why':'actual completed updates, not the requested budget'}
    return output, docs, code, answer


def test_verified_counter_requires_exact_numeric_bytes_and_source_quotes(tmp_path):
    output, docs, code, answer = evidence(tmp_path)
    result = pa.validate(answer, output=output, docs=docs, code=code)
    assert result['status'] == 'observed'
    assert result['evidence']['native_counter']['value'] == 17


@pytest.mark.parametrize('quote', ['step += 1', '', '   ', 'step += 2'])
def test_exact_short_update_statement_is_not_rejected_by_arbitrary_length(tmp_path, quote):
    output, docs, code, answer = evidence(tmp_path)
    source = code[0]['path']
    source.write_text('optimizer.step()\nstep += 1\nsave(step)\n')
    code[0].update(content=source.read_text(), sha256=digest(source))
    answer['source_evidence'] = [
        {'source_id': 'source-0', 'quote': 'optimizer.step()'},
        {'source_id': 'source-0', 'quote': quote},
        {'source_id': 'source-0', 'quote': 'save(step)'}]
    if quote == 'step += 1':
        assert pa.validate(answer, output=output, docs=docs, code=code)['status'] == 'observed'
    else:
        with pytest.raises(ValueError):
            pa.validate(answer, output=output, docs=docs, code=code)


@pytest.mark.parametrize('fault', ['zero','invented_field','invented_quote','changed_doc','changed_source','no_review'])
def test_bad_progress_claims_remain_unverified(tmp_path, fault):
    output, docs, code, answer = evidence(tmp_path)
    if fault == 'zero': answer['counter']['value'] = 0
    if fault == 'invented_field': answer['counter']['field'] = ['optimizer_step']
    if fault == 'invented_quote': answer['source_evidence'][0]['quote'] = 'native_step += 1 # invented'
    if fault == 'changed_doc': (output/'work.json').write_text('{"done":{"updates":999}}')
    if fault == 'changed_source': code[0]['path'].write_text('print("fake")')
    if fault == 'no_review': answer['verdict'] = 'unverified'
    with pytest.raises(ValueError): pa.validate(answer, output=output, docs=docs, code=code)


def test_fresh_metadata_is_numeric_only_and_symlink_safe(tmp_path):
    output, _, _, _ = evidence(tmp_path)
    (output/'secret.json').write_text('{"token":"sensitive-text", "count":2}')
    (output/'link.json').symlink_to(output/'work.json')
    rows = pa.candidates(output, time.time()-10)
    assert 'sensitive-text' not in json.dumps(rows)
    assert 'link.json' not in [row['path'] for row in rows]
    assert pa.candidates(output, time.time()+1) == []


def test_argv_verification_uses_distinct_outputs_without_creating_them(tmp_path):
    rows = iter([{'source':'def stage_argv_train(i): return ["python", i["output"]]'}]*2)
    used = []
    class Client:
        def chat_with_metadata(self,*args,**kw): return json.dumps(next(rows)), {}
    def inputs():
        path = tmp_path/('attempt-'+str(len(used)))
        assert not path.exists()
        used.append(str(path))
        return {'output':str(path)}
    commands = []
    def verify(argv):
        commands.append(argv)
        return {'ok':len(commands)==2,'error':'first attempt failed'}
    source, _, _ = ed.generate_argv(Client(),'train',entrypoint='train.py',invocation='python train.py',
        repository_files=['train.py'], verify=verify, attempts=2, verification_inputs_factory=inputs)
    assert source and len(used)==2 and used[0]!=used[1]
    assert [argv[-1] for argv in commands] == used


def test_source_trace_follows_project_imports_without_importing_code(tmp_path):
    repo = tmp_path/'checkout'
    repo.mkdir()
    (repo/'worker.py').write_text('raise RuntimeError("must never execute")\n')
    trainer = repo/'trainer.py'
    trainer.write_text('import worker\n')
    rows = pa.sources(repo,tmp_path,[str(trainer)])
    assert len(rows)==2
    assert any('must never execute' in row['content'] for row in rows)


def test_audit_uses_context_managed_supervisor_and_revalidates_cached_evidence(tmp_path, monkeypatch):
    output, docs, code, answer = evidence(tmp_path)
    code[0]['excerpt'] = code[0]['content']
    monkeypatch.setattr(pa, 'sources', lambda *a: code)
    class Client:
        role = 'init'
        calls = 0
        @contextmanager
        def as_role(self, role):
            prior = self.role
            self.role = role
            try:
                yield
            finally:
                self.role = prior
        def chat_with_metadata(self, system, payload, **kw):
            assert self.role == 'supervisor'
            assert kw['read_only'] is True
            assert json.loads(payload)['numeric_documents']
            self.calls += 1
            return json.dumps(answer), {}
    client = Client()
    kwargs = dict(repo=tmp_path,root=tmp_path,output=output,argv=[],since=0,
                  native_evidence_ref='evidence/'+'f'*32+'.json')
    result = pa.audit(client, **kwargs)
    assert result['status'] == 'observed'
    assert client.role == 'init' and client.calls == 1
    assert pa.audit(client, **kwargs)['status'] == 'observed'
    assert client.calls == 1
    code[0]['path'].write_text('different producer')
    assert pa.audit(client, **kwargs)['status'] == 'unknown'
    assert client.calls == 1
