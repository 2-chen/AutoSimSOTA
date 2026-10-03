import json
import os
import sys
import time

import pytest

from autosim.research.agent_client import AgentRuntimeClientError
from autosim.research.common import read_json
from autosim.research.process_executor import run_process_stream
from autosim.research.repair_schema import proposal_errors
from autosim.research.review_transaction import review_call, review_view
from autosim.research.recorder import action_explanation


def test_fast_producer_survives_slow_consumer_and_preserves_terminal_record(tmp_path):
    received = []
    def consume(line):
        time.sleep(.012)
        received.append(line)
    code = "import os\nfor i in range(100): os.write(1, (str(i)+':'+'x'*16000+'\\n').encode())\nos.write(1,b'{\"type\":\"result\"}\\n')"
    result = run_process_stream([sys.executable, '-c', code], cwd=tmp_path,
        env=os.environ.copy(), timeout=.8, on_stdout_line=consume, decouple_callbacks=True)
    assert not result.timed_out and result.error is None and result.returncode == 0
    assert len(received) == 101 and received[-1] == '{"type":"result"}'
    assert result.stdout.endswith('{"type":"result"}\n')


def test_split_utf8_frames_and_incomplete_final_frame_are_delivered_not_repaired(tmp_path):
    received = []
    code = "import os,time\na='中文'.encode()\nos.write(1,a[:2]);time.sleep(.05);os.write(1,a[2:]+b'\\n{\"type')"
    result = run_process_stream([sys.executable, '-c', code], cwd=tmp_path,
        env=os.environ.copy(), timeout=2, on_stdout_line=received.append, decouple_callbacks=True)
    assert received == ['中文', '{"type']
    assert result.stdout.endswith('{"type')


def test_decoupled_complete_line_limit_and_callback_failure_fail_closed(tmp_path):
    result = run_process_stream([sys.executable, '-c', "print('x'*500)"],
        cwd=tmp_path, env=os.environ.copy(), timeout=2, decouple_callbacks=True, max_line_bytes=100)
    assert result.error is not None
    def bad(_):raise ValueError('consumer failed')
    result = run_process_stream([sys.executable, '-c', "print('hello')"],
        cwd=tmp_path, env=os.environ.copy(), timeout=2, decouple_callbacks=True, on_stdout_line=bad)
    assert result.error is not None


class Reviewer:
    model = 'fixture'
    supports_main_agent = True
    def __init__(self, fail=False): self.calls=[];self.fail=fail
    def chat_with_metadata(self, system, user, **options):
        self.calls.append(options)
        if self.fail or len(self.calls) == 1:
            raise AgentRuntimeClientError('truncated',failure_category='stream_protocol',
                runtime_failure={'evidence_ref':'evidence/'+'a'*32+'.json'})
        return json.dumps({'approved':True,'reason':'source-backed','citations':[]}), {'turn_id':'b'*32}


def test_review_retries_only_review_then_reuses_identical_bundle(tmp_path):
    client=Reviewer()
    content, meta=review_call(client,instructions='review',payload={'source':'x'},output=tmp_path)
    assert json.loads(content)['approved']
    assert [c['output_format'] for c in client.calls] == ['json','stream-json']
    assert all(c['decision_only'] and not c['include_research_context'] for c in client.calls)
    content, meta=review_call(client,instructions='review',payload={'source':'x'},output=tmp_path)
    assert meta['cache_hit'] and len(client.calls)==2
    review_call(client,instructions='review',payload={'source':'changed'},output=tmp_path)
    assert len(client.calls)==3


def test_exhausted_review_bundle_cannot_loop_or_execute(tmp_path):
    client=Reviewer(fail=True)
    for _ in range(2):
        with pytest.raises(AgentRuntimeClientError):
            review_call(client,instructions='review',payload={},output=tmp_path)
    assert len(client.calls)==2
    assert review_view(tmp_path)[0]['status']=='transport_unavailable'
    assert review_view(tmp_path)[0]['attempts'][0]['evidence_ref'].startswith('evidence/')


def test_proposal_returns_all_field_faults_at_once():
    errors=proposal_errors({'probe_commands':['echo'], 'repair_mode':'invented',
                           'install_replacement_evidence':['prose'], 'commands':'not-list'})
    assert len(errors)==4
    assert any('use probes' in error for error in errors)


def test_run_explains_raised_as_model_failure_not_native_install():
    name,label,detail=action_explanation({'step':'build_the_environment','outcome':'raised',
        'because':'objective failed (stream_protocol)'})
    assert '环境' in name and '异常' in label and '不是原生安装失败' in detail


def test_model_budget_refusal_is_not_retried(tmp_path):
    class Refused(Reviewer):
        def chat_with_metadata(self, *args, **kwargs):
            self.calls.append(kwargs)
            raise AgentRuntimeClientError('run budget empty',failure_category='run_model_budget')
    client=Refused()
    with pytest.raises(AgentRuntimeClientError):
        review_call(client,instructions='review',payload={},output=tmp_path)
    assert len(client.calls)==1


def test_changed_source_invalidates_review_cache_even_with_same_excerpt(tmp_path):
    client=Reviewer()
    review_call(client,instructions='review',payload={'source':'x','source_sha256':'a'},output=tmp_path)
    review_call(client,instructions='review',payload={'source':'x','source_sha256':'b'},output=tmp_path)
    assert len(client.calls)==3
