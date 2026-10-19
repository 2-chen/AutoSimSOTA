import json
from pathlib import Path

import pytest

from autosim.research.evidence_store import (capture_attempt_evidence,
                                              read_attempt_evidence)
from autosim.research.agent_runtime import _McpExecutor, handle_mcp_message


def test_capture_and_read_evidence_with_stable_id(tmp_path):
    output = tmp_path / "run"
    log = output / "research" / "derived" / "evaluate" / "attempts" / ("a" * 32) / "output.log"
    log.parent.mkdir(parents=True)
    log.write_text("[error] missing policy\n", encoding="utf-8")
    row = capture_attempt_evidence(
        output, attempt_id="a" * 32, log=log,
        receipt_ref="research/derived/attempts/" + "a" * 32 + "/receipt.json",
        status="completed", returncode=0, termination_reason="normal_exit",
        metric_artifact={"status": "missing"})
    assert row["evidence_ref"] == "evidence/" + "a" * 32 + ".json"
    found = read_attempt_evidence(output, "a" * 32, limit=8)
    assert found["text"] == "[error] "
    assert found["metric_artifact"]["status"] == "missing"
    log.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        read_attempt_evidence(output, "a" * 32)


def test_read_only_mcp_can_read_evidence_but_not_execute(tmp_path):
    output = tmp_path / "run"
    checkout = output / "checkout"
    checkout.mkdir(parents=True)
    (output / "agent_fixture.json").write_text(
        json.dumps({"workspace": str(checkout)}), encoding="utf-8")
    log = output / "research" / "derived" / "train" / "output.log"
    log.parent.mkdir(parents=True)
    log.write_text("ImportError: dependency missing", encoding="utf-8")
    capture_attempt_evidence(
        output, attempt_id="b" * 32, log=log, receipt_ref="receipt.json",
        status="failed", returncode=1, termination_reason="nonzero_exit")
    executor = _McpExecutor(workspace=checkout, output=output, allow_commands=False)
    names = [row["name"] for row in handle_mcp_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        executor)["result"]["tools"]]
    assert set(names) == {"read_evidence", "search_public_sources", "read_public_source",
                          "inspect_workspace_resources"}
    answer = handle_mcp_message({"jsonrpc": "2.0", "id": 2,
                                 "method": "tools/call", "params": {
                                     "name": "read_evidence", "arguments": {
                                         "evidence_id": "b" * 32}}}, executor)
    assert "ImportError" in answer["result"]["content"][0]["text"]
    forbidden = handle_mcp_message({"jsonrpc": "2.0", "id": 3,
                                    "method": "tools/call", "params": {
                                        "name": "run_command", "arguments": {
                                            "argv": ["true"]}}}, executor)
    assert forbidden["result"]["isError"] is True


def test_evidence_rejects_external_log(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    external = tmp_path / "private.log"
    external.write_text("private", encoding="utf-8")
    with pytest.raises(ValueError, match="inside this run"):
        capture_attempt_evidence(output, attempt_id="c" * 32, log=external,
                                 receipt_ref="receipt.json", status="failed",
                                 returncode=1, termination_reason="nonzero_exit")


def test_large_native_log_streams_and_verifies_unpaged_bytes(tmp_path, monkeypatch):
    from autosim.research.common import read_json
    output=tmp_path/'run';output.mkdir()
    log=output/'output.log';size=64*1024*1024+64
    tail=b'completed native updates before timeout\n'
    with log.open('wb') as stream:
        stream.truncate(size)
        stream.seek(size-len(tail));stream.write(tail)
    monkeypatch.setattr(Path,'read_bytes',lambda _: pytest.fail('whole-file read forbidden'))
    row=capture_attempt_evidence(output,attempt_id='d'*32,log=log,
        receipt_ref='receipt.json',status='timed_out',returncode=None,
        termination_reason='timeout')
    assert tail.decode().strip() in row['excerpt']
    assert read_json(output/row['evidence_ref'])['log_bytes']==size
    found=read_attempt_evidence(output,'d'*32,offset=size-len(tail),limit=len(tail))
    assert found['text']==tail.decode() and found['next_offset']==size
    assert found['status']=='timed_out' and found['termination_reason']=='timeout'
    with log.open('r+b') as stream:
        stream.seek(0);stream.write(b'X')
    with pytest.raises(ValueError,match='changed'):
        read_attempt_evidence(output,'d'*32,offset=size-len(tail),limit=len(tail))


def test_disk_log_bound_is_still_enforced(tmp_path,monkeypatch):
    import autosim.research.evidence_store as store
    output=tmp_path/'run';output.mkdir();log=output/'output.log'
    log.write_bytes(b'x'*65);monkeypatch.setattr(store,'_MAX_LOG',64)
    with pytest.raises(ValueError,match='store limit'):
        capture_attempt_evidence(output,attempt_id='e'*32,log=log,
            receipt_ref='receipt.json',status='failed',returncode=1,
            termination_reason='nonzero_exit')
    assert not (output/'evidence').exists()


def test_log_mutation_during_hash_refuses_capture(tmp_path,monkeypatch):
    import autosim.research.evidence_store as store
    original=store.hashlib.sha256
    output=tmp_path/'run';output.mkdir();log=output/'output.log';log.write_bytes(b'original')
    class ChangingHash:
        def __init__(self):self.inner=original()
        def update(self,chunk):
            self.inner.update(chunk);log.write_bytes(b'changed!')
        def hexdigest(self):return self.inner.hexdigest()
    monkeypatch.setattr(store.hashlib,'sha256',ChangingHash)
    with pytest.raises(ValueError,match='changed while reading'):
        capture_attempt_evidence(output,attempt_id='f'*32,log=log,
            receipt_ref='receipt.json',status='failed',returncode=1,
            termination_reason='nonzero_exit')
    assert not (output/'evidence').exists()
