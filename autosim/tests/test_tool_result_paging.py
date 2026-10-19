import json
from types import SimpleNamespace

from autosim.research.agent_runtime import handle_mcp_message


def test_mcp_read_can_fully_recover_paged_tool_text_without_recursive_paging(tmp_path):
    from autosim.research.evidence_store import read_attempt_evidence
    body = 'source text line\n' * 8000
    executor = SimpleNamespace(output=tmp_path, allow_commands=False, allow_native=True,
        native_probe=lambda _: {'status': 'completed', 'stdout': body, 'returncode': 0,
            'attempt_id': 'native-original'},
        read_evidence=lambda args: read_attempt_evidence(tmp_path, **args))
    result = handle_mcp_message({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': 'inspect_native_environment', 'arguments': {}}}, executor)
    assert not result['result']['isError']
    surfaced = json.loads(result['result']['content'][0]['text'])
    page = surfaced['stdout']
    assert page['context_paged'] and surfaced['attempt_id'] == 'native-original'
    chunks, offset = [], 0
    while True:
        result = handle_mcp_message({'id': 2, 'method': 'tools/call',
            'params': {'name': 'read_evidence', 'arguments': {
                'evidence_id': page['evidence_id'], 'offset': offset}}}, executor)
        packet = json.loads(result['result']['content'][0]['text'])
        assert isinstance(packet['text'], str) and 'context_paged' not in packet
        chunks.append(packet['text'])
        offset = packet['next_offset']
        if offset == packet['log_bytes']:
            break
    assert json.loads(''.join(chunks)) == body
