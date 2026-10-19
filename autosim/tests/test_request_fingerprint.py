import datetime as dt
import json

from autosim.research.deepseek_gateway import TurnCostGate
from autosim.research.request_fingerprint import request_fingerprint


def test_fingerprint_is_content_free_and_detects_envelope_churn():
    request = {'model': 'deepseek-flash', 'max_tokens': 100,
        'system': [{'type': 'text', 'text': 'private system /secret/path'}],
        'tools': [{'name': 'tool', 'input_schema': {'type': 'object'}}],
        'messages': [{'role': 'user', 'content': 'private dataset excerpt'}],
        'metadata': {'key': 'private-key'}}
    before = json.loads(json.dumps(request))
    profile = request_fingerprint(request, body_bytes=400)
    assert request == before
    assert 'private' not in json.dumps(profile) and '/secret/path' not in json.dumps(profile)
    assert profile['not_provider_cache_measurement']
    assert request_fingerprint(request, body_bytes=400) == profile
    request['messages'].append({'role': 'user', 'content': 'new request'})
    following = request_fingerprint(request, body_bytes=500)
    assert following['system_sha256'] == profile['system_sha256']
    assert following['envelope_sha256'] == profile['envelope_sha256']
    assert following['first_message_sha256'] == profile['first_message_sha256']
    request['system'][0]['text'] += ' changed'
    assert request_fingerprint(request, body_bytes=500)['envelope_sha256'] != profile['envelope_sha256']


def test_profiles_do_not_change_accounting_or_release_unknown_holds():
    at = dt.datetime(2026, 10, 14, 2, tzinfo=dt.timezone.utc)
    profile = request_fingerprint({'model': 'deepseek-flash', 'max_tokens': 100}, body_bytes=100)
    gate = TurnCostGate(limit_usd=1, model='deepseek-flash')
    first = gate.reserve(model='deepseek-flash', max_tokens=100, at=at,
                         input_bytes=100, request_profile=profile)
    gate.settle(first, {'input_tokens': 80, 'cache_read_input_tokens': 20, 'output_tokens': 10})
    second = gate.reserve(model='deepseek-flash', max_tokens=100, at=at,
                          input_bytes=100, request_profile=profile)
    gate.settle(second, None)
    report = gate.snapshot()
    assert report['receipts'][0]['request_profile'] == profile
    assert report['receipts'][1]['request_profile'] == profile
    assert report['receipts'][0]['status'] == 'priced'
    assert report['receipts'][1]['status'] == 'unknown'
    assert report['unknown_reserved_usd'] > 0
    assert report['accounting_ceiling_usd'] > report['cost_usd']
