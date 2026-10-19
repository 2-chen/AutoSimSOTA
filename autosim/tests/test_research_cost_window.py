"""CPU-only audit fixtures: no provider, budget or native execution."""
from copy import deepcopy
import json

import pytest

from autosim.research.common import atomic_json
from tools.research_cost_window import report, timestamp

START, END = '2026-10-15T08:00:00Z', '2026-10-15T09:00:00Z'


def receipt(at=START, **fields):
    return {'at': at, 'status': 'priced', 'cost_usd': .2,
        'usage': {'input_hit_tokens': 900, 'input_miss_tokens': 100, 'output_tokens': 20},
        'request_profile': {'system_sha256': '1'*64, 'tools_sha256': '2'*64,
            'first_message_sha256': '3'*64, 'max_output_tokens': 20}, **fields}


def write(root, turn, receipts, role='scheduler'):
    atomic_json(root / f'agent/processes/{turn}.json', {'attempt_id': turn, 'role': role})
    atomic_json(root / f'agent/pricing/{turn}.json', {'turn_id': turn, 'gateway': {'receipts': receipts}})


def test_half_open_window_timezone_and_roles_do_not_reprice(tmp_path):
    write(tmp_path, 'a'*32, [receipt(START), receipt('2026-10-15T16:30:00+08:00'),
        receipt(END), receipt('2026-10-15T07:59:59Z')])
    worker = tmp_path / 'agent_workers/worker';write(worker, 'b'*32, [receipt()], 'fix')
    (tmp_path / 'agent/cost_ledger.json').write_text('not readable accounting; do not touch')
    before = (tmp_path / 'agent/cost_ledger.json').read_bytes()
    value = report(tmp_path, START, END)
    assert value['totals']['requests'] == 3
    assert value['totals']['known_cost_usd'] == pytest.approx(.6)
    assert value['roles']['scheduler']['requests'] == 2
    assert value['roles']['fix']['requests'] == 1
    assert value['totals']['cache_hit_ratio'] == .9
    assert value['totals']['requests_at_output_limit'] == 3
    assert value['worker_roots_included'] == 1
    assert value['cost_per_valid_experiment_usd'] is None
    assert (tmp_path / 'agent/cost_ledger.json').read_bytes() == before


@pytest.mark.parametrize('fault', ['exact', 'changed_cost', 'changed_time', 'undated'])
def test_duplicate_turn_index_is_deduped_or_quarantined_before_time_filter(tmp_path, fault):
    first = receipt();second = deepcopy(first)
    if fault == 'changed_cost': second['cost_usd'] = 99
    if fault == 'changed_time': second['at'] = '2026-10-15T07:00:00Z'
    if fault == 'undated': first.pop('at');second.pop('at')
    write(tmp_path, 'a'*32, [first])
    write(tmp_path/'agent_workers/copy', 'a'*32, [second])
    result = report(tmp_path, START, END)
    if fault in {'changed_cost','changed_time'}:
        assert result['totals']['requests'] == 0
        assert result['conflicting_requests_excluded'] == [{'turn_id':'a'*32,'request_index':0}]
    else:
        assert result['duplicate_copies_excluded'] == 1
        assert result['totals']['requests'] == (0 if fault == 'undated' else 1)
        assert result['undated_receipts_unallocated'] == (1 if fault == 'undated' else 0)


def test_identical_requests_inside_one_turn_are_distinct_attempts(tmp_path):
    write(tmp_path, 'a'*32, [receipt(), receipt()])
    assert report(tmp_path, START, END)['totals']['requests'] == 2


@pytest.mark.parametrize('usage', [None, {}, {'input_hit_tokens': True,
    'input_miss_tokens': 0,'output_tokens': 1}, {'input_hit_tokens': 0,
    'input_miss_tokens': -1, 'output_tokens': 3}])
def test_missing_or_invalid_usage_still_preserves_known_cost(tmp_path, usage):
    write(tmp_path, 'a'*32, [receipt(usage=usage)])
    totals = report(tmp_path, START, END)['totals']
    assert totals['requests_without_usage'] == 1
    assert totals['requests_with_usage'] == 0
    assert totals['known_cost_usd'] == .2
    assert totals['cache_hit_ratio'] is None


def test_unknown_usage_holds_and_bad_dates_are_not_free(tmp_path):
    write(tmp_path, 'a'*32, [receipt(status='unknown', cost_usd=None, reserved_usd=4, usage=None),
        receipt(at='naive invalid time', cost_usd=99)])
    value = report(tmp_path, START, END)
    assert value['totals']['requests_without_cost'] == 1
    assert value['totals']['unknown_reserved_usd'] == 4
    assert value['undated_receipts_unallocated'] == 1
    assert value['limitations']


def test_low_hit_rate_and_prefix_churn_are_diagnostics_not_waste_claim(tmp_path):
    cold = receipt(usage={'input_hit_tokens':0,'input_miss_tokens':1000,'output_tokens':4})
    cold['request_profile']['system_sha256'] = '4'*64
    write(tmp_path, 'a'*32, [receipt(), cold])
    totals = report(tmp_path, START, END)['totals']
    assert totals['requests_below_half_hit'] == 1
    assert totals['cost_of_below_half_hit_requests_usd'] == .2
    assert totals['distinct_system_prefixes'] == 2
    assert totals['distinct_tool_schemas'] == 1


@pytest.mark.parametrize('start,end', [('2026-10-15T08:00:00',END), (END,START), (START,START)])
def test_rejects_ambiguous_window(tmp_path,start,end):
    with pytest.raises(ValueError):report(tmp_path,start,end)


def test_unsafe_receipt_not_read_and_no_prompt_or_secret_storage(tmp_path):
    write(tmp_path, 'a'*32, [receipt()])
    path = tmp_path / ('agent/pricing/' + 'a'*32 + '.json')
    path.unlink();target = tmp_path / 'private.json';target.write_text('private sentinel')
    path.symlink_to(target)
    result = report(tmp_path,START,END)
    assert result['skipped_records']
    assert 'private sentinel' not in json.dumps(result)
    unsafe = tmp_path / 'linked';unsafe.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):report(unsafe,START,END)
