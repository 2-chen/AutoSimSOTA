"""Read-only inventory and prospective freeze for SYSTEM research comparisons.

This is not a scheduler. It never starts runs, chooses a candidate, grants budget,
reads held-out outcomes, or upgrades development observations to formal results.
Data-arm matrices remain in data_study.py and are not a prerequisite here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from autosim.research.common import digest, immutable_json, object_digest
from tools.data_study import (PROJECT, harness_identity, identifier, regular_json,
                              safe_destination)

PROTOCOL_FIELDS = {
    'resource_package', 'allowed_interventions', 'shared_resources_and_tools',
    'baseline_calibration', 'native_development_evaluation', 'final_confirmation',
    'initialization', 'stopping_and_failure_rules', 'memory_isolation',
    'interventions_and_repairs', 'full_cost_accounting', 'comparison_scope',
}
WORKFLOWS = {'expert_demonstrations', 'fixed_data', 'online_learning', 'source_controller'}
SYSTEMS = {'original_recipe', 'generic_research', 'autosimsota',
           'autoresearch_robotics', 'ablation'}


def cells(manifest):
    return {f"{t['id']}--{s['id']}--r{r}": {
        'task_id': t['id'], 'system_id': s['id'], 'research_replica': r,
        'workflow': t['workflow'], 'role': t['role'],
    } for t, s, r in itertools.product(manifest['tasks'], manifest['systems'],
                                     manifest['research_replicas'])}


def _text(value, *, unresolved=False):
    if value is None and unresolved:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 8000:
        raise ValueError('expected bounded nonempty text, or draft null')


def validate(manifest):
    required = {'schema_version', 'study_id', 'phase', 'tasks', 'systems',
                'research_replicas', 'protocol', 'bindings'}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
        raise ValueError('invalid system study schema')
    identifier(manifest['study_id'])
    if manifest['phase'] not in {'draft', 'development', 'formal'}:
        raise ValueError('unknown study phase')
    draft = manifest['phase'] == 'draft'
    for field, maximum in [('tasks', 64), ('systems', 16)]:
        values = manifest[field]
        if not isinstance(values, list) or not 1 <= len(values) <= maximum:
            raise ValueError('expected bounded nonempty ' + field)
        ids = []
        for row in values:
            keys = ({'id', 'repository', 'native_task', 'workflow', 'role', 'admission'}
                    if field == 'tasks' else
                    {'id', 'kind', 'model_configuration', 'capability_manifest'})
            if not isinstance(row, dict) or set(row) != keys:
                raise ValueError('invalid ' + field + ' record')
            ids.append(identifier(row['id']))
            if field == 'tasks':
                _text(row['repository']); _text(row['native_task'], unresolved=draft)
                if row['workflow'] not in WORKFLOWS or row['role'] not in {
                        'development', 'task_transfer', 'repository_holdout'}:
                    raise ValueError('invalid task workflow or transfer claim')
                # Admission is a separately audited resource/native capability report,
                # not a boolean that this matrix can infer from a repository name.
                _text(row['admission'], unresolved=draft)
            else:
                if row['kind'] not in SYSTEMS:
                    raise ValueError('unknown system kind')
                _text(row['model_configuration'], unresolved=draft)
                _text(row['capability_manifest'], unresolved=draft)
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate ' + field + ' IDs')
    replicas = manifest['research_replicas']
    if (not isinstance(replicas, list) or not 1 <= len(replicas) <= 32
            or any(type(x) is not int or not 0 <= x < 2**32 for x in replicas)
            or len(replicas) != len(set(replicas))):
        raise ValueError('distinct research repetition IDs required; episodes are not repetitions')
    protocol = manifest['protocol']
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_FIELDS:
        raise ValueError('complete system comparison protocol required')
    for value in protocol.values():
        _text(value, unresolved=draft)
    if manifest['phase'] == 'formal':
        kinds = {s['kind'] for s in manifest['systems']}
        if not {'original_recipe', 'generic_research', 'autosimsota'} <= kinds or len(replicas) < 3:
            raise ValueError('formal comparison needs original/generic/full systems and >=3 repetitions')
    matrix = cells(manifest)
    bindings = manifest['bindings']
    if not isinstance(bindings, dict) or set(bindings) - set(matrix):
        raise ValueError('bindings must refer to planned system cells')
    outputs = set()
    for binding in bindings.values():
        if not isinstance(binding, dict) or set(binding) != {'output', 'repository', 'run_id'}:
            raise ValueError('explicit output, isolated repository and run_id required')
        identifier(binding['run_id'])
        for key in ('output', 'repository'):
            if not isinstance(binding[key], str) or not Path(binding[key]).is_absolute():
                raise ValueError('absolute binding paths required')
            safe_destination(binding[key])
        output = safe_destination(binding['output'])
        repository = Path(binding['repository']).resolve()
        if repository == output.resolve() or not repository.is_relative_to(output.resolve()):
            raise ValueError('isolated repository must be nested in its trial output')
        resolved = output.resolve()
        if any(resolved.is_relative_to(other) or other.is_relative_to(resolved) for other in outputs):
            raise ValueError('research repetition outputs must be distinct and not nested')
        outputs.add(resolved)
    return manifest


def identity(project=PROJECT):
    return object_digest({'harness': harness_identity(project),
                         'system_study_tool': digest(Path(__file__)),
                         'protocol_notes': digest(PROJECT / 'docs/paper/SYSTEM_STUDY_PROTOCOL.md')})


def freeze(manifest, destination, *, project=PROJECT):
    validate(manifest)
    if manifest['phase'] != 'formal' or set(manifest['bindings']) != set(cells(manifest)):
        raise ValueError('formal prospective freeze requires all preallocated bindings')
    destination = safe_destination(destination)
    for binding in manifest['bindings'].values():
        output = safe_destination(binding['output'])
        if output.exists() or output.is_symlink():
            raise ValueError('cannot relabel an initialized development run as a frozen trial')
        if destination.resolve().is_relative_to(output.resolve()):
            raise ValueError('freeze destination must not initialize a trial output')
    result = {'schema_version': 1, 'manifest': manifest, 'implementation_sha256': identity(project),
              'authority': 'prospective comparison plan; not capability, budget or result approval'}
    result['snapshot_id'] = object_digest(result)
    immutable_json(destination / 'protocol.json', result)
    return result


def inspect(manifest, *, frozen=None, project=PROJECT):
    validate(manifest)
    matches = None
    if frozen is not None:
        if (frozen.get('snapshot_id') != object_digest({k: v for k, v in frozen.items() if k != 'snapshot_id'})
                or frozen.get('manifest') != manifest):
            raise ValueError('frozen plan changed; preserve the old version')
        matches = frozen.get('implementation_sha256') == identity(project)
    rows = []
    for ident, planned in cells(manifest).items():
        row = {'id': ident, **planned, 'status': 'unallocated', 'formal_paper_eligible': False}
        binding = manifest['bindings'].get(ident)
        if binding:
            root = safe_destination(binding['output'])
            row['status'] = 'initialized_unreviewed' if root.exists() else 'not_started'
            # Deliberately no scans for "best", no held-out reads, no inferred zeros.
            row['output'] = str(root)
        rows.append(row)
    return {'schema_version': 1, 'study_id': manifest['study_id'], 'phase': manifest['phase'],
            'planned_cells': len(rows), 'rows': rows,
            'unresolved_protocol_fields': [k for k, v in manifest['protocol'].items() if v is None],
            'unadmitted_tasks': [t['id'] for t in manifest['tasks'] if t['admission'] is None],
            'implementation_matches_freeze': matches, 'formal_paper_eligible': False,
            'pending_result_audit': ['native baseline calibration', 'independent research start and seed audit',
                'actual policy/controller loading, rollout and metrics', 'per-trial final selection and confirmation',
                'all failures, full costs, interventions and scientific fairness'],
            'authority': 'read-only coverage inventory; no execution or scientific approval'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inspect', 'freeze'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args(argv)
    manifest = regular_json(args.manifest)
    if args.command == 'freeze':
        if args.destination is None:
            parser.error('freeze requires --destination; it does not launch or fund runs')
        result = freeze(manifest, args.destination)
    else:
        result = inspect(manifest, frozen=regular_json(args.frozen) if args.frozen else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
