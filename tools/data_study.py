"""Plan, freeze and inspect FDAS experiment matrices without running experiments.

Bindings are explicit, not inferred from filenames or best scores. Evidence retains
its native status: freezing a plan never promotes development scores to paper results.
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
import statistics
from pathlib import Path

from autosim.research.common import atomic_text, digest, immutable_json, object_digest, read_json
from autosim.research.paper_evidence import build as data_evidence
from autosim.research.receipt_verifier import verify_measurement

PROJECT = Path(__file__).resolve().parents[1]
METHODS = {'fixed_official', 'fixed_continuation', 'random_quality', 'targeted_quality', 'fdas',
           'rule_targeted', 'agent_targeted'}
# Legacy campaign labels are retained for old development matrices. Canonical v2
# methods never alias a rule decision into random or reuse its measurement.
ACQUISITION = {'random_quality': ('random_quality','random'),
               'targeted_quality': ('targeted_quality','unfiltered'),
               'fdas': ('fdas','targeted'), 'rule_targeted': ('rule_targeted',),
               'agent_targeted': ('agent_targeted',)}
PROTOCOL_FIELDS = {'accepted_new_episodes', 'official_data_identity', 'expert_identity',
                   'legal_training_domain', 'common_quality_floor', 'initialization',
                   'training_work', 'data_mixture', 'development_evaluation',
                   'heldout_reservation', 'metric', 'cost_accounting', 'exclusion_rules',
                   'model_configuration', 'resource_manifest', 'native_source_revision', 'resource_budget'}


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', value):
        raise ValueError('expected a short path-safe identifier')
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def regular_json(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('study records must not use symlinks')
    if not path.is_file() or path.stat().st_size > 8 * 1024**2:
        raise ValueError('study record missing or oversized')
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError('study record must be an object')
    return value


def safe_destination(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('study destination must not use symlinks')
    return path


def validate(manifest):
    keys = {'schema_version', 'study_id', 'phase', 'track', 'tasks', 'methods',
            'training_seeds', 'agent_replicas', 'protocol', 'bindings'}
    if (not isinstance(manifest, dict) or set(manifest) != keys
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
        raise ValueError('invalid study schema')
    identifier(manifest['study_id'])
    if manifest['phase'] not in {'draft', 'development', 'formal'}:
        raise ValueError('phase must be draft/development/formal')
    if manifest['track'] not in {'equal_accepted_data', 'equal_acquisition_cost', 'equal_total_compute', 'agent_repeat'}:
        raise ValueError('declare one comparison track; do not conflate data and compute equality')
    tasks = manifest['tasks']
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 64:
        raise ValueError('declare 1..64 tasks')
    ids = []
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {'id', 'repository', 'native_task', 'family', 'role'}:
            raise ValueError('task requires id,repository,native_task,family,role')
        ids.append(identifier(task['id']))
        if any(not isinstance(task[k], str) or not task[k].strip() or len(task[k]) > 300
               for k in ('repository', 'native_task', 'family')):
            raise ValueError('task metadata must be nonempty bounded strings')
        if task['role'] not in {'development', 'task_holdout', 'repository_holdout'}:
            raise ValueError('explicitly distinguish task and repository holdout')
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate task ID')
    methods = manifest['methods']
    if (not isinstance(methods, list) or not methods or any(not isinstance(m, str) or m not in METHODS for m in methods)
            or len(set(methods)) != len(methods)):
        raise ValueError('declare distinct supported methods')
    for field in ('training_seeds', 'agent_replicas'):
        values = manifest[field]
        if (not isinstance(values, list) or not 1 <= len(values) <= 32 or
                any(type(v) is not int or not 0 <= v < 2**32 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(field + ' requires distinct integer IDs')
    protocol = manifest['protocol']
    if not isinstance(protocol, dict) or set(protocol) != PROTOCOL_FIELDS:
        raise ValueError('declare the complete comparison protocol (null allowed only in draft)')
    for key, value in protocol.items():
        if key == 'accepted_new_episodes':
            if value is not None and (type(value) is not int or not 1 <= value <= 1000000):
                raise ValueError('accepted_new_episodes requires an integer or draft null')
        elif value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 4000):
            raise ValueError('protocol.' + key + ' requires bounded text or draft null')
    if manifest['phase'] != 'draft' and any(v is None for v in protocol.values()):
        raise ValueError('non-draft protocol cannot contain unresolved fields')
    if manifest['phase'] == 'formal':
        if ('random_quality' not in methods or
                not {'agent_targeted', 'fdas'} & set(methods)):
            raise ValueError('formal study needs random and an agent_targeted or FDAS treatment')
        if 'agent_targeted' in methods and 'rule_targeted' not in methods:
            raise ValueError('formal agent targeting needs the matched rule_targeted control')
        if (manifest['track'] != 'agent_repeat' and len(manifest['training_seeds']) < 3
                or manifest['track'] == 'agent_repeat' and len(manifest['agent_replicas']) < 3):
            raise ValueError('formal track needs at least three independent replication IDs')
    cells = matrix(manifest)
    bindings = manifest['bindings']
    if not isinstance(bindings, dict) or set(bindings) - set(cells):
        raise ValueError('bindings must refer to declared matrix cells')
    seen = set()
    for cell, binding in bindings.items():
        if not isinstance(binding, dict) or set(binding) != {
                'output', 'repository', 'run_id', 'measurement_label', 'campaign_id'}:
            raise ValueError('binding requires explicit output,repository,run_id,measurement_label,campaign_id')
        for key in ('output', 'repository'):
            if not isinstance(binding[key], str) or not Path(binding[key]).is_absolute():
                raise ValueError('binding paths must be absolute')
        identifier(binding['run_id']); identifier(binding['measurement_label'])
        if binding['campaign_id'] is not None:
            identifier(binding['campaign_id'])
        if cells[cell]['method'] in ACQUISITION and binding['campaign_id'] is None:
            raise ValueError('acquisition methods need an explicit data campaign')
        witness = (str(Path(binding['output']).resolve()), binding['run_id'], binding['measurement_label'])
        if witness in seen:
            raise ValueError('one measurement cannot represent multiple cells')
        seen.add(witness)
    return manifest


def matrix(manifest):
    cells = {}
    for task, method, seed, replica in itertools.product(manifest['tasks'], manifest['methods'],
            manifest['training_seeds'], manifest['agent_replicas']):
        ident = f"{task['id']}--{method}--s{seed}--a{replica}"
        cells[ident] = {'id': ident, 'task_id': task['id'], 'repository': task['repository'],
            'task_family': task['family'], 'role': task['role'], 'method': method,
            'training_seed': seed, 'agent_replica': replica}
    return cells


def harness_identity(project=PROJECT):
    """Freeze executable package, prompts/skills and these reporting tools, not credentials."""
    files = sorted(set((project/'autosim/autosim').rglob('*.py')) |
                   set((project/'autosim/autosim/prompts').glob('*.md')) |
                   set((project/'autosim/autosim/skills').rglob('*.md')) |
                   set((project/'autosim/autosim/skills').glob('*.yaml')) |
                   {project/'autosim/pyproject.toml', project/'tools/data_study.py',
                    project/'tools/export_data_research.py'})
    return object_digest({str(p.relative_to(project)): digest(p) for p in files})


def freeze(manifest, destination, *, project=PROJECT):
    validate(manifest)
    if manifest['phase'] != 'formal' or set(manifest['bindings']) != set(matrix(manifest)):
        raise ValueError('freeze requires a finalized formal protocol and all preallocated cell bindings')
    # No retrospectively relabelling a debug trajectory as a frozen formal trial.
    for binding in manifest['bindings'].values():
        root = safe_destination(binding['output'])
        if root.exists() or root.is_symlink():
            raise ValueError('formal output must be uninitialized at preregistration')
    payload = {'schema_version': 1, 'manifest': manifest,
               'harness_sha256': harness_identity(project),
               'authority': 'preregistered plan, not native capability or result approval'}
    payload['snapshot_id'] = object_digest(payload)
    destination = safe_destination(destination)
    immutable_json(destination/'protocol.json', payload)
    return payload


def inspect(manifest, *, frozen=None, project=PROJECT):
    validate(manifest)
    runtime_match = None
    if frozen is not None:
        unsigned = {k: v for k, v in frozen.items() if k != 'snapshot_id'}
        if object_digest(unsigned) != frozen.get('snapshot_id') or frozen.get('manifest') != manifest:
            raise ValueError('frozen protocol identity differs; preserve old study and use a new version')
        runtime_match = frozen.get('harness_sha256') == harness_identity(project)
    reports, rows, costs = {}, [], {}
    for cell, planned in matrix(manifest).items():
        row = {**planned, 'status': 'pending', 'metric_value': None,
               'formal_paper_eligible': False}
        binding = manifest['bindings'].get(cell)
        if binding is None:
            rows.append(row); continue
        output, repo = Path(binding['output']), Path(binding['repository'])
        key = (str(output.resolve()), str(repo.resolve()), binding['run_id'])
        if (not output.exists() and not output.is_symlink() and
                not any(parent.is_symlink() for parent in output.parents)):
            row['pending_reason'] = 'preallocated output has not been initialized; no run started'
            rows.append(row)
            continue
        row['status'] = 'awaiting_evidence'
        try:
            measured = regular_json(output/'research'/binding['run_id']/'measurements'/
                                    (binding['measurement_label']+'.json'))
            if measured.get('label') != binding['measurement_label'] or measured.get('ok') is not True:
                raise ValueError('bound measurement did not complete a valid score')
            if measured.get('confirmation'):
                raise ValueError('do not relabel held-out measurements as development arms')
            if verify_measurement(output/'research'/binding['run_id'], binding['measurement_label']).get('status') != 'consistent':
                raise ValueError('native measurement receipts/identity are not consistent')
            value = measured.get('metric_value')
            if not finite(value):
                raise ValueError('measurement lacks a finite score')
            if planned['method'] in ACQUISITION:
                if key not in reports:
                    reports[key] = data_evidence(output, repo, run_id=binding['run_id'])
                matches = [r for r in reports[key]['rows'] if r['campaign'] == binding['campaign_id']
                    and r['measurement_label'] == binding['measurement_label']
                    and r['arm'] in ACQUISITION[planned['method']] and r['seed'] == planned['training_seed']]
                if len(matches) != 1:
                    raise ValueError('bound campaign arm, declared seed and actual data consumption did not verify')
                row['data_observation'] = matches[0]
            row.update(status='verified_development', metric_value=value,
                metric={k: (measured.get('metric') or {}).get(k) for k in ('name','unit','direction','aggregation')},
                evidence_sha256=digest(output/'research'/binding['run_id']/'measurements'/
                                       (binding['measurement_label']+'.json')),
                seed_status='declared_id_only; actual seed/source/work need independent audit')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            row.update(status='evidence_unavailable', why=str(exc)[:500])
        cost_key = str(output.resolve())
        if output.exists() and cost_key not in costs:
            from autosim.research.paper_evidence import _costs
            try:
                costs[cost_key] = _costs(output)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                costs[cost_key] = {'status':'unavailable', 'why':type(exc).__name__,
                              'scope':'whole run; unknown is not zero'}
        rows.append(row)
    summaries = []
    for task in manifest['tasks']:
        for method, replica in itertools.product(manifest['methods'], manifest['agent_replicas']):
            group = [r for r in rows if r['task_id'] == task['id'] and r['method'] == method
                     and r['agent_replica'] == replica]
            valid = [r for r in group if r['status'] == 'verified_development']
            semantics = {object_digest(r['metric']) for r in valid}
            values = [r['metric_value'] for r in valid] if len(semantics) <= 1 else []
            summaries.append({'task_id':task['id'], 'method':method, 'agent_replica':replica,
                'planned':len(group), 'verified':len(valid), 'complete':len(valid)==len(group) and len(semantics)==1,
                'mean':statistics.mean(values) if values else None,
                'sample_std':statistics.stdev(values) if len(values)>1 else None,
                'aggregation':'descriptive policy-seed summary within one agent replica; actual seed use still needs audit',
                'metric_semantics_match':len(semantics)<=1})
    result = {'schema_version':1, 'study_id':manifest['study_id'], 'phase':manifest['phase'],
        'manifest_sha256':object_digest(manifest),
        'track':manifest['track'], 'rows':rows, 'summaries':summaries,
        'costs':[{'output':key, **value} for key,value in costs.items()],
        'current_harness_matches_freeze':runtime_match, 'formal_paper_eligible':False,
        'missing_protocol_fields':[k for k,v in manifest['protocol'].items() if v is None],
        'formal_gaps':['independent native seed/work and method-fairness audit',
            'held-out outcomes for every prespecified arm, not only the selected best',
            'per-arm cost attribution including failed acquisition and shared initialization',
            'independent system replication and anonymous release audit'],
        'authority':'read-only experiment inventory; no best-seed selection or formal approval'}
    result['design'] = design_view(manifest)
    result['snapshot_id'] = object_digest(result)
    return result


def design_view(manifest):
    """Describe the actual planned contrasts, not scientific validity or execution.

    Acquisition and selection are distinct questions. In particular, an acquisition
    study must not pretend to test Qt or acquire a mandatory extra filtering arm.
    """
    methods = set(manifest['methods'])
    contrasts = []
    definitions = (
        ('targeting_vs_random', 'random_quality', 'agent_targeted', '失败定向获取相对随机获取'),
        ('reasoning_vs_rule', 'rule_targeted', 'agent_targeted', 'Agent目标选择相对固定规则'),
        ('selection_increment', 'agent_targeted', 'fdas', '共同目标获取之上的Qt筛选增量'),
        ('selection_increment_legacy', 'targeted_quality', 'fdas', '旧标记的共同目标池Qt增量'),
        ('legacy_joint_vs_random', 'random_quality', 'fdas', '联合方法相对随机；不能单独归因目标或筛选'),
    )
    for identity, left, right, question in definitions:
        if {left, right} <= methods:
            contrasts.append({'id': identity, 'control': left, 'treatment': right,
                'question': question, 'status': 'planned_not_verified',
                'needs_shared_generator_quality_initialization_work_and_evaluation': True})
    return {'planned_cells': len(matrix(manifest)), 'task_candidates': len(manifest['tasks']),
        'training_seed_ids': len(manifest['training_seeds']),
        'agent_replica_ids': len(manifest['agent_replicas']),
        'independent_units_not_established_by_ids': True,
        'planned_contrasts': contrasts,
        'selection_increment_planned': any(row['id'].startswith('selection_increment') for row in contrasts),
        'equal_total_compute_not_implied': manifest['track'] != 'equal_total_compute',
        'unbound_cells': len(matrix(manifest)) - len(manifest['bindings']),
        'not_a_budget_grant_or_task_capability_check': True}


def markdown(report):
    def cell(value):
        return str(value if value is not None else '待执行').replace('|','\\|').replace('\n',' ')
    labels = {'pending':'待安排', 'awaiting_evidence':'等待证据',
              'evidence_unavailable':'证据未通过', 'verified_development':'已核验开发分数'}
    design = report.get('design') or {}
    if 'selection_increment_planned' not in design:
        selection_note = '旧快照未记录对照问题元数据；请查原协议，不能推断已检验Qt。'
    else:
        selection_note = ('本矩阵包含Qt增量计划，仍须实际执行与公平性核验。' if
                          design['selection_increment_planned'] else
                          '本矩阵不检验Qt筛选增量；不能据此声称获取与筛选均已有效。')
    lines = ['# FDAS 实验矩阵', '', f"研究：{report['study_id']}；轨道：{report['track']}；阶段：{report['phase']}", '',
        '预登记计划不等于正式结果；缺失不填零，不删除失败单元，不挑最好种子。', '',
        '## 计划中的研究问题（不是已完成结论）', '',
        *['- '+row['question']+'：'+row['control']+' → '+row['treatment']
          for row in design.get('planned_contrasts', [])],
        '', selection_note,
        '本轨道不自动等于等总计算；候选任务、seed ID和计划单元均不代表资源授权或独立重复已完成。', '',
        '## 逐单元状态', '',
        '| 任务 | 方法 | 策略 seed | Agent 重复 | 状态 | 开发指标 |',
        '| --- | --- | ---: | ---: | --- | ---: |']
    for row in report['rows']:
        lines.append('| '+' | '.join(cell(v) for v in (row['task_id'],row['method'],
            row['training_seed'],row['agent_replica'],labels[row['status']],row['metric_value']))+' |')
    lines += ['', '## 汇总（含完整性分母）', '',
        '| 任务 | 方法 | Agent 重复 | 已核验 / 计划 | 均值 | 样本标准差 | 全部完成 |',
        '| --- | --- | ---: | --- | ---: | ---: | --- |']
    for row in report['summaries']:
        lines.append('| '+' | '.join(cell(v) for v in (row['task_id'],row['method'],row['agent_replica'],
            f"{row['verified']}/{row['planned']}",row['mean'],row['sample_std'],row['complete']))+' |')
    lines += ['', '策略训练种子与完整 Agent 重复是两种实验单位；上述统计只描述观测，不能证明独立性或因果收益。',
        '', '## 当前断边', '', *['- '+cell(row['id'])+'：'+cell(row['why'])
            for row in report['rows'] if row.get('why')],
        *['- 未定协议：'+key for key in report['missing_protocol_fields']],
        *['- 正式准入待验：'+line for line in report['formal_gaps']], '',
        '费用按原 run 去重展示在 JSON 中；不能按方法数量平均分摊，不能把未知预留当零。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'freeze'])
    parser.add_argument('manifest', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args(argv)
    manifest = regular_json(args.manifest)
    if args.action == 'freeze':
        freeze(manifest, args.destination)
        print(args.destination/'protocol.json')
    else:
        report = inspect(manifest, frozen=regular_json(args.frozen) if args.frozen else None)
        root = safe_destination(args.destination)/report['snapshot_id']
        immutable_json(root/'analysis.json', report)
        atomic_text(root/'MATRIX.md', markdown(report))
        print(root/'MATRIX.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
