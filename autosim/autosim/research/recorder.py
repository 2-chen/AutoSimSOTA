"""Evidence-bound presentation, optional read-only narration, and demo requests.

No scoring, shell execution, or simulation happens here. The Scheduler alone resolves
requests through the existing native executor. Heartbeats never call the model.
"""

from __future__ import annotations

import html
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .common import atomic_json, atomic_text, now, object_digest, read_json, sanitize_model_payload, digest
from .prompt_library import load_prompt

START = "<!-- AUTOSIM_PRESENTATION_START -->"
END = "<!-- AUTOSIM_PRESENTATION_END -->"
NATIVE_START = "<!-- AUTOSIM_NATIVE_TRIALS_START -->"
NATIVE_END = "<!-- AUTOSIM_NATIVE_TRIALS_END -->"
ACTIVE_START = "<!-- AUTOSIM_ACTIVE_CANDIDATE_START -->"
ACTIVE_END = "<!-- AUTOSIM_ACTIVE_CANDIDATE_END -->"
NARRATIVE_SCOPE_START = '<!-- AUTOSIM_NARRATIVE_SCOPE_START -->'
NARRATIVE_SCOPE_END = '<!-- AUTOSIM_NARRATIVE_SCOPE_END -->'
FIELDS = ("current", "purpose", "finding", "blocker", "next")
CHARTS = {"comparison", "training", "cost"}
DEFAULT_TIMEOUT = 180
FACT_REVIEW_CONTRACT_VERSION = 7
SYSTEM = load_prompt("recorder")


def read(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    try:
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or
                path.stat().st_size > 8 * 1024**2):
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, RuntimeError):
        return {}


def _narrative_scope(narrative, status, current_action, root):
    saved = narrative.get('observed_run_scope')
    if not isinstance(saved, dict):
        revision = narrative.get('revision')
        source = read(root, f'report/snapshots/{revision}.json') if isinstance(revision,str) and re.fullmatch(r'[0-9a-f]{64}',revision) else {}
        saved = {key:source.get(key) for key in ('status','current_action')}
    stale = any(saved.get(key) != value for key,value in
                (('status',status),('current_action',current_action)))
    message = ('状态或动作已改变：以下是旧快照的研究说明，其中的停止和阻碍描述不是当前状态；实时进度以页首为准。'
               if stale else '以下说明依据写作时的快照；实时进度见页首。')
    return '\n'.join([NARRATIVE_SCOPE_START, message,
        f"说明写于 {cell(narrative.get('written_at'))}；快照状态 {cell(saved.get('status') or '未知')}，"
        f"动作 {cell(saved.get('current_action') or '未知')}。", NARRATIVE_SCOPE_END])


def refresh_narrative_scope(content, root, *, status, current_action):
    """Mark old prose as historical using live facts; no model or score changes."""
    if NARRATIVE_SCOPE_START not in content or NARRATIVE_SCOPE_END not in content:
        return content
    narrative = read(Path(root), 'report/narrative.json')
    start, rest = content.split(NARRATIVE_SCOPE_START,1)
    _, end = rest.split(NARRATIVE_SCOPE_END,1)
    return start + _narrative_scope(narrative,status,current_action,Path(root)) + end


def active_candidate(root: Path, run_id: str) -> dict[str, Any]:
    """Bounded controller declarations, not progress or measurement evidence."""
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        return {}
    relative = f"research/{run_id}/controller_session.json"
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        return {}
    session = read(root, relative)
    pending = session.get("pending_action")
    if (session.get("run_id") != run_id or session.get("status") != "running"
            or not isinstance(pending, dict)):
        return {}
    index = pending.get("round")
    if (type(index) is not int or index < 1 or session.get("action") != f"round_{index}"):
        return {}
    idea = pending.get("idea") or {}
    if not isinstance(idea, dict) or not isinstance(idea.get("label"), str):
        return {}
    changed = pending.get("changed", {})
    if not isinstance(changed, dict):
        return {}
    paths = changed.get("files") or [changed.get("file")]
    if not isinstance(paths, list):
        return {}
    files = [p for p in paths[:8] if isinstance(p, str)
             and re.fullmatch(r"[A-Za-z0-9_./-]{1,200}", p)
             and not Path(p).is_absolute() and ".." not in Path(p).parts]
    settings = pending.get("settings") or {}
    settings = {key: value for key, value in list(settings.items())[:16]
                if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key)
                and (finite(value) or isinstance(value, bool))} if isinstance(settings, dict) else {}
    mechanism = str(idea.get("mechanism") or "")
    reason = str(idea.get("why") or "")
    return {"round": index, "label": idea["label"][:200],
            "phase": str(pending.get("phase") or "unknown")[:80], "files": files,
            "proposed_hypothesis": {
                "mechanism": mechanism[:1000], "mechanism_truncated": len(mechanism) > 1000,
                "reason": reason[:700], "reason_truncated": len(reason) > 700,
                "authority": "Agent proposal; not verified execution or outcome"},
            "requested_settings": settings,
            "settings_scope": "numeric subset; full configuration in controller record",
            "frozen_policy_requested": isinstance(pending.get("frozen_policy_request"), dict),
            "source_ref": relative, "authority": "controller_declaration_not_completion"}


def active_candidate_panel(candidate: dict[str, Any]) -> str:
    lines = [ACTIVE_START]
    if candidate:
        phase = {"measurement_starting": "已提交实验；执行与计分状态见页首和实验表",
                 "change_starting": "正在应用候选修改",
                 "collection_starting": "正在准备候选数据采集"}.get(candidate["phase"], "候选处理中")
        files = "、".join(candidate["files"]) or "未记录源码修改"
        settings = "；".join(f"{key}={value}" for key, value in candidate["requested_settings"].items()) or "未展示"
        frozen = candidate["frozen_policy_requested"]
        mode = ("已请求冻结权重复用，本轮仅评测，不安排新训练；实际加载仍待核验" if frozen else
                "未请求冻结权重复用，不能声称权重保持不变")
        settings_caption = "继承的原训练设置（不是本轮新增训练量）" if frozen else "请求的数值设置"
        lines += ["### 本轮正在验证什么", "",
                  f"第 {candidate['round']} 轮：{cell(candidate['label'])}。{phase}。", "",
                  f"改动位置：{cell(files, 500)}。{settings_caption}：{cell(settings, 500)}；完整配置见提交记录。", "",
                  f"{mode}。{'继承设置不代表本轮发生训练' if frozen else '设置是请求量，不是已完成步数'}；执行状态见页首。", "",
                  f"[候选提交记录]({candidate['source_ref']})", ""]
    return "\n".join(lines + [ACTIVE_END])


def refresh_active_candidate_panel(content: str, root: Path, *, status: str = "",
                                   destination_root: Path | None = None) -> str:
    """Advance a small fact panel without an LLM call or tree scan."""
    if START not in content or END not in content:
        return content
    # Native heartbeats run from research/<run_id>, including when their destination
    # is the outer RUN.md. Controller declarations live under the owning run root.
    owner = root
    if (root.parent.name == "research" and
            re.fullmatch(r"[A-Za-z0-9_-]{1,100}", root.name) and
            (root / "controller_session.json").is_file()):
        owner = root.parent.parent
    state = read(owner, "run_state.json")
    candidate = (active_candidate(owner, state.get("run_id"))
                 if state.get("status") == "running" and status in {"", "running"} else {})
    if (candidate and destination_root is not None and
            destination_root.resolve() == (owner / "research" / state["run_id"]).resolve()):
        candidate = {**candidate, "source_ref": "controller_session.json"}
    panel = active_candidate_panel(candidate)
    prefix, rest = content.split(START, 1)
    body, suffix = rest.split(END, 1)
    if ACTIVE_START in body and ACTIVE_END in body:
        before, rest = body.split(ACTIVE_START, 1)
        _, after = rest.split(ACTIVE_END, 1)
        body = before + panel + after
    elif candidate:
        body = "\n" + panel + "\n" + body
    return prefix + START + body + END + suffix


def native_table(native: list[dict]) -> str:
    """A fact-only bounded panel shared by snapshots and live telemetry."""
    lines = [NATIVE_START, '### 最近的原生试跑（不计分）', '',
        '这里记录命令验收，不是正式 baseline/candidate。退出成功也不等于策略提升；完成进度、加载身份和 rollout 仍需分别核验。', '',
        '| 阶段/尝试 | 实际结果 | 用时 | 原生错误或输出摘要 | 证据 |',
        '| --- | --- | --- | --- | --- |']
    for attempt in native:
        state = {'completed':'命令退出成功','failed':'命令失败','interrupted':'已中断',
            'timed_out':'达到单次期限','running_unverified':'运行记录待核验'}.get(attempt['status'], attempt['status'])
        seconds = attempt.get('seconds')
        duration = f'{seconds:.1f} 秒' if finite(seconds) else '未结算'
        live = attempt.get('live_telemetry') or {}
        if not finite(seconds) and finite(attempt.get('elapsed_wall_seconds')):
            duration = f"约 {attempt['elapsed_wall_seconds']:.0f} 秒（未结算）"
        plain = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', str(
            attempt.get('excerpt') or live.get('tail') or ''))
        plain = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', plain)
        text = [line.strip() for line in plain.splitlines() if line.strip()]
        excerpt = ' / '.join(text[-3:]) if text else '尚无封存结果'
        if live and not attempt.get('evidence_verified'):
            excerpt = '日志末尾（未封存）：' + excerpt
        ref = str(attempt.get('evidence_ref') or attempt['receipt_ref'])
        if not re.fullmatch(r'[A-Za-z0-9_./-]+',ref) or '..' in Path(ref).parts or Path(ref).is_absolute():
            continue
        label = attempt.get('stage') or '历史试跑'
        lines.append(f"| {cell(label)} / {cell(attempt['id'][:10])} | {cell(state)} | {duration} | {cell(excerpt,240)} | [查看]({ref}) |")
    return '\n'.join(lines + ['', NATIVE_END])


def refresh_native_panel(content: str, root: Path) -> str:
    """Replace only the native panel, leaving narration and measurements untouched."""
    from .stage_verification import recent_attempts
    if START not in content or END not in content:
        return content
    prefix, rest = content.split(START, 1)
    body, suffix = rest.split(END, 1)
    native = recent_attempts(root)
    if not native:
        return content
    panel = native_table(native)
    if NATIVE_START in body and NATIVE_END in body:
        before, remaining = body.split(NATIVE_START, 1)
        _, after = remaining.split(NATIVE_END, 1)
        body = before + panel + after
    elif '### 最近的原生试跑（不计分）' in body:
        before, remaining = body.split('### 最近的原生试跑（不计分）', 1)
        position = remaining.find('\n### ')
        after = remaining[position:] if position >= 0 else ''
        body = before + panel + '\n' + after
    else:
        body += '\n\n' + panel + '\n'
    return prefix + START + body + END + suffix


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cell(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value if value is not None else "未记录").split())[:limit]
    return html.escape(text, quote=False).replace("|", "&#124;").replace("`", "&#96;").replace("[", "&#91;").replace("]", "&#93;").replace("*", "&#42;").replace("_", "&#95;")


def number(value: Any) -> str:
    return f"{value:.6g}" if finite(value) else "未测得"


def cost_points(rows):
    """Display each receipt-bound native attempt once, never charge reuse twice."""
    seen=set()
    points=[]
    for row in rows:
        identities=row.get('stage_attempt_ids') or {}
        for stage,value in (row.get('stage_seconds') or {}).items():
            if not finite(value) or value<0:
                continue
            attempt=identities.get(stage)
            key=(stage,attempt) if isinstance(attempt,str) and re.fullmatch(r'[0-9a-f]{32}',attempt) else None
            if key is not None and key in seen:
                continue
            if key is not None:
                seen.add(key)
            points.append((row['label']+'/'+stage,value))
    return points


def operation_name(step):
    """Translate known lifecycle operations; retain unknown names without guessing."""
    names = {"build_the_environment": "准备／恢复运行环境", "research_task": "专项调查",
            'choose':'主智能体决策', 'update_research_plan':'保存研究计划',
            'register_data_version':'登记数据版本', 'register_data_campaign':'登记数据对照',
            'record_data_campaign_arm':'记录数据对照结果', 'submit_native_job':'提交原生作业',
            'review_report_demo':'审核并展示真实demo', 'configure_confirmation_protocol':'设定独立评测条件',
            'read_the_checkout':'调查仓库能力', 'declare':'登记任务与评测协议',
            'derive_a_command':'推导并验收原生命令',
            'submit_research_task':'提交专项调查', 'poll_research_task':'检查专项调查进度',
            'poll_native_job':'检查原生作业进度', 'verify_metric':'核验原生指标',
            'confirm_best':'独立确认最佳候选', 'retry_failed_action':'复验失败操作',
            'train':'执行训练', 'evaluate':'执行原生评测', 'collect':'采集训练数据',
            'prepare_data':'处理训练数据', 'record_demo':'录制仿真demo',
            "reconcile_interrupted_action": "核对中断操作", "run_the_loop": "训练与评测",
            'correct_policy_handoff':'恢复已训练产物并评测',
            'recover_unscored_baseline':'恢复原训练产物并验收基线',
            'diagnose_development_failure':'Fix 诊断开发评测交接故障',
            'extend_research_rounds':'分配后续研究轮次',
            "wait_for_jobs": "等待后台任务"}
    if isinstance(step, str) and step.startswith('derive:'):
        stage = step.partition(':')[2]
        return '验收原生入口：' + names.get(stage, stage)
    return names.get(step, step or "尚无记录")


def action_explanation(row):
    name = operation_name(row.get('step'))
    outcome = row.get("outcome") or "未记录"
    label = {"raised": "操作抛出异常，未正常完成", "repair proposal rejected": "修复提案未通过校验",
             "checkpoint failure": "准备操作返回了失败证据，环境未就绪",
             "checkpoint": "准备操作已返回，仍需能力核验",
             "reported": "调查报告已返回，不代表实验成功",
             "not attempted": "未执行（原因见操作记录）",
             "done": "操作已完成", "already done": "已有验收，无需重复执行",
             "recorded": "已记录", "registered": "已登记", "submitted": "已提交，尚待返回",
             "rejected": "请求未通过校验",
             "yielded": "本段已返回主控，不代表实验成功"}.get(outcome, outcome)
    reason = str(row.get("because") or "")
    if outcome == "raised" and "stream_protocol" in reason:
        detail = "模型调用的输出协议异常，没有可采用的最终结果；这不是原生安装失败或环境就绪的证据。"
    elif outcome == "raised" and "wall_timeout" in reason:
        detail = "模型调用超过局部时间窗口；不能据此断言仿真或安装失败。"
    else:
        detail = reason
    return name, label, detail


def pending_requests(root: Path) -> list[dict[str, Any]]:
    return [row for row in read(root, "report/demo_requests.json").get("requests", [])
            if isinstance(row, dict) and row.get("status") == "pending"]


def execution_facts(native, measurements):
    """Typed receipt facts: native success is neither an import nor a formal score."""
    stages = {}
    for row in native:
        stage = str(row.get('stage') or 'unknown')
        count = stages.setdefault(stage,{'completed':0,'failed':0,'unverified':0,'gpu_granted':0})
        if row.get('evidence_verified') and row.get('status')=='completed' and row.get('returncode')==0:
            count['completed'] += 1
        elif row.get('evidence_verified') and row.get('returncode') not in {None,0}:
            count['failed'] += 1
        else:
            count['unverified'] += 1
        if row.get('gpu_allowed') is True:
            count['gpu_granted'] += 1
    return {'stages':stages,'formal_scored_measurements':sum(
        finite(m.get('metric_value')) for m in measurements),
        'scope':'latest 32 native receipts, not full experiment history; GPU grants are not utilization',
        'native_event_digest':object_digest([{k:r.get(k) for k in
            ('id','status','returncode','evidence_verified','gpu_allowed')} for r in native])}


def make_snapshot(root: Path, view: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    from .harness_repair import status as harness_repair_status
    from .stage_verification import recent_attempts
    """Keep presentation facts immutable and separate event triggers from live telemetry."""
    rows = [dict(row) for row in view.get("measurements", []) if not row.get("confirmation")]
    baseline = next((row for row in rows if row.get("label") == "baseline"), {})
    policy_groups = {}
    for row in rows:
        identity = row.get('policy_sha256')
        if isinstance(identity, str) and re.fullmatch(r'[0-9a-f]{64}', identity):
            # Comparison within this evidence snapshot, not a raw weight hash
            # or a claim that the observer reverified the archive bytes.
            row['policy_identity_group'] = policy_groups.setdefault(identity, f'policy_{len(policy_groups)+1}')
        row["id"] = "m-" + object_digest(row.get("measurement_ref"))[:16]
        comparable = bool(not str(row.get('label', '')).startswith('reference_') and
                          row.get('candidate_validity') != 'source_mismatch' and
                          view.get("source_consistency") == "stable" and row.get("protocol_sha256") and
                          row.get("protocol_sha256") == baseline.get("protocol_sha256") and
                          row.get("metric_name") == baseline.get("metric_name") and
                          row.get("metric_unit") == baseline.get("metric_unit") and
                          row.get("metric_direction") in {"maximize", "minimize"} and
                          row.get("metric_direction") == baseline.get("metric_direction") and
                          finite(row.get("metric_value")) and finite(baseline.get("metric_value")))
        row["comparable"] = comparable
        row["delta"] = row["metric_value"] - baseline["metric_value"] if comparable else None
    actions = context.get("actions", [])[-8:]
    progress_path = root / "provision_progress.json"
    progress = read(root, "provision_progress.json") if progress_path.is_file() else {}
    ledger_path = root / "agent" / "cost_ledger.json"
    ledger = read(root, "agent/cost_ledger.json") if ledger_path.is_file() else {}
    entries = ledger.get("entries", [])
    reuse = read(root, "environment_selection.json")
    current_environment = read(root, 'environment.json')
    cursor = read(root, 'provision_cursor.json')
    environment_queue = {'python':(cursor.get('planned') or {}).get('python') or current_environment.get('python'),
        'passed':(current_environment.get('verdict') or {}).get('passed') is True,
        'pending':cursor.get('pending') or [], 'next_probe':cursor.get('next_probe'),
        'probe_count':len(cursor.get('probes') or []), 'prefix':cursor.get('prefix')}
    pool_policy = read(root, "environment_pool.json")
    cache_view = read(root, "package_cache_view.json")
    publication = read(root, "environment_cache_publication.json")
    environment_reuse = {"enabled": pool_policy.get("enabled", False),
        "mode": reuse.get('mode'),
        "source_bindings": [{k:b.get(k) for k in ('module','origin')} for b in
                            ((reuse.get('result') or {}).get('overlay') or {}).get('bindings', [])],
        "selected_id": reuse.get("id"), "selection_reason": reuse.get("reason"),
        "base_verification": reuse.get('base_verification') or {},
        "clone_ok": (reuse.get("result") or {}).get("ok"),
        "available_wheels": len(cache_view.get("links") or []),
        "snapshot_status": (publication.get("snapshot") or {}).get("status"),
        "publication_status": publication.get("status")}
    model_budget = {"limit_usd": ledger.get("limit_usd"),
        "updated_at":ledger.get('updated_at'),
        "pending_reserved_usd":sum(float(r.get('reserved_usd') or 0) for r in entries if r.get('status')=='reserved'),
        "unknown_reserved_usd":sum(float(r.get('reserved_usd') or 0) for r in entries if r.get('status')=='unknown'),
        "spent_usd": sum(float(row.get("actual_usd") or 0)
                         for row in entries if row.get("status") == "settled"),
        "held_usd": sum(float(row.get("reserved_usd") or 0)
                        for row in entries if row.get("status") in {"reserved", "unknown"})}
    # Context is supplied by Preparation, never inferred from heartbeat process activity.
    from .review_transaction import review_view
    from .monitor_findings import view as monitor_findings_view
    from .data_preview import view as data_preview_view
    facts = {"status": view.get("status"), "measurements": rows,
             "baseline_reference": view.get('baseline_reference') or {'status':'not_declared'},
             "experiment_lifecycle": view.get('experiment_lifecycle') or {},
             "data_versions": view.get('data_versions') or [],
             "data_previews": data_preview_view(root),
             "data_campaigns": view.get('data_campaigns') or [],
             "actions": actions, "plan": context.get("plan") or {},
             'research_progress':view.get('research_progress') or {'status':'unavailable'},
             'last_completed_action':actions[-1] if actions else {},
             "monitor_findings": monitor_findings_view(root, str(view.get('run_id') or '')),
             "review_transactions": review_view(root),
             "recovery_transaction": context.get("recovery_transaction") or {},
             "provisioning": progress, "model_budget": model_budget,
             "installation_recovery_inventory": read(root, "installation_recovery_inventory.json"),
             "harness_repair": harness_repair_status(root),
             "native_stage_verifications": recent_attempts(root,limit=32),
             "environment_reuse": environment_reuse, 'environment_queue':environment_queue,
             "media_events": read(root, f"research/{view['run_id']}/media_events.json").get('rows') or [],
             "verified_media": [row for row in view.get("verified_media", [])
                                if row.get("trigger") != "final_confirmation"],
             "telemetry_attempts": view.get("telemetry_attempts") or [],
             "budget": view.get("budget") or {}, "run_id": view["run_id"],
             "source_consistency": view.get("source_consistency"),
             "source_revision_vector": view.get("source_revision_vector") or {}}
    facts["runtime_failure"] = read(root, "agent/runtime_failure.json")
    facts['model_provider_blocker'] = {}
    failed_turn = str(facts['runtime_failure'].get('turn_id') or '')
    if facts['status'] == 'infrastructure_blocked' and re.fullmatch(r'[0-9a-f]{32}', failed_turn):
        from .agent_runtime import blocked_provider_category
        pricing_ref = f'agent/pricing/{failed_turn}.json'
        pricing = read(root, pricing_ref)
        category = blocked_provider_category(pricing.get('gateway'), None)
        if category:
            facts['model_provider_blocker'] = {'category':category,
                'pricing_ref':pricing_ref, 'authority':'recorded upstream HTTP status, not an account balance or settled usage claim'}
    if facts['runtime_failure']:
        facts['runtime_failure'] = {**facts['runtime_failure'],
            'observation_scope':'historical failure record; not evidence of current liveness or a current blocker; corroborate against run and last_completed_action'}
    facts["active_candidate"] = (active_candidate(root, view["run_id"])
                                 if view.get("status") == "running" else {})
    facts["runtime_recovery"] = read(root, "runtime_blocker.json")
    schedule = read(root, "scheduling_metrics.json")
    grouped = {}
    for event in schedule.get("events") or []:
        name = str(event.get("kind") or "unknown")
        item = grouped.setdefault(name, {"count": 0, "seconds": 0.0})
        item["count"] += 1
        if finite(event.get("seconds")):
            item["seconds"] += event["seconds"]
    grouped = schedule.get("aggregates") or grouped
    facts["scheduling"] = {"policy": read(root, "scheduler_policy.json"),
        "jobs": context.get("native_jobs") or [], "agents": context.get("agent_tasks") or [],
        "timing": grouped}
    screening = []
    for path in sorted((root / "research" / str(view["run_id"]) / "screening").glob("*/trials/*/trial.json"))[-32:]:
        row = read(root, str(path.relative_to(root)))
        if row:
            screening.append({key: row.get(key) for key in
                ("trial_id", "idea_label", "rung", "status", "metric_value", "measurement_ref")})
    facts["screening"] = screening
    facts['execution_facts'] = execution_facts(facts['native_stage_verifications'],facts['measurements'])
    event_facts = {key: facts[key] for key in ("status", "measurements", "actions", "plan", "verified_media", "recovery_transaction", 'execution_facts')}
    event_facts['fact_review_contract_version'] = FACT_REVIEW_CONTRACT_VERSION
    event_facts['research_progress'] = facts['research_progress']
    if facts['monitor_findings']:
        event_facts['operator_diagnostic_ids'] = [row['evidence_id'] for row in facts['monitor_findings']]
    demo = read(root, "environment_demo.json")
    if demo.get("status") == "recorded_unscored":
        from .common import digest
        from .evidence_store import read_attempt_evidence
        try:
            read_attempt_evidence(root, demo["evidence_id"], limit=1)
            identity = demo.get("id")
            if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{32}", identity):
                raise ValueError("invalid environment preview identity")
            if read(root, f"environment_demos/{identity}/receipt.json") != demo:
                raise ValueError("environment preview receipt changed")
            if not isinstance(demo.get("media"), list) or not 1 <= len(demo["media"]) <= 3:
                raise ValueError("invalid preview media list")
            for media in demo.get("media", []):
                path = root / media["path"]
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or digest(path) != media["sha256"]:
                    raise ValueError("environment preview changed")
                preview = media.get("preview") or {}
                frame = root / preview["path"]
                if (preview.get("parent_sha256") != media["sha256"] or frame.is_symlink()
                        or not frame.resolve().is_relative_to(root.resolve())
                        or digest(frame) != preview["sha256"]):
                    raise ValueError("environment preview frame changed")
            facts["environment_demo"] = demo
        except (OSError, ValueError, KeyError, TypeError):
            facts["environment_demo"] = {"status": "evidence_unavailable"}
    else:
        facts["environment_demo"] = demo
    event_facts["environment_demo"] = facts["environment_demo"]
    event_facts["provisioning"] = progress
    event_facts["installation_recovery_inventory"] = (facts["installation_recovery_inventory"] or {}).get("digest")
    event_facts["runtime_failure"] = facts["runtime_failure"]
    event_facts["runtime_recovery"] = facts["runtime_recovery"]
    event_facts['baseline_reference'] = facts['baseline_reference']
    event_facts['experiment_lifecycle'] = facts['experiment_lifecycle']
    event_facts['data_versions'] = facts['data_versions']
    event_facts['data_previews'] = [{k:r.get(k) for k in ('data_version_id','episode_id','sha256','preview_ref')}
                                  for r in facts['data_previews']]
    event_facts['data_campaigns'] = facts['data_campaigns']
    event_facts['monitor_findings'] = [row.get('evidence_id')
                                    for row in facts['monitor_findings']]
    event_facts["harness_repair"] = facts["harness_repair"]
    event_facts["environment_reuse"] = environment_reuse
    event_facts['environment_queue'] = environment_queue
    event_facts["screening"] = screening
    event_facts["active_candidate"] = facts["active_candidate"]
    event_facts["background_states"] = [[row.get("job_id"), row.get("status")]
                                      for row in facts["scheduling"]["jobs"]]
    event_facts["agent_states"] = [[row.get("id"), row.get("status"), row.get("stale")]
                                 for row in facts["scheduling"]["agents"]]
    # Scalar samples change frequently: plots update without a new model call on each point.
    event_facts["training_states"] = [
        (row.get("attempt_id"), row.get("stage_status"), row.get("errors"))
        for row in facts["telemetry_attempts"]]
    snapshot = {**facts, "current_action": view.get("current_action"),
                "view_revision": view["view_revision"],
                "event_revision": object_digest(event_facts)}
    snapshot["revision"] = object_digest(snapshot)
    path = root / "report" / "snapshots" / (snapshot["revision"] + ".json")
    if not path.exists():
        atomic_json(path, {**snapshot, "created_at": now()})
    atomic_json(root / "report" / "presentation.json", snapshot)
    return snapshot


def _evidence(snapshot: dict[str, Any]) -> dict[str, Any]:
    evidence = {"run": {key: snapshot.get(key) for key in ("status", "current_action", "run_id")},
                "plan": {'authority':'model working intent/historical notes, NOT current execution or cursor facts',
                         'document':snapshot["plan"]},
                'research_progress':snapshot.get('research_progress') or {'status':'unavailable'},
                'last_completed_action':snapshot.get('last_completed_action') or {}}
    evidence["runtime_failure"] = snapshot.get("runtime_failure") or {}
    evidence["runtime_recovery"] = snapshot.get("runtime_recovery") or {}
    evidence['monitor_findings'] = snapshot.get('monitor_findings') or []
    evidence['baseline_reference'] = snapshot.get('baseline_reference') or {}
    evidence['experiment_lifecycle'] = snapshot.get('experiment_lifecycle') or {}
    evidence['execution_facts'] = snapshot.get('execution_facts') or {}
    evidence['data_versions'] = snapshot.get('data_versions') or []
    evidence['data_previews'] = snapshot.get('data_previews') or []
    evidence['data_campaigns'] = snapshot.get('data_campaigns') or []
    evidence["environment_demo"] = snapshot.get("environment_demo") or {}
    evidence['environment_queue'] = snapshot.get('environment_queue') or {}
    evidence['active_candidate'] = snapshot.get('active_candidate') or {}
    evidence.update({row["id"]: row for row in snapshot["measurements"]})
    evidence.update({f"action-{i}": row for i, row in enumerate(snapshot["actions"])})
    evidence["training"] = [{key: row.get(key) for key in (
        "attempt_id", "stage_status", "metrics", "errors", "chart_ref")}
        for row in snapshot["telemetry_attempts"]]
    evidence["media"] = [{key: row.get(key) for key in (
        "trigger", "measurement_label", "episode_id", "attempt_id")}
        for row in snapshot["verified_media"]]
    evidence["budget"] = snapshot["budget"]
    provisioning = dict(snapshot.get("provisioning") or {})
    attempts = provisioning.get("attempts") or []
    if isinstance(attempts, list) and len(attempts) > 8:
        provisioning.update(attempt_count=len(attempts),
            failed_attempt_count=sum(row.get("ok") is False for row in attempts if isinstance(row, dict)),
            attempts=attempts[-8:], history_scope="latest eight attempts; full history remains in the immutable report snapshot")
    evidence["provisioning"] = provisioning
    inventory = dict(snapshot.get("installation_recovery_inventory") or {})
    wheels = inventory.get("local_wheels")
    if isinstance(wheels, list) and len(wheels) > 8:
        inventory.update(local_wheel_count=len(wheels), local_wheels=wheels[:8],
            inventory_scope="eight examples, not the complete install inventory; Recorder does not select installation resources")
    evidence["installation_recovery_inventory"] = inventory
    evidence["harness_repair"] = snapshot.get("harness_repair") or {}
    evidence["model_budget"] = snapshot.get("model_budget") or {}
    evidence["environment_reuse"] = snapshot.get("environment_reuse") or {}
    evidence["scheduling"] = snapshot.get("scheduling") or {}
    evidence["screening"] = snapshot.get("screening") or []
    return evidence


def _event_handle(snapshot: dict[str, Any]) -> str:
    """A short request handle, not a hash the writer has to reconstruct.

    Full snapshot/event identities stay executor-owned. The handle binds both;
    exact comparison still rejects stale replies, including a different snapshot
    of the same material event. Legacy exact event hashes remain accepted.
    """
    return 'report-' + object_digest({
        'event_revision': snapshot['event_revision'],
        'snapshot_revision': snapshot['revision']})[:24]


def _validate(answer: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    if (not isinstance(answer, dict) or (
            answer.get("event_revision") != snapshot["event_revision"] and (
                not isinstance(snapshot.get("revision"), str) or
                answer.get("event_revision") != _event_handle(snapshot)))):
        raise ValueError("Recorder response does not match the event revision")
    evidence = _evidence(snapshot)
    sections = answer.get("sections")
    if not isinstance(sections, dict) or set(sections) != set(FIELDS):
        raise ValueError("Recorder must answer all five reader questions")
    clean = {}
    for key in FIELDS:
        row = sections[key]
        text = row.get("text") if isinstance(row, dict) else None
        refs = row.get("evidence_ids") if isinstance(row, dict) else None
        if (not isinstance(text, str) or not 1 <= len(text) <= 700 or
                not re.search(r"[\u4e00-\u9fff]", text)):
            size = len(text) if isinstance(text, str) else 'not a string'
            raise ValueError(f"Recorder section {key}: text must contain Chinese and "
                             f"be 1..700 characters (got {size})")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 6:
            size = len(refs) if isinstance(refs, list) else 'not a list'
            raise ValueError(f"Recorder section {key}: evidence_ids must contain "
                             f"1..6 existing IDs (got {size})")
        if any(not isinstance(ref, str) or ref not in evidence for ref in refs):
            raise ValueError(f"Recorder section {key}: unknown or invalid evidence ID; "
                             "use only offered top-level keys in allowed_evidence_ids, "
                             "not nested UUIDs or paths")
        numeric = r"\d+(?:\.\d+)*"
        cited = json.dumps([evidence[ref] for ref in refs], ensure_ascii=False, default=str)
        unsupported = sorted(set(re.findall(numeric, text)) - set(re.findall(numeric, cited)))
        if unsupported:
            raise ValueError(f"Recorder section {key}: prose contains numbers absent "
                             f"from this section's cited evidence: {unsupported[:8]}; remove unsupported "
                             "numbers or cite the actual offered evidence")
        claims = row.get("metric_claims", [])
        if not isinstance(claims, list) or len(claims) > 6:
            raise ValueError("invalid metric claims")
        claim_numbers = set()
        performance_numeric = r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)*(?![A-Za-z0-9_.])"
        for claim in claims:
            if not isinstance(claim, dict) or set(claim) != {"measurement_id", "field", "value", "unit"}:
                raise ValueError("metric claims require measurement, field, value and unit")
            measurement = next((m for m in snapshot["measurements"]
                               if m["id"] == claim["measurement_id"] and m["id"] in refs), None)
            if (measurement is None or claim["field"] not in {"metric_value", "delta"} or
                    not finite(claim["value"]) or
                    claim["value"] != measurement.get(claim["field"]) or
                    claim["unit"] != measurement.get("metric_unit", (measurement.get("metric") or {}).get("unit"))):
                raise ValueError("metric claim does not match its typed measurement")
            claim_numbers.update(re.findall(performance_numeric, str(claim["value"])))
            claim_numbers.update(re.findall(performance_numeric, number(claim["value"])))
        performance = re.finditer(
            r"(?:成功率|准确率|得分|奖励|回报|指标|reward|success_rate)"
            r"(?:(?![。；\n]).){0,30}\d", text)
        # IDs such as round_6/v2 are already checked against cited evidence above;
        # their embedded digits are not numeric policy-performance claims.
        performance_numbers = set()
        for mention in performance:
            for value_match in re.finditer(performance_numeric, mention.group()):
                value = value_match.group()
                following = text[mention.start() + value_match.end():]
                # An explicitly labelled integral episode/sample count is not
                # a policy score. All its digits still require cited evidence
                # above; decimal/signed scores never get this exemption.
                if (re.fullmatch(r"\d+", value) and
                        re.match(r"\s*(?:集|个(?:评测)?回合|episodes?\b|条(?:演示|轨迹))", following)):
                    continue
                performance_numbers.add(value)
        if performance_numbers and (not claims or not performance_numbers <= claim_numbers):
            missing = sorted(performance_numbers - claim_numbers)
            raise ValueError(f"Recorder section {key}: numeric performance prose "
                             f"requires typed metric claims for {missing}; "
                             "identifier digits are not metrics")
        if re.search(r"(?:成功率|准确率|得分|奖励|回报|主指标).{0,15}(?:提高|提升|改善)", text) and not snapshot["measurements"]:
            raise ValueError("no measurement supports an improvement claim")
        clean[key] = {"text": text, "evidence_ids": list(dict.fromkeys(refs))}
        if claims:
            clean[key]["metric_claims"] = claims
    charts = answer.get("charts", [])
    if not isinstance(charts, list) or len(charts) > 3:
        raise ValueError("invalid chart requests")
    for row in charts:
        if (not isinstance(row, dict) or set(row) != {"kind", "why"} or
                row["kind"] not in CHARTS or not isinstance(row["why"], str) or
                not 1 <= len(row["why"]) <= 400):
            raise ValueError("chart requests are declarative choices, not code")
    requests = answer.get("demo_requests", [])
    if not isinstance(requests, list) or len(requests) > 2:
        raise ValueError("invalid demo requests")
    eligible = {row["id"] for row in snapshot["measurements"] if finite(row.get("metric_value"))}
    data_eligible = {row['id'] for row in snapshot.get('data_versions', []) if row.get('status') == 'reviewed_not_consumed'}
    for row in requests:
        if (isinstance(row, dict) and set(row) == {"kind", "purpose"} and
                row["kind"] == "environment_smoke" and isinstance(row["purpose"], str) and
                1 <= len(row["purpose"]) <= 400):
            continue
        if (isinstance(row, dict) and set(row) == {'kind','data_version_id','purpose'}
                and row['kind'] == 'dataset_preview' and row['data_version_id'] in data_eligible
                and isinstance(row['purpose'], str) and 1 <= len(row['purpose']) <= 400):
            continue
        if (not isinstance(row, dict) or set(row) != {"measurement_id", "purpose"} or
                row["measurement_id"] not in eligible or not isinstance(row["purpose"], str) or
                not 1 <= len(row["purpose"]) <= 400):
            raise ValueError("demo requests need a scored development measurement and purpose")
    return {"sections": clean, "charts": charts, "demo_requests": requests}


def _section_validation_errors(proposed: dict, snapshot: dict) -> list[str]:
    """Use the same validator to identify all section errors for one correction.

    Temporary neutral sections exist only in this diagnostic projection, never in
    a saved narrative. The corrected complete response still passes _validate.
    """
    sections = proposed.get('sections')
    if not isinstance(sections, dict):
        return []
    errors = []
    for key in FIELDS:
        scoped = {'event_revision': proposed.get('event_revision'), 'sections': {
            field: sections.get(field) if field == key else
                {'text': '证据不足，暂不能判断。', 'evidence_ids': ['run']}
            for field in FIELDS}}
        try:
            _validate(scoped, snapshot)
        except ValueError as error:
            errors.append(str(error)[:300])
    return list(dict.fromkeys(errors))


class FactReviewUnavailable(RuntimeError):
    """Reviewer transport/format failure cannot be repaired by changing prose."""


def _review_facts(root, client, snapshot, proposed, *, remaining, options, attempt, turns):
    """One independent bounded read-only review, in the original writing window."""
    if remaining <= 0:
        raise TimeoutError('Recorder original writing window exhausted before fact review')
    if not narration_allowance(root)['allowed']:
        raise FactReviewUnavailable('Recorder fact review cannot spend protected recovery budget')
    evidence = _evidence(snapshot)
    refs = {ref for section in proposed['sections'].values() for ref in section['evidence_ids']}
    refs.update({'execution_facts','last_completed_action','environment_queue','harness_repair'})
    refs.update(row['measurement_id'] for row in proposed.get('demo_requests', [])
                if row.get('measurement_id'))
    if refs & {'plan','data_campaigns'}:
        # A requested quota in a plan/campaign is not an observed accepted count.
        # Include current data facts even when the writer cites only its plan.
        refs.update({'data_versions','data_campaigns'})
    if any(row.get('kind') == 'dataset_preview' for row in proposed.get('demo_requests', [])):
        refs.add('data_versions')
    # Review cited facts only, not the entire skill library/telemetry history.
    request = sanitize_model_payload({'phase': 'review_report',
        'event_revision': snapshot['event_revision'], 'proposed_report': proposed,
        'evidence': {ref: evidence[ref] for ref in sorted(refs)},
        'allowed_evidence_ids': sorted(refs),
        'authority': 'source-bound prose verification only; not an experiment or independent benchmark confirmation'})
    if len(json.dumps(request, ensure_ascii=False).encode()) > 96 * 1024:
        raise FactReviewUnavailable('Recorder factual evidence exceeds the bounded review packet')
    import time
    review_started = time.monotonic()
    from .agent_client import role_scope
    from .execution_derive import _object
    try:
        with role_scope(client, 'supervisor'):
            content, metadata = client.chat_with_metadata(load_prompt('recorder-fact-review'),
                json.dumps(request, ensure_ascii=False), **{**options, 'max_tokens': 1500,
                    'timeout': min(options['timeout'], 45, remaining), 'resume_session': False})
        turns.append(metadata)
        if not isinstance(content, str) or len(content) > 12000:
            raise ValueError('fact review response exceeds its bound')
        answer = _object(content)
        issues = answer.get('issues')
        if (set(answer) != {'verdict', 'issues'}
                or answer.get('verdict') not in {'supported', 'unsupported', 'uncertain'}
                or not isinstance(issues, list) or len(issues) > 4
                or (answer['verdict'] == 'supported') != (not issues)):
            raise ValueError('fact review needs a bounded verdict and issues')
        def references_valid(issue):
            return (isinstance(issue, dict) and set(issue) == {'section', 'reason', 'evidence_ids'}
                    and isinstance(issue['section'], str)
                    and issue['section'] in {*FIELDS, 'charts', 'demo_requests'}
                    and isinstance(issue['reason'], str) and 1 <= len(issue['reason']) <= 250
                    and isinstance(issue['evidence_ids'], list) and 1 <= len(issue['evidence_ids']) <= 6
                    and all(isinstance(ref, str) and ref in request['evidence'] for ref in issue['evidence_ids']))
        if any(not references_valid(issue) for issue in issues):
            # A negative scientific judgment must not disappear because its section
            # name/citation is malformed. Ask the SAME independent reviewer for one
            # index-based reference correction, never a rewritten verdict or reason.
            eligible = answer['verdict'] in {'unsupported', 'uncertain'} and all(
                isinstance(issue, dict) and set(issue) == {'section', 'reason', 'evidence_ids'}
                and isinstance(issue['section'], str) and len(issue['section']) <= 48
                and isinstance(issue['reason'], str) and 1 <= len(issue['reason']) <= 250
                and isinstance(issue['evidence_ids'], list) and len(issue['evidence_ids']) <= 6
                for issue in issues)
            left = remaining-(time.monotonic()-review_started)
            if (not eligible or attempt.get('fact_review_format_repair_attempted')
                    or left < 1 or not narration_allowance(root)['allowed']):
                raise ValueError('fact review issue is not bound to the supplied evidence')
            attempt['fact_review_format_repair_attempted'] = True
            original = answer
            repair_request = {**request, 'phase': 'repair_issue_references',
                'original_review': original, 'allowed_sections': [*FIELDS, 'charts', 'demo_requests']}
            attempt['fact_review_reference_repair'] = {
                'original_answer': original, 'original_provider_turn': metadata,
                'request_sha256': object_digest(repair_request)}
            with role_scope(client, 'supervisor'):
                fixed, fixed_meta = client.chat_with_metadata(
                    load_prompt('recorder-fact-reference-repair'),
                    json.dumps(repair_request, ensure_ascii=False),
                    **{**options, 'max_tokens': 700, 'timeout': min(options['timeout'], 45, left),
                       'resume_session': False})
            turns.append(fixed_meta)
            if not isinstance(fixed, str) or len(fixed) > 8000:
                raise ValueError('fact review reference repair exceeds its bound')
            mapping = _object(fixed)
            selections = mapping.get('issue_citations')
            if (set(mapping) != {'issue_citations'} or not isinstance(selections, list)
                    or len(selections) != len(issues)):
                raise ValueError('fact review correction must preserve all original issues')
            by_index = {}
            for selection in selections:
                if (not isinstance(selection, dict) or
                        set(selection) != {'issue_index', 'section', 'evidence_ids'} or
                        type(selection['issue_index']) is not int or
                        not 0 <= selection['issue_index'] < len(issues) or
                        selection['issue_index'] in by_index):
                    raise ValueError('invalid fact review issue selection')
                by_index[selection['issue_index']] = selection
            answer = {'verdict': original['verdict'], 'issues': [
                {'reason': issue['reason'], 'section': by_index[i]['section'],
                 'evidence_ids': by_index[i]['evidence_ids']} for i, issue in enumerate(issues)]}
            issues = answer['issues']
            attempt['fact_review_reference_repair'].update(answer=mapping, provider_turn=fixed_meta)
        for issue in issues:
            if not references_valid(issue):
                raise ValueError('fact review issue is not bound to the supplied evidence')
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        raise FactReviewUnavailable(f'Recorder fact review unavailable: {type(exc).__name__}: {exc}') from exc
    attempt.setdefault('fact_reviews', []).append({'request_sha256': object_digest(request),
        'proposed_report_sha256': object_digest(proposed), 'answer': answer, 'provider_turn': metadata})
    if answer['verdict'] != 'supported':
        raise ValueError('Recorder factual contradiction: ' + json.dumps(issues, ensure_ascii=False))


def narration_allowance(root: Path) -> dict:
    """Read fresh accounting; reporting cannot spend the recovery safety margin."""
    path = root/'agent/cost_ledger.json'
    if not path.exists():
        return {'allowed':True,'reason':'no persistent model ledger'}
    try:
        from .agent_budget import AgentCostLedger
        if path.is_symlink():
            raise ValueError('unsafe model ledger')
        row = read_json(path)
        usage = AgentCostLedger(root,run_id=row['run_id'],limit_usd=row['limit_usd'],
            cost_basis=row.get('cost_basis','cli_reported')).snapshot()
        return {'allowed':usage['remaining_usd'] > usage['recovery_reserve_usd']+1e-9,
            'remaining_usd':usage['remaining_usd'],'protected_usd':usage['recovery_reserve_usd'],
            'reason':'优先保留实验执行和故障恢复额度；事实报告继续更新'}
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        return {'allowed':False,'reason':'模型账本无法核验，暂不调用报告模型：'+type(exc).__name__}


def _semantic_failure(action):
    """Receipt IDs and changing timestamps do not turn a repeated refusal into research."""
    if action.get('outcome') not in {'raised','failed','rejected'}:
        return None
    keys=('step','outcome','failure_category','failure_domain','repair_owner')
    result={key:action.get(key) for key in keys}
    if action.get('failure_category')=='operation_arguments_invalid':
        # RUN.md still updates exact validation text locally; paid prose can batch it.
        result['reason']='operation_arguments_invalid'
    else:
        result['reason']=action.get('because')
        result['evidence_id']=action.get('evidence_id')
    return result


def _select_writing_methods(root, client, selection_packet, options):
    """Reuse the Agent's own advisory choice, never a keyword-selected skill.

    Evidence, prose and factual reviews are NOT cached. A new purpose, catalog,
    instruction, failure category or selected body invalidates the choice.
    """
    from .execution_derive import _object
    from .skills import read_selected_skills
    outline = selection_packet['task_outline']
    key = object_digest({'system': SYSTEM, 'catalog': selection_packet['skill_catalog'],
        'purpose': {name: outline.get(name) for name in
            ('status', 'has_failure', 'has_media', 'purpose', 'failure_category')},
        'has_measurements': bool(outline.get('measurement_count'))})
    saved = read(root, 'report/recorder_skills.json')
    choices = saved.get('choices')
    if (saved.get('key') == key and isinstance(choices, list) and len(choices) <= 2
            and all(isinstance(row, dict) and isinstance(row.get('id'), str)
                and isinstance(row.get('why'), str) and row['why'].strip() for row in choices)):
        try:
            methods = read_selected_skills([row['id'] for row in choices])
            if (len(methods['skills']) == len(choices) and
                    methods['selection'] == saved.get('selection')):
                return methods, {row['id']: row['why'] for row in choices}, None, True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    content, metadata = client.chat_with_metadata(SYSTEM,
        json.dumps(selection_packet, ensure_ascii=False), **options)
    choices = _object(content).get('skill_reads')
    if not isinstance(choices, list) or len(choices) > 2:
        raise ValueError('invalid Recorder skill selection')
    if any(not isinstance(row, dict) or not isinstance(row.get('id'), str) or
           not isinstance(row.get('why'), str) or not row['why'].strip() for row in choices):
        raise ValueError('Recorder skill choice needs a reason')
    choices = [{'id': row['id'], 'why': row['why'][:400]} for row in choices]
    methods = read_selected_skills([row['id'] for row in choices])
    if len(methods['skills']) != len(choices):
        raise ValueError('selected Recorder skill body is unavailable')
    atomic_json(root / 'report/recorder_skills.json', {'key': key, 'choices': choices,
        'selection': methods['selection'], 'selected_at': now(),
        'selection_metadata': metadata, 'scope': 'advisory writing methods only'})
    return methods, {row['id']: row['why'] for row in choices}, metadata, False


def narrate(root: Path, snapshot: dict[str, Any], *, client: Any = None,
            timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """At most one attempt per semantic event, including failures and restarts."""
    previous = read(root, "report/narrative.json")
    if client is None or not getattr(client, "supports_recorder", False):
        return previous
    allowance = narration_allowance(root)
    if not allowance['allowed']:
        atomic_json(root/'report/recorder_deferred.json', {**allowance,'at':now(),'mode':'fact_only'})
        return previous
    if snapshot.get("source_consistency") != "stable":
        return previous
    if snapshot.get('status') in {'budget_waiting','budget_exhausted','internal_error','infrastructure_blocked'}:
        return previous
    actions = snapshot.get('actions') or []
    if actions and actions[-1].get('failure_domain') == 'framework_plan':
        return previous
    if (snapshot.get('status') == 'running' and actions and
            actions[-1].get('step') == 'choose' and actions[-1].get('outcome') == 'decision_rejected'):
        # The deterministic presentation already displays the exact rejected fields.
        # A formatting-retry/Monitor cycle is not a new experiment to narrate.
        return previous
    from .run_record import _report_lock
    with _report_lock(root / "report" / "recorder-turn"):
        previous = read(root, "report/narrative.json")
        trigger = read(root, "report/recorder_trigger.json")
        if trigger.get("event_revision") == snapshot["event_revision"] and trigger.get('status') != 'deferred':
            return previous
        # RUN.md facts/charts are refreshed independently. Batch routine prose updates,
        # but never delay a new measurement, failure, demo or terminal transition.
        important = object_digest({key: snapshot.get(key) for key in
            ("status", "measurements", "runtime_failure", "runtime_recovery", "monitor_findings",
             "verified_media", "environment_demo", "recovery_transaction", 'research_progress','execution_facts',
             'plan', 'data_versions', 'data_campaigns', 'data_previews','harness_repair')} | {
                 'fact_review_contract_version': FACT_REVIEW_CONTRACT_VERSION,
                 "latest_action_failure": _semantic_failure(actions[-1]) if actions else None})
        import time
        if (trigger.get('status') == 'failed' and
                trigger.get('important_event_digest') == important):
            atomic_json(root / 'report/recorder_deferred.json', {
                'at': now(), 'mode': 'fact_only',
                'reason': 'unchanged failed prose deferred until material evidence changes; facts stay live',
                'event_revision': snapshot['event_revision']})
            return previous
        if previous.get("important_event_digest") == important:
            atomic_json(root / "report" / "recorder_deferred.json", {
                "at": now(), "mode": "fact_only", "reason": "unchanged research facts; retain reviewed prose and live tables",
                "event_revision": snapshot["event_revision"]})
            return previous
        attempt = {"event_revision": snapshot["event_revision"], "revision": snapshot["revision"],
                   "attempted_at": now(), "attempted_epoch": time.time(),
                   "important_event_digest": important, "status": "started"}
        atomic_json(root / "report" / "recorder_trigger.json", attempt)
        narration_started = time.monotonic()
        try:
            from .agent_client import role_scope
            from .execution_derive import _object
            from .skills import skills_catalog
            catalog = skills_catalog(query="RUN.md recorder 中文 图表 视频 可读性")
            packet = {"event_revision": _event_handle(snapshot),
                      "evidence": _evidence(snapshot), "skill_catalog": catalog,
                      "instruction": '仅返回 {"skill_reads":[{"id":"目录ID","why":"选择理由"}]}，最多两项。'}
            packet = sanitize_model_payload(packet, local_roots=(root,))
            # Skill selection needs a task outline, not the entire evidence packet.
            # The subsequent writer still receives all evidence and chosen skill bodies.
            selection_packet = {"phase":"skill_selection", "event_revision": packet["event_revision"],
                "skill_catalog": packet["skill_catalog"], "instruction": packet["instruction"],
                "task_outline": {"status": snapshot.get("status"),
                    "measurement_count": len(snapshot.get("measurements") or []),
                    "latest_actions": (snapshot.get("actions") or [])[-2:],
                    "has_failure": bool(snapshot.get("runtime_failure")),
                    "failure_category": (snapshot.get("runtime_failure") or {}).get('category'),
                    "has_media": bool(snapshot.get("verified_media")),
                    "purpose": "中文进度说明、结果对比、阻碍解释及主动请求 demo"}}
            selection_packet = sanitize_model_payload(selection_packet, local_roots=(root,))
            options = {"max_tokens": 1000, "timeout": max(1, timeout / 2),
                       "thinking":"disabled", "resume_session":False,
                       "read_only": True, "auto_skills": False, "include_research_context": False}
            if getattr(client, "supports_main_agent", False):
                options.update(decision_only=True, output_format="json")
            with role_scope(client, "recorder"):
                methods, reasons, metadata, reused = _select_writing_methods(
                    root, client, selection_packet, options)
                attempt['skill_selection_reused'] = reused
                allowance = narration_allowance(root)
                if not allowance['allowed']:
                    attempt.update(status='deferred',reason=allowance['reason'])
                    atomic_json(root/'report/recorder_deferred.json', {**allowance,'at':now(),'mode':'fact_only'})
                    return previous
                packet.pop("skill_catalog")
                packet['allowed_evidence_ids'] = list(packet['evidence'])
                packet['phase']='write_report'
                packet["instruction"] = "根据相同证据快照，返回规定的最终报告 JSON。"
                packet["methods"] = [{"id": row["id"], "method": row["method"]} for row in methods["skills"]]
                remaining=timeout-(time.monotonic()-narration_started)
                if remaining<=0:
                    raise TimeoutError('Recorder original writing window exhausted after skill selection')
                options={**options,'max_tokens':5000,'timeout':min(max(1,timeout/2),remaining)}
                content, final_metadata = client.chat_with_metadata(SYSTEM, json.dumps(packet, ensure_ascii=False), **options)
                turns = [*([metadata] if metadata is not None else []), final_metadata]
                proposed = None
                try:
                    proposed = _object(content)
                    answer = _validate(proposed, snapshot)
                    _review_facts(root, client, snapshot, answer,
                        remaining=timeout-(time.monotonic()-narration_started), options=options,
                        attempt=attempt, turns=turns)
                except ValueError as rejected:
                    remaining = timeout - (time.monotonic() - narration_started)
                    repair_input = proposed if proposed is not None else str(content)
                    if (remaining < 1 or not narration_allowance(root)['allowed']
                            or len(json.dumps(repair_input, ensure_ascii=False)) > 16000):
                        raise
                    # One response correction, not new research or a fresh time budget.
                    packet['format_correction'] = {
                        'error': str(rejected)[:300],
                        'factual_issues': (attempt.get('fact_reviews') or [{}])[-1].get('answer', {}).get('issues', []),
                        'section_errors': (_section_validation_errors(proposed, snapshot)
                            if isinstance(proposed, dict) else
                            ['响应不是合法JSON；字符串中的双引号必须转义。']),
                        'previous_answer': sanitize_model_payload(repair_input, local_roots=(root,)),
                        'instruction': '修正格式、引用及 factual_issues 中指出的事实矛盾。删掉无证据的因果归因、数字和旧轮次；'
                            '不补造事实，不改测量。保持中文简短，返回完整合法JSON；每段最多6个引用，'
                            '涉及性能数字必须有本段metric_claims，字符串内的引号必须转义。'}
                    packet['phase']='format_repair'
                    repair_options = {**options, 'timeout': min(options['timeout'], remaining)}
                    content, repair_metadata = client.chat_with_metadata(
                        SYSTEM, json.dumps(packet, ensure_ascii=False), **repair_options)
                    turns.append(repair_metadata)
                    attempt['format_repair_attempted'] = True
                    answer = _validate(_object(content), snapshot)
                    _review_facts(root, client, snapshot, answer,
                        remaining=timeout-(time.monotonic()-narration_started), options=options,
                        attempt=attempt, turns=turns)
            record = {**answer, **attempt, "written_at": now(), "status": "written",
                      'observed_run_scope':{key:snapshot.get(key) for key in ('status','current_action')},
                      'fact_review_contract_version': FACT_REVIEW_CONTRACT_VERSION,
                      "written_epoch": time.time(),
                      "important_event_digest": important,
                      "model_authored": True, "skill_selection": [
                          {**row, "why": reasons[row["id"]]} for row in methods["selection"]],
                      "turns": turns}
            atomic_json(root / "report" / "narrative.json", record)
            atomic_json(root / "report" / "narratives" / (snapshot["event_revision"] + ".json"), record)
            _enqueue(root, snapshot, answer["demo_requests"])
            attempt["status"] = "written"
            return record
        except Exception as exc:  # reporting never terminates an experiment
            attempt.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:300])
            return previous
        finally:
            atomic_json(root / "report" / "recorder_trigger.json", attempt)


def _enqueue(root: Path, snapshot: dict[str, Any], requests: list[dict[str, Any]]) -> None:
    from .run_record import _report_lock
    with _report_lock(root / "report" / "demo_requests.json"):
        store = read(root, "report/demo_requests.json")
        rows = store.get("requests", [])
        for request in requests:
            if request.get("kind") == "environment_smoke":
                identity = object_digest({"kind": "environment_smoke", "event": snapshot["event_revision"]})[:24]
                if not any(row.get("id") == identity for row in rows):
                    rows.append({"id": identity, **request, "status": "pending",
                                 "created_at": now(), "event_revision": snapshot["event_revision"]})
                continue
            if request.get('kind') == 'dataset_preview':
                path = root/'data_versions'/(request['data_version_id']+'.json')
                card = read(root, str(path.relative_to(root)))
                if path.is_symlink() or card.get('id') != request['data_version_id']:
                    raise ValueError('data preview registration is unavailable')
                sha = digest(path)
                identity = object_digest({'kind':'dataset_preview','version':card['id'],'hash':sha})[:24]
                if not any(row.get('id') == identity for row in rows):
                    rows.append({'id':identity, **request, 'status':'pending', 'created_at':now(),
                        'data_version_sha256':sha, 'event_revision':snapshot['event_revision']})
                continue
            measurement = next(row for row in snapshot["measurements"] if row["id"] == request["measurement_id"])
            # A resolved request for this exact measurement is not requeued by new wording.
            identity = object_digest({"measurement": measurement["measurement_ref"],
                                      "hash": measurement.get("source_sha256")})[:24]
            if any(row.get("id") == identity for row in rows):
                continue
            rows.append({"id": identity, "status": "pending", "created_at": now(),
                         "measurement_ref": measurement["measurement_ref"],
                         "measurement_sha256": measurement.get("source_sha256"),
                         "label": measurement["label"], "purpose": request["purpose"],
                         "event_revision": snapshot["event_revision"], "snapshot_revision": snapshot["revision"]})
        atomic_json(root / "report" / "demo_requests.json", {"requests": rows})


def resolve_request(root: Path, request_id: str, *, status: str, reason: str,
                    result: dict[str, Any] | None = None) -> dict[str, Any]:
    from .run_record import _report_lock
    with _report_lock(root / "report" / "demo_requests.json"):
        store = read(root, "report/demo_requests.json")
        row = next((row for row in store.get("requests", []) if row.get("id") == request_id), None)
        if row is None or row.get("status") != "pending":
            raise ValueError("request is not pending; never replay an unknown capture")
        row.update(status=status, reason=reason[:600], resolved_at=now(), result=result or {})
        atomic_json(root / "report" / "demo_requests.json", store)
        return dict(row)


def complete_capture(root: Path, request_id: str, result: dict) -> None:
    from .run_record import _report_lock
    with _report_lock(root / 'report/demo_requests.json'):
        store = read(root, 'report/demo_requests.json')
        row = next((r for r in store.get('requests', []) if r.get('id') == request_id), None)
        if row is None or row.get('status') != 'capturing':
            raise ValueError('capture request ownership changed; do not replay')
        row.update(status=result.get('status') or 'unknown', result=result, resolved_at=now())
        atomic_json(root/'report/demo_requests.json', store)


def bar_svg(points: list[tuple[str, float]], title: str, unit: str) -> str:
    """One scale, explicit zero and units. Never mix metrics or invent missing values."""
    points = [(str(name), float(value)) for name, value in points if finite(value)]
    low = min([0.0] + [value for _, value in points])
    high = max([0.0] + [value for _, value in points])
    span = high - low or 1
    x = lambda value: 190 + (value - low) / span * 460
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 {80 + 34 * len(points)}" role="img">',
             f'<title>{html.escape(title)}</title><rect width="100%" height="100%" fill="white"/>',
             f'<text x="12" y="22" font-size="14">{html.escape(title)} ({html.escape(unit or "原始单位")})</text>',
             f'<text x="190" y="43" font-size="11">{low:.6g}</text><text x="650" y="43" font-size="11">{high:.6g}</text>']
    for index, (name, value) in enumerate(points):
        y = 58 + index * 34
        left, right = sorted((x(0), x(value)))
        parts.extend([f'<text x="12" y="{y+15}" font-size="12">{html.escape(name[:24])}</text>',
                      f'<rect x="{left:.2f}" y="{y}" width="{right-left:.2f}" height="20" fill="#3679ad"/>',
                      f'<text x="690" y="{y+15}" font-size="12">{value:.6g}</text>'])
    parts.append('</svg>')
    return ''.join(parts)


def campaign_panel(root: Path, snapshot: dict[str, Any]) -> list[str]:
    campaigns = snapshot.get('data_campaigns') or []
    if not campaigns:
        return []
    lines = ['### 从失败到数据的实验', '',
             '先写明要解决的问题，再对比定向采集和等量随机采集。未完成的实验保持待验证，不填零分。', '']
    for campaign in campaigns:
        if campaign.get('status') == 'evidence_unavailable':
            lines += [f"实验 {cell(campaign['id'])} 的引用证据已不可用，暂不展示结果。", '']
            continue
        lines += [f"**{cell(campaign['id'])}**：{cell(campaign['hypothesis'])}", '',
                  f"起点为 {cell(campaign['development_label'])}，开发指标 {number(campaign['starting_score'])}。"
                  f"[完整采集与筛选计划]({campaign['plan_ref']})", '']
        names = {'targeted': '失败驱动采集与筛选', 'random': '等量随机采集', 'unfiltered': '无目标筛选对照',
            'fixed_official':'已有数据训练', 'fixed_continuation':'不增数据续训',
            'random_quality':'随机补采＋共同质量底线', 'rule_targeted':'固定规则定向补采',
            'agent_targeted':'Agent定向补采', 'targeted_quality':'定向补采＋共同质量底线',
            'fdas':'定向补采＋任务相关筛选'}
        if campaign.get('schema_version') == 2:
            axes = {'accepted_episodes':'等有效episode量', 'attempts':'等采集尝试上限',
                    'acquisition_seconds':'等采集执行时间', 'total_compute':'等完整资源包'}
            lines += ['本实验控制的是'+axes[campaign['comparison']['budget_axis']]+'；其他成本仍须报告。', '']
        if campaign.get('rule_allocation'):
            allocation = campaign['rule_allocation']['allocation']
            lines += ['规则对照按封存的开发失败频率分配，而不是由LLM逐轮挑目标：'+
                      '、'.join(f'{cell(k)}：{v}' for k,v in allocation.items())+'。', '']
        lines += ['| 方法 | 种子 | 全量报告 episode | 训练使用版本 | 开发指标 | 进展 |',
                  '| --- | ---: | ---: | --- | ---: | --- |']
        points, notes = [], []
        for arm in campaign['comparison']['arms']:
            for seed in campaign['comparison']['seeds']:
                row = next((r for r in campaign['observations'] if r['arm'] == arm and r['seed'] == seed), None)
                if row:
                    if row.get('status') in {'unsupported','budget_limited'}:
                        explanation = '能力边界' if row['status'] == 'unsupported' else '总预算受限'
                        lines += [f"| {names[arm]} | {seed} | 未完成 | — | — | "
                                  f"[{explanation}]({row['record_ref']}) |"]
                        notes.append(f"{names[arm]}：{cell(row.get('reason'))}。这是未完成的实验，不是零分。")
                        continue
                    lines += [f"| {names[arm]} | {seed} | {cell(row['reported_full_episodes'])} | "
                              f"{cell(row['training_version_id'])} | {number(row['metric_value'])} | "
                              f"[已消费并计分]({row['record_ref']}) |"]
                    points.append((f'{names[arm]} / {seed}', row['metric_value']))
                    if row.get('count_status') == 'incomplete_or_unknown':
                        notes.append(f"{names[arm]}有开发成绩，但新增数量未达到预定目标或仍未知；不标作等量比较完成。")
                else:
                    lines += [f'| {names[arm]} | {seed} | 未产出 | — | — | 待执行 |']
        lines += ['', '数量来自全量数据报告；是否为不重叠新增轨迹、实际种子/训练配方和总成本是否匹配，仍需独立审核。', '']
        for note in notes:
            lines += [note, '']
        if len(points) >= 2:
            ref = f"report/charts/{snapshot['revision'][:24]}-{campaign['id']}-campaign.svg"
            atomic_text(root/ref, bar_svg(points, '数据实验开发指标对比', '原生指标'))
            lines += [f'![数据实验对比]({ref})', '']
    return lines


def data_panel(root: Path, snapshot: dict[str, Any]) -> list[str]:
    """Readable data lineage/exposure, from registered metadata and measurements only."""
    versions = snapshot.get('data_versions') or []
    if not versions:return []
    lines = ['### 数据资产与使用情况', '',
        '原始数据保留，处理结果单独登记。下表分开记录数据数量和训练曝光；未知值不按零填写。', '',
        '| 版本 / 类型 | 原始来源 | 数量及范围 | 健康检查 | 大小（字节） | 证据 |',
        '| --- | --- | --- | --- | ---: | --- |']
    kinds = {'unknown':'来源未登记','existing':'已有数据','official':'官方来源（已审核）',
             'collected':'原生采集','processed':'可逆处理结果','synthetic':'合成数据'}
    points = []
    for version in versions[:64]:
        ident = version['id']
        if not isinstance(ident,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',ident):continue
        profile = version.get('profile') or {}
        scope = {'full':'全量检查','sample':'抽样检查，非全量','metadata':'原生元数据'}.get(profile.get('scope'),'未检查')
        units = {'episodes':'条 episode','transitions':'条 transition','samples':'个 sample',
                 'successful_episodes':'条成功 episode','failed_episodes':'条失败 episode','invalid_episodes':'条无效 episode'}
        counts = profile.get('counts') or {}
        quantities = '；'.join(f'{count} {units[key]}' for key,count in counts.items() if key in units)
        issues = profile.get('issues') or {}
        health = '；'.join(f'{name}：{count}' for name,count in issues.items()) or '未报告检查项'
        if profile.get('issues_categories_omitted'):
            health += '；其余检查项见数据卡'
        parents = '、'.join(version.get('parents') or []) or '未登记父版本'
        refs = f'[数据卡](data_versions/{ident}.json)'
        producer = version.get('producer_ref') or ''
        if re.fullmatch(r'evidence/[0-9a-f]{32}\.json',producer):refs += f' · [生产证据]({producer})'
        lines.append(f"| {cell(ident)} / {cell(kinds.get(version.get('kind'), '来源未登记'))} | "
                     f"{cell(parents)} | {cell(quantities or '数量未知')}；{cell(scope)} | "
                     f"{cell(health)} | {cell(version.get('bytes'))} | {refs} |")
        count = counts.get('episodes')
        if profile.get('scope') == 'full' and isinstance(count,int) and not isinstance(count,bool) and count>=0:
            points.append((ident,count))
    lines += ['', '检查结果来自经源码审核的原生报告；条数和无异常记录本身不证明质量或提分。', '']
    funnels = [row for row in versions if (row.get('profile') or {}).get('funnel')]
    if funnels:
        lines += ['#### 新数据走到了哪一步', '',
            '请求数量和实际产出分开；作业超时不抹掉完整轨迹。格式通过不等于专家监督、来源独立或训练有效。', '',
            '| 数据版本 | 请求 | 尝试 | 完整提交 | 格式通过 | 选择保留 | 来源／监督缺口 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
        descriptions, diagrams = [], []
        for row in funnels:
            profile = row['profile']; funnel = profile['funnel']; known = profile.get('manifest') or {}
            chosen = (profile.get('selection') or {}).get('counts', {}).get('selected', '未知')
            gap = []
            if known.get('unknown_source_families'):
                gap.append(f"{known['unknown_source_families']}条的来源族未确认")
            if known.get('unknown_supervision'):
                gap.append(f"{known['unknown_supervision']}条的监督类型未确认")
            lines.append(f"| {cell(row['id'])} | {cell(funnel.get('requested_episodes','未知'))} | "
                f"{cell(funnel.get('attempted_episodes','未知'))} | {cell(funnel.get('committed_episodes','未知'))} | "
                f"{cell(funnel.get('format_valid_episodes','未知'))} | {cell(chosen)} | {cell('；'.join(gap) or '见数据卡')} |")
            if row.get('partial_output'):
                descriptions.append(f"{cell(row['id'])}来自未完成的采集作业：原失败状态保留，当前登记的是另行核验的完整产物，未将作业改成成功。")
            labels = [('requested_episodes','请求'), ('attempted_episodes','实际尝试'),
                      ('committed_episodes','完整提交'), ('format_valid_episodes','格式通过')]
            points_for_version = [(label, funnel[key]) for key,label in labels if type(funnel.get(key)) is int]
            if type(chosen) is int:
                points_for_version.append(('选择保留', chosen))
            if len(points_for_version) >= 2:
                ref = f"report/charts/{snapshot['revision'][:24]}-{row['id']}-data-funnel.svg"
                atomic_text(root/ref, bar_svg(points_for_version, row['id']+' 数据产出', 'episode'))
                diagrams.append(f"![{cell(row['id'])} 数据产出漏斗]({ref})")
        lines += ['', *descriptions, '', *diagrams, '']
    if len(points) >= 2:
        relative = f"report/charts/{snapshot['revision'][:24]}-data-episodes.svg"
        atomic_text(root/relative, bar_svg(points, '数据规模对比（仅全量 episode 计数）', 'episode'))
        lines += [f'![已登记数据数量对比]({relative})', '',
                  '父版本与派生版本可能包含重叠轨迹，不能相加为新增数据量。', '']
    measured = [row for row in snapshot.get('measurements', []) if row.get('data_versions')]
    if measured:
        lines += ['#### 哪些数据真正进入训练', '',
            '样本次包含重复读取，不能当作新增样本数；来源份额是实际读取计数，不是配置中的混合权重。', '',
            '| 实验 | 数据版本 | 实读样本次 | 消费阶段 | batch 数 / 不重复 sample | 实际来源曝光 | 对应开发指标 |',
            '| --- | --- | ---: | --- | --- | --- | --- |']
        for row in measured:
            reads = row.get('data_exposure') or []
            for ident in row['data_versions'][:16]:
                witness = next((item for item in reads if item.get('data_version_id') == ident), {})
                source = '；'.join(f'{k}：{v}' for k,v in (witness.get('source_samples') or {}).items()) or '未记录'
                work = f"{witness.get('batches','未知')} / {witness.get('unique_samples','未知')}"
                stage = ('进入实际训练batch' if witness.get('consumption_stage') == 'training_batch' else
                         '样本读取，可能含预取' if witness.get('consumption_stage') == 'sample_read' else '旧回执未区分')
                value = '候选来源无效' if row.get('candidate_validity') == 'source_mismatch' else number(row.get('metric_value'))
                lines.append(f"| {cell(row['label'])} | {cell(ident)} | {cell(witness.get('samples_read','未知'))} | "
                             f"{cell(stage)} | {cell(work)} | {cell(source)} | {cell(row.get('metric_name'))}：{value} |")
        lines.append('')
    previews = snapshot.get('data_previews') or []
    if previews:
        lines += ['#### 真实数据示例', '', '以下是已登记数据的完整轨迹片段，按有界样本展示，不代表总体成功率。', '']
        for row in previews[:3]:
            lines += [f"[播放数据录像](<{row['path']}>) · 数据版本 {cell(row['data_version_id'])} · episode {cell(row['episode_id'])}",
                      '', cell(row['purpose'], 400), '']
            if row.get('preview_ref'):
                lines += [f"![数据轨迹真实帧](<{row['preview_ref']}>)", '']
    return lines


def render(root: Path, snapshot: dict[str, Any], narrative: dict[str, Any]) -> str:
    lines = [START, "## 当前进展", ""]
    lines += [active_candidate_panel(snapshot.get("active_candidate") or {}), ""]
    lifecycle = snapshot.get('experiment_lifecycle') or {}
    metrics = lifecycle.get('metrics') or {}
    if metrics:
        lines += ['### 实验闭环概览', '',
            f"已核验开发分数：{cell(metrics.get('valid_scored_development',0))} 项；"
            f"训练完成、尚待计分：{cell(metrics.get('completed_training_pending_score',0))} 项。", '',
            '范围：近期回执；正式计分仍需核验策略加载、rollout 和指标。', '']
    candidate_rows = lifecycle.get('candidates') or []
    if candidate_rows:
        lines += ['研究问题来自 Agent 记录，不是干预已生效或因果效果已成立的证明。', '',
                  '| 实验 | 研究问题 | 当前结果 | 数据消费 | 下一环节 |',
                  '| --- | --- | --- | --- | --- |']
        edge_names = {'measurement_to_commit':'完成采用与报告，不重复训练',
            'training_to_policy':'修复权重选择或归档，保留已训练产物',
            'data_to_training':'核验真实 loader / batch 消费',
            'policy_to_score':'修复评测连接，保留权重和冻结协议'}
        for candidate in candidate_rows[-8:]:
            ref = str(candidate.get('measurement_ref') or '')
            link = (f"[{cell(candidate.get('label'))}]({ref})" if
                    re.fullmatch(r'[A-Za-z0-9_./-]+',ref) and not Path(ref).is_absolute()
                    and '..' not in Path(ref).parts else cell(candidate.get('label')))
            value = (f"{cell(candidate.get('metric_name'))}：{cell(candidate.get('metric_value'))}"
                     if candidate.get('status') == 'scored' else '尚无核验分数')
            consumption = {'verified':'已核验','unknown':'尚缺实际消费证据',
                'not_applicable':'无需训练数据','not_required':'未声明核验要求'}.get(
                    candidate.get('data_consumption'),candidate.get('data_consumption') or '未知')
            edge = (candidate.get('recovery') or {}).get('edge')
            next_step = edge_names.get(edge,'按原生证据定位失败环节')
            question = (candidate.get('research_note') or {}).get('hypothesis') or candidate.get('idea')
            if candidate.get('status') == 'unverified':
                issues = (candidate.get('verification') or {}).get('issues') or []
                next_step = '先复核现有回执，不重复训练' + ('：' + str(issues[0]) if issues else '')
            lines.append(f"| {link} | {cell(question or '尚未记录', 180)} | {value} | {cell(consumption)} | {cell(next_step)} |")
        lines.append('')
    transactions = lifecycle.get('transactions') or []
    if transactions and transactions[-1].get('status') in {'running','interrupted'}:
        latest = transactions[-1]
        phase_names = {'planned':'准备提交实验', 'train':'执行训练', 'train_returned':'训练操作已返回',
            'evaluate':'执行冻结评测', 'evaluate_returned':'评测操作已返回',
            'prepare_data':'准备训练输入', 'collect':'采集数据',
            'reconcile_existing_ownership':'等待核对中断作业的真实状态'}
        lines += [f"最近实验 `{cell(latest.get('label'))}`："
                  f"{cell(phase_names.get(latest.get('phase'),'核对原生阶段回执'))}。", '']
    reference = snapshot.get('baseline_reference') or {'status':'not_declared'}
    reference_status = {'not_declared':'尚未核验', 'ready':'来源已审核，待实测',
        'evaluating':'实测中', 'reproduced':'达到已审核参考值',
        'performance_mismatch':'实测与参考值不一致', 'evaluation_failed':'评测失败',
        'unavailable':'资源不可获得（已审核）', 'unverified':'证据需要修复',
        'rejected':'来源审核未通过'}.get(reference.get('status'), reference.get('status'))
    lines += ['### 基线复现状态', '',
        '本地 baseline 的有效测量不等于官方性能复现；原基线与历史不重置。', '',
        f"已发布基线：{cell(reference_status)}。来源审核不等于对远端权重字节的独立认证。", '']
    if reference.get('expected_metric') is not None:
        lines += ['| 已发布参考值 | 实测值 | 样本数 |', '| --- | --- | --- |',
            f"| {cell(reference.get('expected_metric'))} | {cell(reference.get('measured_metric'))} | {cell(reference.get('samples'))} |", '']
    demo = snapshot.get("environment_demo") or {}
    experiments=(snapshot.get('experiment_lifecycle') or {}).get('experiments') or []
    if experiments:
        lines += ['### 长期训练与正式采用', '',
            '训练结束不等于已经计分：必须由主 Agent 选择采用，核验策略加载并执行冻结评测。', '',
            '| 候选 | 作业状态 | 正式指标 | 证据 |', '| --- | --- | ---: | --- |']
        names={'completed':'训练完成，待采用','running':'训练中','queued':'排队中',
               'adoption_blocked':'训练完成，但源码身份不满足正式采用条件',
               'pending':'待启动','scored':'正式测量已核验','unscored':'尚未获得有效正式分数',
               'failed':'作业失败','cancelled':'已取消'}
        for job in experiments:
            ref=job.get('measurement_ref') or job.get('request_ref')
            lines.append(f"| {cell(job.get('idea_label'))} | {cell(names.get(job.get('status'),job.get('status')))} | "
                         f"{cell(job.get('metric_value'))} | [查看]({ref}) |")
        lines.append('')
    lines += data_panel(root, snapshot)
    lines += campaign_panel(root, snapshot)
    if demo.get("status") == "recorded_unscored":
        lines += ["### 环境预览（不计分）", "",
                  "这是 Agent 提交并由原生执行器录制的预览。任务与策略身份尚未审计，不能用它证明得分或 SOTA。", "",
                  f"观察目的：{cell(demo.get('purpose'))}", ""]
        for media in demo.get("media", []):
            relative = media.get("path", "")
            if (re.fullmatch(r"[A-Za-z0-9_./-]+", relative) and
                    not Path(relative).is_absolute() and ".." not in Path(relative).parts):
                lines.append(f"[播放环境预览]({relative})")
            preview = media.get("preview") or {}
            frame = str(preview.get("path") or "")
            if (re.fullmatch(r"[A-Za-z0-9_./-]+", frame) and not Path(frame).is_absolute()
                    and ".." not in Path(frame).parts):
                lines.append(f"![实际录制帧，非性能证据]({preview['path']})")
        lines.append("")
    status = snapshot.get("status") or "unknown"
    localized = {"running": "运行中", "completed": "运行已结束", "adaptation_unresolved": "接入未完成，已停止",
                 "budget_exhausted": "预算边界触发，已停止", "stopped": "已停止", "internal_error": "内部错误，已停止",
                 "budget_waiting": "模型额度被用量预留占用，等待核清后续跑（非实际费用已耗尽）",
                 "infrastructure_blocked": "运行基础设施／框架阻塞，需核对具体故障后续跑",
                 "created": "已创建", "paused": "已暂停", "action_limit": "局部动作额度已到，已暂停"}.get(status, status)
    provider_blocker = snapshot.get('model_provider_blocker') or {}
    if status == 'infrastructure_blocked' and provider_blocker.get('category'):
        localized = '模型 API 账户／额度阻塞，需处理上游服务后续跑'
    rows = snapshot["measurements"]
    measured = sum(finite(row.get("metric_value")) and row.get('candidate_validity') != 'source_mismatch' for row in rows)
    lines += [f"状态：**{cell(localized)}**。已记录 {len(rows)} 份开发测量记录，其中 {measured} 份有指标且未被来源核验判无效。", "",
              '同一次原生评测的失败记录和评分恢复记录分别保留；恢复记录不代表新增 GPU 试跑。', '']
    if provider_blocker.get('category'):
        reason = ('上游模型 API 返回 HTTP 402（支付／账户额度拒绝），不是本地实验预算耗尽或仿真环境失败。'
            if provider_blocker['category'] == 'provider_payment_required' else
            '上游模型 API 返回 HTTP 401／403（账户访问拒绝），需要处理端点访问权限。')
        lines += [reason+' 不继续重试同一端点或通过其他角色重复请求；未知用量预留保留。', '',
                  f"[查看上游拒绝证据]({provider_blocker['pricing_ref']})", '']
    incumbent = next((row for row in rows if row.get('is_incumbent') is True
                      and finite(row.get('metric_value'))), None)
    if incumbent:
        baseline = next((row for row in rows if row.get('label')=='baseline'), {})
        lines += [f"当前已采用最佳：**{cell(incumbent.get('metric_name'))} = "
                  f"{cell(incumbent['metric_value'])}**，样本数 {cell(incumbent.get('samples'))}。", "",
                  f"原始基线：{cell(baseline.get('metric_value', '未测得'))}；"
                  f"最佳测量：[查看实验]({incumbent['measurement_ref']})。这是开发结果。", ""]
    fault = snapshot.get("runtime_failure") or {}
    recovery = snapshot.get("runtime_recovery") or {}
    terminal_action = (snapshot.get("actions") or [{}])[-1]
    name, result_label, detail = action_explanation(terminal_action)
    active_action = status == "running" and bool(snapshot.get("current_action"))
    if active_action:
        lines += [f"正在执行：**{cell(operation_name(snapshot['current_action']))}**。", "",
                  "下方是最近已返回的操作记录，不是当前动作的结果或停止原因。", ""]
        action_heading = "最近已返回的操作（历史）"
    else:
        action_heading = "最近一次操作"
    lines += [f"{action_heading}：**{cell(name)}**；状态：**{cell(result_label)}**。", ""]
    if detail:
        if active_action:
            lines += ["<details>", "<summary>历史操作说明摘录（按需展开）</summary>", "",
                      cell(detail, 700), "", "</details>", ""]
        else:
            lines += [f"具体说明：{cell(detail, 700)}", ""]
    findings = snapshot.get('monitor_findings') or []
    if findings:
        lines += ['### 外部监视诊断（不是原生实验结果）', '',
                  '以下反例由操作者检查时产生，不是系统自主发现，也不代表原生训练崩溃或正式得分。'
                  '主 Agent 需要独立核验；修正应进入新的审核候选，原基线与历史保留。', '',
                  '| 关联尝试 | 诊断说明 | 独立证据 |', '| --- | --- | --- |']
        for finding in findings:
            lines.append(f"| {cell(finding['native_attempt_id'][:10])} | "
                         f"{cell(finding['summary'], 700)} | [查看反例]({finding['evidence_ref']}) |")
        lines.append('')
    faults = [r for r in snapshot.get('actions', []) if r.get('outcome') in {'raised', 'rejected', 'failed'}]
    if faults:
        lines += ['### 最近故障（执行回执）', '', '| 操作 | 故障 | 证据 |', '| --- | --- | --- |']
        for failure in faults[-3:]:
            reason = str(failure.get('because') or '')
            summary = ('命令组装／参数合同拒绝，未进入原生训练' if
                       'produced an ambiguous command:' in reason else
                       '修复申请的证据 ID 无效，尚未进入修复执行' if
                       'unsafe evidence id' in reason else reason)
            ref = str(failure.get('receipt_ref') or '')
            link = f'[查看回执]({ref})' if re.fullmatch(r'action_receipts/[0-9a-f]{16}\.json', ref) else '见操作记录'
            lines.append(f'| {cell(action_explanation(failure)[0])} | {cell(summary, 250)} | {link} |')
        lines.append('')
    native = snapshot.get('native_stage_verifications') or []
    if native:
        facts = snapshot.get('execution_facts') or execution_facts(native,snapshot.get('measurements') or [])
        lines += ['### 实际执行与正式计分', '',
            '下表是近期封存回执，不是完整实验历史。原生试跑成功不等于正式计分；GPU授权不等于利用率。', '',
            '| 原生阶段 | 正常退出试跑 | 非零退出 | GPU授权次数 |', '| --- | ---: | ---: | ---: |']
        for stage, counts in sorted(facts['stages'].items()):
            lines.append(f"| {cell(stage)} | {counts['completed']} | {counts['failed']} | {counts['gpu_granted']} |")
        lines += ['',f"正式计分测量：{facts['formal_scored_measurements']} 项。命令验收与正式计分分开统计。",'']
        lines += [native_table(native), '']
    queue = snapshot.get('environment_queue') or {}
    if queue.get('prefix'):
        lines += ['### 当前环境与安装待办', '',
            f"当前计划 Python：{cell(queue.get('python'))}；环境验收：{'已通过' if queue.get('passed') else '未通过'}。", '',
            '以下是当前隔离环境的队列，不是历史环境的失败命令。历史研究计划可能过期，以当前回执为准。', '',
            '| 顺序 | 尚未执行的安装命令 |', '| --- | --- |']
        for i, command in enumerate(queue.get('pending') or [], 1):
            lines.append(f'| {i} | {cell(command, 500)} |')
        lines += ['', f"探针游标：{cell(queue.get('next_probe'))} / {cell(queue.get('probe_count'))}；安装完成也不代表能力验收通过。", '']
        reuse = snapshot.get('environment_reuse') or {}
        if reuse.get('mode') == 'overlay':
            lines += ['依赖策略：只读复用已有包，新增/替换包写入本次环境；旧可编辑安装钩子不继承。', '',
                '已连接的源码模块：'+cell('、'.join(b.get('module','') for b in reuse.get('source_bindings', [])), 1000)+'。', '']
    reviews = snapshot.get("review_transactions") or []
    if reviews:
        review = reviews[0]
        label = {"running": "正在进行独立审核", "completed": "独立审核结果已返回，仍需原生执行与复验",
                 "validated": "审核意见及引用已通过校验，尚未证明原生能力恢复",
                 "validation_rejected": "审核意见已返回，但引用或合同未通过校验，需要修正提案",
                 "transport_interrupted": "审核调用中断，保留原提案恢复审核",
                 "transport_unavailable": "同一证据的审核通道恢复未成功，不重复安装"}.get(
                     review.get("status"), "审核记录状态待核对")
        lines += [f"审核进度：{cell(label)}。审核结果不是环境就绪或性能提升的证明。", ""]
        if review.get("validation_errors"):
            lines += [f"需修正：{cell('; '.join(review['validation_errors']), 700)}。", ""]
    proposal_failures = [row for row in snapshot.get("actions") or []
                         if row.get("failure_domain") == "framework_plan"]
    if proposal_failures:
        latest = proposal_failures[-1]
        actions = snapshot.get("actions") or []
        last_index = max(index for index, row in enumerate(actions)
                         if row.get("failure_domain") == "framework_plan")
        resolved = any((row.get("step") == "build_the_environment" and row.get("outcome") in {"done", "already done"}) or
                       (row.get("replayed_step") == "build_the_environment" and
                        row.get("outcome") == "reverified") for row in actions[last_index + 1:])
        installation_started = bool((snapshot.get("provisioning") or {}).get("attempts"))
        crossed = installation_started and any(row.get("step") == "build_the_environment" or
            row.get("replayed_step") == "build_the_environment" for row in actions[last_index + 1:])
        lines += ["### 历史环境方案校验失败（后续已进入原生安装）" if crossed and not resolved else
                  "### 环境方案校验失败（后续已复验通过）" if resolved else "### 环境方案尚未通过校验", "",
                  ("这是历史方案错误，后续已执行原生安装；不代表完整环境就绪，当前阻碍以最新安装与复验回执为准。"
                   if crossed else "后续方案已复验通过。" if resolved else
                   "失败发生在模型方案解析或合同校验阶段。" +
                   ("已有安装操作记录，不能据此声称从未安装；当前方案仍需修正。" if installation_started else
                    "尚未执行原生安装、训练或仿真。") + "应由环境规划器读取被拒绝的方案并修正。"), "",
                  f"具体原因：{cell(latest.get('because'), 1200)}。", ""]
        for ref in latest.get("evidence_refs") or []:
            if re.fullmatch(r"evidence/[0-9a-f]{32}\.json", ref):
                lines += [f"[被拒绝方案及校验错误]({ref})", ""]
    transaction = snapshot.get("recovery_transaction") or {}
    pending = transaction.get("pending") or {}
    trial = transaction.get("latest_revalidation") or {}
    if pending or trial:
        historical = (not pending and not transaction.get('unresolved_failure') and
            trial.get('replayed_step')=='build_the_environment' and
            (snapshot.get('environment_queue') or {}).get('passed') is True)
        lines += ["### 历史复验记录（环境现已通过）" if historical else "### 故障修复与复验", "",
                  ("下表保留当时的复验；后续环境验收已通过，不是当前失败或停止原因。" if historical else
                   "修复建议不等于恢复成功；下面记录原操作是否实际重新执行。"), "",
                  "| 项目 | 当前证据 |", "| --- | --- |"]
        if pending:
            owner = {"environment_planner": "环境规划器", "environment_executor": "环境执行器 / 安装修复 Fix", "source_fix": "源码 Fix"}.get(pending.get("owner"), pending.get("owner"))
            lines += [f"| 待复验原操作 | {cell(pending.get('original_step'))} |",
                      f"| 修复责任 | {cell(owner)} |",
                      f"| 下一操作 | {cell(pending.get('next_action'))}；总预算与准入仍有效 |"]
        if trial:
            outcome = {"reverified": "原操作复验通过", "reverification_failed": "原操作复验仍失败",
                       "revalidation_in_progress": "安装已有进展，完整复验尚未完成"}.get(trial.get("outcome"), trial.get("outcome"))
            if historical and trial.get('outcome') != 'reverified':
                outcome = '当时未通过；后续环境验收已通过'
            lines += [f"| 最近复验 | {cell(trial.get('replayed_step'))}：{cell(outcome)} |",
                      f"| 原操作结果 | {cell(trial.get('original_outcome'))}；不代表正式指标提升 |"]
        lines.append("")
    unresolved = transaction.get("unresolved_failure") or {}
    if unresolved.get("failure_domain") in {"native_install", "native_probe"}:
        probe_failure = unresolved.get("failure_domain") == "native_probe"
        lines += ["### 原生能力探针尚未通过" if probe_failure else "### 原生安装故障尚未解除", "",
                  ("能力探针失败已封存；解释器存在或包已安装不代表实际任务可用。" if probe_failure else
                  "安装命令失败已封存；中间修复命令成功不等于完整环境就绪。") +
                  "应修正安装操作并完成能力探针，不能将未使用的重试额度当作已恢复。", "",
                  f"失败操作证据 ID：{cell(unresolved.get('evidence_id'))}。", ""]
        if transaction.get("status") == "unresolved_requires_new_evidence":
            lines += ["相同操作的有界复验未恢复；需要新的源码/错误证据或明确框架边界，不能反复执行同一错误命令。", ""]
    if terminal_action.get("step") == "stop":
        if status == 'running':
            lines += ['<details>', '<summary>上次停止说明（历史，当前已续跑）</summary>', '',
                      f"历史调度说明：{cell(terminal_action.get('because') or terminal_action.get('why'), 1500)}。", '',
                      '当前状态和本轮故障看实时动作及原生回执，不沿用该历史停止判断。', '',
                      '</details>', '']
        else:
            lines += ["### 为什么停止", "",
                      f"调度 Agent 的停止说明：{cell(terminal_action.get('because') or terminal_action.get('why'), 1500)}。", "",
                      "停止说明是 Agent 的判断；应结合下面的具体执行证据核对，不能单凭它认定资源不足。", ""]
    if fault.get("role") == "recorder" and status not in {"completed"}:
        lines += ["### 报告写作回合异常（不代表主实验停止）", "",
            f"记录 Agent 的历史异常：{cell(fault.get('message'), 700)}。", "",
            "这是写作子任务的证据，不能据此认定 Scheduler、安装、训练或仿真已停止。"
            "主流程是否推进应看原生回执与任务进程；本页心跳刷新也不证明实验有进展。", ""]
        relative = fault.get("evidence_ref") or ""
        if re.fullmatch(r"evidence/[0-9a-f]{32}\.json", relative):
            lines += [f"[写作异常证据]({relative})", ""]
    elif fault and status not in {"completed"}:
        result = recovery.get("recovery") or {}
        recovery_status = {"revalidated": "已通过原安全检查，尚需恢复实际调度",
                           "blocked": "未恢复，需要处理具体原因", "unsupported": "尚无安全恢复操作"}.get(
                               result.get("status"), "尚无已核验恢复")
        historical = status == 'running'
        lines += ["### 历史框架 / 运行故障记录" if historical else "### 框架 / 运行故障与恢复", "",
            *(["当前记录状态为运行中；以下保留的是历史封存错误和当时的恢复判断。"
                "不能据此认定当前已停止或仍因同一故障阻塞；实际进展须核对实时操作、回执与进程。", ""]
               if historical else []),
            *( ["安全检查发现源码副本内的外部链接或疑似敏感文件，阻止了 Agent 启动；"
                "这不表示模型上下文耗尽或 benchmark 不可运行。", ""]
               if fault.get("category") == "workspace_guard" else []),
            f"最近封存的具体错误：{cell(fault.get('message'), 700)}。", "",
            *(["这是模型规划回合中断，不是原生安装、reset、训练或 rollout 失败。"
                "修复应先检查实际回合时限与已封存的工具进展，不能据此判定仿真仓库不可运行。", ""]
              if fault.get("native_operation_status") == "not_inferred" else []),
            f"{'历史恢复判断' if historical else '恢复状态'}：{cell(recovery_status)}；"
            f"说明：{cell(result.get('reason') or (result.get('answer') or {}).get('reason') or '以封存证据为准', 700)}。", "",
            "这是启动/恢复证据，不等同于仿真不可实现，也不等同于总预算耗尽。", ""]
        relative = fault.get("evidence_ref") or ""
        if re.fullmatch(r"evidence/[0-9a-f]{32}\.json", relative):
            lines += [f"[完整错误证据]({relative})", ""]
    current_narrative = narrative.get("event_revision") == snapshot["event_revision"]
    if narrative.get("sections") and current_narrative:
        lines += [_narrative_scope(narrative,snapshot.get('status'),snapshot.get('current_action'),root), '']
        for key, title in zip(FIELDS, ("正在做什么", "为什么做", "最新发现", "当前阻碍", "下一步")):
            row = narrative["sections"][key]
            lines += [f"**{title}：**{cell(row['text'], 700)}", ""]
        lines += [f"以上为模型解释，不替代下方实测数据；依据 [写作快照](report/snapshots/{narrative['revision']}.json)，"
                  f"写于 {cell(narrative.get('written_at'))}。", ""]
        if narrative["revision"] != snapshot["revision"]:
            lines += ["图表/运行状态已有更新；叙述仍基于上述快照，未将新遥测解释为新研究结论。", ""]
    else:
        action = snapshot.get("current_action")
        lines += [f"当前动作：{cell(operation_name(action) if action else '没有已确认的活动动作')}。", "",
                  "研究解释尚未更新，不能从心跳推断实验成功。以下是上次保存的计划，可能已过时；其中的问题和后续动作不是当前故障或当前调度指令。", ""]
        plan = snapshot.get("plan") or {}
        lines += [f"研究目的（已记录计划）：{cell(plan.get('objective') or '尚未记录')}。", "",
                  f"上次计划中的问题（未复核当前状态）：{cell('；'.join(plan.get('open_questions') or []) or '尚未记录')}。", "",
                  f"上次计划中的后续动作（不是当前指令）：{cell('；'.join(plan.get('next_actions') or []) or '尚未记录')}。", ""]
        if narrative.get("sections"):
            lines += [f"上一份叙述已过期，见 [历史说明](report/narratives/{narrative['event_revision']}.json)。", ""]
    model = snapshot.get("model_budget") or {}
    data_strategy = (snapshot.get("plan") or {}).get("data_strategy") or {}
    if data_strategy:
        lines += ["### 数据提分计划（尚非采集结果）", "",
                  "这是最近保存的数据计划，不证明当前仍采用该路线；实际采集与消费看原生回执和数据表。", "",
                  f"选择路径：{cell(data_strategy.get('route'))}；原因：{cell(data_strategy.get('why'), 700)}。", "",
                  "| 需要补什么 | 计划依据 |", "| --- | --- |",
                  *[f"| 目标 {index + 1} | {cell(target, 700)} |" for index, target in enumerate(data_strategy.get("targets") or [])],
                  "", f"生产者到训练器：{cell(' → '.join(data_strategy.get('producer_to_loader') or []), 1000)}。", "",
                  f"下一次小试：{cell(data_strategy.get('next_probe'), 700)}。", "",
                  "以上是 Agent 计划；只有原生采集回执、实际 loader 消费与后续同协议分数才能证明执行和效果。", ""]
    scheduling = snapshot.get("scheduling") or {}
    if scheduling.get("policy"):
        lines += ["### 后台任务与资源调度", "",
            "主 Agent 决定任务和优先级；执行器负责容量、来源身份和总预算。排队不算执行进展。", ""]
        jobs = scheduling.get("jobs") or []
        if jobs:
            statuses = {"queued": "等待资源", "running": "运行中", "completed": "已完成",
                        "failed": "失败", "cancelled": "已取消", "outcome_unknown": "结果未知"}
            reasons = {"priority_queue": "等待队列优先级", "host_capacity": "等待 CPU/内存容量",
                       "gpu_busy": "GPU 正忙", "gpu_lease_race": "GPU 租约竞争，重新排队"}
            lines += ["| 原生阶段 | 状态 | CPU / 内存 MiB / GPU | 等待原因 | 局部窗口（秒） |",
                      "| --- | --- | --- | --- | --- |"]
            for job in jobs:
                resources = job.get("resources") or {}
                lines += [f"| {cell(job.get('stage'))} | {cell(statuses.get(job.get('status'), job.get('status')))} | "
                    f"{cell(resources.get('cpu'))} / {cell(resources.get('memory_mib'))} / {cell(resources.get('gpu'))} | "
                    f"{cell(reasons.get(job.get('wait_reason'), job.get('wait_reason') or '无已记录等待'))} | {number(job.get('window_seconds'))} |"]
            lines += [""]
        agents = scheduling.get("agents") or []
        if agents:
            lines += ["| 只读调查角色 | 状态 | 源码是否已变更 |", "| --- | --- | --- |"]
            for task in agents:
                lines += [f"| {cell(task.get('role'))} | {cell(task.get('status'))} | {'是，须复核' if task.get('stale') else '未标记变更'} |"]
            lines += [""]
        timing = scheduling.get("timing") or {}
        if timing:
            titles = {"controller_action": "主流程操作", "model_turn": "模型回合（含 admission）",
                "readonly_task": "只读调查", "recorder": "报告写作", "resource_wait": "原生资源等待",
                "graph_resource_wait": "依赖图资源等待", "screening_trial": "筛选实验", "formal_measurement": "正式测量"}
            lines += ["| 调度事件 | 次数 | 累计秒数 |", "| --- | --- | --- |"]
            for kind, row in sorted(timing.items()):
                lines += [f"| {cell(titles.get(kind, kind))} | {row['count']} | {number(row['seconds'])} |"]
            lines += ["", "以上耗时可能重叠，不能相加当作墙钟耗时或 GPU 利用率。", ""]
    if snapshot.get("screening"):
        lines += ["### 低成本候选筛选（不计入正式提升）", "",
            "仅比较同保真档位；低预算结果需返回完整研究流程，再做独立确认。", "",
            "| 候选 | 档位（从零开始） | 状态 | 筛选指标 |", "| --- | --- | --- | --- |"]
        for trial in snapshot["screening"]:
            lines += [f"| {cell(trial.get('idea_label'))} | {cell(trial.get('rung'))} | {cell(trial.get('status'))} | {number(trial.get('metric_value'))} |"]
        lines += [""]
    if model.get("limit_usd") is not None:
        spent = float(model.get("spent_usd") or 0)
        held = float(model.get("held_usd") or 0)
        limit = float(model["limit_usd"])
        lines += ["### 模型费用预算", "",
            "| 总上限 | 已结算 | 在途/未知用量预留 | 可用余额 |",
            "| --- | --- | --- | --- |",
            f"| ${limit:.2f} | ${spent:.5f} | ${held:.5f} | ${max(0, limit-spent-held):.5f} |", "",
            "局部额度不足不代表总预算耗尽；预留不是实际消费。GPU 与墙钟预算独立计算。", ""]
        lines += [f"尚未结算回合预留：${float(model.get('pending_reserved_usd') or 0):.5f}；"
                  f"未知用量保留：${float(model.get('unknown_reserved_usd') or 0):.5f}。"
                  "未结算不等于进程仍活跃；没有可靠用量证据不能清零。", "",
                  f"账本更新时间：{cell(model.get('updated_at') or '未记录')}。", ""]
        if snapshot.get('status') == 'budget_waiting' and limit-spent-held > 1e-9:
            lines += ['账本已有可用余额；这不代表已停止的实验自动恢复，需核对服务或明确续跑。','']
    recovery_resources = snapshot.get("installation_recovery_inventory") or {}
    if recovery_resources:
        wheel_count = len(recovery_resources.get("local_wheels") or [])
        candidates = len(recovery_resources.get("environment_candidates") or [])
        lines += ["### 安装恢复可用资源", "",
            f"本轮缓存发现 {wheel_count} 个 wheel，另有 {candidates} 个环境复用候选。"
            "这是恢复线索，不是已安装、兼容或仿真就绪的证明。下载源不可达不等于本地资源不存在。", "",
            "[资源清单与核验要求](installation_recovery_inventory.json)。"
            "是否复用及采用哪条安装路径由 Agent 根据失败证据决定。", ""]
        if recovery_resources.get("scan_complete") is not True:
            lines += ["清单扫描未完成，不能据此断言没有其他缓存资源。", ""]
    repairs = (snapshot.get("harness_repair") or {}).get("repairs") or []
    if repairs:
        lines += ["### 框架修复候选（未部署）", "",
            "Fix 只在独立副本提出补丁。语法通过不代表原故障已修好；"
            "必须独立复验原操作和回归测试。当前不会自动替换运行中的 Python 框架。", "",
            "| 候选 | 状态 | 原故障证据 |", "| --- | --- | --- |"]
        for repair in repairs:
            lines += [f"| {cell(repair.get('id'))} | {cell(repair.get('status'))} | {cell(repair.get('failure_evidence_id'))} |"]
        lines += [""]
    attempts = (snapshot.get("provisioning") or {}).get("attempts") or []
    reuse = snapshot.get("environment_reuse") or {}
    if reuse.get("enabled"):
        if reuse.get('mode') == 'overlay':
            reuse_text = '只读依赖连接已建立，仍需原生探针' if reuse.get('clone_ok') else '只读依赖连接未确认成功'
        else:
            reuse_text = '已复制，仍需原生探针' if reuse.get('clone_ok') else '未确认复制成功'
        publication_status = reuse.get("snapshot_status") or reuse.get("publication_status") or "not_published"
        publication_text = {
            "published": "已发布（后续运行仍须复验）", "already_present": "已有同身份快照",
            "not_cacheable": "环境不适合独立封存", "capacity_skipped": "空间不足，未发布",
            "incomplete_snapshot": "已有未完成副本，保留待检查", "clone_failed": "复制失败，已保留证据",
            "deadline_reached": "剩余时间不足，未发布", "cache_unavailable": "缓存不可用，未发布",
            "disabled": "未启用", "not_published": "尚未发布",
            "template_only": "已保存底座与增量模板，不复制大环境（后续仍须复验）",
        }.get(publication_status, publication_status)
        lines += ["### 环境复用", "",
            f"可供安装器使用的共享 wheel：{reuse.get('available_wheels', 0)} 个（不是实测命中数）。",
            f"基础环境选择：{cell(reuse.get('selected_id') or '未选择公共基础环境')}。",
            f"选择依据：{cell(reuse.get('selection_reason') or '尚未记录')}。",
            f"复用状态：{reuse_text}；"
            f"快照发布：{cell(publication_text)}。", "",
            "缓存命中与安装成功都不等于仿真就绪；数据、权重和资源连接独立验证。", ""]
    base_check = reuse.get('base_verification') or {}
    if base_check.get('status'):
        capability_names = {'cuda_compute': 'GPU 运算', 'backward_optimizer': '反向传播与参数更新',
            'weights_roundtrip': '权重保存和重载', 'reset': '初始状态重置',
            'scene_initialize': '场景初始化', 'physics_step': '物理步进',
            'offscreen_frame': '离屏画面', 'python_subprocess': 'Python 子进程', 'local_io': '本地读写'}
        capabilities = '、'.join(capability_names.get(name, name) for name in base_check.get('capabilities') or []) or '尚无通过记录'
        status = {'verified': '已验收', 'stale': '证据已失效，需复验', 'unverified': '未验收'}.get(base_check['status'], base_check['status'])
        lines += [f"选择底座时的运行栈检查：{cell(status)}；能力：{cell(capabilities)}。",
                  '这只说明底层运行栈，不代表本仓库任务、数据采集或训练评测已经通过。', '']
    if attempts:
        lines += ["### 环境准备的实际操作", "",
            "以下操作已经执行；安装成功不等于仿真环境已核验。", "",
            "| 操作 | 命令（截取） | 结果 | 耗时（秒） | 完整证据 |",
            "| --- | --- | --- | --- | --- |"]
        for index, row in enumerate(attempts):
            ref = str(row.get("evidence_ref") or "")
            safe = bool(re.fullmatch(r"evidence/[0-9a-f]{32}\.json", ref))
            link = f"[回执]({ref})" if safe else "未封存"
            lines += [f"| {index+1} | {cell(str(row.get('command') or '未记录')[-180:])} | "
                      f"{'成功' if row.get('ok') else '失败'} | "
                      f"{cell(row.get('seconds', '未记录'))} | {link} |"]
        lines += ["", "[累计安装日志](build.log) · [逐操作进度](provision_progress.json)", ""]
    trigger = read(root, "report/recorder_trigger.json")
    if not narration_allowance(root)['allowed']:
        lines += ['报告写作暂用事实模式：优先保留实验执行和故障恢复额度，表格、图表及已核验媒体仍更新。','']
    if trigger.get("status") == "failed":
        lines += ["写作服务本轮未完成，保留事实展示；[失败原因](report/recorder_trigger.json)。", ""]
    lines += ["## 实验对比", "", "只对同一已记录评测协议、指标、单位与方向计算原始差值；差值不是显著性结论。", "",
              "| 方案 | 主要改动 | 数据 | 已记录阶段耗时 | 开发指标 | 相对基线差值 | 样本数 | 可比性 / 处置 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        direction = {"minimize": "越低越好", "maximize": "越高越好"}.get(row.get("metric_direction"), "方向未记录")
        metric = cell(row.get("metric_name")) + ": " + number(row.get("metric_value")) + "（" + direction + "）"
        delta = number(row.get("delta")) if row.get("comparable") else "不计算"
        seconds = row.get("stage_seconds") or {}
        cost = "；".join((f"{cell(k)} 已复用（原尝试 {number(v)} 秒）" if
                           k in (row.get('reused_stages') or []) else
                           f"{cell(k)} {number(v)} 秒") for k,v in seconds.items() if finite(v)) or "未记录"
        disposition = row.get("disposition") or "未记录保留/回退结论"
        display_label = row['label'] + ((' / ' + str(row['idea'])) if row.get('idea') else '')
        lines.append(f"| [{cell(display_label)}]({row['measurement_ref']}) | {cell(row.get('changes') or '未记录')} | "
                     f"{cell(row.get('data_summary') or '未记录')} | {cost} | {metric} | {delta} | "
                     f"{cell(row.get('samples'))} | {'同协议' if row.get('comparable') else '可比性未确立'}；{cell(disposition)} |")
    if not rows:
        lines += ["", "尚无开发测量。没有分数不等于成功率为零。"]
    lines += ["", "耗时来自原生回执；复用训练不是又训练一次，耗时图按同一原生尝试去重。它不是总成本或 GPU 小时；正式留出结果不参与此开发对比。", ""]
    chosen = {row["kind"]: row["why"] for row in narrative.get("charts", [])} if current_narrative else {
        "comparison": "比较同协议候选与基线的开发指标。", "training": "观察训练是否产生学习信号。", "cost": "观察已记录的训练与评测时间花在哪里。"}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("comparable"):
            key = object_digest([row.get("protocol_sha256"), row.get("metric_name"), row.get("metric_unit")])[:16]
            groups.setdefault(key, []).append(row)
    if "comparison" in chosen:
        for key, group in groups.items():
            if len(group) < 2:
                continue
            # Top-level links and immutable filenames bind every plot to this exact snapshot.
            relative = f"report/charts/{snapshot['revision'][:24]}-{key}.svg"
            atomic_text(root / relative, bar_svg([(r['label'], r['metric_value']) for r in group],
                        group[0]['metric_name'], group[0].get('metric_unit') or ""))
            lines += [f"{cell(chosen['comparison'])}", "", f"![同协议开发指标比较]({relative})", ""]
    if "cost" in chosen:
        points = cost_points(rows)
        if len(points) >= 2:
            relative = f"report/charts/{snapshot['revision'][:24]}-cost.svg"
            atomic_text(root / relative, bar_svg(points[:24], "已记录阶段耗时（非完整成本）", "秒"))
            lines += [cell(chosen['cost']), "", f"![已记录阶段耗时]({relative})", ""]
    if "training" in chosen:
        lines += ["### 训练观察", "", cell(chosen["training"]), ""]
        for row in snapshot.get("telemetry_attempts", [])[-3:]:
            ref = row.get("chart_ref")
            if ref and re.fullmatch(r"report/telemetry/[A-Za-z0-9_.-]+\.svg", ref):
                path = root / ref
                expected = snapshot.get("source_revision_vector", {}).get(ref)
                try:
                    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 2 * 1024**2:
                        raise ValueError("unsafe telemetry plot")
                    data = path.read_bytes()
                    actual = hashlib.sha256(data).hexdigest()
                    if actual != expected:
                        raise ValueError("telemetry plot changed after snapshot")
                    frozen = f"report/charts/telemetry-{actual}.svg"
                    if not (root / frozen).exists():
                        atomic_text(root / frozen, data.decode("utf-8"))
                    lines += [f"![原生训练遥测，非正式分数]({frozen})", ""]
                except (OSError, ValueError, UnicodeError):
                    lines += ["训练曲线与本页快照暂不一致，等待下次刷新，不展示错配数据。", ""]
        if not any(row.get("chart_ref") for row in snapshot.get("telemetry_attempts", [])):
            lines += ["尚无可展示的训练曲线：未接收到可用遥测，不能据此判定训练没有学习。", ""]
    lines += ["## 行为演示", ""]
    media = snapshot.get("verified_media", [])
    for row in media[:3]:
        lines += [f"- [查看真实录像](<{row['path']}>)：{cell(row.get('measurement_label') or row.get('trigger'))}；"
                  f"episode：{cell(row.get('episode_id'))}。录像说明行为，不代替统计；未核对初态则不称配对比较。"]
        if row.get("preview_ref"):
            lines += ["", f"![对应录像真实帧](<{row['preview_ref']}>)", ""]
    if not media:
        lines += ["当前没有已核验 demo；可能尚未完成原生录制或媒体核验，不能用旧录像替代。", ""]
        events = snapshot.get('media_events') or []
        if events:
            event = events[-1]
            reason = ('尚未接入原生录像入口，需要 Agent 根据源码补齐录制能力' if
                      event.get('status') == 'unavailable' else event.get('why') or '见录像回执')
            lines += [f"最近录制结果：{cell(event.get('status'))}；{cell(reason, 400)}。", '']
    requests = read(root, "report/demo_requests.json").get("requests", [])
    for row in requests[-6:]:
        lines += [f"- 录制请求 {cell(row.get('id'))}：{cell(row.get('purpose'))}；状态 {cell(row.get('status'))}；"
                  f"{cell(row.get('reason') or '等待 Scheduler 决定预算与执行')}。"]
    links = f"[本页证据快照](report/snapshots/{snapshot['revision']}.json)"
    if narrative:
        links += " · [写作与图表选择](report/narrative.json)"
    lines += ["", links, "", END]
    return '\n'.join(lines)


def _writing_workspace(root: Path, client: Any):
    """A stable empty read-only workspace, with fresh conversations for every review.

    Only presentation uses this workspace. Scientific reader snapshots and their
    isolation are unchanged. Configuration changes allocate another worker and keep
    every old receipt; no session, charge or unknown reservation is discarded.
    """
    from .agent_tasks import snapshot as worker_snapshot
    from .run_record import _report_lock
    import uuid
    configuration = {key: getattr(client, key, None) for key in
        ('run_id', 'model', 'turn_budget_usd', 'total_budget_usd', 'timeout')}
    configuration['workspace'] = str(client.workspace)
    with _report_lock(root / 'report/recorder-workspace'):
        if (root / 'agent_workers').is_symlink():
            raise ValueError('unsafe Recorder worker namespace')
        saved = read(root, 'report/recorder_worker.json')
        identity = saved.get('worker_id')
        if saved.get('configuration') == configuration and isinstance(identity, str) and re.fullmatch(r'[0-9a-f]{32}', identity):
            worker = root / 'agent_workers' / identity
            checkout = worker / 'checkout'
            manifest = read(worker, 'workspace_snapshot.json')
            parent = read(worker, 'worker_parent.json')
            session = read(worker, 'agent/session.json')
            if (worker.is_dir() and checkout.is_dir() and not worker.is_symlink()
                    and not checkout.is_symlink() and manifest.get('source_bytes') == 0
                    and manifest.get('reader_source_files') == []
                    and manifest.get('reader_source_identity') is None
                    and manifest.get('destination') == str(checkout)
                    and parent == {'output': str(root.resolve()), 'read_only': True}
                    and session.get('status') != 'running'):
                return worker, checkout
        identity = uuid.uuid4().hex
        worker, checkout = worker_snapshot(root, client.workspace, identity, empty=True)
        atomic_json(root / 'report/recorder_worker.json', {'worker_id': identity,
            'configuration': configuration, 'scope': 'empty read-only presentation workspace'})
        return worker, checkout


def _background_narration(root: Path, snapshot: dict[str, Any], client: Any, timeout: float):
    """A single coalescing worker per run; never hold the main decision thread."""
    from .common import atomic_text
    from .run_record import _report_lock
    import time
    while True:
        with _ASYNC_LOCK:
            request = _ASYNC_PENDING.pop(str(root), None)
            if request is None:
                _ASYNC_RUNNING.discard(str(root))
                return
        snapshot, client, timeout = request
        try:
            worker, checkout = _writing_workspace(root, client)
            writer = client.fork_readonly(output=worker, workspace=checkout, role="recorder")
            began = time.monotonic()
            narrative = narrate(root, snapshot, client=writer, timeout=timeout)
            from .scheduling import note
            note(root, kind="recorder", identity=snapshot["event_revision"],
                 seconds=time.monotonic()-began, status=narrative.get("status"))
            latest = read(root, "report/async_request.json")
            if latest.get("event_revision") != snapshot["event_revision"]:
                continue
            # Writing itself settles additional model usage. Rebuild live facts after
            # settlement, without another model call or a new narrative event.
            view = read(root, "report/view.json")
            if view:
                snapshot = make_snapshot(root, view, read(root, "report/context.json"))
            presentation = render(root, snapshot, narrative)
            with _report_lock(root / "RUN.md"):
                if read(root, "report/async_request.json").get("event_revision") != snapshot["event_revision"]:
                    continue
                destination = root / "RUN.md"
                if destination.is_file():
                    old = destination.read_text()
                    if START in old and END in old:
                        before, rest = old.split(START, 1)
                        _, after = rest.split(END, 1)
                        atomic_text(destination, before + presentation + after)
                        from .run_record import _publish_presentation_html
                        _publish_presentation_html(destination, before + presentation + after)
        except Exception as exc:
            atomic_json(root / "report/async_error.json", {"at": now(), "error": type(exc).__name__})


import threading as _threading
from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
_ASYNC_POOL = _ThreadPoolExecutor(max_workers=2, thread_name_prefix="autosim-recorder")
_ASYNC_LOCK = _threading.Lock()
_ASYNC_PENDING: dict = {}
_ASYNC_RUNNING: set = set()


def refresh(root: Path, view: dict[str, Any], *, context: dict[str, Any] | None = None,
            client: Any = None, timeout: float = DEFAULT_TIMEOUT) -> str:
    if context is None:
        context = read(root, "report/context.json")
    else:
        atomic_json(root / "report" / "context.json", context)
    snapshot = make_snapshot(root, view, context)
    allowance = narration_allowance(root)
    if not allowance['allowed']:
        atomic_json(root/'report/recorder_deferred.json', {**allowance,'at':now(),'mode':'fact_only'})
        return render(root,snapshot,read(root,'report/narrative.json'))
    from .scheduling import policy
    asynchronous = policy(root).get("async_recorder") and hasattr(client, "fork_readonly")
    if asynchronous:
        atomic_json(root / "report/async_request.json", {
            "event_revision": snapshot["event_revision"], "at": now()})
        previous = read(root, "report/recorder_trigger.json")
        if previous.get("event_revision") != snapshot["event_revision"]:
            with _ASYNC_LOCK:
                _ASYNC_PENDING[str(root)] = (snapshot, client, timeout)
                if str(root) not in _ASYNC_RUNNING:
                    _ASYNC_RUNNING.add(str(root))
                    _ASYNC_POOL.submit(_background_narration, root, snapshot, client, timeout)
        narrative = read(root, "report/narrative.json")
    else:
        narrative = narrate(root, snapshot, client=client, timeout=timeout)
    return render(root, snapshot, narrative)
