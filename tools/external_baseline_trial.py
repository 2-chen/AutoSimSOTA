#!/usr/bin/env python3
"""Run upstream coding-agent research instructions on isolated native repositories.

Only isolation, provider accounting, and process supervision are shared with AutoSim.
The upstream instructions and the coding agent choose and implement the experiments.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "autoresearch_runs/external_systems_20261016_v1"
REFERENCES = ROOT / "autoresearch_cache/reference_repos"
ENPIRE = REFERENCES / "NVlabs-ENPIRE-20261016"
AUTORESEARCH = REFERENCES / "jellyheadandrew-autoresearch-robotics-ff30cc4"
OLD_RUN = ROOT / "autoresearch_runs/robosyn_required_data_20261013_v1"
BASE_PYTHON = ROOT / ".venv/bin/python"
SYSTEMS = {"enpire": ENPIRE, "autoresearch_robotics": AUTORESEARCH}
SOURCES = {
    "robosyn": Path("/home/wbc/下载/autoresearch/test/RoboSynChallenge"),
    "robotwin": Path("/home/wbc/下载/autoresearch/test/RoboTwin"),
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(source):
    files = {}
    for directory in ("policy", "scripts", "script", "envs", "robosynchallenge", "configs", "task_config"):
        base = source / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if (path.is_file() and not path.is_symlink()
                    and path.suffix in {".py", ".sh", ".yml", ".yaml", ".json"}
                    and not any(p in {"act_ckpt", "processed_data", "__pycache__", ".git", ".venv"}
                                for p in path.relative_to(base).parts)):
                files[str(path.relative_to(source))] = digest(path)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return {"source": str(source), "commit": commit, "files": files}


def copy_code(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    excludes = [".git", "__pycache__", ".venv", "/assets", "/data", "/lerobot_dataset",
                "/checkpoints", "processed_data", "act_ckpt", "eval_result", "outputs",
                "*.ckpt", "*.safetensors", "*.mp4", "*.hdf5", "*.h5", "*.parquet"]
    subprocess.run(["rsync", "-a", "--max-size=10m", *[f"--exclude={x}" for x in excludes],
                    str(source) + "/", str(destination) + "/"], check=True)


def prepare():
    STUDY.mkdir(parents=True, exist_ok=True)
    if (STUDY / "manifest.json").exists():
        raise RuntimeError("study already prepared; do not replace its inputs")
    prior = json.loads((OLD_RUN / "task_gpu_budget.json").read_text())
    manifest = {"created_at": now(), "scope": "native-port pilot, not unmodified upstream benchmark",
                "model": "deepseek-flash", "model_usd_per_arm": 5,
                "wall_seconds_per_arm": 4500, "maximum_reserved_gpu_seconds_per_repository": 9000,
                "prior_robosyn_gpu_seconds": prior["charged_seconds"],
                "prior_robosyn_gpu_ledger": str(OLD_RUN / "task_gpu_budget.json"),
                "old_ledgers_modified": False, "trials": []}
    if prior["charged_seconds"] + 9000 > prior["cap_seconds"]:
        raise RuntimeError("pilot would exceed inherited RoboSyn GPU envelope")
    for name, source in SOURCES.items():
        save(STUDY / f"source_{name}_before.json", source_identity(source))
    for system, upstream in SYSTEMS.items():
        if not upstream.is_dir():
            raise FileNotFoundError(upstream)
        for benchmark, source in SOURCES.items():
            trial = STUDY / f"{system}_{benchmark}"
            bench = trial / "benchmark"
            copy_code(source, bench)
            for relative in (["assets", "checkpoints", "lerobot_dataset"] if benchmark == "robosyn"
                             else ["assets", "data", "policy/ACT/act_ckpt", "policy/ACT/processed_data"]):
                target, original = bench / relative, source / relative
                if original.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(original, target_is_directory=True)
            for directory in ("home", "tmp", "logs", "outputs", "runtime"):
                (trial / directory).mkdir(parents=True, exist_ok=True)
            runtime = trial / "runtime"
            # Borrow installed dependencies, but prevent an old editable finder from
            # redirecting the new benchmark's imports to a previous experiment.
            (runtime / "sitecustomize.py").write_text(
                "import sys\n"
                "m=sys.modules.get('autosim_overlay_sources')\n"
                "if m is not None:\n"
                "    keep={'embodichain','embodichain_tasks','embodichain_tasks.configs',"
                "'coal','eigenpy','hppfcl','pinocchio'}\n"
                "    for k in list(m.MAPPING):\n"
                "        if k not in keep: del m.MAPPING[k]\n")
            upstream_read = ("enpire/policy/autoresearch_instruction.md" if system == "enpire" else "program.md")
            row = {"name": trial.name, "system": system, "benchmark": benchmark,
                   "upstream": str(upstream), "upstream_instruction": upstream_read,
                   "upstream_instruction_sha256": digest(upstream / upstream_read),
                   "source": str(source), "worktree": str(bench),
                   "python": str(OLD_RUN / "env/bin/python") if benchmark == "robosyn" else str(BASE_PYTHON),
                   "task": "click_bell" if benchmark == "robosyn" else "beat_block_hammer",
                   "setting": "random" if benchmark == "robosyn" else "demo_clean",
                   "starting_checkpoint": str(source / "checkpoints/ACT_sim_click_bell") if benchmark == "robosyn"
                   else str(source / "policy/ACT/act_ckpt/act-beat_block_hammer/baseline"),
                   "development_episodes": 20, "development_seed": 2026,
                   "confirmation_episodes": 50, "confirmation_seed": 2027,
                   "status": "prepared", "model_budget_usd": 5, "wall_limit_seconds": 4500}
            if system == "enpire":
                row["upstream_commit"] = subprocess.check_output(
                    ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
            else:
                row["upstream_commit"] = "ff30cc481af9343b9c96f448b5ed72b75bfcd562"
            save(trial / "inputs.json", row)
            (trial / "INTEGRATION.md").write_text(integration(row), encoding="utf-8")
            save(trial / "state.json", {"status": "prepared", "at": now()})
            manifest["trials"].append(row)
    save(STUDY / "manifest.json", manifest)
    render()


def integration(row):
    framework = ("""ENPIRE 公开入口是 coding agent 配合其 Auto-Research Contract。阅读
Environment/Policy/TrialRunner/ArtifactStore，接入原生仿真及训练器；不用安装真机 PLD。
记录实际调用了哪些上游模块。若仅遵循研究合同并调用原生批评测，明确标为
“ENPIRE 合同驱动的原生适配”，不能说已复现完整真机框架或另一系统的模板。
""" if row["system"] == "enpire" else """原版 autoresearch-robotics 没有本任务模板。
自行创建 prepare.py/evaluate.py/train.py 薄入口，映射到原生训练/评测与策略文件，
遵循其 program.md 的分析→假设→执行→评测/视觉分析→记忆→保留/回退循环。
上游10秒Fetch训练窗口替换为本任务预算内合理时长，记录模板和研究权限变化。
""")
    visual = ("""当前 deepseek-flash 支持图像，代理已显式启用 inline image。
请用 Read 实际读取原生仿真视频提取的 PNG/JPEG 帧，提供基于图像的观察并引用帧路径。
不能把路径当作看过图像。保存少量成功/失败原生演示视频和关键帧，在中文文档展示。
""" if row.get("allow_images") else """本 arm 使用旧文本代理，只能文本反馈。
视觉能力限制来自本次运行配置，不应解释为 deepseek-flash 模型不支持图像。
可生成/展示视频，但不能编造视觉观察。本 arm 单列为文本条件诊断，不与视觉条件混报。
""")
    return f"""# 原生仓库试跑输入

系统：{row['system']}；任务：{row['task']}；配置：{row['setting']}。
上游源码只读路径：{row['upstream']}，版本：{row['upstream_commit']}。
首先完整阅读上游 `{row['upstream_instruction']}`；ENPIRE 同时阅读其 AGENTS.md、
Environment/Policy 接口和 TrialRunner。遵循其自主提案、执行、证据、保留/回退循环。
本地可编辑目标：`benchmark/`，原仓库与已有数据/权重/环境均只读连接。
Python：{row['python']}。PATH 中的 python 即此环境。已有环境优先，无需从零安装。
初始参考权重：{row['starting_checkpoint']}。先核对其架构与归一化，再原生实测。
RoboSyn 是已有发布 ACT 权重；RoboTwin 是本机已有 baseline 训练产物，不能称官方 SOTA。

## 本次接入与研究任务

这是 native-port pilot。你需要自行阅读当前仓库，建立必要的薄适配，并开展一次真实改进。
{framework}
先复用已提供的环境。确实缺失时记录证据和具体需要；不修改共享环境。
ENPIRE 可在本 trial 建独立依赖 overlay；autoresearch-robotics 的禁止新增依赖规则
适用于冻结后的研究修改，原生模板接入时确需补齐的依赖另记为准备成本。

先建立共同起点真实基线，再自主提出并运行至少一个候选，原生评测后保留/回退。
允许优化策略/训练/合法部署参数，以及在原生训练域新增/筛选/混合数据。
是否采集及研究路线由你决定，保存假设、diff、真实命令、日志、策略/数据身份和决定。
每次实验用独立输出目录，严禁覆盖已有数据或 checkpoint。新数据写到本 trial outputs。
先在此隔离 benchmark 内 git init 并提交源码快照，不要操作父项目或源仓库 git。
原生评测的脚本、任务、物理、reset、成功定义和种子不属于研究变量。
RoboTwin test_num 原为写死100，本次只允许准备阶段接入20/50集控制，之后冻结；
不得跳过原生专家有效初态检查，不得改变任务步数/成功条件来提分。

开发评测：固定任务/配置，seed={row['development_seed']}，episodes=20，基线与候选相同。
候选只通过完整20集产生分数；部分进度及异常不是0分。若找到更好候选，先提交最终
选择，再分别对基线和最终候选做 seed={row['confirmation_seed']}、50集确认，确认不再用于搜索。
来不及完成确认则明确尚待确认，绝不能把开发小样本提升写成公开SOTA。
正常训练/评测可后台运行并检查进度，不因一次工具返回未完成就重启。
本 trial 总墙钟75分钟，模型额度$5；这是整个试点预算，单个命令按任务合理安排。
不要耗尽总窗口反复总结。优先真实baseline→候选→评测；最多两个研究候选。
实际时间读取 state.json 的 started_at / elapsed_seconds，并用 date 核对，不凭感觉估计。

## 记录与展示

维护中文 `RESEARCH.md`，包含当前阶段、基线/候选对比表、数据增量、实际结果、
下一步及原生视频链接。机器结果写 `pilot_results.json`，附原生 metrics/log 路径，
不手造分数。没有完成则填 null，并注明阻碍。每次阶段完成更新记录。
保存原生仿真视频；不因媒体失败重复训练或计分。
{visual}
不要访问宿主密钥、修改现有环境、终止其他进程或调用真机；资源都已显式连接。
你已获得开始本试点和上述仿真实验的授权，不等待再次确认。
"""


def sandbox(row, command):
    trial = STUDY / row["name"]
    args = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user", "--unshare-pid",
            "--unshare-ipc", "--ro-bind", "/", "/", "--tmpfs", "/home/wbc", "--tmpfs", "/tmp",
            "--proc", "/proc", "--dev-bind", "/dev", "/dev"]
    mounts = [Path("/home/wbc/miniconda3"), Path("/home/wbc/.local/node"),
              SOURCES[row["benchmark"]], SYSTEMS[row["system"]], ROOT / "EmbodiChain",
              ROOT / ".venv", OLD_RUN / "env", OLD_RUN / ".borrowed_dependencies",
              Path("/home/wbc/.cache/embodichain_data"), Path("/home/wbc/.cache/torch")]
    for path in mounts:
        if path.exists():
            args += ["--ro-bind", str(path), str(path)]
    args += ["--bind", str(trial), str(trial)]
    for relative in (["scripts/eval_policy.py", "robosynchallenge", "configs"] if row["benchmark"] == "robosyn"
                     else ["script/eval_policy.py", "envs", "task_config"]):
        path = trial / "benchmark" / relative
        args += ["--ro-bind", str(path), str(path)]
    args += ["--chdir", str(trial), "--", *command]
    return args


def environment(row, gateway):
    trial = STUDY / row["name"]
    keep = {k: v for k, v in os.environ.items() if k in {
        "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"}}
    keep.update({"HOME": str(trial / "home"), "TMPDIR": str(trial / "tmp"),
                 "PATH": str(Path(row["python"]).parent) + ":/home/wbc/.local/node/bin:/usr/local/bin:/usr/bin:/bin:/home/wbc/miniconda3/envs/robotwin5090/bin",
                 "PYTHONPATH": ":".join(map(str, [trial / "runtime", trial / "benchmark",
                       trial / "benchmark/policy", ROOT / "EmbodiChain", SYSTEMS[row["system"]]])),
                 "LD_LIBRARY_PATH": "/home/wbc/miniconda3/envs/robotwin5090/lib",
                 "ANTHROPIC_API_KEY": gateway.local_token, "ANTHROPIC_BASE_URL": gateway.base_url,
                 "ANTHROPIC_MODEL": "deepseek-flash", "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-flash",
                 "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-flash", "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-flash",
                 "CLAUDE_CONFIG_DIR": str(trial / "home/.claude"),
                 "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1",
                 "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192", "MAX_THINKING_TOKENS": "0",
                 "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                 "HF_HOME": str(trial / "home/.cache/huggingface"),
                 "HF_LEROBOT_HOME": str(trial / "outputs/datasets"),
                 "TORCH_HOME": "/home/wbc/.cache/torch", "WANDB_MODE": "disabled",
                 "EMBODICHAIN_DATA_ROOT": "/home/wbc/.cache/embodichain_data",
                 "XDG_CACHE_HOME": str(trial / "home/.cache"),
                 "MPLCONFIGDIR": str(trial / "home/.config/matplotlib"),
                 "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
                 "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    if row["benchmark"] == "robotwin":
        keep["AUTOSIM_DISABLE_RT"] = "1"
    keep["NO_PROXY"] = "127.0.0.1,localhost"
    return keep


def _trial_file(trial, path):
    """Only local regular artifacts; resource symlinks are not report inputs."""
    try:
        return (path.is_file() and path.stat().st_size <= 2 * 1024**2
                and path.resolve().is_relative_to(trial.resolve())
                and not any(p.is_symlink() for p in (path, *path.parents)
                            if p.is_relative_to(trial)))
    except (OSError, ValueError):
        return False


def submitted_result(trial):
    """Discover native-port submissions without recursive or silent conflict picking.

    A submission is a request for audit, never authority for a scientific score.
    The two explicit roots cover CLI cwd differences without reading data/assets.
    """
    documents, paths = [], []
    for path in (trial / "pilot_results.json", trial / "benchmark/pilot_results.json"):
        if not path.exists() and not path.is_symlink():
            continue
        paths.append(str(path.relative_to(trial)))
        try:
            if not _trial_file(trial, path):
                raise ValueError("unsafe or oversized submission")
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or not value:
                raise ValueError("submission must be a nonempty object")
            documents.append(value)
        except (OSError, ValueError, TypeError):
            return {"status": "invalid", "paths": paths}
    if not documents:
        return {"status": "missing", "paths": []}
    if any(value != documents[0] for value in documents[1:]):
        return {"status": "conflict", "paths": paths}
    return {"status": "ready", "paths": paths, "value": documents[0]}


def native_metrics(trial):
    # Discovery only, not metric admission. Prune invalid attempts and resource
    # links before walking; never traverse a 100GB data tree to find a report.
    paths = set()
    names = {"evaluation_metrics.json", "_result.txt", "metrics.json", "summary.json"}
    for relative in ("benchmark/eval_result", "benchmark/outputs", "outputs",
                     "benchmark/research_out", "research_out"):
        base = trial / relative
        if not base.is_dir() or any(p.is_symlink() for p in (base, *base.parents)
                                   if p.is_relative_to(trial)):
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(("_invalid", "."))
                             and not (Path(directory) / d).is_symlink())
            for name in sorted(names.intersection(files)):
                path = Path(directory) / name
                if _trial_file(trial, path):
                    paths.add(str(path.relative_to(STUDY)))
            if len(paths) >= 500:
                return sorted(paths)[:500]
    return sorted(paths)


def render():
    manifest_path = STUDY / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    lines = ["# 外部自主研究系统试跑", "", f"更新时间：{now()}", "",
             "本次接入 ENPIRE 与 autoresearch-robotics，分别在 RoboSyn click_bell / random 和",
             "RoboTwin beat_block_hammer / demo_clean 上试跑。使用原生训练与评测，源码隔离、资源只读连接。",
             "两者均为明确记录适配的 native-port pilot；结果需区分接入成功、真实提分与独立确认。", "",
             "| 系统 | 仓库 | 反馈条件 | 状态 | 已结算模型费用 | 原生指标文件 | 研究记录 |",
             "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for row in manifest["trials"]:
        trial = STUDY / row["name"]
        state = json.loads((trial / "state.json").read_text())
        costpath = trial / "model_usage.json"
        cost = json.loads(costpath.read_text()) if costpath.exists() else {}
        metrics = native_metrics(trial)
        report = next((p for p in (trial / "RESEARCH.md", trial / "benchmark/RESEARCH.md")
                       if _trial_file(trial, p)), None)
        record = f"[{row['name']}]({report.relative_to(STUDY).as_posix()})" if report else "尚未发现研究记录"
        mode = "图像可用" if row.get("allow_images") else "文本条件诊断"
        lines.append(f"| {row['system']} | {row['benchmark']} | {mode} | {state['status']} | "
                     f"${cost.get('cost_usd', 0):.4f} | {len(metrics)} | {record} |")
    lines += ["", "每个 arm 最多 $5 模型费用、75分钟整个研究窗口，GPU 串行。首个文本条件单列，",
              "追加一个独立 ENPIRE/RoboSyn 图像条件试跑后，RoboSyn上限3.75小时，RoboTwin2.5小时。",
              "新试点费用与旧账分别保存，RoboSyn GPU总额核对旧消耗，",
              "不会清除旧消费或未知预留。实际模型费用由 provider usage 估算，不使用 Claude 的默认美元估价。",
              "", "开发固定20集、seed2026；若有改进，最终策略和基线分别50集、seed2027确认。",
              "未完成的结果保持空缺，阶段性提升不等于公开排行榜SOTA。",
              "指标文件为有界产物索引（每arm最多500项），含诊断与冒烟，未经独立核验不代表有效分数。",
              "状态保留原调度记录，报告链接按当前产物发现；已开放图像不表示实际取得图像反馈。",
              "", "## 原生证据", ""]
    for row in manifest["trials"]:
        for relative in native_metrics(STUDY / row["name"]):
            lines.append(f"- [{relative}](<{relative}>)")
    (STUDY / "RUN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_trial(row):
    sys.path.insert(0, str(ROOT / "autosim"))
    from autosim.llm_client import load_credential_file
    from autosim.research.deepseek_gateway import DeepSeekTurnGateway
    load_credential_file(project_root=ROOT)
    trial = STUDY / row["name"]
    initial = json.loads((trial / "state.json").read_text())
    resuming = initial["status"] == "resume_pending"
    if initial["status"] != "prepared" and not resuming:
        raise RuntimeError(f"refusing implicit replay: {row['name']} is {initial['status']}")
    started, session = time.monotonic(), str(uuid.uuid4())
    prior_usage = {}
    if resuming:
        session = initial["session_id"]
        prior_usage = json.loads((trial / "model_usage.json").read_text())
        save(trial / f"model_usage_before_resume_{len(list((trial / 'logs').glob('agent_*.jsonl')))}.json", prior_usage)
        prior_elapsed = max(0.0, (dt.datetime.now(dt.timezone.utc) -
                                dt.datetime.fromisoformat(initial["started_at"])).total_seconds())
        started -= prior_elapsed
    state = {"status": "starting", "started_at": now(), "session_id": session,
             "supervisor_pid": os.getpid(), "upstream_commit": row["upstream_commit"]}
    if resuming:
        state["started_at"] = initial["started_at"]
        state["resumed_at"] = now()
    prior_cost = float(prior_usage.get("cost_usd", 0))
    prior_unknown = float(prior_usage.get("unknown_reserved_usd", 0)) + float(prior_usage.get("held_usd", 0))
    def checkpoint(value):
        combined = dict(value)
        if prior_usage:
            combined.update(cost_usd=prior_cost + value["cost_usd"],
                unknown_reserved_usd=prior_unknown + value["unknown_reserved_usd"],
                accounting_ceiling_usd=prior_cost + prior_unknown + value["accounting_ceiling_usd"],
                requests=prior_usage["requests"] + value["requests"],
                receipts=prior_usage["receipts"] + value["receipts"], limit_usd=row["model_budget_usd"],
                inherited_unsettled_usd=prior_unknown,
                unknown=bool(value["unknown"] or prior_unknown))
        save(trial / "model_usage.json", combined)
    save(trial / "state.json", state)
    upstream = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    if not upstream.endswith("/anthropic"):
        upstream = upstream.removesuffix("/v1") + "/anthropic"
    gateway = DeepSeekTurnGateway(upstream_base_url=upstream, upstream_key=os.environ["DEEPSEEK_API_KEY"],
                model="deepseek-flash", limit_usd=row["model_budget_usd"] - prior_cost - prior_unknown,
                generation_policy={"max_output_tokens": 8192, "thinking": "disabled"},
                checkpoint=checkpoint,
                allow_images=bool(row.get("allow_images")))
    prompt = ("请完整阅读当前 INTEGRATION.md，并按其中指定的上游研究指令开始本任务。"
              "这是获准的仿真试点，请自主完成适配、真实基线和至少一个候选的原生评测，"
              "维护中文RESEARCH.md与真实证据。不要只制定计划或等待确认。")
    if resuming:
        prompt = ("继续同一个研究会话，先完整阅读 OPERATOR_RECOVERY.md。操作者已停止本trial的旧进程组；"
                  "复用已有源码和产物，按该文件修复配置／适配并复验，不从零调查或清空旧账。"
                  "未完成／配置无效评测不能当0分；配置稳定后重测共同基线，再自主优化。")
    first_log = len(list((trial / "logs").glob("agent_*.jsonl")))
    try:
        with gateway:
            env = environment(row, gateway)
            # Validate isolation and the actual CUDA/import bindings before paying.
            probe = [row["python"], "-c", "import os,json,torch; assert torch.ones(4,device='cuda').sum().item()==4; print(json.dumps({"
                     "'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0),"
                     "'cwd':os.getcwd(),'source_credentials_visible':os.path.exists('" + str(ROOT / ".env") + "')}))"]
            check = subprocess.run(sandbox(row, probe), env=env, capture_output=True, text=True, timeout=90)
            (trial / "logs/preflight.log").write_text(check.stdout + check.stderr)
            if check.returncode or '"cuda": true' not in check.stdout or '"source_credentials_visible": false' not in check.stdout:
                raise RuntimeError("isolated CUDA preflight failed; see logs/preflight.log")
            for turn in range(3):
                if time.monotonic() - started >= row["wall_limit_seconds"]:
                    break
                cli = ["/home/wbc/.local/node/bin/claude", "--bare", "--print", "--verbose",
                       "--model", "deepseek-flash", "--output-format", "stream-json",
                       "--permission-mode", "bypassPermissions", "--dangerously-skip-permissions",
                       "--disable-slash-commands", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                       "--tools", "Bash,Read,Edit,Write,Glob,Grep,TaskOutput,TaskStop",
                       "--settings", '{"attribution":{"commit":"","pr":""}}']
                cli += ["--session-id", session] if turn == 0 and not resuming else ["--resume", session]
                log_index = first_log + turn
                stdout_path = trial / f"logs/agent_{log_index}.jsonl"
                with stdout_path.open("w") as stdout, (trial / f"logs/agent_{log_index}.stderr").open("w") as stderr:
                    process = subprocess.Popen(sandbox(row, cli), env=env, stdin=subprocess.PIPE,
                                               stdout=stdout, stderr=stderr, text=True, start_new_session=True)
                    process.stdin.write(prompt)
                    process.stdin.close()
                    state.update(status="running", agent_pid=process.pid, turn=turn, last_update=now())
                    save(trial / "state.json", state)
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        state.update(elapsed_seconds=elapsed, last_update=now(), native_metrics=native_metrics(trial))
                        save(trial / "state.json", state)
                        render()
                        if elapsed >= row["wall_limit_seconds"]:
                            stop_group(process)
                            state["status"] = "wall_limit"
                            break
                        time.sleep(10)
                state["last_exit_code"] = process.returncode
                if state["status"] == "wall_limit":
                    break
                if process.returncode != 0:
                    state["status"] = "agent_error"
                    break
                submission = submitted_result(trial)
                if submission["status"] in {"invalid", "conflict"}:
                    state.update(status="submission_needs_review", submission=submission)
                    break  # Never buy another native experiment to fix report metadata.
                if submission["status"] == "ready":
                    value = submission["value"]
                    state["submission_refs"] = submission["paths"]
                    if value.get("completed") is True or value.get("terminal_boundary") is True:
                        state["status"] = "agent_finished"
                        break
                    # Completion metadata is not scientific validation. A submitted
                    # result goes to audit; don't run more experiments merely because
                    # an upstream agent uses a different result schema.
                    if isinstance(value, dict) and value:
                        state["status"] = "submitted_needs_audit"
                        break
                prompt = ("继续执行当前已授权试点。请检查实际进程和已产出证据，复用已完成操作，"
                          "推进真实baseline和至少一个候选，更新RESEARCH.md与pilot_results.json。"
                          "completed=true只能在baseline及候选原生评测完成时填写；"
                          "如有已证实不可解决的边界写terminal_boundary=true并附日志。不要仅总结后停止。")
            if state["status"] == "running":
                state["status"] = "needs_followup"
    except Exception as exc:
        state.update(status="launcher_error", error=f"{type(exc).__name__}: {str(exc)[:220]}")
    finally:
        elapsed = time.monotonic() - started
        state.update(finished_at=now(), elapsed_seconds=elapsed,
                     conservative_device_occupancy_seconds=elapsed, native_metrics=native_metrics(trial))
        save(trial / "state.json", state)
        render()


def run_all(only=None):
    manifest = json.loads((STUDY / "manifest.json").read_text())
    with (STUDY / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(STUDY / "supervisor.json", {"pid": os.getpid(), "started_at": now(), "status": "running"})
        for row in manifest["trials"]:
            if only and row["name"] != only:
                continue
            current = json.loads((STUDY / row["name"] / "state.json").read_text())
            if current["status"] in {"prepared", "resume_pending"}:
                run_trial(row)
        for name, source in SOURCES.items():
            before = json.loads((STUDY / f"source_{name}_before.json").read_text())
            after = source_identity(source)
            save(STUDY / f"source_{name}_after.json", {"checked_at": now(), "unchanged": before == after,
                                                      "identity": after})
        save(STUDY / "supervisor.json", {"pid": os.getpid(), "finished_at": now(), "status": "finished"})
        render()


def vision_preflight():
    import base64
    import http.client
    import io
    from urllib.parse import urlsplit
    from PIL import Image
    sys.path.insert(0, str(ROOT / "autosim"))
    from autosim.llm_client import load_credential_file
    from autosim.research.deepseek_gateway import DeepSeekTurnGateway
    load_credential_file(project_root=ROOT)
    canvas = Image.new("RGB", (64, 64), "red")
    canvas.paste("blue", (0, 32, 64, 64))
    binary = io.BytesIO()
    canvas.save(binary, format="PNG")
    image_data = base64.b64encode(binary.getvalue()).decode()
    base = os.environ["DEEPSEEK_BASE_URL"].rstrip("/").removesuffix("/v1")
    if not base.endswith("/anthropic"):
        base += "/anthropic"
    with DeepSeekTurnGateway(upstream_base_url=base, upstream_key=os.environ["DEEPSEEK_API_KEY"],
            model="deepseek-flash", limit_usd=.4, allow_images=True,
            generation_policy={"max_output_tokens": 100, "thinking": "disabled"},
            checkpoint=lambda value: save(STUDY / "vision_preflight_usage.json", value)) as gateway:
        url = urlsplit(gateway.base_url)
        connection = http.client.HTTPConnection(url.hostname, url.port, timeout=120)
        payload = {"model": "deepseek-flash", "max_tokens": 100,
                   "messages": [{"role": "user", "content": [
                       {"type": "text", "text": "Identify the top-half and bottom-half colors. Reply briefly in English."},
                       {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                       "data": image_data}}]}]}
        connection.request("POST", "/v1/messages", body=json.dumps(payload), headers={
            "Authorization": "Bearer " + gateway.local_token, "Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        text = " ".join(c.get("text", "") for c in body.get("content", []) if c.get("type") == "text")
        result = {"at": now(), "http_status": response.status, "response": text,
                  "verified": response.status == 200 and "red" in text.lower() and "blue" in text.lower(),
                  "expected": "top red, bottom blue", "usage": gateway.snapshot(),
                  "official_vision_source": "https://api-docs.deepseek.com/guides/vision/"}
        save(STUDY / "vision_preflight.json", result)
        print(json.dumps({k: result[k] for k in ("http_status", "response", "verified")}))
        if not result["verified"]:
            raise RuntimeError("real provider image preflight did not verify colors")


def configure_vision():
    manifest = json.loads((STUDY / "manifest.json").read_text())
    if not json.loads((STUDY / "vision_preflight.json").read_text()).get("verified"):
        raise RuntimeError("vision provider preflight must pass first")
    with (STUDY / "supervisor.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.get("vision_configured"):
            raise RuntimeError("vision configuration already installed; do not reset runs")
        first = manifest["trials"][0]
        if json.loads((STUDY / first["name"] / "state.json").read_text())["status"] == "running":
            raise RuntimeError("wait for the text diagnostic to finish")
        first["allow_images"] = False
        first["scope"] = "completed text-feedback diagnostic; excluded from image-feedback comparison"
        for row in manifest["trials"][1:]:
            trial = STUDY / row["name"]
            if json.loads((trial / "state.json").read_text())["status"] != "waiting_vision_support":
                raise RuntimeError("pending trial changed unexpectedly")
            row["allow_images"] = True
            save(trial / "inputs.json", row)
            (trial / "INTEGRATION.md").write_text(integration(row))
            save(trial / "state.json", {"status": "prepared", "at": now(), "vision_preflight": "verified"})
        row = dict(first)
        row.update(name="enpire_robosyn_vision", allow_images=True, scope="image-feedback native-port pilot")
        trial = STUDY / row["name"]
        if trial.exists():
            raise RuntimeError("vision trial directory already exists")
        row["worktree"] = str(trial / "benchmark")
        copy_code(SOURCES["robosyn"], trial / "benchmark")
        for relative in ("assets", "checkpoints", "lerobot_dataset"):
            (trial / "benchmark" / relative).symlink_to(SOURCES["robosyn"] / relative, target_is_directory=True)
        for directory in ("home", "tmp", "logs", "outputs", "runtime"):
            (trial / directory).mkdir(parents=True, exist_ok=True)
        generic_runtime = (STUDY / first["name"] / "runtime/sitecustomize.py").read_text().split("# Thin adapter:")[0]
        (trial / "runtime/sitecustomize.py").write_text(generic_runtime)
        save(trial / "inputs.json", row)
        (trial / "INTEGRATION.md").write_text(integration(row))
        save(trial / "state.json", {"status": "prepared", "at": now(), "vision_preflight": "verified"})
        manifest["trials"].append(row)
        manifest.update(vision_configured=True, vision_source="https://api-docs.deepseek.com/guides/vision/",
                        maximum_reserved_gpu_seconds_per_repository={"robosyn": 13500, "robotwin": 9000},
                        model_maximum_usd=25.4)
        if manifest["prior_robosyn_gpu_seconds"] + 13500 > 86400:
            raise RuntimeError("vision extension exceeds inherited GPU envelope")
        save(STUDY / "manifest.json", manifest)
        save(STUDY / "vision_amendment.json", {"at": now(), "reason": "official Flash vision verified",
             "old_text_trial_preserved": first["name"], "new_independent_trial": row["name"],
             "prior_usage_reset": False, "preflight": "vision_preflight.json"})
        render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run", "render", "vision-preflight", "configure-vision"])
    parser.add_argument("--only")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run_all(args.only)
    elif args.command == "vision-preflight":
        vision_preflight()
    elif args.command == "configure-vision":
        configure_vision()
    else:
        render()
