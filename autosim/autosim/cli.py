#!/usr/bin/env python3
"""
AutoSim CLI — 自动化机器人策略优化流水线

对标 AutoSOTA 的交互模型：
  autosim init                              → 创建工作目录
  autosim --repo /path/to/clone --task T    → 一键全流程
  autosim sessions                          → 历史记录
  autosim inspect <ref>                     → 查看详情
"""

import sys, os, argparse, json, yaml
from pathlib import Path
from datetime import datetime

AUTOSIM_HOME = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path.cwd()
DEFAULT_ROBOTWIN_REPO = os.environ.get("ROBOTWIN_HOME", str(PROJECT_ROOT / "RoboTwin"))


def _find_task_config(task_name: str) -> Path:
    """Find the onboard config for a task."""
    paths = [
        WORKSPACE_ROOT / "output" / f"onboard_{task_name}.yaml",
        WORKSPACE_ROOT / ".autosim" / "tasks" / task_name / "config.yaml",
    ]
    for p in paths:
        if p.exists():
            return p
    return None


# ── autosim init ────────────────────────────────────────────────

def cmd_init(args):
    """Initialize workspace with config scaffold (like AutoSOTA's autosota init)."""
    ws = WORKSPACE_ROOT

    # Create directory structure
    (ws / ".autosim" / "tasks").mkdir(parents=True, exist_ok=True)
    (ws / ".autosim" / "runs").mkdir(parents=True, exist_ok=True)
    (ws / "task").mkdir(exist_ok=True)
    (ws / "logs").mkdir(exist_ok=True)
    (ws / "optimized_code").mkdir(exist_ok=True)

    # Create config.yaml
    config_path = ws / "config.yaml"
    if config_path.exists() and not args.force:
        print(f"  Config already exists: {config_path}")
        print(f"  Use --force to overwrite")
    else:
        config = {
            "llm_model": "deepseek-v4-flash",
            "llm_api_key": "",
            "llm_base_url": "http://10.1.21.21:3000/v1",
            "eval_seeds": 15,
            "max_iterations": 24,
            "target_improvement_pct": 30.0,
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"  Created: config.yaml")

    # Create task/target.md template
    target_path = ws / "task" / "target.md"
    if not target_path.exists() or args.force:
        target_path.write_text("""**任务**: beat_block_hammer
**仓库**: /path/to/RoboTwin

## 主要指标

| 指标 | 说明 | 优化方向 | 当前基线值 |
|------|------|----------|------------|
| `success_rate` | 任务成功率 | 越高越好 ↑ | ?% |

## 目标

在主指标 `success_rate` 上相比基线提升 30% 以上。
""")
        print(f"  Created: task/target.md")

    print(f"""
  工作目录: {ws}/
  ├── config.yaml          ← 填入 API key + 模型名
  ├── task/
  │   └── target.md        ← 填写优化目标
  ├── .autosim/            ← 运行时生成
  ├── logs/                ← 优化日志
  └── optimized_code/      ← 最优结果导出

  下一步: 编辑 config.yaml + task/target.md
          然后: autosim --repo /path/to/clone --task beat_block_hammer
""")
    return 0


# ── autosim run --repo ... --adapter act ────────────────────────

def cmd_run(args):
    """Generic closed-loop optimization with any adapter."""
    from autosim.optimize.closed_loop import ClosedLoopOptimizer

    # Map adapter names to classes
    adapter_map = {
        "act": "autosim.adapters.act_adapter.ACTAdapter",
        "robotwin": "autosim.adapters.robotwin_adapter.RoboTwinAdapter",
    }

    adapter_cls_path = args.adapter or "act"
    if adapter_cls_path in adapter_map:
        adapter_cls_path = adapter_map[adapter_cls_path]

    # Dynamic import
    mod_path, cls_name = adapter_cls_path.rsplit(".", 1)
    import importlib
    try:
        mod = importlib.import_module(mod_path)
        adapter_cls = getattr(mod, cls_name)
    except (ImportError, AttributeError) as exc:
        print(f"  Unsupported or unavailable adapter: {args.adapter}")
        print(f"  Tried: {adapter_cls_path}")
        print(f"  Error: {exc}")
        return 1

    if (args.adapter or "act") == "act" or adapter_cls_path.endswith(".ACTAdapter"):
        adapter = adapter_cls(
            repo_path=args.repo,
            task_name=args.task,
            data_name=args.data_name,
            task_config=args.task_config,
            expert_data_num=args.expert_data_num,
            epochs=args.epochs,
            seed=args.seed,
            batch_size=args.batch_size,
            timeout=args.timeout,
            gpu_id=args.gpu,
            dry_run=args.dry_run,
            eval_metric=args.metric,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            instruction_type=args.instruction_type,
            temporal_agg=not args.no_temporal_agg,
            device=args.device,
            python_executable=args.python_executable,
        )
    else:
        adapter = adapter_cls(repo_path=args.repo)
    allow_llm = not (args.no_llm or args.dry_run)
    opt = ClosedLoopOptimizer(
        adapter,
        output_dir=args.output or "output",
        allow_llm=allow_llm,
        idea_file=args.idea_file,
    )
    result = opt.run(max_iterations=args.max_iter)

    improv = result.get("improvement_pct", 0)
    print(f"\n  Improvement: {improv:+.1f}%")
    return 0


# ── autosim system --adapter act ────────────────────────────────

def cmd_system(args):
    """System-level simulation optimization over adapter candidates."""
    if args.adapter != "act":
        print(f"  Unsupported system adapter: {args.adapter}")
        return 1

    from autosim.adapters.act_adapter import ACTAdapter
    from autosim.optimize.simulation_optimizer import SimulationOptimizer
    from autosim.benchmark import (
        BenchmarkCandidateGenerator,
        LLMBenchmarkCandidateGenerator,
        load_benchmark_spec,
    )

    spec = load_benchmark_spec(
        path=args.benchmark_spec,
        task=args.task,
        task_config=args.task_config,
        baseline_ckpt=args.baseline_ckpt,
        eval_episodes=args.eval_episodes,
        max_rounds=args.max_rounds,
        candidate_limit=args.candidate_limit or 4,
        target_score=args.target_score,
    )

    if args.benchmark_mode and not args.dry_run and not args.train_output_root:
        print("  ERROR: --benchmark-mode without --dry-run requires --train-output-root")
        print("  This prevents generated checkpoints from being written into the source tree by accident.")
        return 1
    if args.checkpoint_candidates:
        spec.checkpoint_candidates = [
            item.strip() for item in args.checkpoint_candidates.split(",") if item.strip()
        ]

    expert_data_num = args.expert_data_num if args.expert_data_num is not None else spec.expert_data_num
    data_name = args.data_name or spec.data_name

    adapter = ACTAdapter(
        repo_path=args.repo,
        task_name=spec.task,
        data_name=data_name,
        task_config=spec.task_config,
        expert_data_num=expert_data_num,
        epochs=args.epochs,
        batch_size=args.batch_size,
        timeout=args.timeout,
        gpu_id=args.gpu,
        output_root=args.train_output_root,
        dry_run=args.dry_run,
        eval_metric=spec.metric,
        eval_episodes=spec.eval_episodes,
        eval_seed=args.eval_seed if args.eval_seed is not None else spec.eval_seed,
        instruction_type=spec.instruction_type,
        temporal_agg=spec.temporal_agg and not args.no_temporal_agg,
        device=args.device,
        python_executable=args.python_executable,
    )

    baseline = adapter.candidate_from_checkpoint(
        spec.baseline_ckpt,
        description="baseline checkpoint for simulator comparison",
    )
    optimizer = SimulationOptimizer(
        adapter,
        output_dir=args.output,
        skip_known_bad=args.skip_known_bad,
    )

    spec_path = None
    if args.save_benchmark_spec:
        spec_path = spec.save(args.save_benchmark_spec)
        print(f"  Benchmark spec saved: {spec_path}")

    if args.benchmark_mode:
        param_space = adapter.get_param_space()
        llm_candidates = []
        if args.llm_candidates:
            print("  Generating LLM ALGO/CODE benchmark candidates...")
            try:
                llm_candidates = LLMBenchmarkCandidateGenerator(
                    spec=spec,
                    adapter=adapter,
                    param_space=param_space,
                    limit=args.llm_candidate_limit,
                    model=args.llm_model,
                ).generate()
                print(f"  LLM candidates: {len(llm_candidates)}")
            except Exception as exc:
                print(f"  LLM candidate generation failed: {exc}")
                llm_candidates = []
        generator = BenchmarkCandidateGenerator(
            spec,
            param_space,
            llm_candidates=llm_candidates,
        )
        per_round_limit = args.candidate_limit or spec.candidate_limit

        def candidate_provider(_round_index, seen):
            return generator.next_batch(seen, per_round_limit)
    elif args.scan_checkpoints:
        def candidate_provider(_round_index, seen):
            return adapter.discover_checkpoint_candidates(
                limit=args.candidate_limit,
                include_prefix=args.candidate_prefix,
                exclude_names=seen,
            )
    else:
        candidate_names = [c.strip() for c in args.candidates.split(",") if c.strip()]
        candidates = [
            adapter.candidate_from_checkpoint(name, description="system candidate checkpoint")
            for name in candidate_names
        ]

        def candidate_provider(round_index, _seen):
            return candidates if round_index == 1 else []

    result = optimizer.run_until(
        baseline=baseline,
        candidate_provider=candidate_provider,
        target_score=spec.target_score,
        max_rounds=(spec.max_rounds if args.benchmark_mode
                    else (args.max_rounds or spec.max_rounds)),
        patience=args.patience,
    )

    best = result.get("best_overall") or result.get("best")
    if best:
        candidate = best["candidate"]["name"]
        score = best["score"]
        baseline_score = result["baseline"]["score"]
        print(f"\n  System best: {candidate} {score:.4f} vs baseline {baseline_score:.4f}")
    return 0


# ── autosim [task_name] --repo ... ──────────────────────────────

def cmd_main(args):
    """Main entry point: onboard + optimize in one command."""
    from autosim.onboard.discover import discover_task, run_baseline
    from autosim.ideas.param_ideas import generate_ideas
    from autosim.optimize.loop import OptimizeLoop, set_output_dir

    task_name = args.task
    repo_path = args.repo

    print("=" * 60)
    print(f"  AutoSim — {task_name}")
    print(f"  Repo: {repo_path}")
    print("=" * 60)

    # Phase 1: Onboard
    print(f"\n── PHASE 1: Onboard ──")
    if args.skip_onboard:
        config_path = _find_task_config(task_name)
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            print(f"  Using existing config: {config_path}")
            baseline_rate = config.get('baseline_rate', 0.0)
        else:
            print(f"  ERROR: No existing config for {task_name}. Run onboard first.")
            return 1
    else:
        task_info = discover_task(repo_path, task_name)
        baseline = run_baseline(repo_path, task_name, task_info, args.baseline_seeds)
        baseline_rate = baseline['rate']
        config = {
            "repo_path": repo_path,
            "task_name": task_name,
            "task_file": task_info['task_file'],
            "primary_metric": "success_rate",
            "metric_direction": "higher",
            "embodiment": str(task_info['embodiment']),
            "baseline_rate": baseline_rate,
            "params": task_info['params'],
            "max_iterations": args.max_iter,
            "target_improvement_pct": args.target_pct,
        }
        # Save onboard config
        out_dir = WORKSPACE_ROOT / ".autosim" / "tasks" / task_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"  Baseline: {baseline_rate*100:.1f}%")

    # Phase 2: Idea Generation
    print(f"\n── PHASE 2: Ideas ──")
    ideas = generate_ideas(config['params'])
    print(f"  Generated {len(ideas)} candidates")

    # Phase 3: Optimize
    print(f"\n── PHASE 3: Optimize ──")
    run_dir = WORKSPACE_ROOT / ".autosim" / "runs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    set_output_dir(str(run_dir))

    loop = OptimizeLoop(config, ideas)
    result = loop.run(max_iterations=args.max_iter, seeds_per_eval=args.eval_seeds)

    best_rate = result['best_rate']
    improvement = ((best_rate - baseline_rate) / baseline_rate * 100) if baseline_rate else 0.0

    print(f"\n{'=' * 60}")
    print(f"  Optimization Complete")
    print(f"{'=' * 60}")
    print(f"  Task:       {task_name}")
    print(f"  Baseline:   {baseline_rate*100:.1f}%")
    print(f"  Best:       {best_rate*100:.1f}% (+{improvement:.1f}%)")
    print(f"  Run dir:    {run_dir}")
    print(f"  Scores:     {run_dir}/scores.jsonl")

    # Save result
    with open(run_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    if not args.skip_export:
        print(f"\n  Next: autosim demo --task {task_name}")
    return 0


# ── autosim sessions ────────────────────────────────────────────

def cmd_sessions(args):
    """List past optimization runs (like AutoSOTA's autosota sessions)."""
    runs_dir = WORKSPACE_ROOT / ".autosim" / "runs"
    if not runs_dir.exists():
        print("  No runs yet.")
        return 0

    runs = sorted(runs_dir.iterdir(), reverse=True)
    print(f"  {'Task':<20s} {'Run':<25s} {'Status':<12s} {'Iters':<8s} {'Best':>10s}")
    print(f"  {'─'*75}")

    for r in runs:
        if not r.is_dir():
            continue
        result_file = r / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            task = data.get('task_name', '?')
            baseline = data.get('baseline_rate', 0) * 100
            best = data.get('best_rate', 0) * 100
            iters = data.get('num_evaluations', '?')
            print(f"  {task:<20s} {r.name:<25s} {'✓ complete':<12s} {str(iters):<8s} {best:>8.1f}%")
        else:
            print(f"  {'?':<20s} {r.name:<25s} {'~ aborted':<12s}")

    return 0


# ── autosim inspect ─────────────────────────────────────────────

def cmd_inspect(args):
    """View run details (like AutoSOTA's autosita inspect)."""
    runs_dir = WORKSPACE_ROOT / ".autosim" / "runs"
    ref = args.ref

    # Find run by name, task, or 'latest'
    run_path = None
    if ref == "latest":
        runs = sorted(runs_dir.iterdir(), reverse=True)
        run_path = runs[0] if runs else None
    else:
        for r in runs_dir.iterdir():
            if r.name == ref or r.name.endswith(ref):
                run_path = r
                break
        if not run_path:
            # Try matching by task name via result.json
            for r in sorted(runs_dir.iterdir(), reverse=True):
                rf = r / "result.json"
                if rf.exists():
                    with open(rf) as f:
                        if json.load(f).get('task_name') == ref:
                            run_path = r
                            break

    if not run_path or not run_path.exists():
        print(f"  Run not found: {ref}")
        return 1

    result_file = run_path / "result.json"
    scores_file = run_path / "scores.jsonl"

    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        print(f"  Task:      {data.get('task_name', '?')}")
        print(f"  Baseline:  {data.get('baseline_rate', 0)*100:.1f}%")
        print(f"  Best:      {data.get('best_rate', 0)*100:.1f}%")
        print(f"  Params:    {json.dumps(data.get('best_params', {}), indent=2)}")
        print(f"  Evals:     {data.get('num_evaluations', '?')}")

    if args.scores and scores_file.exists():
        print(f"\n  Scores:")
        with open(scores_file) as f:
            for line in f:
                print(f"    {line.strip()}")

    return 0


# ── autosim demo ─────────────────────────────────────────────────

def cmd_demo(args):
    """Generate comparison video (like AutoSOTA's export)."""
    task_name = args.task
    config_path = _find_task_config(task_name)
    if not config_path:
        print(f"  No config for {task_name}. Run autosim --repo ... --task {task_name} first.")
        return 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Find best result
    runs_dir = WORKSPACE_ROOT / ".autosim" / "runs"
    best_params = None
    for r in sorted(runs_dir.iterdir(), reverse=True):
        rf = r / "result.json"
        if rf.exists():
            with open(rf) as f:
                data = json.load(f)
            if data.get('task_name') == task_name:
                best_params = data.get('best_params')
                break

    if not best_params:
        best_params = {k: v['default'] for k, v in config['params'].items()}

    baseline_params = {k: v['default'] for k, v in config['params'].items()}

    print(f"  Baseline: {config['baseline_rate']*100:.0f}%")
    print(f"  Best params: {json.dumps(best_params)}")

    from autosim.export.demo import run_comparison_demo
    video = run_comparison_demo(
        config['repo_path'], task_name,
        baseline_params, best_params,
        cases=args.cases,
        output=args.output or "output/comparison.mp4",
    )
    print(f"\n  Video: {video}")
    return 0


# ── autosim scene ─────────────────────────────────────────────────

def cmd_scene(args):
    """Scene management: generate, list, build, optimize"""
    from autosim.scene import (
        SceneGenerator, SceneRegistry, SceneBuilder,
        get_template, list_templates,
    )
    from autosim.mcp_client import MCPClient

    if args.scene_cmd == "list":
        if args.template:
            print(f"\n  Available templates:")
            for t in list_templates():
                template = get_template(t)
                print(f"    {t:<25s} {template.description[:60]}")
        registry = SceneRegistry(args.registry or ".autosim/scenes")
        scenes = registry.list(task_category=args.task_type)
        print(f"\n  Registry scenes ({len(scenes)}):")
        for s in scenes:
            print(f"    {s['name']:<30s} {s['task_category']:<12s} {s.get('description','')[:50]}")
        return 0

    if args.scene_cmd == "generate":
        gen = SceneGenerator()
        registry = SceneRegistry(args.registry or ".autosim/scenes")
        result = gen.from_description(args.description, template_name=args.template)
        if result.success:
            registry.save(result.config, tags=[result.template_used, result.config.task.category])
            print(f"\n  ✓ Generated: {result.config.name}")
            print(f"    Template: {result.template_used}")
            print(f"    Objects:  {[o.name for o in result.config.objects]}")
            print(f"    Task:     {result.config.task.description}")
        else:
            print(f"\n  ✗ Failed: {result.error}")
        return 0

    if args.scene_cmd == "build":
        registry = SceneRegistry(args.registry or ".autosim/scenes")
        config = registry.load(args.name)
        client = MCPClient(mock=args.mock)
        if not args.mock:
            client.connect()
        builder = SceneBuilder(client, verbose=True)
        result = builder.build(config)
        if not args.mock:
            builder.teardown()
        print(f"\n  ✓ Scene built: {len(result.get('objects', []))} objects")
        return 0

    if args.scene_cmd == "optimize":
        from autosim.adapters.isaac_adapter import IsaacSimAdapter
        from autosim.optimize.closed_loop import ClosedLoopOptimizer
        from autosim.scene import ContinuousLearner

        registry = SceneRegistry(args.registry or ".autosim/scenes")
        config = registry.load(args.name)

        task_type = args.task_type or config.task.category
        adapter = IsaacSimAdapter(
            task_type=task_type,
            robot_type=args.robot or "franka",
            mock=args.mock,
            num_seeds=args.seeds or 10,
        )
        opt = ClosedLoopOptimizer(adapter, output_dir=args.output or f"output/scene_{args.name}")
        result = opt.run(max_iterations=args.max_iter or 8)

        learner = ContinuousLearner()
        learner.ingest_run(result, task_type=task_type, robot=args.robot or "franka")
        print(f"\n  Optimization: baseline {result.get('baseline_score',0):.3f} → {result.get('best_score',0):.3f}")
        return 0

    print("  Unknown scene subcommand. Use: autosim scene {list,generate,build,optimize}")
    return 0


# ── Main ─────────────────────────────────────────────────────────

def main():
    # Handle no-args case
    if len(sys.argv) == 1:
        sys.argv.append("--help")

    # Route based on first positional argument
    first_arg = sys.argv[1] if len(sys.argv) > 1 else ""

    if first_arg == "research":
        # Repository-in, optimized-repository-out simulator AutoResearch.  Keep
        # its parser and durable state machine isolated from the legacy task CLI.
        from autosim.research.repository_autoresearch import main as research_main

        return research_main(sys.argv[2:])

    if first_arg == "scout":
        # Read an unfamiliar benchmark and report what it can do. Separate from
        # `research` because it answers a different question -- `scout` asks whether this
        # benchmark can support a research loop at all, `research` runs one.
        from autosim.research.scout import main as scout_main

        return scout_main(sys.argv[2:])

    if first_arg == "run":
        parser = argparse.ArgumentParser(description="AutoSim Run — Generic Optimization")
        parser.add_argument("run_cmd", nargs="?")
        parser.add_argument("--repo", default=DEFAULT_ROBOTWIN_REPO)
        parser.add_argument("--adapter", default="act")
        parser.add_argument("--task", default="beat_block_hammer")
        parser.add_argument("--data-name", default=None,
                            help="Registered ACT dataset, e.g. sim-beat_block_hammer-demo_randomized-15")
        parser.add_argument("--task-config", default="demo_randomized")
        parser.add_argument("--expert-data-num", type=int, default=15)
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--batch-size", type=int, default=4)
        parser.add_argument("--timeout", type=int, default=1800)
        parser.add_argument("--gpu", default=None)
        parser.add_argument("--metric", choices=["success_rate", "val_loss"], default="success_rate",
                            help="Optimize deployed task success_rate or training val_loss")
        parser.add_argument("--eval-episodes", type=int, default=10,
                            help="Number of successful expert seeds to evaluate per candidate")
        parser.add_argument("--eval-seed", type=int, default=0)
        parser.add_argument("--instruction-type", default="unseen")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--python-executable", default=None,
                            help="Python executable for ACT training/evaluation subprocesses")
        parser.add_argument("--no-temporal-agg", action="store_true")
        parser.add_argument("--dry-run", action="store_true",
                            help="Use deterministic proxy scores without launching ACT training")
        parser.add_argument("--no-llm", action="store_true",
                            help="Disable LLM calls and use local/offline optimization only")
        parser.add_argument("--idea-file", default=None,
                            help="JSON file with offline ALGO/CODE/PARAM ideas")
        parser.add_argument("--max-iter", type=int, default=8)
        parser.add_argument("--output", default="output")
        args = parser.parse_args()
        return cmd_run(args)

    if first_arg == "system":
        parser = argparse.ArgumentParser(description="AutoSim System — Simulation Candidate Optimizer")
        parser.add_argument("system_cmd", nargs="?")
        parser.add_argument("--repo", default=DEFAULT_ROBOTWIN_REPO)
        parser.add_argument("--adapter", default="act")
        parser.add_argument("--task", default="beat_block_hammer")
        parser.add_argument("--task-config", default="demo_clean")
        parser.add_argument("--benchmark-mode", action="store_true",
                            help="Generate train_recipe candidates from a BenchmarkSpec")
        parser.add_argument("--benchmark-spec", default=None,
                            help="YAML benchmark spec. Defaults to a RoboTwin ACT spec from CLI args")
        parser.add_argument("--save-benchmark-spec", default=None,
                            help="Write the resolved benchmark spec to this YAML path")
        parser.add_argument("--checkpoint-candidates", default=None,
                            help="Comma-separated checkpoint candidates to evaluate in benchmark mode")
        parser.add_argument("--llm-candidates", action="store_true",
                            help="Generate ALGO/CODE patch_train_recipe candidates with the configured LLM API")
        parser.add_argument("--llm-candidate-limit", type=int, default=4,
                            help="Maximum LLM-generated ALGO/CODE candidates to prepend to the benchmark queue")
        parser.add_argument("--llm-model", default=None,
                            help="Override AUTOSIM_LLM_MODEL for LLM candidate generation")
        parser.add_argument("--baseline-ckpt", default="demo_clean-50",
                            help="Baseline checkpoint directory name under policy/ACT/act_ckpt/act-<task>")
        parser.add_argument("--data-name", default=None,
                            help="Registered ACT dataset name; default is sim-<task>-<task_config>-<expert_data_num>")
        parser.add_argument("--expert-data-num", type=int, default=None,
                            help="Expert demo count used to build the default ACT dataset name")
        parser.add_argument("--epochs", type=int, default=100,
                            help="Training epochs for generated train_recipe candidates")
        parser.add_argument("--batch-size", type=int, default=4,
                            help="ACT train/val batch size for generated train_recipe candidates")
        parser.add_argument("--train-output-root", default=None,
                            help="Checkpoint root for generated train_recipe candidates")
        parser.add_argument("--candidates", default="autosim_algo1_scheduler",
                            help="Comma-separated checkpoint directory names to compare")
        parser.add_argument("--scan-checkpoints", action="store_true",
                            help="Continuously discover checkpoint candidates from the task ckpt directory")
        parser.add_argument("--candidate-limit", type=int, default=None,
                            help="Maximum newly discovered checkpoint candidates per round")
        parser.add_argument("--candidate-prefix", default=None,
                            help="Only discover checkpoint directories with this prefix")
        parser.add_argument("--max-rounds", type=int, default=None,
                            help="Maximum system optimization rounds")
        parser.add_argument("--target-score", type=float, default=None,
                            help="Stop when best score reaches this target, e.g. known SOTA")
        parser.add_argument("--patience", type=int, default=3,
                            help="Stop after this many rounds without improvement/new useful candidates")
        parser.add_argument("--skip-known-bad", action="store_true",
                            help="Skip method families that the experience graph marks as repeatedly negative")
        parser.add_argument("--timeout", type=int, default=1800)
        parser.add_argument("--gpu", default=None)
        parser.add_argument("--eval-episodes", type=int, default=None)
        parser.add_argument("--eval-seed", type=int, default=None)
        parser.add_argument("--instruction-type", default="unseen")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--python-executable", default=None,
                            help="Python executable for ACT training/evaluation subprocesses")
        parser.add_argument("--no-temporal-agg", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--output", default="output/system")
        args = parser.parse_args()
        return cmd_system(args)

    if first_arg in ("init", "sessions", "inspect", "demo", "scene"):
        # Subcommand mode
        parser = argparse.ArgumentParser(description="AutoSim CLI")
        sub = parser.add_subparsers(dest="command")
        p = sub.add_parser("init")
        p.add_argument("--force", action="store_true")
        sub.add_parser("sessions")
        p = sub.add_parser("inspect")
        p.add_argument("ref"); p.add_argument("--scores", action="store_true")
        p = sub.add_parser("demo")
        p.add_argument("--task", required=True); p.add_argument("--cases", type=int, default=8); p.add_argument("--output")
        p = sub.add_parser("scene")
        p.add_argument("scene_cmd", nargs="?", choices=["list", "generate", "build", "optimize"],
                       default="list", help="Scene subcommand")
        p.add_argument("--template", action="store_true", help="Include built-in templates in list")
        p.add_argument("--description", "-d", default="Franka grasping a cube",
                       help="Natural language scene description (for generate)")
        p.add_argument("--name", "-n", default="my_scene", help="Scene name")
        p.add_argument("--task-type", default=None, help="Filter by task type")
        p.add_argument("--robot", default="franka")
        p.add_argument("--mock", action="store_true", default=True)
        p.add_argument("--no-mock", action="store_false", dest="mock")
        p.add_argument("--registry", default=".autosim/scenes", help="Scene registry directory")
        p.add_argument("--seeds", type=int, default=10, help="Eval seeds")
        p.add_argument("--max-iter", type=int, default=8)
        p.add_argument("--output", default=None, help="Output directory")
        args = parser.parse_args()

        if args.command == "init": return cmd_init(args)
        elif args.command == "sessions": return cmd_sessions(args)
        elif args.command == "inspect": return cmd_inspect(args)
        elif args.command == "demo": return cmd_demo(args)
        elif args.command == "scene": return cmd_scene(args)
    else:
        # Main flow: autosim <task> --repo <path>
        parser = argparse.ArgumentParser(description="AutoSim — 自动化机器人策略优化")
        parser.add_argument("task", help="Task name")
        parser.add_argument("--repo", required=True)
        parser.add_argument("--baseline-seeds", type=int, default=30)
        parser.add_argument("--max-iter", type=int, default=24)
        parser.add_argument("--target-pct", type=float, default=30.0)
        parser.add_argument("--eval-seeds", type=int, default=15)
        parser.add_argument("--skip-onboard", action="store_true")
        parser.add_argument("--skip-export", action="store_true")
        args = parser.parse_args()
        return cmd_main(args)


if __name__ == "__main__":
    sys.exit(main())
