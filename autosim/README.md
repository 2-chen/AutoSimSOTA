# AutoSim: 具身智能自动化研究系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> 本文包含早期实现及历史使用说明。当前独立项目入口与已知限制以[根目录README](../README.md)为准；整体许可证尚待维护者确认。

结合 **AutoSOTA** 的闭环优化范式、**isaac-sim-mcp** 的仿真控制和 **RoboTwin** 的任务定义，实现具身智能领域的自动化研究——通过仿真中的迭代优化，自动发现更好的机器人控制策略。

## 目录

- [核心思想](#核心思想)
- [架构概览](#架构概览)
- [快速开始](#快速开始)
- [CLI 使用指南](#cli-使用指南)
- [两种优化模式](#两种优化模式)
- [项目结构](#项目结构)
- [核心模块详解](#核心模块详解)
- [扩展指南](#扩展指南)
- [配置参考](#配置参考)
- [输出格式](#输出格式)
- [Demo 脚本](#demo-脚本)
- [依赖](#依赖)
- [引用](#引用)

## 核心思想

AutoSim 将机器人策略优化建模为一个**闭环迭代搜索问题**：

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  BASELINE ──→ 策略建议 ──→ 仿真执行 ──→ 指标提取 ──→ 反馈   │
│      ↑                                                        │
│      └──────────── 如果提升不足，继续迭代 ←────────────────┘  │
│                                                              │
│  收敛条件: 达到目标提升百分比 OR 达到最大迭代次数               │
└──────────────────────────────────────────────────────────────┘
```

每次迭代：
1. **策略**根据历史记录建议一组候选参数
2. 在 Isaac Sim（或 Mock 模式）中**执行任务**
3. **提取指标**（末端距离、成功率等）
4. **反馈**给策略，更新搜索分布
5. 如果超越目标提升，**提前收敛**

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                         AutoSim                                  │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │   Optimizer   │   │  Task Suite  │   │    MCP Client     │   │
│  │ (AutoSOTA 灵感)│──▶│(RoboTwin 灵感)│──▶│ (isaac-sim-mcp)   │   │
│  └──────────────┘   └──────────────┘   └───────────────────┘   │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  参数搜索策略         任务定义范式          仿真通信层            │
│  · random            · Franka 到达         · Socket 协议          │
│  · grid              · RoboTwin 任务       · Python 代码注入      │
│  · hill_climb        · 自定义任务          · 场景管理              │
│  · cma_es                                                        │
│  · LLM-driven                                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 三个项目的融合方式

| 项目 | 核心贡献 | AutoSim 中的复用 |
|------|---------|-----------------|
| **AutoSOTA** | 闭环优化循环（baseline → idea → eval → iterate） | `optimizer.py` + `llm_optimizer.py` 完整复现该模式 |
| **isaac-sim-mcp** | 自然语言 → Isaac Sim 控制（MCP 协议） | `mcp_client.py` 封装 Socket 通信 |
| **RoboTwin** | 任务定义范式（setup → play → check） | `tasks/` 任务系统 + `onboard/` 自动发现 |

## 快速开始

### 安装

```bash
cd /path/to/AutoSimSOTA/autosim
pip install -r requirements.txt
```

### 初始化工作目录

```bash
# 创建 .autosim/ 目录结构和配置文件
python -m autosim.cli init

# 强制覆盖已有配置
python -m autosim.cli init --force
```

初始化后编辑 `config.yaml`（填入 API key）和 `task/target.md`（填写优化目标）。

### Mock 模式（无需 Isaac Sim）

Mock 模式使用简化的 Franka 正运动学模型，无需实际运行 Isaac Sim，适合快速验证优化策略：

```bash
# 默认 CMA-ES 策略，50 轮迭代
python demo/run_demo.py

# 随机搜索策略，10 轮
python demo/run_demo.py --strategy random --iterations 10

# 爬山策略，指定目标位置
python demo/run_demo.py --strategy hill_climb --target 0.5,0.0,0.4

# 网格搜索
python demo/run_demo.py --strategy grid --iterations 25
```

### 连接真实 Isaac Sim

```bash
# 1）先启动 Isaac Sim 并加载 MCP 扩展
# 2）运行优化
python demo/run_demo.py --no-mock --host localhost --port 8766
```

### 一键全流程（RoboTwin 任务）

```bash
# 自动完成: Onboard → Ideas → Optimize → Export
python -m autosim.cli beat_block_hammer --repo /path/to/RoboTwin

# 跳过 onboard（使用缓存）
python -m autosim.cli beat_block_hammer --repo /path/to/RoboTwin --skip-onboard

# 自定义参数
python -m autosim.cli beat_block_hammer \
    --repo /path/to/RoboTwin \
    --baseline-seeds 50 \
    --max-iter 30 \
    --target-pct 30.0 \
    --eval-seeds 15
```

## CLI 使用指南

AutoSim 提供完整的命令行接口，对标 AutoSOTA 的交互模型：

```bash
# 初始化工作目录
autosim init [--force]

# 一键全流程优化
autosim <task_name> --repo <repo_path> [options]

# 通用优化（指定 adapter）
autosim run --repo <path> --adapter act [--max-iter 8]

# 系统级仿真候选优化：不训练模型，只统一评测 baseline/candidates
autosim system \
  --repo /path/to/AutoSimSOTA/RoboTwin \
  --adapter act \
  --task beat_block_hammer \
  --task-config demo_clean \
  --baseline-ckpt demo_clean-50 \
  --candidates autosim_algo1_scheduler,autosim_combo_best \
  --eval-episodes 10 \
  --output output/system_act

# 持续优化模式：按轮扫描候选池，直到达到目标分数或停止条件
autosim system \
  --repo /path/to/AutoSimSOTA/RoboTwin \
  --adapter act \
  --task beat_block_hammer \
  --task-config demo_clean \
  --baseline-ckpt demo_clean-50 \
  --scan-checkpoints \
  --candidate-prefix autosim_ \
  --candidate-limit 4 \
  --max-rounds 20 \
  --target-score 0.90 \
  --patience 5 \
  --skip-known-bad \
  --eval-episodes 10 \
  --output output/system_act_loop

# 查看历史运行记录
autosim sessions

# 查看某次运行详情
autosim inspect latest [--scores]
autosim inspect <run_id> [--scores]
autosim inspect <task_name> [--scores]

# 生成对比视频
autosim demo --task <task_name> [--cases 8] [--output output/comparison.mp4]
```

### 主流程参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | (必填) | 项目仓库路径 |
| `--baseline-seeds` | 30 | 基线评估的随机种子数 |
| `--max-iter` | 24 | 最大优化迭代次数 |
| `--target-pct` | 30.0 | 目标提升百分比 |
| `--eval-seeds` | 15 | 每次评估的种子数 |
| `--skip-onboard` | false | 跳过任务发现，使用缓存配置 |
| `--skip-export` | false | 跳过最终对比视频导出 |

### 系统级仿真优化入口

`autosim system` 是面向“AutoSim 作为自动优化系统”的入口。它不假设优化对象一定是模型权重，也不在系统层写死 ACT 训练逻辑；系统只管理：

1. `baseline`：明确的基线候选，例如 RoboTwin 官方 `demo_clean-50` checkpoint
2. `candidates`：可被仿真评测的候选对象，例如不同 checkpoint、策略配置、控制器、adapter 输出
3. `adapter`：把候选转换成具体仿真环境调用，返回统一 `EvalResult`
4. `SimulationOptimizer`：按同一指标评测、排序、记录 manifest

输出文件为：

```text
output/system_act/simulation_optimization_result.json
```

这个 JSON 保存 baseline、每个候选的 `success_rate`、原始指标、相对 baseline 的提升和 checkpoint 路径，避免把不同来源的 baseline 混用。

持续模式使用 `SimulationOptimizer.run_until()`：

- `--scan-checkpoints`：每轮重新扫描候选池，适合外部训练器/搜索器持续产出新 checkpoint
- `--target-score`：达到目标分数即停止，可用作当前 SOTA 阈值
- `--max-rounds`：最大优化轮数，避免无限运行
- `--patience`：连续若干轮无提升后停止
- `--candidate-limit`：每轮最多评测的新候选数
- `--skip-known-bad`：读取经验网络，跳过多次被证明无效的方法族

每次评测都会同步维护一张经验网络：

```text
output/system_act_loop/optimization_experience_graph.json
output/system_act_loop/optimization_experience_summary.md
```

网络中的节点是每一次优化尝试，边表示“同一方法族”等关系。每个节点保存：

- 优化方法族：例如 `learning_rate_tuning`、`kl_weight_tuning`、`scheduler`
- 优化假设：为什么要尝试这个候选
- 实际效果：相对 baseline 的提升或退化
- 经验结论：保留、谨慎扩展、避免重复

如果目录中已有旧的 `simulation_optimization_result.json`，AutoSim 会在新一轮启动时自动把历史评测导入经验网络，避免忘记之前已经验证过的失败方向。

因此真正的系统闭环是：

```text
明确 baseline -> 发现候选 -> 仿真评测 -> 更新 best -> 写 manifest
      ↑                                                    |
      └────────────── 未达到 target/SOTA 时继续下一轮 ─────┘
```

## 两种优化模式

### 模式 1：经典优化策略

适合参数空间明确、可微分的连续优化问题。

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| `random` | 均匀随机采样 | 基线对比、高维探索 |
| `grid` | 参数空间网格搜索 | 低维参数空间（<4 维） |
| `hill_climb` | 爬山法 + 随机重启动 | 平滑的局部最优 |
| `cma_es` | 协方差矩阵自适应进化策略 | 高维连续参数（**推荐**） |

```python
from autosim.optimizer import EmbodiedOptimizer, create_strategy

# CMA-ES 自动维护高斯分布，逐步收敛到最优区域
optimizer = EmbodiedOptimizer(task, opt_config)
run = optimizer.optimize(strategy="cma_es")
```

### 模式 2：LLM 驱动优化

适合需要修改代码逻辑（而不仅仅是参数）的场景。LLM 分析完整源码，生成 CODE/ALGO/PARAM 三级优化建议，并通过仿真反馈闭环迭代。

```
PHASE 1: Deep Code Analysis   →  完整源码分析（策略、动作原语、决策点、失败模式）
PHASE 2: Idea Library         →  生成 12-15 个精确 diff（含 exact old_code/new_code）
PHASE 3: Closed-Loop Feedback →  逐个应用 → 仿真评估 → LLM 反思 → 改进下一轮
PHASE 4: PARAM Fine-tuning    →  在最佳 CODE/ALGO 状态下微调参数
```

AutoSOTA 优先级: **ALGO > CODE > PARAM**

```bash
# LLM 优化模式
python -m autosim.llm_optimizer \
    --repo /path/to/RoboTwin \
    --task beat_block_hammer \
    --baseline-rate 0.45 \
    --max-iter 8 \
    --seeds 12
```

配置环境变量:
```bash
export AUTOSIM_LLM_BASE_URL=http://10.1.21.21:3000/v1
export AUTOSIM_LLM_API_KEY=sk-...
export AUTOSIM_LLM_MODEL=deepseek-v4-flash
```

### ACT 模型接入验证

`autosim run --adapter act` 会通过 `TaskAdapter` 接入 RoboTwin 的
`policy/ACT/imitate_episodes.py` 训练候选模型，再调用 RoboTwin 部署评测入口统计
任务 `success_rate`（越高越好）。如果没有 DeepSeek API key，或当前环境不能联网，
闭环优化器会自动退化为本地参数搜索；如果加上 `--dry-run`，则不会启动真实训练，
只使用确定性代理分数验证整个闭环。

```bash
# 验证 AutoSim 闭环，不依赖 GPU/数据集/API
python -m autosim.cli run \
    --adapter act \
    --repo /path/to/AutoSimSOTA/RoboTwin \
    --task beat_block_hammer \
    --metric success_rate \
    --expert-data-num 15 \
    --epochs 3 \
    --max-iter 3 \
    --dry-run \
    --output /tmp/autosim_act_dryrun

# 真实训练评估：需要先用 RoboTwin/policy/ACT/process_data.sh 准备 processed_data
python -m autosim.cli run \
    --adapter act \
    --repo /path/to/AutoSimSOTA/RoboTwin \
    --task beat_block_hammer \
    --task-config demo_randomized \
    --expert-data-num 15 \
    --metric success_rate \
    --eval-episodes 10 \
    --epochs 100 \
    --gpu 0 \
    --max-iter 8 \
    --output output/act_closed_loop

# 如只想快速筛训练超参，也可以改用验证集 loss
python -m autosim.cli run \
    --adapter act \
    --repo /path/to/AutoSimSOTA/RoboTwin \
    --task beat_block_hammer \
    --metric val_loss \
    --epochs 20 \
    --max-iter 8
```

## 项目结构

```
autosim/
├── autosim/                          # 核心 Python 包
│   ├── __init__.py                   # 版本声明 (v0.2.0)
│   ├── cli.py                        # CLI 入口 (init/sessions/inspect/demo/run)
│   ├── config.py                     # 配置管理 (TaskConfig, OptimizerConfig)
│   ├── optimizer.py                  # 经典优化引擎 (random/grid/hill_climb/cma_es)
│   ├── llm_optimizer.py             # LLM 驱动优化器 (DeepSeek API)
│   ├── mcp_client.py                # Isaac Sim MCP Socket 通信客户端
│   ├── mock_planner.py              # Mock 运动规划器 (线性插值)
│   ├── metrics.py                   # 指标追踪与优化曲线绘制
│   ├── record_score.py              # 原子化分数记录 (兼容 AutoSOTA)
│   │
│   ├── adapters/                    # 通用任务适配器框架
│   │   ├── __init__.py
│   │   ├── base.py                  # TaskAdapter 抽象基类
│   │   ├── act_adapter.py           # ACT 行为克隆适配器
│   │   ├── isaac_adapter.py         # Isaac Sim 仿真适配器
│   │   └── robotwin_adapter.py      # RoboTwin 任务适配器
│   │
│   ├── tasks/                       # 任务定义
│   │   ├── __init__.py
│   │   ├── task_base.py             # 任务基类 (setup → execute → extract_metrics)
│   │   └── franka_reach.py          # Franka 机械臂到达任务
│   │
│   ├── onboard/                     # 任务发现与基线
│   │   ├── __init__.py
│   │   └── discover.py              # 自动扫描仓库，提取参数，运行基线
│   │
│   ├── ideas/                       # 候选参数生成
│   │   ├── __init__.py
│   │   └── param_ideas.py           # 三级候选生成 (random + local + grid)
│   │
│   ├── optimize/                    # 闭环优化循环
│   │   ├── __init__.py
│   │   ├── loop.py                  # OptimizeLoop (AutoSOTA 风格)
│   │   └── closed_loop.py           # ClosedLoopOptimizer (通用 adapter 版本)
│   │
│   ├── export/                      # 结果导出
│   │   ├── __init__.py
│   │   └── demo.py                  # 对比视频生成
│   │
│   └── mock/                        # Mock Isaac Sim
│       ├── __init__.py
│       └── mock_isaac.py            # 简化的 Franka FK 模拟
│
├── demo/                            # Demo 脚本
│   ├── config.yaml                  # Demo 配置
│   ├── run_demo.py                  # 经典优化 Demo
│   ├── run_robot_demo.py            # RoboTwin 任务 Demo
│   ├── run_visual_demo.py           # 可视化 Demo
│   ├── run_comparison_demo.py       # 对比 Demo
│   ├── run_ik_demo.py               # 逆运动学 Demo
│   ├── run_act_eval.py              # ACT 评估
│   ├── optimize_act.py              # ACT 参数优化
│   ├── optimize_act_v2.py           # ACT 优化 v2
│   ├── optimize_act_closed_loop.py  # ACT 闭环优化
│   ├── run_strategy_opt.py          # 策略优化
│   ├── run_robotwin_task.py         # RoboTwin 任务运行
│   ├── run_robotwin_opt.py          # RoboTwin 优化
│   ├── calibrate_camera.py          # 相机标定
│   ├── save_aloha_video.py          # Aloha 视频保存
│   ├── save_act_video.py            # ACT 视频保存
│   └── save_comparison_video.py     # 对比视频保存
│
├── task/                            # 任务目标定义
│   └── target.md
├── output/                          # 优化结果输出
├── optimized_code/                  # 最优代码导出
├── logs/                            # 运行日志
├── config.yaml                      # 全局配置
├── requirements.txt
├── pyproject.toml
├── autosim.sh                       # Shell 包装脚本
└── README.md
```

## 核心模块详解

### TaskAdapter — 通用适配器框架

任何仿真/训练项目只需实现三个方法，AutoSim 就能自动优化：

```python
from autosim.adapters.base import TaskAdapter, ParamDef, EvalResult

class MyAdapter(TaskAdapter):
    def get_param_space(self) -> Dict[str, ParamDef]:
        """返回可优化的超参空间"""
        return {
            "learning_rate": ParamDef("learning_rate", 1e-4, (1e-5, 1e-2)),
            "batch_size": ParamDef("batch_size", 64, (16, 256), dtype="int"),
        }

    def evaluate(self, params: Dict) -> EvalResult:
        """用给定参数训练/评估 → 返回分数"""
        # 训练模型，运行仿真...
        return EvalResult(score=0.85, success=True, metrics={"loss": 0.12})

    def get_source_files(self) -> Dict[str, str]:
        """返回源码文件供 LLM 分析"""
        return {"train.py": open("train.py").read()}
```

内置适配器:
- `ACTAdapter` — ACT 行为克隆训练
- `RoboTwinAdapter` — RoboTwin 仿真任务
- `IsaacAdapter` — Isaac Sim 直接控制

### MCP Client — 仿真通信

```python
from autosim.mcp_client import MCPClient

# Mock 模式（离线测试）
client = MCPClient(mock=True)

# 真实 Isaac Sim
client = MCPClient(host="localhost", port=8766, mock=False)

# 高层 API
client.get_scene_info()
client.create_robot(robot_type="franka", position=[0, 0, 0])
client.execute_script("print('hello from Isaac Sim')")
```

### Onboard — 任务自动发现

```python
from autosim.onboard.discover import discover_task, run_baseline

# 自动扫描仓库，提取策略参数
task_info = discover_task("/path/to/RoboTwin", "beat_block_hammer")
# → {name, embodiment, params: {pre_grasp_dis: {default, range}, ...}, ...}

# 运行基线评估
baseline = run_baseline("/path/to/RoboTwin", "beat_block_hammer", task_info, num_seeds=50)
# → {rate: 0.45, success: 22, total: 50}
```

### Ideas — 候选参数生成

```python
from autosim.ideas.param_ideas import generate_ideas

# 三级候选: random_uniform + local_mutation + grid_search
ideas = generate_ideas(params, n_random=20, n_local=10, n_grid=5)
# 每个 idea 带 _tier (PARAM/CODE/ALGO) 和 _source 元数据
```

### 分数追踪

```python
from autosim.record_score import record_score, get_best_score

# 原子化记录（幂等，兼容 AutoSOTA 格式）
record_score("output/scores.jsonl", iteration=5, idea_id="LLM-005",
             title="Grasp offset tuning", status="success",
             primary_score=0.72, params={...}, is_best=True)

# 查询最佳分数
best = get_best_score("output/scores.jsonl", direction="higher")
```

## 扩展指南

### 添加新的机器人任务

```python
# autosim/tasks/my_grasp.py
from autosim.tasks.task_base import BaseTask, TaskResult

class MyGraspTask(BaseTask):
    TASK_TYPE = "my_grasp"

    def _build_setup_script(self) -> str:
        return '''
from omni.isaac.core.prims import XFormPrim
# 放置待抓取物体...
'''

    def _build_execute_script(self, params) -> str:
        return f'''
# 用指定参数执行抓取
gripper_open = {params['gripper_open']}
# ...
print("AUTOSIM_RESULT: " + json.dumps({{"grasp_success": success}}))
'''

    def _extract_metrics(self, result) -> dict:
        return {"grasp_success": result.get("grasp_success", 0)}
```

### 添加新的优化策略

```python
# 在 autosim/optimizer.py 中添加
class BayesianOptimization(OptimizationStrategy):
    def suggest(self) -> Dict[str, Any]:
        # GP-based acquisition function
        ...

STRATEGIES["bayesian"] = BayesianOptimization
```

### 添加新的仓库适配器

```python
# 在 autosim/onboard/discover.py 中注册
@register_adapter("MyRepo")
def discover_myrepo(repo_path: str, task_name: str) -> dict:
    # 扫描仓库，提取任务和参数
    return {
        "name": task_name,
        "embodiment": [...],
        "params": {...},
        "task_file": "...",
    }
```

### 添加新的 TaskAdapter

```python
# autosim/adapters/my_adapter.py
from autosim.adapters.base import TaskAdapter, ParamDef, EvalResult

class MyAdapter(TaskAdapter):
    def get_param_space(self): ...
    def evaluate(self, params): ...
    def get_source_files(self): ...
```

## 配置参考

### config.yaml

```yaml
# LLM 模型配置
llm_model: deepseek-v4-flash
llm_api_key: ""
llm_base_url: http://10.1.21.21:3000/v1

# 优化参数
eval_seeds: 15           # 每次评估的随机种子数
max_iterations: 24       # 最大优化迭代次数
target_improvement_pct: 30.0  # 目标提升百分比
```

### 环境变量

```bash
# OpenAI-compatible API（LLM 优化模式）
export AUTOSIM_LLM_BASE_URL=http://10.1.21.21:3000/v1
export AUTOSIM_LLM_API_KEY=sk-...
export AUTOSIM_LLM_MODEL=deepseek-v4-flash

# 仿真连接
export ISAAC_SIM_HOST=localhost
export ISAAC_SIM_PORT=8766
```

## 输出格式

### scores.jsonl

每行一条 JSON 记录，兼容 AutoSOTA 格式：

```json
{"iteration": 0, "primary_score": 0.45, "metrics": {"success_rate": 0.45}, "success": true, "params": {"pre_grasp_dis": 0.12, "grasp_dis": 0.01}}
{"iteration": 1, "primary_score": 0.52, "metrics": {"success_rate": 0.52}, "success": true, "params": {"pre_grasp_dis": 0.11, "grasp_dis": 0.02}}
```

### run_*.json

完整运行记录（包含所有迭代详情）：

```json
{
  "task_name": "beat_block_hammer",
  "primary_metric": "success_rate",
  "metric_direction": "higher",
  "baseline_score": 0.45,
  "best_score": 0.72,
  "best_params": {"pre_grasp_dis": 0.08, "grasp_dis": 0.03},
  "best_iteration": 15,
  "num_iterations": 24,
  "improvement_pct": 60.0
}
```

### 优化曲线

运行 `metrics.py` 自动生成 `output/optimization_curve.png`，显示每轮分数和最优曲线。

## Demo 脚本

| 脚本 | 用途 |
|------|------|
| `run_demo.py` | 经典优化 Demo（Franka 到达任务） |
| `run_robot_demo.py` | RoboTwin 任务运行 |
| `run_robotwin_task.py` | RoboTwin 单任务执行 |
| `run_robotwin_opt.py` | RoboTwin 参数优化 |
| `run_act_eval.py` | ACT 模型评估 |
| `optimize_act.py` | ACT 参数优化（v1） |
| `optimize_act_v2.py` | ACT 参数优化（v2） |
| `optimize_act_closed_loop.py` | ACT 闭环优化 |
| `run_strategy_opt.py` | 策略参数优化 |
| `run_visual_demo.py` | 可视化 Demo |
| `run_comparison_demo.py` | 对比 Demo |
| `run_ik_demo.py` | 逆运动学求解 Demo |
| `calibrate_camera.py` | 相机标定工具 |
| `save_aloha_video.py` | Aloha 双臂视频录制 |
| `save_act_video.py` | ACT 策略视频录制 |
| `save_comparison_video.py` | 基线 vs 优化对比视频 |

## 依赖

- **Python** ≥ 3.10
- **numpy** ≥ 1.26 — 数值计算
- **pyyaml** ≥ 6.0 — 配置解析
- **scipy** ≥ 1.10 — CMA-ES 等高级优化（可选）
- **matplotlib** ≥ 3.5 — 优化曲线绘制（可选）
- **Isaac Sim** 4.2.0+ — 真实仿真模式（可选）
- **RoboTwin** — RoboTwin 任务优化（可选）

## 引用

如果 AutoSim 对您的研究有帮助，请引用相关项目：

- **AutoSOTA**: [arXiv:2604.05550](https://arxiv.org/abs/2604.05550) — 闭环自动化研究范式
- **RoboTwin**: [arXiv:2506.18088](https://arxiv.org/abs/2506.18088) — 双臂机器人基准
- **Isaac Sim MCP**: [GitHub](https://github.com/omni-mcp/isaac-sim-mcp) — MCP 仿真控制协议

## License

MIT License
