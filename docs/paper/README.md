# 论文工作区

核心问题：在相同合法专家、数据量或计算预算下，开发失败驱动的自主数据获取与筛选，能否优于随机扩充和固定数据续训？不是把多 Agent 架构本身当成贡献。

- `tex/main.tex`：英文匿名方法/实验协议草稿，结果尚缺；`tex/main.pdf` 为编译预览。
- `OUTLINE.md`：中文论文结构。
- `PILOT_BRIEF.md`：供真实研究系统读取的冻结试点任务书。
- `evidence/`：只读开发证据快照，不能冒充冻结系统的正式主表。
- 方法、门槛和对照详见 [研究协议](../FAILURE_DRIVEN_DATA_RESEARCH.md)。

## 编译

```bash
cd docs/paper/tex
TEXINPUTS=vendor//: BSTINPUTS=vendor//: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

当前 vendor 是固定提交的官方 2026 模板，2027 最终格式仍须复核。`0000` 不是正式投稿编号。

## 导出真实开发证据

```bash
PYTHONPATH=autosim:. .venv/bin/python tools/export_data_research.py \
  autoresearch_runs/robosyn_fdas_pilot_20261008_v1 \
  autoresearch_runs/robosyn_fdas_pilot_20261008_v1/checkout \
  docs/paper/evidence
```

导出器复验 measurement 回执和当前数据身份，仅输出成功核验的观察。初始状态摘要不是实验结果；未知费用和活动 GPU 租约不能视作已完整结算。Episode CI 不替代训练种子/独立 Agent 复现；无初始状态哈希时不得冒充配对评测。

## 投稿门槛

真实新增数据 → 被训练实际消费 → 权重实际加载 → 原生 rollout → 等量随机对照 → 固定数据等计算对照 → 多种子与留出 → 人工干预审计。缺少任一关键证据时不写已完成、不填数字。
