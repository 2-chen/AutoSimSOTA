# 论文工作区

核心问题：能否从原生仿真仓库自主构建并演化研究流程，在固定科学协议和有限总资源内持续产生可信指标改进？数据获取／筛选是重要领域分支，其随机／规则对照不是整个系统的唯一目标。系统比较、未见仓库和正式数据对照均未完成，不把多Agent架构本身当贡献。

- `tex/main.tex`：英文匿名方法/实验协议草稿，含 24%→32% 开发案例；正式数据方法对照尚缺；`tex/main.pdf` 为编译预览。
- `OUTLINE.md`：中文论文结构。
- `PILOT_BRIEF.md`：供真实研究系统读取的冻结试点任务书。
- [方法规格](METHOD_SPECIFICATION.md)：系统自主择路／原生连接的定义，以及FDAS数据分支的机制与边界。
- [数据专项执行规范](STUDY_EXECUTION.md)：真实数据试点恢复、直接对照、冻结/多种子/留出；不是系统主实验总协议。
- [系统主实验协议](SYSTEM_STUDY_PROTOCOL.md)与[系统矩阵草案](SYSTEM_EXPERIMENT_MATRIX.json)：原始配方／通用研究／完整系统，按独立研究重复而非采集arm建表；18个草案单元均未分配。tools/system_study.py提供只读覆盖检查及前瞻计划封存，不启动实验／读取留出／签发正式结果；实际任务准入、预算和结果层仍待验收。
- `EXPERIMENT_MATRIX.json`：四候选任务×五方法×三训练seed的获取草案（60单元），包含随机/规则/Agent，尚未冻结或执行；正式先准入两任务。
- [Qt增量草案](EXPERIMENT_MATRIX_QT.json)：增加fdas，同目标获取池上研究Qt（72候选单元）；不复制targeted_quality为重复方法。只读汇总工具为`tools/data_study.py`，两个矩阵都不代表费用授权或已完成结果。
- `evidence/`：只读开发证据快照，不能冒充冻结系统的正式主表。
- `DEVELOPMENT_CASE_ROBOSYN_CLOSURE.md`：10月13日新增真实恢复评测、同权重部署负结果、2k候选20%→30%局部观察及20条专家数据供给试点；已补入英文稿，尚非FDAS数据收益。
- [10月19日固定权重部署消融](DEVELOPMENT_CASE_ROBOSYN_MATCHED_DATA.md)：同一基线策略、相同100个episode seed，执行块50→4，成功率56%→0%，平均推理调用4.42→91、原生耗时674.2→1293.6秒。实际加载／冻结父链／rollout／指标均复验一致；[机器证据](evidence/robosyn_fixed_policy_ablation_20261019.json)。这是有效开发负消融，不是数据收益或SOTA；40条新采数据的学习对照仍未完成。
- [本轮机器证据](evidence/robosyn_closure_20261013.json)：六项有效measurement的聚合分数、原生时长、receipt ID、measurement/bundle/权重hash；全数只读复核，不含原始数据或视频。
- [强制数据轨道的恢复计分](DEVELOPMENT_CASE_ROBOSYN_REQUIRED_DATA.md)：10月15日Main自主接回原round_1训练产物，50集开发评测9/50，对照本轨道6/50；无重训，消费仍unknown，非数据获取收益。[机器摘要](evidence/robosyn_required_data_20261015.json)
- [同轨道官方权重校准](evidence/robosyn_reference_20261015.json)：Main另行完成24/50（48%）参考评测，实际加载／rollout／指标回执一致；不替换本地baseline，不算新数据／自主候选收益或统计等价复现。
- [10月16日诊断候选的恢复负结果](evidence/robosyn_round3_recovery_20261016.json)：Main无重训接回round_3，正式4/50（8%），未超过best18%；之前12%探针不择优替换、不合并，数据消费仍unknown。
- [10月17日自主长训与Fix恢复](evidence/robosyn_training_volume_20261017.json)：Main自行选80K更新，Fix／Main复用原产物完成50集计分12/50（24%），高于本地12%／先前18%、低于单列参考48%；是学习量不等的单seed开发观察，不是数据方法或SOTA结论。已补入英文稿与开发案例；维护暂停时尚未采用，不代改best指针。
- [外部系统原生接入试点](EXTERNAL_BASELINE_PILOT.md)：已补入英文稿，ENPIRE／autoresearch-robotics 的 RoboTwin 基线及候选均0/20；文本 RoboSyn 诊断11/20、候选10/20与6/20、原基线确认21/50；两个图像可用的 random arm 无有效基线。clear诊断与主协议分开，[17项机器证据](evidence/external_native_port_20261016.json)保留实际加载身份及报告偏差。不是正式公平排名或SOTA结论。
- [RoboSyn 开发案例及证据索引](DEVELOPMENT_CASE_ROBOSYN.md)：同一权重下的自主部署参数优化；不宣称 SOTA 或失败驱动数据方法已有效。
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

系统：可信基线→自主干预→原生计分→保留／回退并继续→匹配预算的系统对照→独立研究重复／确认→迁移及全成本／干预审计。数据贡献另外要求真实新增数据→实际batch消费→策略计分→随机／规则／固定数据对照。只完成一种证据不宣称另一种贡献成立；以[总规划](../../plan/MASTER_PLAN.md)的未来验收为准。
