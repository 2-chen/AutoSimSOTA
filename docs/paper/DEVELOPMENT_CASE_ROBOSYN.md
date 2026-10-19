# RoboSyn 开发案例：自主部署优化的可观察收益

记录日期：2026-10-12。性质：开发观察及内部证据索引，不是冻结正式主表。

## 可以写进论文的结果

在 RoboSyn 的 click_bell 任务中，本地训练的 ACT 基线在 100 个开发 episode 上取得 24 次成功。主 Agent 自主提出将每次模型查询后执行的动作数量从 8 增至预测块的完整 16 步。候选复用相同权重，没有新增训练，在后续 100 个开发 episode 上取得 32 次成功：观察到提升 **8 个百分点**。

这表明系统在开发阶段自主选出并执行了一项测得收益的部署优化。它不是 SOTA 结论，也不是失败驱动数据获取与筛选方法的有效性证据。

| 测量 | 成功次数 / episode | 如何解释 |
| --- | ---: | --- |
| 本地训练基线 | 24 / 100 | 本轮优化的原始比较起点 |
| 全动作块执行候选 | 32 / 100 | 复用基线权重的开发提升，无新训练 |
| 已发布权重的本地参考测量 | 55 / 100 | 单独报告，不混作优化起点；协议可比性和官方性能复现仍须审核 |
| 独立确认 | 未执行 | 源码身份检查在 rollout 前拒绝；不是零分，也不是执行后的评测失败 |

本地基线和候选记录均声明 task=click_bell、setting=random、episodes=100、seed=0。政策执行长度属于策略部署参数，不是改任务成功条件。相同 seed 标签不证明初态配对；本例不作显著性、独立留出泛化或多种子稳健性结论。

## 证据索引

以下是内部开发路径，匿名投稿时须转换成不含用户名、私有路径或公开身份的证据编号；不得直接复制整份运行记录进入匿名包。

- [原基线测量](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/research/derived/measurements/baseline.json)：0.24；训练 attempt `bff2e51f548a44efbc8dd43bf48e1b32`；评测 attempt `3dd8f865912a4ce3953aa86a18e8db62`。
- [候选测量](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/research/derived/measurements/round_2.json)：0.32；评测 attempt `fabe6b5c56a94b8aaaca0cacd5f0dde8`，原生退出码 0。
- [Agent 候选库](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/research/derived/ideas.json)：`act-full-chunk-execution` 的机制、代码改动和源依据。
- [候选执行回执](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/action_receipts/557387ecf9831f71.json)：由系统请求并执行，不是操作者手动选定候选参数。
- [发布权重参考测量](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/research/derived/measurements/reference_recovered_13f958906285408c.json)：0.55；不等同于已认证的官方复现。
- [确认前置失败](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/research/derived/measurements/confirmation.json)：`ran=false`，`where=source_identity`，原文为 `baseline and selected policy do not both have matching, content-addressed source states for independent confirmation`。
- [干预日志](../../autoresearch_runs/robosyn_fdas_preregistered_20261011_v1/OPERATOR_INTERVENTIONS.md)：框架修复、预算修订、操作者验收与重启；整轮不得描述为无人介入的冻结系统运行。

基线和候选复用的策略 SHA-256：

`377d27c9d8d9f9fddcdcc6a15e83369a7b493728c465f011dbdee81153906c69`。

## 归因与边界

- 收益来自动作执行长度的开发优化，不来自新增数据或新权重。已评分候选明确记录 `no_new_training_or_data_claim=true`。
- 先前限时采集留下 69 个 episode parquet 文件，尚无完整数据健康认证及候选训练消费证据；不能写成“新增数据使成功率达到 32%”。
- 独立确认的失败被外层概括为“没有可用分数”，误导了后续 Agent 调查；这里按底层 `ran=false` 记录为前置拒绝，不采用 Agent 的执行后失败推断。
- 训练数据消费绑定仍标记 unknown；权重加载与 rollout 指标证据不替代完整数据来源/公平性证明。
- 开发搜索后选择最佳结果有选择偏差；100 个 episode 不是 100 次独立训练或 Agent 复现。
- 数据方法主表仍缺等量随机、等计算固定数据续训、多种子与独立确认，不用此例填充这些空白。

## 稿件中的英文表述

> In a development case on RoboSyn click_bell, the agent selected a deployment-side action-horizon change while reusing the same frozen ACT weights. The observed success count increased from 24 to 32 out of 100 development episodes, an 8-percentage-point increase. This single development comparison does not establish statistical significance, held-out improvement, or SOTA. Independent confirmation was rejected before rollout by source-state identity checks. The enclosing run required operator framework repairs and restarts, and the result does not validate failure-driven data acquisition, since newly collected outputs were not consumed by the scored candidate.

正式结果补齐后保留这条开发记录，不回溯修改其身份或把后来的确认冒充本轮已完成结果。
