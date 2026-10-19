# FDAS 实验矩阵

研究：fdas_acquisition_equal_data_draft_20261015；轨道：equal_accepted_data；阶段：draft

预登记计划不等于正式结果；缺失不填零，不删除失败单元，不挑最好种子。

## 计划中的研究问题（不是已完成结论）

- 失败定向获取相对随机获取：random_quality → agent_targeted
- Agent目标选择相对固定规则：rule_targeted → agent_targeted

本矩阵不检验Qt筛选增量；不能据此声称获取与筛选均已有效。
本轨道不自动等于等总计算；候选任务、seed ID和计划单元均不代表资源授权或独立重复已完成。

## 逐单元状态

| 任务 | 方法 | 策略 seed | Agent 重复 | 状态 | 开发指标 |
| --- | --- | ---: | ---: | --- | ---: |
| bell | fixed_official | 0 | 0 | 待安排 | 待执行 |
| bell | fixed_official | 1 | 0 | 待安排 | 待执行 |
| bell | fixed_official | 2 | 0 | 待安排 | 待执行 |
| bell | fixed_continuation | 0 | 0 | 待安排 | 待执行 |
| bell | fixed_continuation | 1 | 0 | 待安排 | 待执行 |
| bell | fixed_continuation | 2 | 0 | 待安排 | 待执行 |
| bell | random_quality | 0 | 0 | 待安排 | 待执行 |
| bell | random_quality | 1 | 0 | 待安排 | 待执行 |
| bell | random_quality | 2 | 0 | 待安排 | 待执行 |
| bell | rule_targeted | 0 | 0 | 待安排 | 待执行 |
| bell | rule_targeted | 1 | 0 | 待安排 | 待执行 |
| bell | rule_targeted | 2 | 0 | 待安排 | 待执行 |
| bell | agent_targeted | 0 | 0 | 待安排 | 待执行 |
| bell | agent_targeted | 1 | 0 | 待安排 | 待执行 |
| bell | agent_targeted | 2 | 0 | 待安排 | 待执行 |
| drawer | fixed_official | 0 | 0 | 待安排 | 待执行 |
| drawer | fixed_official | 1 | 0 | 待安排 | 待执行 |
| drawer | fixed_official | 2 | 0 | 待安排 | 待执行 |
| drawer | fixed_continuation | 0 | 0 | 待安排 | 待执行 |
| drawer | fixed_continuation | 1 | 0 | 待安排 | 待执行 |
| drawer | fixed_continuation | 2 | 0 | 待安排 | 待执行 |
| drawer | random_quality | 0 | 0 | 待安排 | 待执行 |
| drawer | random_quality | 1 | 0 | 待安排 | 待执行 |
| drawer | random_quality | 2 | 0 | 待安排 | 待执行 |
| drawer | rule_targeted | 0 | 0 | 待安排 | 待执行 |
| drawer | rule_targeted | 1 | 0 | 待安排 | 待执行 |
| drawer | rule_targeted | 2 | 0 | 待安排 | 待执行 |
| drawer | agent_targeted | 0 | 0 | 待安排 | 待执行 |
| drawer | agent_targeted | 1 | 0 | 待安排 | 待执行 |
| drawer | agent_targeted | 2 | 0 | 待安排 | 待执行 |
| assembly | fixed_official | 0 | 0 | 待安排 | 待执行 |
| assembly | fixed_official | 1 | 0 | 待安排 | 待执行 |
| assembly | fixed_official | 2 | 0 | 待安排 | 待执行 |
| assembly | fixed_continuation | 0 | 0 | 待安排 | 待执行 |
| assembly | fixed_continuation | 1 | 0 | 待安排 | 待执行 |
| assembly | fixed_continuation | 2 | 0 | 待安排 | 待执行 |
| assembly | random_quality | 0 | 0 | 待安排 | 待执行 |
| assembly | random_quality | 1 | 0 | 待安排 | 待执行 |
| assembly | random_quality | 2 | 0 | 待安排 | 待执行 |
| assembly | rule_targeted | 0 | 0 | 待安排 | 待执行 |
| assembly | rule_targeted | 1 | 0 | 待安排 | 待执行 |
| assembly | rule_targeted | 2 | 0 | 待安排 | 待执行 |
| assembly | agent_targeted | 0 | 0 | 待安排 | 待执行 |
| assembly | agent_targeted | 1 | 0 | 待安排 | 待执行 |
| assembly | agent_targeted | 2 | 0 | 待安排 | 待执行 |
| rearrange | fixed_official | 0 | 0 | 待安排 | 待执行 |
| rearrange | fixed_official | 1 | 0 | 待安排 | 待执行 |
| rearrange | fixed_official | 2 | 0 | 待安排 | 待执行 |
| rearrange | fixed_continuation | 0 | 0 | 待安排 | 待执行 |
| rearrange | fixed_continuation | 1 | 0 | 待安排 | 待执行 |
| rearrange | fixed_continuation | 2 | 0 | 待安排 | 待执行 |
| rearrange | random_quality | 0 | 0 | 待安排 | 待执行 |
| rearrange | random_quality | 1 | 0 | 待安排 | 待执行 |
| rearrange | random_quality | 2 | 0 | 待安排 | 待执行 |
| rearrange | rule_targeted | 0 | 0 | 待安排 | 待执行 |
| rearrange | rule_targeted | 1 | 0 | 待安排 | 待执行 |
| rearrange | rule_targeted | 2 | 0 | 待安排 | 待执行 |
| rearrange | agent_targeted | 0 | 0 | 待安排 | 待执行 |
| rearrange | agent_targeted | 1 | 0 | 待安排 | 待执行 |
| rearrange | agent_targeted | 2 | 0 | 待安排 | 待执行 |

## 汇总（含完整性分母）

| 任务 | 方法 | Agent 重复 | 已核验 / 计划 | 均值 | 样本标准差 | 全部完成 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| bell | fixed_official | 0 | 0/3 | 待执行 | 待执行 | False |
| bell | fixed_continuation | 0 | 0/3 | 待执行 | 待执行 | False |
| bell | random_quality | 0 | 0/3 | 待执行 | 待执行 | False |
| bell | rule_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| bell | agent_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| drawer | fixed_official | 0 | 0/3 | 待执行 | 待执行 | False |
| drawer | fixed_continuation | 0 | 0/3 | 待执行 | 待执行 | False |
| drawer | random_quality | 0 | 0/3 | 待执行 | 待执行 | False |
| drawer | rule_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| drawer | agent_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| assembly | fixed_official | 0 | 0/3 | 待执行 | 待执行 | False |
| assembly | fixed_continuation | 0 | 0/3 | 待执行 | 待执行 | False |
| assembly | random_quality | 0 | 0/3 | 待执行 | 待执行 | False |
| assembly | rule_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| assembly | agent_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| rearrange | fixed_official | 0 | 0/3 | 待执行 | 待执行 | False |
| rearrange | fixed_continuation | 0 | 0/3 | 待执行 | 待执行 | False |
| rearrange | random_quality | 0 | 0/3 | 待执行 | 待执行 | False |
| rearrange | rule_targeted | 0 | 0/3 | 待执行 | 待执行 | False |
| rearrange | agent_targeted | 0 | 0/3 | 待执行 | 待执行 | False |

策略训练种子与完整 Agent 重复是两种实验单位；上述统计只描述观测，不能证明独立性或因果收益。

## 当前断边

- 未定协议：model_configuration
- 未定协议：resource_manifest
- 未定协议：native_source_revision
- 未定协议：resource_budget
- 未定协议：accepted_new_episodes
- 未定协议：official_data_identity
- 未定协议：expert_identity
- 未定协议：legal_training_domain
- 未定协议：initialization
- 未定协议：training_work
- 未定协议：data_mixture
- 未定协议：development_evaluation
- 未定协议：heldout_reservation
- 正式准入待验：independent native seed/work and method-fairness audit
- 正式准入待验：held-out outcomes for every prespecified arm, not only the selected best
- 正式准入待验：per-arm cost attribution including failed acquisition and shared initialization
- 正式准入待验：independent system replication and anonymous release audit

费用按原 run 去重展示在 JSON 中；不能按方法数量平均分摊，不能把未知预留当零。
