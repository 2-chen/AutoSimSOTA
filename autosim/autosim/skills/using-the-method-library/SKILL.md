---
name: using-the-method-library
description: 技能库中文导航；按当前问题选择方法，区分文献、历史测量与方法建议，并按需读取正文和参考资料。
scope: general
confidence: orientation — not an experimental result
evidence: 2026-09-29 library audit and primary-source survey; no new simulator run.
---

# 按问题使用技能库

先看短目录，主 Agent 根据当前不确定性选择最多 3 项并记录理由，再读取正文。关键词只是推荐，不替 Agent 决策；用不到可以不选。不要整库阅读或把全部正文塞进 system prompt。

## 方法与运行合同（contract）

技能提供可反驳的方法，不授予执行权限。可以说明为何不采用某条建议，但不能违背运行器的总预算、代码隔离、资源权限、正式评测协议与证据真实性。已有接口不能表达的方法，应明确能力缺口，不杜撰操作。

读取 description 判断触发场景；读取 confidence 区分方法建议、借鉴证据和本地测量；读取 evidence 查来源/版本/局限；再用 method 设计当前实验。没有数字不代表无价值，有数字也不代表可泛化。新条目的细致证据写在正文，未设置 confidence 的元数据会显示 unspecified，不等于高置信度。

benchmark scope 限制历史材料适用范围；general 只表示方法可跨仓库考虑，不表示其中结论普遍成立。官方文档也要与本地 revision 核对。

## 1. 选择研究方向

- [choosing-simulation-improvements](../choosing-simulation-improvements/SKILL.md)：按失败、资源和协议选择下一假设；先区分修复、复现、本地增分与 SOTA。
- [spending-a-budget-across-ideas](../spending-a-budget-across-ideas/SKILL.md)：低保真筛选、长期作业与局部预算分配；先验证代理指标。
- [make-your-prior-earn-it](../make-your-prior-earn-it/SKILL.md)：何时需要随机/简单对照来检验 LLM 选法的价值。

## 2. 接入仓库与修复

- [benchmark-shapes](../benchmark-shapes/SKILL.md)：逐项调查任务、评测、观测、控制与资源差异，无“统一默认值”。
- [adapting-an-unfamiliar-benchmark](../adapting-an-unfamiliar-benchmark/SKILL.md)：从源码/资源建立可核验能力图。
- [finding-how-a-benchmark-runs](../finding-how-a-benchmark-runs/SKILL.md)：追 wrapper 到原生采集、转换、训练、评测入口。
- [what-the-benchmark-does-not-ship](../what-the-benchmark-does-not-ship/SKILL.md)：显式已有资源、官方搜索下载、完整性与消费验证。
- [building-an-environment-that-runs](../building-an-environment-that-runs/SKILL.md)：用最小执行验证版本、硬件与环境。
- [making-a-stage-run](../making-a-stage-run/SKILL.md)：参数/环境/路径/消费者故障分类和原操作复验。
- [the-declaration-is-a-claim](../the-declaration-is-a-claim/SKILL.md)：检验声明的配置值、能力和实际可运行性。
- [read-your-failures](../read-your-failures/SKILL.md)：稳定错误证据、诊断、有限重试与 Fix 交接。

## 3. 数据供给与价值

先验证是否能采集与消费，再判断是否值得扩量。以下条目各回答不同问题，不是连续强制步骤。

- [where-successes-come-from](../where-successes-come-from/SKILL.md)：谁提供动作，是否能自主产生训练轨迹？
- [connecting-native-data-collection](../connecting-native-data-collection/SKILL.md)：原生采集→转换→真实 loader 如何接通？
- [budget-attempts-not-episodes](../budget-attempts-not-episodes/SKILL.md)：失败重试和接受目标实际花多少资源？
- [generator-success-is-not-data-value](../generator-success-is-not-data-value/SKILL.md)：采集产出率与下游策略收益有何区别？
- [diversity-before-volume](../diversity-before-volume/SKILL.md)：新增覆盖还是重复已有场景？
- [more-of-the-same-saturates](../more-of-the-same-saturates/SKILL.md)：真饱和还是欠训练/噪声？
- [adding-more-can-subtract](../adding-more-can-subtract/SKILL.md)：为何加入数据后退步？
- [quality-needs-reweighting](../quality-needs-reweighting/SKILL.md)：是否值得过滤或重加权，而非必须重加权？
- [weight-the-small-set](../weight-the-small-set/SKILL.md)：少量新数据是否得到实际曝光？

## 4. 策略、控制与学习

- [aligning-policy-inputs-and-actions](../aligning-policy-inputs-and-actions/SKILL.md)：训练输入、部署预处理、动作和时序是否一致？
- [validating-policy-training](../validating-policy-training/SKILL.md)：有没有有效更新，哪份产物被实际加载和计分？
- [tuning-action-chunks-and-feedback](../tuning-action-chunks-and-feedback/SKILL.md)：分块、重规划、时序融合与推理延迟。
- [improving-observations-and-memory](../improving-observations-and-memory/SKILL.md)：增强、合法多视角/状态/历史，以及有条件的 3D。
- [adapting-pretrained-policies](../adapting-pretrained-policies/SKILL.md)：匹配权重、官方资源、LoRA/模型切换的实际成本。
- [learning-from-policy-failures](../learning-from-policy-failures/SKILL.md)：专家纠错/DAgger 与原生在线 RL 的不同前提。

## 5. 有效比较与人类记录

- [hold-one-variable](../hold-one-variable/SKILL.md)：发现隐藏的 sampler/config 混杂，区分组合增益与单项归因。
- [control-same-budget](../control-same-budget/SKILL.md)：选择计算/数据成本清楚的对照，发布 checkpoint 仍可作基线。
- [report-paired-and-intervaled](../report-paired-and-intervaled/SKILL.md)：episode 身份、差值区间与独立确认。
- [resolution-before-repetition](../resolution-before-repetition/SKILL.md)：更多样本、更好诊断还是改变假设？
- [progress-beats-binary](../progress-beats-binary/SKILL.md)：用辅助进度解释失败，不更换正式指标。
- [writing-research-updates](../writing-research-updates/SKILL.md)：中文 RUN.md 围绕问题、证据和认识变化写作。
- [presenting-simulation-demos](../presenting-simulation-demos/SKILL.md)：真实视频/图表、样本来源和结论边界。

## 6. 历史材料与按需文献

[robosynchallenge-measurements](../robosynchallenge-measurements/SKILL.md) 仅适用于匹配范围的历史回顾；不是当前代码/资源仍具备相同能力的证明。新仓库不能使用其数值作先验阈值。

需要论文和优先级依据时，读取 [方法调研](../choosing-simulation-improvements/references/literature-review.md)；需要了解删改依据时，读取 [技能审计](../choosing-simulation-improvements/references/library-audit.md)。参考文档不单独注册为技能，不自动注入上下文。

当前显式选读接口返回技能正文，不递归读取这些链接。正文包含可独立使用的方法；参考文件只能在运行环境确实提供相应只读访问时打开，不能假定宿主技能路径已挂载到目标仓库，也不能虚构已读文献。无访问条件时使用正文的来源链接按已有权限调查，或说明未读细节。

## 维护方式

保持现有 ID/目录稳定以支持历史回执；修改内容更新 manifest 版本并保留选择时正文哈希。新增条目同时登记 manifest、触发场景、输入/输出、适用/失效条件和验证方式，并加入本导航。

一项技能应解决一个可判断的问题：证据是什么、何时可用、最小实验、何时停止、怎样避免假提升。模型名、仓库名、论文分数和本机路径不能变成通用规则。这里的方法库不是某次 run 的 idea 库或失败记忆；运行结果仍留在对应账本，成功迁移的经验才考虑提炼。
