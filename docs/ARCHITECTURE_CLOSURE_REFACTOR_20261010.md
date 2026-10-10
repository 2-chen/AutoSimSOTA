# 架构收敛：把付费执行转化为可信分数

目标是仓库驱动的自动指标优化。角色数量、开放式写作、环境安装次数和合同测试数量都不是研究收益。本轮是渐进式收敛，不重写全部运行器，不重置旧实验。

## 一个研究决定中心

Main/Scheduler 选择假设、数据和学习工作量、候选及恢复动作。Data/Resource/Objective/Fix/Supervisor 返回局部调查或审核，不能替换全局目标、改评测、重设总预算。内层 DerivedResearch 执行 Main 已选择的候选，维护测量、最佳版本、回退和历史；保留 legacy 兼容。

`Preparation._decision_facts()` 为 Main、同步/异步任务、Monitor 与 Fix 提供同一实验/数据/接口目录和操作参数合同。子 Agent 可给最多三项 `operation_proposals`：`do / arguments / why / evidence_refs / expected_postcondition`；仍受原 16 KB 报告上限约束。Main 根据当前合同和真实证据审核后执行，提案不是命令授权。较大参数在短记忆中以 SHA 和完整交接引用替代，不截断为半个可执行 JSON。

## 一个实验索引，不复制执行权

`experiment_transaction` 在真实评分调用中记录计划→阶段→回执→测量返回，每次调用独立 ID，绑定 settings、attempt/evidence ID 与 measurement SHA。中断保留已返回阶段；索引 I/O 失败不抹掉原生结果。

原 native/session/finalization 仍负责执行所有权和恢复。索引不能认证分数、清进程、重放未知工作、改 best 或预算；“运行中”不是存活证明，“测量返回”不是候选采用完成。未全面替换既有图执行/候选恢复分支，避免引入第二个真相来源。

`experiment_view` 汇总同步测量、baseline/候选长期训练、事务与原候选比较引用。`receipt_verifier` 不一致的结果不显示核验分数；开发交接排除确认/留出测量及事务。发布参考与本地候选收益分轨。

恢复提示按断边：数据→训练由 Data 跟进；训练→策略选择/归档由 Fix 跟进；策略→rollout/指标由 Fix 跟进；测量→采用/报告由 Scheduler 跟进。仍须用封存 ID 和实际进程/session 证据确认可安全执行，索引不授权重试。

命令派生内的 LLM 修订者可明确返回 `environment_handoff.reason`，由执行器附上本次失败的原生 ID，立即交回 Main 的 `build_the_environment`/Fix 环境规划路线。该故障会反驳旧 passing environment，保留所选 producer 和调用输入，修后再复验原入口；不得藏安装于 staging，也不得把交接说成恢复成功。解释器/工作目录/显示变量可用源码支持的调用修订解决时仍留在派生中，不用缺包关键词自动决定安装。缺原生失败证据的请求不取得环境修复所有权。

明确 `repair_owner=main_agent/resource/environment_planner/environment_executor` 的交接不额外购买 checkout-only Fix：后者不能替 Main 选输入、替 Resource 增预算或替环境规划器执行安装。普通 source/candidate 故障的 Fix 路线和独立核验保持不变。

## 缩小无关失效、复用已接受接口

新 native job 的 `source_stage_scope=[stage]` 只绑定实际执行的命令构造器和参数，因此新 collect/evaluate 构造器不使训练失效。启动前、结束后、独立学习审核、baseline/candidate adoption 共用算法。

全部 inventoried/materialized checkout 源码仍绑定；没有把静态 imports 当完整依赖证明。实际 settings/argv、数据身份、权重与冻结协议继续单独核验。旧 request 没有 scope 时保持原 global SHA，不无证据迁移历史。

旧完成作业在 Main 明确 `inspect_native_job` 且原 global 身份**仍然匹配**时，可封存附加 `source_binding.json`：原 request/result/native receipt SHA、已完成零退出的封存证据、原 global 与当前阶段身份。此后新增无关 builder 可以复用，原记录一字不改；原身份已变、缺封存回执、源码/阶段/原证据变动均不能补证。工作进程启动仍直接检查原 request，附加证明只服务完成作业交接，不授权重放。

`execution_interface` 将 LLM 已接受的阶段适配器保存为内容寻址版本：阶段对象、源码身份、环境记录、显式资源绑定、派生记录、封存原生验证。Main 明确调用 `derive_a_command(stage=..., reuse_verified=true)`，身份及原生证据一致才复用，不重复模型派生/验证。输入变化拒绝，不自动重建环境；startup-only 仍不是正式训练。

当前是同 run 精确复用，不是跨仓库/跨 run 无条件安装包。资源挂载身份不证明数据内容和实际 batch 消费，缓存也不证明环境从此永远可用、学习充分或已计分。跨 run 必须重新绑定资源/输出/环境，不能复制绝对路径。

未完成的调用修订也需要保存：handoff 返回 `invocation_draft_ref`，包含当时的基上下文与 revised row，但仍为 unverified。Main 修改 checkpoint 后，可明确同时选择 `draft_ref` 和 `draft_rebind_inputs=["checkpoint"]`；除该调用输入外，原始阶段声明、入口源码 SHA、任务和解释器必须完全相同。它保留 Agent 修过的包装调用/环境，仍走完整新派生及原生复验，不迁移接受接口/旧分数、不自动选新 checkpoint。旧 draft 只有不可逆 base SHA、没有基上下文的，不能无证据重绑定；其他输入重绑定目前不开放。

真实试点发现第二类空转：调用输入或环境已经改变，LLM 仍引用旧 checkpoint/SDL 回执直接返回 invocation_handoff，而没有执行新调用。现在每次生成命令提供当前调用上下文指纹（阶段声明、除临时输出外的调用输入、实际环境/工作目录和入口文件 SHA）；原生执行将该指纹写入执行器末尾日志，再封存证据。历史交接必须指纹相同、原生封存一致，且不是当前修订链已处理的 ID。receipt 字段被单独修改不能冒充指纹匹配；旧无指纹或不同上下文只作历史建议。它不认证完整依赖源码、资源字节、训练消费、加载或分数；这些仍由对应核验负责，也不限制 Agent 用旧经验设计新命令。

## 缩短计分关键路径

Main 控制候选时，非必要 benchmark 研究问答在既有 finalization 中标记跳过；交给计分提交后的按需任务。评分有效性、源码/策略加载/rollout/metric 核验、rubric、best、源码回退、idea 与 history/session 提交仍保留。

demo 和部分 rubric 仍在 finalization，本轮没有忽略其未知副作用。进一步外移为独立事务需证明冻结依赖、恢复和不重复执行；不能仅吞异常“加速”。

## 数据闭环和人类展示

修复真实 schema 错配：native measurement 的 `scored` 是策略说明文字，data registry 原先错误要求布尔 True。现在不再据该类型拒绝，但明确未计分/发布参考/留出仍拒绝，并在登记前复验原生 receipt。保留 policy/rollout/metric 见证、新增采集祖先、版本实际消费、immutable 记录和公平性独立审核。

Data-first 依据开发失败、coverage、真实 loader/控制及学习充分性判断瓶颈。先小试合法专家与数据消费，再扩大采集；错误标签/重复数据不是收益。不修 loader 或明显欠训就盲目加数据，也不是数据优先。等量 targeted/random、共同质量底线与不增数据等计算续训分别做对照。

RUN.md 中文闭环概览和最近实验表展示核验分数、完成待计分训练、数据消费与下一断边。实时事实不依赖 Recorder 重新写作；使用原有图表/真实原生视频与帧、语义限频，不造展示录像代替实验。效率视图新增训练完成/计分/待计分计数，标注近期有界范围；重叠调度时长不能当实际 GPU 计算或费用。

完整回归还发现 DeepSeek 价格卡在2026-10-10到期，会阻止真实网关启动。已复核 [官方价格页](https://api-docs.deepseek.com/quick_start/pricing/)：Flash USD 费率未变；原有效费率 ID/历史账本保留，增加核验日期和来源，复核期限延至2026-11-10。继续到期拒绝，不停用预算防线、不重算旧用量。

## 相关工作的采用边界

| 工作 | 采用 | 不照搬 |
| --- | --- | --- |
| [AutoSOTA](https://arxiv.org/abs/2604.05550) / [CLI](https://github.com/tsinghua-fib-lab/AutoSOTA/blob/main/cli_guide.md) | 资源、目标、Init/Fix、候选、评测、独立审核，技能与记忆分离 | 论文职责不等于独立进程；CLI 已可运行环境前提不等于任意仿真仓库可接入 |
| [AIDE](https://arxiv.org/abs/2502.13138) | 候选关系、debug/improve/explore、证据摘要 | 仿真候选不仅是脚本，还包含数据、策略及协议 |
| [AI Scientist-v2](https://arxiv.org/abs/2504.08066) | 实验组织、分析、可视化 | 不把论文问答放在每次计分提交前 |
| [MLE-bench](https://arxiv.org/abs/2410.07095) | 独立验收、完整任务测量 | 不把 mock 合同通过当真实指标优化 |
| [Research-agent search](https://arxiv.org/abs/2507.02554) | 操作与搜索策略联合比较 | 不认定 MCTS 必胜，先稳定候选计分 |
| [DataMIL](https://arxiv.org/abs/2505.09603)、[RESample](https://arxiv.org/abs/2510.17640) | 策略相关数据选择、失败/OOD 覆盖 | 失败驱动采数据不自动构成新颖贡献，必须实测消费与公平对照 |

## 分层验收与剩余工作

1. 工程：身份/缓存/事务/交接/留出隔离/展示正负例和完整回归。
2. 原生 CPU：真实 toy train→权重→evaluate→receipt→measurement→事务展示；不是仿真收益。
3. P0 仿真：接续原 RoboSyn pilot，保留环境、已完成训练、原采集及全部账本。Main 自主完成策略→评分命令→正式测量，不由操作者代选权重/参数；框架刷新记录为开发干预。
4. P1 恢复：封存 ID→具体提案→受控修复→复验原断边，不重放未知副作用；既有恢复事务继续生效，统一索引不替代执行核验。
5. P2 数据：新增消费→同协议计分→等量 targeted/random→不增数据等计算对照，无此链不批准论文结果。
6. P3 泛化：先资源可复现的机械臂模仿学习仓库，按执行类型分族、报告资源缺口，再扩 online RL、VLA/多服务、非机械臂。
7. P4 效率：首次可信分数时间/总成本、有效候选、训练采用率、恢复/重复失败/干预、数据净收益；真实候选积累后比较搜索和上下文策略。

工程和 CPU 通过不证明真实仿真/全部跨仓库对照通过。精确测试快照、在线状态与尚待工作写入 `plan/EXECUTION_STATUS.md`。
