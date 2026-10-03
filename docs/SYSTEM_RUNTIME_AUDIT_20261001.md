# AutoSimSOTA 运行逻辑审计

> 修复更新：以下正文保留修复前的审计证据。2026-10-01 用户授权修复后，A01–A12 已完成代码修正和回归用例；设计缺口的处理范围、未完成项和最终测试结果见文末“修复交付”。不把合同测试通过等同于真实 baseline/SOTA。

日期：2026-10-01。检查对象：当前工作树、RoboSyn 原 run 的实际轨迹，以及启动、续跑、调度、安装、原生作业、证据和 RUN.md 之间的连接。

本轮是检查，不是修复发布：未修改生产 Python，未手动安装实验依赖，未停止或重启在线服务，未调用付费模型替实验 Agent 操作。新增此报告和独立诊断脚本。

## 结论

当前问题不能归结为“仓库资源不足”或“LLM 找不到路径”。存在真实的框架缺陷，也存在 Agent 能力与执行接口不匹配。

最重要的断点是：**发现问题 → 获得合法可执行的修复 → 执行修复 → 用新证据解除旧故障 → 恢复计分**，这些步骤尚未构成统一、可恢复的事务。系统已经有 Scheduler、Fix、Monitor，但各自返回报告不等于这条链条完成。

本次整理 **12 项明确缺陷、8 项运行设计缺口、2 项当前 RoboSyn 接线/兼容性问题**。不把全部问题都说成本次 RoboSyn 停滞的原因：其中部分为故障注入发现的潜在阻断；当前轨迹直接证实的是安装恢复循环、权限错配、报告格式失败、源码接线问题及流式协议失败。

## 实际运行证据

对象：[RoboSyn RUN.md](../autoresearch_runs/robosyn_installrecovery_20261001_v1/RUN.md)。服务 `autosim-robosyn-installrecovery-20261001-v1.service`，审计期间只读检查仍为 active/running，MainPID 3095475。这只表示控制进程存活，不表示仿真实验成功。

约 16:33 CST 的读取快照：

- 原生环境操作记录 53 条：returncode=0 为 27 条，returncode=1 为 25 条，另 1 条未知。这些是安装/探针，不是 53 次训练实验。
- 主控制器仍在准备阶段，尚无正式 baseline、训练/评测分数。
- 最近环境失败 ID：`d9e2b618f3e141fdbdeeee64d96cbb80`，原操作耗时 613.4 秒。
- 安装替换提案被拒绝：`unsafe replacement source reference`，`native_operation_launched=false`。后续 checkpoint 再次返回同一个旧失败 ID。
- Init 的原生诊断发现 EmbodiChain 搜索路径不匹配，以及 `embodichain_tasks` 导入失败；报告同时明确说无法写 RUN ROOT，因此没有执行主 Agent 指派的 wheel 安装。
- 一轮 Init 调查因 `research report exceeds 16000 bytes` 整体拒收，主 Agent 又发起了一轮压缩报告的调查。
- 新增 Scheduler 错误 ID：`7ac7cec75ea04c5a8b88c1770ed238cc`。进程 rc=0、未超时，但没有最终响应，收到不完整 JSON 行，框架分类为 `stream_protocol`。不能据此断言是 DeepSeek 服务端错误；CLI/网关/输出链路仍需分层定位。

证据读取来自 `run_state.json`、`environment.json`、`provision_progress.json`、原 run 的 `agent/turn_failures/` 和隔离 checkout，不改写这些文件。

## 一、明确缺陷

严重度含义：P0 可造成执行权永久卡住；P1 可阻断关键路径或错误处理；P2 影响调度质量、资源效率和可观察性。

### A01 · P0：长期作业启动不是完整事务，启动失败遗留永久占位

位置：[native_jobs.py](../autosim/autosim/research/native_jobs.py)，`submit()` / `_active()`。

先落盘 request、control，再调用 `Popen`。如果工作目录、进程启动或日志打开失败，未写失败终态，也没有回滚占位。`_active()` 只看 request 存在且 result 不存在，继续把它视为持有执行权。

后果：prepare 会禁止环境构建、命令推导、声明刷新、计分等操作。没有真正启动的作业也能封死关键路径。

复现：故障注入 `Popen → OSError` 后，`active_jobs()` 仍有一个作业，且无 result。应将“占位→启动→进程身份确认”做成可核对事务，启动失败写封存终态；不得凭空认定未知副作用不存在。

### A02 · P0：死作业取消后不释放执行权

位置：同上，`status()` / `cancel()` / `_active()`。

死 worker 无 result 时显示 `outcome_unknown`。`cancel()` 只是写 `cancelled=true`，依赖 worker 自己处理；死 worker 不会再处理，所以 request 永远保持 active。

复现：构造身份已退出、没有 result 的作业，取消后仍占位。需要单独的 reconciliation：核对 worker 与原生子进程/回执，明确保留未知结果、禁止采用其指标，然后关闭已核对的所有权。不能直接删除目录或无条件释放活作业。

### A03 · P1：续跑队列中的“已批准无法继续”被降格成普通 checkpoint

位置：[provision.py](../autosim/autosim/research/provision.py)，`build()` 的失败队头预检。

`resume()` 对独立审核通过的 unbuildable 返回 `([], None)`。普通失败处理检查 transcript 的 unbuildable，但新加入的队头预检将其当作“没有修复方案”，直接 `checkpoint()`。

复现：同样经过独立边界审核，续跑路径返回 `yielded / scheduler_checkpoint`，而非明确边界结果。会导致 Scheduler 继续尝试已经批准关闭的路线。需要有类型的恢复结果，不能让“拒绝、边界成立、暂未产出方案”共享空列表语义。

### A04 · P1：子 Agent 报告格式失败导致整轮调查丢失，没有报告级补救

位置：[main_agent.py](../autosim/autosim/research/main_agent.py)，`validate_report()`；[prepare.py](../autosim/autosim/research/prepare.py)，`_step_research_task()`。

报告同时有字符、条数和 UTF-8 总字节约束；`max_tokens=4000` 不保证中文 JSON 小于 16000 字节。接收后直接 validate，失败抛出，没有先封存原始报告、再请求一次只压缩结构的修正。

本 run 已真实发生。诊断用例也说明字段都在单项限额内，整体仍会超字节。**限制大小本身不是 bug；缺少可恢复的格式处理才是。** 应保留原轮次的来源和观察，不重做调查，更不能截断后假装结果完整。

### A05 · P1：原生作业代码身份漏掉运行中获取/新增的实际源码

位置：[native_jobs.py](../autosim/autosim/research/native_jobs.py)，`source_identity()`。

只哈希最初 workspace_snapshot 的 source_entries 和 derived_stages，后来下载的引擎、新增训练 bridge 等不进入身份。

复现：新增 dependency.py，计算身份，然后修改其内容，身份不变。前后代码一致检查可能放过实际执行代码变化。

应将 Agent 物化的实际源码作为显式来源加入运行代码清单，至少绑定所选入口及其依赖闭包；不能通过把 checkpoint、视频、数据统统哈希来解决。

### A06 · P1：子 Agent 混合目录快照漏读新下载源码

位置：[agent_tasks.py](../autosim/autosim/research/agent_tasks.py)，`snapshot()`。

只有 scope 完全找不到初始清单文件时，才扫描 materialized source。如果目录同时有原始文件和新下载源码，只复制原始文件，静默遗漏新文件。

复现：scope `.` 内原有 train.py、新增 new_dependency.py，快照只有 train.py。新 dependency 单文件 scope 虽能工作，目录 scope 仍不完整。可能造成 Resource/Fix 错判缺模块、看不到真实消费者。应安全合并两类来源，并明确展示任何过滤或不完整范围。

### A07 · P1：局部动作预算用历史总步数计算，与实际循环不一致

位置：[prepare.py](../autosim/autosim/research/prepare.py)，`_run_cycle()`、控制决策/ActionReceipt 的 `controller_actions_remaining`。

循环按当前 segment 执行 `max_steps` 次，但公布的剩余额度是 `max_steps - len(self.steps)`。历史续跑、派生探针步骤也计入 len。

进入新段且历史已有 80 步时，即使本段尚未执行任何动作，公布余额也为 0；实际还能执行新段。会误导 Scheduler/Monitor 认为没有修复空间，证据账本也与执行器不一致。应分开记录全局历史、segment 动作和派生操作，不改变总 GPU 预算。

### A08 · P2：“最近动作”其实是首次出现的步骤类型顺序

位置：同上，`_latest_per_step()`，以及 Monitor、Fix、技能选择调用端。

字典更新不会改变插入顺序，因此输出按每种 step 首次出现的次序排列。对结果取 `[-6:]` 不是时间上最近六次操作。

复现：build 最早出现，六种其他动作后又 build 失败，最近列表中没有 build。当前失败仍能通过 failed_action 单独传入，但上下文中的“最近过程”和技能推荐会失真。需要分别提供 latest_by_operation 和真实 chronological recent_actions。

### A09 · P2：无进展指纹仍哈希整份环境遥测

位置：同上，`_supervision_fingerprint()`。

注释要求排除时间戳/遥测，但直接哈希完整 facts.environment，其中含 latest_installation_attempt、repair_rejection、资源清单元数据等。

复现：只改同一证据的 seconds，指纹改变；没有任何新能力或新执行结果。实际有没有因此跳过某次停滞检测尚未测定，但实现没有满足其排除遥测的契约。应明确哈希故障身份、待办版本、能力验证及实际产物，而非整个显示对象。

### A10 · P2：RUN.md 中 Agent 执行耗时错误归零

位置：[run_record.py](../autosim/autosim/research/run_record.py)，`refresh_live_status()`；[agent_runtime.py](../autosim/autosim/research/agent_runtime.py) 的事件/心跳调用。

无 running.json 时，只有 fallback_current 与 current_action.step 完全相同才采用开始时间。实际 fallback 是 `waiting for model/tool event` 或事件名，step 是 coding_agent_turn。

复现：已开始的 Agent turn 显示 0 秒。当前 RUN 的这类显示也已观察到。应按稳定 action/turn ID 关联，不按展示文本比较；分别显示 Agent 耗时、原生作业耗时、无新证据时长。

### A11 · P1：不可读长期作业仍阻塞执行，却从 Agent 状态中消失

位置：[prepare.py](../autosim/autosim/research/prepare.py)，`_native_job_view()`；[native_jobs.py](../autosim/autosim/research/native_jobs.py)，`_active()`。

request 已落盘但 control/result 损坏或缺失时，active 仍占位，view 捕获异常后直接 continue。主 Agent 看不到阻断它的作业，也拿不到结构化的异常证据。

这是由代码分支确认的故障路径，本轮未对真实 run 注入坏文件。应显示 unreadable/outcome_unknown 占位、错误证据及合法 reconciliation 入口，而不是隐藏。

### A12 · P1：合法大源码引用被一律判作“不安全”，没有片段引用入口

位置：[provision.py](../autosim/autosim/research/provision.py)，`review_probe_replacement()`。

source_ref 的行号被剥离，随后按整文件大小检查；超过 65536 字节就返回 `unsafe replacement source reference`。对大训练器/仿真环境文件，即使只引用几行正确源码，也无法通过。

复现：正常的约 75 KB 源码文件引用被拒绝。保持路径、凭证和读取大小的保护是必要的；应提供有界行段/摘要、文件哈希及准确拒绝原因。当前实际安装提案的同名拒绝是否就是这个大小条件触发，尚未证明，不能混为一谈。

## 二、运行设计缺口：不能简单靠再给 Agent 更多 token 修复

### R01 · 高优先级：任务要求与工具权限不匹配

主 Agent 把 wheel 材料化、安装实际 run-local prefix 的任务交给 research_task(init)。Init 可以编辑隔离 checkout、做 CPU 诊断，但不能写实际前缀/缓存；原生检查挂载是只读，普通诊断也不暴露完整 RUN ROOT。

这是合理的隔离边界，**不能通过给任意 Bash 全盘写权限解决**。缺的是可提交、审核、执行并复验的安装/配置操作接口。Init 本轮已明确反馈权限不允许，说明 LLM 察觉了，执行连接却没完成。

### R02 · 高优先级：已存在的解释器与已验证环境混在一个字段里

env_python() 只有所有环境 probes 通过才返回解释器，Scheduler 的 environment.interpreter 也因此为空。实际 run-local Python 和 native_context 已存在，但状态没有明确展示 existing_unverified_interpreter。

禁止未验证环境进入正式计分正确；把“未核验”显示成“没有解释器”不正确。应分别表达解释器存在、已安装能力、已通过消费者探针、仿真 readiness、正式计分资格。

### R03 · 高优先级：恢复提案被拒绝，却继续用旧执行失败充当本次失败

队头预检没有执行新原生操作时，checkpoint() 取历史 latest_attempt。prepare 据其 ok=false 又报告“一次原生准备操作已封存”、checkpoint failure，再调用 Monitor。

当前轨迹同一 d9e2… ID 重复返回，Agent 因历史失败次数逐渐避开 build，改走 Recorder、harness、research_task 等不能完成安装的路径。安装恢复事务还用连续相同 template 的失败次数关闭 retry，不表达源码/环境/方案版本是否已变化。

应区分 proposal_rejected、repair_pending、native_failed、native_succeeded；前者交回拒绝证据和可修订契约，而不是增加原生重试数。允许新方案或新前提下再验证，禁止无新证据重放。

### R04 · 高优先级：流式协议故障证据不足以定位输出链路

新错误中 rc=0、未超时、没有 final_result，最后只有不完整 JSON 行。runtime 收集了 bounded stdout/stderr，但返回/封存主要保留投影事件及截短错误，未持久化完整的有界脱敏原始流。

因此现在无法可靠区分 CLI 提前退出、stdout 尾部不完整、协议消息类型问题、网关请求异常。本轮不将严格拒绝无最终结果的响应判为 bug，也不建议采用半条 JSON 作为执行命令。

应封存逐流字节数、尾部完整性、终态/EOF、进程及网关请求身份；保存有界脱敏原始流或稳定证据引用，然后让 Fix 定位。报告不得泄露 auth header、环境密钥或私有内容。

### R05 · 高优先级：长期作业只解决部分阶段，不覆盖卡住的引导阶段

submit_native_job 需要已验证 derived stage，安装/环境引导不能提交；安装仍同步执行，默认单次 3600 秒，探针 1800 秒。日志正文在命令结束后才写入主日志，主 Scheduler 等操作结束才能重新决定。

候选阶段另有接口断点：detached producer 的正式采用只允许 initial baseline，不能据此认为 candidate 长作业完整接入计分。

建议统一 job 生命周期，但保持安装、采集、训练、评测各自的权限和协议；通过资源阶段类型区别，而不是 benchmark 名字分支。

### R06 · 中高优先级：harness 自修复目前是候选提案，不是自主发布能力

[harness_repair.py](../autosim/autosim/research/harness_repair.py) 只允许两个辅助模块，activation_allowed=false，也没有自动测试、灰度加载、复验和回滚链。prepare/provision 等主要恢复逻辑不在范围内。

这样避免系统自改崩溃是必要保护，但目前不能期待 Fix 自动修掉本文多数框架错误。需要独立验证/发布边界，不能仅扩大 ALLOWED 或打开 live Python 写权限。RUN.md 应准确称“修复候选已提出”，不能称“框架已修复”。

### R07 · 中优先级：CLI 模型窗口与实际 specialist 窗口不一致

即使启动参数 agent-timeout-seconds=900，research_task 和异步调查调用仍固定最多 240 秒；独立审核等又有 180 秒窗口。Agent 不能用当前任务参数申请调整它们。

因此用户以为给了 900 秒，实际关键角色仍只有四分钟。这不是总 GPU 预算耗尽，应在状态中展示“配置上限 / 本操作有效窗口 / 谁决定 / 能否调整”。

### R08 · 中优先级：可编辑调查一开始就清空所有已验证命令

_step_research_task 对 init/fix/scheduler 在执行前设置 resurvey/metric revalidation，并归档后清空 stages、parameters、verified。即使最终只做只读诊断，或者格式失败，也要重新推导。

提前失效能防止未知副作用污染计分，不建议取消保护。应区分只读诊断与编辑事务；编辑路径以实际差异、消费者依赖和异常终态决定重新验证范围，减少已证实未改动时的全量重做。

## 三、当前 RoboSyn 本身的连接问题

### N01：EmbodiChain 目录布局与评估入口不一致

隔离 checkout 的 scripts/eval_policy.py:37–41 默认从 checkout 的兄弟目录找 EmbodiChain，而实际引擎位于 checkout/EmbodiChain。该入口支持 EMBODICHAIN_ROOT，因此有合法配置连接路线，不应继续只靠重复 pip 安装。

Init 报告已通过实际导入链指出这个问题。本轮只读检查源码确认默认路径计算和环境变量入口；未人工修改实验配置。后续应由系统提出配置、执行原生探针，并记录实际消费者采用的值。

### N02：任务源码需要 embodichain_tasks，当前获取的引擎接口可能不匹配

click_bell.py:25 导入 embodichain_tasks.tableware.base_agent_env。Init 修正诊断中的搜索路径后，导入链停在此模块；Scheduler 进一步观察所取引擎有 embodichain/lab/gym/envs/tasks/tableware 而非该顶层命名空间。

这是版本/包布局兼容性问题的强线索，尚不能断言资源根本不存在。需核对 RoboSyn 原生要求的引擎 commit/tag、安装方式、私有/公开任务包，以及旧成功环境的实际来源。不要伪造一个同名空模块绕过错误，也不要仅凭模块相似就改评测任务实现。

## 四、为什么以前能跑，现在反复卡住

旧成功路径可能保留了手工/历史环境中的引擎路径、包版本、已安装模块和可用资源；新隔离路径将这些隐式连接切断，这是需要适配的真实差异。但框架又叠加了声明、验证、审核、续跑和预算门槛，而每个门槛的失败还未接进一致的恢复接口。

于是容易形成：旧故障仍未解除 → 安装替换未批准 → 返回旧失败 → Monitor 提醒不能重复 → 主 Agent 改派没有安装权限的角色 → 角色交付失败/只返回诊断 → 再次调度，直到模型协议/局部窗口/预算使控制流暂停。

这里并非没有 Agent，也并非 Agent 完全不理解源码。最新 Init 已找出路径和模块问题。关键是其判断尚不能稳定变成受控的修复和原操作复验。

## 五、测试结果与覆盖空档

[诊断脚本](audits/runtime_20261001_reproductions.py) 执行结果：**10 passed**。它的含义是“复现了当前坏行为/约束边界”，不是“这十项已经修好”。全部使用临时 fixture 或 mock，不启动真实 GPU 实验、不访问模型 API。

```bash
PYTHONPATH=autosim .venv/bin/python -m pytest -q docs/audits/runtime_20261001_reproductions.py
```

本轮同时运行现有 install_queue_refresh、refresh_continuation、main_agent、harness_repair 四组合同测试：**43 passed**。合同测试通过，与本次缺陷同时存在，说明现有测试主要验证局部预设交接，未覆盖所有跨层故障路径。

需要增加的验收不是再统计多少 unit test，而是：

1. 真正用户 CLI 入口，从已存在但未核验的环境到 baseline；不人工修配置。
2. 真实/模拟安装替换拒绝后，修订提案并产生不同的新原生回执。
3. worker 启动失败、controller/worker 分别崩溃、缺失回执后的安全续跑。
4. 下载引擎、修复 bridge 后代码身份与子 Agent 视图一致。
5. 报告超长、模型 JSON/协议错误只恢复交付，不重复无关实验。
6. 安装/采集/训练长期作业可观察、可取消、可延长，并在复验后正式计分。
7. 同一故障重复时不无限触发 Monitor/Recorder，变化确实到达实际能力/原生证据才算进展。

## 六、建议修复顺序与完成定义

### 第一批：先使失败状态可见、执行权可恢复

优先 A01、A02、A11、R03、R04。完成标准：每个占位和失败都有稳定身份、责任执行器、合法后续操作和可核对终态；不会出现“作业不可见但封死全部操作”。

### 第二批：修复 Agent 到执行器的连接

优先 R01、R02、A03、A04、A12。提供有版本的安装/配置提案：操作、cwd、执行路径 alias、父错误 ID、源码片段、预计能力、原操作 disposition；独立审核后交执行器，修后核验真实消费者。主 Agent 可以修订提案，不直接改安全内核。

### 第三批：一致的代码身份与调度上下文

优先 A05、A06、A07、A08、A09、R08。分开源码清单、资源绑定、产物清单；明确当前段预算和最近事件。保留旧结果但不要让旧累计失败掩盖新方案。

### 第四批：长作业与人类可观察性

优先 R05、R07、A10。先支持准备期的可观察/可取消作业，再补候选的训练产物采用和正式评测。RUN.md 展示“正在执行什么、多久没有新证据、当前错误和恢复方案、还缺哪个真实验收”，不要只展示程序存活。

### 第五批：harness 候选验证与最小真实验收

完善 R06 的影子测试、受控发布、回滚，不让 Fix 直接覆盖在线框架。用一个资源可复现的机械臂仓库证明完整链条，再用第二个仓库证明泛化；两者都需保存数据消费、checkpoint 实际加载、rollout 和指标身份。绝不以 mock baseline、导入成功或“Agent 已建议修复”代替成功实验。

这轮不建议先增加 Agent 数量、继续翻倍预算、添加 benchmark 特例或重开 run。应先修复这些通用控制与执行连接，再让原任务在既有资源和账本上续跑。

## 修复交付（2026-10-01）

### 明确缺陷的修复

| 项目 | 已实现的变化 |
| --- | --- |
| A01 | worker 启动失败写带证据的 failed 终态，不保留无执行副作用的永久占位。 |
| A02 | 死 worker 取消进入 reconciliation；须确认 worker 和该 job 全部原生子进程已退出，否则不释放，不采纳未知指标。 |
| A03 | 失败队头 preflight 保留独立批准的 unbuildable 语义，并关闭对应 cursor；不降格为普通 checkpoint。 |
| A04 | 报告先严格核验；不合格时封存原文，只做一次只读格式压缩重交，不重跑调查，原始证据仍可读取。 |
| A05 | 原生 job 身份新增有界公共源码扫描，捕获后来获取/新增的实际代码变化；不纳入数据、模型和录像目录。 |
| A06 | 目录 scope 合并初始清单与新增公共源码；默认 worker 快照也纳入新增源码，仍有隐私、路径和大小保护。 |
| A07 | 新 segment 单独计数；历史动作及 derive 子探针不消耗本段公布的动作额度。 |
| A08 | step 汇总按最近发生排序，另向 Monitor/Fix 提供真正按时间顺序的最近动作。 |
| A09 | 环境进展指纹只取能力/结果字段和稳定执行证据身份，不把 seconds 等遥测变化计作能力进展。 |
| A10 | RUN 耗时取实际 action/turn 开始时间，不再要求展示标签与步骤名相同；兼容数值与 ISO 时间。 |
| A11 | 损坏/缺失 job 记录显式显示 unreadable；无效、异 job 或损坏终态不释放执行权。 |
| A12 | 大源码支持有界行段、原文校验和全文件哈希；超大引用要求收窄，而非一概判为不安全。凭据、外链和越界仍拒绝。 |

### 设计缺口的处理范围

- **R01/R03**：新增 `build_the_environment(repair_proposal=...)`。Scheduler 可以提交 Init/Fix 发现的修复，不需要诊断 shell 获得实际环境写权限。提案必须绑定当前封存失败 ID；错误安装替换、能力探针修正和批量退役仍经过原独立审核。提案被拒明确记录 `repair_contract / native_operation_launched=false`，不冒充新安装失败。需要修订提案，而不是重复重放旧命令。
- 提案使用 `run_path` alias 时还必须回传 `execution_aliases_digest`，防止资源清单变化后同一别名指向另一构件；陈旧映射拒绝，不猜测替代文件。
- **R02**：状态分开显示已核验 interpreter 与已存在但未核验的 interpreter，发布 native inspection 状态；未通过的环境仍不能计分。
- **R04**：失败模型回合保存脱敏、大小有界的原始 stdout/stderr、尾部换行状态和稳定证据 ID；只读 worker 的证据也归入父 run 可访问的证据库。协议/传输失败会重建角色上下文，不继续盲用损坏会话；不采用半条 JSON 或没有最终结果的响应。
- **R07**：同步/异步 specialist 使用配置的实际模型窗口，不再固定 240 秒。同步 task 可申请 `timeout_seconds`，受模型配置与总墙钟限制。环境安装可申请 `operation_timeout_seconds`，不放宽 GPU/总预算。
- **R08**：`research_task(mode="inspect")` 强制只读，不清空已验证命令；编辑模式仍在交出控制权前失效旧验证，防止未知副作用污染计分。

### 验证与仍未完成的范围

最终全量修复回归：**1449 passed、2 skipped，190.38 秒**，含执行别名版本保护；`git diff --check` 通过。进度见 [EXECUTION_STATUS.md](../plan/EXECUTION_STATUS.md)。`docs/audits/runtime_20261001_reproductions.py` 已从“断言复现旧坏行为”转换为“断言修复后的行为”。新增 `test_runtime_audit_repairs.py` 覆盖跨层交接、拒绝、权限及活子进程反例。

未停止/重启原 RoboSyn 服务，未修改它的安装队列、实验源码或预算，也未由外部助手代填具体 RoboSyn 修复。因此在线已加载的旧 Python 代码不能视为已切换新版本。

**尚未完成，不能宣称全部设计缺口已闭环：**

1. R05：安装引导仍是同步原生操作，未变成可独立提交、运行中协商/取消的长期环境 job；候选 detached producer 的正式计分采用链也尚未完整接入。本次只提供可申请的安装时间窗口，不将其冒充长期作业系统。
2. R06：harness 仍只提候选，不自动测试后部署 live Python。保留这条安全边界；不能仅靠放大权限来“修复”。
3. RoboSyn N01/N02：实际入口路径和引擎版本/任务包还需要系统用原生回执解决。Resource 已在旧轨迹中找到候选发布 tag，但发现正确版本不是已经安装/加载成功，更不是正式得分。
4. 新版本真实用户 CLI 的环境→采集/训练→实际 checkpoint 加载→rollout→计分完整验收仍未完成。模型流式协议的原始故障原因也尚未证实，仅完善了证据与有界恢复，不能保证供应商/CLI 以后不再失败。
