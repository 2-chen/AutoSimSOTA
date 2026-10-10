# 0/100 基线调查与研究交接修复

日期：2026-10-09。范围：旧 RoboSyn run、最新 FDAS pilot 的留存证据，以及当前通用控制器。此次未重启真实实验，未更改源仓库、旧测量、原基线或历史预算。

## 后续执行层修复（2026-10-09）

先前的来源说明和建议还不够：空 data binding 被当作“无需核验”、复用回执没有 ran 导致训练语义漏判、参考评测与本地搜索共用协议、recent_actions 丢失 retryable 等字段，都是系统自身问题，已修改实现。

现有数据未经版本消费证明时明确为 unknown；不改写旧测量，也不把这次软件修复当实际 loader 核验。Main 收到基线 subject SHA、封存原生 ID 和具体诊断任务，需要完成 training/data/policy/evaluation 的引用式诊断才能继续候选轮。Agent 可选择调查或继续限定的本地优化；记录不能给官方复现或数据消费认证。它不是固定训练配方、自动排定采数路线或跨仓库固定阈值。

官方参考评测使用已审核的 episode 数和独立不可变协议，回执及只读核验器均绑定此身份；不同参考 episode 数不再修改本地协议。已有本地 evaluator/metric/task/留出协议不能被参考路径绕过。执行前拒绝也有终态记录，避免只留下 evaluating 指针。

只读旧 run 已检验：自定义指标 click_bell_success_rate 的声明下界、100 个 episode 原值和缺少 ran 的复用训练回执，均能被新诊断机制识别。实际重新评测发布 DP、真实数据消费复验和新增数据对照仍待在线实验，不能声称已提高旧成功率。

参考异常捕获后会封存稳定 operation_failure ID。自动 Fix 路由也从仅依赖旧 step/outcome 名单，扩展为读取明确 Fix 所有权、失败类别和经字节核验的封存 ID；新研究操作默认只读诊断，由 Main 经审核接口安排修复和复验，不能清未知所有权或未经核查重放仿真。没有新执行回执的启动前故障才可登记终态。独立参考复现成功不把另一份本地 baseline 的官方标签改为 true。

验证：全量收集快照 2811 passed、2 skipped；之后收尾改动由当前代码扩大专项 326 passed 覆盖。没有真实模型/仿真验证；详见执行进度，不能把单元/CPU 合同通过当 AutoSOTA 成果。

## 结论先说

旧 RoboSyn 的 **0/100 是一次真实执行的本地短训测量，不是官方发布基线的复现**。最明确的问题是基线定位和训练配方：用从头训练、很少更新、不同时间窗口的策略作起点，然后容易把“评分链核验通过”解释成“官方复现成功”。后者不成立。

这不等于已证明评测器无问题。我们排除了“未评测而填零”“加载了别的权重”“程序崩溃后被读成零”等解释；仍未排除数据消费、观测/动作对齐、依赖版本和仿真控制等问题。不要用训练不足这一解释跳过这些核查，也不要将真实的低分删掉。

## 1. 本地与发布模型的差异

调查对象：`autoresearch_runs/robosyn_installrecovery_20261001_v1`，Click bell / random，Diffusion Policy。

| 项目 | 本地 baseline | 官方发布的 DP 配置/结果 | 判断 |
| --- | --- | --- | --- |
| 策略来源 | 从头训练产物，非发布权重 | 固定发布 revision 的策略 bundle | 不是同一策略身份 |
| 训练配置中的更新量 | 1024 | 100000 | 本地预算仅为发布配置的 1.024% |
| batch size | 8 | 64 | 单次更新数据暴露也不同 |
| horizon | 16 | 32 | 时间建模不一致 |
| action chunk | 8 | 32 | 控制时序不一致 |
| 训练 seed | 0 | 1000 | 不属于相同训练配方 |
| 评测 episode | 100 | 100 | 样本量一致不代表其他条件一致 |
| 成功数量 | 0 | 51 | 官方参考不是之前拿来比较的 ACT 37 |

发布参数来自 [固定 revision 的 train_config.json](https://huggingface.co/RoboSynChallenge/DP_sim_click_bell/blob/fb8c9551e0fade7c4888a926ab577b85f0684da1/train_config.json)，该文件是发布训练配置，不是我们独立观测到的发布训练计数。[官方仿真评测说明](https://github.com/EDEM-AI/RoboSynChallenge/blob/main/evaluation_results/README.md) 给出 Click bell DP 51/100、ACT 37/100，并指定 DP 使用 10 次去噪及对应代码 commit。仿真参考不等于真实世界榜单 SOTA。

若仅按配置计算，1024×8 与 100000×64 的名义样本处理量相差约 781 倍；这不是实际消费量证据，也不能代替公平的等计算比较。本地同样缺少训练充分性/收敛性判断。

观察、动作形状和 normalization mapping 的配置层面检查：双方都有 14 维 state/action、三路相机，VISUAL 使用 MEAN_STD，STATE/ACTION 使用 MIN_MAX。暂未发现这些配置字段的直接矛盾；相机内容/坐标系、实际归一化统计和动作控制链仍需运行行为证据。

## 2. 0 分到底来自什么

关键证据：

- 训练 attempt：`59522171ac524b9d896023b45948db84`。封存命令明确指定 1024 步，训练约 336.6 秒；实际产物选择记录关联到完成计数 1024 的 checkpoint 容器。
- 旧 measurement.settings 为空，但派生命令参数留下 1024 步，缺少独立正式基线配方选择的充分记录。当前已有的 formal-work 校验要求 Agent 明确给出学习预算，本轮进一步通过基线来源投影和指引区分小试、短训与发布参考；不在执行器中写死新的最低训练步数。
- 本地冻结配置：`research/derived/experiments/baseline/59522171ac524b9d896023b45948db84/policy/train_config.json`。
- 评测 attempt：`da19301fae8c4cc3b40749f299a49d4c`。真实加载的冻结策略身份为 `9896f0933b1ddf7e6135809a09b3440df5369b6035ad81a57d61de1978ccaa93`。
- 封存日志有 100 个完成 episode、同一策略身份的指标事件及结果文件；100 次均失败，均用尽 361 action steps。这是任务超时，不是进程崩溃。
- 本轮重新运行只读 receipt verifier：baseline 0.00、round_18 0.28、round_20_handoff_correction 0.27 均为 `consistent`，包括结果字节、实际加载、rollout/episode 身份及指标重读。

核验边界：上述一致性检查没有独立重跑物理仿真，也不能认证 instrumentation 的全部语义、官方复现条件或统计独立性。

后续 8192/16384 步的记录出现 2/100、28/100、27/100，说明系统确实得到过不同的策略行为和成功 episode，并非所有任务永远被读成零。但期间还有其他干预，不能把提升全部归因于训练步数；28/100 也不能据此称为超过官方 DP。

## 3. 数据路径不是丢失，但消费证据不足

旧运行的 `existing_datasets` 是隔离运行时的只读资源挂载，而不是完整复制的数据目录。停机后本地占位目录为空属预期行为，不能用普通文件查看判定数据丢失。

`workspace_snapshot.json` 明确绑定到源仓库 `lerobot_dataset`。只读检查 backing resource 的 `meta/info.json` 显示 1000 episodes、74000 frames、25 FPS、train split 0:1000；训练配置未显式选择 episode 子集，日志显示了 legacy feature alias 到原生 observation 特征的处理。

但这些是目录元信息、配置和调用证据，不是旧训练实际 batch 消费的见证。旧 measurement 没有可追溯的 `data_consumption`，无法严谨断言当时加载全量、混合比例正确或每一路图像与 action 对齐。

另一个 run 的 **1 episode / 74 frames** 是最新 pilot 的小试数据，不是上述旧基线的数据。不同 run、不同策略族和不同数据不能拼成一个“统一基线”。

## 4. 已实施的通用修复

### 4.1 训练产物只有一个交接入口

`verified_train_artifacts.producers` 同时投影命令验证产物与长 native job 产物。每个 producer 保留 ID、来源、output、封存 native evidence、学习进度及有界目录预览。

长作业必须有零退出 native 证据、稳定的 worker source identity、准确的 job→receipt 关联、fresh artifact 证据，以及独立进度审核。未审核的完成作业列入 pending，告诉 Main 调用 `inspect_native_job`；不再因为不属于 `verification_outputs` 就忽略它。

具体 checkpoint 仍由 Agent 根据保存/加载源码选择。执行器只要求路径来自唯一合法 producer，并冻结、复验所选字节；不自动选 newest、largest 或固定目录名。

### 4.2 消除协议与产物检查的循环依赖

`inspect_native_job` 不再初始化正式研究控制器，也不要求正式 comparison protocol 才能审核学习计数。读状态只重放既有审核，不偷偷调用模型。

正式初始 job adoption 则在核对所有权、准确设置和源身份后，冻结显式选定的 evaluator/split/metric，再验证训练回执并计分。已有协议不得覆盖；设置不一致不得偷偷改写为“可兼容”。候选 adoption、实际加载、rollout 和指标核验继续保留。

### 4.3 三类审核共用准确引用合同

baseline reference、data version、confirmation reservation 共用 source citation 映射。Supervisor 引用模型实际看到的 `source_id + quote`，包括脱敏占位符；本地映射到唯一原始引用并重新核验源码，不要求模型猜出私有路径，也不使用模糊匹配。

原始 quote 变化、伪造 ID、错误映射或审核期间字节变化仍拒绝。

### 4.4 临时阻塞不等于永久失败

已有 native job 占有 run 时，confirmation reservation 返回 `resource_busy / retryable=true / retry_when`；search 已开始时才返回不可回溯的 `protocol_frozen`。这些字段保存在行动记录中，不在持久化时丢掉。

Main 的 `stop_scope=framework` 增加独立只读 Monitor 复核。永久框架边界需要当前封存故障和准确引用；缺证据、参数错误、临时所有权冲突不能仅靠一句 Main 判断确认。未证明时给 Main 一次纠正机会，重复未解决则明确暂停，不反复花费模型预算，也不宣称已经自主修复。用户暂停和硬预算不被此逻辑绕过。

### 4.5 把局部字段错误提前交给 Main

reference、version、confirmation 引用等嵌套结构提前暴露 schema，并给出准确字段路径；发布参考按 kind 条件要求 checkpoint、revision、metric/tolerance 等字段。32 位 evidence/job ID 与 16 位 action ID、64 位摘要区分，避免错误 ID 到达 Fix 才被拒绝。

这是参数合同，不是科学有效性审查。完整 source、数据 lineage、公开 HTTPS 来源和原生语义仍由原有域校验及 Supervisor 判断。asset/recipe 等复杂科学合同不被通用 JSON shape 替代。

### 4.6 Main 看里程碑，不只看最近一句错误

新增短 `research_milestones` 事实投影：本地基线的真实来源、训练 argv、消费见证状态、完成训练交接、待审核 producer 和可重试阻塞。baseline.ok 明确不代表官方复现。

Main 指引要求在关键数据里程碑咨询 Data，再推进采集→质量/profile→不可变数据版本→真实 batch 消费→候选计分；若先走其他路径，解释更高价值瓶颈。不为任何仓库固定数据量、算法、训练配方或采集入口。

还修复了一个审计缺口：已验证的数据消费结果此前在成功 measurement 中未保存；现在新测量保留 `data_consumption`。历史缺证据记录不会被自动补造。

### 4.7 减少不产生实验证据的 token 开销

Recorder 将重复参数拒绝按语义合并，行动 ID/时间戳/错误措辞改变不单独购买叙述；精确事实仍本地实时更新，新真实故障、指标或媒体事件继续触发。

Scheduler 请求按当前可执行操作投影相应过程条款，保留战略、安全和 JSON 输出合同。仅早期 `read/build/stop` 的测试场景，提示正文字符由 63864 降到 50620，减少约 20.7%；所有操作可用时保持完整正文。字符减少不是实测 token/费用减少，更不保证实际缓存命中率。未将整套技能正文重新塞回 prompt。

## 5. 下一次真实验收顺序

1. **先验证能力与官方参考，不先扩大短训。** 在同一任务/策略族/固定发布 revision 下评测完整发布权重，核对固定代码版本、相机、动作、reset、控制频率和推理参数。保持与本地候选收益分离；已有审核拒绝要以准确封存证据纠正。
2. **若参考分数明显不符，先诊断加载到行为链。** 用独立开发 episode 的可视化、动作范围、观测/归一化统计和状态演进定位；先修系统连接，不用额外数据掩盖错误，也不改成功判据。
3. **本地训练显式规划充分性。** Agent/Resource 根据原生学习单位、真实吞吐与剩余总预算选配方。小试只能证明能训练；主基线需交代数据规模、样本消费、实际计数和学习曲线。不写死 100000 步，也不把任何仓库的官方配方强加给其他仓库。
4. **接通 Data follow-through。** 合法专家小试、原生转换、quality/profile、数据版本、真实 batch 消费、候选计分。小试成功后按有效数据增量和收益决定扩量，不无限重复转换同一条 episode。
5. **最后做 FDAS 对照。** 失败引导采集/筛选与等量随机、同计算不增数据对照采用同策略与协议。训练不足导致的机械性增分不是数据方法的独立贡献；未复现官方参考也不妨碍严谨的本地对照，但必须说明范围。

原每仓库总预算、未知用量预留及所有历史消耗保留。已有旧基线不替换，真实研究启动/恢复另记在线验收，不将合同测试结果冒充 SOTA。

## 6. 验证与边界

新增回归覆盖：长作业无 comparison protocol 的独立审核、两种 producer 的显式选择、计数/源码/日志/审核/receipt/job 身份篡改负例、脱敏引用往返、临时忙碌后仍不可回溯的协议边界、停止复核与单次纠正、嵌套字段错误、语义叙述合并及提示投影。

CPU 集成测试真实执行 toy trainer，复用其原产物建立正式协议并进入 evaluator 获得测量，禁止第二次训练。这个测试不是 RoboSyn rollout，也没有调用付费模型。宿主全量收集快照为 **2792 passed、2 skipped（455.72秒）**；随后补充的两条引用交接测试由最终专项 **208 passed（11.55秒）**覆盖。完整边界记入 `plan/EXECUTION_STATUS.md`。

此次没有新增真实采集、训练、官方权重评测或性能分数。现有框架修复改善了 Agent 得到正确事实和交接权限的条件，不能宣称已证明真实 LLM 会完成 FDAS 或大多数仿真仓库都会成功。
