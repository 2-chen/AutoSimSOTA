---
name: making-a-stage-run
description: 原生入口运行失败时按证据定位参数、环境、资源与消费者连接；修复后复验原操作，不靠猜通用配置名。
---

# 让原生阶段真正跑起来

候选补丁必须在实际消费者启动时仍然存在。若 staging 每次重建训练桥接器，它可能覆盖已审核学习改动；相同分数不能据此判定候选无效。检查 Main 的 candidate_source_integrity_warnings 与原生 receipt 的 candidate_source_integrity。执行器只在精确回到已审核 preimage 时重放同一补丁，其他变化拒绝；不替换候选、不修改评测。发现旧候选未保留改动时保留旧记录，留出未暴露前在总预算内追加局部轮次，提出/选择新的审核候选再验。不能拿“命令成功”和文件存在代替干预生效证明。

接入验证的工作量不是正式研究的默认值。首次基线前，由 Main 根据源码、实际成本和剩余总预算提交 `run_the_loop(training_settings={steps: ...}, evaluation_settings={episodes: ..., seed: ...})`。仅当已验证 scorer 使用 caller episodes 时设置它；原生配置控制评测则遵循其协议。执行器在训练前拒绝缺失/非正 episode 数，不自动沿用单 episode 探针，也不统一规定 50 或 100 次。episodes 必须为正整数，seed 为非负整数；指标、任务、成功阈值不能通过此接口调整。首次计分协议冻结后，候选与独立确认保持相同设置，不重写原 baseline。

调用方输入与入口默认值分属不同权限：Main 已选定的 checkpoint/data/task 等输入不会被入口 parameters 覆盖。若失败证据和 loader 源码证明选错输入（例如传了 bundle 目录，但 loader 只打开文件），入口修订 Agent 返回 `caller_input_handoff: {input: "checkpoint", reason: "源码与失败依据、所需类型"}`，立即交还 Main 从已验证 producer 重新选择，再复验。不能反复改 --agent 默认值、在 builder 固定本次权重，或把选择错误说成环境缺失/仓库不可运行。交接不自动修改输入，也不代表评测通过。

方法性故障分类，不限定只有三种错误。执行时保存完整日志、原命令/cwd/config、退出或超时原因、产物和稳定 evidence ID；摘要保留 traceback 的因果链和尾部，必要时按 ID 读正文。

- 参数解析失败：读当前 CLI/help/config schema；使用完整 flag，注意 argparse 缩写可能把名字错误表现成值错误。
- 原生仅接收 JSON/YAML 配置时，配置字段不是命令行 flag，也不一定支持 `key=value`。先沿源码确认工作量、数据和输出如何进入 consumer。需要动态传递时，在隔离 checkout 内创建可幂等生成的可执行配置桥接器：接收调用方数值工作量、数据、输出和稀疏候选设置，执行时在本次输出槽写入配置，再调用不变的原生 trainer。纯 argv 函数只调用桥接器，不写文件；仅改变配置文件名、固定 smoke JSON 或 `--debug` 不能证明实际工作量被绑定。换算 epoch/update 需有源码依据，最终以原生已完成进度和实际加载权重核验。
- import/动态库/渲染/设备失败：检查实际解释器、环境和权限，不靠换数据参数修复。
- 路径/配置错误：沿真实 consumer 找到 resolved 值、cwd、资源绑定与缓存；不要猜 data_root 或 output_dir。
- 训练/执行已开始后失败：检查数据样本、模型兼容性、数值、内存、作业信号与内部阶段，不把所有退出归为超时。

路径按 shell 与配置解析器各自的规则传递并小试。非 ASCII 路径不必然不可用；只有实际复现解析问题时才考虑安全别名。相对路径可以合法，只须绑定 cwd；不要把宿主绝对路径塞进不同命名空间。

HDF5 句柄在 spawn/pickle 路径下可能失败，num_workers=0 可作为诊断/临时修复；独立 worker 内延迟打开文件也可能是合适方案，不把零 worker 变成统一永久规则。隔离所需的环境变量不能为省事随意撤销。

缺数据先检查授权资源绑定和官方来源；不假设数据必在邻居目录。输出位置可能由代码生成，追实际构造函数；只能写 run-local 位置，不重定向回源仓库。输入 dict 的缺失、空值、0 和 False 要按 schema 区分。

每次原生操作的 output 是独立槽，可能尚不存在。staging 的 `AUTOSIM_STAGE_OUTPUT`、`AUTOSIM_STAGE_PYTHON`、`AUTOSIM_STAGE_REPO` 是本次调用实际路径，不是模板：用 shell 引号传递，例如原生文件 writer 要求父目录时显式执行 `mkdir -p "$AUTOSIM_STAGE_OUTPUT"`；纯 argv 则从 `i["output"]` 构造文件路径。若源码要求 output 本身不存在，不创建它。固定创建 checkout/eval 不能修复写入动态 caller output 的程序。入口验收与正式执行提供同样通道；是否创建及创建哪一层由 Agent 沿原生源码决定。

修复形成可审计差异，重试原失败阶段并比较新回执；后续还须验证实际消费者，文件产生不等于训练/评分成功。候选已回滚时通过新候选复验，不能把 Fix 的说明或探针当正式得分。

已封存的同阶段失败可用于 invocation_handoff，不需为得到“本轮失败”重放已知错误。命令在启动前被结构检查拒绝时，可使用该次提案的精确 response_sha256 作为 proposal_sha256 请求交接，不伪造 native evidence。交接只是诊断/修订请求；修后仍要执行和验收真实消费者，不能将旧失败或配置生成算作通过。
