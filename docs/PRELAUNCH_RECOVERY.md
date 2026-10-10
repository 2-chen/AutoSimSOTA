# 命令拒绝、故障证据与恢复合同

## 这次修复的真实故障

2026-10-05 的 robomimic Lift run 已完成基线与多轮训练评测，随后在候选 `dataloader-workers-4` 的训练命令组装阶段停止。原始失败回执 `c9c5b837730dfcd1` 与后续修复申请回执 `3e673296cda096b6` 保留不动。

根因链条如下：

1. 原生桥接入口使用 argparse `action='append'` 接受多次配置赋值。
2. 通用覆盖器把一个配置赋值从 `--set key=value` 中单独移走，留下空的 `--set`。
3. 重复参数校验还把原生合法的多次赋值误判为重复单值参数。
4. 异常发生在原生训练启动之前，没有生成 Fix 所需的 32 位证据 ID。
5. 候选的写前 checkpoint 已写为 `running/measurement_starting`，异常退出没有完成测量事务。恢复申请又因把 64 位回执摘要误当证据 ID 被拒绝，主 Agent 不再得到可执行研究动作。

这不是环境故障、GPU 缺失或总模型预算耗尽。

## 通用修复方式

### 配置赋值与参数语义

执行器只静态阅读实际入口文件中显式的 argparse append/extend 声明，不 import 或执行入口，不维护 benchmark 或 flag 名称白名单。入口的相对路径按原生 working directory 解析。

对已证明可以重复出现的参数：不同 `key=value` 配置项可以共存；同一键的冲突仍拒绝。已明确选择的 sparse override 可替换同键默认值，且必须保留参数与赋值的完整分组。`--option key=value` 和 `--option=key=value` 都保留原始表达方式。

普通单值参数仍拒绝冲突重复。输出路径等普通值即使含 `=`，也不能被当作配置项改写。不认识的动态 CLI/parser 不放宽校验，交由 Agent 根据源码修订调用合同。

### 启动前故障也封存证据

`operation_failure.py` 写入独立日志及 32 位证据 ID，沿用既有证据存储的内容校验，不能用回执 SHA 替代 ID、覆盖旧证据或引用外部路径。

训练/evaluate 的 argv 构造失败生成 `ran=false`、`command_started=false` 的 attempt receipt，并作为失败测量返回。候选事务正常完成失败记录、源码回退、Idea 结果与暂停 checkpoint；Scheduler/Fix 获得准确 evidence_id。失败不冒充成功，也不计为零分。

其他控制器异常也封存，但记录 `native_operation_status=not_inferred`：不能仅凭外层异常断言此前没有任何原生动作。预算耗尽与持久化故障不伪装成普通参数故障。

### 旧 running checkpoint 的恢复

`prelaunch_recovery.py` 是有界兼容路径，主 Agent 通过既有 `reconcile_interrupted_action` 显式选择。它仅接受：

- 身份与内容摘要验证通过的原始 run_the_loop 回执，准确指向本轮参数候选，且明确是 argv 构造拒绝；
- 尚无本轮 measurement、源码修改、采集结果或待解决原生后台任务；
- pending checkpoint 之后没有任何正式原生 attempt；
- 已有历史与 round 索引一致，原研究输入身份不变；
- 用原 pending settings 重新构造命令能够通过当前校验。

核对成功后保存旧 pending 与原回执引用，仅改为可选择候选的 paused 状态。不消耗该轮，不训练、不评测、不挑选新候选；原基线、历史、冻结协议与全部预算保持不变。真实原生复验须由主 Agent 再选择已审核候选执行。

已启动任务、不确定副作用或未收尾源码事务仍进入原有严格恢复路径，不能通过此入口重放。

## RUN.md 展示修正

文档区分基础设施/框架故障与环境故障，展示最近失败操作及回执。实验表从 controller history 补齐 hypothesis/改动与回退结论，样本数可引用已验证的 metric lineage，不把请求 episode 数冒充完成数。

源码候选未实际生效时保留原分数，但不计算优化差值、不进入可比提升曲线。没有原生录制入口的自动录制请求现在会留下媒体边界记录，而不是无声丢弃；不会补造视频或宣称已有 demo。

## 本轮范围

修复代码、回归测试与报告投影；没有重启 benchmark、扩大预算、改变留出协议或修改源仓库。旧 run 的独立 confirmation 未预声明，仍须保持这一边界，不能事后冒充预注册验收。

最终完整宿主回归：2123 passed、2 skipped（350.81 秒）。实际旧 run 的 argv 构造只读重放通过，旧 pending 被识别为可显式核对恢复。报告已通过生成器更新；状态、会话、测量、协议与预算原件摘要不变。这些验证不冒充新增真实训练或留出验收。
