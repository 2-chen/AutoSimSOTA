# 研究产物的生命周期与磁盘压力恢复

## 本轮交付

保存参数的决策由 Main/Fix 根据原生源码作出；独立 Supervisor 审核保存、选择、学习和恢复语义。执行器负责路径、字节身份、引用、活动作业、容量准入和删除事务。没有添加 RoboSyn/LIBERO/RoboTwin 名称分支，也没有统一写死 checkpoint 频率。

| 接口 | 输入/结果 | 硬边界 |
|---|---|---|
| `inspect_storage` | 文件系统余量、分项占用、生产者样本、登记和事务 | 只读；总览最近 8 个生产者，每个最多 24 个文件；登记用 registry_offset/registry_limit 分页，展示文件名不代替完整清单 |
| `inspect_storage(evidence_id, offset, limit)` | 按稳定证据 ID 分页读取完整生产者目录 | 每页 1–64，扫描上限 32768，超限报不完整 |
| `configure_storage_controls` | 原生正整数保存/日志参数、源码引用、可选容量估算 | 仅 TRAIN 工程参数；独立审核，保存/选择/学习/评测语义不变，源文件变化后重新审核 |
| `register_storage_artifact` | exact refs、sealed evidence_id、角色、理由、源码 | 1–256 个精确 payload 文件；独立分类；禁止源代码、数据生产者、元数据和框架归档 |
| `plan_storage_gc` | 1–64 个登记 ID、理由、源码 | 独立审核；中间产物必须有同一生产者的保留终点或恢复组 |
| `commit_storage_gc` | exact plan_id | 运行级授权、闲置、引用/指纹/源码复核；持久化删除日志；不接受通配符或任意宿主路径 |
| `reconcile_storage_gc` | 中断的 plan_id | 只核对已删和剩余文件，不重放删除；剩余内容需新登记和审核 |

`source_evidence` 是 `[{path, quote}]`，quote 必须是精确当前原生源码。path 默认 checkout-relative；借用依赖可用 `dependency:<索引>:<相对路径>`，索引仅来自已校验的 `native_context.json`。不存在的源码、任意宿主绝对路径或未证明保存规则的 CLI 参数不算依据。

## 授权与保留范围

默认不开自动删除。新启动的操作者可加 `--storage-auto-gc`；不携带该参数的续跑保留既有策略，不默默增加权限。也可独立启停运行级策略：

```bash
PYTHONPATH=autosim .venv/bin/python -m autosim.research.storage_lifecycle /absolute/run --auto-gc enable
PYTHONPATH=autosim .venv/bin/python -m autosim.research.storage_lifecycle /absolute/run --auto-gc disable
```

此授权不允许任意 shell 清理。GC 仅接受 run 内 `research/` 或 `verification_outputs/` 下明确属于已知原生训练/评测/媒体生产者的 payload。terminal/resume 登记不可自动降级；测量、baseline/best、注册数据、已接受验证交接、媒体记录等引用自动保护。硬链接的已测量文件别名也保护。

源仓库、外部挂载、其他 run、数据采集产物、源码、日志/配置/计数、证据和框架归档均不属于自动 GC。环境和共享缓存暂不自动清理：它们可能是只读依赖或有跨 run 消费者，现有所有权证据不足。Agent 可调查、建议受控维护或新存储位置，但不能自行扩展删除范围。

原生作业准入和 GC 共用运行级锁，同步正式执行与原生命令验收均持有持久化 stage lease；长期作业及未解决的执行记录也阻止清理。异常/未知进程不按 PID 猜测终止。`committing` 事务阻止新作业，先核对再恢复。

删除不可恢复；日志、元数据和逐文件清单保留。部分完成的清理也保留已删 refs，使策略发现不再把退休的 metadata-only 目录当作可评测 checkpoint。

## 容量准入与运行观测

作业可通过 `resources.storage_required_bytes` 提交预计新增体积；源审阅的存储控制可登记 `checkpoint_bytes`（一次保存）和 `projected_bytes`（本轮新增）。正式执行采用各估算的最大值，不能用小估算覆盖已有大估算，并为最低安全余量留空间。

`/tmp/autosim-storage-reservations-<uid>/` 是私有协作租约目录。同一文件系统上的其他活跃预留从实际空闲中扣除；完整预留保留到作业结束，因此可能保守地重复考虑已写出的体积。它不是磁盘预分配/全机配额，其他宿主程序和估算误差仍可造成空间不足。未知租约不自动释放；不能为了启动任务重置账本。

执行器每轮只读取磁盘余量，不调用模型或全盘扫描；持久化记录合并为首次、状态变化或约每 30 秒一次。空间低于下一次保存的安全估计（1.25 倍）或 64 MiB 紧急线时，请求停止当前写入。正式作业与验收试跑均记录 `storage_pressure`，关联原生 attempt 和稳定证据 ID。写观测文件遇到 ENOSPC 时，仍把停止信号和内存中的压力证据交给执行器。

停止不保证新保存成功，未保存更新可能丢失。Main/Fix 需核验现有完整恢复状态，决定复用、续训、修订保存参数、申请精确清理或新存储；修后复验原操作，不因 ENOSPC 重训或换算法。

## 冻结副本和报告

正式普通/图执行路径的权重冻结使用 `research/<id>/artifact_store/<fingerprint>/`，同一源内容重复冻结复用同一已核验对象。首次复制尝试独立 COW，失败回退普通副本，绝不硬链接可变原生权重。对象被改写时拒绝复用；拷贝完成、manifest 未写的中断槽仅在重新核验后补齐 manifest，不覆盖内容。导出仍带真实独立文件，不变成只适用于本机的指针。

RUN.md / RUN.html 自动增加中文“存储与中间产物”面板：最近空闲、Agent 估算、上次分类占用、授权状态、最近事务登记体积及未知清理提醒。心跳更新无需额外模型调用；估算与观测分开显示，登记体积不冒充文件系统实际释放空间。

## 验收边界

本轮采用临时 CPU fixture 和脚本 Reviewer，验证保护引用、字节/源码变化、符号链接、完整目录分页、并发预留、事务中断、流式取消、证据封存、COW/内容复用、短目录与主动选读。它们不证明 DeepSeek 在真实仓库已经选择合适的保存参数，也不证明仿真指标提高。旧 RoboSyn 保持停止，未开启自动回收、未修改保存频率、未新增真实清理。
