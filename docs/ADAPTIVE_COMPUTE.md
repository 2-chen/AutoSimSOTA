# 自适应算力执行

新建 research CLI 运行默认使用自动资源发现和统一调度。算力规划与研究提案分别配置；实际设备由当前容器的授权范围、UUID/CUDA 对应关系、资源余量和能力探针共同决定。

```bash
# 在主项目 AutoSimSOTA/ 下使用；这是后续使用入口，本次未启动完整研究。
PYTHONPATH=autosim .venv/bin/autosim research RoboSynChallenge \
  --task click_bell --gpus auto --compute-controller api \
  --compute-env-file .env --compute-codegen validated \
  --hours 24 --gpu-hours 96

# 静态预览：不运行 GPU/API 探针。
PYTHONPATH=autosim .venv/bin/autosim research RoboSynChallenge \
  --task click_bell --gpus auto --dry-run

# 显式保留旧单卡路径。
PYTHONPATH=autosim .venv/bin/autosim research RoboSynChallenge --gpu 0
```

`--compute-controller heuristic` 为默认算力控制器；`api` 从指定私有 `.env` 读取现有服务配置。支持 `DEEPSEEK_`、`AUTOSIM_LLM_`、`OPENAI_` 前缀的 `API_KEY`、`BASE_URL`、`MODEL`。密钥不写进 GPU 工作进程参数或验证报告。

`--compute-codegen validated` 要求自动执行模式与算力 API controller。数据审计入口先对当前数据测量：小工作量或剩余时间不足时直接保留原函数。工作量足够时，API 仅接收 `_padding_audit` 函数，候选经过语法白名单、受限子进程、等价性、重复 A/B 和当前工作收益检查后，才在该次审计上下文中使用。上下文退出后恢复原函数。该入口目前只支持这一数值函数，不是任意仓库修改器。

资源合同包含 CPU、RAM、共享内存、临时盘、I/O 槽位、每卡显存和独立的 CUDA/训练/物理/渲染能力。GPU 叶任务取得 UUID 租约与预算后执行；数据审计为零卡任务。采集和评测固定逻辑块及输入摘要，设备放置另存执行记录；恢复时不会因卡数变化重新生成 seed bank 或重复已完成工作。

耗时画像按代码、环境、设备类别、工作量形状及执行资源分组；种子和 checkpoint 内容仍属于恢复身份，不把每个新候选误当成全新的耗时类别。至少三份可比样本后才标为实测画像。

原生引擎的可写缓存按节点、GPU UUID 和工作空间隔离，并在独占租约下复用；每个工作的生成 URDF、临时文件单独保存。Torch 权重和只读仿真资产共享，SCO 容器使用预缓存依赖及离线数据模式。缓存隔离能避免并发写同一目录，但原生加载崩溃是否解决仍以实机回执为准。

恢复使用原 `--output-root` 和 `--run-id`，同时保持科学参数与输入。历史单卡协议继续使用原路径；转用新执行协议需新 run-id。跨节点遇到未结束 claim 时，必须先有原分配终止证据才能继续，系统会拒绝不确定的重放。

当前多卡训练能力是多个独立单卡 ACT 任务。ACT DDP launcher 受完整训练合同摘要门禁控制；线性模型 NCCL/checkpoint 探针通过也不能代替 ACT 采样、优化器及恢复语义的认证。

## 本次 SCO 组件验证

```bash
# 默认仅准备冻结源码、校验缓存依赖并显示命令。
.venv/bin/python tools/sco_compute_validation.py \
  --validation-id matrix_example --gpus 4 --seconds 3300

# 用户授权后提交独立规格；依次取 1、2、4、8。
.venv/bin/python tools/sco_compute_validation.py \
  --validation-id matrix_example --gpus 4 --seconds 3300 --submit

# 控制机从实际分配快照调用 API；计算容器不需要联网。
.venv/bin/python tools/compute_control_response.py \
  --request ../compute_validation/matrix_example/4gpu/compute_control_request.json \
  --env-file .env

# 只读取回执，生成矩阵与成本报告。
.venv/bin/python tools/summarize_compute_validation.py
```

提交器冻结主项目 `autosim/` 与 `tools/`，使用已缓存的镜像、ACT checkpoint、数据及仿真资产。任务有独立输出与最长执行时间，总矩阵采用 15 GPU-h 预占上限；已结束任务按应用账本及平台终态分别对账。成员配额从实际提交错误确认，429 后不会自动循环创建任务。

验证入口不调用完整 AutoResearch。它检查实际卡数、固定 CUDA 工作队列、原生评测并发、固定 16 episode 合并、采集、数据审计、ACT 短训练、恢复与进程回收，以及真实 API 参数应用。失败组件保留日志与失败状态。四种规格必须有各自独立分配；一个八卡容器的掩码不能替代该矩阵。

当前执行状态与已知限制以外层 `plan/adaptive_compute_execution_20260914.md` 和 `compute_validation/report_20260914/compute_report.md` 为准。第二 GPU 类型实机、生产 SCO backend、多节点和完整研究闭环仍属于后续范围。
