# 已有环境受控复用

## 2026-10-03：预验收运行栈＋仓库增量适配

`autosim environments verify` 现在可以对已有环境做独立、短时、原生验收，通过后自动登记为只读底座。增量包写本次 overlay，当前源码、数据和资产分别连接；没有按 benchmark 名称选环境的规则。

| 类别 | 实际检查 | 本机验收 |
| --- | --- | --- |
| Python 基础 | 子进程、隔离目录读写；不是所有编译器都已验收 | Python 3.10 通过 |
| GPU 训练 | CUDA 运算、反向传播、参数更新、权重保存后重载并核对输出 | Torch 2.7.1+cu128／RTX 5090 通过 |
| MuJoCo 离屏 | 初始状态重置、真实物理步进、EGL 渲染并保存非空画面 | MuJoCo 3.8.1 通过 |
| SAPIEN 离屏 | 场景初始化、物理步进、Vulkan 渲染；2／3 系列探针分开 | SAPIEN 3.0.1 通过；2 系列未真实验收 |

四类能力由两套已有前缀承载，不是下载四套环境。三个 GPU 检查的租约占用合计 **3.755 秒**，无依赖安装、网络下载或模型调用。SAPIEN 底座的 Torch 版本没有因此获得 5090 训练认证。结果在 `autoresearch_cache/base_checks/`：`base_check.json`、`base_verification.json`、中文 `CHECK.md`、稳定 evidence ID；渲染另保存真实 `frame.png`。它们不是 benchmark demo、采集数据或计分结果。

### 使用方式

登记支持 venv／含 editable 的只读复用，不再只允许可克隆 conda；登记本身仍不等于通过：

```bash
autosim environments register --prefix /path/to/existing-env --label gpu-base
autosim environments list
autosim environments verify --prefix /path/to/existing-env \
  --profile torch-cuda --output /path/to/fresh-check \
  --wall-seconds 120 --gpu-seconds 60 --label gpu-base
```

profile 另有 `python-toolchain`、`mujoco-headless`、`sapien-headless`。输出须是新目录，GPU／渲染检查必须显式给 GPU 额度；CPU 基础检查不需 GPU。维护检查最多 600 秒墙钟、600 GPU 秒，实际走现有独占租约、占用复核和计量，不默认获得 24 小时额度、不停止外部占用进程。此维护不新建 benchmark run 清空旧用量，也不增加旧实验预算。

### 选择、记忆和安全边界

- Main／Init 短目录增加实际能力、版本变体、证据 ID 和失效原因，不传宿主前缀。排序只是推荐，Agent 仍按当前 Python／ABI／引擎约束自主选 ID、模式、绑定和理由；先用零安装消费者探针，失败才增量补缺。
- 核验解释器字节、依赖及钩子元数据、探针源码、封存日志／回执、画面及驱动／GPU UUID。登记变更刷新目录；缓存目录仍重查证据，变化不能沿用通过能力。Python CPU 能力不因 GPU 不可见而失效。
- GPU 查询失败不把报错文字当设备名称，不据此认定无物理 GPU。当前执行器无法核验 GPU 身份时不展示当前 GPU 能力，保留历史证书及原因；GPU 可访问的实际执行器重新核验，没有自动绕过权限。
- 成功 overlay 模板记录原底座身份和局部增量 pins，供后续同底座选择参考，不是盲目全装清单。旧 run overlay 不能登记为独立底座，否则过滤旧 `.pth` 会丢失借用依赖。
- 默认准备通过后仅发布小模板和已验证 wheel，不再自动复制大型 conda 环境。新 run 可由操作者显式使用 `--publish-environment-snapshots`；既有陌生前缀拒绝覆盖。
- RUN.md 用中文区分选择时的底座能力与当前仓库验收。CHECK.md 自动展示真实引擎帧；失败不发布成功画面或得分。

这是活的只读引用，不是完整依赖字节冻结镜像。已有引擎可用就避免重新下载；缺失能力留下失败证据，由 Agent 受审增量修复，当前不自动构建容器镜像或安装所有引擎。数据、资产和 checkpoint 不塞入底座，任务仍须原生消费者、reset／step／render 和策略 rollout 验收。

当前通用 MuJoCo／SAPIEN 2 底座探针对单物理 GPU 做验收，多 GPU 节点需补充受审的渲染器身份探针，不能假设 EGL／Vulkan 默认枚举与租约 GPU 一致。这是维护探针的边界，不是认定多 GPU 仿真仓库无法运行；SAPIEN 3 已按 CUDA 设备显式选择渲染器。渲染底座的设备发现目前需要兼容 Torch 的 CUDA 身份查询，不宣称支持所有无 Torch 的独立引擎环境。

最终回归：**1650 passed、2 skipped（262.44 秒）**；新增底座专项 **34 passed（12.01 秒）**。包含真实 CPU 隔离消费者、Agent 选底座后零安装推进、证据失效、默认不复制大环境及中文维护／RUN 报告。外层权限失败已在宿主复验，不靠放宽沙箱通过。

API 参考：[MuJoCo 官方文档](https://mujoco.readthedocs.io/en/3.2.4/programming/visualization.html)、[SAPIEN 2 相机文档](https://sapien.ucsd.edu/docs/2.2/tutorial/rendering/camera.html)、[SAPIEN 3 运行栈 API](https://sapien-sim.github.io/docs/api/sapien.pysapien.render.html)。

以下保留前轮历史快照；“仍缺 sklearn”等不代表最新环境准备状态。

## 为什么不能“直接拿旧环境训练”，也不应该全部重装

旧 Python 环境同时包含第三方依赖和可编辑源码引用。第三方依赖通常可以继续使用；可编辑引用可能指向已移动、删除的源码，也可能让候选实验改写源仓库。不能安全克隆一个 venv，不代表其中的 Torch、NumPy 等依赖不能复用。

`overlay` 模式把这两部分分开：只读借用已有依赖，在本次运行创建轻量 Python 前缀；新增和替换包由原生隔离执行器写入本次前缀。不是重新安装整个环境，也不是完整复制。

## Agent 如何选择

1. 主 Agent 从短目录比较 Python、ABI、依赖版本与 `overlay_reusable`，按消费者证据选择基础环境。没有按 benchmark 名称选环境的规则。
2. 准备阶段可提交 `build_the_environment`，参数为 `base_environment_id`、`base_environment_mode: "overlay"` 和 `environment_selection_reason`。不能与故障修复提案合并，也不能静默替换已验证阶段的环境。
3. Init 在规划中读取 `source_binding_options`，按需要选择 `source_binding_ids`。目录只给出 ID、模块名和来源类别，不要求模型猜测宿主路径。当前 checkout 中的同名可编辑模块自动优先绑定；外部依赖源码必须显式选择。
4. `commands` 可以为空，但原生消费者 `probes` 必须非空。先验证已有包、模块、配置和入口；缺什么再修什么。没有证据，不应反复重装兼容的大型依赖。
5. 失败保留回执和稳定证据 ID，沿用 Scheduler/Fix 的恢复事务和原操作复验。失败复用不会静默退回从零安装；Agent 必须明确重新选择。

`clone` 仍适用于可安全重定位的 conda 环境，`reconstruct` 适用于确有兼容性/隔离问题、无法复用的环境。三种模式都是提案，不能代替消费者验收。

## 创建中断、绑定修订和身份复验

- 创建底座前保存 `environment_bootstrap.json` 和构建游标；前缀内 `overlay_creation.json` 区分准备中、解释器已创建和连接已完成。中断后沿用同一计划/前缀，不重建已创建的解释器；没有匹配事务的既有目录拒绝覆盖。
- 显式切换的去重身份包含实际绑定 ID。同一底座补选缺失模块是新提案，底座和绑定都未变化才拒绝重复。主 Agent 可在 `build_the_environment` 提交 `source_binding_ids`，Init 必须遵守选择，不让模型猜宿主目录。
- “已经完成”校验原生上下文和借用元数据，不仅比较保存的身份字符串。基础环境变化后不能沿用旧通过记录。
- 复验保持基础解释器和 Python ABI，不改源仓库，不重建环境，不丢弃本次局部安装。核验原选定模块的相同来源，归档旧 manifest，建立新的不可变依赖视图，再执行消费者探针；绑定来源失效需要 Agent 明确修订。有原生长期作业运行时不刷新依赖。复验失败的当前记录保持未通过。
- `native_context.json` 绑定当前 overlay 定义及生成启动文件的摘要；绑定/视图变化产生新执行身份，启动文件被改写时拒绝旧上下文。完成/失败关闭 bootstrap，避免误恢复已结束计划。
- 通用扫描只推荐 ABI 匹配的 `*/lib/pythonX.Y/site-packages` 下已有模块。Agent 选 ID 后才接入；排除启动定制模块、越界符号链接和任意旧 `.pth`，不按 benchmark 或厂商名字添加安装规则。

真实 CPU 只读验证中，显式接入已有嵌套依赖后 `pinocchio`、`eigenpy`、`hppfcl`、`coal` 均能实际导入，无重新安装。证据 `e33585425e8b47ba983e8e79fb862359` 位于 `/tmp/autosim-overlay-check-f6eqlroh/evidence/`；不代替 RoboSyn 完整导入、GPU 物理/渲染或实验评分验收。

最终补查覆盖上下文正常、丢失和损坏三种情况。受控准备不信任坏上下文的路径映射，保存错误证据后重新发布；普通执行器的默认校验不放宽。真实完整 `import robosynchallenge` 复核仍在 Open3D 导入链缺 `sklearn`（证据 `4a0d9a791fa7404fa31f8e987bb8b291`），基础元数据未变化。

本轮最终代码回归：**1551 passed、2 skipped（240.75 秒）**；相关定向集合 **204 passed（43.23 秒）**，覆盖中断续跑、绑定修订、身份变化、局部安装保留、复验失败撤销及运行中长期作业保护。修改模块编译和 diff 空白检查通过。

## 隔离与证据

- 创建 venv 时使用 `-I -S --copies --without-pip`，不执行被发现环境的启动钩子，也不依赖一串旧解释器符号链接。
- 静态解析 editable finder 的字面量 `MAPPING`，不执行其 Python 代码；所有继承 `.pth`、`.egg-link`、editable 辅助模块和启动定制文件均不继承。
- 非 editable 的本地源码安装是已安装副本，不因其构建来源是 `file:` 就整体排除。主 venv 优先于继承环境；同名分发的旧版本元数据不混入新版本视图。
- 过滤依赖视图是只读来源的符号链接；运行前缀中的本地安装优先于借用依赖。原生执行器、CPU 诊断和 demo 挂载共同保护基础环境与外部依赖源码。
- 借用视图放在 `sys.prefix` 外的 run-owned 目录并只读挂载，使 pip 识别其为外部安装，不卸载基础包。本地 wheel 集成测试已证明可正常安装新版本、由本次前缀覆盖旧包，基础文件/元数据不变。
- `overlay.json` 记录基础身份、依赖元数据/钩子摘要、选定源码绑定和跳过的钩子。`native_context.json` 记录执行器认可的只读挂载；跨进程排序稳定，变化后拒绝沿用旧上下文。
- RUN.md 中文展示当前复用方式、已连接模块和仍待验收的探针，不把依赖连接写成“完整复制成功”。

## 验收结果与边界（2026-10-02）

真实 CPU 测试使用 `/tmp/autosim-overlay-check-f6eqlroh` 的独立源码快照。资源及策略目录仅为此诊断只读连接；策略目录没有作为可修改候选源码验收。未运行训练、采集、reset/step、评测或模型调用。

| 检查 | 结果 | 原生证据 |
| --- | --- | --- |
| 无重新安装复用 Torch、NumPy、LeRobot，执行 CPU 张量运算，并验证 pip 外部包身份 | 通过：Torch 2.7.1+cu128、NumPy 1.26.4，张量和为 3；借用 Torch 的 local=False | `6e856fdbfa5b4ed1b9a95a8937b956fc` |
| 当前 RoboSyn 模块身份、EmbodiChain 与任务模块来源连接 | 顶层模块定位通过：RoboSyn 在临时 checkout，依赖源码来自显式绑定；不是完整任务导入通过 | 同上 |
| 完整 `import robosynchallenge` | 未通过：Open3D 导入链缺 `sklearn`；尚不能宣称仿真已就绪 | `450066aec4ea4804b4f9233e77b8460e` |

回执与日志位于上述临时目录的 `evidence/`、`native_diagnostics/`。合成合同测试另外验证旧钩子不执行、未知/重复 ID 拒绝、基础元数据变化拒绝、本地包覆盖、多进程上下文、先探针不安装，以及真实只读沙箱拒绝写回基础包。

回归测试：全量 **1536 passed、2 skipped（229.74 秒）**；最后启动定制包/字节码过滤补查后的相关定向集合 **83 passed（10.29 秒）**。本地 pip 覆盖测试通过原生隔离执行器运行，无网络下载。

这是**活的只读依赖引用，不是不可变环境快照**。对包及外部源码内容的完整字节冻结、非标准可执行 `.pth` 插件的安全迁移、GPU 渲染/物理和原生训练/评测仍需额外验收。忽略的钩子若承载必要功能，应根据具体故障新增受审连接或安装修复，不能重新执行任意旧钩子。基础包实际缺失或版本冲突仍需增量修复，不能靠 stub 冒充依赖。此次没有重启旧真实 run、增加预算或清除历史费用。
