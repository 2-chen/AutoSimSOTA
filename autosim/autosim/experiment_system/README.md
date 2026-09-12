# 可验证的自动采集—训练—评测运行层

这是独立的新系统版本，不修改 `autosim.research` 及现有 RoboSyn 冻结实验。
目前实现了运行层、RoboSyn 桥接和 RoboTwin 原生插件；是否完成真实任务由产物验收决定。

## 已实现的机制

- 显式观测/动作/时序契约；证据哈希绑定的能力状态，声明不等于验证。
- 作业输入冻结、依赖检查、独立尝试、输出校验后提交、成本记录。
- 控制器退出后 supervisor 继续持有作业与 GPU 锁；恢复时复用有效完成回执。
- 超时清理本作业启动的嵌套独立会话进程；不按进程名误杀其他任务。
- 未知执行状态必须审核；评测失败不自动重试，不把零成功率当作基础设施故障。
- 全量 parquet/视频时间轴与帧数检查、数值合法性、重复轨迹提示。
- 开发证据生成补采请求，不确定诊断回退全随机；修复记录要求版本和回归证据。
- ACT/DP 标准数据训练入口；RoboTwin 使用原生专家及原生 `eval_policy` 主循环。

## 运行

在 `autosim/` 目录：

```bash
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.cli bootstrap
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.cli inventory
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.cli cpu-acceptance
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.cli audit-existing
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.continuation --max-hours 48
../AutoSimSOTA/.venv/bin/python -m autosim.experiment_system.cli status
```

默认产物：`output/experiment_system_20260906`。首次 continuation 冻结新代码；修改后必须新建版本，不能边运行边修改冻结源码。
共享 GPU 锁 `/tmp/autosim-robosyn-gpu-0.lock` 与旧系统一致，旧任务运行中只能等待。

原生 RoboTwin 固定于提交 `c3ddfa8b97d5519efa828b075999bd0006778e5e`，位于产物目录 `native/RoboTwin`。
原工作区的既有改动不进入这个工作树。官方 cuRobo v0.7.8 固定提交
`d64c4b005459db10c5dd867d8b30a87d5bda9bdb` 在 `runtime_env` 中安装。
编译使用现有 CUDA 12.1、`8.9+PTX`、两路编译；RTX 5090 上的实际内核运行仍必须通过 GPU 验收。
该新环境只读复用现有 Python 包，并非依赖完全封闭的容器。cuRobo 打包元数据显示 0.0.0，因此必须同时记录源提交与二进制哈希，不能只依赖包版本字符串。

迁移任务预先限定为 `pick_dual_bottles`、`stack_blocks_two`、`open_laptop`，均使用 `demo_randomized`。
官方随机纹理固定 HF revision `785feb15aa4a4f532395ad2b1d2be5f28cb561ad`，压缩包约 11 GB，按原生资产结构安装；不关闭随机背景绕过缺失资产。

## 验收与边界

CPU 测试包含真实子进程故障，以及合成 HDF5 → LeRobot → 视频解码的转换测试。
这些不是机器人成功率，也不是系统优于其他论文的证据。

集成队列对 RoboSyn ClickBell 与三个 RoboTwin 任务执行有界采集、准入、ACT/DP 各 200 更新、原生 3 局评测。
这只验证链路，不证明策略有效；阶段失败写入独立记录。RoboTwin 保留官方的 expert-check 种子筛选，不能把原始连续种子列表误称为实际评测 seed bank。

RoboTwin 转换使用下一条记录的关节状态作动作标签，末帧没有下一步标签因此丢弃。
视频统一编码 25 FPS 只是数据索引时间，不声称原生 TOPP 控制为固定 25 Hz。
该训练适配不是官方 ACT 配方的逐项复现，正式比较必须把预处理差异单列。

尚未完成：完整动作单位/原生包装一致性实测、对恶意代码的评测隔离、整套依赖和资产闭包、
自主修复操作执行与回归后的生产发布、失败补采 ticket 的跨 benchmark 执行闭环、
系统机制消融、独立操作者接入成本实验、重复训练/研究及独立最终测试。
`feedback.py` 当前是带验证的请求与修复选择机制，不是“已经验证的自动视觉因果诊断”。

能力选择器支持官方数据回退，但当前新集成队列用于测通原生采集能力，不会将回退记为自动采集成功。
语义契约中未知的单位/频率明确标记未验证；配置盘点不计入完整流程通过数。
