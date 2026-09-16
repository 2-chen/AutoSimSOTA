# 外部兼容补丁快照

`robosyn.patch`、`robotwin.patch`与`xpolicylab.patch`分别是整理时对应外部Git工作树的已跟踪文件差异；对应`.base.txt`登记官方/上游HEAD。三者必须应用于各自仓库，不能跨仓库应用。

RoboSyn补丁基于官方提交`6f555fc3c9f514bb8fea6c7de036e68b9390fa5b`，归档以下5个适配文件：

- `scripts/run_env.py`：有界定向/纠正采集、场景profile和采集清单；
- `policy/act/scripts/train.py`：观测契约、数据组合/抽样、增强和训练回执；
- `robosynchallenge/managers/datasets.py`：数据访问适配；
- `policy/act/deploy_policy.py`：部署参数适配；
- `scripts/eval_policy.py`：有界评测、初始化和逐局证据。

应用前在匹配版本的独立副本执行`git apply --check /path/to/patch`，审查通过再应用。不要对已修改的当前工作树重复应用。

`robosyn_checkpoint_compat.py`是**未跟踪**兼容文件的归档（不在`robosyn.patch`中，因为它从未被提交到上游仓库）。它必须部署为benchmark仓库内的`policy/act/checkpoint_compat.py`：

- `RoboSynChallenge/policy/act/checkpoint_compat.py`（训练/契约校验路径）
- `RoboSynChallenge_eval_clean/policy/act/checkpoint_compat.py`（评测路径）

三份拷贝内容必须一致，sha256为`d51e642ed93f1c627cc08913e113c07d9df0f86f7ca69add24f960cc0eda87d5`。`autosim/research/repository_harness.py`与`repository_autoresearch.py`会把它计入checkpoint契约摘要，缺失时`harness_artifacts`路径会在原生评测前抛`RuntimeError`。该文件针对`lerobot==0.3.3`（该版本的`ACTPolicy`已自带inline normalization，故`_restore_inline_normalization`为`False`，行为与内置实现逐位一致）；若运行时缺少inline normalization，它会从checkpoint内嵌buffer恢复，并拒绝`strict=False`加载。

本快照不包含仿真资产、权重或完整上游仓库，也不是完整安装器。RoboSyn上述已跟踪运行插桩已归档；RoboTwin的未跟踪环境配置仍需独立审计。上游许可证保持适用；发布前核查补丁来源与许可。
