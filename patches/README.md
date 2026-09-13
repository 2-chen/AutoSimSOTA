# 外部兼容补丁快照

`robosyn.patch`、`robotwin.patch`与`xpolicylab.patch`分别是整理时对应外部Git工作树的已跟踪文件差异；对应`.base.txt`登记官方/上游HEAD。三者必须应用于各自仓库，不能跨仓库应用。

RoboSyn补丁基于官方提交`6f555fc3c9f514bb8fea6c7de036e68b9390fa5b`，归档以下5个适配文件：

- `scripts/run_env.py`：有界定向/纠正采集、场景profile和采集清单；
- `policy/act/scripts/train.py`：观测契约、数据组合/抽样、增强和训练回执；
- `robosynchallenge/managers/datasets.py`：数据访问适配；
- `policy/act/deploy_policy.py`：部署参数适配；
- `scripts/eval_policy.py`：有界评测、初始化和逐局证据。

应用前在匹配版本的独立副本执行`git apply --check /path/to/patch`，审查通过再应用。不要对已修改的当前工作树重复应用。

本快照不包含未跟踪文件、仿真资产、权重或完整上游仓库，也不是完整安装器。RoboSyn上述已跟踪运行插桩已归档；RoboTwin的未跟踪环境配置仍需独立审计。上游许可证保持适用；发布前核查补丁来源与许可。
