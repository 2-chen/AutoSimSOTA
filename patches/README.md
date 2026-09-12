# 外部兼容补丁快照

`robotwin.patch`与`xpolicylab.patch`分别是整理时对应外部Git工作树的已跟踪文件差异；对应`.base.txt`登记HEAD。两者应用于各自仓库，不能将XPolicyLab补丁应用于RoboTwin根目录。

应用前在匹配版本的独立副本执行`git apply --check /path/to/patch`，审查通过再应用。不要对已修改的当前工作树重复应用。

本快照不包含未跟踪文件、仿真资产、权重或完整上游仓库，也不是完整安装器。RoboSyn历史修改与RoboTwin的未跟踪环境配置尚需独立审计后归档。上游许可证保持适用；发布前核查补丁来源与许可。
