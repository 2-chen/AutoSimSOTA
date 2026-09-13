# GitHub发布整理与剩余限制

核查日期：2026-09-12。初次整理不上传GitHub、不提交密钥、不删除实验工件；随后用户明确授权将核心代码推送至公开仓库`2-chen/AutoSimSOTA`。

## 已整理

根目录忽略凭据、环境、上游仓库、数据和实验输出；提供核心安装入口与无密钥模板。第三方Git工作树修改保存在`patches/`，同时登记HEAD；未将整个第三方仓库纳入核心实现。嵌套上游仓库仍保留在本地，不能用`git add -f`绕过忽略规则。

## 可移植性尚未完成

1. RoboSyn已移到外部，但`repository_autoresearch.py`的真实运行仍检查benchmark/evaluator必须位于项目内部；不能承诺外部输入已能长跑。应改为明确登记的外部benchmark依赖，而非隐式扫描父项目；保留核心代码、凭据和输出边界。
2. `research/runtime.py`的项目根目录判断依赖内置RoboSyn目录存在，移动后可能误判；EmbodiChain和资产位置也仍使用约定目录。需以明确根目录与外部依赖配置替代存在性猜测。
3. `research/runtime.py`与`research/robotwin_jobs.py`仍包含开发机conda路径。需显式配置runtime Python、库路径和资产位置，并用清洁工作目录验证。
4. 部分兼容逻辑位于外部benchmark。RoboSyn的5个已跟踪插桩文件现已归档到`patches/robosyn.patch`并登记基线提交；补丁不包含未跟踪脚本、资产或完整上游仓库。正式一键安装前仍需把补丁应用步骤自动化，并补齐其他backend的版本锁与未跟踪兼容文件审计。
5. 源码包含历史开发路径、旧实验入口和说明；这些不是密钥，但不等于通用可运行配置。暂不删除，避免破坏历史复现。
6. 自动多卡、跨设备兼容、完整预算恢复仍在规划中；不要在发布说明中写成已有能力。

## 发布前验收

本轮测试结果：默认测试为205通过、14失败、2个模块跳过；失败涉及仍绑定旧RoboSyn目录的发现/配置/采集测试，不应声称全量回归通过。已为两个直接导入外部训练脚本的模块提供显式路径，以下定向测试13项通过：

```bash
AUTOSIM_ROBOSYN_REPO=/absolute/path/to/RoboSynChallenge .venv/bin/python -m pytest -q autosim/tests/test_robosyn_data_pipeline.py autosim/tests/test_robosyn_v3_training.py
```

该环境变量目前仅接入上述两个测试模块，不是整个运行时的统一路径配置；其余测试与运行时依赖解耦仍待完成。缺少PyTorch的纯核心环境也不能直接运行所有训练相关测试。当前是上传内容整理完成，不是清洁安装/CI验收完成。

- 审查`git status --short`和待提交差异；确认不存在`.env`、私钥、数据或模型。不要把本地运行报告整目录强制加入Git。
- 核查补丁中没有凭据；逐一确认上游许可证与核心代码来源，再决定整体LICENSE。
- 完成CPU测试；在无旧工作区依赖的环境验证外部仓库发现和一次原生闭环后，才能宣称便携部署完成。
- 曾在聊天或其他渠道公开过的API令牌应轮换；`.gitignore`不能撤销已泄露的凭据。
- 用户已确认目标为公开仓库`https://github.com/2-chen/AutoSimSOTA`并授权提交推送；许可证仍待维护者确认。发布此开发快照不表示上述运行/测试限制已解决。
