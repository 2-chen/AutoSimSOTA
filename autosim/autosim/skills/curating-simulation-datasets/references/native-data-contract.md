# 原生数据卡与训练曝光合同

用于 `register_data_version` 的可选元数据，以及原生采集/处理/loader 的有界日志事件。它们已接入执行器，不要求安装统一数据格式或额外平台。Agent 决定当前仓库怎么读取、检查、选择和混合；Supervisor 审核精确源码及冻结协议。先小试，不为可选统计拖延实际训练。

## 1. 登记资产

已有四个必需字段不变；`dataset_ref` 指实际能由原生 loader 消费的文件/目录，不是只写了一份选择清单就假定原生 loader 理解它：

```json
{
  "id": "new_training_data",
  "dataset_ref": "research/.../native_outputs/.../dataset",
  "source_refs": [{"ref": "relative_native_source.py", "quote": "exact current source text"}],
  "reason": "本次合法训练数据的来源与 loader 连接",
  "asset": {
    "kind": "processed",
    "parents": ["raw_batch"],
    "producer_evidence_id": "32位十六进制的实际原生attempt ID",
    "license": "来源许可与适用范围的简短声明",
    "description": "按完整 episode 选择，不覆盖原数据"
  }
}
```

`kind` 可选 existing/official/collected/processed/synthetic，描述来源类别，不认证动作提供者质量。已有/官方数据可以只登记来源，不需要伪造本系统采集回执；可选 `profile_evidence_id` 引用独立原生检查。collected/processed/synthetic 必须给成功封存 collect/prepare_data 的生产者 ID；processed 必须给已登记 parent。旧卡没有 asset 仍可加载，来源、数量和检查未知，不猜补。

父卡包括实际字节身份、记录 hash 和资源路径。生产者数据必须属于其真实产物位置，stage/output/working directory 已封存，不能拿 train 日志或另一个目录的文件冒充采集。处理中保持 raw 单独位置，输出不与任何父目录重叠。

## 2. 原生健康/数量报告

从当前 native reader 检查合法训练数据，完成 writer close/finalize 后，向 stdout 输出一行：

```text
AUTOSIM_DATA_PROFILE {"identity_field":"content_sha256","identity":"真实数据字节SHA256","scope":"full","format":"当前原生格式和计数单位","training_only":true,"counts":{"episodes":20,"transitions":2400,"samples":2400},"issues":{"truncated_episodes":0,"nonfinite_actions":0},"coverage":{"scene_count":4},"parents":[]}
```

上例数字只是字段示意，不是默认配额。文件身份是 content_sha256；目录身份是 sha256，即系统 `artifact_identity` 的有序相对文件名/hash 清单摘要，不能拿单个文件或整目录名称代替。原生代码在实际读写后计算，不回显输入常量。

每个已封存 attempt 对该数据身份恰好一个匹配事件，每行不超过 16000 字节。scope 为 full/sample/metadata：sample 表示抽样范围里的数量，不能当成全量；metadata 表示原生元数据记录，不冒充实扫。counts 只支持 episodes/transitions/samples/successful_episodes/failed_episodes/invalid_episodes；后三项若填写必须互斥，总数不超过 episodes。未知字段不填。issues 和 coverage 是最多 32 项非负整数计数，各键不超过 64 字符，不放 raw episode、画面或动作序列。

检查什么由源码语义决定：字段/有限值/边界/截断/时间对齐/控制单位/坐标/gripper/训练划分/缓存与根目录等；小批真实 loader 和必要原生回放能回答不同问题。不要把 `training_only=true` 当作独立用途证明，Supervisor 仍必须审核合法范围及源码。没有报告的检查项不是零；“未发现异常”不是正确动作或优质数据认证。

处理后的事件还需要 parents 和 recipe：

```json
{
  "parents": [{"id":"raw_batch","identity_field":"sha256","identity":"父数据身份","record_sha256":"父数据卡文件SHA256"}],
  "recipe": {"operation":"native complete-episode selection","parameters":{"grouping":"source_episode","criterion":"documented development-only criterion"},"preserves_episode_groups":true}
}
```

父 ID/hash 必须与实际卡、实际读取的 raw 一致；parameters 是不超过 4 KiB 的 JSON。大规模选择索引写在原生输出里并纳入该数据版本身份，而不是塞进 profile 或主 Agent 上下文。转换源码/配置用 source_refs 提供精确证据；完整 episode 与来源链不得跨训练/验证混用。当前未实现通用 splitter 或自动去重，布尔声明不能代替原生行为审核。

## 3. 实际训练曝光

继续用已审核轴的 `data-version:<id>`；系统解析为授权 native 路径。loader 实际消费 batch 后发原消费事件，可以增加：

```text
AUTOSIM_TRAIN_DATA_CONSUMPTION {"data_version_id":"mixed_view","setting":"dataset_root","identity_field":"sha256","identity":"实际输入身份","samples_read":100,"exposure":{"batches":10,"unique_samples":70,"source_samples":{"official_data":80,"new_expert_data":20}}}
```

只能在实际 batch 读取处累计，不使用请求训练量、目录发现条数或配置混合比例冒充。samples_read 是样本读取次数，允许重复；unique_samples 需要可核验的原生样本 ID，没有能力就省略，不要求把所有 ID 注入日志。source_samples 只允许当前版本和直接父版本 ID，和必须等于该次 samples_read。每个版本/setting 恰好一个最终汇总事件；不同窗口/重复采样要说明计数单位。没有 exposure 的旧事件仍通过基础读取合同，但 RUN.md 不宣称有完整曝光。

正式采用前和训练后复核数据卡、实际数据及父链、生产/检查封存日志和回执；矛盾或变动不计分。真实曝光也不能证明训练充分、标签正确或指标提高；结论仍来自冻结 evaluator 的实际加载/rollout/指标链。

## 4. 可见性与边界

Main/Recorder 获得元数据短目录：类型、父 ID、profile 计数/范围、有限检查项、证据和卡 hash。完整卡按需读取；短目录不扫描 raw、展开 episode 或重复加载全部配方。新卡触发技能选择状态更新，普通心跳不触发。RUN.md 自动展示中文来源/数量/检查表、实际曝光与开发指标；至少两个全量 episode 计数才画数量对照，父/派生数据不能相加为“新增”。

仍保留 64 GiB/10000 目录项和有界父链检查。未实现分片发布、断点续采、转换缓存、跨 run 索引、自动采集数量优化器、数据价值评分、独立备份或 raw GC。卡和 recipe 不复制/删除/上传数据，也不修改源仓库和冻结协议。旧会话缺采集/处理边时，不能事后偷改 stages 解除验收；应先按实际能力报告缺口。
