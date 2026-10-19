# 原生数据卡与训练曝光合同

用于 `register_data_version` 的可选元数据，以及原生采集/处理/loader 的有界日志事件。它们已接入执行器，不要求安装统一数据格式或额外平台。Agent 决定当前仓库怎么读取、检查、选择和混合；Supervisor 审核精确源码及冻结协议。先小试，不为可选统计拖延实际训练。

## 1. 登记资产

登记与消费是两个阶段，不能互为先决条件。登记审核数据身份、合法训练域、来源与格式，以及源码中实际可达的 root→样本 reader 连接；不要求先训练、先安装日志 emitter 或先出现非空 `AUTOSIM_TRAIN_SELECTION_JSON`（它只在选择已登记 handle 后才由执行器填入）。登记后由 Main 在受审核候选中显式选择 handle，审核安装后的消费连接，最后以真实训练 batch 事件证明消费；登记成功不认证样本已被训练使用。冻结研究意图不是违反 benchmark 协议的豁免，实际协议禁令与保留集隔离仍须遵守。

先核对生产数据的实际 schema 与 reader 要求，不因仓库存在 converter 就默认转换。输入 staging 空目录不是已采集的数据；必须通过明确的输入绑定连接原 producer。原路径未找到导致自动下载，应视为输入绑定故障而非数据缺失；不得伪造版本号、改名冒充另一格式或重复采集来掩盖连接问题。

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

`kind` 可选 existing/official/collected/processed/synthetic，描述来源类别，不认证动作提供者质量。已有/官方数据可以只登记来源，不伪造采集回执；可选 `profile_evidence_id` 引用原生汇总检查。新数据必须给封存 collect/prepare_data 的生产者 ID；processed 必须给已登记 parent。旧卡没有 asset 仍可加载，来源、数量和检查未知，不猜补。

collected 可另给 `manifest_evidence_id`：成功的只读全量 native inspector 检查实际提交的 episode/视频/数据字节。原生采集器不必自行打印 profile；即使原生产者 failed/timed_out/cancelled/interrupted，完整可用部分也可另行审核登记。生产者失败/退出码保留，不能写成完成请求数量。专家语义、合法训练域仍由独立审核决定，格式通过不自动填 training_only 或 expert。

默认 `producer_scope:"this_run"`。用户授权、账本继承已核验的新前瞻研究可用 `producer_scope:"budget_parent"` 复用原研究采集数据；必须做本研究的新全量检查，保留原失败/谱系/消费。它是共同起点，不是新 arm 的新增获取。旧确认记录不能删除以开放研究。

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

父 ID/hash 必须与实际卡、实际读取的 raw 一致；parameters 是不超过 4 KiB 的 JSON。大规模选择索引写在原生输出里并纳入该数据版本身份，而不是塞进主 Agent 上下文。转换源码/配置用 source_refs 提供精确证据；完整 episode 与来源链不得跨训练/验证混用。框架提供可逆 manifest 选择，不统一转换原生格式或猜来源划分；声明不能代替原生行为审核。

### 只读格式检查与可逆视图

原生环境确为 LeRobot v2.1 时，Data/Main 可选择现有 `inspect_native_environment` 的 `data_profile_request`：format、checkout/resource 相对 root、max_episodes、decode_videos、include_manifest。不是此格式则读当前 native reader 设计有源码的检查，不强制转换。完整 manifest/选择保存在回执；返回短摘要和全局 evidence_id，用 read_evidence 分页，不重复展开所有 episode。

新数据不在只读 worker 快照时，使用 root="@producer" 和 producer={evidence_id,artifact_index,relative_path?,scope:"this_run"}。索引来自封存 collect/prepare_data 的 candidate_paths；只有选中的数据目录只读挂载，不能请求 host path 或整个 run。授权继承研究可用 scope="budget_parent"。

只读worker不复制原生环境，可借用研究owner已发布的解释器/依赖只读执行；原生checkout地址映射为委派快照，而非开放父源码。未在source_paths里的模块不会自动出现。CPU前后原生context须稳定，native_context_stable=false的观察不能登记为健康manifest或coverage证明；源码捕获ID由owner封存，Main可直接解析，不跨worker猜路径。

`manifest_fields` 将真实 episode 元数据字段映射为 source_family/source_instance/split/supervision；conditions 每项是 {metadata_field,unit,source_ref}。未知字段保持 null，不能把常量 "train"/"reset_expert" 当作字段名制造证明。

`selection` 接受 target={field,op,value,unit}、limit、deduplicate(bytes/source_instance/none)、require_split、require_supervision。op 为 eq/in/between/lt/le/gt/ge，无表达式执行。缺控制观测/单位不符不会当负观察；缺原生 episode 内容身份时不能将共享 HDF5 文件判为重复。split 默认 train；原生未记录 split 时，仅在独立数据用途审核下可显式取 null，而非伪填元数据。视图保留完整 episode，仍须原生 converter/loader 真正消费所选集合。

`coverage_reference={evidence_id,target}` 比较此前封存 manifest 与本次实际条件。返回 observed_coverage_change/no_observed_coverage_change/needs_native_observation；不同覆盖不直接证明控制因果性，仍需相同合法域/seed流和原生路由审核。no-op 或不可观察时先诊断或改走事后筛选，不自动增采。

其他原生格式可用同一只读工具的 format="source_manifest"、inspector_ref="checkout相对.py"、root（或同样producer grant）。由Main/Init/Fix连接当前原生reader的有界检查脚本，不强制HDF5/其他格式转LeRobot。脚本从AUTOSIM_DATA_ROOT取已授权根，stdout只发一个JSON对象：format、scope、counts（episodes/transitions/samples等原生单位）、issues、video_decode_scope（full或not_applicable）、episode_manifest；manifest仍需上述实际文件/hash/提交/观察条件合同。源码限16KiB、CPU只读，统计须分批/有界；未知监督/覆盖不填常量，视频适用时不能用not_applicable逃检查。Source/字节封存在回执，版本登记额外由Supervisor核验manifest_observations_supported，实际source变动会拒绝旧见证；工具标签或打印JSON不是观察认证。Data只读，缺脚本时请求Main委派实现。

## 3. 实际训练曝光

继续用已审核轴的 `data-version:<id>`；系统解析为授权 native 路径。loader 实际消费 batch 后发原消费事件，可以增加：

```text
AUTOSIM_TRAIN_DATA_CONSUMPTION {"data_version_id":"mixed_view","setting":"dataset_root","identity_field":"sha256","identity":"实际输入身份","reader_id":"trainer:0","consumption_stage":"training_batch","samples_read":100,"exposure":{"batches":10,"unique_samples":70,"source_samples":{"official_data":80,"new_expert_data":20}}}
```

在真正被训练更新使用的 batch 处发 consumption_stage="training_batch"；__getitem__/预取只记 sample_read（旧事件默认此类型），不是已经训练。samples_read 是累计样本读取次数，允许重复；进度事件可单调递增，系统取每个 reader 最后值，不把累积计数相加。每个独立 reader 生命周期用不同稳定 reader_id，计数回退或重复冲突必须修复。若同时有 batch 与预取只采用 batch 口径，不能叠加。

unique_samples 需要可核验的原生样本 ID，没有能力就省略；多 reader 没有全局去重见证时不相加 unique。source_samples 只允许当前版本和直接父版本 ID，和须等于实际 samples_read；不由混合配置推算。setting 是实际选择 data-version 的轴名，不硬编码 dataset_root 或旧版本ID。没有 exposure 的旧事件保持基础读取事实，但不满足 v2 数据 arm 的 batch 消费要求。

正式采用前和训练后复核数据卡、实际数据及父链、生产/检查封存日志和回执；矛盾或变动不计分。真实曝光也不能证明训练充分、标签正确或指标提高；结论仍来自冻结 evaluator 的实际加载/rollout/指标链。

## 4. 可见性与边界

Main/Recorder 获得元数据短目录：类型、父 ID、profile 计数/范围、有限检查项、证据和卡 hash。完整卡按需读取；短目录不扫描 raw、展开 episode 或重复加载全部配方。新卡触发技能选择状态更新，普通心跳不触发。RUN.md 自动展示中文来源/数量/检查表、实际曝光与开发指标；至少两个全量 episode 计数才画数量对照，父/派生数据不能相加为“新增”。

仍保留 64 GiB/10000 目录项和有界父链检查。未实现分片发布、通用断点续采/转换缓存、跨 run 数据仓库、学习型数据价值评分、独立备份或 raw GC。Card/recipe/manifest 不复制、删除或上传 raw。旧会话缺采集/处理边或已经确认时，不能偷改 stages/删除确认开放研究；依授权建立继承账本的前瞻研究。

## 5. 原生采集控制

`submit_native_job(stage="collect", collection_settings={episodes,seed?,controls?,control_source_refs?}, ...)`：controls 为最多12个原生字段到JSON值，传入 builder 的专用 `i["collection_controls"]`，不通过评测或训练调参表注入。精确 source refs 与独立 Supervisor 核验训练合法域、caller→consumer、动作供给；builder/source/实际值身份封存在作业中并在 worker 执行前复核。数量和 seed 同样要真实消费，未使用的输入拒绝。

字段名或argv存在不证明场景改变。Data要求先记录真实条件并用小试比较 coverage，再根据产出率、下游成本和候选收益建议扩量。无直接控制但目标可观测时允许随机池后选，拒绝/额外采集成本不能隐去。不要填通用 object_pose/press_depth 默认值。

## 6. 前瞻数据对照 v2

现有 register_data_campaign 可用 schema_version:2。仍需 id/development_label/hypothesis/acquisition/selection/comparison；comparison 包含 arms、usable_episodes、training_recipe、seeds、compute_accounting，加 budget_axis、budget_limit、quality_floor、frozen_factors、rule。

arms 可选 fixed_official/fixed_continuation/random_quality/rule_targeted/agent_targeted/targeted_quality/fdas；至少 random_quality+agent_targeted。先做可完成的最小比较，不要求一轮跑齐七种。v1 targeted/random/unfiltered 历史记录不改名。

- budget_axis 为 accepted_episodes/attempts/acquisition_seconds/total_compute。前两者为整数，接受量与 usable_episodes 一致；完整计算的 budget_limit 是明确 {gpu_seconds?,cpu_seconds?,wall_seconds?,model_usd?} 资源包，不加总不同单位。共同 Q0 和初始化/学习量/评测因素在干预前冻结。
- rule_targeted 时 rule={algorithm:"failure_frequency_v1",strata_evidence_id,source_refs}；原生开发诊断发唯一 AUTOSIM_FAILURE_STRATA 后接 [{id,failures,evaluated},...]。从真实开发记录统计，不打印模型常量。已冻结的失败频率/最大余数分配只决定配额，实际训练域/生成使用仍需查证；未运行规则对照不声称LLM优于规则。
- 正式训练和开发计分后 record_data_campaign_arm 的字段为 campaign_id/arm/seed/measurement_label/training_version_id；新增 arm 再给 acquired_version_id，后者必须是本研究实际 collect 后代；fixed controls 不填新增版本。不同 arm/seed 不能复用同一个 measurement；batch 消费是必要条件，登记仍不是公平性认证。
- 数量未匹配保持 incomplete_or_unknown；不能事后降低目标、补零或复制轨迹。预算轴对应的全部原生成本仍需独立审查，matching 数字不是 matching 成本。
- v2 已声明工作未完成时先做数据 arm，不抢跑 confirm_best。真正无安全下一步时，可向 record_data_campaign_arm 给 boundary={status:"unsupported"|"budget_limited",reason,evidence_refs:[稳定ID],source_refs:[精确引用]}，不同时给分数/版本。Supervisor 独立审核；可修故障不是永久边界，边界是未完成实验而非零分。

Recorder 自动用中文展示版本/部分失败/漏斗/未知项、实际 batch 曝光、方法与预算口径、对照与边界；demo 仍需现有原生媒体准入，不把格式检查视频算成策略得分。

Recorder可请求{kind:"dataset_preview",data_version_id,purpose}；Main通过现有review_report_demo核验登记卡/完整manifest并复用少量已存在的完整native视频，每episode一个相机、最多三个，不额外跑GPU。身份变化或留出后关闭；默认样本不是代表性成功/失败统计。RUN.md/HTML显示视频、真实帧和版本关联；没有native视频则报告边界，不合成。
