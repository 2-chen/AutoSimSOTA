# 从 tests/test_harness_sco.py 移走的一个测试

`test_remote_shell_verifies_frozen_bwrap_and_sets_its_path_before_entrypoint`

**它验的性质是对的,而且重要**:远端启动脚本在进入 entrypoint 之前,要先校验冻结的
bwrap 二进制的哈希,并把 `AUTOSIM_REPAIR_BWRAP` 设好;哈希不对就不进入。

**它站不住了,有两个原因**:

1. 被测对象 `tools/run_full_research_sco.sh` 是旧世界的启动器,已随旧世界移出。
   测试把这个脚本从 `tools/` 拷进去跑 —— 脚本没了,测试就没有对象。
2. 它依赖 `/data/AutoResearch/AutoSimSOTA/harness_validation/dependencies/bwrap`,
   这台机器上没有。所以在移出之前它就已经是红的。

**这条性质应该重新测**,对象换成新的远端启动器。原文:

```python
@pytest.mark.parametrize("valid", [True, False])
```
