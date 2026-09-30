# 从 tests/test_scout.py 移走的两个测试

`test_the_round_one_rule_follows_the_plan` 和 `test_a_supplied_plan_is_frozen_into_the_protocol`
测的是旧循环 `repository_autoresearch.validate_proposal` 的语义 —— 一个按 milestone 收窄
取值菜单的验证器。通用路径的验证器是 `research/decision.validate_proposal`,语义不同,
由 `tests/test_derived_research.py` 覆盖。

原文保存在 `test_scout.py` 的 git 历史里。
