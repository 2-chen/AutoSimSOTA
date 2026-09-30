import concurrent.futures

import pytest

from autosim.research.agent_budget import AgentCostBudgetError, AgentCostLedger


def test_flexible_allowance_expands_and_releases_but_preserves_other_holds(tmp_path):
    ledger = AgentCostLedger(tmp_path, run_id="flex", limit_usd=1)
    first = ledger.reserve(session_id="s", role="init", requested_usd=.2)
    other = ledger.reserve(session_id="t", role="recorder", requested_usd=.1)
    assert ledger.extend(first["reservation_id"], .8) == pytest.approx(.8)
    with pytest.raises(AgentCostBudgetError, match="cannot reserve"):
        ledger.extend(first["reservation_id"], .91)
    ledger.settle(first["reservation_id"], actual_usd=.05, launched=True)
    tail = ledger.reserve(session_id="u", role="fix", requested_usd=2, allow_partial=True)
    assert tail["allowed_usd"] == pytest.approx(.85)
    assert ledger.snapshot()["reserved_usd"] == pytest.approx(.95)
    assert other["reservation_id"] != tail["reservation_id"]


def test_provider_cost_releases_unused_reservation_but_never_starts_partial_turn(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.10)
    first = ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.07)
    assert first["allowed_usd"] == pytest.approx(0.07)
    after_first = ledger.settle(first["reservation_id"], actual_usd=0.04, launched=True)
    assert after_first["spent_usd"] == pytest.approx(0.04)
    assert after_first["reserved_usd"] == pytest.approx(0)

    with pytest.raises(AgentCostBudgetError, match="full.*turn reservation"):
        ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.08)

    second = ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.06)
    assert second["allowed_usd"] == pytest.approx(0.06)
    assert second["remaining_usd"] == pytest.approx(0)


def test_missing_provider_usage_stays_reserved_and_blocks_future_spend(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.05)
    reservation = ledger.reserve(session_id="s1", role="fix", requested_usd=0.05)
    usage = ledger.settle(reservation["reservation_id"], actual_usd=None, launched=True)
    assert usage["spent_usd"] == pytest.approx(0)
    assert usage["reserved_usd"] == pytest.approx(0.05)
    assert usage["unknown_entries"] == 1
    with pytest.raises(AgentCostBudgetError, match="exhausted"):
        ledger.reserve(session_id="s2", role="scheduler", requested_usd=0.01)


def test_unknown_usage_requires_explicit_idempotent_provider_reconciliation(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.05)
    reservation = ledger.reserve(session_id="s1", role="fix", requested_usd=0.05)
    ledger.settle(reservation["reservation_id"], actual_usd=None, launched=True)

    with pytest.raises(ValueError, match="evidence reference"):
        ledger.reconcile_unknown(reservation["reservation_id"], actual_usd=0.01,
                                 evidence=" ")
    reconciled = ledger.reconcile_unknown(
        reservation["reservation_id"], actual_usd=0.012,
        evidence="provider billing record 2026-09-29")
    assert reconciled["spent_usd"] == pytest.approx(0.012)
    assert reconciled["reserved_usd"] == pytest.approx(0)
    assert reconciled["remaining_usd"] == pytest.approx(0.038)
    assert ledger.reconcile_unknown(
        reservation["reservation_id"], actual_usd=0.012,
        evidence="provider billing record 2026-09-29") == reconciled
    with pytest.raises(AgentCostBudgetError, match="already reconciled"):
        ledger.reconcile_unknown(reservation["reservation_id"], actual_usd=0.02,
                                 evidence="different provider record")


def test_unlaunched_turn_releases_reservation(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.05)
    reservation = ledger.reserve(session_id="s1", role="resource", requested_usd=0.05)
    usage = ledger.settle(reservation["reservation_id"], actual_usd=None, launched=False)
    assert usage["remaining_usd"] == pytest.approx(0.05)
    assert usage["entry_count"] == 1


def test_provider_cost_overrun_is_recorded_and_refuses_next_turn(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.05)
    reservation = ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.05)
    usage = ledger.settle(reservation["reservation_id"], actual_usd=0.051, launched=True)
    assert usage["spent_usd"] == pytest.approx(0.051)
    assert usage["remaining_usd"] == 0
    with pytest.raises(AgentCostBudgetError, match="exhausted"):
        ledger.reserve(session_id="s2", role="fix", requested_usd=0.001)


def test_small_remaining_total_cannot_be_passed_as_a_smaller_provider_cap(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.20)
    first = ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.15)
    ledger.settle(first["reservation_id"], actual_usd=0.149, launched=True)

    with pytest.raises(AgentCostBudgetError, match=r"leaving \$0\.051000.*full \$0\.150000"):
        ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.15)

    assert ledger.snapshot()["spent_usd"] == pytest.approx(0.149)
    assert ledger.snapshot()["remaining_usd"] == pytest.approx(0.051)


def test_budget_is_frozen_across_resume_and_settlement_is_idempotent(tmp_path):
    output = tmp_path / "run"
    ledger = AgentCostLedger(output, run_id="r1", limit_usd=0.10)
    reservation = ledger.reserve(session_id="s1", role="scheduler", requested_usd=0.04)
    assert ledger.settle(reservation["reservation_id"], actual_usd=0.02,
                         launched=True) == ledger.settle(
                             reservation["reservation_id"], actual_usd=0.02, launched=True)
    with pytest.raises(AgentCostBudgetError, match="silently change"):
        AgentCostLedger(output, run_id="r1", limit_usd=0.20).snapshot()


def test_concurrent_turns_cannot_overreserve_the_run_limit(tmp_path):
    ledger = AgentCostLedger(tmp_path / "run", run_id="r1", limit_usd=0.10)

    def reserve(index):
        try:
            return ledger.reserve(session_id=f"s{index}", role="scheduler",
                                  requested_usd=0.03)
        except AgentCostBudgetError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))

    accepted = [row for row in results if row is not None]
    assert len(accepted) == 3
    assert sum(row["allowed_usd"] for row in accepted) == pytest.approx(0.09)
    assert ledger.snapshot()["remaining_usd"] == pytest.approx(0.01)


def test_historical_cli_ledger_cannot_be_reinterpreted_as_official_pricing(tmp_path):
    output = tmp_path / "run"
    old = AgentCostLedger(output, run_id="r1", limit_usd=1)
    old.reserve(session_id="s1", role="scheduler", requested_usd=.1)
    with pytest.raises(AgentCostBudgetError, match="accounting basis changed"):
        AgentCostLedger(output, run_id="r1", limit_usd=1,
                        cost_basis="deepseek_official_estimate_v1").snapshot()
    assert old.snapshot()["cost_basis"] == "cli_reported"
