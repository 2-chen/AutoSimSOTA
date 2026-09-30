"""A second run cannot reset the user's 24 GPU-hour repository cap."""

import json
import hashlib
import subprocess

import pytest

from autosim.research import repository_budget


def test_repository_reservations_survive_restart_and_share_one_cap(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(repository_budget.time, "time", lambda: clock[0])
    repo = tmp_path / "repo"
    repo.mkdir()
    ledger = tmp_path / "ledger"
    first = repository_budget.RepositoryBudget(ledger, repo=repo, cap_seconds=100)
    first.reserve(tmp_path / "run1", wall_seconds=60)
    reopened = repository_budget.RepositoryBudget(ledger, repo=repo, cap_seconds=100)
    reopened.reserve(tmp_path / "run1", wall_seconds=60)
    with pytest.raises(ValueError, match="insufficient unreserved"):
        reopened.reserve(tmp_path / "run2", wall_seconds=41)
    reopened.reserve(tmp_path / "run2", wall_seconds=40)
    clock[0] = 110.0
    first.finish(tmp_path / "run1")
    assert reopened.state()["charged_seconds"] == 10
    assert reopened.state()["reserved_seconds"] == 40
    reopened.reserve(tmp_path / "run3", wall_seconds=50)
    with pytest.raises(ValueError, match="insufficient unreserved"):
        reopened.reserve(tmp_path / "run4", wall_seconds=1)


def test_repository_cap_cannot_be_raised_or_reset_by_a_new_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    ledger = tmp_path / "ledger"
    budget = repository_budget.RepositoryBudget(ledger, repo=repo, cap_seconds=86400)
    budget.reserve(tmp_path / "run1", wall_seconds=86400)
    with pytest.raises(ValueError, match="insufficient unreserved"):
        budget.reserve(tmp_path / "run2", wall_seconds=1)
    with pytest.raises(ValueError, match="24-hour user cap"):
        repository_budget.RepositoryBudget(ledger, repo=repo, cap_seconds=86401)
    with pytest.raises(ValueError, match="cannot silently change"):
        repository_budget.RepositoryBudget(ledger, repo=repo,
                                           cap_seconds=86000).reserve(
                                               tmp_path / "run3", wall_seconds=1)


def test_finished_run_can_resume_only_with_remaining_original_deadline(tmp_path, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(repository_budget.time, "time", lambda: clock[0])
    repo = tmp_path / "repo"
    repo.mkdir()
    budget = repository_budget.RepositoryBudget(tmp_path / "ledger", repo=repo,
                                                 cap_seconds=100)
    output = tmp_path / "run"
    budget.reserve(output, wall_seconds=50)
    clock[0] = 110.0
    budget.finish(output)
    resumed = budget.reserve(output, wall_seconds=50)
    assert resumed["reserved_seconds"] == 40
    clock[0] = 115.0
    budget.finish(output)
    assert budget.state()["charged_seconds"] == 15
    clock[0] = 151.0
    with pytest.raises(ValueError, match="deadline has expired"):
        budget.reserve(output, wall_seconds=50)


def test_isolated_copy_chain_shares_the_original_repository_budget(tmp_path):
    original = tmp_path / "sources" / "maniskill"
    first_run = tmp_path / "run_a"
    first = first_run / "checkout"
    second_run = tmp_path / "run_b"
    second = second_run / "checkout"
    for path in (original, first, second):
        path.mkdir(parents=True)
    (first_run / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(original), "destination": str(first)}), encoding="utf-8")
    (second_run / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(first), "destination": str(second)}), encoding="utf-8")

    assert repository_budget.source_repository(second) == original.resolve()
    assert repository_budget.repository_key(second) == repository_budget.repository_key(original)


def test_budget_provenance_requires_a_matching_destination(tmp_path):
    original = tmp_path / "sources" / "unrelated"
    checkout = tmp_path / "run" / "checkout"
    original.mkdir(parents=True)
    checkout.mkdir(parents=True)
    (checkout.parent / "workspace_snapshot.json").write_text(json.dumps({
        "source": str(original), "destination": str(tmp_path / "not-this-checkout")}),
        encoding="utf-8")

    assert repository_budget.source_repository(checkout) == checkout.resolve()


def test_a_nested_directory_does_not_inherit_its_parent_git_remote(tmp_path):
    parent = tmp_path / "outer_repository"
    checkout = parent / "checkout"
    parent.mkdir()
    checkout.mkdir()
    subprocess.run(["git", "init", str(parent)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(parent), "remote", "add", "origin",
                    "https://example.invalid/outer.git"], capture_output=True, check=True)

    expected = repository_budget.object_digest({"repository": str(checkout.resolve())})
    assert repository_budget.repository_key(checkout) == expected


def _write_alias_run(tmp_path, *, original, name="run", charged=60.0):
    output = tmp_path / name
    checkout = output / "checkout"
    checkout.mkdir(parents=True)
    manifest = {"source": str(original.resolve()), "destination": str(checkout.resolve())}
    (output / "workspace_snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output, {
        "status": "finished", "started_epoch": 100.0, "budget_started_epoch": 100.0,
        "wall_seconds": 100.0, "reserved_seconds": 0.0, "charged_seconds": charged,
        "accounting": "one-GPU full wall-time upper bound", "finished_epoch": 160.0,
    }


def _write_path_keyed_ledger(ledger, checkout, rows):
    old_key = repository_budget.object_digest({"repository": str(checkout.resolve())})
    path = ledger / f"{old_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "repository_key": old_key,
                               "cap_seconds": 100, "runs": rows}), encoding="utf-8")
    return path


def test_proven_legacy_alias_is_merged_once_and_consumes_shared_cap(tmp_path):
    original = tmp_path / "source" / "robot"
    original.mkdir(parents=True)
    ledger = tmp_path / "ledger"
    output, row = _write_alias_run(tmp_path, original=original)
    alias = _write_path_keyed_ledger(ledger, output / "checkout", {str(output): row})
    alias_hash = hashlib.sha256(alias.read_bytes()).hexdigest()

    budget = repository_budget.RepositoryBudget(ledger, repo=original, cap_seconds=100)
    first = budget.state()
    assert first["charged_seconds"] == 60
    assert first["runs"] == 1
    assert budget.state()["charged_seconds"] == 60
    with pytest.raises(ValueError, match="insufficient unreserved"):
        budget.reserve(tmp_path / "over_cap", wall_seconds=41)
    budget.reserve(tmp_path / "at_cap", wall_seconds=40)

    canonical = json.loads(budget.path.read_text(encoding="utf-8"))
    assert canonical["legacy_sources"][alias.stem] == {
        "filename": alias.name, "sha256": alias_hash, "imported_runs": 1}
    assert hashlib.sha256(alias.read_bytes()).hexdigest() == alias_hash


def test_unrelated_alias_ledger_is_not_imported(tmp_path):
    original = tmp_path / "source" / "robot"
    unrelated = tmp_path / "source" / "other"
    original.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    ledger = tmp_path / "ledger"
    output, row = _write_alias_run(tmp_path, original=unrelated)
    _write_path_keyed_ledger(ledger, output / "checkout", {str(output): row})

    budget = repository_budget.RepositoryBudget(ledger, repo=original, cap_seconds=100)
    assert budget.state()["charged_seconds"] == 0
    assert budget.state()["runs"] == 0


def test_proven_alias_with_unverifiable_extra_run_fails_closed(tmp_path):
    original = tmp_path / "source" / "robot"
    original.mkdir(parents=True)
    ledger = tmp_path / "ledger"
    output, row = _write_alias_run(tmp_path, original=original)
    rows = {str(output): row, str(tmp_path / "missing_run"): dict(row)}
    _write_path_keyed_ledger(ledger, output / "checkout", rows)

    budget = repository_budget.RepositoryBudget(ledger, repo=original, cap_seconds=100)
    with pytest.raises(ValueError, match="mixes verified and unknown runs"):
        budget.state()
    assert not budget.path.exists()


def test_changed_legacy_source_after_import_requires_reconciliation(tmp_path):
    original = tmp_path / "source" / "robot"
    original.mkdir(parents=True)
    ledger = tmp_path / "ledger"
    output, row = _write_alias_run(tmp_path, original=original)
    alias = _write_path_keyed_ledger(ledger, output / "checkout", {str(output): row})
    budget = repository_budget.RepositoryBudget(ledger, repo=original, cap_seconds=100)
    assert budget.state()["charged_seconds"] == 60

    changed = json.loads(alias.read_text(encoding="utf-8"))
    changed["runs"][str(output)]["charged_seconds"] = 61
    alias.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="changed after import"):
        budget.reserve(tmp_path / "new_run", wall_seconds=1)


def test_alias_manifest_must_bind_destination_to_the_run_output(tmp_path):
    original = tmp_path / "source" / "robot"
    original.mkdir(parents=True)
    ledger = tmp_path / "ledger"
    output, row = _write_alias_run(tmp_path, original=original)
    manifest = json.loads((output / "workspace_snapshot.json").read_text(encoding="utf-8"))
    manifest["destination"] = str((tmp_path / "elsewhere" / "checkout").resolve())
    (output / "workspace_snapshot.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_path_keyed_ledger(ledger, output / "checkout", {str(output): row})

    budget = repository_budget.RepositoryBudget(ledger, repo=original, cap_seconds=100)
    assert budget.state()["charged_seconds"] == 0
    assert budget.state()["runs"] == 0
