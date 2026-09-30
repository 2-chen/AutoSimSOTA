from autosim.research.common import atomic_json, digest, object_digest
from autosim.research.policy_provenance import source_snapshot_policy


def test_run_generated_checkpoint_is_not_shipped(tmp_path):
    output = tmp_path / "run"
    repo = output / "checkout"
    repo.mkdir(parents=True)
    released = repo / "released.pth"
    released.write_bytes(b"release")
    entries = [["file", "released.pth", released.stat().st_size,
                digest(released), 0o644]]
    atomic_json(output / "workspace_snapshot.json", {
        "schema_version": 2, "destination": str(repo.resolve()),
        "source_entries": entries,
        "source_tree_fingerprint": object_digest(entries)})
    generated = repo / "experiments" / "candidate.pth"
    generated.parent.mkdir()
    generated.write_bytes(b"new policy")
    assert source_snapshot_policy(released, repo=repo, output=output)["status"] == "source_snapshot"
    assert source_snapshot_policy(generated, repo=repo, output=output)["status"] == "run_generated"
    released.write_bytes(b"altered")
    assert source_snapshot_policy(released, repo=repo, output=output)["status"] == "changed"
