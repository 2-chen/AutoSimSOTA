from autosim.research.data_version import fingerprint_dataset


def test_dataset_fingerprint_reuses_hashes_only_after_metadata_match(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    (root / "meta/info.json").write_text("{}")
    (root / "data/episode.parquet").write_bytes(b"parquet-fixture")
    store = tmp_path / "versions"
    first = fingerprint_dataset(root, store)
    second = fingerprint_dataset(root, store)
    assert first["content_id"] == second["content_id"]
    assert second["cache_validation"] == "all_relative_paths_size_and_mtime_ns_matched"

    (root / "data/episode.parquet").write_bytes(b"changed-fixture")
    third = fingerprint_dataset(root, store)
    assert third["content_id"] != first["content_id"]
