import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.queue_manifest import versioned_manifest
from autosim.research.common import digest, read_json


class QueueManifestTest(unittest.TestCase):
    def test_restart_reuses_same_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = versioned_manifest(root, {"protocol": 1, "sources": {"a": "one"}})
            before = digest(first)
            second = versioned_manifest(root, {"protocol": 1, "sources": {"a": "one"}})
            self.assertEqual(first, second)
            self.assertEqual(before, digest(second))

    def test_changed_source_appends_lineage_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = versioned_manifest(root, {"protocol": 1, "sources": {"a": "one"}})
            first_hash = digest(first)
            second = versioned_manifest(root, {"protocol": 1, "sources": {"a": "two"}})
            self.assertEqual(second.name, "manifest.v2.json")
            self.assertEqual(digest(first), first_hash)
            self.assertEqual(read_json(second)["supersedes_sha256"], first_hash)
            self.assertEqual(versioned_manifest(root, {"protocol": 1, "sources": {"a": "two"}}), second)

    def test_each_protocol_change_is_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            versioned_manifest(root, {"protocol": 1})
            versioned_manifest(root, {"protocol": 2})
            third = versioned_manifest(root, {"protocol": 3})
            self.assertEqual(third.name, "manifest.v3.json")


if __name__ == "__main__":
    unittest.main()
