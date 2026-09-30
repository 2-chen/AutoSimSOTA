import tempfile
import unittest
from pathlib import Path

from autosim.experiment_validation.asset_audit import inventory
from autosim.research.common import digest


class AssetAuditTest(unittest.TestCase):
    def test_hashes_nested_files_and_flags_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            asset = root / "nested/asset.bin"
            asset.write_bytes(b"fixture-asset")
            (root / "link").symlink_to(asset)
            report = inventory(root, lambda *args: None)
            self.assertEqual(report["file_count"], 1)
            self.assertEqual(report["files"]["nested/asset.bin"]["sha256"], digest(asset))
            self.assertEqual(report["issues"][0]["kind"], "symlink_requires_explicit_review")


if __name__ == "__main__":
    unittest.main()
