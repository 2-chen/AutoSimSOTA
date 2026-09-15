import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from autosim.research.common import assert_frozen, freeze_files, read_json
from autosim.research.controller import controller_source_files
from autosim.research.production import run as production_run


class ControllerSourceFilesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = SimpleNamespace(workspace=self.root, repo=self.root / "benchmark",
                                       eval_repo=self.root / "evaluation", output=self.root / "output")
        self.sources = [
            self.runtime.eval_repo / "robosynchallenge/tasks/task.py",
            self.runtime.eval_repo / "configs/task/eval.json",
            self.root / "AutoSimSOTA/EmbodiChain/embodichain/lab/envs/tasks/task.py",
            self.runtime.repo / "policy/act/model/policy.py",
            self.runtime.eval_repo / "robosynchallenge/env/scene.py",
            self.runtime.eval_repo / "scripts/eval_policy.py",
            self.runtime.repo / "policy/act/scripts/train.py",
            self.runtime.repo / "scripts/run_env.py",
            self.runtime.eval_repo / "policy/act/deploy_policy.py",
        ]
        for path in self.sources:
            self.write(path)

    @staticmethod
    def write(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n" if path.suffix == ".json" else "VALUE = 1\n")

    def test_environments_and_linked_directories_are_pruned_before_traversal(self):
        act = self.runtime.repo / "policy/act"
        excluded = []
        for name in (".venv", "venv", ".git", "__pycache__", ".pytest_cache", "site-packages"):
            path = act / name / "nested/package.py"
            self.write(path)
            excluded.append(path)
        named_environment = act / "training_interpreter"
        self.write(named_environment / "nested/package.py")
        (named_environment / "pyvenv.cfg").write_text("home = /usr/bin\n")
        excluded.append(named_environment / "nested/package.py")
        external = self.root / "external_packages"
        self.write(external / "package.py")
        (act / "linked_packages").symlink_to(external, target_is_directory=True)
        excluded.append(act / "linked_packages/package.py")
        # Verify pruning itself, rather than merely filtering files after an
        # expensive traversal has already walked an environment tree.
        scan = os.scandir
        visited = []
        def checked_scan(path):
            path = Path(path)
            visited.append(path)
            self.assertFalse(any(path == file.parents[1] or file.parents[1] in path.parents
                                 for file in excluded[:-1]))
            self.assertNotEqual(path, act / "linked_packages")
            return scan(path)
        with mock.patch("os.scandir", side_effect=checked_scan):
            files = controller_source_files(self.runtime)
        self.assertTrue(visited)
        self.assertTrue(set(self.sources).issubset(files))
        self.assertFalse(set(excluded).intersection(files))

    def test_real_benchmark_and_embodichain_source_changes_still_break_freeze(self):
        frozen = freeze_files(controller_source_files(self.runtime))
        ignored = self.runtime.repo / "policy/act/.venv/lib/package.py"
        self.write(ignored)
        assert_frozen(frozen)
        for path in self.sources[:4]:
            with self.subTest(source=str(path.relative_to(self.root))):
                original = path.read_text()
                path.write_text(original + "# changed scientific source\n")
                with self.assertRaisesRegex(RuntimeError, "frozen protocol changed"):
                    assert_frozen(frozen)
                path.write_text(original)

    def test_production_continuation_uses_same_pruned_source_inventory(self):
        excluded = self.runtime.repo / "policy/act/.venv/lib/package.py"
        self.write(excluded)
        production_run(self.runtime, [], suite_hours=0)
        frozen = read_json(self.runtime.output / "production_continuation/frozen_core.json")
        self.assertNotIn(str(excluded), frozen)
        self.assertIn(str(self.runtime.repo / "policy/act/model/policy.py"), frozen)
        with mock.patch("autosim.research.production.source_tree_files", side_effect=AssertionError("unnecessary re-enumeration")):
            production_run(self.runtime, [], suite_hours=0)


if __name__ == "__main__":
    unittest.main()
