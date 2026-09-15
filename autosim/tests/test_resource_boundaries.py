import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from autosim.research.common import atomic_json
from autosim.research.devices import cuda_child_env, discover, outer_allowed
from autosim.research.host_resources import capacity


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"index": i, "uuid": f"GPU-{i}abcdef"} for i in range(4)]

    def test_explicit_masks_survive_child_queries(self):
        for value in ("", "-1", "none", "void"):
            env = {"CUDA_VISIBLE_DEVICES": value}
            rows, outer = outer_allowed(env, self.rows)
            self.assertEqual(rows, [])
            self.assertEqual(cuda_child_env(env, outer), env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", cuda_child_env({}, {}))

    def test_uuid_order_and_intersection(self):
        rows, _ = outer_allowed({"CUDA_VISIBLE_DEVICES": "GPU-3abc,GPU-1abc",
                                 "NVIDIA_VISIBLE_DEVICES": "1,3"}, self.rows)
        self.assertEqual([r["index"] for r in rows], [3, 1])

    def test_invalid_identifier_stops_cuda_enumeration(self):
        rows, _ = outer_allowed({"CUDA_VISIBLE_DEVICES": "2,-1,3"}, self.rows)
        self.assertEqual([r["index"] for r in rows], [2])

    def test_nvml_listing_without_cuda_access_is_not_an_allocation(self):
        def runner(argv):
            return "0, GPU-abcdef, NVIDIA GPU, 32000, 0, 0, 580, 12" if "--query-gpu" in str(argv) else ""
        report = discover(runner=runner, environ={}, torch_rows=[])
        self.assertEqual(report["allowed"], [])


class CapacityTests(unittest.TestCase):
    def test_v2_takes_parent_limits_and_current_memory_into_account(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cg, proc = root / "cgroup", root / "proc"
            (cg / "jobs/worker").mkdir(parents=True)
            (proc / "self").mkdir(parents=True)
            (proc / "self/cgroup").write_text("0::/jobs/worker\n")
            (proc / "meminfo").write_text("MemTotal: 1000000000 kB\nMemAvailable: 900000000 kB\n")
            (cg / "cpu.max").write_text("max 100000")
            (cg / "jobs/cpu.max").write_text("250000 100000")
            (cg / "jobs/worker/cpu.max").write_text("800000 100000")
            (cg / "jobs/memory.max").write_text(str(16 * 2**30))
            (cg / "jobs/memory.current").write_text(str(5 * 2**30))
            got = capacity(cgroup=cg, proc=proc, affinity=set(range(56)))
            self.assertEqual(got["cpu"]["effective_cpus"], 2.5)
            self.assertEqual(got["memory"]["total_mib"], 16 * 1024)
            self.assertEqual(got["memory"]["available_mib"], 11 * 1024)

    def test_v1_limits_and_affinity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for leaf in ("proc/self", "cg/cpu/job", "cg/memory/job"):
                (root / leaf).mkdir(parents=True)
            (root / "proc/self/cgroup").write_text("2:cpu,cpuacct:/job\n3:memory:/job\n")
            (root / "cg/cpu/job/cpu.cfs_quota_us").write_text("400000")
            (root / "cg/cpu/job/cpu.cfs_period_us").write_text("100000")
            (root / "cg/memory/job/memory.limit_in_bytes").write_text(str(2**30))
            (root / "cg/memory/job/memory.usage_in_bytes").write_text(str(2**29))
            got = capacity(cgroup=root / "cg", proc=root / "proc", affinity={1, 4})
            self.assertEqual(got["cpu"]["effective_cpus"], 2)
            self.assertEqual(got["memory"]["available_mib"], 512)


class DurableWriteTests(unittest.TestCase):
    def test_concurrent_publishers_never_share_a_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "state.json"
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda n: atomic_json(output, {"generation": n, "payload": "x" * 1000}), range(32)))
            record = json.loads(output.read_text())
            self.assertIn(record["generation"], range(32))
            self.assertEqual(record["payload"], "x" * 1000)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
