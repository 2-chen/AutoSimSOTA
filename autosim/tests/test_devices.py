import tempfile
import types
import unittest
from pathlib import Path

from autosim.research.devices import (DEFAULT_DEVICE_ENV, NoCompatibleDevice,
                                      align_process_defaults, apply_capability, build_plan,
                                      default_device_index, describe, discover, gpu_class,
                                      identity_is_safe, legacy_index_lock_held,
                                      normalize_uuid, ordinal_of, parse_compute_apps,
                                      parse_gpu_csv, probe_receipt_key, renderer_for, select,
                                      shard_count, shard_plan)

IDLE = """0, GPU-eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8, NVIDIA GeForce RTX 5090, 32607, 12, 0, 580.95.05, 12.0
1, GPU-aaaa1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 8, 0, 580.95.05, 12.0
2, GPU-cccc1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 4, 0, 580.95.05, 12.0
3, GPU-bbbb1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 6, 0, 580.95.05, 12.0
"""

BUSY = """0, GPU-eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8, NVIDIA GeForce RTX 5090, 32607, 31120, 97, 580.95.05, 12.0
1, GPU-aaaa1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, 8, 0, 580.95.05, 12.0
2, [N/A], NVIDIA GeForce RTX 5090, 32607, 4, 0, 580.95.05, 12.0
3, GPU-bbbb1111-2222-3333-4444-555566667777, NVIDIA GeForce RTX 5090, 32607, [N/A], [N/A], 580.95.05, 12.0
"""

TORCH_ROWS = [
    {"index": 0, "uuid": "eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8", "name": "NVIDIA GeForce RTX 5090"},
    {"index": 1, "uuid": "aaaa1111-2222-3333-4444-555566667777", "name": "NVIDIA GeForce RTX 5090"},
    {"index": 2, "uuid": "cccc1111-2222-3333-4444-555566667777", "name": "NVIDIA GeForce RTX 5090"},
    {"index": 3, "uuid": "bbbb1111-2222-3333-4444-555566667777", "name": "NVIDIA GeForce RTX 5090"},
]

ALL_VERIFIED = {"passed": True, "devices": {
    "eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8": "verified_sim",
    "aaaa1111-2222-3333-4444-555566667777": "verified_sim",
    "cccc1111-2222-3333-4444-555566667777": "verified_sim",
    "bbbb1111-2222-3333-4444-555566667777": "verified_sim"}}

NO_LOCK = staticmethod(lambda index: False)


def make_report(*, table=IDLE, compute_apps="", environ=None, torch_rows=None):
    def runner(command, **kwargs):
        if "--query-compute-apps" in " ".join(command):
            return compute_apps
        return table
    return discover(runner=runner, environ=environ or {}, torch_rows=torch_rows or TORCH_ROWS,
                    disk_path=Path(tempfile.gettempdir()))


def ready_report(**kwargs):
    return apply_capability(make_report(**kwargs), ALL_VERIFIED, index_lock_held=NO_LOCK)


class ParsingTests(unittest.TestCase):
    def test_parses_uuid_model_memory_driver_and_compute_capability(self):
        rows = parse_gpu_csv(IDLE)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(rows[1]["model"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(rows[1]["memory_used_mib"], 8)
        self.assertEqual(rows[1]["driver_version"], "580.95.05")
        self.assertEqual(rows[1]["compute_capability"], "12.0")

    def test_absent_values_stay_none_rather_than_being_guessed(self):
        rows = parse_gpu_csv(BUSY)
        self.assertIsNone(rows[2]["uuid"])
        self.assertIsNone(rows[3]["memory_used_mib"])
        self.assertIsNone(rows[3]["utilization_gpu_pct"])

    def test_compute_apps_parses_pids_by_uuid(self):
        rows = parse_compute_apps("GPU-aaaa1111-2222-3333-4444-555566667777, 28141, 31000")
        self.assertEqual(rows, [{"gpu_uuid": "GPU-aaaa1111-2222-3333-4444-555566667777",
                                 "pid": 28141, "used_memory_mib": 31000}])

    def test_uuid_prefix_and_case_do_not_matter_for_comparison(self):
        self.assertEqual(normalize_uuid("GPU-eed8f088-ad"), normalize_uuid("eed8f088-ad"))


class DiscoveryTests(unittest.TestCase):
    def test_outer_cuda_visible_devices_restricts_the_allowed_range(self):
        report = make_report(environ={"CUDA_VISIBLE_DEVICES": "1,2"})
        self.assertEqual(report["allowed"], [1, 2])
        self.assertIn("CUDA_VISIBLE_DEVICES", report["outer"]["reason"])

    def test_nvidia_visible_devices_void_exposes_nothing(self):
        report = apply_capability(make_report(environ={"NVIDIA_VISIBLE_DEVICES": "void"}),
                                  ALL_VERIFIED, index_lock_held=NO_LOCK)
        self.assertEqual(report["allowed"], [])
        self.assertIn("no device is allowed", describe(report))

    def test_nvidia_visible_devices_uuids_select_by_identity_not_index(self):
        report = make_report(environ={"NVIDIA_VISIBLE_DEVICES":
                                      "GPU-bbbb1111-2222-3333-4444-555566667777"})
        self.assertEqual(report["allowed"], [3])

    def test_torch_and_nvidia_smi_order_disagreement_is_recorded(self):
        swapped = [dict(row, uuid=TORCH_ROWS[(row["index"] + 1) % 4]["uuid"]) for row in TORCH_ROWS]
        self.assertIsNotNone(make_report(torch_rows=swapped)["cuda_order_mismatch"])
        self.assertIsNone(make_report()["cuda_order_mismatch"])

    def test_report_covers_cpu_memory_disk_and_index_to_uuid_mapping(self):
        report = make_report()
        self.assertEqual(report["visible_index_to_uuid"]["1"],
                         "GPU-aaaa1111-2222-3333-4444-555566667777")
        self.assertIn("logical_count", report["cpu"])
        self.assertIn("free_bytes", report["disk"])


class CapabilityTests(unittest.TestCase):
    def test_without_a_probe_receipt_no_device_is_usable(self):
        report = apply_capability(make_report(), None, index_lock_held=NO_LOCK)
        self.assertEqual(report["usable"], [])
        self.assertTrue(all(g["capability"] == "unknown" for g in report["gpus"]))

    def test_probe_receipt_makes_the_devices_it_verified_usable(self):
        report = ready_report()
        self.assertEqual([g["index"] for g in report["usable"]], [0, 1, 2, 3])
        self.assertIn("4 of 4 allowed devices usable", describe(report))

    def test_device_without_a_uuid_cannot_be_leased_and_is_unusable(self):
        report = apply_capability(make_report(table=BUSY), ALL_VERIFIED, index_lock_held=NO_LOCK)
        self.assertEqual(report["gpus"][2]["capability"], "unknown")
        self.assertNotIn(2, [g["index"] for g in report["usable"]])

    def test_unreadable_memory_is_flagged_as_unverified_occupancy(self):
        report = apply_capability(make_report(table=BUSY), ALL_VERIFIED, index_lock_held=NO_LOCK)
        self.assertTrue(report["gpus"][3]["occupancy_unverified"])
        self.assertFalse(report["gpus"][1]["occupancy_unverified"])

    def test_busy_device_is_reported_as_waiting_with_its_evidence(self):
        report = apply_capability(
            make_report(table=BUSY,
                        compute_apps="GPU-aaaa1111-2222-3333-4444-555566667777, 28141, 31000"),
            ALL_VERIFIED, index_lock_held=NO_LOCK)
        waiting = {g["index"]: g for g in report["waiting"]}
        self.assertEqual(waiting[0]["busy_evidence"], ["31120 MiB in use"])
        self.assertIn("28141", waiting[1]["busy_evidence"][0])
        self.assertNotIn(0, [g["index"] for g in report["usable"]])
        self.assertIn("index 0 busy", describe(report))

    def test_a_legacy_index_lock_counts_as_occupancy_evidence(self):
        report = apply_capability(make_report(), ALL_VERIFIED,
                                  index_lock_held=lambda index: index == 1)
        self.assertEqual(report["gpus"][1]["busy_evidence"], ["legacy index lock held"])
        self.assertNotIn(1, [g["index"] for g in report["usable"]])

    def test_incompatible_device_keeps_its_verdict_and_stays_out(self):
        receipt = {"passed": True, "devices": {**ALL_VERIFIED["devices"],
                                              "aaaa1111-2222-3333-4444-555566667777": "incompatible"}}
        report = apply_capability(make_report(), receipt, index_lock_held=NO_LOCK)
        self.assertEqual(report["gpus"][1]["capability"], "incompatible")
        self.assertNotIn(1, [g["index"] for g in report["usable"]])

    def test_legacy_index_lock_probe_answers_for_a_real_lock_file(self):
        import fcntl
        path = Path("/tmp/autosim-robosyn-gpu-99.lock")
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertFalse(legacy_index_lock_held(99))
        with path.open("w") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertTrue(legacy_index_lock_held(99))


class SelectionTests(unittest.TestCase):
    def test_pinned_index_addresses_the_engine_by_physical_index(self):
        device = {"index": 2, "uuid": "GPU-x", "model": "NVIDIA GeForce RTX 5090"}
        chosen = select(device, "pinned_index")
        self.assertEqual((chosen.vulkan_gpu_id, chosen.torch_index, chosen.cuda_visible),
                         (2, 0, "2"))

    def test_identity_mode_keeps_cuda_index_equal_to_the_physical_index(self):
        device = {"index": 2, "uuid": "GPU-x", "model": "NVIDIA GeForce RTX 5090"}
        chosen = select(device, "identity")
        self.assertEqual((chosen.vulkan_gpu_id, chosen.torch_index, chosen.cuda_visible),
                         (2, 2, None))

    def test_isolated_mode_sees_one_device_as_index_zero(self):
        device = {"index": 2, "uuid": "GPU-x", "model": "NVIDIA GeForce RTX 5090"}
        chosen = select(device, "node_isolated")
        self.assertEqual((chosen.vulkan_gpu_id, chosen.torch_index, chosen.cuda_visible),
                         (0, 0, "0"))

    def test_renderer_is_always_explicit_never_auto(self):
        self.assertEqual(renderer_for("NVIDIA GeForce RTX 5090"), "hybrid")
        self.assertEqual(renderer_for("NVIDIA H100 80GB HBM3"), "fast-rt")
        for model in ("NVIDIA GeForce RTX 5090", "NVIDIA H100 PCIe", "unknown"):
            self.assertNotEqual(renderer_for(model), "auto")

    def test_gpu_class_is_stable_and_ignores_punctuation(self):
        self.assertEqual(gpu_class("NVIDIA GeForce RTX 5090"), gpu_class("nvidia geforce rtx 5090"))


class IdentitySafetyTests(unittest.TestCase):
    """Identity addressing means "the engine index *is* the CUDA ordinal" -- nothing may renumber.

    The 5090 pool measured the failure this refuses to plan around: with one visible card the
    engine's ``cuDeviceGet(&m_cudaDevice, 3)`` (``OptixDevice.cpp``) returns
    ``CUDA_ERROR_INVALID_DEVICE`` ("Invalid device ID: 3. Available devices: 0-0").
    """

    def plan(self, report=None, **kwargs):
        options = {"report": report if report is not None else ready_report(),
                   "requested": "auto", "mode": "identity", "max_parallel_jobs": None,
                   "gpu_hours": None, "hours": 24.0, "probe_receipt_sha256": "abc"}
        options.update(kwargs)
        return build_plan(**options)

    def test_identity_is_safe_when_every_card_keeps_its_own_index(self):
        self.assertEqual(identity_is_safe(ready_report()), (True, ""))
        report = ready_report(environ={"CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        self.assertEqual(identity_is_safe(report), (True, ""))
        plan = self.plan(report=report)
        self.assertEqual([g["selection"]["torch_index"] for g in plan["usable"]], [0, 1, 2, 3])
        self.assertEqual([g["selection"]["vulkan_gpu_id"] for g in plan["usable"]], [0, 1, 2, 3])
        self.assertTrue(all(g["selection"]["cuda_visible"] is None for g in plan["usable"]),
                        "identity must not narrow the visible set")

    def test_identity_is_refused_when_the_outer_filter_renumbers_the_cards(self):
        for visible in ("1,2", "3,2,1,0", "0,1"):
            report = ready_report(environ={"CUDA_VISIBLE_DEVICES": visible})
            safe, why = identity_is_safe(report)
            self.assertFalse(safe, visible)
            self.assertIn(visible, why)
            with self.assertRaises(NoCompatibleDevice):
                self.plan(report=report)

    def test_pinned_index_still_honours_a_restricting_outer_filter(self):
        report = ready_report(environ={"CUDA_VISIBLE_DEVICES": "1,2"})
        plan = self.plan(report=report, mode="pinned_index")
        self.assertEqual([g["selection"]["cuda_visible"] for g in plan["usable"]], ["1", "2"])
        self.assertEqual([g["selection"]["torch_index"] for g in plan["usable"]], [0, 0])


class PlanTests(unittest.TestCase):
    def plan(self, report=None, **kwargs):
        options = {"report": report if report is not None else ready_report(), "requested": "auto",
                   "mode": "pinned_index", "max_parallel_jobs": None, "gpu_hours": None,
                   "hours": 24.0, "probe_receipt_sha256": "abc"}
        options.update(kwargs)
        return build_plan(**options)

    def test_plan_records_mode_selection_and_uuid_mapping(self):
        plan = self.plan()
        self.assertEqual(plan["mode"], "pinned_index")
        self.assertEqual([d["index"] for d in plan["usable"]], [0, 1, 2, 3])
        self.assertEqual(plan["usable"][2]["selection"]["vulkan_gpu_id"], 2)
        self.assertEqual(plan["usable"][2]["selection"]["torch_index"], 0)
        self.assertEqual(plan["visible_index_to_uuid"]["0"],
                         "GPU-eed8f088-ad1c-3c82-e41f-c8bd3d8e80d8")

    def test_requested_subset_must_stay_inside_the_allowed_range(self):
        with self.assertRaises(ValueError):
            self.plan(requested="0,5")
        self.assertEqual([d["index"] for d in self.plan(requested="1,3")["usable"]], [1, 3])

    def test_waiting_devices_are_recorded_with_a_reason(self):
        plan = self.plan(report=apply_capability(make_report(table=BUSY), ALL_VERIFIED,
                                                 index_lock_held=NO_LOCK))
        self.assertEqual([d["index"] for d in plan["usable"]], [1, 3])
        reasons = {d["index"]: d["reason"] for d in plan["waiting"]}
        self.assertIn("31120 MiB in use", reasons[0])
        self.assertIn("no UUID", reasons[2])

    def test_default_gpu_hours_limit_is_hours_times_usable_devices(self):
        self.assertEqual(self.plan(hours=24.0)["gpu_hours_limit"], 96.0)
        self.assertEqual(self.plan(hours=24.0, gpu_hours=8)["gpu_hours_limit"], 8)

    def test_max_parallel_jobs_is_clamped_to_the_usable_set(self):
        self.assertEqual(self.plan(max_parallel_jobs=9)["max_parallel_jobs"], 4)
        self.assertEqual(self.plan(max_parallel_jobs=2)["max_parallel_jobs"], 2)
        with self.assertRaises(ValueError):
            self.plan(max_parallel_jobs=0)

    def test_no_usable_device_raises_with_the_reason_attached(self):
        with self.assertRaises(NoCompatibleDevice):
            self.plan(report=apply_capability(make_report(), None, index_lock_held=NO_LOCK))

    def test_probe_receipt_key_is_scoped_to_the_node_and_device_set(self):
        key = probe_receipt_key(host="h", driver="580.95.05", image="img", worker_spec="spec",
                                uuids=["GPU-a", "GPU-b"])
        self.assertEqual(key, probe_receipt_key(host="h", driver="580.95.05", image="img",
                                               worker_spec="spec", uuids=["GPU-b", "GPU-a"]))
        self.assertNotEqual(key, probe_receipt_key(host="h2", driver="580.95.05", image="img",
                                                  worker_spec="spec", uuids=["GPU-a", "GPU-b"]))


class ShardPolicyTests(unittest.TestCase):
    def test_shard_plan_is_contiguous_and_covers_every_episode_once(self):
        for episodes, count in ((40, 4), (7, 3), (200, 4), (9, 1)):
            blocks = shard_plan(episodes, count)
            self.assertEqual(sum(size for _, size in blocks), episodes)
            self.assertEqual([offset for offset, _ in blocks][0], 0)
            covered = [index for offset, size in blocks for index in range(offset, offset + size)]
            self.assertEqual(covered, list(range(episodes)))
            self.assertLessEqual(max(size for _, size in blocks) - min(size for _, size in blocks), 1)

    def test_more_shards_than_episodes_is_refused(self):
        with self.assertRaises(ValueError):
            shard_plan(2, 3)

    def test_small_banks_stay_serial_and_large_banks_shard(self):
        cost = {"construction_seconds": 360, "marginal_seconds": 40}
        self.assertEqual(shard_count(3, 4, max_parallel_jobs=4, cost_model=cost), 1)
        self.assertEqual(shard_count(8, 4, max_parallel_jobs=4, cost_model=cost), 1)
        self.assertEqual(shard_count(40, 4, max_parallel_jobs=4, cost_model=cost), 3)
        self.assertEqual(shard_count(200, 4, max_parallel_jobs=4, cost_model=cost), 4)

    def test_shard_count_never_exceeds_ready_devices_or_max_parallel_jobs(self):
        cost = {"construction_seconds": 360, "marginal_seconds": 40}
        self.assertEqual(shard_count(200, 2, max_parallel_jobs=4, cost_model=cost), 2)
        self.assertEqual(shard_count(200, 4, max_parallel_jobs=2, cost_model=cost), 2)


class FakeCuda:
    def __init__(self, *, available=True, failure=None):
        self.available, self.failure, self.selected = available, failure, []

    def is_available(self):
        return self.available

    def set_device(self, index):
        if self.failure is not None:
            raise self.failure
        self.selected.append(index)


def torch_like(cuda):
    return types.SimpleNamespace(cuda=cuda)


class FakeWarp:
    def __init__(self, *, failure=None):
        self.failure, self.default = failure, None

    def set_device(self, ident):
        if self.failure is not None:
            raise self.failure
        self.default = ident


def importer(*, torch_module=None, warp_module=None):
    """Answer the two imports ``align_process_defaults`` makes, and nothing else."""
    def load(name):
        if name == "torch" and torch_module is not None:
            return torch_module
        if name == "warp" and warp_module is not None:
            return warp_module
        raise ModuleNotFoundError(f"No module named {name!r}")
    return load


class DefaultDeviceTests(unittest.TestCase):
    """The second half of the device contract: what a *default* device resolves to.

    ``CUDA_VISIBLE_DEVICES`` cannot fix this one -- an identity process needs every card
    visible so the engine can resolve its physical index -- so the libraries that pick a card
    when a call does not name one have to be pointed at the leased card instead.  Measured
    (second multi-GPU probe): 1030 MiB appeared on card 0 at 0.0% utilization while an
    identity arm ran on card 3.
    """

    def test_the_default_device_is_pointed_at_the_leased_card(self):
        cuda, warp = FakeCuda(), FakeWarp()
        record = align_process_defaults(3, importer=importer(torch_module=torch_like(cuda),
                                                             warp_module=warp))
        self.assertEqual(cuda.selected, [3])
        self.assertEqual(warp.default, "cuda:3")
        self.assertEqual(record, {"torch_index": 3, "torch_default": 3, "warp_default": 3})

    def test_a_process_without_torch_or_warp_still_starts(self):
        """A diagnostic that can fail is a diagnostic that turns a measurement into a crash."""
        record = align_process_defaults(2, importer=importer())
        self.assertEqual(record["torch_default"], "unavailable: ModuleNotFoundError")
        self.assertEqual(record["warp_default"], "unavailable: ModuleNotFoundError")

    def test_an_ordinal_this_process_cannot_use_is_reported_not_raised(self):
        record = align_process_defaults(3, importer=importer(
            torch_module=torch_like(FakeCuda(failure=RuntimeError("invalid device ordinal"))),
            warp_module=FakeWarp(failure=RuntimeError("invalid device ordinal"))))
        self.assertEqual(record["torch_default"], "unavailable: RuntimeError")
        self.assertEqual(record["warp_default"], "unavailable: RuntimeError")

    def test_a_cpu_only_process_says_so_and_leaves_torch_alone(self):
        cuda = FakeCuda(available=False)
        record = align_process_defaults(1, importer=importer(torch_module=torch_like(cuda),
                                                             warp_module=FakeWarp()))
        self.assertEqual(record["torch_default"], "no_cuda")
        self.assertEqual(cuda.selected, [])

    def test_the_policy_child_never_imports_warp(self):
        """The child is torch-only: it must not pay for a library it does not use."""
        asked = []

        def load(name):
            asked.append(name)
            return torch_like(FakeCuda())

        record = align_process_defaults(3, warp=False, importer=load)
        self.assertEqual(asked, ["torch"])
        self.assertIsNone(record["warp_default"])

    def test_ordinal_of_reads_the_torch_device_spellings(self):
        self.assertEqual(ordinal_of("cuda"), 0)
        self.assertEqual(ordinal_of("cuda:3"), 3)
        self.assertEqual(ordinal_of("CUDA:2"), 2)
        for other in ("cpu", "", None, "cuda:x", "cuda:"):
            self.assertIsNone(ordinal_of(other), other)

    def test_only_a_device_plan_asks_for_alignment(self):
        self.assertIsNone(default_device_index({}))
        self.assertIsNone(default_device_index({DEFAULT_DEVICE_ENV: ""}))
        self.assertIsNone(default_device_index({DEFAULT_DEVICE_ENV: "not-a-number"}))
        self.assertEqual(default_device_index({DEFAULT_DEVICE_ENV: "3"}), 3)


if __name__ == "__main__":
    unittest.main()
