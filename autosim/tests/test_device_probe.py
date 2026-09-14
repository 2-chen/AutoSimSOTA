import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autosim.research import device_probe, evaluation
from autosim.research.devices import DEFAULT_DEVICE_ENV, SimDeviceSelection
from autosim.research.runtime import Runtime

ALLOCATION = ["GPU-aaa", "GPU-bbb"]
IDLE = {"GPU-aaa": {"peak_memory_mib": 12, "mean_utilization_pct": 0.0, "samples": 3},
        "GPU-bbb": {"peak_memory_mib": 8, "mean_utilization_pct": 0.0, "samples": 3}}


def arm(**overrides):
    row = {"name": "B", "status": "completed", "execution_mode": "real_simulation",
           "episode_count": 1, "worker_failure": None}
    row.update(overrides)
    return row


def peak(**memory):
    return {uuid: {"peak_memory_mib": value, "mean_utilization_pct": 40.0, "samples": 5}
            for uuid, value in memory.items()}


class JudgeTests(unittest.TestCase):
    def verdict(self, row=None, peaks=None, baseline=None):
        return device_probe.judge(row or arm(), own_uuid="GPU-aaa", allocation=ALLOCATION,
                                  baseline=baseline if baseline is not None else IDLE,
                                  peak=peaks if peaks is not None else peak(**{"GPU-aaa": 4000,
                                                                               "GPU-bbb": 8}))

    def test_the_leased_device_grew_and_every_other_stayed_idle(self):
        result = self.verdict()
        self.assertTrue(result["passed"])
        self.assertEqual(result["growth_mib"]["GPU-aaa"]["growth_mib"], 3988)
        self.assertEqual(result["growth_mib"]["GPU-bbb"]["growth_mib"], 0)

    def test_a_leaking_neighbour_fails_even_when_the_leased_device_grew(self):
        result = self.verdict(peaks=peak(**{"GPU-aaa": 4000, "GPU-bbb": 2600}))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["others_stayed_idle"])

    def test_the_leased_device_not_growing_fails_the_arm(self):
        result = self.verdict(peaks=peak(**{"GPU-aaa": 40, "GPU-bbb": 8}))
        self.assertFalse(result["checks"]["own_device_grew"])

    def test_growth_is_measured_against_the_pre_run_baseline(self):
        baseline = {"GPU-aaa": {"peak_memory_mib": 3700, "samples": 1},
                    "GPU-bbb": {"peak_memory_mib": 8, "samples": 1}}
        result = self.verdict(peaks=peak(**{"GPU-aaa": 5000, "GPU-bbb": 8}), baseline=baseline)
        self.assertEqual(result["growth_mib"]["GPU-aaa"]["growth_mib"], 1300)
        self.assertTrue(result["passed"])

    def test_a_crashed_arm_fails_even_when_memory_moved(self):
        result = self.verdict(row=arm(status="failed", execution_mode=None, episode_count=None,
                                      worker_failure={"error": "SIGABRT"}),
                              peaks=peak(**{"GPU-aaa": 4000, "GPU-bbb": 8}))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["no_worker_failure"])

    def test_an_unreadable_memory_reading_is_not_evidence_of_growth(self):
        result = self.verdict(peaks=peak(**{"GPU-aaa": 40, "GPU-bbb": 8}))
        self.assertIsNotNone(result["growth_mib"]["GPU-aaa"]["growth_mib"])
        self.assertFalse(result["checks"]["own_device_grew"])


def negative(status="failed", **receipt):
    row = {"name": "C", "status": status,
           "startup_receipt": {"status": "failed", "returncode": -6,
                               "startup_phase": "environment_constructing",
                               "initializations_recorded": False, **receipt}}
    return row


class NegativeControlTests(unittest.TestCase):
    """The negative control's claim is narrow, so failing it is not enough to satisfy it."""

    def test_a_native_abort_before_the_first_reset_reproduces_it(self):
        self.assertTrue(device_probe.reproduced_native_abort(negative()))
        self.assertTrue(device_probe.reproduced_native_abort(negative(returncode=-11)))

    def test_a_command_line_error_is_not_a_reproduced_abort(self):
        # rc=2 in 1.8 s with no startup.json: the first probe read this as a falsification
        # of the model when it was a defect in the probe's own command line.
        row = negative(returncode=2, startup_phase=None)
        self.assertFalse(device_probe.reproduced_native_abort(row))

    def test_an_abort_after_the_first_reset_is_not_the_startup_abort(self):
        self.assertFalse(device_probe.reproduced_native_abort(negative(initializations_recorded=True)))
        self.assertFalse(device_probe.reproduced_native_abort(negative(startup_phase="running")))

    def test_a_missing_process_receipt_is_not_evidence_of_anything(self):
        self.assertFalse(device_probe.reproduced_native_abort({"name": "C", "status": "failed"}))
        self.assertFalse(device_probe.reproduced_native_abort({"name": "C", "status": "completed"}))


class MergedCaptureTests(unittest.TestCase):
    def test_the_diagnostic_reads_stderr_because_that_is_where_the_engine_logs(self):
        completed = mock.Mock(returncode=0, stdout="", stderr="[EmbodiChain WARNING]: boom\n")
        with mock.patch.object(device_probe.subprocess, "run", return_value=completed) as run:
            captured = device_probe.run_merged(["python", "-c", "x"], env={"A": "1"}, timeout=5)
        self.assertEqual(captured["returncode"], 0)
        self.assertIn("boom", captured["text"])
        self.assertEqual(run.call_args.kwargs["env"], {"A": "1"})

    def test_a_failure_to_launch_is_recorded_rather_than_raised(self):
        with mock.patch.object(device_probe.subprocess, "run", side_effect=OSError("no python")):
            captured = device_probe.run_merged(["python"], env={}, timeout=5)
        self.assertIsNone(captured["returncode"])
        self.assertIn("__error__", captured["text"])


class EnvironmentMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def runner(command, **kwargs):
            self.calls.append(list(command))
            index = command[-1] if command[-2] == "-c" and len(command) > 2 else None
            if "torch" in " ".join(command) and "-c" in command:
                return json.dumps({"count": 4, "rows": [
                    {"i": i, "name": "NVIDIA GeForce RTX 5090", "uuid": f"u{i}"}
                    for i in range(4)]})
            if "select_default_renderer" in " ".join(command):
                return f"RESULT_MARKER hybrid\n" if index else "RESULT_MARKER hybrid\n"
            return "  vulkan_physical_device_count=4\n    [0] uuid=GPU-aaa luid=0\n"
        self.patch = mock.patch.object(device_probe, "run_text", runner)
        self.runner = self.patch.start()
        self.addCleanup(self.patch.stop)
        self.patch_merged = mock.patch.object(
            device_probe, "run_merged",
            lambda command, **kw: {"returncode": 0, "text": runner(command) or ""})
        self.patch_merged.start()
        self.addCleanup(self.patch_merged.stop)

    def measure(self, devices=None):
        devices = devices or [{"index": i, "uuid": f"GPU-{i}"} for i in range(4)]
        return device_probe.environment_measurements(
            Path("/usr/bin/python"), {"PATH": "/usr/bin"}, devices,
            vulkan_probe=Path(__file__), loader_select=None)

    def test_shared_environments_are_measured_once_and_each_index_separately(self):
        table = self.measure()
        self.assertEqual(sorted(table["shared"]), ["cvd_all", "outer_default"])
        self.assertEqual(sorted(table["by_index"]), ["0", "1", "2", "3"])

    def test_loader_select_is_recorded_only_when_asked_for(self):
        self.assertNotIn("loader_select", self.measure()["shared"])
        table = device_probe.environment_measurements(
            Path("/usr/bin/python"), {}, [{"index": 0, "uuid": "GPU-0"}],
            vulkan_probe=Path(__file__), loader_select="0x10de:0x2b85")
        self.assertIn("loader_select", table["shared"])

    def test_a_failing_query_is_recorded_rather_than_guessed(self):
        with mock.patch.object(device_probe, "run_text", lambda command, **kw: "__error__: rc=9"):
            table = self.measure([{"index": 0, "uuid": "GPU-0"}])
        entry = table["by_index"]["0"]
        self.assertIn("error", entry["torch"])
        self.assertIsNone(entry["vulkan"]["physical_devices"])

    def test_the_engine_warning_on_stderr_is_what_the_row_records(self):
        """Six identical bare markers is what a blind measurement looks like.

        The engine answers ``select_default_renderer`` through ``log_warning`` on stderr, so a
        stdout-only capture makes every environment -- including the one predicted to fail --
        read the same. The row has to carry the line that distinguishes them.
        """
        def merged(command, **kwargs):
            if "select_default_renderer" in " ".join(command):
                return {"returncode": 0, "text": "RESULT_MARKER hybrid\n[EmbodiChain WARNING]: "
                                                 "Failed to query GPU name for device 3 (boom). "
                                                 "Defaulting renderer to 'hybrid'."}
            return {"returncode": 0, "text": ""}

        with mock.patch.object(device_probe, "run_merged", merged):
            row = self.measure([{"index": 3, "uuid": "GPU-3"}])["by_index"]["3"]
        self.assertIn("Failed to query GPU name for device 3", row["renderer_auto"])
        self.assertEqual(row["renderer_auto_returncode"], 0)


class NodeIsolationTests(unittest.TestCase):
    """The free measurement that decides whether the deeper mechanism is available.

    Every identity arm on a card other than 0 leaves ~525 MiB on card 0, and it is neither
    torch's nor warp's default device (both are aligned and the receipt shows it).  Hiding
    the other cards' device nodes would make "device 0" mean the leased card for *every*
    subsystem at once, including the unnamed one -- so whether this container permits it is
    worth seconds of measurement rather than an assumption either way.
    """

    DEVICES = [{"index": i, "uuid": f"GPU-{i}"} for i in range(4)]

    def setUp(self):
        self.calls = []

    def measure(self, results):
        def merged(command, **kwargs):
            self.calls.append(list(command))
            for needle, payload in results:
                if needle in " ".join(command):
                    return payload if isinstance(payload, dict) else {"returncode": 0, "text": payload}
            return {"returncode": 0, "text": ""}
        with mock.patch.object(device_probe, "run_merged", merged):
            return device_probe.node_isolation_measurement(
                Path("/usr/bin/python"), {"PATH": "/usr/bin"}, self.DEVICES, dri=Path("/nonexistent-dri"))

    def test_the_namespace_is_tried_three_ways_because_each_can_fail_alone(self):
        measurement = self.measure([])
        self.assertEqual(sorted(measurement["runs"]), ["namespace_only", "user_namespace",
                                                       "with_cuda_visible_0"])
        self.assertEqual(len(self.calls), 3)
        self.assertIn("unshare", self.calls[0][0])

    def test_it_hides_every_card_but_the_one_it_is_testing(self):
        measurement = self.measure([])
        self.assertEqual(measurement["target_index"], 3)
        self.assertEqual(measurement["hidden_nodes"],
                         ["/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2"])
        self.assertNotIn("/dev/nvidia3", " ".join(self.calls[0]))

    def test_a_surviving_card_that_renumbers_to_zero_is_the_feasible_answer(self):
        measurement = self.measure([("unshare", {"returncode": 0, "text": "RESULT_MARKER " + json.dumps(
            {"nvidia_smi": "0, GPU-3", "torch_count": 1, "torch_rows": [{"i": 0}]})})])
        self.assertTrue(measurement["runs"]["namespace_only"]["feasible"])
        self.assertTrue(measurement["feasible"])

    def test_a_survivor_that_keeps_its_own_index_is_infeasible_and_says_so(self):
        """Hiding a node is not the same as renumbering it: the minor can survive."""
        measurement = self.measure([("unshare", {"returncode": 0, "text": "RESULT_MARKER " + json.dumps(
            {"nvidia_smi": "3, GPU-3", "torch_count": 1, "torch_rows": [{"i": 0}]})})])
        row = measurement["runs"]["namespace_only"]
        self.assertEqual(row["survivor_index"], "3")
        self.assertFalse(row["feasible"])
        self.assertIn("output", row)

    def test_a_refused_unshare_is_recorded_with_its_own_output(self):
        measurement = self.measure([("unshare", {"returncode": 1,
                                             "text": "unshare: unshare failed: Operation not permitted"})])
        row = measurement["runs"]["namespace_only"]
        self.assertIsNone(row["survivor_index"])
        self.assertFalse(row["feasible"])
        self.assertIn("Operation not permitted", row["output"])

    def test_a_bind_that_failed_disqualifies_the_run_even_if_one_card_remains(self):
        """A partially hidden node set looks like a one-card node and is not one."""
        measurement = self.measure([("unshare", {"returncode": 0, "text":
                                             "BIND_FAILED /dev/nvidia1\nRESULT_MARKER " + json.dumps(
                                                 {"nvidia_smi": "0, GPU-3", "torch_count": 1})})])
        row = measurement["runs"]["namespace_only"]
        self.assertEqual(row["bind_failures"], ["/dev/nvidia1"])
        self.assertFalse(row["feasible"])

    def test_every_variant_must_work_because_production_uses_the_cvd_spelling(self):
        measurement = self.measure([("--map-root-user", {"returncode": 1, "text": "denied"})])
        self.assertFalse(measurement["feasible"])
        self.assertEqual(measurement["runs"]["user_namespace"]["returncode"], 1)

    def test_an_unreadable_dri_directory_is_not_an_exception(self):
        measurement = self.measure([])
        self.assertIn("unreadable", str(measurement["dri_nodes"]))

    def test_feasibility_is_never_a_verification(self):
        """It must not be able to reach the verdict: only a real episode can verify a mode."""
        source = Path(device_probe.__file__).read_text(encoding="utf-8")
        block = source.split("def probe_verdict(")[1].split("\ndef ")[0]
        self.assertNotIn("node_isolation", block)


class StartupCensusTests(unittest.TestCase):
    """The timeline, not the number: which phase put memory on a card nobody leased.

    ``startup.json`` keeps only the latest phase, and the latest phase is never the answer
    -- memory already held before the engine is constructed is a library import, the same
    memory appearing during construction is the engine.  Opposite fixes, one number.
    """

    def test_this_process_and_its_children_are_attributed_separately(self):
        mine, child = os.getpid(), os.getpid() + 1
        apps = (f"GPU-aaa, {mine}, 500\n"
                f"GPU-aaa, {child}, 40\n"
                f"GPU-bbb, {mine}, 10\n"
                "GPU-ccc, 999999, 999\n")
        with mock.patch("autosim.research.devices.run_text", lambda command, **kw: apps), \
             mock.patch("autosim.research.accounting.describe_process",
                        lambda pid, **kw: {"parent": mine if pid == child else 999}):
            census = evaluation._own_memory_mib()
        self.assertEqual(census["own"], {"GPU-aaa": 500, "GPU-bbb": 10})
        self.assertEqual(census["children"], {"GPU-aaa": 40})

    def test_a_failing_query_is_recorded_rather_than_raised(self):
        with mock.patch("autosim.research.devices.run_text", lambda command, **kw: "__error__: rc=9"):
            self.assertIn("error", evaluation._own_memory_mib())

    def test_the_census_is_appended_so_the_timeline_survives(self):
        from autosim.research.runtime import startup_census

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with mock.patch.object(evaluation, "_own_memory_mib",
                                  lambda: {"own": {"GPU-aaa": 7}, "children": {}}):
                evaluation._ALIGNMENT.clear()
                evaluation.startup_phase(output, "environment_constructing", policy_has_acted=False)
                evaluation.startup_phase(output, "environment_ready", policy_has_acted=False)
            rows = startup_census(Path(tmp))
            self.assertEqual([row["phase"] for row in rows],
                             ["environment_constructing", "environment_ready"])
            self.assertEqual(rows[0]["memory"]["own"], {"GPU-aaa": 7})
            self.assertEqual(json.loads((Path(tmp) / "startup.json").read_text())["phase"],
                             "environment_ready")

    def test_a_malformed_census_line_is_skipped_rather_than_raised(self):
        from autosim.research.runtime import startup_census

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "startup_census.jsonl").write_text('{"phase": "a"}\nnot json\n\n{"phase": "b"}\n')
            self.assertEqual([row["phase"] for row in startup_census(Path(tmp))], ["a", "b"])

    def test_a_missing_census_is_an_empty_timeline(self):
        from autosim.research.runtime import startup_census

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(startup_census(Path(tmp)), [])

    def test_the_alignment_phase_is_written_before_the_engine_is_built(self):
        """The pre-environment census is what separates an import from the engine."""
        source = Path(evaluation.__file__).read_text(encoding="utf-8")
        align = source.split("def _select_cuda(")[1].split("\ndef ")[0]
        self.assertIn('startup_phase(output, "devices_aligned"', align)

    def test_the_import_rows_bracket_every_module_scope_import(self):
        """One row before anything official loads, one after: the import gets a when."""
        source = Path(evaluation.__file__).read_text(encoding="utf-8")
        main = source.split("def main() -> None:")[1]
        start = main.index('startup_phase(output, "main_start"')
        loaded = main.index('startup_phase(output, "official_imported"')
        exec_module = main.index("module_spec.loader.exec_module(official)")
        self.assertLess(start, exec_module)
        self.assertLess(exec_module, loaded)


class ConcurrentSamplerTests(unittest.TestCase):
    """The 2026-09-13 ladder's concurrent arms reported zero growth on every card.

    That is not what a failed overlap looks like -- it is what *no sampling* looks like: the
    two episodes completed, the shared sampler was created and read once for its baseline,
    and it was never entered, so "peak" after the join was the baseline again.  The check
    then said two devices never ran at the same time, which the evidence did not support.
    """

    def test_a_sampler_that_is_never_entered_reports_the_baseline_back(self):
        from autosim.research.accounting import UtilizationSampler

        readings = iter(["GPU-aaa, 0, 1\n", "GPU-aaa, 0, 4096\n"])
        sampler = UtilizationSampler(interval_seconds=0.01,
                                     runner=lambda command, **kw: next(readings, "GPU-aaa, 0, 4096\n"))
        sampler._sample_once()
        baseline = sampler.summary()
        self.assertEqual(baseline["GPU-aaa"]["peak_memory_mib"], 1)
        self.assertEqual(sampler.summary()["GPU-aaa"]["peak_memory_mib"], 1)

    def test_entering_the_sampler_is_what_makes_the_overlap_visible(self):
        import time as clock

        from autosim.research.accounting import UtilizationSampler

        state = {"mib": 1}
        sampler = UtilizationSampler(interval_seconds=0.01,
                                     runner=lambda command, **kw: f"GPU-aaa, 0, {state['mib']}\n")
        sampler._sample_once()
        baseline = sampler.summary()["GPU-aaa"]["peak_memory_mib"]
        with sampler:
            state["mib"] = 4096
            clock.sleep(0.2)
        self.assertEqual(baseline, 1)
        self.assertGreater(sampler.summary()["GPU-aaa"]["peak_memory_mib"], 1)

    def test_the_ladder_enters_the_shared_sampler_around_the_overlap_window(self):
        source = Path(device_probe.__file__).read_text(encoding="utf-8")
        block = source.split('if "concurrent" in wanted:')[1].split("# The two diagnostic arms")[0]
        self.assertIn("with sampler:", block)
        self.assertLess(block.index("with sampler:"), block.index("thread.start()"))
        self.assertLess(block.index("thread.join()"), block.index("peak = sampler.summary()"))


class RuntimeDeviceBindingTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.runtime = Runtime(self.root, self.root / "out", repo_path=self.root,
                               eval_repo_path=self.root, python_path=Path("/usr/bin/python"))
        self.selection = SimDeviceSelection("pinned_index", 2, "GPU-ccc", 2, 0, "2", "hybrid", {})

    def test_the_legacy_path_passes_no_device_flag_at_all(self):
        self.assertEqual(self.runtime.collector_device_flags(), [])
        self.assertEqual(self.runtime.evaluation_device_flags(), [])
        self.assertEqual(self.runtime.device_metadata(), {})
        environment = self.runtime.environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn(DEFAULT_DEVICE_ENV, environment,
                         "the legacy path has nothing to align: ordinal 0 is the leased card")

    def test_a_device_plan_tells_every_child_which_card_is_ordinal_zero(self):
        """A plan is what makes "the default device" ambiguous, so a plan is what aligns it.

        The evaluator, the collector and the policy child each read this variable before they
        touch CUDA (see ``devices.align_process_defaults``); the trainer gets it through its
        narrowed selection, where ordinal 0 means the leased card by construction.
        """
        identity = SimDeviceSelection("identity", 3, "GPU-ddd", 3, 3, None, "hybrid", {})
        bound = self.runtime.for_job(identity, job="eval_1", output=self.root / "job1")
        self.assertEqual(bound.environment()[DEFAULT_DEVICE_ENV], "3")
        # The trainer is bound to one visible card, so its ordinal 0 -- and its variable -- is
        # the leased card by construction rather than by alignment.
        self.runtime.selection = identity
        environment = self.runtime.environment(selection=self.runtime.training_selection())
        self.assertEqual((environment[DEFAULT_DEVICE_ENV], environment["CUDA_VISIBLE_DEVICES"]),
                         ("0", "3"))

    def test_each_subprocess_gets_the_flags_its_own_cli_defines(self):
        """The collector and the evaluator spell the engine index differently.

        Asserting the strings against the two argparsers rather than against each other is
        the point: the first multi-GPU probe died on ``--gpu_id`` reaching an evaluator that
        only knows ``--device-gpu-id``.
        """
        bound = self.runtime.for_job(self.selection, job="eval_1")
        self.assertEqual(bound.collector_device_flags(),
                         ["--gpu_id", "2", "--renderer", "hybrid"])
        self.assertEqual(bound.evaluation_device_flags(),
                         ["--device-gpu-id", "2", "--renderer", "hybrid",
                          "--device-torch-index", "0"])
        # And against the CLIs themselves, read from source: a unit test that only compares
        # our two spellings to each other would have passed the day the probe died.
        evaluator_source = (Path(__file__).parents[1] / "autosim/research/evaluation.py").read_text()
        for flag in bound.evaluation_device_flags()[::2]:
            self.assertIn(f'"{flag}"', evaluator_source)
        # Both accepted layouts: next to the platform (outer) or inside it (inner root).
        candidates = [root / "EmbodiChain/embodichain/lab/gym/utils/gym_utils.py"
                      for root in (Path(__file__).parents[2], Path(__file__).parents[3])]
        collector_source = next((path for path in candidates if path.is_file()), None)
        if collector_source is None:
            self.skipTest("the EmbodiChain checkout defining the collector CLI is absent")
        for flag in bound.collector_device_flags()[::2]:
            self.assertIn(f'"{flag}"', collector_source.read_text())
        metadata = bound.device_metadata()
        self.assertEqual((metadata["device_uuid"], metadata["device_index"],
                          metadata["device_torch_index"], metadata["job"]),
                         ("GPU-ccc", 2, 0, "eval_1"))
        self.assertNotIn("renderer", metadata["device_mode"])

    def test_binding_a_job_never_mutates_the_shared_runtime(self):
        self.runtime.selection = self.selection
        bound = self.runtime.for_job(self.selection, job="eval_1", output=self.root / "job1")
        self.assertIsNone(self.runtime.job)
        self.assertEqual(bound.output, self.root / "job1")
        self.assertIs(self.runtime.selection, self.selection)   # parent untouched
        self.assertEqual(self.runtime.selection.extra_env, {})
        self.assertEqual((bound.selection.mode, bound.selection.index,
                          bound.selection.vulkan_gpu_id, bound.selection.torch_index),
                         ("pinned_index", 2, 2, 0))

    def test_per_job_caches_are_isolated_and_created(self):
        bound = self.runtime.for_job(self.selection, job="eval_1", output=self.root / "job1")
        other = self.runtime.for_job(self.selection, job="eval_2", output=self.root / "job2")
        self.assertNotEqual(bound.selection.extra_env["XDG_CACHE_HOME"],
                            other.selection.extra_env["XDG_CACHE_HOME"])
        for path in bound.selection.extra_env.values():
            self.assertTrue(Path(path).is_dir(), path)
        environment = bound.environment()
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(environment["TMPDIR"], bound.selection.extra_env["TMPDIR"])

    def test_identity_mode_keeps_the_whole_visible_set(self):
        """Identity addressing is only correct *because* nothing renumbers the cards.

        Narrowing an identity selection to one visible card makes every index above 0 an
        invalid CUDA ordinal -- measured on the 5090 pool: ``CUDA_VISIBLE_DEVICES=3`` with
        ``gpu_id=3`` aborts in ``OptixDevice.cpp`` (``cuDeviceGet(&m_cudaDevice, 3)`` ->
        ``CUDA_ERROR_INVALID_DEVICE``, "Available devices: 0-0"), which is why the engine
        index and the process index agreeing per card is the whole contract.
        """
        identity = SimDeviceSelection("identity", 2, "GPU-ccc", 2, 2, None, "hybrid", {})
        bound = self.runtime.for_job(identity, job="eval_1", output=self.root / "job1")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}):
            self.assertEqual(bound.environment()["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            environment = bound.environment()
        self.assertNotEqual(environment.get("CUDA_VISIBLE_DEVICES"), "0",
                            "an identity selection must never narrow the visible set")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", environment,
                         "an unset outer value stays unset; an empty one is not exported")
        self.assertEqual(bound.evaluation_device_flags(),
                         ["--device-gpu-id", "2", "--renderer", "hybrid",
                          "--device-torch-index", "2"])

    def test_a_selection_without_a_job_still_contributes_device_metadata(self):
        bound = self.runtime.for_job(self.selection, job="")
        self.assertNotIn("job", bound.device_metadata())

    def test_training_is_narrowed_to_one_card_so_its_ordinal_stays_honest(self):
        """``--device cuda`` is ordinal 0 of the *trainer's* view, not of the node.

        Under identity addressing the trainer's view is the whole node, so a job leased to
        card 3 would train on card 0 -- the same split the engine had, one space over.  The
        training view is therefore narrowed to the leased card, where ordinal 0 means the
        leased card in both modes (and training can never touch a neighbour's memory).
        """
        identity = SimDeviceSelection("identity", 3, "GPU-ddd", 3, 3, None, "hybrid", {})
        self.runtime.selection = identity
        training = self.runtime.training_selection()
        self.assertEqual((training.mode, training.index, training.vulkan_gpu_id,
                          training.torch_index, training.cuda_visible),
                         ("identity", 3, 3, 0, "3"))
        self.assertEqual(training.uuid, identity.uuid)      # the same card, not a new one
        self.assertEqual(self.runtime.environment(selection=training)["CUDA_VISIBLE_DEVICES"],
                         "3")
        self.assertIsNone(identity.cuda_visible, "the shared selection is left untouched")

    def test_the_three_environment_building_entry_points_align_their_default_device(self):
        """Dropping any of these three calls silently restores the cross-card residual.

        Nothing in a receipt pointed at a single owner of the second probe's 1030 MiB on card
        0, so the rule is enforced structurally instead: every process that builds an
        environment or loads a policy under a plan -- the evaluator shim, the collection
        worker, the policy child -- asks ``devices`` for the plan's ordinal before it touches
        CUDA, and only when a plan set one.
        """
        research = Path(__file__).parents[1] / "autosim/research"
        for name in ("evaluation.py", "collection_worker.py", "policy_rpc.py"):
            source = (research / name).read_text()
            self.assertIn("align_process_defaults", source, name)
            self.assertIn("default_device_index", source, name)

    def test_a_pinned_selection_reaches_training_unchanged(self):
        self.runtime.selection = self.selection
        self.assertEqual(self.runtime.training_selection(), self.selection)

    def test_an_unbound_runtime_trains_on_the_legacy_device(self):
        self.assertIsNone(self.runtime.training_selection())


def four_devices():
    return [{"index": index, "uuid": f"GPU-{index}"} for index in range(4)]


def judged(name, index, *, passed=True, abort=None, checks=None):
    row = {"name": name, "device_index": index,
           "judged": {"passed": passed, "own_device_uuid": f"GPU-{index}",
                      "checks": checks or {"completed": passed, "real_simulation": passed,
                                           "one_episode": passed, "no_worker_failure": True,
                                           "own_device_grew": passed,
                                           "others_stayed_idle": passed}}}
    if abort is not None:
        row["abort"] = abort
    return row


ORDINAL_ABORT = {"kind": "invalid_device_ordinal", "cards_touched_before_abort": [],
                 "evidence": "Invalid device ID: 1. Available devices: 0-0"}
SPLIT_ABORT = {"kind": "aborted_after_a_card_started", "cards_touched_before_abort": ["GPU-3"],
               "evidence": "1 card(s) had already started the engine"}


def diagnostics_ok():
    return [{"name": "C", "status": "failed", "startup_receipt": {
                "status": "failed", "returncode": -6,
                "startup_phase": "environment_constructing",
                "initializations_recorded": False}, "abort": SPLIT_ABORT},
            {"name": "D", "prediction_met": True, "returncode": 0}]


class ProbeVerdictTests(unittest.TestCase):
    """Which mode a node verified, on which devices -- and what a failure is *called*.

    The ladder runs C (negative control), D (cheap diagnostic), B_i (preferred mode, every
    allocated card), A_i (fallback, every card it must cover) and the concurrent pair.
    """

    def verdict(self, *arms, mode="pinned_index"):
        return device_probe.probe_verdict(arms=list(arms), allocation=four_devices(),
                                          requested_mode=mode)

    def concurrent(self, *a):
        return [judged("concurrent_a", 3, passed=a[0] if a else True),
                judged("concurrent_b", 2, passed=a[1] if len(a) > 1 else True)]

    def test_the_fallback_wins_when_the_preferred_mode_only_works_on_card_zero(self):
        """The measured shape of this cluster: index 0 passes, indices 1+ die at cuDeviceGet."""
        arms = diagnostics_ok() + [judged("B_0", 0)] + \
            [judged(f"B_{i}", i, passed=False, abort=ORDINAL_ABORT) for i in (1, 2, 3)] + \
            [judged(f"A_{i}", i) for i in range(4)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertEqual(verdict["selection_mode"], "identity")
        self.assertEqual(verdict["winner"], "fallback")
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["verified_devices"], ["GPU-0", "GPU-1", "GPU-2", "GPU-3"])
        self.assertTrue(verdict["verified_every_allocated_device"])
        self.assertEqual(verdict["model_discrepancies"], [])
        self.assertEqual(len(verdict["mode_limitations"]), 1)
        self.assertIn("only valid on physical index 0", verdict["mode_limitations"][0])
        self.assertEqual(verdict["devices_verified_under_preferred_mode"]["1"]["abort"]["kind"],
                         "invalid_device_ordinal")

    def test_the_preferred_mode_wins_when_it_covers_every_card(self):
        arms = diagnostics_ok() + [judged(f"B_{i}", i) for i in range(4)] + \
            [judged("A_3", 3)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertEqual((verdict["winner"], verdict["selection_mode"], verdict["passed"]),
                         ("preferred", "pinned_index", True))
        self.assertEqual(verdict["mode_limitations"], [])

    def test_a_mode_that_missed_one_card_is_not_a_capability(self):
        arms = diagnostics_ok() + [judged("B_0", 0)] + \
            [judged(f"B_{i}", i, passed=False, abort=ORDINAL_ABORT) for i in (1, 2, 3)] + \
            [judged(f"A_{i}", i, passed=i != 2) for i in range(4)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertIsNone(verdict["selection_mode"])
        self.assertFalse(verdict["passed"])
        self.assertIn("no_addressing_mode_verified_every_device",
                      verdict["model_discrepancies"])

    def test_a_mode_that_never_ran_on_some_card_is_not_verified_for_it(self):
        """Only devices that ran an episode under the winning mode may be called usable."""
        arms = diagnostics_ok() + [judged("B_0", 0)] + [judged("A_3", 3)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertEqual((verdict["selection_mode"], verdict["passed"]), ("pinned_index", False))
        self.assertIn("the_winning_mode_did_not_verify_every_allocated_device",
                      verdict["model_discrepancies"])

    def test_a_preferred_failure_with_no_named_cause_is_a_discrepancy(self):
        arms = diagnostics_ok() + [judged("B_0", 0),
                                   judged("B_1", 1, passed=False)] + \
            [judged(f"A_{i}", i) for i in range(4)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertEqual(verdict["selection_mode"], "identity")
        self.assertIn("preferred_mode_failed_without_a_named_cause",
                      verdict["model_discrepancies"])

    def test_concurrency_is_part_of_the_verdict_not_a_footnote(self):
        arms = diagnostics_ok() + [judged(f"B_{i}", i) for i in range(4)] + \
            [judged("A_3", 3)] + self.concurrent(True, False)
        verdict = self.verdict(*arms)
        self.assertFalse(verdict["passed"])
        self.assertIn("two_devices_never_ran_an_episode_at_the_same_time",
                      verdict["model_discrepancies"])

    def test_a_negative_control_that_did_not_abort_blocks_the_receipt(self):
        negative = {"name": "C", "status": "completed", "startup_receipt": {}}
        diagnostic = {"name": "D", "prediction_met": True, "returncode": 0}
        arms = [negative, diagnostic] + [judged(f"B_{i}", i) for i in range(4)] + \
            [judged("A_3", 3)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertFalse(verdict["passed"])
        self.assertIn("negative_control_did_not_reproduce_the_abort",
                      verdict["model_discrepancies"])

    def test_a_negative_control_that_died_before_reaching_a_card_is_inconclusive(self):
        """The second probe's C arm died like the ordinal arms did -- same rc, same phase.

        Accepting that as "reproduced the historical abort" is how a ladder ends up citing a
        crash it never observed, so a control that never reached a card is called inconclusive
        and blocks the receipt instead.
        """
        negative = {"name": "C", "status": "failed", "startup_receipt": {
                        "status": "failed", "returncode": -6,
                        "startup_phase": "environment_constructing",
                        "initializations_recorded": False},
                    "abort": {"kind": "aborted_before_any_card_moved"}}
        diagnostic = {"name": "D", "prediction_met": True, "returncode": 0}
        arms = [negative, diagnostic] + [judged(f"B_{i}", i) for i in range(4)] + \
            [judged("A_3", 3)] + self.concurrent()
        verdict = self.verdict(*arms)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["negative_control_reproduced_abort"], False)
        self.assertIn("negative_control_aborted_before_reaching_a_card",
                      verdict["model_discrepancies"])


class CrossCardResidualTests(unittest.TestCase):
    """A card nobody leased that moved anyway is reported with a *name*, not a shrug.

    The second probe measured 1030 MiB on card 0 while an identity arm ran on card 3 and
    recorded no process, so the finding could not be acted on.  Whether that memory is a
    library defaulting to device 0 (bounded, idle, the arm's own process) or somebody else's
    work is the difference between "documented" and "unsafe", and only the holder says which.
    """

    GROWTHS = {"GPU-3": {"growth_mib": 4606, "mean_utilization_pct": 5.9,
                         "processes": {"4242": {"mib": 4606, "cmd": "python -m eval", "parent": 1}}},
               "GPU-0": {"growth_mib": 1030, "mean_utilization_pct": 0.0,
                         "processes": {"4242": {"mib": 1030, "cmd": "python -m eval", "parent": 1}}},
               "GPU-1": {"growth_mib": 3, "mean_utilization_pct": 0.0, "processes": {}}}

    def test_only_devices_over_the_ceiling_are_reported_and_each_names_its_holders(self):
        rows = device_probe.foreign_holders(self.GROWTHS, ["GPU-0", "GPU-1"])
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["device_uuid"], rows[0]["growth_mib"]), ("GPU-0", 1030))
        self.assertEqual(rows[0]["mean_utilization_pct"], 0.0)
        self.assertEqual(rows[0]["holders"], [{"pid": "4242", "mib": 1030,
                                                "cmd": "python -m eval", "parent": 1}])

    def test_the_verdict_carries_the_residual_and_the_pid_that_ran_the_arm(self):
        arm = judged("A_3", 3)
        arm["own_pid"] = 4242
        arm["judged"]["foreign_holders"] = device_probe.foreign_holders(self.GROWTHS,
                                                                        ["GPU-0", "GPU-1"])
        verdict = device_probe.probe_verdict(arms=[arm], allocation=four_devices(),
                                             requested_mode="identity")
        row = verdict["cross_card_residual"][0]
        self.assertEqual((row["arm"], row["device_index"], row["own_pid"], row["growth_mib"]),
                         ("A_3", 3, 4242, 1030))
        self.assertEqual(row["holders"][0]["pid"], "4242")
        # A residual is evidence, not a veto on its own: the device that ran the episode and
        # the other devices' own checks still decide, which is what the reason list shows.
        self.assertEqual(verdict["devices_verified_under_winning_mode"]["3"]["own_pid"], 4242)

    def test_judge_separates_the_ceiling_from_the_attribution(self):
        """The check stays strict; what changes is that failing it now has a witness."""
        peak = {uuid: {"peak_memory_mib": row["growth_mib"] + 1, "mean_utilization_pct": 0.0,
                       "samples": 5, "processes": row["processes"]}
                for uuid, row in self.GROWTHS.items()}
        baseline = {uuid: {"peak_memory_mib": 1} for uuid in self.GROWTHS}
        result = device_probe.judge({"status": "completed", "execution_mode": "real_simulation",
                                     "episode_count": 1},
                                    own_uuid="GPU-3", allocation=list(self.GROWTHS),
                                    baseline=baseline, peak=peak)
        self.assertFalse(result["checks"]["others_stayed_idle"])
        self.assertEqual([row["device_uuid"] for row in result["foreign_holders"]], ["GPU-0"])

    def test_a_phase_receipt_names_its_process_so_a_holder_can_be_matched_to_it(self):
        """``own_pid`` is read from the arm's phase receipt; without it a holder has no owner."""
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            evaluation.startup_phase(output, "environment_constructing", policy_has_acted=False)
            receipt = json.loads((output / "startup.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["pid"], os.getpid())
            self.assertEqual(receipt["phase"], "environment_constructing")
            self.assertEqual(receipt["default_device"], {},
                             "a process that never selected a device reports no alignment")


#: Card 0's residual, copied from the run of record (job ``pt-2xib2ilt``, 2026-09-14, arms
#: ``A_*/startup_census.jsonl`` + ``device_probe.json``): every evaluator process parks one
#: CUDA context on card 0 while ``make_env_from_configs`` builds the environment, and the
#: alignment's own context sits on the leased card from ``devices_aligned`` onward.
ZERO_CARD = "GPU-8be632a8-d7c8-bc44-b7f3-7b226e1cbdff"
LEASED_CARD = "GPU-2f24b1f9-73ce-1599-e1be-742cc3c43f8e"
RUN_ROOT = "/data/AutoResearch/AutoSimSOTA/AutoSimSOTA"
#: Truncated at 300 chars by ``accounting.describe_process``, exactly as the receipt stores it
#: -- which is why the checkout check looks for the *prefix* and not for the arm's own output.
WORKER_CMD = ("/data/AutoResearch/AutoSimSOTA/AutoSimSOTA/.venv/bin/python -m "
              "autosim.research.evaluation --task click_bell --checkpoint "
              "/data/AutoResearch/AutoSimSOTA/AutoSimSOTA/RoboSynChallenge/checkpoints/"
              "ACT_sim_click_bell --output /data/AutoResearch/AutoSimSOTA/AutoSimSOTA/"
              "autoresearch_runs/RoboSynChallenge/")
CENSUS_PHASES = ("main_start", "official_imported", "devices_aligned",
                 "environment_constructing", "environment_ready", "evaluation_reset_started")


def measured_census(*, zero_card_at=("environment_ready", "evaluation_reset_started"),
                    leased_card_at=("devices_aligned", "environment_constructing",
                                    "environment_ready", "evaluation_reset_started")) -> list[dict]:
    """The per-phase timeline an arm writes, with only the *presence* of each card varying."""
    rows = []
    for phase in CENSUS_PHASES:
        own = {}
        if phase in leased_card_at:
            own[LEASED_CARD] = 532 if phase in ("devices_aligned",
                                                "environment_constructing") else 3577
        if phase in zero_card_at:
            own[ZERO_CARD] = 506
        rows.append({"phase": phase, "memory": {"own": own, "children": {}}})
    return rows


def residual(*, growth=525, utilization=0.0, uuid=ZERO_CARD, holders=None) -> dict:
    """A residual row as ``foreign_holders`` builds it inside ``judge``."""
    row = {"growth_mib": growth, "mean_utilization_pct": utilization,
           "processes": holders if holders is not None else
           {"3673": {"mib": 506, "cmd": WORKER_CMD, "parent": 362}}}
    return device_probe.foreign_holders({uuid: row}, [uuid])[0]


def worker(pid="3673", mib=506, cmd=WORKER_CMD) -> dict:
    return {pid: {"mib": mib, "cmd": cmd, "parent": 362}}


class EngineContextExceptionTests(unittest.TestCase):
    """The one residual that is accepted, and everything that must still be refused.

    ``others_stayed_idle`` has to stop calling a *measured, bounded, attributable, idle* CUDA
    context "somebody else's work" -- otherwise the 4-GPU plan is refused forever by a fact
    about the engine that no available lever moves.  What it must not do is accept the shape
    blindly: the holder, the size, the utilization and the timeline all have to agree, and
    each of them is a falsification point below.
    """

    def accepted(self, row=None, *, census=None, run_root=RUN_ROOT):
        return device_probe.engine_own_context(
            row if row is not None else residual(),
            census=measured_census() if census is None else census, run_root=run_root)

    def test_the_measured_shape_is_accepted(self):
        result = self.accepted()
        self.assertTrue(result["accepted"], result["reason"])
        self.assertEqual(result["holder_pids"], ["3673"])

    def test_two_concurrent_workers_are_two_contexts_not_one_leak(self):
        """The pair doubles the *card's* growth to 1046 MiB; the bound is per holder, so the
        concurrency arm is not refused for containing its sibling's context as well."""
        row = residual(growth=1046, holders={**worker("6430"), **worker("6431")})
        self.assertTrue(self.accepted(row)["accepted"])

    def test_a_single_holder_above_the_bound_is_not_a_context(self):
        result = self.accepted(residual(holders=worker(mib=1200)))
        self.assertFalse(result["accepted"])
        self.assertIn("single-context bound", result["reason"])

    def test_a_busy_card_is_computing_not_only_holding_a_context(self):
        self.assertFalse(self.accepted(residual(utilization=5.9))["accepted"])

    def test_a_holder_that_is_not_this_projects_worker_is_never_accepted(self):
        result = self.accepted(residual(holders=worker(cmd="/usr/bin/python3 -m torchrun "
                                                           "--nproc_per_node 4 train.py")))
        self.assertFalse(result["accepted"])
        self.assertIn("not this project's worker", result["reason"])

    def test_a_worker_from_another_checkout_is_somebody_elses_run(self):
        """A job that never installed the platform root we are running from is not our job."""
        other = WORKER_CMD.replace(RUN_ROOT, "/data/other/AutoSimSOTA")
        result = self.accepted(residual(holders=worker(cmd=other)))
        self.assertFalse(result["accepted"])
        self.assertIn("does not belong to this checkout", result["reason"])

    def test_a_growth_no_process_could_be_named_for_is_a_failure(self):
        result = self.accepted(residual(holders={}))
        self.assertFalse(result["accepted"])
        self.assertIn("no process", result["reason"])

    def test_a_card_already_holding_memory_before_the_engine_is_built_is_not_accepted(self):
        result = self.accepted(census=measured_census(zero_card_at=("official_imported",
                                                                    "environment_ready")))
        self.assertFalse(result["accepted"])
        self.assertIn("already holding memory before", result["reason"])

    def test_without_a_timeline_nothing_is_accepted(self):
        for census in ([], measured_census(zero_card_at=()), [{"phase": "devices_aligned",
                                                               "memory": {}}]):
            with self.subTest(census=census):
                self.assertFalse(self.accepted(census=census)["accepted"],
                                 "an unplaceable growth is not a pass")

    def test_a_child_holding_the_card_counts_as_this_arms_memory(self):
        census = measured_census(zero_card_at=())
        for row in census:
            if row["phase"] == "environment_ready":
                row["memory"]["children"] = {ZERO_CARD: 506}
        self.assertTrue(self.accepted(census=census)["accepted"])

    def test_a_card_held_by_a_stranger_at_an_early_phase_does_not_appear_in_our_census(self):
        """The census only ever names this process and its children, so it cannot be used to
        prove anything about a foreign holder -- hence the cmd check above."""
        self.assertEqual(device_probe.appeared_only_while_the_engine_was_built(
            "GPU-somebody-else", measured_census()), False)

    def test_acceptance_is_recorded_per_card_and_never_silences_the_check(self):
        """The verdict keeps the residual *and* the acceptance, so the receipt still shows the
        megabytes: accepting is a documented exception, not a clean measurement."""
        peaks = {LEASED_CARD: {"peak_memory_mib": 5000, "mean_utilization_pct": 40.0,
                               "samples": 5, "processes": worker("1000", 4000)},
                 ZERO_CARD: {"peak_memory_mib": 526, "mean_utilization_pct": 0.0,
                             "samples": 5, "processes": worker()}}
        baseline = {LEASED_CARD: {"peak_memory_mib": 1}, ZERO_CARD: {"peak_memory_mib": 1}}
        result = device_probe.judge({"status": "completed", "execution_mode": "real_simulation",
                                     "episode_count": 1},
                                    own_uuid=LEASED_CARD, allocation=[LEASED_CARD, ZERO_CARD],
                                    baseline=baseline, peak=peaks,
                                    census=measured_census(), run_root=RUN_ROOT)
        self.assertTrue(result["checks"]["others_stayed_idle"], result["foreign_holders"])
        self.assertTrue(result["passed"])
        self.assertEqual([row["device_uuid"] for row in result["accepted_residuals"]],
                         [ZERO_CARD])
        self.assertEqual(result["foreign_holders"][0]["growth_mib"], 525)

    def test_a_foreign_holder_alongside_an_accepted_one_still_fails_the_arm(self):
        peaks = {LEASED_CARD: {"peak_memory_mib": 5000, "mean_utilization_pct": 40.0,
                               "samples": 5, "processes": worker("1000", 4000)},
                 ZERO_CARD: {"peak_memory_mib": 1526, "mean_utilization_pct": 0.0, "samples": 5,
                             "processes": {**worker(), **worker("9999", 1000,
                                                                "/usr/bin/python3 train.py")}}}
        baseline = {uuid: {"peak_memory_mib": 1} for uuid in peaks}
        result = device_probe.judge({"status": "completed", "execution_mode": "real_simulation",
                                     "episode_count": 1},
                                    own_uuid=LEASED_CARD, allocation=list(peaks),
                                    baseline=baseline, peak=peaks,
                                    census=measured_census(), run_root=RUN_ROOT)
        self.assertFalse(result["checks"]["others_stayed_idle"])
        self.assertFalse(result["passed"])

    def test_the_verdict_carries_the_standing_of_every_residual(self):
        arm = {"name": "A_1", "device_index": 1, "own_pid": 3673,
               "judged": {"passed": True, "own_device_uuid": LEASED_CARD,
                          "checks": {"others_stayed_idle": True},
                          "foreign_holders": [
                   {**residual(), **self.accepted()},
                   {**residual(uuid=LEASED_CARD), "accepted": False, "reason": "no process "
                    "could be named for the growth"}]}}
        verdict = device_probe.probe_verdict(arms=[arm], allocation=[],
                                             requested_mode="identity")
        self.assertEqual([row["accepted"] for row in verdict["cross_card_residual"]],
                         [True, False])
        self.assertEqual(verdict["cross_card_residual"][1]["reason"],
                         "no process could be named for the growth")

    def test_a_run_without_a_timeline_is_refused_by_the_judge_not_by_the_caller(self):
        """``record`` passes whatever census the arm produced, including nothing at all."""
        result = device_probe.judge({"status": "completed", "execution_mode": "real_simulation",
                                    "episode_count": 1},
                                    own_uuid=LEASED_CARD, allocation=[LEASED_CARD, ZERO_CARD],
                                    baseline={uuid: {"peak_memory_mib": 1}
                                              for uuid in (LEASED_CARD, ZERO_CARD)},
                                    peak={LEASED_CARD: {"peak_memory_mib": 5000,
                                                        "mean_utilization_pct": 40.0, "samples": 5,
                                                        "processes": worker("1000", 4000)},
                                          ZERO_CARD: {"peak_memory_mib": 526,
                                                      "mean_utilization_pct": 0.0, "samples": 5,
                                                      "processes": worker()}},
                                    census=[], run_root=RUN_ROOT)
        self.assertFalse(result["checks"]["others_stayed_idle"])


class AbortClassificationTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def classify(self, text, growths=None):
        (self.root / "process").mkdir(exist_ok=True)
        (self.root / "process/stdout.log").write_text(text, encoding="utf-8")
        return device_probe.classify_abort(output=self.root, growths=growths or {})

    def test_the_driver_ordinal_abort_is_named_by_the_engines_own_message(self):
        result = self.classify("Using CUDA device: NVIDIA GeForce RTX 5090 | ID: 0\n"
                               "Invalid device ID: 1. Available devices: 0-0\n"
                               "ERROR: OptixDevice.cpp(132): cuDeviceGet(&m_cudaDevice, "
                               "m_ordinal) (101) CUDA_ERROR_INVALID_DEVICE: invalid device "
                               "ordinal\n", {"GPU-1": {"growth_mib": 3}})
        self.assertEqual(result["kind"], "invalid_device_ordinal")
        self.assertIn("CUDA_ERROR_INVALID_DEVICE", result["evidence"])
        self.assertEqual(result["cards_touched_before_abort"], [])

    def test_a_death_after_a_card_started_is_not_a_death_before_one(self):
        after = self.classify("...\n[EmbodiChain INFO] Physics successfully bound\n",
                              {"GPU-3": {"growth_mib": 1633}, "GPU-0": {"growth_mib": 62}})
        self.assertEqual(after["kind"], "aborted_after_a_card_started")
        self.assertEqual(after["cards_touched_before_abort"], ["GPU-3"])
        before = self.classify("...\n", {"GPU-1": {"growth_mib": 3}})
        self.assertEqual(before["kind"], "aborted_before_any_card_moved")
        self.assertEqual(before["cards_touched_before_abort"], [])

    def test_a_missing_log_still_classifies_from_what_moved(self):
        result = device_probe.classify_abort(output=self.root / "absent",
                                             growths={"GPU-2": {"growth_mib": 900}})
        self.assertEqual(result["kind"], "aborted_after_a_card_started")


if __name__ == "__main__":
    unittest.main()
