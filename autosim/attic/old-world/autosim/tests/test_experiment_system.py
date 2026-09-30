"""System acceptance uses real CPU child processes, never simulated policy scores."""
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from autosim.research.common import atomic_json, digest, read_json
from autosim.experiment_system.contracts import Capability, TaskContract, assert_compatible, select_workflow
from autosim.experiment_system.executor import Executor, Job, failure_action, lease
from autosim.experiment_system.feedback import Repair, collection_ticket
from autosim.experiment_system.quality import audit_episode


class SystemTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.json"
        atomic_json(self.source, {"version": 1})
        self.sources = {str(self.source): digest(self.source)}
        self.executor = Executor(self.root / "jobs")

    def tearDown(self):
        self.temp.cleanup()

    def job(self, **kwargs):
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{\"status\":\"completed\"}')", "{attempt}/result.json"]
        defaults = dict(name="test", stage="probe", command=command, cwd=str(self.root),
                        outputs={"result.json": "result"}, sources=self.sources, timeout_seconds=5,
                        budget_seconds=10, max_attempts=2)
        defaults.update(kwargs)
        return Job(**defaults)

    def contract(self):
        return TaskContract("fixture", "task", {"state_dim": 1, "cameras": {"rgb": [2, 2, 3]}},
                            {"dimension": 1, "mode": "absolute", "units": ["rad"], "order": ["arm"], "frame": "joint"},
                            {"max_actions": 10, "control_hz": 25, "recorded_hz": 25},
                            {"native_entry": "fixture", "success_authority": "native", "setting": "random"}, self.sources)

    def test_success_and_idempotent_reuse(self):
        first = self.executor.run(self.job())
        self.assertEqual(first["status"], "committed")
        self.assertEqual(first, self.executor.run(self.job()))
        self.assertEqual(len(list((self.root / "jobs/test").glob("attempt_*"))), 1)

    def test_changed_inputs_rejected(self):
        self.executor.register(self.job())
        with self.assertRaises(ValueError):
            self.executor.register(self.job(timeout_seconds=4))

    def test_changed_source_rejected(self):
        job = self.job()
        atomic_json(self.source, {"version": 2})
        with self.assertRaises(ValueError):
            self.executor.run(job)

    def test_committed_output_tamper_rejected(self):
        self.executor.run(self.job())
        atomic_json(self.root / "jobs/test/attempt_001/result.json", {"status": "completed", "altered": True})
        with self.assertRaises(ValueError):
            self.executor.run(self.job())

    def test_exit_zero_without_output_is_not_success(self):
        result = self.executor.run(self.job(command=[sys.executable, "-c", "pass"]))
        self.assertEqual(result["status"], "requires_artifact_audit")

    def test_invalid_json_not_admitted(self):
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{')", "{attempt}/result.json"]
        self.assertEqual(self.executor.run(self.job(command=command))["status"], "requires_artifact_audit")

    def test_missing_dependency_waits_without_attempt(self):
        self.assertEqual(self.executor.run(self.job(dependencies=("upstream",)))["status"], "waiting_dependency")
        self.assertFalse(list((self.root / "jobs/test").glob("attempt_*")))

    def test_job_lock_prevents_duplicate(self):
        self.executor.register(self.job())
        with lease(self.root / "jobs/test/job.lock"):
            self.assertEqual(self.executor.run(self.job())["status"], "waiting_lease")

    def test_evaluation_failure_never_retries(self):
        job = self.job(stage="evaluate", command=[sys.executable, "-c", "raise SystemExit(1)"])
        self.assertEqual(self.executor.run(job)["status"], "quarantine_evaluation")
        self.assertEqual(self.executor.run(job)["status"], "quarantine_evaluation")
        self.assertEqual(len(list((self.root / "jobs/test").glob("attempt_*"))), 1)

    def test_successful_execution_with_zero_policy_score_commits(self):
        command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{\"status\":\"completed\",\"success_rate\":0}')", "{attempt}/result.json"]
        self.assertEqual(self.executor.run(self.job(stage="evaluate", command=command))["status"], "committed")

    def test_timeout_has_bounded_retry_and_cost(self):
        job = self.job(command=[sys.executable, "-c", "import time; time.sleep(5)"], timeout_seconds=1, budget_seconds=4)
        self.assertEqual(self.executor.run(job)["status"], "retry_from_new_attempt")
        self.assertEqual(self.executor.run(job)["status"], "retry_from_new_attempt")
        self.assertEqual(self.executor.run(job)["status"], "budget_exhausted")
        self.assertEqual(self.executor.summary()["jobs"][0]["failed_attempts"], 2)

    def test_unknown_partial_execution_requires_audit(self):
        self.executor.register(self.job())
        (self.root / "jobs/test/attempt_001").mkdir()
        self.assertEqual(self.executor.run(self.job())["status"], "requires_audit_unknown_execution")

    def test_timeout_cleans_independent_session_grandchild(self):
        command = [sys.executable, "-c",
            "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)",
            "{attempt}/grandchild.pid"]
        job = self.job(command=command, timeout_seconds=1)
        self.executor.run(job)
        pid = int((self.root / "jobs/test/attempt_001/grandchild.pid").read_text())
        for _ in range(100):
            stat = Path(f"/proc/{pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.01)
        else:
            self.fail("nested process survived supervisor timeout")

    def test_controller_killed_supervisor_retains_lock_then_recovers(self):
        command = [sys.executable, "-c", "import time,sys; from pathlib import Path; time.sleep(1.5); Path(sys.argv[1]).write_text('{\"status\":\"completed\"}')", "{attempt}/result.json"]
        job = self.job(command=command)
        self.executor.register(job)
        controller = subprocess.Popen([sys.executable, "-c",
            "import sys; from pathlib import Path; from autosim.research.common import read_json; from autosim.experiment_system.executor import Executor,Job; e=Executor(Path(sys.argv[1])); e.run(Job(**read_json(Path(sys.argv[1])/'test/job.json')))", str(self.root / "jobs")])
        start = time.monotonic()
        try:
            while not (self.root / "jobs/test/attempt_001/started.json").exists():
                if time.monotonic() - start > 10:
                    self.fail("supervisor did not start")
                time.sleep(.02)
            controller.kill()
            controller.wait()
            self.assertEqual(self.executor.run(job)["status"], "waiting_lease")
            while not (self.root / "jobs/test/attempt_001/receipt.json").exists():
                if time.monotonic() - start > 10:
                    self.fail("supervisor did not complete")
                time.sleep(.02)
            # Receipt can be visible just before supervisor releases its leases.
            for _ in range(100):
                result = self.executor.run(job)
                if result["status"] != "waiting_lease":
                    break
                time.sleep(.02)
            self.assertEqual(result["status"], "committed")
            self.assertEqual(len(list((self.root / "jobs/test").glob("attempt_*"))), 1)
        finally:
            if controller.poll() is None:
                controller.kill()
                controller.wait()

    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError):
            self.job(outputs={"../outside": "json"}).validate()

    def test_credential_manifest_rejected(self):
        with self.assertRaises(ValueError):
            self.job(environment={"API_TOKEN": "not-a-real-token"}).validate()

    def test_declared_capabilities_not_usable(self):
        c = self.contract()
        cap = Capability("expert_collection", "declared", c.signature, self.sources, "test")
        self.assertFalse(cap.usable(c))

    def test_workflow_is_capability_gated(self):
        c = self.contract()
        caps = [Capability(n, "validated", c.signature, self.sources, "fixture evidence") for n in
                ("semantic_probe", "native_evaluation", "train_reload:act", "official_data")]
        self.assertEqual(select_workflow(c, caps, "act")["mode"], "official_data_only")
        caps.append(Capability("expert_collection", "validated", c.signature, self.sources, "fixture"))
        self.assertTrue(select_workflow(c, caps, "act")["automatic_collection"])

    def test_same_dimension_unit_mismatch_rejected(self):
        c = self.contract()
        observed = asdict(c)
        observed["action"]["units"] = ["degree"]
        with self.assertRaises(ValueError):
            assert_compatible(c, observed)

    def test_stale_capability_rejected(self):
        c = self.contract()
        cap = Capability("expert_collection", "validated", "old", self.sources, "old probe")
        self.assertFalse(cap.usable(c))

    def episode(self):
        return [{"observation.state": [i / 10], "action": [i / 10], "frame_index": i,
                 "episode_index": 0, "timestamp": i / 25} for i in range(5)]

    def audit(self, rows):
        return audit_episode(rows, state_key="observation.state", state_dim=1, action_dim=1, fps=25)

    def test_valid_episode(self):
        self.assertTrue(self.audit(self.episode())["passed"])

    def test_nonfinite_data_rejected(self):
        rows = self.episode()
        rows[2]["action"] = [float("nan")]
        self.assertFalse(self.audit(rows)["passed"])

    def test_timestamp_shift_rejected(self):
        rows = self.episode()
        rows[2]["timestamp"] += .1
        self.assertFalse(self.audit(rows)["passed"])

    def test_frame_gap_rejected(self):
        rows = self.episode()
        rows[2]["frame_index"] = 3
        self.assertFalse(self.audit(rows)["passed"])

    def test_shape_rejected(self):
        rows = self.episode()
        for row in rows:
            row["action"] = [1, 2]
        self.assertFalse(self.audit(rows)["passed"])

    def test_final_data_cannot_generate_ticket(self):
        with self.assertRaises(ValueError):
            collection_ticket({"purpose": "final"}, supported_profiles={"full_random"}, budget=10)

    def test_uncertain_diagnosis_retains_random_coverage(self):
        result = collection_ticket({"purpose": "development", "source": str(self.source),
                                    "failures": [{"confidence": "heuristic", "collection_profile": "targeted_camera"}]},
                                   supported_profiles={"full_random", "targeted_camera"}, budget=10)
        self.assertEqual(result["random_coverage_episodes"], 10)

    def test_repair_requires_regression_evidence(self):
        repair = Repair("retry", "infrastructure", "train", "retry_new_attempt", self.sources, {})
        self.assertFalse(repair.applicable("infrastructure", "train"))

    def test_repair_cannot_change_evaluation(self):
        evidence = self.root / "regression.json"
        atomic_json(evidence, {"passed": True})
        repair = Repair("retry", "infrastructure", "train", "retry_new_attempt", self.sources,
                        {str(evidence): digest(evidence)})
        self.assertTrue(repair.applicable("infrastructure", "train"))
        self.assertFalse(repair.applicable("infrastructure", "evaluate"))

    def test_robotwin_converter_next_state_alignment_and_rgb(self):
        import cv2
        import h5py
        import numpy as np
        import pyarrow.parquet as pq
        from autosim.experiment_system.robotwin_worker import convert
        from autosim.experiment_system.quality import audit_dataset
        raw = self.root / "native.hdf5"
        state = np.arange(5 * 14, dtype=np.float32).reshape(5, 14) / 100
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[:] = [210, 40, 10]
        bits = cv2.imencode(".jpg", rgb)[1].tobytes()
        with h5py.File(raw, "w") as handle:
            for key, values in (("left_arm", state[:, :6]), ("left_gripper", state[:, 6]),
                                ("right_arm", state[:, 7:13]), ("right_gripper", state[:, 13])):
                handle.create_dataset("joint_action/" + key, data=values)
            for camera in ("head_camera", "left_camera", "right_camera"):
                handle.create_dataset(f"observation/{camera}/rgb", data=[bits] * 5, dtype=f"S{len(bits)}")
        destination = convert([str(raw)], self.root / "converted", "fixture")
        parquet = next((destination / "data").rglob("*.parquet"))
        rows = pq.read_table(parquet).to_pylist()
        self.assertEqual(len(rows), 4)
        np.testing.assert_allclose(rows[0]["observation.state"], state[0])
        np.testing.assert_allclose(rows[0]["action"], state[1])
        np.testing.assert_allclose(rows[-1]["action"], state[-1])
        audit = audit_dataset(destination, state_dim=14, action_dim=14)
        self.assertTrue(audit["passed"], audit["errors"])
        import av
        with av.open(str(next((destination / "videos").rglob("*.mp4")))) as container:
            decoded = next(container.decode(video=0)).to_ndarray(format="rgb24")
        self.assertGreater(decoded[..., 0].mean(), decoded[..., 2].mean() + 100)

    def test_native_inference_whitelist(self):
        import numpy as np
        from autosim.experiment_system.native_policy import eval
        captured = {}
        class Model:
            contract = {"cameras": ["head_camera"]}
            def predict(self, batch):
                captured["batch"] = batch
                return {"action": np.zeros((1, 14))}
        class Environment:
            def take_action(self, action):
                captured["action"] = action
        observation = {"joint_action": {"vector": np.zeros(14)},
                       "observation": {"head_camera": {"rgb": np.zeros((2, 2, 3))}},
                       "privileged_target_pose": [1, 2, 3], "seed": 123}
        eval(Environment(), Model(), observation)
        self.assertEqual(set(captured["batch"]), {"robot", "sensor"})
        self.assertEqual(set(captured["batch"]["robot"]), {"qpos"})


if __name__ == "__main__":
    unittest.main()
