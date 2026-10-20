"""Exercise migration decisions without contacting hosts or allocating a GPU."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import migrate_when_idle as migration


class MigrationTests(unittest.TestCase):
    def test_low_utilization_does_not_make_occupied_gpu_idle(self):
        gpu = dict(pids=[123], memory_mib=20, utilization=0)
        self.assertFalse(migration.idle(gpu))
        gpu["pids"] = []
        self.assertTrue(migration.idle(gpu))
        gpu["memory_mib"] = 512
        self.assertFalse(migration.idle(gpu))

    def test_origin_starting_prevents_target_launch(self):
        cfg = dict(origin_job_id="123", torch_host="torch")
        with patch.object(migration, "ssh") as ssh, \
             patch.object(migration, "origin_status", return_value=("RUNNING", "None")), \
             patch.object(migration, "remote") as remote, \
             patch.object(migration, "record"):
            self.assertEqual(migration.migrate(cfg, "gpu"), 2)
            remote.assert_not_called()
            self.assertFalse(any(call.args[1][0] == "scancel" for call in ssh.call_args_list))

    def test_rejected_hold_does_not_launch_or_cancel(self):
        cfg = dict(origin_job_id="123", torch_host="torch")
        with patch.object(migration, "ssh", side_effect=RuntimeError("hold rejected")) as ssh, \
             patch.object(migration, "remote") as remote, \
             patch.object(migration, "record"):
            self.assertEqual(migration.migrate(cfg, "gpu"), 2)
            remote.assert_not_called()
            ssh.assert_called_once_with("torch", ["scontrol", "hold", "123"])

    def exercise(self, target, expected):
        with tempfile.TemporaryDirectory() as folder:
            cfg = dict(origin_job_id="123", torch_host="torch", target_host="145",
                       output="/unused", local_dir=folder, interval_seconds=0)
            with patch.object(migration, "ssh") as ssh, \
                 patch.object(migration, "origin_status", return_value=("PENDING", "JobHeldUser")), \
                 patch.object(migration, "remote", side_effect=[{"pid": 9}, target]), \
                 patch.object(migration, "record"):
                self.assertEqual(migration.migrate(cfg, "gpu"), expected)
                return [call.args[1] for call in ssh.call_args_list], Path(folder).exists()

    def test_failed_target_releases_original_job(self):
        calls, _ = self.exercise(dict(runner=dict(state="FAILED"), cells=0), 2)
        self.assertEqual(calls, [["scontrol", "hold", "123"], ["scontrol", "release", "123"]])

    def test_successful_target_cancels_only_pending_original(self):
        calls, _ = self.exercise(dict(runner=dict(state="SUCCEEDED"), cells=66,
                                      summary_complete=True), 0)
        self.assertEqual(calls, [["scontrol", "hold", "123"],
                                ["scancel", "--state=PENDING", "123"]])


if __name__ == "__main__":
    unittest.main()
