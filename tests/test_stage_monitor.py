import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "scripts/monitor_stage.py"
spec = importlib.util.spec_from_file_location("monitor", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MonitorTests(unittest.TestCase):
    def test_scheduler_success_is_not_artifact_success(self):
        jobs = [dict(state="COMPLETED", exit_code="0:0")]
        self.assertEqual(m.outcome(jobs, False), ("INCOMPLETE", True))
        self.assertEqual(m.outcome(jobs, True), ("SUCCESS", True))
        self.assertEqual(m.outcome([dict(state="FAILED", exit_code="1:0")], True),
                         ("FAILED", True))
        self.assertEqual(m.outcome([dict(state="UNKNOWN", exit_code="")], True),
                         ("RUNNING_OR_PENDING", False))
        self.assertEqual(m.parse_accounting("123_0|CANCELLED by 9|0:15|00:03\n")["123_0"]["state"],
                         "CANCELLED")

    def test_ssh_failure_does_not_turn_old_success_into_current_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(local_dir=tmp, host="torch", train_job="123",
                report_job="124", interval=60, max_hours=1, once=True,
                runtime="/runtime", manifest="manifest.json", report_root="/report",
                output_root="/runs", remote_python="python3", remote_script="/monitor.py")
            m.write_json(Path(tmp)/"latest.json", dict(status="SUCCESS", terminal=True))
            error = subprocess.CalledProcessError(255, ["ssh"], stderr="Permission denied")
            with patch.object(m.subprocess, "run", side_effect=error):
                self.assertEqual(m.watch(args), 0)
            value = json.loads((Path(tmp)/"latest.json").read_text())
            self.assertEqual(value["status"], "MONITOR_DISCONNECTED")
            self.assertFalse(value["terminal"])

    def test_partial_metrics_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(experiment_id="experiment", steps=5)
            m.write_json(root/"cfg.json", cfg)
            entry = dict(config="cfg.json", label="model", model="ode_original",
                         steps=5, evaluate_steps=[5], sha256="hash")
            m.write_json(root/"manifest.json", dict(datasets=["data"], entries=[entry],
                                                   run_count=1, eval_checkpoint_count=1))
            folder = root/"runs/experiment/data/ode_original/seed000"
            folder.mkdir(parents=True)
            (folder/"metrics.jsonl").write_text('{"step": 4}\n{"step":')
            (folder/"COMPLETE").write_text("5\n")
            m.write_json(folder/"config.json", cfg)
            report = root/"report"
            m.write_json(report/"stage-summary.json", dict(complete=True))
            m.write_json(report/"diagnostics/data/model-step0000005.json",
                         dict(dataset="data", label="model", step=5, seed=0, config_sha256="hash"))
            args = SimpleNamespace(runtime=str(root), report_root=str(report),
                manifest="manifest.json", output_root=str(root/"runs"), train_job="123", report_job="124")
            returns = [SimpleNamespace(stdout="123_0|COMPLETED|0:0|00:01\n124|COMPLETED|0:0|00:01\n"),
                       SimpleNamespace(stdout="", stderr="")]
            with patch.object(m.subprocess, "run", side_effect=returns):
                value = m.probe(args)
            self.assertEqual(value["status"], "INCOMPLETE")
            self.assertEqual(value["progress"][0]["step"], 4)
            self.assertEqual(value["completed_runs"], 0)


if __name__ == "__main__":
    unittest.main()
