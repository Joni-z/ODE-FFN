import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"scripts"))
spec = importlib.util.spec_from_file_location("submit_stage", root/"scripts/submit_stage_monitored.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SubmissionTests(unittest.TestCase):
    def argv(self, folder):
        return ["submit", "--runtime", "/runtime", "--report-root", "/report",
                "--manifest", "manifest.json", "--local-dir", folder]

    def test_monitor_attaches_and_duplicate_receipt_blocks_resubmission(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [SimpleNamespace(stdout=s) for s in ("OK", "", "help", "123\n", "124\n")]
            with patch.object(sys, "argv", self.argv(tmp)), \
                 patch.object(m.subprocess, "run", side_effect=replies) as remote, \
                 patch.object(m.subprocess, "Popen", return_value=SimpleNamespace(pid=999)) as launch:
                m.main()
                self.assertIn("--dependency=afterany:123", remote.call_args_list[-1].args[0][-1])
                self.assertTrue(launch.call_args.kwargs["start_new_session"])
                self.assertIn("123", launch.call_args.args[0])
            receipt = json.loads((Path(tmp)/"submission.json").read_text())
            self.assertEqual(receipt["monitor_pid"], 999)
            with patch.object(sys, "argv", self.argv(tmp)), \
                 patch.object(m.subprocess, "run") as remote:
                with self.assertRaises(FileExistsError):
                    m.main()
                remote.assert_not_called()

    def test_report_submission_failure_retains_training_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            replies = [SimpleNamespace(stdout=s) for s in ("OK", "", "help", "123\n")]
            replies.append(subprocess.CalledProcessError(255, ["ssh"]))
            with patch.object(sys, "argv", self.argv(tmp)), \
                 patch.object(m.subprocess, "run", side_effect=replies), \
                 patch.object(m.subprocess, "Popen") as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    m.main()
                launch.assert_not_called()
            receipt = json.loads((Path(tmp)/"submission.json").read_text())
            self.assertEqual(receipt["training_array_job_id"], "123")
            self.assertTrue(receipt["requires_reconciliation"])


if __name__ == "__main__":
    unittest.main()
