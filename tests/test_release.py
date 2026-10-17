"""Offline tests of data isolation, execution setup, and result aggregation."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sldbench import list_tasks
from tools.summarize_runs import aggregate


class BenchmarkIntegrationTests(unittest.TestCase):
    def test_every_selected_task_has_both_settings(self):
        manifest = json.loads((ROOT / "benchmark/tasks-manifest.json").read_text())
        expected = {t + s for t in list_tasks(selected=True) for s in ("", "_costaware")}
        self.assertEqual(set(manifest["tasks"]), expected)
        for task in list_tasks(selected=True):
            # Both settings are graded against the very same test file.
            p = manifest["files"][f"{task}/tests/data/test.parquet"]
            a = manifest["files"][f"{task}_costaware/tests/data/test.parquet"]
            self.assertEqual(p, a)
            self.assertEqual(p["source"], f"datasets/{task}/test.parquet")
            # No held-out table is copied into an agent environment.
            for suffix in ("", "_costaware"):
                prefix = f"{task}{suffix}/environment/"
                for destination, spec in manifest["files"].items():
                    if destination.startswith(prefix):
                        self.assertNotEqual(spec["source"], p["source"])

    def test_dry_run_schedules_180_trials_without_credentials(self):
        process = subprocess.run([sys.executable, str(ROOT / "run.py"), "--all", "--repeats", "3", "--dry-run"],
                                 capture_output=True, text=True, check=True)
        config = json.loads(process.stdout)
        self.assertEqual(len(config["tasks"]) * config["repeats"], 180)
        self.assertEqual(config["agent_seconds"], 3600)
        self.assertEqual(config["effort"], "high")
        self.assertIsNone(config["max_turns"])
        self.assertFalse(config["smoke"])
        command = config["command"]
        self.assertEqual(Path(command[command.index("--jobs-dir") + 1]), ROOT / "runs")

    def test_invalid_task_is_rejected_before_api_access(self):
        process = subprocess.run([sys.executable, str(ROOT / "run.py"), "--task", "not_a_task", "--dry-run"],
                                 capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("Unknown task", process.stderr)


class AggregationTests(unittest.TestCase):
    def fixture(self):
        rows = [{"task": task, "setting": "active", "trial": f"{task}-{i}",
                 "r2": score, "flops_fraction_of_pool": cost}
                for task, scores, cost in [("a", [0.1, 0.2, 0.9], 0.2), ("b", [-1.0, -0.2, 0.3], 0.4)]
                for i, score in enumerate(scores)]
        config = {"tasks": ["a_costaware", "b_costaware"], "repeats": 3, "model": "fixture"}
        return rows, config

    def test_best_at_three_is_per_task_then_median(self):
        rows, config = self.fixture()
        result = aggregate(rows, config)["settings"]["active"]
        self.assertAlmostEqual(result["median_best_at_3_r2"], 0.6)
        self.assertAlmostEqual(result["median_r2"], 0.15)
        self.assertAlmostEqual(result["median_cost_share"], 0.3)
        self.assertEqual(result["task_best_at_3_r2"], {"a": 0.9, "b": 0.3})

    def test_missing_repeat_is_not_silently_aggregated(self):
        rows, config = self.fixture()
        with self.assertRaises(ValueError):
            aggregate(rows[:-1], config)

    def test_nonfinite_score_is_not_silently_discarded(self):
        rows, config = self.fixture()
        rows[0]["r2"] = float("inf")
        with self.assertRaises(ValueError):
            aggregate(rows, config)

    def test_duplicate_trial_is_rejected(self):
        rows, config = self.fixture()
        rows[1]["trial"] = rows[0]["trial"]
        with self.assertRaises(ValueError):
            aggregate(rows, config)

    def test_single_repeat_does_not_claim_best_at_three(self):
        rows, config = self.fixture()
        config["repeats"] = 1
        result = aggregate([rows[0], rows[3]], config)
        self.assertNotIn("median_best_at_3_r2", result["settings"]["active"])

    def test_aggregate_retains_framework_identity(self):
        rows, config = self.fixture()
        config.update(agent="gemini-cli", agent_version="0.59.0", protocol="Gemini generateContent")
        result = aggregate(rows, config)
        self.assertEqual(result["agent"], "gemini-cli")
        self.assertEqual(result["agent_version"], "0.59.0")
        self.assertEqual(result["protocol"], "Gemini generateContent")


if __name__ == "__main__":
    unittest.main()
