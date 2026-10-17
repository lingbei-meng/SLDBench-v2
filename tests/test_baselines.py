"""Regression checks for purchases, aggregation and reference inputs."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from baselines.acquisition import (candidate_order, greedy_fill, purchase,
                                   task_level_table, selected_tasks, source_mapping,
                                   build_jobs, job_key)

ROOT = Path(__file__).resolve().parents[1]


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.cat = pd.DataFrame({"obs_id": ["a", "b", "c", "d", "e"],
                                 "cost_flops": [1., 100., 2., 3., 4.],
                                 "_group_key": ["x", "x", "y", "y", "y"]})

    def test_expensive_candidate_does_not_stop_affordable_purchases(self):
        self.assertEqual(greedy_fill(self.cat, [1, 0, 4, 2, 3], 6), [0, 4])
        # Cost=5, remaining=1; later observations cannot exceed the cap.

    def test_rules_use_no_targets_and_respect_budget(self):
        cat = self.cat.copy()
        # Altering an irrelevant outcome must not affect the selected indices.
        cat["target"] = [100, -100, 7, 9, -1]
        for rule in ("cheapest", "random", "group_cheapest", "price_spread"):
            chosen = purchase(cat, rule, 2, .1)
            self.assertEqual(chosen, purchase(self.cat, rule, 2, .1))
            self.assertLessEqual(float(cat.loc[chosen, "cost_flops"].sum()), 11)

    def test_group_rule_visits_each_group_before_its_second_point(self):
        order = candidate_order(self.cat, "group_cheapest", None)
        self.assertEqual(order[:2], [0, 2])
        self.assertEqual(len(set(order)), len(self.cat))

    def test_random_order_independent_of_catalogue_row_order(self):
        order = candidate_order(self.cat, "random", 1)
        shuffled = self.cat.iloc[::-1]
        self.assertEqual(order, candidate_order(shuffled, "random", 1))
        self.assertNotEqual(order, candidate_order(self.cat, "random", 2))

    def test_full_pool_is_shared_and_grid_has_1290_distinct_fits(self):
        tasks = selected_tasks(ROOT)
        jobs = build_jobs(ROOT, tasks, Path("scratch"))
        self.assertEqual(len(jobs), 1290)
        self.assertEqual(len({job_key(j) for j in jobs}), 1290)
        self.assertEqual(sum(j["rule"] == "full_pool" for j in jobs), 30)
        self.assertEqual(purchase(self.cat, "full_pool", None, 1), list(self.cat.index))

    def test_random_aggregation_takes_seed_median_within_each_task(self):
        records = []
        for task, scores in (("a", [-50, .1, .2]), ("b", [.6, .8, 1.])):
            for seed, score in enumerate(scores):
                records.append(dict(task=task, rule="random", seed=seed, rho=.1,
                                    r2=score, realised_cost_fraction=.09, n_points=4, status="ok"))
        table = task_level_table(pd.DataFrame(records))
        random = table[table.rule.eq("random")].set_index("task")
        self.assertAlmostEqual(random.loc["a", "r2"], .1)
        self.assertAlmostEqual(random.loc["b", "r2"], .8)
        self.assertAlmostEqual(random.r2.median(), .45)

    def test_reference_laws_and_contracts_match_archived_sources(self):
        provenance = json.loads((ROOT / "baselines/provenance.json").read_text())
        self.assertEqual(len(provenance["reference_laws"]), 30)
        for task, expected in provenance["reference_laws"].items():
            data = (ROOT / "baselines/reference_laws" / (task + ".py")).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
        self.assertEqual(len(source_mapping(ROOT, selected_tasks(ROOT))), 30)



if __name__ == "__main__":
    unittest.main()
