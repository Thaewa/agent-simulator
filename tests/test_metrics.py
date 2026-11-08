import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from agents import Wasp, WaspRole, Larvae
from metrics import (METRICS_SCHEMA, distance_per_event, normalize_aggregate,
                     ratio, summarize_agents, summarize_runs)
from simulation_fixture import LoggedSimulationTestCase


class TestMetrics(unittest.TestCase):
    def agent(self, distance, feeds, role=WaspRole.FORAGER):
        return SimpleNamespace(distance_traveled=distance, feed_count_larvae=feeds,
                               feed_count_wasp=0, step_count=10, role=role,
                               feeder_distance_traveled=distance,
                               feeder_feed_count_larvae=feeds, feeder_step_count=10)

    def test_distance_per_event_includes_nonfeeding_adults(self):
        agents = [self.agent(4, 2), self.agent(6, 0)]
        self.assertEqual(distance_per_event(agents, "feed_count_larvae"), 5)

    def test_no_feed_is_missing_distance_cost_not_zero(self):
        self.assertTrue(np.isnan(distance_per_event([self.agent(4, 0)], "feed_count_larvae")))
        self.assertTrue(np.isnan(ratio(0, 0)))

    def test_rates_use_steps_and_counts_remain_explicit(self):
        result = summarize_agents([self.agent(4, 2), self.agent(6, 0)], [], 10)
        self.assertEqual(result["mean_feed_count_larvae"], 1)
        self.assertEqual(result["mean_feed_rate_larvae"], 0.1)
        self.assertEqual(result["std_feed_rate_larvae"], 0.1)

    def test_role_metrics_do_not_depend_on_final_role(self):
        agent = self.agent(4, 2)
        first = summarize_agents([agent], [], 10)
        agent.role = WaspRole.FEEDER
        second = summarize_agents([agent], [], 10)
        self.assertEqual(first["sum_distance_traveled_feeders"], second["sum_distance_traveled_feeders"])
        self.assertEqual(first["sum_feed_count_larvae"], 2)

    def test_zero_steps_has_no_defined_rate(self):
        result = summarize_agents([self.agent(0, 0)], [], 0)
        self.assertTrue(np.isnan(result["mean_feed_rate_larvae"]))

    def test_hunger_fraction_is_explicit_final_state(self):
        larvae = [SimpleNamespace(hunger=h) for h in (1, 1.49, 1.5, 3)]
        result = summarize_agents([], larvae, 1)
        self.assertEqual(result["final_low_hunger_fraction"], 0.5)

    def test_distance_per_step_does_not_divide_population_twice(self):
        frame = pd.DataFrame([{
            "metrics_schema": METRICS_SCHEMA, "__mode__": "tsp", "simulation_id": 1,
            "sum_distance_traveled_feeders": 20, "sum_step_count_feeders": 40,
            "sum_feed_count_larvae": 4, "num_wasps": 10, "num_larvae": 2,
            "num_feeder_participants": 4,
        }])
        row = normalize_aggregate(frame).iloc[0]
        self.assertEqual(row.distance_per_wasp_per_step, 0.5)
        self.assertEqual(row.norm_walk_to_feed_eff, 0.2)
        self.assertEqual(row.norm_step_to_feed_eff, 0.1)
        self.assertEqual(row.norm_steps, 10)

    def test_uncertainty_is_between_run_sd(self):
        frame = pd.DataFrame({"metrics_schema": [METRICS_SCHEMA] * 2,
                              "__mode__": ["tsp"] * 2, "simulation_id": [1, 2],
                              "rate": [0.1, 0.3], "within_run_sd": [99, 99]})
        row = summarize_runs(frame, ["rate"]).iloc[0]
        self.assertAlmostEqual(row.rate_mean, 0.2)
        self.assertAlmostEqual(row.rate_std, np.std([0.1, 0.3], ddof=1))
        single = summarize_runs(frame.iloc[:1], ["rate"]).iloc[0]
        self.assertTrue(np.isnan(single.rate_std))

    def test_historical_schema_and_duplicate_run_ids_are_rejected(self):
        frame = pd.DataFrame({"__mode__": ["tsp"], "simulation_id": [1], "rate": [1]})
        with self.assertRaises(ValueError):
            summarize_runs(frame, ["rate"])
        frame["metrics_schema"] = METRICS_SCHEMA
        with self.assertRaises(ValueError):
            summarize_runs(pd.concat([frame, frame]), ["rate"])

    def test_real_actions_are_attributed_to_the_role_at_action_time(self):
        wasp = Wasp("probe", outer_nest_radius=4, food=5)
        wasp.role = WaspRole.FEEDER
        wasp.updateHungerCueRange()
        larva = Larvae("larva", 0, 0, hunger=3)
        grid = np.array([[x, y] for x in (-1, -0.5, 0, 0.5, 1)
                         for y in (-1, -0.5, 0, 0.5, 1)])
        wasp.next_step = {"x": 0.5, "y": 0}
        wasp.move(np.empty((0, 2)), grid)
        wasp.feed(larva)
        wasp.role = WaspRole.FORAGER
        wasp.next_step = {"x": 0.5, "y": 0}
        wasp.move(np.empty((0, 2)), grid)
        wasp.feed(larva)
        self.assertEqual(wasp.feeder_step_count, 1)
        self.assertEqual(wasp.feeder_distance_traveled, 0.5)
        self.assertEqual(wasp.feeder_feed_count_larvae, 1)
        self.assertEqual(wasp.step_count, 2)
        self.assertEqual(wasp.distance_traveled, 1)
        self.assertEqual(wasp.feed_count_larvae, 2)


class TestLoggedRoleMetrics(LoggedSimulationTestCase):
    def test_aggregate_retains_activity_of_agents_that_switched_roles(self):
        self.sim.role_changes_frequency = 2
        self.sim.max_role_changes = 1
        self.sim.potential_feeder_to_forager = 1
        self.feeder.rolePersistence = 0
        self.feeder.feeder_to_forager_probability = 1
        self.feeder.hungerCuesHigh = Mock(return_value=True)
        self.feeder.hungerCuesLow = Mock(return_value=False)
        self.forager.hungerCuesLow = Mock(return_value=False)
        self.run_simulation(3)
        row = pd.read_csv(self.logs / "aggregate_results_greedy.csv").iloc[0]
        self.assertEqual(self.feeder.role, WaspRole.FORAGER)
        self.assertEqual(row.sum_step_count_feeders, 2)
        self.assertEqual(row.num_feeder_participants, 1)
        self.assertAlmostEqual(row.sum_distance_traveled_feeders,
                               self.feeder.feeder_distance_traveled, places=5)
        self.assertEqual(row.total_wasp_steps, 6)
        self.assertEqual(row.metrics_schema, METRICS_SCHEMA)


if __name__ == "__main__":
    unittest.main()
