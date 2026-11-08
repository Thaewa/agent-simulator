"""Keep biological bouts distinct from transfer events and legacy proxies."""
import unittest
from unittest.mock import patch

import pandas as pd

from agents import FeedingBout, FoodTransferEvent, WaspRole
from simulation_fixture import LoggedSimulationTestCase


class TestBoutSemantics(LoggedSimulationTestCase):
    def test_transfer_api_and_legacy_constructor_remain_compatible(self):
        self.assertIs(FeedingBout, FoodTransferEvent)
        event = FeedingBout(feeder=self.forager, receiver=self.feeder, t=4, amount=0.25)
        self.assertIs(event.donor, self.forager)
        self.assertIs(event.feeder, self.forager)
        self.assertIs(event.receiver, self.feeder)
        self.assertEqual((event.timestamp, event.amount), (4, 0.25))

    def test_successful_unload_returns_a_transfer_event(self):
        self.forager.x = self.feeder.x + 0.5
        self.forager.y = self.feeder.y
        self.forager.chanceOfFeeding = 1
        self.forager.role = WaspRole.FORAGER
        self.feeder.food, self.feeder.hunger = 0, 3
        events = self.forager.feedNearby([self.feeder], 2)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], FoodTransferEvent)
        self.assertIs(events[0].donor, self.forager)
        self.assertEqual(self.forager.feed_count_wasp, 1)
        self.assertEqual(self.sim.finished_bouts, 0)

    def test_positive_proxy_count_is_not_reported_as_biological_bouts(self):
        # A legacy threshold can close a transfer group without any observation
        # of an arrival or food-source lineage. Preserve it, but label it honestly.
        self.forager.food = self.feeder.food = 0
        self.sim.active_bouts[self.forager] = [self.feeder]
        with patch("builtins.print"):
            self.sim.count_bouts()
        self.assertEqual(self.sim.finished_bouts, 1)
        with patch.object(self.sim, "stepAgent"), patch.object(self.sim, "moveAgent"):
            self.run_simulation(1)
        result = pd.read_csv(self.sim.logger.aggregate_log_path).iloc[0]
        self.assertEqual(result.bout_count_semantics, "legacy_food_threshold_proxy_v1")
        self.assertEqual(result.biological_bout_status, "unavailable")
        self.assertEqual(result.legacy_bout_proxy_count, 1)
        self.assertEqual(result.total_adult_transfer_events, 0)


if __name__ == "__main__":
    unittest.main()
