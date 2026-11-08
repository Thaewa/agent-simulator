import unittest
from unittest.mock import patch
import numpy as np
from simulation_fixture import LoggedSimulationTestCase

class TestAgents(LoggedSimulationTestCase):
    def test_feed_larvae(self):
        donor_before, receiver_before = self.feeder.food, self.larvae.food
        self.feeder.feed(self.larvae)
        self.assertEqual(self.larvae.hunger, self.larvae.noHunger)
        self.assertEqual(donor_before - self.feeder.food, self.feeder.foodTransferToLarvae)
        self.assertEqual(self.larvae.food - receiver_before, self.feeder.foodTransferToLarvae)
        self.assertEqual(self.feeder.feed_count_larvae, 1)

    def test_forage_increases_food_at_nearby_site(self):
        old_food = self.forager.food
        with patch("agents.np.random.rand", return_value=0):
            self.forager.forage([np.array(self.forager.getPosition())])
        self.assertGreater(self.forager.food, old_food)
        self.assertTrue(any("foraged" in event for event in self.forager.storedEvents))

    def test_move_applies_planned_position(self):
        self.sim.createGrid()
        self.feeder.next_step = {"x": 0.5, "y": 0.5}
        self.feeder.move(np.empty((0, 2)), self.sim.grid)
        self.assertEqual(self.feeder.getPosition(), [0.5, 0.5])
        self.assertAlmostEqual(self.feeder.distance_traveled, np.sqrt(0.5))

    def test_step_forager_returns_events_and_keeps_food_nonnegative(self):
        events = self.forager.step(t=1, agents=self.sim.agents, forage=self.sim.forage)
        self.assertIsInstance(events, list)
        self.assertGreaterEqual(self.forager.food, 0)

if __name__ == "__main__":
    unittest.main()
