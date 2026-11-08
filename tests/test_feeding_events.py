"""Feeding probability, successful bout creation and per-step event counts."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents import Larvae, Wasp, WaspRole
from simulation_fixture import LoggedSimulationTestCase


class TestFeedingEvents(unittest.TestCase):
    def wasp(self, name="donor", food=5, role=WaspRole.FORAGER):
        agent = Wasp(name, outer_nest_radius=4, food=food, max_food=6)
        agent.role = role
        agent.updateHungerCueRange()
        return agent

    def test_empty_receivers_always_return_list(self):
        self.assertEqual(self.wasp().feedNearby([], 1), [])

    def test_probability_zero_never_feeds_and_one_always_feeds(self):
        for probability in (0, 1):
            with self.subTest(probability=probability):
                donor = self.wasp(role=WaspRole.FEEDER)
                donor.chanceOfFeeding = probability
                larva = Larvae("larva", 0, 0, hunger=3)
                with patch("agents.random.random", return_value=0.5):
                    donor.feedNearby([larva], 1)
                self.assertEqual(donor.feed_count_larvae, probability)

    def test_probability_threshold(self):
        for draw, expected in ((0.79, 1), (0.8, 0), (0.81, 0)):
            with self.subTest(draw=draw):
                donor = self.wasp(role=WaspRole.FEEDER)
                larva = Larvae("larva", 0, 0, hunger=3)
                with patch("agents.random.random", return_value=draw):
                    donor.feedNearby([larva], 1)
                self.assertEqual(donor.feed_count_larvae, expected)

    def test_failed_transfer_creates_no_bout(self):
        donor = self.wasp()
        donor.chanceOfFeeding = 1
        receiver = self.wasp("receiver", food=6, role=WaspRole.FEEDER)
        receiver.hunger = 3
        with patch("agents.random.random", return_value=0):
            bouts = donor.feedNearby([receiver], 4)
        self.assertEqual(bouts, [])
        self.assertEqual(donor.feed_count_wasp, 0)

    def test_insufficient_donor_creates_no_bout(self):
        donor = self.wasp(food=0.5)
        donor.chanceOfFeeding = 1
        receiver = self.wasp("receiver", food=0, role=WaspRole.FEEDER)
        receiver.hunger = 3
        with patch("agents.random.random", return_value=0):
            self.assertEqual(donor.feedNearby([receiver], 1), [])
        self.assertEqual((donor.food, receiver.food), (0.5, 0))

    def test_rejected_unload_creates_no_larval_event(self):
        donor = self.wasp()
        donor.chanceOfFeeding = 1
        donor.hungerCue = 1
        donor.unloadOnlyChance = 0
        larva = Larvae("larva", 0, 0, hunger=3)
        with patch("agents.random.random", return_value=0), \
             patch("agents.np.random.rand", return_value=0.5):
            self.assertEqual(donor.feedNearby([larva], 1), [])
        self.assertEqual((donor.food, larva.food, donor.feed_count_larvae), (5, 0, 0))

    def test_successful_adult_transfer_records_actual_amount(self):
        donor = self.wasp()
        donor.chanceOfFeeding = 1
        receiver = self.wasp("receiver", food=5.75, role=WaspRole.FEEDER)
        receiver.hunger = 3
        with patch("agents.random.random", return_value=0):
            bouts = donor.feedNearby([receiver], 4)
        self.assertEqual(len(bouts), 1)
        self.assertIs(bouts[0].feeder, donor)
        self.assertIs(bouts[0].receiver, receiver)
        self.assertEqual((bouts[0].timestamp, bouts[0].amount), (4, 0.25))
        self.assertEqual((donor.food, receiver.food), (4.75, 6))

    def test_larval_feed_does_not_create_adult_transfer_bout(self):
        donor = self.wasp()
        donor.chanceOfFeeding = 1
        larva = Larvae("larva", 0, 0, hunger=3)
        with patch("agents.random.random", return_value=0):
            bouts = donor.feedNearby([larva], 1)
        self.assertEqual(donor.feed_count_larvae, 1)
        self.assertEqual(bouts, [])

    def test_self_is_not_a_receiver_or_a_random_trial(self):
        donor = self.wasp(role=WaspRole.FEEDER)
        donor.hunger = 3
        with patch("agents.random.random") as draw:
            self.assertEqual(donor.feedNearby([donor], 1), [])
        draw.assert_not_called()
        self.assertEqual(donor.feed_count_wasp, 0)


class TestNestEventCounts(LoggedSimulationTestCase):
    def test_adult_transfer_events_are_step_counts_not_cumulative(self):
        self.feeder.food, self.feeder.hunger = 0, 3
        self.forager.x, self.forager.y = 0.5, 0.5
        self.forager.chanceOfFeeding = 1
        self.larvae.hunger = 0
        with patch.object(self.sim, "moveAgent"):
            self.run_simulation(3)
        nest = pd.read_csv(self.logs / "nest_log_greedy.csv")
        self.assertEqual(nest.transfer_events.tolist(), [0, 1, 0, 0])
        aggregate = pd.read_csv(self.logs / "aggregate_results_greedy.csv").iloc[0]
        self.assertEqual(aggregate.total_adult_transfer_events, nest.transfer_events.sum())

    def test_larval_feeding_events_are_step_counts_not_cumulative(self):
        self.feeder.chanceOfFeeding = 1
        self.forager.food = 0
        self.larvae.x, self.larvae.y = 0.5, 0.5
        with patch.object(self.sim, "moveAgent"):
            self.run_simulation(3)
        nest = pd.read_csv(self.logs / "nest_log_greedy.csv")
        self.assertEqual(nest.feeding_events.tolist(), [0, 1, 0, 0])
        aggregate = pd.read_csv(self.logs / "aggregate_results_greedy.csv").iloc[0]
        self.assertEqual(aggregate.total_larval_feeding_events, nest.feeding_events.sum())

    def test_foraging_events_are_step_counts_not_cumulative(self):
        self.feeder.food = self.forager.food = 0
        self.forager.chanceOfForaging = 1
        self.sim.forage = [np.array(self.forager.getPosition())]
        with patch.object(self.sim, "moveAgent"):
            self.run_simulation(3)
        nest = pd.read_csv(self.logs / "nest_log_greedy.csv")
        self.assertEqual(nest.foraging_events.tolist(), [0, 1, 1, 1])
        aggregate = pd.read_csv(self.logs / "aggregate_results_greedy.csv").iloc[0]
        self.assertEqual(aggregate.total_foraging_events, nest.foraging_events.sum())


if __name__ == "__main__":
    unittest.main()
