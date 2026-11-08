"""Food capacity and conservation at collection and transfer boundaries."""

import unittest
from unittest.mock import patch

import numpy as np

from agents import Larvae, Wasp, WaspRole


class TestFoodLimits(unittest.TestCase):
    def wasp(self, name="donor", food=5, capacity=6, role=WaspRole.FEEDER):
        agent = Wasp(name, outer_nest_radius=4, food=food, max_food=capacity)
        agent.role = role
        agent.updateHungerCueRange()
        return agent

    def forage(self, agent, sites=((0, 0),)):
        with patch("agents.np.random.rand", return_value=0):
            agent.forage([np.array(site) for site in sites])

    def test_forage_uses_increment_when_capacity_is_larger(self):
        agent = self.wasp(food=2, capacity=20)
        agent.forageIncrease = 3
        self.forage(agent)
        self.assertEqual(agent.food, 5)

    def test_forage_caps_at_remaining_capacity(self):
        agent = self.wasp(food=2, capacity=6)
        self.forage(agent)
        self.assertEqual(agent.food, 6)

    def test_multiple_nearby_sites_cannot_overfill(self):
        agent = self.wasp(food=5.75, capacity=6)
        self.forage(agent, ((0, 0), (0.5, 0), (0, 0.5)))
        self.assertEqual(agent.food, 6)
        self.assertEqual(len(agent.storedEvents), 1)

    def test_full_wasp_does_not_collect_or_log_collection(self):
        agent = self.wasp(food=6, capacity=6)
        self.forage(agent)
        self.assertEqual(agent.food, 6)
        self.assertEqual(agent.storedEvents, [])

    def test_wasp_transfer_caps_recipient_and_conserves_food(self):
        for role, high in ((WaspRole.FEEDER, False), (WaspRole.FORAGER, True)):
            with self.subTest(role=role):
                donor = self.wasp(role=role)
                if high:
                    donor.hungerCue = 10
                recipient = self.wasp("receiver", food=5.75)
                total = donor.food + recipient.food
                donor.feed(recipient)
                self.assertEqual(recipient.food, 6)
                self.assertEqual(donor.food, 4.75)
                self.assertEqual(donor.food + recipient.food, total)
                self.assertEqual(donor.feed_count_wasp, 1)

    def test_full_recipient_does_not_consume_food_or_count_event(self):
        donor, recipient = self.wasp(), self.wasp("receiver", food=6)
        donor.feedWasp(recipient, 1)
        self.assertEqual((donor.food, recipient.food, donor.feed_count_wasp), (5, 6, 0))
        self.assertEqual(donor.storedEvents, [])

    def test_insufficient_donor_food_never_goes_negative(self):
        for kind in ("wasp", "larva"):
            with self.subTest(kind=kind):
                donor = self.wasp(food=0.5)
                target = self.wasp("receiver", food=0) if kind == "wasp" else Larvae("larva", 0, 0)
                action = donor.feedWasp if kind == "wasp" else donor.feedLarvae
                action(target, 1)
                self.assertEqual(donor.food, 0.5)
                self.assertEqual(target.food, 0)
                self.assertEqual(donor.feed_count_wasp + donor.feed_count_larvae, 0)

    def test_zero_transfer_has_no_effect(self):
        donor, recipient = self.wasp(), self.wasp("receiver", food=0)
        donor.feedWasp(recipient, 0)
        self.assertEqual((donor.food, recipient.food, donor.feed_count_wasp), (5, 0, 0))

    def test_self_transfer_is_not_a_feeding_event(self):
        donor = self.wasp()
        donor.feedWasp(donor, 1)
        self.assertEqual(donor.food, 5)
        self.assertEqual(donor.feed_count_wasp, 0)

    def test_last_complete_ration_can_be_fed_in_normal_step(self):
        donor = self.wasp(food=1)
        larva = Larvae("larva", 0, 0, hunger=3)
        with patch("agents.random.random", return_value=0):
            donor.step(t=1, agents=[donor, larva], forage=[])
        self.assertEqual((donor.food, larva.food, donor.feed_count_larvae), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
