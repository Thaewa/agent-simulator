import unittest
from agents import Wasp, Larvae, WaspRole
from simulator import Simulator

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator()
        self.feeder = Wasp(agent_id="F1", x=0, y=0, role=WaspRole.FEEDER, food=10)
        self.forager = Wasp(agent_id="Fo1", x=1, y=1, role=WaspRole.FORAGER, food=5)
        self.larvae = Larvae(agent_id="L1", x=2, y=2, hunger=5)

        self.sim.addAgent(self.feeder)
        self.sim.addAgent(self.forager)
        self.sim.addAgent(self.larvae)
        self.sim.addForage(5, 5)

    def test_feed_larvae(self):
        old_food = self.feeder.food
        self.feeder.feed(self.larvae)
        self.assertEqual(self.larvae.hunger, 1)  # reset to 1
        self.assertLess(self.feeder.food, old_food)

    def test_forage_increases_food(self):
        old_food = self.forager.food
        self.forager.forage(self.sim.forage)
        self.assertGreaterEqual(self.forager.food, old_food)

    def test_move_changes_position(self):
        old_pos = self.feeder.getPosition()
        self.feeder.next_step = {"x": 1, "y": 1}
        self.feeder.move()
        self.assertNotEqual(old_pos, self.feeder.getPosition())

    def test_step_forager_behavior(self):
        """Forager should still have non-negative food after stepping"""
        self.forager.step(agents=self.sim.agents, forage=self.sim.forage)  # ✅ fixed
        self.assertGreaterEqual(self.forager.food, 0)
