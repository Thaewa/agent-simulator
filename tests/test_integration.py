import unittest
from simulator import Simulator
from agents import Wasp, Larvae, WaspRole

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator()
        self.w1 = Wasp("W1", 0, 0, WaspRole.FORAGER, food=5)
        self.w2 = Wasp("W2", 1, 1, WaspRole.FEEDER, food=5)
        self.l1 = Larvae("L1", 2, 2, hunger=5)
        self.sim.addAgent(self.w1)
        self.sim.addAgent(self.w2)
        self.sim.addAgent(self.l1)
        self.sim.addForage(5, 5)

    def test_run_simulation_report(self):
        report = self.sim.runSimulation(3)
        self.assertIsInstance(report, dict)
        self.assertIn("movements", report)
        self.assertIn("feedLarvae", report)
        self.assertIn("hungerLarvae", report)
        self.assertIn("hungerWasp", report)

    def test_hunger_threshold_behavior(self):
        self.l1.hunger = 10
        report = self.sim.runSimulation(1)
        self.assertIn("hungerLarvae", report)
        self.assertGreater(report["hungerLarvae"]["L1"][-1], 0)
