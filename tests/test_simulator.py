import unittest
from simulator import Simulator
from agents import Wasp, Larvae, WaspRole

class TestSimulator(unittest.TestCase):
    def setUp(self):
        # Create a simulator with at least:
        # - 1 feeder wasp
        # - 1 forager wasp
        # - 1 larvae
        # Also add at least one forage point (required by simulation conditions).
        self.sim = Simulator()
        self.feeder = Wasp("F1", 0, 0, WaspRole.FEEDER, food=5)
        self.forager = Wasp("Fo1", 1, 1, WaspRole.FORAGER, food=5)
        self.larvae = Larvae("L1", 2, 2, hunger=3)

        self.sim.addAgent(self.feeder)
        self.sim.addAgent(self.forager)
        self.sim.addAgent(self.larvae)
        self.sim.addForage(3, 3)

    def test_add_and_remove_agent(self):
        """Agent should be correctly added and removed from the simulation"""
        w = Wasp("Wtemp", 0, 0, WaspRole.FORAGER)
        self.sim.addAgent(w)
        self.assertIn(w, self.sim.agents)   # Ensure agent was added
        self.sim.removeAgent(w)
        self.assertNotIn(w, self.sim.agents)  # Ensure agent was removed

    def test_step_advances_time(self):
        """Calling step() should advance the simulation time by 1"""
        self.sim.step()
        self.assertEqual(self.sim.currentTime, 1)

    def test_aggregate_movements(self):
        """Initial agent position should appear in the movement history"""
        movements = self.sim.aggregateMovements()
        self.assertIn("F1", movements)  # Feeder's ID should exist in history
        self.assertEqual(movements["F1"], [[0, 0]])  # Position must match initial

    def test_run_simulation_advances_time(self):
        """runSimulation() should return a report with required keys"""
        report = self.sim.runSimulation(2)
        self.assertIn("movements", report)
        self.assertIn("feedLarvae", report)
        self.assertIn("hungerLarvae", report)
        self.assertIn("hungerWasp", report)

    def test_aggregate_hunger(self):
        """Hunger values for larvae and wasps should be correctly aggregated"""
        hunger_larvae = self.sim.aggregateHungerLarvae()
        hunger_wasp = self.sim.aggregateHungerWasp()
        self.assertEqual(hunger_larvae["L1"], [3])  # Initial hunger of larvae
        self.assertEqual(hunger_wasp["F1"], [0])    # Feeder hunger should start at 0


if __name__ == "__main__":
    unittest.main()
