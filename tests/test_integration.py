import numpy as np
from simulation_fixture import LoggedSimulationTestCase

class TestIntegration(LoggedSimulationTestCase):
    def test_run_simulation_report(self):
        report = self.run_simulation(3)
        self.assertEqual(set(report), {"movements", "feedLarvae", "hungerLarvae", "hungerWasp"})
        self.assertEqual(set(report["hungerWasp"]), {"F1", "Fo1"})
        self.assertEqual(set(report["hungerLarvae"]), {"L1"})
        self.assertEqual(self.sim.currentTime, 3)
        for agent in self.sim.agents:
            self.assertTrue(np.isfinite([agent.x, agent.y, agent.food, agent.hunger]).all())
            self.assertGreaterEqual(agent.food, 0)

    def test_unfed_larva_accumulates_hunger(self):
        self.larvae.hunger = 10
        self.feeder.food = self.forager.food = self.larvae.food = 0
        report = self.run_simulation(1)
        self.assertAlmostEqual(report["hungerLarvae"]["L1"][-1], 10 + self.larvae.hungerRate)
