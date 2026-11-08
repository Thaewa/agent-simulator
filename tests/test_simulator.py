import unittest
import pandas as pd
from agents import Wasp, WaspRole
from simulation_fixture import LoggedSimulationTestCase

class TestSimulator(LoggedSimulationTestCase):
    def test_add_and_remove_agent(self):
        wasp = Wasp("Wtemp", outer_nest_radius=4, x=0, y=0)
        wasp.role = WaspRole.FORAGER
        self.sim.addAgent(wasp)
        self.assertIn(wasp, self.sim.agents)
        self.sim.removeAgent(wasp)
        self.assertNotIn(wasp, self.sim.agents)

    def test_one_simulation_step_advances_time(self):
        self.run_simulation(1)
        self.assertEqual(self.sim.currentTime, 1)

    def test_aggregate_movements(self):
        self.assertEqual(self.sim.aggregateMovements()["F1"], [[0, 0]])

    def test_run_simulation_advances_time(self):
        report = self.run_simulation(2)
        self.assertEqual(self.sim.currentTime, 2)
        self.assertEqual(set(report), {"movements", "feedLarvae", "hungerLarvae", "hungerWasp"})

    def test_aggregate_hunger(self):
        self.assertEqual(self.sim.aggregateHungerLarvae()["L1"], [3])
        self.assertEqual(self.sim.aggregateHungerWasp()["F1"], [0])

    def test_logs_contain_initial_state_then_each_completed_step(self):
        self.run_simulation(3)
        nest = pd.read_csv(self.logs / "nest_log_greedy.csv")
        larvae = pd.read_csv(self.logs / "larvae_log_greedy.csv")
        agents = pd.read_csv(self.logs / "agent_log_greedy.csv")
        self.assertEqual(nest.timestamp.tolist(), [0, 1, 2, 3])
        self.assertEqual(larvae.timestamp.tolist(), [0, 1, 2, 3])
        for identifier in ("F1", "Fo1"):
            self.assertEqual(agents[agents.agent_id == identifier].timestamp.tolist(), [0, 1, 2, 3])
        self.assertFalse(nest.duplicated(["simulation_id", "timestamp"]).any())
        self.assertEqual(larvae[larvae.timestamp == 0].iloc[0].hunger_level, 3)
        last = agents[(agents.agent_id == "F1") & (agents.timestamp == 3)].iloc[0]
        self.assertEqual([last.position_x, last.position_y], self.feeder.getPosition())

    def test_zero_steps_only_logs_initial_state(self):
        self.run_simulation(0)
        nest = pd.read_csv(self.logs / "nest_log_greedy.csv")
        self.assertEqual(nest.timestamp.tolist(), [0])
        self.assertEqual(self.sim.currentTime, 0)

if __name__ == "__main__":
    unittest.main()
