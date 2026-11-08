"""Scheduling, probability, quota, ratio and cooldown checks for role changes."""

import unittest
from unittest.mock import Mock, patch
import pandas as pd

from agents import Wasp, WaspRole
from simulator import Simulator
from simulation_fixture import LoggedSimulationTestCase


class TestRoleSwitching(unittest.TestCase):
    def make_wasp(self, role=WaspRole.FEEDER, cooldown=50):
        agent = Wasp("role-probe", outer_nest_radius=4, role_persistence=cooldown,
                     feeder_to_forager_probability=1, forager_to_feeder_probability=1)
        agent.role = role
        agent.rolePersistence = 0
        agent.hungerCuesHigh = Mock(return_value=role == WaspRole.FEEDER)
        agent.hungerCuesLow = Mock(return_value=role == WaspRole.FORAGER)
        return agent

    def make_sim(self, frequency=10, quota=2):
        sim = Simulator(role_changes_frequency=frequency, max_role_changes=quota)
        sim.potential_feeder_to_forager = 0.3
        return sim

    def test_changes_only_at_scheduled_steps(self):
        for time, expected in ((1, WaspRole.FEEDER), (9, WaspRole.FEEDER),
                               (10, WaspRole.FORAGER), (11, WaspRole.FEEDER)):
            with self.subTest(time=time):
                agent = self.make_wasp()
                with patch("simulator.np.random.random", return_value=0.5):
                    self.make_sim().updateRoles(time, agent, 0, 2, 20)
                self.assertEqual(agent.role, expected)

    def test_probability_zero_never_switches_and_one_always_switches(self):
        for role in (WaspRole.FEEDER, WaspRole.FORAGER):
            for probability in (0, 1):
                with self.subTest(role=role, probability=probability):
                    agent = self.make_wasp(role)
                    agent.feeder_to_forager_probability = probability
                    agent.forager_to_feeder_probability = probability
                    with patch("simulator.np.random.random", return_value=0.5):
                        self.make_sim().updateRoles(10, agent, 0, 4, 20)
                    self.assertEqual(agent.role != role, bool(probability))

    def test_probability_threshold_uses_less_than(self):
        for draw, switched in ((0.79, True), (0.8, False), (0.81, False)):
            with self.subTest(draw=draw):
                agent = self.make_wasp()
                agent.feeder_to_forager_probability = 0.8
                with patch("simulator.np.random.random", return_value=draw):
                    self.make_sim().updateRoles(10, agent, 0, 2, 20)
                self.assertEqual(agent.role == WaspRole.FORAGER, switched)

    def test_exhausted_quota_does_not_attempt_an_extra_change(self):
        agent = self.make_wasp()
        with patch("simulator.np.random.random") as draw:
            count, total = self.make_sim().updateRoles(10, agent, 2, 2, 20)
        draw.assert_not_called()
        self.assertEqual((count, total, agent.role), (2, 2, WaspRole.FEEDER))

    def test_quota_is_shared_across_agents_at_scheduled_step(self):
        sim = self.make_sim()
        count, total = 0, 2
        agents = [self.make_wasp() for _ in range(4)]
        with patch("simulator.np.random.random", return_value=0.5):
            for agent in agents:
                count, total = sim.updateRoles(10, agent, count, total, 20)
        self.assertEqual((count, total), (2, 4))
        self.assertEqual(sum(a.role == WaspRole.FORAGER for a in agents), 2)

    def test_upper_forager_boundary_is_inclusive(self):
        agent = self.make_wasp()
        with patch("simulator.np.random.random", return_value=0.5):
            count, total = self.make_sim().updateRoles(10, agent, 0, 5, 20)
        self.assertEqual((count, total, agent.role), (1, 6, WaspRole.FORAGER))

    def test_lower_forager_boundary_is_inclusive(self):
        agent = self.make_wasp(WaspRole.FORAGER)
        with patch("simulator.np.random.random", return_value=0.5):
            count, total = self.make_sim().updateRoles(10, agent, 0, 3, 20)
        self.assertEqual((count, total, agent.role), (1, 2, WaspRole.FEEDER))

    def test_last_forager_cannot_switch_away(self):
        agent = self.make_wasp(WaspRole.FORAGER)
        sim = self.make_sim()
        sim.forager_ratio = 0
        with patch("simulator.np.random.random") as draw:
            sim.updateRoles(10, agent, 0, 1, 20)
        draw.assert_not_called()
        self.assertEqual(agent.role, WaspRole.FORAGER)

    def test_successful_change_restarts_full_cooldown(self):
        sim = self.make_sim(frequency=1)
        agent = self.make_wasp(cooldown=3)
        with patch("simulator.np.random.random", return_value=0.5):
            _, total = sim.updateRoles(1, agent, 0, 2, 20)
            self.assertEqual(agent.role, WaspRole.FORAGER)
            self.assertEqual(agent.rolePersistence, 3)
            agent.hungerCuesHigh.return_value = False
            agent.hungerCuesLow.return_value = True
            for time in (2, 3):
                sim.updateRoles(time, agent, 0, total, 20)
                self.assertEqual(agent.role, WaspRole.FORAGER)
            sim.updateRoles(4, agent, 0, total, 20)
            self.assertEqual(agent.role, WaspRole.FEEDER)
            self.assertEqual(agent.rolePersistence, 3)

    def test_cooldown_counts_down_on_unscheduled_steps(self):
        agent = self.make_wasp()
        agent.rolePersistence = 3
        self.make_sim().updateRoles(1, agent, 2, 2, 20)
        self.assertEqual(agent.rolePersistence, 2)


class TestRoleSwitchClock(LoggedSimulationTestCase):
    def test_simulation_passes_current_step_not_horizon(self):
        self.sim.role_changes_frequency = 2
        with patch.object(self.sim, "updateRoles", wraps=self.sim.updateRoles) as update:
            self.run_simulation(3)
        self.assertEqual([call.args[0] for call in update.call_args_list], [1, 1, 2, 2, 3, 3])

    def test_agent_and_nest_logs_agree_after_role_change(self):
        self.sim.role_changes_frequency = 2
        self.sim.max_role_changes = 1
        self.sim.potential_feeder_to_forager = 1
        self.feeder.rolePersistence = 0
        self.feeder.feeder_to_forager_probability = 1
        self.feeder.hungerCuesHigh = Mock(return_value=True)
        self.feeder.hungerCuesLow = Mock(return_value=False)
        self.forager.hungerCuesLow = Mock(return_value=False)
        self.run_simulation(2)
        agents = pd.read_csv(self.logs / "agent_log_greedy.csv")
        nests = pd.read_csv(self.logs / "nest_log_greedy.csv")
        adults = agents[agents.agent_id.isin(["F1", "Fo1"])]
        feeder = adults[adults.agent_id == "F1"].set_index("timestamp")
        self.assertEqual(feeder.loc[1, "agent_role"], "FEEDER")
        self.assertEqual(feeder.loc[2, "agent_role"], "FORAGER")
        for _, row in nests.iterrows():
            current = adults[adults.timestamp == row.timestamp]
            self.assertEqual(int(current.agent_role.eq("FORAGER").sum()), row.total_foragers)
            self.assertEqual(int(current.agent_role.eq("FEEDER").sum()), row.total_feeders)


if __name__ == "__main__":
    unittest.main()
