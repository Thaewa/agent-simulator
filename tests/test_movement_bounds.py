import unittest
from unittest.mock import patch

import numpy as np

from agents import Wasp, WaspRole
from simulator import Simulator


class TestMovementBounds(unittest.TestCase):
    def agent(self, position=(0, 0), role=WaspRole.FORAGER, radius=1):
        agent = Wasp("probe", outer_nest_radius=radius, x=position[0], y=position[1], max_step_trials=2)
        agent.role = role
        return agent

    def grid(self):
        X, Y = np.meshgrid(np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5))
        return np.column_stack((X.ravel(), Y.ravel()))

    def test_grid_membership_accepts_one_matching_row(self):
        agent = self.agent()
        agent.next_step = {"x": 0.5, "y": 0}
        self.assertTrue(agent.positionInGrid(self.grid()))
        agent.next_step = {"x": 2, "y": 0}
        self.assertFalse(agent.positionInGrid(self.grid()))

    def test_stay_is_not_randomized(self):
        agent = self.agent()
        with patch("agents.np.random.choice") as choose:
            agent.move(np.empty((0, 2)), self.grid())
        choose.assert_not_called()
        self.assertEqual(agent.getPosition(), [0, 0])
        self.assertEqual(agent.distance_traveled, 0)
        self.assertEqual(agent.step_count, 1)

    def test_out_of_grid_move_falls_back_to_stay(self):
        agent = self.agent((1, 0))
        agent.next_step = {"x": 0.5, "y": 0}
        with patch("agents.np.random.choice", return_value=np.array([0])):
            agent.move(np.empty((0, 2)), self.grid())
        self.assertEqual(agent.getPosition(), [1, 0])

    def test_feeder_can_reach_closed_nest_boundary(self):
        agent = self.agent((0.5, 0), WaspRole.FEEDER)
        agent.next_step = {"x": 0.5, "y": 0}
        with patch("agents.np.random.choice", return_value=np.array([0])):
            agent.move(np.empty((0, 2)), self.grid())
        self.assertEqual(agent.getPosition(), [1, 0])

    def test_feeder_cannot_leave_nest(self):
        agent = self.agent((0.5, 0.5), WaspRole.FEEDER)
        agent.next_step = {"x": 0.5, "y": 0.5}
        with patch("agents.np.random.choice", return_value=np.array([0])):
            agent.move(np.empty((0, 2)), self.grid())
        self.assertEqual(agent.getPosition(), [0.5, 0.5])

    def test_occupied_destination_is_not_entered(self):
        agent = self.agent()
        agent.next_step = {"x": 0.5, "y": 0}
        with patch("agents.np.random.choice", return_value=np.array([0])):
            agent.move(np.array([[0.5, 0]]), self.grid())
        self.assertEqual(agent.getPosition(), [0, 0])

    def test_retry_limit_stops_in_place(self):
        agent = self.agent()
        agent.next_step = {"x": 0.5, "y": 0.5}
        with patch("agents.np.random.choice", return_value=np.array([0.5])) as choose:
            agent.move(np.empty((0, 2)), np.array([[0, 0]]))
        self.assertEqual(agent.getPosition(), [0, 0])
        self.assertLessEqual(choose.call_count, 4)

    def test_oversized_action_cannot_jump_across_grid(self):
        agent = self.agent()
        agent.next_step = {"x": 1, "y": 0}
        with patch("agents.np.random.choice", return_value=np.array([0])):
            agent.move(np.empty((0, 2)), self.grid())
        self.assertEqual(agent.getPosition(), [0, 0])

    def test_invalid_start_is_rejected(self):
        agent = self.agent((2, 0))
        with self.assertRaises(ValueError):
            agent.move(np.empty((0, 2)), self.grid())

    def test_nonfinite_action_is_rejected(self):
        agent = self.agent()
        agent.next_step = {"x": float("nan"), "y": 0}
        with self.assertRaises(ValueError):
            agent.move(np.empty((0, 2)), self.grid())

    def test_forager_may_leave_nest_but_stays_on_grid(self):
        agent = self.agent((0.5, 0.5), radius=0.5)
        agent.next_step = {"x": 0.5, "y": 0.5}
        agent.move(np.empty((0, 2)), self.grid())
        self.assertEqual(agent.getPosition(), [1, 1])

    def test_initial_overlap_is_rejected(self):
        agent = self.agent()
        with self.assertRaises(ValueError):
            agent.move(np.array([[0, 0]]), self.grid())

    def test_stationary_neighbor_blocks_a_swap(self):
        first, second = self.agent(), self.agent((0.5, 0))
        first.id, second.id = "A", "B"
        first.next_step, second.next_step = {"x": 0.5, "y": 0}, {"x": -0.5, "y": 0}
        sim = Simulator()
        sim.grid = self.grid()
        sim.addAgent(first)
        sim.addAgent(second)
        with patch("agents.np.random.choice", return_value=np.array([0])):
            sim.moveAgent(first)
            self.assertNotEqual(first.getPosition(), second.getPosition())
            sim.moveAgent(second)
        self.assertEqual(first.getPosition(), [0, 0])
        self.assertEqual(second.getPosition(), [0.5, 0])

    def test_sequential_first_arrival_wins_contested_destination(self):
        first, second = self.agent((-0.5, 0)), self.agent((0.5, 0))
        first.id, second.id = "A", "B"
        first.next_step, second.next_step = {"x": 0.5, "y": 0}, {"x": -0.5, "y": 0}
        sim = Simulator()
        sim.grid = self.grid()
        sim.addAgent(first)
        sim.addAgent(second)
        with patch("agents.np.random.choice", return_value=np.array([0])):
            sim.moveAgent(first)
            sim.moveAgent(second)
        self.assertEqual(first.getPosition(), [0, 0])
        self.assertEqual(second.getPosition(), [0.5, 0])

    def test_created_grid_contains_half_cells_and_remote_forage(self):
        sim = Simulator()
        sim.addAgent(self.agent())
        sim.addForage(5.5, 0)
        sim.createGrid()
        cells = set(map(tuple, sim.grid))
        self.assertIn((0.5, 0), cells)
        self.assertIn((5.5, 0), cells)


if __name__ == "__main__":
    unittest.main()
