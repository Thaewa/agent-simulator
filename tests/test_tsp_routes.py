"""Small, deterministic route tests using the real graph and TSP solver."""

import unittest
from unittest.mock import patch

import numpy as np
import networkx as nx

from agents import Wasp, WaspRole


class TestTspRoutes(unittest.TestCase):
    def make_wasp(self, mode="tsp", food=3, position=(0.5, 0.5)):
        wasp = Wasp("route-probe", outer_nest_radius=4,
                    x=position[0], y=position[1], path_finding=mode, food=food)
        wasp.role = WaspRole.FEEDER
        return wasp

    def plan(self, wasp, targets):
        targets = np.asarray(targets, dtype=float)
        wasp.get_path(targets, np.arange(len(targets)))
        first = (wasp.x + wasp.next_step["x"], wasp.y + wasp.next_step["y"])
        return [first] + list(wasp.path or [])

    def test_integer_and_half_cell_targets_remain_exact(self):
        for offset in (0, 0.5):
            with self.subTest(offset=offset):
                wasp = self.make_wasp(food=2, position=(offset, offset))
                targets = np.array([[1, 0], [0, 1], [1, 1]]) + offset
                route = self.plan(wasp, targets)
                self.assertEqual(len(set(map(tuple, targets)) & set(route)), 2)
                self.assertTrue(all(x * 2 == int(x * 2) and y * 2 == int(y * 2)
                                    for x, y in route))

    def test_enough_food_for_all_targets_still_plans(self):
        targets = [(1, 0.5), (1.5, 0.5), (1.5, 1)]
        for food in (3, 6):
            with self.subTest(food=food):
                route = self.plan(self.make_wasp(food=food), targets)
                self.assertTrue(set(targets).issubset(route))

    def test_single_target_produces_a_nonzero_step(self):
        wasp = self.make_wasp(food=2)
        route = self.plan(wasp, [(1.5, 0.5)])
        self.assertEqual(route[-1], (1.5, 0.5))
        self.assertNotEqual(wasp.next_step, {"x": 0, "y": 0})

    def test_open_tour_is_open_and_closed_tour_returns_to_first_target(self):
        targets = [(1, 0.5), (1.5, 0.5), (1.5, 1)]
        solver = nx.approximation.traveling_salesman_problem
        for mode, closed in (("tsp", False), ("tsp-hamiltonian", True)):
            with self.subTest(mode=mode):
                tours = []

                def record_tour(*args, **kwargs):
                    tour = solver(*args, **kwargs)
                    tours.append(tour)
                    return tour

                with patch("agents.nx.approximation.traveling_salesman_problem",
                           side_effect=record_tour):
                    route = self.plan(self.make_wasp(mode=mode), targets)
                self.assertTrue(set(targets).issubset(route))
                # The connector can pass a target before reaching the tour start.
                # Check the actual solver tour, not that incidental first visit.
                self.assertEqual(tours[0][0] == tours[0][-1], closed)

    def test_real_move_follows_route_without_stationary_prefix(self):
        for mode in ("tsp", "tsp-hamiltonian"):
            with self.subTest(mode=mode):
                wasp = self.make_wasp(mode=mode)
                targets = np.array([[1, 0.5], [1.5, 0.5], [1.5, 1]])
                planned = self.plan(wasp, targets)
                X, Y = np.meshgrid(np.arange(-4, 4.5, 0.5), np.arange(-4, 4.5, 0.5))
                grid = np.column_stack((X.ravel(), Y.ravel()))
                visited = []
                for expected in planned:
                    previous = np.array(wasp.getPosition())
                    wasp.move(np.empty((0, 2)), grid)
                    self.assertEqual(tuple(wasp.getPosition()), expected)
                    displacement = np.abs(np.array(wasp.getPosition()) - previous)
                    self.assertGreater(float(displacement.max()), 0)
                    self.assertLessEqual(float(displacement.max()), 0.5)
                    visited.append(tuple(wasp.getPosition()))
                    if wasp.path:
                        wasp.shortest_path_finding(targets)
                self.assertTrue(set(map(tuple, targets)).issubset(visited))

    def test_no_targets_or_no_capacity_keeps_greedy_fallback(self):
        for targets, food in (([], 3), ([(1, 0.5)], 0.5)):
            with self.subTest(targets=targets, food=food):
                wasp = self.make_wasp(food=food)
                wasp.next_step = {"x": -0.5, "y": 0}
                self.plan(wasp, targets)
                self.assertIsNone(wasp.path)
                self.assertEqual(wasp.next_step, {"x": -0.5, "y": 0})

    def test_target_at_current_position_does_not_create_zero_step(self):
        wasp = self.make_wasp()
        wasp.next_step = {"x": -0.5, "y": 0}
        self.plan(wasp, [(0.5, 0.5)])
        self.assertIsNone(wasp.path)
        self.assertEqual(wasp.next_step, {"x": -0.5, "y": 0})

    def test_mixed_negative_and_positive_half_cell_targets(self):
        wasp = self.make_wasp(position=(-0.5, -0.5))
        targets = [(-1, -0.5), (0.5, -1), (-0.5, 0.5)]
        route = self.plan(wasp, targets)
        self.assertTrue(set(targets).issubset(route))
        previous = np.array(wasp.getPosition())
        for node in route:
            self.assertLessEqual(float(np.abs(np.array(node) - previous).max()), 0.5)
            previous = np.array(node)

    def test_deviation_discards_route_instead_of_jumping_two_cells(self):
        wasp = self.make_wasp()
        wasp.path = [(1.5, 0.5)]
        wasp.next_step = {"x": -0.5, "y": 0}
        wasp.shortest_path_finding(np.array([[1.5, 0.5]]))
        self.assertIsNone(wasp.path)
        self.assertEqual(wasp.next_step, {"x": -0.5, "y": 0})


if __name__ == "__main__":
    unittest.main()
