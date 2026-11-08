"""Regression checks for mode selection; no simulations or log writes."""

import unittest
import contextlib
import io
from unittest.mock import patch

import numpy as np

from agents import Wasp, WaspRole
from simulator import instanceGenerator
from main import parse_args


class TestPathfindingModes(unittest.TestCase):
    def make_wasp(self, mode="greedy", food=3, role=WaspRole.FEEDER):
        wasp = Wasp("probe", outer_nest_radius=4, path_finding=mode, food=food)
        wasp.role = role
        return wasp

    def test_constructor_normalizes_mode(self):
        for supplied, expected in (("TSP", "tsp"),
                                   (" TSP-Hamiltonian ", "tsp-hamiltonian"),
                                   ("RANDOM_WALK-BIASED", "random_walk-biased")):
            with self.subTest(mode=supplied):
                self.assertEqual(self.make_wasp(supplied).path_finding, expected)

    def test_direct_assignment_normalizes_mode(self):
        wasp = self.make_wasp()
        wasp.path_finding = "TSP-Hamiltonian"
        self.assertEqual(wasp.path_finding, "tsp-hamiltonian")

    def test_cli_normalizes_before_config_and_logger_use(self):
        with patch("sys.argv", ["main.py", "--mode", "TSP-Hamiltonian"]):
            self.assertEqual(parse_args().mode, "tsp-hamiltonian")

    def test_cli_rejects_unknown_mode_before_simulation(self):
        with patch("sys.argv", ["main.py", "--mode", "tspp"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args()
        self.assertEqual(error.exception.code, 2)

    def test_unknown_mode_is_rejected(self):
        for mode in ("tspp", "not-tsp", "unknown", ""):
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    self.make_wasp(mode)

    def test_batch_tsp_modes_reach_route_planner(self):
        for mode in ("tsp", "tsp-hamiltonian", "TSP", "TSP-Hamiltonian"):
            with self.subTest(mode=mode):
                wasp = self.make_wasp(mode)
                larvae = np.array([[1, 0], [0, 1]])
                with patch.object(wasp, "greedy_path_finding") as greedy, \
                     patch.object(wasp, "shortest_path_finding") as route:
                    wasp.feelGradient(None, None, larvaePositions=larvae)
                greedy.assert_called_once()
                route.assert_called_once_with(larvae)

    def test_non_tsp_modes_do_not_request_tsp_routes(self):
        for mode in ("greedy", "random_walk", "random_walk-biased"):
            with self.subTest(mode=mode):
                wasp = self.make_wasp(mode)
                with patch.object(wasp, "greedy_path_finding"), \
                     patch.object(wasp, "random_walk_path_finding") as random_walk, \
                     patch.object(wasp, "shortest_path_finding") as route:
                    wasp.feelGradient(None, None)
                route.assert_not_called()
                self.assertEqual(random_walk.call_count, int(mode != "greedy"))

    def test_tsp_keeps_feeder_and_food_restrictions(self):
        for role, food in ((WaspRole.FORAGER, 3), (WaspRole.FEEDER, 1)):
            with self.subTest(role=role, food=food):
                wasp = self.make_wasp("tsp", food=food, role=role)
                with patch.object(wasp, "greedy_path_finding") as greedy, \
                     patch.object(wasp, "shortest_path_finding") as route:
                    wasp.feelGradient(None, None)
                greedy.assert_called_once()
                route.assert_not_called()

    def test_open_and_closed_tours_pass_correct_cycle_option(self):
        X, Y = np.meshgrid(np.arange(3), np.arange(3))
        for mode, closed in (("tsp", False), ("TSP", False),
                             ("tsp-hamiltonian", True), ("TSP-Hamiltonian", True)):
            with self.subTest(mode=mode):
                wasp = self.make_wasp(mode)
                with patch("agents.nx.approximation.traveling_salesman_problem",
                           return_value=[(0, 0), (1, 0)]) as planner:
                    wasp.estimate_path(None, None, np.array([[0, 0], [1, 0]]), X, Y)
                self.assertEqual(planner.call_args.kwargs["cycle"], closed)

    def test_generator_mode_override_is_normalized(self):
        generator = instanceGenerator()
        generator.waspDictionary({})
        generator.simulatorDictionary({"pathfinding_mode": "TSP-Hamiltonian"})
        self.assertEqual(generator.generateWasp(0, 0).path_finding, "tsp-hamiltonian")

    def test_generator_without_override_keeps_wasp_mode(self):
        generator = instanceGenerator()
        generator.waspDictionary({"path_finding": "TSP"})
        # The recovered generator supplies the radius separately.
        self.assertEqual(generator.generateWasp(0, 0).path_finding, "tsp")


if __name__ == "__main__":
    unittest.main()
