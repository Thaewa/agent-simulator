import math
import unittest

import numpy as np

from agents import Larvae, Wasp, WaspRole
from simulator import Simulator, instanceGenerator
from utils import grid_graph_from_array


class TestInstanceGeometry(unittest.TestCase):
    def test_generated_regions_and_identities_do_not_overlap(self):
        previous = np.random.get_state()
        self.addCleanup(np.random.set_state, previous)
        generator = instanceGenerator(number_of_cells=30, nest_fill_percentage=0.95)
        generator.waspDictionary({})
        generator.larvaeDictionary({"hunger_rate": 0.3, "hunger_rate_multiplier_l2": 2,
                                    "hunger_rate_multiplier_l3": 3})
        generator.simulatorDictionary({"forager_ratio": 0.1, "potential_feeder_to_forager_buffer": 0.2})
        inner_radius = math.ceil(math.sqrt(30 * 0.5 / math.pi))
        for seed in range(20):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                sim = generator.generateSimulator()
                self.assertEqual(len({a.id for a in sim.agents}), len(sim.agents))
                for kind in (Wasp, Larvae):
                    population = [a for a in sim.agents if isinstance(a, kind)]
                    self.assertEqual(len({tuple(a.getPosition()) for a in population}), len(population))
                for wasp in (a for a in sim.agents if isinstance(a, Wasp)):
                    if wasp.role == WaspRole.FORAGER:
                        self.assertGreater(wasp.x**2 + wasp.y**2, inner_radius**2)

    def test_graph_excludes_inactive_cells(self):
        X, Y = np.meshgrid(np.arange(2), np.arange(2))
        active = np.array([[True, False], [False, True]])
        graph = grid_graph_from_array(active, X, Y)
        self.assertEqual(set(graph), {(0, 0), (1, 1)})
        self.assertEqual(len(graph.edges), 1)

    def test_duplicate_agent_id_cannot_overwrite_movement_history(self):
        sim = Simulator()
        first = Wasp("same", outer_nest_radius=4)
        second = Wasp("same", outer_nest_radius=4, x=1)
        sim.addAgent(first)
        with self.assertRaises(ValueError):
            sim.addAgent(second)
        self.assertEqual(sim.movementHistory["same"], [[0, 0]])

    def test_feeder_route_respects_nest_mask(self):
        wasp = Wasp("route", outer_nest_radius=1, food=3)
        wasp.role = WaspRole.FEEDER
        targets = np.array([[1, 0], [0.5, 0.5], [0, 1]])
        wasp.get_path(targets, np.arange(len(targets)))
        first = (wasp.x + wasp.next_step["x"], wasp.y + wasp.next_step["y"])
        route = [first] + list(wasp.path or [])
        self.assertTrue(set(map(tuple, targets)).issubset(route))
        self.assertTrue(all(x*x + y*y <= 1 for x, y in route))

    def test_feeder_does_not_plan_to_exterior_target(self):
        wasp = Wasp("route", outer_nest_radius=1, food=3)
        wasp.role = WaspRole.FEEDER
        wasp.get_path(np.array([[1.5, 0]]), np.array([0]))
        self.assertIsNone(wasp.path)
        self.assertEqual(wasp.next_step, {"x": 0, "y": 0})


if __name__ == "__main__":
    unittest.main()
