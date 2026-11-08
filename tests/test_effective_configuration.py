import unittest
from unittest.mock import patch

import numpy as np

from agents import Larvae, Wasp
from simulator import Simulator, instanceGenerator
from run_experiment import effective_configuration


class TestEffectiveConfiguration(unittest.TestCase):
    def test_larval_rate_uses_configured_base_and_multiplier(self):
        for base, multiplier in ((0.3, 1), (0.3, 2), (0.3, 3), (0, 3)):
            with self.subTest(base=base, multiplier=multiplier):
                larva = Larvae("larva", 0, 0, hunger_rate=base, hungerMultiplier=multiplier)
                self.assertAlmostEqual(larva.hungerRate, base * multiplier)

    def test_real_metabolism_uses_effective_larval_rate(self):
        larva = Larvae("larva", 0, 0, hunger=1, food=0, hunger_rate=0.3, hungerMultiplier=2)
        sim = Simulator()
        for _ in range(4):
            sim.updateAgent(larva)
        self.assertAlmostEqual(larva.hunger, 3.4)
        self.assertEqual(larva.food, 0)
        larva.food = 2
        before = larva.hunger
        sim.updateAgent(larva)
        self.assertAlmostEqual(larva.food, 1.4)
        self.assertEqual(larva.hunger, before)

    def test_generator_reaches_l3_instead_of_repeating_l2(self):
        generator = instanceGenerator()
        generator.larvaeDictionary({"hunger_rate": 0.3, "hunger_rate_multiplier_l2": 2,
                                    "hunger_rate_multiplier_l3": 4})
        with patch("simulator.np.random.choice",
                   side_effect=lambda choices, size: np.array([choices[-1]])):
            larva = generator.generateLarvae(0, 0)
        self.assertEqual(larva.hungerMultiplier, 4)
        self.assertAlmostEqual(larva.hungerRate, 1.2)

    def test_wasp_uses_its_own_configured_rate(self):
        wasp = Wasp("wasp", outer_nest_radius=4, hunger_rate=0.1, food=2)
        Simulator().updateAgent(wasp)
        self.assertAlmostEqual(wasp.food, 1.9)

    def test_nonfinite_or_negative_metabolic_parameters_are_rejected(self):
        for bad in (-0.1, float("nan"), float("inf")):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    Larvae("larva", 0, 0, hunger_rate=bad)
                with self.assertRaises(ValueError):
                    Larvae("larva", 0, 0, hungerMultiplier=bad)
                with self.assertRaises(ValueError):
                    Wasp("wasp", outer_nest_radius=4, hunger_rate=bad)

    def test_legacy_generator_hunger_fields_do_not_override_larvae_section(self):
        generator = instanceGenerator(hunger_rate=9, larvae_hunger_multiplier=99)
        generator.larvaeDictionary({"hunger_rate": 0.3, "hunger_rate_multiplier_l2": 2,
                                    "hunger_rate_multiplier_l3": 4})
        with patch("simulator.np.random.choice",
                   side_effect=lambda choices, size: np.array([choices[0]])):
            larva = generator.generateLarvae(0, 0)
        self.assertAlmostEqual(larva.hungerRate, 0.3)

    def test_invalid_unselected_multiplier_is_rejected_at_configuration_load(self):
        with self.assertRaises(ValueError):
            instanceGenerator().larvaeDictionary({
                "hunger_rate": 0.3, "hunger_rate_multiplier_l2": 2,
                "hunger_rate_multiplier_l3": -1,
            })

    def test_manifest_report_reflects_actual_rates_and_ignored_keys(self):
        sim = Simulator()
        sim.addAgent(Wasp("adult", outer_nest_radius=4, hunger_rate=0.1))
        sim.addAgent(Larvae("larva", 0, 0, hunger_rate=0.3, hungerMultiplier=2))
        result = effective_configuration({"instance_generator": {"hunger_rate": 9}}, sim)
        self.assertEqual(result["effective_adult_rates"], [0.1])
        self.assertEqual(result["effective_larval_rates"], [0.6])
        self.assertEqual(result["ignored_legacy_keys"], ["instance_generator.hunger_rate"])


if __name__ == "__main__":
    unittest.main()
