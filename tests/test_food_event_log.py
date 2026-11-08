"""Lossless event records and observational equivalence of the journal."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents import Larvae, Wasp, WaspRole
from logger import DataLogger, validate_food_event_log
from pathfinding import PATHFINDING_MODES
from smoke_test import scenario, SEED, STEPS
from simulation_fixture import LoggedSimulationTestCase


class TestFoodEventIntegration(LoggedSimulationTestCase):
    def read_events(self):
        return pd.read_csv(self.sim.logger.event_log_path, keep_default_na=False)

    def test_multiple_events_preserve_order_and_do_not_repeat_on_idle_steps(self):
        second = Larvae("L2", x=-1, y=0, hunger=3)
        self.sim.addAgent(second)

        def actions(agent, *unused):
            if agent is self.feeder and self.sim.currentTime == 1:
                agent.feedLarvae(self.larvae, 1)
                agent.feedLarvae(second, 1)

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent"):
            self.run_simulation(3)
        events = self.read_events()
        self.assertEqual(events.recipient_id.tolist(), ["L1", "L2"])
        self.assertEqual(events.timestamp.tolist(), [1, 1])
        self.assertEqual(events.event_id.tolist(), [1, 2])
        self.assertEqual(events.step_event_order.tolist(), [1, 2])
        self.assertEqual(events.amount.tolist(), [1, 1])
        self.assertTrue(events.event_type.eq("larval_feed").all())
        validate_food_event_log(self.sim.logger)
        self.assertIsNone(self.feeder._food_event_sink)

    def test_event_positions_and_roles_precede_state_updates(self):
        def actions(agent, *unused):
            if agent is self.feeder:
                agent.feedLarvae(self.larvae, 1)

        def move(agent):
            if agent is self.feeder:
                agent.x = 0.5
                agent.role = WaspRole.FORAGER

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent", side_effect=move):
            self.run_simulation(1)
        event = self.read_events().iloc[0]
        state = pd.read_csv(self.sim.logger.agent_log_path).query("agent_id == 'F1' and timestamp == 1").iloc[0]
        self.assertEqual((event.actor_x, event.actor_role), (0, "FEEDER"))
        self.assertEqual((state.position_x, state.agent_role), (0.5, "FORAGER"))
        self.assertEqual((event.recipient_x, event.recipient_y), (2, 2))

    def test_within_step_order_restarts_but_event_id_continues(self):
        def actions(agent, *unused):
            if agent is self.feeder:
                agent.feedLarvae(self.larvae, 1)

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent"):
            self.run_simulation(2)
        events = self.read_events()
        self.assertEqual(events.event_id.tolist(), [1, 2])
        self.assertEqual(events.timestamp.tolist(), [1, 2])
        self.assertEqual(events.step_event_order.tolist(), [1, 1])
        validate_food_event_log(self.sim.logger)

    def test_partial_transfer_and_collection_record_actual_amounts(self):
        self.feeder.food = self.feeder.maxFood - 0.25
        self.forager.food = 5
        self.forager.maxFood = 6
        self.forager.forageIncrease = 3
        self.forager.chanceOfForaging = 1

        def actions(agent, *unused):
            if agent is self.forager:
                self.assertEqual(agent.feedWasp(self.feeder, 1), 0.25)
                agent.forage([agent.getPosition()])

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent"):
            self.run_simulation(1)
        events = self.read_events()
        self.assertEqual(events.event_type.tolist(), ["adult_transfer", "collection"])
        self.assertEqual(events.amount.tolist(), [0.25, 1.25])
        self.assertEqual(events.recipient_id.tolist(), ["F1", ""])
        self.assertEqual((float(events.iloc[1].source_x), float(events.iloc[1].source_y)), (1, 1))
        self.assertTrue(events.amount_unit.eq("model_food").all())
        validate_food_event_log(self.sim.logger)

    def test_noop_attempts_leave_header_only_journal(self):
        self.feeder.food = 0

        def actions(agent, *unused):
            if agent is self.feeder:
                self.assertEqual(agent.feedLarvae(self.larvae, 1), 0)
                self.assertEqual(agent.feedWasp(agent, 1), 0)

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent"):
            self.run_simulation(2)
        self.assertTrue(self.read_events().empty)
        validate_food_event_log(self.sim.logger)

    def test_sink_is_restored_on_action_exception(self):
        previous = lambda *args: None
        self.feeder._food_event_sink = previous
        with patch.object(self.sim, "stepAgent", side_effect=RuntimeError("probe")):
            with self.assertRaisesRegex(RuntimeError, "probe"):
                self.run_simulation(1)
        self.assertIs(self.feeder._food_event_sink, previous)

    def test_reconciliation_rejects_missing_event(self):
        def actions(agent, *unused):
            if agent is self.feeder:
                agent.feedLarvae(self.larvae, 1)

        with patch.object(self.sim, "stepAgent", side_effect=actions), patch.object(self.sim, "moveAgent"):
            self.run_simulation(1)
        events = pd.read_csv(self.sim.logger.event_log_path)
        events.iloc[:0].to_csv(self.sim.logger.event_log_path, index=False)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_food_event_log(self.sim.logger)


class TestEventLogEquivalence(unittest.TestCase):
    def test_all_modes_keep_states_counters_and_rng_unchanged(self):
        py_state, np_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, py_state)
        self.addCleanup(np.random.set_state, np_state)
        with tempfile.TemporaryDirectory() as temporary:
            for mode in PATHFINDING_MODES:
                with self.subTest(mode=mode):
                    outcomes = []
                    for enabled in (False, True):
                        random.seed(SEED)
                        np.random.seed(SEED)
                        output = Path(temporary) / str(enabled)
                        with contextlib.redirect_stdout(io.StringIO()):
                            sim = scenario(mode, output)
                            sim.logger.enable_event_log = enabled
                            sim.runSimulation(STEPS)
                        outcomes.append((
                            [(a.id, a.getPosition(), a.food, a.hunger, a.storedEvents,
                              getattr(a, "feed_count_larvae", 0), getattr(a, "feed_count_wasp", 0),
                              getattr(a, "forage_count", 0)) for a in sim.agents],
                            [(output / mode / f"{kind}_log_{mode}.csv").read_bytes()
                             for kind in ("agent", "larvae", "nest")],
                            random.getstate(),
                            tuple(value.tolist() if isinstance(value, np.ndarray) else value
                                  for value in np.random.get_state()),
                        ))
                        if enabled:
                            validate_food_event_log(sim.logger)
                    self.assertEqual(outcomes[0], outcomes[1])

    def test_disabled_logger_does_not_create_event_file(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            logger = DataLogger("greedy", simulation_id=1, output_dir=temporary, enable_event_log=False)
            self.assertFalse(Path(logger.event_log_path).exists())


if __name__ == "__main__":
    unittest.main()
