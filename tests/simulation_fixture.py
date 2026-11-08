"""Current-API simulation fixture with isolated logs and random state."""
import contextlib
import io
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
from agents import Larvae, Wasp, WaspRole
from logger import DataLogger
from simulator import Simulator

class LoggedSimulationTestCase(unittest.TestCase):
    def setUp(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, python_state)
        self.addCleanup(np.random.set_state, numpy_state)
        random.seed(42)
        np.random.seed(42)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.logs = Path(temporary.name) / "greedy"
        self.sim = Simulator()
        self.feeder = Wasp("F1", outer_nest_radius=4, x=0, y=0, food=5)
        self.feeder.role = WaspRole.FEEDER
        self.forager = Wasp("Fo1", outer_nest_radius=4, x=1, y=1, food=5)
        self.forager.role = WaspRole.FORAGER
        for wasp in (self.feeder, self.forager):
            wasp.updateHungerCueRange()
        self.larvae = Larvae("L1", x=2, y=2, hunger=3)
        for agent in (self.feeder, self.forager, self.larvae):
            self.sim.addAgent(agent)
        self.sim.addForage(3, 3)
        self.sim.config = {"simulator": {"pathfinding_mode": "greedy"}}
        with contextlib.redirect_stdout(io.StringIO()):
            self.sim.logger = DataLogger("greedy", simulation_id=1, output_dir=temporary.name)

    def run_simulation(self, steps):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.sim.runSimulation(steps)
