# simulator.py
# Defines the Simulator class that manages agents and runs the simulation
from __future__ import annotations

from typing import List, Optional, Dict, Tuple
from agents import Agent, Wasp, Larvae
from agents import AgentType, WaspRole
import numpy as np
from utils import gaussian_attraction, in_circle, outside_circle
from logger import DataLogger  #Added: Data logging for simulation


class instanceGenerator:
    r"""
    Factory for randomized simulation instances (grid, larvae, wasps, forage sites).

    The generator draws grid sizes and placements stochastically, then
    populates larvae and wasps according to the provided ratios and
    fill percentages. It finally returns a :class:`Simulator` populated
    with agents and foraging locations.

    :param larvae_to_wasps_ratio: Ratio of larvae count to wasp count used to derive
                                  the number of wasps from the number of larvae.
    :type larvae_to_wasps_ratio: float
    :param percentage_foragers: Fraction of wasps assigned the ``FORAGER`` role.
    :type percentage_foragers: float
    :param min_number_of_cells: Lower bound on the number of grid cells used to set nest radius.
    :type min_number_of_cells: int
    :param max_number_of_cells: Upper bound on the number of grid cells used to set nest radius.
    :type max_number_of_cells: int
    :param nest_fill_percentage: Fraction of available nest cells to populate with larvae.
                                 Split equally between inner and outer nest regions.
    :type nest_fill_percentage: float
    :param forage_fill_percentage: Fraction of available forage ring cells to activate as forage sites.
    :type forage_fill_percentage: float
    :param larvae_hunger_multiplier: Multiplier applied to larvae hunger rate for inner-nest larvae.
    :type larvae_hunger_multiplier: float
    :param mean_food_capacity: Mean of the normal distribution used for wasp capacity sampling.
    :type mean_food_capacity: float
    :param std_food_capacity: Standard deviation of capacity sampling for wasps.
    :type std_food_capacity: float
    :param forage_distance: Radial gap between nest radius and forage ring radius.
    :type forage_distance: int
    :param hunger_rate: Base per-step hunger rate assigned to generated agents.
    :type hunger_rate: float
    :param proportion_inner_larvae: Fraction of the nest radius defining the inner-nest radius.
    :type proportion_inner_larvae: float
    :param grid_buffer: Buffer added to nest and forage ring radii to ensure they
    :type grid_buffer: int
    :param forage_buffer: Buffer added to nest and forage ring radii to ensure they
    :type forage_buffer: int

    **Derived attributes**

    :ivar larvae_to_wasps_ratio: See parameter of the same name.
    :vartype larvae_to_wasps_ratio: float
    :ivar percentage_foragers: See parameter of the same name.
    :vartype percentage_foragers: float
    :ivar min_number_of_cells: See parameter of the same name.
    :vartype min_number_of_cells: int
    :ivar max_number_of_cells: See parameter of the same name.
    :vartype max_number_of_cells: int
    :ivar nest_fill_percentage: See parameter of the same name.
    :vartype nest_fill_percentage: float
    :ivar mean_food_capacity: See parameter of the same name.
    :vartype mean_food_capacity: float
    :ivar std_food_capacity: See parameter of the same name.
    :vartype std_food_capacity: float
    :ivar forage_distance: See parameter of the same name.
    :vartype forage_distance: int
    :ivar forage_fill_percentage: See parameter of the same name.
    :vartype forage_fill_percentage: float
    :ivar larvae_hunger_multiplier: See parameter of the same name.
    :vartype larvae_hunger_multiplier: float
    :ivar agent_hunger_ratio: Base per-step hunger rate assigned to agents.
    :vartype agent_hunger_ratio: float
    :ivar proportion_inner_larvae: See parameter of the same name.
    :vartype proportion_inner_larvae: float
    :ivar grid_buffer: Extra cells included when building the grid (padding).
    :vartype grid_buffer: int
    :ivar forage_buffer: Extra padding around the forage radius to form a ring.
    :vartype forage_buffer: int
    """

    def __init__(
        self,
        larvae_to_wasps_ratio: float = 0.3,
        percentage_foragers: float = 0.1,
        min_number_of_cells: int = 100,
        max_number_of_cells: int = 120,
        nest_fill_percentage: float = 0.3,
        forage_fill_percentage: float = 0.1,
        larvae_hunger_multiplier: float = 3.0,
        mean_food_capacity: float = 10.0,
        std_food_capacity: float = 3.0,
        forage_distance: int = 10,
        hunger_rate: float = 0.2,
        proportion_inner_larvae: float = 0.5,
        grid_buffer:int = 1,
        forage_buffer:int = 10

    ) -> None:
        self.larvae_to_wasps_ratio = larvae_to_wasps_ratio
        self.percentage_foragers = percentage_foragers
        self.min_number_of_cells = min_number_of_cells
        self.max_number_of_cells = max_number_of_cells
        self.nest_fill_percentage = nest_fill_percentage
        self.mean_food_capacity = mean_food_capacity
        self.std_food_capacity = std_food_capacity
        self.forage_distance = forage_distance
        self.forage_fill_percentage = forage_fill_percentage
        self.larvae_hunger_multiplier = larvae_hunger_multiplier
        self.agent_hunger_ratio = hunger_rate
        self.proportion_inner_larvae = proportion_inner_larvae
        self.grid_buffer = grid_buffer
        self.forage_buffer = forage_buffer

    # -------------------------
    # Building blocks
    # -------------------------
    def waspDictionary(self, wasp_config: dict):
        self.wasp_dictionary = wasp_config
    def larvaeDictionary(self, larvae_config: dict):
        self.larvae_dictionary = larvae_config
    def simulatorDictionary(self, simulator_config: dict):
        self.simulator_dictionary = simulator_config
    def generateLarvae(self, x: int, y: int) -> "Larvae":
        """
        Create a :class:`Larvae` at a grid location with randomized initial food/hunger.

        :param x: X-coordinate.
        :type x: int
        :param y: Y-coordinate.
        :type y: int
        :param hunger_multiplier: Multiplier applied to the larvae's base hunger rate.
        :type hunger_multiplier: float
        :return: Newly created larvae agent.
        :rtype: Larvae
        """
        random_food = np.random.choice([0.9, 3, 5], 1)[0]
        random_hunger = 1 if random_food > 1 else 2
        agent_id = "L" + str(x) + str(y)
        random_hunger_multiplier = np.random.choice([1.0, self.larvae_dictionary['hunger_rate_multiplier']], 1)[0]
        return Larvae(
            agent_id,
            x=x,
            y=y,
            hunger=random_hunger,
            food=random_food,
            hungerMultiplier= random_hunger_multiplier,
            hunger_rate=self.larvae_dictionary['hunger_rate'],
        )

    def generateGrid(self, radius: int) -> np.ndarray:
        """
        Build a rectangular grid covering ``[-radius, radius + grid_buffer)`` in both axes.

        :param radius: Half-width of the square region (exclusive of the last index).
        :type radius: int
        :return: ``(N, 2)`` array of integer coordinates (x, y).
        :rtype: numpy.ndarray
        """
        grid_x, grid_y = np.mgrid[-radius : radius + self.grid_buffer, -radius : radius + self.grid_buffer]
        grid = np.concatenate((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1)), axis=1)
        return grid

    def addForaging(self, chosen_forage_indices: np.ndarray, grid: np.ndarray, simulator: "Simulator") -> "Simulator":
        """
        Add forage locations to the simulator at selected grid indices.

        :param chosen_forage_indices: Indices into ``grid`` identifying forage cells.
        :type chosen_forage_indices: numpy.ndarray
        :param grid: ``(N, 2)`` coordinate array.
        :type grid: numpy.ndarray
        :param simulator: Simulator to mutate.
        :type simulator: Simulator
        :return: The same simulator, for chaining.
        :rtype: Simulator
        """
        for row in grid[chosen_forage_indices, :]:
            simulator.addForage(row[0], row[1])
        return simulator

    def addLarvaes(  # (kept name for backward compatibility)
        self,
        chosen_nest_indices: np.ndarray,
        chosen_inner_nest_indices: np.ndarray,
        grid: np.ndarray,
        simulator: "Simulator",
    ) -> "Simulator":
        """
        Add larvae to inner and outer nest regions.

        :param chosen_nest_indices: Indices for outer-nest larvae.
        :type chosen_nest_indices: numpy.ndarray
        :param chosen_inner_nest_indices: Indices for inner-nest larvae (use hunger multiplier).
        :type chosen_inner_nest_indices: numpy.ndarray
        :param grid: ``(N, 2)`` coordinate array.
        :type grid: numpy.ndarray
        :param simulator: Simulator to mutate.
        :type simulator: Simulator
        :return: The same simulator, for chaining.
        :rtype: Simulator
        """
        for row in grid[chosen_inner_nest_indices, :]:
            simulator.addAgent(self.generateLarvae(row[0], row[1],))
        for row in grid[chosen_nest_indices, :]:
            simulator.addAgent(self.generateLarvae(row[0], row[1]))
        return simulator

    def addWasps(
        self,
        chosen_nest_indices_feeders: np.ndarray,
        chosen_inner_nest_indices_foragers: np.ndarray,
        grid: np.ndarray,
        simulator: "Simulator",
    ) -> "Simulator":
        """
        Add wasps to the simulator and assign roles to selected indices.

        :param chosen_nest_indices_feeders: Indices (into ``grid``) for feeder wasps.
        :type chosen_nest_indices_feeders: numpy.ndarray
        :param chosen_inner_nest_indices_foragers: Indices (into ``grid``) for forager wasps.
        :type chosen_inner_nest_indices_foragers: numpy.ndarray
        :param grid: ``(N, 2)`` coordinate array.
        :type grid: numpy.ndarray
        :param simulator: Simulator to mutate.
        :type simulator: Simulator
        :return: The same simulator, for chaining.
        :rtype: Simulator
        """
        for row in chosen_nest_indices_feeders:
            wasp = self.generateWasp(grid[row, 0], grid[row, 1])
            wasp.role = WaspRole.FEEDER
            simulator.addAgent(wasp)
        for row in chosen_inner_nest_indices_foragers:
            wasp = self.generateWasp(grid[row, 0], grid[row, 1])
            wasp.role = WaspRole.FORAGER
            simulator.addAgent(wasp)
        return simulator

    def generateWasp(self, x: int, y: int) -> "Wasp":
        """
        Create a :class:`Wasp` at a grid location with sampled capacity and defaults.

        :param x: X-coordinate.
        :type x: int
        :param y: Y-coordinate.
        :type y: int
        :param path_finding: Path-finding mode to assign.
        :type path_finding: str
        :param outer_nest_radius: Optional nest radius constraint for the wasp.
        :type outer_nest_radius: int | None
        :return: Newly created wasp agent.
        :rtype: Wasp
        """
        max_food = np.random.normal(self.mean_food_capacity, self.std_food_capacity, 1)[0]
        wasp = Wasp(
                agent_id="W" + str(x) + str(y),
                x=x,
                y=y,
                max_food=max_food,
                **self.wasp_dictionary,
            )
        return wasp

    # -------------------------
    # High-level instance maker
    # -------------------------

    def generateSimulator(self) -> "Simulator":
        """
        Create and populate a :class:`Simulator` with grid, larvae, wasps, and forage.

        Steps:
        1. Draw number of cells and derive radii (nest, inner nest, forage).
        2. Build grid with padding.
        3. Sample inner/outer larvae placements and forage ring cells.
        4. Compute wasp counts and split into foragers/feeders.
        5. Populate the simulator accordingly.

        :return: A populated simulator instance.
        :rtype: Simulator
        """
        simulator = Simulator()

        number_of_cells = np.random.randint(self.min_number_of_cells, self.max_number_of_cells + 1)
        radius_nest = int(number_of_cells ** (1 / 2)) + 1
        radius_inner_nest = int(radius_nest * self.proportion_inner_larvae)
        radius_forage = radius_nest + self.forage_distance

        grid = self.generateGrid(radius_forage + self.forage_buffer)
        simulator.grid = grid

        inner_nest_indices = np.where(in_circle(grid[:, 0], grid[:, 1], radius_inner_nest))[0]
        inner_chosen_nest_indices_larvaes = np.random.choice(
            inner_nest_indices,
            int(self.nest_fill_percentage / 2 * inner_nest_indices.shape[0]),
            replace=False,
        )

        nest_indices = np.where(
            in_circle(grid[:, 0], grid[:, 1], radius_nest)
            & outside_circle(grid[:, 0], grid[:, 1], radius_inner_nest)
        )[0]
        chosen_nest_indices_larvaes = np.random.choice(
            nest_indices, int(self.nest_fill_percentage / 2 * nest_indices.shape[0]), replace=False
        )

        forage_indices = np.where(
            in_circle(grid[:, 0], grid[:, 1], radius_forage)
            & outside_circle(grid[:, 0], grid[:, 1], radius_nest + int(self.forage_buffer / 2))
        )[0]
        chosen_forage_indices_forage = np.random.choice(
            forage_indices, int(self.forage_fill_percentage * forage_indices.shape[0]), replace=False
        )

        total_wasps = int(
            (len(chosen_nest_indices_larvaes) + len(inner_chosen_nest_indices_larvaes)) * self.larvae_to_wasps_ratio
        )
        total_foragers = max(1, int(total_wasps * self.percentage_foragers))
        total_feeders = total_wasps - total_foragers

        chosen_nest_indices_feeders = np.random.choice(inner_nest_indices, total_feeders, replace=False)
        chosen_nest_indices_foragers = np.random.choice(nest_indices, total_foragers, replace=False)

        simulator = self.addForaging(chosen_forage_indices_forage, grid, simulator)
        simulator = self.addLarvaes(chosen_nest_indices_larvaes, inner_chosen_nest_indices_larvaes, grid, simulator)
        simulator = self.addWasps(
            chosen_nest_indices_feeders, chosen_nest_indices_foragers, grid, simulator
        )

        return simulator



class Simulator:
    r"""
    Manage a collection of agents and coordinate the simulation.

    The simulator advances time, maintains a grid and forage sites, accumulates
    gradient fields, and aggregates results from agents' actions.

    :param forager_ratio: Ratio of forager wasps to total wasps.
    :type forager_ratio: float
    :param potential_feeder_to_forager: Probability of a feeder wasp changing to a forager wasp.
    :type potential_feeder_to_forager: float
    :param max_role_changes: Maximum number of role changes allowed for a wasp.
    :type max_role_changes: int
    :param role_changes_frequency: Frequency at which role changes are considered.
    :type role_changes_frequency: int

    **Attributes initialized in ``__init__``**

    :ivar currentTime: Current simulation step (time index).
    :vartype currentTime: int
    :ivar agents: List of all agents in the simulation.
    :vartype agents: List[Agent]
    :ivar movementHistory: Map from agent ID to its sequence of positions.
    :vartype movementHistory: Dict[str, List[List[int]]]
    :ivar gradients: Accumulated gradient fields keyed by :class:`WaspRole`.
    :vartype gradients: Dict[WaspRole, np.ndarray | list]
    :ivar grid: Flattened grid coordinates ``(N, 2)`` or ``None`` if not created.
    :vartype grid: numpy.ndarray | None
    :ivar forage: List of forage locations (each as a length-2 array).
    :vartype forage: List[numpy.ndarray]
    :ivar forager_ratio: Target ratio of foragers among wasps.
    :vartype forager_ratio: float
    :ivar potential_feeder_to_forager: Additional allowance for converting feeders to foragers.
    :vartype potential_feeder_to_forager: float
    :ivar max_role_changes: Max role changes allowed per ``role_changes_frequency`` window.
    :vartype max_role_changes: int
    :ivar role_changes_frequency: Steps between role reassessment opportunities.
    :vartype role_changes_frequency: int
    """

    def __init__(self, forager_ratio: float=0.10, potential_feeder_to_forager: float=0.25, max_role_changes: int=2, role_changes_frequency: int=100) -> None:
        """Initialize the simulator and default state."""
        self.currentTime: int = 0
        self.agents: List[Agent] = []
        self.movementHistory: Dict[str, List[List[int]]] = {}
        self.gradients: Dict[WaspRole, List[float] | np.ndarray] = {WaspRole.FEEDER: [], WaspRole.FORAGER: []}
        self.grid: Optional[np.ndarray] = None
        self.forage: List[np.ndarray] = []
        self.forager_ratio: float = forager_ratio
        self.potential_feeder_to_forager: float = potential_feeder_to_forager
        self.max_role_changes: int = max_role_changes
        self.role_changes_frequency: int = role_changes_frequency
        # Initialize DataLogger
        self.logger = DataLogger(
            agent_log_path="agent_log.csv",
            nest_log_path="nest_log.csv"
        )

    # ---------------------------
    # Core methods
    # ---------------------------

    def step(self) -> None:
        """
        Advance the simulation by one time unit.

        Calls ``step(t=..., agents=..., forage=...)`` on each agent, then
        increments :attr:`currentTime`.

        :return: ``None``.
        :rtype: None
        """
        for agent in self.agents:
            agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)
        self.currentTime += 1

    def addAgent(self, agent: Agent) -> None:
        """
        Add a new agent to the simulation and initialize its movement history.

        :param agent: Agent to add.
        :type agent: Agent
        :return: ``None``.
        :rtype: None
        """
        self.agents.append(agent)
        self.movementHistory[agent.id] = [agent.getPosition()]

    def addForage(self, x: int, y: int) -> None:
        """
        Add a foraging location.

        :param x: X-coordinate of the forage site.
        :type x: int
        :param y: Y-coordinate of the forage site.
        :type y: int
        :return: ``None``.
        :rtype: None
        """
        self.forage.append(np.array([x, y]))

    def removeAgent(self, agent: Agent) -> None:
        """
        Remove an agent from the simulation (if present).

        :param agent: Agent to remove.
        :type agent: Agent
        :return: ``None``.
        :rtype: None
        """
        if agent in self.agents:
            self.agents.remove(agent)

    def accumulateGradients(self) -> None:
        """
        Accumulate larvae-generated gradients into the role-specific fields.

        For each hungry larva, computes a Gaussian-like field centered on the larva
        and adds it to both feeder and forager gradient buffers. If gradient arrays
        are uninitialized, they are created to match the current grid size.

        :return: ``None``.
        :rtype: None
        """
        agents = self.agents
        larvae = [agent for agent in agents if agent.type == AgentType.LARVAE]

        if len(self.gradients[WaspRole.FEEDER]) == 0:
            self.gradients[WaspRole.FEEDER] = np.zeros_like(self.grid[:, 0])
        if len(self.gradients[WaspRole.FORAGER]) == 0:
            self.gradients[WaspRole.FORAGER] = np.zeros_like(self.grid[:, 0])

        for agent in larvae:
            if agent.hunger > agent.noHunger:
                x0, y0 = agent.getPosition()
                spread = agent.radius
                peak = max(agent.hunger / min(agent.food + 0.1, 0.1), 0.1)
                gradient = gaussian_attraction(self.grid[:, 0], self.grid[:, 1], x0, y0, spread, peak)

                self.gradients[WaspRole.FEEDER] = self.gradients[WaspRole.FEEDER] + gradient
                self.gradients[WaspRole.FORAGER] = self.gradients[WaspRole.FORAGER] + gradient

    def aggregateMovements(self) -> Dict[str, List[List[int]]]:
        """
        Collect movement data for all agents.

        :return: Mapping from agent ID to list of positions.
        :rtype: Dict[str, List[List[int]]]
        """
        # If you want to record positions every call, uncomment:
        # for agent in self.agents:
        #     self.movementHistory[agent.id].append(agent.getPosition())
        return self.movementHistory

    def aggregateFeedLarvae(self) -> Dict[str, Dict[str, int]]:
        """
        Collect feeding events performed by wasps.

        Parses wasp event logs and counts how many times each target was fed.

        :return: Mapping ``{wasp_id: {target_id: count}}``.
        :rtype: Dict[str, Dict[str, int]]
        """
        result: Dict[str, Dict[str, int]] = {}
        for agent in self.agents:
            if isinstance(agent, Wasp):
                result[agent.id] = {}
                for event in agent.storedEvents:
                    if "fed" in event:
                        parts = event.split()
                        if len(parts) >= 3:
                            target = parts[2]
                            result[agent.id][target] = result[agent.id].get(target, 0) + 1
        return result

    def aggregateHungerLarvae(self) -> Dict[str, List[float]]:
        """
        Collect hunger values for larvae agents.

        :return: Mapping from larvae ID to a list containing current hunger.
        :rtype: Dict[str, List[float]]
        """
        return {a.id: [a.hunger] for a in self.agents if isinstance(a, Larvae)}

    def aggregateHungerWasp(self) -> Dict[str, List[float]]:
        """
        Collect hunger values for wasp agents.

        :return: Mapping from wasp ID to a list containing current hunger.
        :rtype: Dict[str, List[float]]
        """
        return {a.id: [a.hunger] for a in self.agents if isinstance(a, Wasp)}

    def verifyNumberAgents(self, min_feeders: int = 1, min_foragers: int = 1, min_larvae: int = 1) -> bool:
        """
        Verify that minimum agent counts are satisfied.

        :param min_feeders: Minimum number of feeder wasps.
        :type min_feeders: int
        :param min_foragers: Minimum number of forager wasps.
        :type min_foragers: int
        :param min_larvae: Minimum number of larvae.
        :type min_larvae: int
        :return: ``True`` if all minima are met, else ``False``.
        :rtype: bool
        """
        count_feeders = 0
        count_larvae = 0
        count_foragers = 0
        for agent in self.agents:
            if isinstance(agent, Wasp):
                if agent.role == WaspRole.FEEDER:
                    count_feeders += 1
                elif agent.role == WaspRole.FORAGER:
                    count_foragers += 1
            elif isinstance(agent, Larvae):
                count_larvae += 1
        return (count_feeders >= min_feeders) and (count_foragers >= min_foragers) and (count_larvae >= min_larvae)

    def createGrid(self, padding: int = 3) -> None:
        """
        Create a grid covering the bounding box of current agent positions.

        The grid is stored as a flattened ``(N, 2)`` array of integer coordinates.
        Gradient arrays are initialized (zeros) for each :class:`WaspRole`.

        :param padding: Extra cells to include around the bounding box.
        :type padding: int
        :return: ``None``.
        :rtype: None
        """
        positions_dict = {"x": [agent.x for agent in self.agents], "y": [agent.y for agent in self.agents]}
        xmin = min(positions_dict["x"])
        xmax = max(positions_dict["x"])
        ymin = min(positions_dict["y"])
        ymax = max(positions_dict["y"])

        x, y = np.meshgrid(
            np.arange(xmin - padding, xmax + padding + 1),
            np.arange(ymin - padding, ymax + padding + 1),
        )
        self.grid = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)

        for role in self.gradients:
            self.gradients[role] = np.zeros(self.grid.shape[0])

    def verifyGrid(self) -> bool:
        """
        Check that the grid is initialized and has shape ``(N, 2)``.

        :return: ``True`` if a valid grid exists, else ``False``.
        :rtype: bool
        """
        return (self.grid is not None) and (self.grid.shape[0] > 0) and (self.grid.shape[1] == 2)

    def verifyForage(self) -> bool:
        """
        Check that at least one forage point exists.

        :return: ``True`` if any forage points are present, else ``False``.
        :rtype: bool
        """
        return len(self.forage) > 0

    def verifySimulationConditions(self) -> bool:
        """
        Verify that the simulation is ready to run.

        Conditions:
          1. Minimum agent counts are satisfied.
          2. The grid is initialized and valid (created on-demand if ``None``).
          3. At least one forage point exists.

        :return: ``True`` if all conditions hold, else ``False``.
        :rtype: bool
        """
        num_agents_ok = self.verifyNumberAgents()
        if self.grid is None:
            self.createGrid()
        grid_ok = self.verifyGrid()
        forage_ok = self.verifyForage()
        return num_agents_ok and grid_ok and forage_ok

    def runSimulation(self, t: int) -> Dict[str, Dict]:
        """
        Run the simulation for ``t`` steps and return an aggregate report.

        During each step:
          - Accumulate larvae gradients.
          - For each wasp, sense gradients and step/move.
          - Update agents' hunger/food and possibly adjust roles.
          - Advance :attr:`currentTime` and clear gradient buffers.

        :param t: Number of steps to simulate.
        :type t: int
        :raises ValueError: If simulation conditions are not met.
        :return: Report with keys ``movements``, ``feedLarvae``, ``hungerLarvae``, ``hungerWasp``.
        :rtype: Dict[str, Dict]
        """
        if not self.verifySimulationConditions():
            raise ValueError("Simulation conditions not met")

        i = 0
        larvae = [agent for agent in self.agents if isinstance(agent, Larvae)]
        larvaes_position = [agent.getPosition() for agent in larvae]
        wasps = [agent for agent in self.agents if isinstance(agent, Wasp)]
        total_wasp = len(wasps)
        total_foragers = sum(1 for agent in wasps if agent.role == WaspRole.FORAGER)

        while i < t:
            self.accumulateGradients()

            count_roles = 0
            j = 0
            while j < len(self.agents):
                agent = self.agents[j]
                if isinstance(agent, Wasp):
                    self.stepAgent(agent, wasps, larvaes_position)
                j += 1

            j = 0
            while j < len(self.agents):
                agent = self.agents[j]
                if isinstance(agent, Wasp):
                    self.moveAgent(agent)
                    # Log agent state for this timestep
                    if isinstance(agent, (Wasp, Larvae)):
                        self.logger.log_agent(
                            timestamp=self.currentTime,
                            agent_id=agent.id,
                            agent_role=str(agent.role.name if isinstance(agent, Wasp) else "Larvae"),
                            action=agent.storedEvents[-1] if agent.storedEvents else "none",
                            target_id=self._parse_target_from_event(agent.storedEvents[-1]) if agent.storedEvents else "",
                            position_x=agent.x,
                            position_y=agent.y,
                            hunger_level=getattr(agent, "hunger", 0),
                            hunger_cue=getattr(agent, "hunger_cue", 0),
                            food_stored=getattr(agent, "food", 0),
                            nest_layer=(
                                1 if isinstance(agent, Larvae)
                                else 2 if agent.role == WaspRole.FEEDER
                                else 3
                            ),
                            larvae_hunger_avg=self._calc_avg_hunger(AgentType.LARVAE),
                            total_food_in_nest=self._calc_total_food(),
                            rush_intensity=self._estimate_rush(),
                            exploration_bias=0.0
                        )

                self.updateAgent(agent)
                if isinstance(agent, Wasp):
                    count_roles, total_foragers = self.updateRoles(
                        t, agent, count_roles, total_foragers, total_wasp
                    )
                j += 1

            self.currentTime += 1
            # Log nest-level stats once per timestep
            self.logger.log_nest(
                timestamp=self.currentTime,
                total_foragers=len([a for a in self.agents if isinstance(a, Wasp) and a.role == WaspRole.FORAGER]),
                total_feeders=len([a for a in self.agents if isinstance(a, Wasp) and a.role == WaspRole.FEEDER]),
                foraging_events=sum("foraged" in e for a in self.agents for e in getattr(a, "storedEvents", [])),
                feeding_events=sum("fed" in e and "transfer" not in e for a in self.agents for e in getattr(a, "storedEvents", [])),
                transfer_events=sum("transfer" in e for a in self.agents for e in getattr(a, "storedEvents", [])),
                avg_hunger_foragers=self._calc_avg_hunger(WaspRole.FORAGER),
                avg_hunger_feeders=self._calc_avg_hunger(WaspRole.FEEDER),
                avg_hunger_larvae=self._calc_avg_hunger(AgentType.LARVAE),
                food_balance_in_nest=self._calc_total_food(),
                rush_intensity=self._estimate_rush(),
                exploration_bias=0.0,
                active_cells=len([a for a in self.agents if isinstance(a, Larvae) and getattr(a, "hunger", 0) > 1]),
                nest_size=len(self.agents)
            )

            # ======================================================================
            # Added for larvae logging (per timestep)
            # ----------------------------------------------------------------------
            for larva in [a for a in self.agents if a.__class__.__name__ == "Larvae"]:
                self.logger.log_larvae(
                    timestamp=self.currentTime,
                    larva_id=getattr(larva, "id", "L?"),
                    position_x=larva.x,
                    position_y=larva.y,
                    hunger_level=larva.hunger,
                    food_received=getattr(larva, "food", 0),
                    distance_to_nest=np.sqrt(larva.x**2 + larva.y**2)
                )

            if i % 10 == 0:
                print(f"Step {i}")
            i += 1
            self.clearGradients()
        
        # ======================================================================
        # Added for aggregate logging (after simulation ends)
        # ----------------------------------------------------------------------
        from datetime import datetime
        wasps = [a for a in self.agents if a.__class__.__name__ == "Wasp"]
        larvae = [a for a in self.agents if a.__class__.__name__ == "Larvae"]

        feed_counts = [w.feed_count for w in wasps] if wasps else []
        mean_feed_freq = np.mean(feed_counts) if feed_counts else 0
        std_feed_freq = np.std(feed_counts) if feed_counts else 0
        num_wasps = len(wasps)
        num_larvae = len(larvae)
        wasp_to_larvae_ratio = num_wasps / num_larvae if num_larvae > 0 else 0

        self.logger.log_aggregate(
            simulation_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            pathfinding_mode=getattr(self.agents[0], "path_finding", "unknown"),
            mean_hunger_larvae=np.mean([a.hunger for a in self.agents if a.__class__.__name__ == "Larvae"]),
            max_hunger_larvae=np.max([a.hunger for a in self.agents if a.__class__.__name__ == "Larvae"]),
            min_hunger_larvae=np.min([a.hunger for a in self.agents if a.__class__.__name__ == "Larvae"]),
            mean_distance_per_feed=self._calc_distance_per_feed(),
            feeding_efficiency=self._calc_feeding_efficiency(),
            mean_feed_freq=mean_feed_freq,  # new field
            std_feed_freq=std_feed_freq,    # new column
            num_wasps=num_wasps,                 # 
            num_larvae=num_larvae,               # 
            wasp_to_larvae_ratio=wasp_to_larvae_ratio  # 
        )

        report: Dict[str, Dict] = {}
        report["movements"] = self.aggregateMovements()
        report["feedLarvae"] = self.aggregateFeedLarvae()
        report["hungerLarvae"] = self.aggregateHungerLarvae()
        report["hungerWasp"] = self.aggregateHungerWasp()
        return report

    # ======================================================================
    #  Added helper metrics
    # ----------------------------------------------------------------------
    def _calc_distance_per_feed(self):
        distances = []
        for wasp in [a for a in self.agents if a.__class__.__name__ == "Wasp"]:
            if wasp.feed_count > 0 and hasattr(wasp, "distance_traveled"):
                distances.append(wasp.distance_traveled / wasp.feed_count)
        return np.mean(distances) if distances else 0

    def _calc_feeding_efficiency(self):
        larvae = [a for a in self.agents if a.__class__.__name__ == "Larvae"]
        if not larvae:
            return 0
        fed = sum(1 for a in larvae if getattr(a, "food", 0) > 0)
        return fed / len(larvae)

    # ---------------------------
    # Internal helpers
    # ---------------------------

    def updateAgent(self, agent: Agent) -> None:
        """
        Update an agent's hunger and food given its current stores.

        - If ``food < 1``, increase hunger by ``hungerRate``.
        - Reduce ``food`` by ``hungerRate`` if sufficient, else set to 0.

        :param agent: Agent to update.
        :type agent: Agent
        :return: ``None``.
        :rtype: None
        """
        if agent.food < 1:
            agent.hunger += agent.hungerRate
        if agent.food - agent.hungerRate > 0:
            agent.food -= agent.hungerRate
        else:
            agent.food = 0

    def stepAgent(self, agent: Wasp, wasps: List[Wasp], larvaes_position: List[List[int]]) -> None:
        """
        Execute perception and stepping logic for a wasp.

        :param agent: Wasp to step.
        :type agent: Wasp
        :param wasps: All wasp agents (for role-based interactions).
        :type wasps: List[Wasp]
        :param larvaes_position: Positions of all larvae.
        :type larvaes_position: List[List[int]]
        :return: ``None``.
        :rtype: None
        """
        if agent.inOuterNest():
            local_hunger_cue = agent.estimateLocalHungerCue(self.gradients[agent.role], self.grid)
            agent.updateHungerCue(local_hunger_cue / self.gradients[agent.role].shape[0])

        position_foragers = [agent_.getPosition() for agent_ in wasps if agent_.role == WaspRole.FORAGER]
        position_wasp = [wasp.getPosition() for wasp in wasps if agent.id != wasp.id]

        if agent.role == WaspRole.FORAGER:
            agent.feelGradient(self.grid, self.gradients, self.forage, larvaePositions=np.array(larvaes_position))
            agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)
        elif agent.role == WaspRole.FEEDER:
            agent.feelGradient(
                self.grid,
                self.gradients,
                foragersPositions=np.array(position_foragers),
                waspPositions=np.array(position_wasp),
                larvaePositions=np.array(larvaes_position),
            )
            agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)
        else:
            agent.step(t=self.currentTime)

    def moveAgent(self, agent: Wasp) -> None:
        """
        Move a wasp applying collision avoidance and record its position.

        :param agent: Wasp to move.
        :type agent: Wasp
        :return: ``None``.
        :rtype: None
        """
        wasp_agents = [a for a in self.agents if isinstance(a, Wasp)]
        agent_next_positions = np.array(
            [[wasp.x + wasp.next_step["x"], wasp.y + wasp.next_step["y"]] for wasp in wasp_agents if wasp != agent]
        )
        agent.move(agent_next_positions, self.grid)

        current_pos = agent.getPosition()
        if self.movementHistory[agent.id][-1] != current_pos:
            self.movementHistory[agent.id].append(current_pos)

    def updateRoles(
        self,
        t: int,
        agent: Wasp,
        count_roles: int,
        total_foragers: int,
        total_wasp: int,
    ) -> Tuple[int, int]:
        """
        Possibly update a wasp's role based on cues, persistence, and global ratios.

        :param t: Current time horizon used for role-change scheduling.
        :type t: int
        :param agent: Wasp under consideration.
        :type agent: Wasp
        :param count_roles: Number of role changes performed in the current window.
        :type count_roles: int
        :param total_foragers: Current total number of foragers.
        :type total_foragers: int
        :param total_wasp: Total number of wasps.
        :type total_wasp: int
        :return: Updated ``(count_roles, total_foragers)``.
        :rtype: Tuple[int, int]
        """
        if t % self.role_changes_frequency == 0 and count_roles <= self.max_role_changes:
            if (
                agent.role == WaspRole.FEEDER
                and agent.hungerCuesHigh()
                and ((total_foragers + 1) / total_wasp) < (self.forager_ratio + self.potential_feeder_to_forager)
            ):
                if np.random.random() > agent.feeder_to_forager_probability and agent.rolePersistence == 0 and agent.inOuterNest():
                    agent.role = WaspRole.FORAGER
                    total_foragers += 1
                    agent.updateRolePersistence()
                    count_roles += 1

            if (
                agent.role == WaspRole.FORAGER
                and agent.hungerCuesLow()
                and ((total_foragers - 1) / total_wasp) > max((1 / total_wasp), (self.forager_ratio))
            ):
                if np.random.random() > agent.forager_to_feeder_probability and agent.rolePersistence == 0 and agent.inOuterNest():
                    agent.role = WaspRole.FEEDER
                    total_foragers -= 1
                    agent.updateRolePersistence()
                    count_roles += 1

        if agent.rolePersistence > 0:
            agent.rolePersistence -= 1
        return count_roles, total_foragers

    def clearGradients(self) -> None:
        """
        Reset gradient accumulators to zero arrays matching the grid size.

        :return: ``None``.
        :rtype: None
        """
        if self.grid is None:
            return
        for role in self.gradients:
            self.gradients[role] = np.zeros(self.grid.shape[0])

    # === Logger helper functions ===
    def _calc_avg_hunger(self, target_type):
        if target_type == AgentType.LARVAE:
            larvae = [a.hunger for a in self.agents if isinstance(a, Larvae)]
            return np.mean(larvae) if larvae else 0
        elif target_type == WaspRole.FORAGER:
            foragers = [a.hunger for a in self.agents if isinstance(a, Wasp) and a.role == WaspRole.FORAGER]
            return np.mean(foragers) if foragers else 0
        elif target_type == WaspRole.FEEDER:
            feeders = [a.hunger for a in self.agents if isinstance(a, Wasp) and a.role == WaspRole.FEEDER]
            return np.mean(feeders) if feeders else 0
        return 0

    def _calc_total_food(self):
        return sum(getattr(a, "food", 0) for a in self.agents)

    def _estimate_rush(self):
        feeders = [a for a in self.agents if isinstance(a, Wasp) and a.role == WaspRole.FEEDER]
        hungry = sum(1 for a in feeders if getattr(a, "hunger", 0) > 1)
        return hungry / len(feeders) if feeders else 0

    def _parse_target_from_event(self, event: str) -> str:
        if not event:
            return ""
        parts = event.split()
        for p in parts:
            if p.startswith("W") or p.startswith("L"):
                return p
        return ""
