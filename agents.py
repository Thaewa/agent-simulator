# agents.py
# Defines Agent (abstract), Wasp, and Larvae classes based on UML
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from enum import Enum
from utils import gaussian_attraction,estimate_gradient,m_closest_rows,grid_graph_from_array,in_circle
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------
# Enums
# ---------------------------
class AgentType(Enum):
    """
    Enum to classify agent types in the simulation.

    .. attribute:: WASP

       Represents a wasp agent.

    .. attribute:: LARVAE

       Represents a larvae agent.
    """
    WASP = "Wasp"
    LARVAE = "Larvae"

class WaspRole(Enum):
    """
    Enum to classify the role of a wasp agent.

    .. attribute:: FEEDER

       Wasp responsible for feeding larvae.

    .. attribute:: FORAGER

       Wasp responsible for collecting food.
    """
    FEEDER = "Feeder"
    FORAGER = "Forager"

# ---------------------------
# Abstract Base Class
# ---------------------------

class Agent(ABC):
    r"""
    Abstract base class representing a generic agent in the simulation.

    :param agent_id: Unique identifier for the agent.
    :type agent_id: str
    :param x: X-coordinate of the agent in the environment.
    :type x: int
    :param y: Y-coordinate of the agent in the environment.
    :type y: int
    :param agent_type: The type of agent (e.g., Wasp or Larvae).
    :type agent_type: AgentType
    :param hunger: Initial hunger level of the agent. Defaults to ``0``.
    :type hunger: int, optional
    :param food: The amount of food currently stored by the agent. Defaults to ``0``.
    :type food: int, optional
    :param radius: The radius of the agent. Defaults to ``5``.
    :type radius: int, optional

    :ivar id: Unique identifier for the agent.
    :vartype id: str
    :ivar x: X-coordinate of the agent in the environment.
    :vartype x: int
    :ivar y: Y-coordinate of the agent in the environment.
    :vartype y: int
    :ivar type: The type of agent (e.g., Wasp or Larvae).
    :vartype type: AgentType
    :ivar hunger: Hunger level of the agent.
    :vartype hunger: int
    :ivar storedEvents: Log of events performed by the agent.
    :vartype storedEvents: List[str]
    :ivar radius: The radius of the agent.
    :vartype radius: int
    :ivar food: The amount of food currently stored by the agent.
    :vartype food: int
    :ivar hungerRate: The rate at which the agent's hunger level increases.
    :vartype hungerRate: float
    """

    def __init__(
        self,
        agent_id: str,
        x: int,
        y: int,
        agent_type: "AgentType",
        hunger: int = 0,
        food: int = 0,
        radius: int = 5,
    ) -> None:
        self.id: str = agent_id
        self.x: int = x
        self.y: int = y
        self.type: "AgentType" = agent_type
        self.hunger: int = hunger
        self.noHunger: int = 1
        self.storedEvents: List[str] = []
        self.radius: int = radius
        self.food: int = food
        self.hungerRate: float = 0.01

    def getPosition(self) -> List[int]:
        """
        Return the current position of the agent.

        :return: A list ``[x, y]`` representing the position.
        :rtype: List[int]
        """
        return [self.x, self.y]

    @abstractmethod
    def generateEvent(self) -> str:
        """
        Generate a new event string.

        This method must be implemented by subclasses.

        :return: A description of the generated event.
        :rtype: str
        """
        raise NotImplementedError

    def getType(self) -> "AgentType":
        """
        Return the type of the agent.

        :return: Type of the agent (e.g., Wasp or Larvae).
        :rtype: AgentType
        """
        return self.type

    @abstractmethod
    def step(self, t: int) -> None:
        """
        Perform one simulation step for this agent.

        This method may update position, hunger, food, and/or log events.

        :param t: Current simulation time.
        :type t: int
        :return: This method returns ``None``.
        :rtype: None
        """
        raise NotImplementedError
# ---------------------------
# Subclass: Wasp
# ---------------------------

class Wasp(Agent):
    r"""
    Represents a wasp agent in the simulation.

    :param agent_id: Unique identifier for the agent.
    :type agent_id: str
    :param x: Initial X-coordinate of the wasp.
    :type x: int
    :param y: Initial Y-coordinate of the wasp.
    :type y: int
    :param path_finding: Path-finding strategy (e.g. ``"greedy"``, ``"random_walk"``, ``"biased_random_walk"``,
                         strings containing ``"TSP"`` or ``"Hamiltonian"`` for tour planning).
    :type path_finding: str
    :param hunger: Initial hunger level.
    :type hunger: int
    :param food: Initial amount of stored food.
    :type food: int
    :param max_food: Maximum amount of food the wasp can hold.
    :type max_food: int
    :param outer_nest_radius: Radius threshold used for nest-related constraints.
    :type outer_nest_radius: int
    :param hunger_rate: Per-step hunger increase rate.
    :type hunger_rate: float

    **Key attributes**

    :ivar path_finding: Path-finding mode string.
    :vartype path_finding: str
    :ivar smellRadius: Radius of olfactory perception (in grid units).
    :vartype smellRadius: int
    :ivar smellIntensity: Intensity scaling of olfactory perception.
    :vartype smellIntensity: float
    :ivar next_step: Next planned step as a dictionary with keys ``'x'`` and ``'y'``.
    :vartype next_step: dict
    :ivar chanceOfFeeding: Probability threshold for attempting to feed nearby agents.
    :vartype chanceOfFeeding: float
    :ivar forageIncrease: Amount of food gained when a foraging attempt succeeds.
    :vartype forageIncrease: int
    :ivar maxFood: Maximum food capacity.
    :vartype maxFood: int
    :ivar chanceOfForaging: Probability threshold for attempting to forage.
    :vartype chanceOfForaging: float
    :ivar foodTransferToWasp: Units of food transferred when feeding a wasp.
    :vartype foodTransferToWasp: int
    :ivar foodTransferToLarvae: Units of food transferred when feeding a larva.
    :vartype foodTransferToLarvae: int
    :ivar minHungerCue: Minimum observed hunger cue (tracking range).
    :vartype minHungerCue: float | None
    :ivar maxHungerCue: Maximum observed hunger cue (tracking range).
    :vartype maxHungerCue: float | None
    :ivar outerNestRadius: Outer nest radius constraint.
    :vartype outerNestRadius: int
    :ivar hunger_rate: Per-step hunger increase rate (duplicate of constructor arg for convenience).
    :vartype hunger_rate: float
    :ivar smellrepulsion: Repulsion strength/radius parameter for wasp-to-wasp spacing.
    :vartype smellrepulsion: int
    :ivar repulsionRadius: Spread parameter for repulsion gradient.
    :vartype repulsionRadius: int
    :ivar rolePersistence: Internal counter controlling how long a role is maintained.
    :vartype rolePersistence: int
    :ivar hungerCueDecay: Multiplicative decay applied to the internal ``hungerCue`` each step.
    :vartype hungerCueDecay: float
    :ivar prevStep: Previous position (only used for random-walk variants).
    :vartype prevStep: List[int] | None
    :ivar path: Planned path (sequence of grid nodes) for shortest/TSP strategies.
    :vartype path: list | None
    :ivar rolePersistenceUpdate: Increment applied to ``rolePersistence``.
    :vartype rolePersistenceUpdate: int
    :ivar unloadOnlyChance: Probability of unloading only to larvae when foraging and cues are high.
    :vartype unloadOnlyChance: float
    :ivar hungerCuesLowThreshold: Lower threshold factor for determining "low" hunger cues.
    :vartype hungerCuesLowThreshold: float
    :ivar hungerCuesHighThreshold: Upper threshold factor for determining "high" hunger cues.
    :vartype hungerCuesHighThreshold: float
    :ivar maxStepTrials: Maximum retries when sampling a collision-free next step.
    :vartype maxStepTrials: int
    :ivar oneStepList: Allowed per-axis step choices.
    :vartype oneStepList: List[int]
    :ivar smellRadiusBuffer: Additional radius buffer used in shortest-path detection.
    :vartype smellRadiusBuffer: int
    """

    def __init__(
        self,
        agent_id: str,
        x: int,
        y: int,
        path_finding: str = "greedy",
        hunger: int = 0,
        food: int = 0,
        max_food: int = 10,
        outer_nest_radius: int = 10,
        hunger_rate: float = 0.01,
    ) -> None:
        super().__init__(agent_id, x, y, AgentType.WASP, hunger, food, outer_nest_radius)
        self.path_finding: str = path_finding
        self.smellRadius: int = 2
        self.smellIntensity: float = 5.0
        self.next_step = {"x": 0, "y": 0}
        self.chanceOfFeeding: float = 0.0
        self.forageIncrease: int = 10
        self.maxFood: int = max_food
        self.chanceOfForaging: float = 0.3
        self.foodTransferToWasp: int = 1
        self.foodTransferToLarvae: int = 1
        self.minHungerCue = None
        self.maxHungerCue = None
        self.outerNestRadius = outer_nest_radius
        self.hunger_rate = hunger_rate
        self.smellrepulsion = 2
        self.repulsionRadius = 1
        self.rolePersistence = 50
        self.hungerCueDecay = 0.9
        self.prevStep = None if path_finding != "random_walk" else self.getPosition()
        self.path = None
        self.rolePersistenceUpdate = 0
        self.unloadOnlyChance = 0.8
        self.hungerCuesLowThreshold = 0.25
        self.hungerCuesHighThreshold = 0.55
        self.maxStepTrials = 10
        self.oneStepList = [-1, 0, 1]
        self.smellRadiusBuffer = 2

    def updateRolePersistence(self) -> None:
        """
        Update the role persistence by adding ``rolePersistenceUpdate``.

        :return: ``None``.
        :rtype: None
        """
        self.rolePersistence += self.rolePersistenceUpdate

    def updateHungerCueRange(self) -> None:
        """
        Update the observed min/max hunger cue bounds from the current ``hungerCue``.

        Initializes ``minHungerCue``/``maxHungerCue`` on first call.

        :return: ``None``.
        :rtype: None
        """
        updates = self.hungerCue
        if self.minHungerCue is None:
            self.minHungerCue = updates
        if self.maxHungerCue is None:
            self.maxHungerCue = updates
        self.minHungerCue = updates if self.minHungerCue > updates else self.minHungerCue
        self.maxHungerCue = updates if self.maxHungerCue < updates else self.maxHungerCue

    # Extra methods
    def feedWasp(self, wasp: Agent, passed_food: int) -> None:
        """
        Transfer food to another wasp and update hunger cues.

        :param wasp: Target wasp to receive food.
        :type wasp: Agent
        :param passed_food: Units of food to transfer.
        :type passed_food: int
        :return: ``None``.
        :rtype: None
        """
        if wasp.hungerCue > 0.0:
            self.hungerCue = (self.hungerCue + wasp.hungerCue) / 2
            self.food -= passed_food
            wasp.food += passed_food
            wasp.hunger = self.noHunger
            self.storedEvents.append(f"{self.id} fed {wasp.id} (transfer)")

    def feedLarvae(self, larvae: Agent, passed_food: int) -> None:
        """
        Transfer food to a larva and reset its hunger.

        :param larvae: Target larva to receive food.
        :type larvae: Agent
        :param passed_food: Units of food to transfer.
        :type passed_food: int
        :return: ``None``.
        :rtype: None
        """
        self.food -= passed_food
        larvae.food += passed_food
        larvae.hunger = larvae.noHunger
        self.storedEvents.append(f"{self.id} fed {larvae.id}")

    def feed(self, target: Agent) -> None:
        """
        Feed an agent by transferring a unit of food (if available).

        If the target is a larva, ``foodTransferToLarvae`` is used; if a wasp, ``foodTransferToWasp`` is used.
        Certain role/cue conditions can bias transfers (e.g., unloading to larvae for foragers at high cues).

        :param target: The agent to feed.
        :type target: Agent
        :return: ``None``.
        :rtype: None
        """
        passed_food = (
            self.foodTransferToLarvae
            if target.type == AgentType.LARVAE
            else self.foodTransferToWasp
        )
        if self.food > passed_food:
            if self.hungerCuesHigh() and self.role == WaspRole.FORAGER:
                if isinstance(target, Larvae):
                    if np.random.rand() < self.unloadOnlyChance:
                        self.feedLarvae(target, passed_food)
                else:
                    self.feedWasp(target, passed_food)
            else:
                if isinstance(target, Larvae):
                    self.feedLarvae(target, passed_food)
                else:
                    self.feedWasp(target, passed_food)
        else:
            self.storedEvents.append(
                f"{self.id} tried to feed {target.id} but had no food"
            )

    def forage(self, forage: List[np.ndarray]) -> None:
        """
        Move towards foraging points and possibly gain food.

        If within smell radius of any foraging point and a random draw exceeds
        ``chanceOfForaging`` (and capacity allows), increment stored food by
        ``forageIncrease`` and reset hunger.

        :param forage: List of foraging point coordinates.
        :type forage: List[numpy.ndarray]
        :return: ``None``.
        :rtype: None
        """
        forage = np.array(forage)
        idx = self.nearbyEntity(forage)
        if idx.shape[0] > 0:
            for id in idx:
                if random.random() > self.chanceOfForaging and (
                    self.food + self.forageIncrease < self.maxFood
                ):
                    self.food += self.maxFood  # (Behavior preserved as in original code)
                    self.hunger = self.noHunger
                    self.storedEvents.append(
                        f"{self.id} is in location {self.getPosition()} and foraged food at location {forage[id]}"
                    )

    def inOuterNest(self) -> bool:
        """
        Check if the next position remains inside the outer nest radius.

        :return: ``True`` if the next position is inside; otherwise ``False``.
        :rtype: bool
        """
        return ((self.x + self.next_step["x"]) ** 2 + (self.y + self.next_step["y"]) ** 2) < self.outerNestRadius**2

    def hungerCuesLow(self) -> bool:
        """
        Determine if the current hunger cue is below the low threshold.

        :return: ``True`` if low; otherwise ``False``.
        :rtype: bool
        """
        return self.hungerCue < (
            (self.maxHungerCue + self.minHungerCue) * self.hungerCuesLowThreshold
        )

    def foragerMovesCondition(self) -> bool:
        """
        Condition used to decide whether a forager should move.

        :return: ``True`` if movement condition is satisfied; otherwise ``False``.
        :rtype: bool
        """
        return not self.hungerCuesLow()

    def positionInGrid(self, grid: np.ndarray) -> bool:
        """
        Check whether the next position occurs more than once in a grid
        (used to detect duplicates/collisions in a list of positions).

        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: ``True`` if duplicate/occupied; otherwise ``False``.
        :rtype: bool
        """
        mask = np.all(
            grid == np.array([self.x + self.next_step["x"], self.y + self.next_step["y"]]).T,
            axis=1,
        )
        idx = np.flatnonzero(mask)
        return len(idx) > 1

    def move(self, next_step_array: np.ndarray, grid: np.ndarray) -> None:
        """
        Move the agent by one step according to ``next_step`` with collision avoidance.
        Logs a movement event after the update.

        :param next_step_array: Array of reserved/planned next steps for other agents.
        :type next_step_array: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        self._generateNewNextStepIfNecessary(next_step_array, grid)
        self._updatePosition()
        self._generateMoveEvent()

    def _generateNewNextStepIfNecessary(self, next_step_array: np.ndarray, grid: np.ndarray) -> None:
        """
        Helper to ensure ``next_step`` is set to a valid, unoccupied step.

        :param next_step_array: Array of reserved/planned next steps for other agents.
        :type next_step_array: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        if self.next_step["x"] == 0 and self.next_step["y"] == 0:
            self.next_step["x"] = np.random.choice(self.oneStepList, 1)[0]
            self.next_step["y"] = np.random.choice(self.oneStepList, 1)[0]
        if next_step_array.shape[0] != 0 or (not self.positionInGrid(grid)):
            next_step_array_condition = next_step_array == np.array(
                [self.x + self.next_step["x"], self.y + self.next_step["y"]]
            ).T
            count = 0
            while (
                np.any(np.all(next_step_array_condition, axis=1)) and (not self.positionInGrid(grid))
            ) or (self.role == WaspRole.FEEDER and (not self.inOuterNest())):
                self.next_step["x"] = np.random.choice(self.oneStepList, 1)[0]
                self.next_step["y"] = np.random.choice(self.oneStepList, 1)[0]
                next_step_array_condition = next_step_array == np.array(
                    [self.x + self.next_step["x"], self.y + self.next_step["y"]]
                ).T
                count += 1
                if count > self.maxStepTrials:
                    self.next_step["x"] = 0.0
                    self.next_step["y"] = 0.0
                    break

    def _updatePosition(self) -> None:
        """
        Apply ``next_step`` to the wasp's position and reset ``next_step``.

        :return: ``None``.
        :rtype: None
        """
        if self.path_finding == "random_walk":
            self.prevStep = self.getPosition()
        self.x += self.next_step["x"]
        self.y += self.next_step["y"]
        self.next_step = {"x": 0, "y": 0}

    def _generateMoveEvent(self) -> None:
        """
        Log a movement event to ``storedEvents``.

        :return: ``None``.
        :rtype: None
        """
        new_pos = self.getPosition()
        self.storedEvents.append(
            f"{self.id} moved to {new_pos} current hunger level is {format(self.hunger, '.2f')} and food stored is {format(self.food, '.2f')}"
        )

    def hungerCuesHigh(self) -> bool:
        """
        Determine if the current hunger cue is above the high threshold.

        :return: ``True`` if high; otherwise ``False``.
        :rtype: bool
        """
        return self.hungerCue > (
            self.hungerCuesHighThreshold * ((self.maxHungerCue + self.minHungerCue))
        )

    def estimateLocalHungerCue(self, gradientField, grid: np.ndarray) -> float:
        """
        Estimate the local hunger cue within ``smellRadius``.

        :param gradientField: Field of hunger cues sampled on the grid.
        :type gradientField: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: Sum of cues within the local neighborhood.
        :rtype: float
        """
        index = np.where(in_circle((grid[:, 0] - self.x), (grid[:, 1] - self.y), self.smellRadius))[0]
        return sum(gradientField[index])

    def updateHungerCue(self, hungerCue: float) -> None:
        """
        Set the current internal hunger cue.

        :param hungerCue: New hunger cue value.
        :type hungerCue: float
        :return: ``None``.
        :rtype: None
        """
        self.hungerCue = hungerCue

    def updateFeltGradient(self, grid: np.ndarray, x0: float, y0: float, feltGradient: np.ndarray) -> np.ndarray:
        """
        Add an attraction around ``(x0, y0)`` to a felt gradient using a Gaussian-like kernel.

        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :param x0: X-coordinate of the attraction peak.
        :type x0: float
        :param y0: Y-coordinate of the attraction peak.
        :type y0: float
        :param feltGradient: Current felt gradient to be updated (shape compatible with grid sampling).
        :type feltGradient: numpy.ndarray
        :return: Updated felt gradient.
        :rtype: numpy.ndarray
        """
        spread = self.smellRadius
        peak = (self.hunger / max(self.food, 0.1)) / self.smellIntensity
        gradient = gaussian_attraction(grid[:, 0], grid[:, 1], x0, y0, spread, peak)
        feltGradient = feltGradient + gradient
        return feltGradient

    def feelForageGradient(self, gradientField, forage, grid: np.ndarray) -> np.ndarray:
        """
        Build a felt gradient using foraging points as attractors.

        :param gradientField: Dictionary mapping role to gradient arrays.
        :type gradientField: Dict[WaspRole, numpy.ndarray]
        :param forage: Iterable of coordinates for foraging points.
        :type forage: List[tuple[float, float]]
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: Felt gradient array.
        :rtype: numpy.ndarray
        """
        feltGradient = np.zeros_like(gradientField[self.role])
        for x0, y0 in forage:
            feltGradient = self.updateFeltGradient(grid, x0, y0, feltGradient)
        return feltGradient

    def feelForagersGradient(self, gradientField, foragersPositions: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Build a felt gradient using nearby forager wasps as attractors.

        :param gradientField: Dictionary mapping role to gradient arrays.
        :type gradientField: Dict[WaspRole, numpy.ndarray]
        :param foragersPositions: ``(M, 2)`` array of forager positions.
        :type foragersPositions: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: Felt gradient array.
        :rtype: numpy.ndarray
        """
        foragers_index = np.where(
            in_circle(foragersPositions[:, 0], foragersPositions[:, 1], self.outerNestRadius)
        )[0]
        foragersPositions = foragersPositions[foragers_index]
        feltGradient = np.zeros_like(gradientField[self.role])
        for x0, y0 in foragersPositions:
            feltGradient = self.updateFeltGradient(grid, x0, y0, feltGradient)
        return feltGradient

    def addRepulsionGradient(self, feltGradient: np.ndarray, waspPositions: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """
        Add a repulsive component to the felt gradient around nearby wasps.

        :param feltGradient: Current felt gradient.
        :type feltGradient: numpy.ndarray
        :param waspPositions: ``(K, 2)`` array of wasp positions.
        :type waspPositions: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: Updated felt gradient with repulsion applied.
        :rtype: numpy.ndarray
        """
        wasp_index = np.where(
            (((waspPositions[:, 0] - self.x) ** 2 + (waspPositions[:, 1] - self.y) ** 2) <= self.smellrepulsion)
        )[0]
        for index in wasp_index:
            x0, y0 = waspPositions[index]
            gradient = gaussian_attraction(
                grid[:, 0], grid[:, 1], x0, y0, self.smellRadius, self.repulsionRadius
            )
            feltGradient = feltGradient - gradient
        return feltGradient

    def greedy_path_finding(
        self,
        grid: np.ndarray,
        gradientField: Dict[WaspRole, np.ndarray],
        forage: List[Tuple[float, float]] = None,
        foragersPositions: np.ndarray = None,
        waspPositions: np.ndarray = None,
    ) -> None:
        """
        Greedy path-finding that follows the (possibly modified) gradient field.

        Depending on the role and food level, the felt gradient may incorporate
        foraging points (for foragers) or forager positions (for feeders) and
        repulsion from other wasps.

        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :param gradientField: Dictionary mapping ``WaspRole`` to gradient arrays.
        :type gradientField: Dict[WaspRole, numpy.ndarray]
        :param forage: Optional list of foraging point coordinates.
        :type forage: List[tuple[float, float]] | None
        :param foragersPositions: Optional ``(M, 2)`` array of forager positions.
        :type foragersPositions: numpy.ndarray | None
        :param waspPositions: Optional ``(K, 2)`` array of wasp positions.
        :type waspPositions: numpy.ndarray | None
        :return: ``None``.
        :rtype: None
        """
        if self.role == WaspRole.FORAGER:
            if self.food < 1:
                feltGradient = self.feelForageGradient(gradientField, forage, grid)
            else:
                feltGradient = gradientField[self.role]

        if self.role == WaspRole.FEEDER:
            if self.food < 1:
                feltGradient = self.feelForagersGradient(gradientField, foragersPositions, grid)
            else:
                feltGradient = gradientField[self.role]
                feltGradient = self.addRepulsionGradient(feltGradient, waspPositions, grid)

        dZdx, dZdy = estimate_gradient(grid, feltGradient)
        dZ = np.column_stack((dZdx, dZdy))
        mask = np.all(grid == np.array([self.x, self.y]).T, axis=1)
        idx = np.flatnonzero(mask)
        sign_displacement = np.sign(dZ[idx])

        self.next_step[list(self.next_step.keys())[0]] += sign_displacement[0, 0].item()
        self.next_step[list(self.next_step.keys())[1]] += sign_displacement[0, 1].item()

    def random_walk_path_finding(self) -> None:
        """
        Random-walk (optionally biased) path-finding by sampling a one-step move.

        :return: ``None``.
        :rtype: None
        """
        new_x = np.random.choice(self.oneStepList, 1)[0]
        new_y = np.random.choice(self.oneStepList, 1)[0]
        if "biased" in self.path_finding:
            while self.prevStep[0] == (self.x + new_x) and self.prevStep[1] == (self.y + new_y):
                new_x = np.random.choice(self.oneStepList, 1)[0]
                new_y = np.random.choice(self.oneStepList, 1)[0]

    def estimate_path(self, relevant_larvae_position, X, Y):
        """
        Estimate a visiting path over relevant larvae positions using a TSP heuristic.

        :param relevant_larvae_position: Array of larvae positions.
        :type relevant_larvae_position: numpy.ndarray
        :param X: Meshgrid X-array covering the search area.
        :type X: numpy.ndarray
        :param Y: Meshgrid Y-array covering the search area.
        :type Y: numpy.ndarray
        :return: A concatenation of the shortest path from the current position to the tour start and the TSP tour itself.
        :rtype: list
        """
        arr_full = np.ones_like(X, dtype=bool)
        G = grid_graph_from_array(arr_full, X, Y)
        int_position = [int(pos) for pos in self.getPosition()]
        nodes = np.array(relevant_larvae_position)
        nodes = (nodes.astype(int)).tolist()
        nodes_list = [tuple(node) for node in nodes]
        tsp = nx.approximation.traveling_salesman_problem(
            G, nodes=nodes_list, cycle=True if "Hamiltonian" in self.path_finding else False
        )
        int_position = [int(pos) for pos in self.getPosition()]
        added_path = nx.shortest_path(G, tuple(int_position), tsp[0])
        return added_path + tsp

    def get_path(self, larvaePositions: np.ndarray, index: np.ndarray) -> None:
        """
        Compute a local path to a subset of larvae and prime ``next_step`` accordingly.

        :param larvaePositions: ``(L, 2)`` array of larvae positions.
        :type larvaePositions: numpy.ndarray
        :param index: Indices of larvae within an expanded smell neighborhood.
        :type index: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        min_x = larvaePositions[index][:, 0].min()
        max_x = larvaePositions[index][:, 0].max()
        min_y = larvaePositions[index][:, 1].min()
        max_y = larvaePositions[index][:, 1].max()
        x = np.arange(min(min_x - 2, self.x - 2), max(max_x + 2, self.x + 2))
        y = np.arange(min(min_y - 2, self.y - 2), max(max_y + 2, self.y + 2))
        X, Y = np.meshgrid(x, y)
        m = int(self.food / self.foodTransferToLarvae)
        _, relevant_larvae_position = m_closest_rows(larvaePositions[index], np.array(self.getPosition()), min(len(index), m))
        if relevant_larvae_position.shape[0] == 1:
            pass
        else:
            self.path = self.estimate_path(x, y, relevant_larvae_position, X, Y)
            self.next_step["x"] = self.path[0][0] - self.x
            self.next_step["y"] = self.path[0][1] - self.y
            self.path = self.path[1:]

    def shortest_path_finding(self, larvaePositions: np.ndarray) -> None:
        """
        Shortest-path behavior toward nearby larvae clusters, with basic path tracking.

        :param larvaePositions: ``(L, 2)`` array of larvae positions.
        :type larvaePositions: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        if self.path is None:
            index = np.where(
                in_circle((larvaePositions[:, 0] - self.x), (larvaePositions[:, 1] - self.y), (self.smellRadius + self.smellRadiusBuffer))
                & in_circle((larvaePositions[:, 0] - self.x), (larvaePositions[:, 1] - self.y), (0))
            )[0]
            if len(index) == 0:
                pass
            else:
                self.get_path(larvaePositions, index)
        else:
            if abs(self.path[0][0] - self.x) > 1 or abs(self.path[0][1] - self.y) > 1:
                self.path = None
            else:
                self.next_step["x"] = self.path[0][0] - self.x
                self.next_step["y"] = self.path[0][1] - self.y
                if len(self.path) == 1:
                    self.path = None
                else:
                    self.path = self.path[1:]

    def feelGradient(
        self,
        grid: np.ndarray,
        gradientField: Dict[WaspRole, np.ndarray],
        forage: List[Tuple[float, float]] = None,
        foragersPositions: np.ndarray = None,
        waspPositions: np.ndarray = None,
        larvaePositions: np.ndarray = None,
    ) -> None:
        """
        Sense the environment and choose a movement policy based on role and resources.

        If the wasp lacks sufficient food, it may ignore larvae fields and incorporate
        foraging points (for foragers) or forager positions (for feeders). Depending on
        ``path_finding``, it performs greedy, random-walk, or TSP-inspired movement.
        Logs a "sensed gradient" event.

        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :param gradientField: Dictionary mapping ``WaspRole`` to gradient arrays.
        :type gradientField: Dict[WaspRole, numpy.ndarray]
        :param forage: Optional list of foraging point coordinates.
        :type forage: List[tuple[float, float]] | None
        :param foragersPositions: Optional ``(M, 2)`` array of forager positions.
        :type foragersPositions: numpy.ndarray | None
        :param waspPositions: Optional ``(K, 2)`` array of wasp positions (used for repulsion).
        :type waspPositions: numpy.ndarray | None
        :param larvaePositions: Optional ``(L, 2)`` array of larvae positions (used for shortest/TSP strategies).
        :type larvaePositions: numpy.ndarray | None
        :return: ``None``.
        :rtype: None
        """
        hungerCue = sum(gradientField[self.role]) / gradientField[self.role].shape[0]
        self.updateHungerCue(hungerCue)
        self.updateHungerCueRange()

        if self.food < 1.1:
            self.greedy_path_finding(grid, gradientField, forage, foragersPositions, waspPositions)
        elif self.food < self.foodTransferToWasp and self.role == WaspRole.FORAGER:
            self.greedy_path_finding(grid, gradientField, forage, foragersPositions, waspPositions)
        else:
            if self.path_finding == "greedy":
                self.greedy_path_finding(grid, gradientField, forage, foragersPositions, waspPositions)
            elif "random_walk" in self.path_finding:
                self.random_walk_path_finding()
            elif "TSP" in self.path_finding:
                self.greedy_path_finding(grid, gradientField, forage, foragersPositions, waspPositions)
                self.shortest_path_finding(larvaePositions)
            self.storedEvents.append(f"{self.id} sensed gradient ")

    def generateEvent(self) -> str:
        """
        Generate a wasp-specific event string.

        :return: Event description.
        :rtype: str
        """
        return f"Wasp {self.id} event"

    def nearbyEntity(self, entities: np.ndarray) -> np.ndarray:
        """
        Return indices of entities within a small Chebyshev neighborhood.

        :param entities: ``(E, 2)`` array of entity positions.
        :type entities: numpy.ndarray
        :return: Indices of entities within the neighborhood.
        :rtype: numpy.ndarray
        """
        entities = np.abs(entities - np.array([self.x, self.y]))
        masks = (entities <= 2).all(axis=1)
        idx = np.flatnonzero(masks)
        return idx

    def feedNearby(self, agents: List[Agent]) -> None:
        """
        Attempt to feed nearby agents according to ``chanceOfFeeding`` and their hunger.

        :param agents: Candidate agents to consider for feeding.
        :type agents: List[Agent]
        :return: ``None``.
        :rtype: None
        """
        if not agents:
            return

        agents_position = np.array([agent.getPosition() for agent in agents])
        idx = self.nearbyEntity(agents_position)
        if idx.shape[0] > 0:
            for id in idx:
                if random.random() > self.chanceOfFeeding and agents[id].hunger > 1:
                    self.feed(agents[id])

    def step(self, t: int = None, agents: List[Agent] = None, forage=None) -> None:
        """
        Advance the wasp's state by one simulation step.

        Splits agent lists, performs feeding (depending on role and hunger cues),
        optionally forages, and applies hunger-cue decay.

        :param t: Current simulation time (not used directly).
        :type t: int | None
        :param agents: All agents in the simulation.
        :type agents: List[Agent] | None
        :param forage: Optional list of foraging point coordinates.
        :type forage: List[tuple[float, float]] | None
        :return: ``None``.
        :rtype: None
        """
        wasps = [agent for agent in agents if agent.type == AgentType.WASP]
        wasps_feeders = [wasp for wasp in wasps if wasp.role == WaspRole.FEEDER]
        larvaes = [agent for agent in agents if agent.type == AgentType.LARVAE]
        net_receivers = wasps_feeders + larvaes

        if self.food > 0:
            if self.role == WaspRole.FORAGER:
                if self.hungerCuesHigh():
                    self.feedNearby(wasps_feeders)
                else:
                    self.feedNearby(net_receivers)
            elif self.role == WaspRole.FEEDER:
                self.feedNearby(larvaes)

        if self.role == WaspRole.FORAGER and forage is not None:
            self.forage(forage)

        self.hungerCue = self.hungerCue * self.hungerCueDecay

# ---------------------------
# Subclass: Larvae
# ---------------------------


class Larvae(Agent):
    r"""
    Represents a larvae agent.

    Larvae do not move or forage; they only request food and increase hunger
    according to their internal rate.

    :param agent_id: Unique identifier for the larvae.
    :type agent_id: str
    :param x: Initial X-coordinate.
    :type x: int
    :param y: Initial Y-coordinate.
    :type y: int
    :param hunger: Initial hunger level. Defaults to ``0``.
    :type hunger: int, optional
    :param food: Initial amount of stored food. Defaults to ``0``.
    :type food: int, optional
    :param hungerMultiplier: Multiplier applied to the base ``hungerRate`` inherited
                             from :class:`Agent`. The effective rate becomes
                             ``hungerMultiplier * hungerRate``.
    :type hungerMultiplier: float, optional
    :param hunger_rate: Per-step hunger increase rate stored on this instance
                        (kept for parity with :class:`Wasp`). Defaults to ``0.01``.
    :type hunger_rate: float, optional

    **Key attributes**

    :ivar hunger_rate: Per-step hunger increase rate stored on this instance.
    :vartype hunger_rate: float
    :ivar hungerMultiplier: Scalar applied to the inherited ``hungerRate``.
    :vartype hungerMultiplier: float
    :ivar hungerRate: Effective per-step hunger rate after applying ``hungerMultiplier``.
    :vartype hungerRate: float
    """

    def __init__(
        self,
        agent_id: str,
        x: int,
        y: int,
        hunger: int = 0,
        food: int = 0,
        hungerMultiplier: float = 1.0,
        hunger_rate: float = 0.01,
    ) -> None:
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger, food)
        self.hunger_rate = hunger_rate
        self.hungerMultiplier = hungerMultiplier
        # scale the inherited Agent.hungerRate
        self.hungerRate = self.hungerMultiplier * self.hungerRate

    def generateEvent(self) -> str:
        """
        Generate a larvae-specific event string.

        :return: Event description.
        :rtype: str
        """
        return f"Larvae {self.id} event"

    def step(self, t: int | None = None, agents: List["Agent"] | None = None, forage=None) -> None:
        """
        Advance the larvae's state by one time step.

        Larvae cannot move or forage; they only increase hunger and request food.

        :param t: Current simulation time (used for logging).
        :type t: int | None
        :param agents: Unused for larvae (present for API compatibility).
        :type agents: List[Agent] | None
        :param forage: Unused for larvae (present for API compatibility).
        :type forage: Any
        :return: ``None``.
        :rtype: None
        """
        self.hunger += self.hungerRate
        self.storedEvents.append(f"{self.id} asked for food at time {t}")
