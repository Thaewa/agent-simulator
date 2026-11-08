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
from pathfinding import normalize_pathfinding_mode


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

       Wasp responsible for feeding larvae. A feeder is a wasp that stays within the nest, it can receive and transfer
         food to other wasps and to larvae. 

    .. attribute:: FORAGER

       Wasp responsible for collecting food. The forager is a wasp that can move outside the nest, collect food,
       and transfer it to either wasps or larvae. 
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

    def _updatePosition(self):
        """Base fallback so subclasses can safely call super()._updatePosition()."""
        if hasattr(self, "next_step"):
            self.x += self.next_step.get("x", 0)
            self.y += self.next_step.get("y", 0)

    
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
    :param path_finding: Path-finding strategy (e.g. ``"greedy"``, ``"random_walk"``, ``"random_walk-biased"``,
                         ``"tsp"`` or ``"tsp-hamiltonian"`` for tour planning; names are case-insensitive).
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
    :param smell_radius: Radius of olfactory perception (in grid units).
    :type smell_radius: int
    :param smell_intensity: Intensity scaling of olfactory perception.
    :type smell_intensity: float
    :param chance_of_feeding: Probability of feeding larvae when encountering one.
    :type chance_of_feeding: float
    :param forage_increase: Amount of food added to the foraging site when a wasp forages.
    :type forage_increase: int
    :param chance_of_foraging: Probability of foraging at a foraging site.
    :type chance_of_foraging: float
    :param food_transfer_to_wasp: Amount of food transferred from a wasp to another when the latter is hungry and encounters the former.
    :type food_transfer_to_wasp: int
    :param food_transfer_to_larvae: Amount of food transferred from a wasp to a larva when the wasp is not hungry and encounters the larva.
    :type food_transfer_to_larvae: int
    :param role_persistence: Number of simulation steps before reassessing the role of a wasp.
    :type role_persistence: int
    :param hunger_cue_decay: Rate of decay of hunger cues from a foraging site.
    :type hunger_cue_decay: float
    :param unload_only_chance: Probability of a wasp unloading food at a foraging site without foraging.
    :type unload_only_chance: float
    :param hunger_cues_low_threshold: Lower bound on the hunger cue threshold.
    :type hunger_cues_low_threshold: float
    :param hunger_cues_high_threshold: Upper bound on the hunger cue threshold.
    :type hunger_cues_high_threshold: float
    :param max_step_trials: Maximum number of times a wasp will attempt to take a step before giving up.
    :type max_step_trials: int
    :param smell_radius_buffer: Additional radius added to the smell radius to prevent wasps from getting stuck at the edge of the foraging site.
    :type smell_radius_buffer: int
    :param feeder_to_forager_probability: Probability of a wasp becoming a forager when being a feeder and hunger cues are high.
    :type feeder_to_forager_probability: float
    :param forager_to_feeder_probability: Probability of a wasp becoming a feeder when being a forager and hunger cues are low.
    :type forager_to_feeder_probability: float

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
    :var maxHungerCue: Maximum observed hunger cue (tracking range).
    :vartype maxHungerCue: float | None
    :ivar outerNestRadius: Outer nest radius constraint.
    :vartype outerNestRadius: int
    :ivar hunger_rate: Per-step hunger increase rate (duplicate of constructor arg for convenience).
    :vartype hunger_rate: float
    :ivar rolePersistence: Internal counter controlling how long a role is maintained.
    :vartype rolePersistence: int
    :ivar hungerCueDecay: Multiplicative decay applied to the internal ``hungerCue`` each step.
    :vartype hungerCueDecay: float
    :ivar prevStep: Previous position (only used for random-walk variants).
    :vartype prevStep: List[int] | None
    :ivar path: Planned path (sequence of grid nodes) for shortest/TSP strategies.
    :vartype path: list | None
    :ivar rolePersistenceUpdate: Configured cooldown restored after a role change.
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
    :ivar feederToForagerProbability: Probability of a wasp becoming a forager when being a feeder and hunger cues are high.
    :vartype feederToForagerProbability: float
    :ivar foragerToFeederProbability: Probability of a wasp becoming a feeder when being a forager and hunger cues are low.
    :vartype foragerToFeederProbability: float
    """

    def __init__(
        self,
        agent_id: str,
        outer_nest_radius: int,
        x: int = 0,
        y: int = 0,
        path_finding: str = "greedy",
        hunger: int = 0,
        food: int = 0,
        max_food: int = 10,
        hunger_rate: float = 0.01,
        smell_radius: int = 2,
        smell_intensity: float = 5.0,
        chance_of_feeding: float = 0.8,
        forage_increase: int = 10,
        chance_of_foraging: float = 0.8,
        food_transfer_to_wasp: int = 1,
        food_transfer_to_larvae: int = 1,
        role_persistence: int = 50,
        hunger_cue_decay: float = 0.9,
        unload_only_chance: float = 0.8,
        hunger_cues_low_threshold: float = 0.25,
        hunger_cues_high_threshold: float = 0.55,
        max_step_trials: int = 10,
        smell_radius_buffer: int = 2,
        feeder_to_forager_probability: float = 0.8,
        forager_to_feeder_probability: float = 0.2,
    ) -> None:
        if not np.isfinite(hunger_rate) or hunger_rate < 0:
            raise ValueError("Wasp hunger_rate must be finite and nonnegative")
        super().__init__(agent_id, x, y, AgentType.WASP, hunger, food, outer_nest_radius)
        self.path_finding: str = path_finding
        self.smellRadius: int = smell_radius
        self.smellIntensity: float = smell_intensity
        self.next_step = {"x": 0, "y": 0}
        self.chanceOfFeeding: float = chance_of_feeding
        self.forageIncrease: int = forage_increase
        self.maxFood: int = max_food
        self.chanceOfForaging: float = chance_of_foraging
        self.foodTransferToWasp: int = food_transfer_to_wasp
        self.foodTransferToLarvae: int = food_transfer_to_larvae
        self.minHungerCue = None
        self.maxHungerCue = None
        self.outerNestRadius = outer_nest_radius
        self.hungerRate = hunger_rate
        self.rolePersistence = role_persistence
        self.hungerCueDecay = hunger_cue_decay
        self.prevStep = self.getPosition()
        self.path = None
        self.rolePersistenceUpdate = role_persistence
        self.unloadOnlyChance = unload_only_chance
        self.hungerCuesLowThreshold = hunger_cues_low_threshold
        self.hungerCuesHighThreshold = hunger_cues_high_threshold
        self.maxStepTrials = max_step_trials
        self.oneStepList = [-0.5, 0, 0.5]
        self.smellRadiusBuffer = smell_radius_buffer
        self.feeder_to_forager_probability = feeder_to_forager_probability
        self.forager_to_feeder_probability = forager_to_feeder_probability
        self.hungerCue = 0
        self.distance_traveled = 0.0 # <---- Newly added
        self.feed_count_larvae = 0 # <---- Newly added
        self.feed_count_wasp = 0 # <---- Newly added
        self.forage_count = 0
        self.hunger_cue = 0.0 # <---- # stores latest hunger cue sensed
        self.step_count = 0  # <---- New field
        self.feeder_step_count = 0
        self.feeder_distance_traveled = 0.0
        self.feeder_feed_count_larvae = 0
        
    @property
    def path_finding(self) -> str:
        return self._path_finding

    @path_finding.setter
    def path_finding(self, mode: str) -> None:
        # The generator can override this after construction as well.
        self._path_finding = normalize_pathfinding_mode(mode)

    def updateRolePersistence(self) -> None:
        """
        Reset the cooldown to the configured ``rolePersistenceUpdate`` interval.
        
        :return: ``None``.
        :rtype: None
        """
        self.rolePersistence = self.rolePersistenceUpdate

    def updateHungerCueRange(self) -> None:
        """
        Update the observed min/max hunger cue bounds from the current ``hungerCue``.

        Initializes ``minHungerCue``/``maxHungerCue`` on first call.

        This method assumes that the agents are able to react to thresholds of local hunger cues.
        As the perceived hunger cue  can change over simulations. These thresholds are build dynamically over the simulation.
        The idea behind this is that the agents are able to gage what is a HIGH and LOW hunger cue for each particular instance of the simulation.

        :return: ``None``.
        :rtype: None
        """
        updates = self.hungerCue
        if self.minHungerCue is None:
            self.minHungerCue = updates
        if self.maxHungerCue is None:
            self.maxHungerCue = updates
        # Update, if current local hunger cue is lower/higher than existing min/high threshold.
        self.minHungerCue = updates if self.minHungerCue > updates else self.minHungerCue
        self.maxHungerCue = updates if self.maxHungerCue < updates else self.maxHungerCue

    # Extra methods
    def feedWasp(self, wasp: Agent, passed_food: int) -> float:
        """
        Transfer up to the requested food, limited by recipient capacity.

        :param wasp: Target wasp to receive food.
        :type wasp: Agent
        :param passed_food: Units of food to transfer.
        :type passed_food: int
        :return: Amount transferred, or zero if no transfer occurred.
        :rtype: float
        """
        if wasp is self or passed_food <= 0 or self.food < passed_food:
            return 0.0
        amount = min(passed_food, max(0, wasp.maxFood - wasp.food))
        if amount <= 0:
            return 0.0
        # When wasps interact they also exchange information of their local hunger cues 
        if wasp.hungerCue > 0.0:
            self.hungerCue = (self.hungerCue + wasp.hungerCue) / 2
        # Update local food storage 
        self.food -= amount
        # Update target wasp
        wasp.food += amount
        # Reset target wasp hunger
        wasp.hunger = self.noHunger
        self.feed_count_wasp += 1 
        self.storedEvents.append(f"{self.id} fed {wasp.id} (transfer)")
        return float(amount)

    def feedLarvae(self, larvae: Agent, passed_food: int) -> float:
        """
        Transfer food to a larva and reset its hunger.

        :param larvae: Target larva to receive food.
        :type larvae: Agent
        :param passed_food: Units of food to transfer.
        :type passed_food: int
        :return: Amount transferred, or zero if no transfer occurred.
        :rtype: float
        """
        if passed_food <= 0 or self.food < passed_food:
            return 0.0
        # Update local food storage
        self.food -= passed_food
        # Update target larva
        larvae.food += passed_food
        # Reset target larva hunger
        larvae.hunger = larvae.noHunger
        self.feed_count_larvae += 1
        if self.role == WaspRole.FEEDER:
            self.feeder_feed_count_larvae += 1
        self.storedEvents.append(f"{self.id} fed {larvae.id}")
        return float(passed_food)

    def feed(self, target: Agent) -> float:
        """
        Feed an agent by transferring a unit of food (if available).

        If the target is a larva, ``foodTransferToLarvae`` is used; if a wasp, ``foodTransferToWasp`` is used.
        Certain role/cue conditions can bias transfers (e.g., unloading to larvae for foragers at high cues).
        This implementation assumes that there can be different amounts of food transfered to wasps and larvae. This can be changed in the configuration file.

        :param target: The agent to feed.
        :type target: Agent
        :return: Amount transferred, or zero if feeding did not succeed.
        :rtype: float
        """
        # Define the amount of food to transfer given the target type
        passed_food = self.foodTransferToLarvae if target.type == AgentType.LARVAE else self.foodTransferToWasp
        # A donor can give its last complete ration.
        if passed_food > 0 and self.food >= passed_food:
            # If the wasp is a forager and the hunger cue is high it is more likely to do an unload than to directly feed larvae
            if self.hungerCuesHigh() and self.role == WaspRole.FORAGER:
                if isinstance(target, Larvae):
                    # If the target is a larva and the hunger cues are high it is unlikely that the larvae will be feeded
                    if np.random.rand() < self.unloadOnlyChance:
                        return self.feedLarvae(target, passed_food)
                else:
                    # If the target is a wasp and the hunger cues are high it is likely that the wasp will be feeded
                    return self.feedWasp(target, passed_food)
            else:
                if isinstance(target, Larvae):
                    # Feeders and foragers feed larvae if hunger cues are not high
                    return self.feedLarvae(target, passed_food)
                else:
                    # The transfer method enforces the recipient's capacity.
                    return self.feedWasp(target, passed_food)
        else:
            self.storedEvents.append(
                f"{self.id} tried to feed {target.id} but had no food"
            )
        return 0.0
                    
    def forage(self, forage: List[np.ndarray]) -> None:
        """
        Move towards foraging points and possibly gain food.

        If near a foraging point and a random draw is below
        ``chanceOfForaging`` (and capacity allows), increment stored food by
        ``forageIncrease`` up to ``maxFood`` and reset hunger.

        :param forage: List of foraging point coordinates.
        :type forage: List[numpy.ndarray]
        :return: ``None``.
        :rtype: None
        """
        if len(forage) == 0:
            return
        forage = np.array(forage)
        # Collect the positions of foraging nearby the wasp
        idx = self.nearbyEntity(forage)
        if idx.shape[0] > 0:
            for id in idx:
                # If the forager wasp is not full and a random draw is less than the chance of foraging realize the forage
                if self.food < self.maxFood and self.forageIncrease > 0 and np.random.rand() < self.chanceOfForaging:
                    self.food += min(self.forageIncrease, self.maxFood - self.food)
                    self.forage_count += 1
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
        return ((self.x + self.next_step["x"]) ** 2 + (self.y + self.next_step["y"]) ** 2) <= self.outerNestRadius**2

    def hungerCuesLow(self) -> bool:
        """
        Determine if the current hunger cue is below the low threshold.
        The hungerCuesLowThreshold can be updated in the config file.
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
        """Whether the proposed next position is an existing grid cell."""
        point = np.array([self.x + self.next_step["x"], self.y + self.next_step["y"]])
        return bool(np.any(np.all(grid == point, axis=1)))

    def move(self, next_step_array: np.ndarray, grid: np.ndarray) -> None:
        """
        Move the agent by one step according to ``next_step`` with collision avoidance.
        Logs a movement event after the update.

        :param next_step_array: Current occupied positions of other adult wasps.
        :type next_step_array: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        # Checks if the next_step attribute is not empty
        self._generateNewNextStepIfNecessary(next_step_array, grid)
        # If the next step is not empty update the position
        self._updatePosition()
        # Generate a movement event
        self._generateMoveEvent()
        self.step_count += 1 # <--- newly added
        if self.role == WaspRole.FEEDER:
            self.feeder_step_count += 1

    def _generateNewNextStepIfNecessary(self, next_step_array: np.ndarray, grid: np.ndarray) -> None:
        """Validate a step against grid, nest and occupied adult cells.

        Movement is sequential. A valid zero action stays still; after bounded
        retries, an invalid action also stays at its valid, unoccupied origin.
        """
        occupied = np.asarray(next_step_array).reshape(-1, 2)
        current = np.array(self.getPosition())
        if not np.isfinite(current).all() or not np.any(np.all(grid == current, axis=1)):
            raise ValueError(f"{self.id} starts outside the grid")
        if np.any(np.all(occupied == current, axis=1)):
            raise ValueError(f"{self.id} starts on another adult's cell")
        if self.role == WaspRole.FEEDER and np.dot(current, current) > self.outerNestRadius**2:
            raise ValueError(f"{self.id} starts outside its feeder nest boundary")
        if not np.isfinite([self.next_step["x"], self.next_step["y"]]).all():
            raise ValueError(f"{self.id} has a non-finite movement action")

        def valid():
            point = current + [self.next_step["x"], self.next_step["y"]]
            return (
                self.next_step["x"] in self.oneStepList
                and self.next_step["y"] in self.oneStepList
                and self.positionInGrid(grid)
                and not np.any(np.all(occupied == point, axis=1))
                and (self.role != WaspRole.FEEDER or self.inOuterNest())
            )

        if valid():
            return
        for _ in range(self.maxStepTrials):
            self.next_step = {
                "x": np.random.choice(self.oneStepList, 1)[0],
                "y": np.random.choice(self.oneStepList, 1)[0],
            }
            if valid():
                return
        self.next_step = {"x": 0, "y": 0}

    def _updatePosition(self) -> None:
        """
        Apply ``next_step`` to the wasp's position and reset ``next_step``.

        :return: ``None``.
        :rtype: None
        """
        if not hasattr(self, "x") or not hasattr(self, "y"):
            return
        
        prev_x, prev_y = self.x, self.y
        # If the path finding is random walk update the previous step position
        if "random_walk" in self.path_finding  :
            self.prevStep = self.getPosition()
        self.x += self.next_step["x"]
        self.y += self.next_step["y"]
        self.next_step = {"x": 0, "y": 0}
        # <--- Newly added: distance traveled ---
        if prev_x is None or prev_y is None: 
            return
        distance = np.sqrt((self.x - prev_x) ** 2 + (self.y - prev_y) ** 2)
        self.distance_traveled += distance
        if self.role == WaspRole.FEEDER:
            self.feeder_distance_traveled += distance

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
        The hungerCuesHighThreshold can be updated in the config file.

        :return: ``True`` if high; otherwise ``False``.
        :rtype: bool
        """
        return self.hungerCue > (
            self.hungerCuesHighThreshold * ((self.maxHungerCue + self.minHungerCue))
        )

    def estimateLocalHungerCue(self, gradientField, grid: np.ndarray) -> float:
        """
        Estimate the local hunger cue within ``smellRadius``.
        This implementation collects the GLOBAL hunger cues and filters out by the smelling capacities of the agent. 
        The GLOBAL hunger cues are collected in the simulator and passed to the agents. In this way only the local hunger cues are further used
        by the agent.
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
        self.hunger_cue = hungerCue # sync for logger

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
        Build a felt gradient using nearby forager wasps as attractors. This will be used by the feeder wasps to reproduce the SURGE mechanism.
        This will allow the feeder wasps to move towards the foragers when their hunger cues are high as they will look for food.

        :param gradientField: Dictionary mapping role to gradient arrays.
        :type gradientField: Dict[WaspRole, numpy.ndarray]
        :param foragersPositions: ``(M, 2)`` array of forager positions.
        :type foragersPositions: numpy.ndarray
        :param grid: Grid positions as an ``(N, 2)`` array.
        :type grid: numpy.ndarray
        :return: Felt gradient array.
        :rtype: numpy.ndarray
        """
        # Filter forager positions that are inside the nest
        foragers_index = np.where(
            in_circle(foragersPositions[:, 0], foragersPositions[:, 1], self.outerNestRadius)
        )[0]
        foragersPositions = foragersPositions[foragers_index]
        feltGradient = np.zeros_like(gradientField[self.role])
        # Add foragers as attractors by the Gaussian Field
        for x0, y0 in foragersPositions:
            feltGradient = self.updateFeltGradient(grid, x0, y0, feltGradient)
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
        foraging points (for foragers) or forager positions (for feeders) 

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
        # If the agent is a forager
        if self.role == WaspRole.FORAGER:
            # Only use the forage positions as attractors if the forager does not have food
            if self.food < 1:
                feltGradient = self.feelForageGradient(gradientField, forage, grid)
            # Only use local larvae hunger information as attractors
            else:
                feltGradient = gradientField[self.role]
        # If the agent is a feeder wasp
        if self.role == WaspRole.FEEDER:
            # If the feeder does not have food then use the forager positions as attractors to promote SURGE behavior
            if self.food < 1:
                feltGradient = self.feelForagersGradient(gradientField, foragersPositions, grid)
            # If the feeder has food, let it be attracted to local hungry larvae and repel other wasps
            else:
                feltGradient = gradientField[self.role]
                
        # Use discrte differences to estimate the gradient
        dZdx, dZdy = estimate_gradient(grid, feltGradient)
        dZ = np.column_stack((dZdx, dZdy))
        mask = np.all(grid == np.array([self.x, self.y]).T, axis=1)
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            # fallback to nearest grid cell (safeguard)
            idx = [np.argmin(np.sum((grid - np.array([self.x, self.y]))**2, axis=1))]
        # take the sign of the gradient to achieve +1 or -1 movement in each axis.
        sign_displacement = np.sign(dZ[idx])/2
        
        # update the next step
        self.next_step[list(self.next_step.keys())[0]] += sign_displacement[0, 0].item()
        self.next_step[list(self.next_step.keys())[1]] += sign_displacement[0, 1].item()

    def random_walk_path_finding(self) -> None:
        """
        Random-walk (optionally biased) path-finding by sampling a one-step move.

        :return: ``None``.
        :rtype: None
        """
        # Randomly choose a step direction within 1,0,-1
        new_x = np.random.choice(self.oneStepList, 1)[0]
        new_y = np.random.choice(self.oneStepList, 1)[0]
        # If path finding mechanism is biased, prevent agent from returning to previous position
        if "biased" in self.path_finding:
            while self.prevStep[0] == (self.x + new_x) and self.prevStep[1] == (self.y + new_y):
                new_x = np.random.choice(self.oneStepList, 1)[0]
                new_y = np.random.choice(self.oneStepList, 1)[0]
        # Update the next step
        self.next_step[list(self.next_step.keys())[0]] = new_x
        self.next_step[list(self.next_step.keys())[1]] = new_y

    def estimate_path(self, x, y, larvae_positions, X=None, Y=None):
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
        if self.role == WaspRole.FEEDER:
            arr_full = in_circle(X, Y, self.outerNestRadius)
        # Create a grid array from the X and Y arrays which represent the X and Y coordinates of each point in the grid near the wasp
        G = grid_graph_from_array(arr_full, X, Y)
        position = tuple(self.getPosition())
        # Preserve half-cell coordinates; integer conversion changes the targets
        # and can produce nodes that do not exist in the local graph.
        nodes_list = list(dict.fromkeys(tuple(node) for node in larvae_positions))
        if not nodes_list:
            return []
        if len(nodes_list) == 1:
            return nx.shortest_path(G, position, nodes_list[0])
        tsp = nx.approximation.traveling_salesman_problem(
            G, nodes=nodes_list, cycle=self.path_finding == "tsp-hamiltonian"
        )
        added_path = nx.shortest_path(G, position, tsp[0])
        # The connector already includes the first tour node.
        return added_path + tsp[1:]

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
        if self.foodTransferToLarvae <= 0:
            raise ValueError("foodTransferToLarvae must be positive for route planning")
        if self.role == WaspRole.FEEDER and len(index):
            candidates = larvaePositions[index]
            index = np.asarray(index)[in_circle(candidates[:, 0], candidates[:, 1], self.outerNestRadius)]
        capacity = int(np.floor(self.food / self.foodTransferToLarvae))
        target_count = min(len(index), capacity)
        if target_count <= 0:
            self.path = None
            return
        _, targets = m_closest_rows(
            larvaePositions[index], np.array(self.getPosition()), target_count
        )
        # Build the smallest rectangular lattice containing the selected targets
        # and the wasp. Match the simulator's half-cell movement resolution.
        step = min(abs(value) for value in self.oneStepList if value != 0)
        points = np.vstack((targets, self.getPosition()))
        lower = np.floor(points.min(axis=0) / step).astype(int)
        upper = np.ceil(points.max(axis=0) / step).astype(int)
        x = np.arange(lower[0], upper[0] + 1) * step
        y = np.arange(lower[1], upper[1] + 1) * step
        X, Y = np.meshgrid(x, y)
        route = self.estimate_path(x, y, targets, X, Y)
        # move() randomizes a zero step. Skip the initial current-position node
        # so that it cannot accidentally replace the first planned movement.
        while route and route[0] == tuple(self.getPosition()):
            route = route[1:]
        if not route:
            self.path = None
            return
        self.next_step["x"] = route[0][0] - self.x
        self.next_step["y"] = route[0][1] - self.y
        self.path = route[1:] or None

    def shortest_path_finding(self, larvaePositions: np.ndarray) -> None:
        """
        Shortest-path behavior toward nearby larvae clusters, with basic path tracking.

        :param larvaePositions: ``(L, 2)`` array of larvae positions.
        :type larvaePositions: numpy.ndarray
        :return: ``None``.
        :rtype: None
        """
        # Find a TSP path towards the larvae if there isn't a current path already in memory.
        if not self.path:
            # Find the larvae within the expanded smell radius of the agent so that only the closest larvae are considered in the path
            index = np.where(
                in_circle((larvaePositions[:, 0] - self.x), (larvaePositions[:, 1] - self.y), (self.smellRadius + self.smellRadiusBuffer)))[0]
            # If there are not enough larvae within the expanded smell radius, do not find a path
            if len(index) == 0:
                pass
            # If there is at least one near larvae, find a path to them 
            else:
                self.get_path(larvaePositions, index)
        # If there already exists a path in memory, update the next step in the path
        else:
            # If the agent is already in a position close to the next step in the path, remove the path
            # This is a safeguard because the non-blocking behavior alters the next step taken by the agent and in turn this affects the validity of the path
            step = min(abs(value) for value in self.oneStepList if value != 0)
            if abs(self.path[0][0] - self.x) > step or abs(self.path[0][1] - self.y) > step:
                self.path = None
            # If there is a path and it is still a valid path because the non-blocking method has not altered it, use the next step in the path
            else:
                self.next_step["x"] = self.path[0][0] - self.x
                self.next_step["y"] = self.path[0][1] - self.y
                # If there are no more steps in the path, remove the path
                if len(self.path) == 1:
                    self.path = None
                # Remove the current step from the path as it will be taken by the agent
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
        :param waspPositions: Optional ``(K, 2)`` array of wasp positions.
        :type waspPositions: numpy.ndarray | None
        :param larvaePositions: Optional ``(L, 2)`` array of larvae positions (used for shortest/TSP strategies).
        :type larvaePositions: numpy.ndarray | None
        :return: ``None``.
        :rtype: None
        """
        # Update the internal hunger cue ranges given last sensed hunger cue
        self.updateHungerCueRange()
        # All agents use greedy path-finding by default
        self.greedy_path_finding(grid, gradientField, forage, foragersPositions, waspPositions)
        
        #Only feeders use other path-finding strategies
        if self.role == WaspRole.FEEDER and self.food>1:
            # If strategy contains random walk, use such strategy
            if "random_walk" in self.path_finding:
                self.random_walk_path_finding()
            # Select only the supported, normalized TSP modes.
            if self.path_finding in ("tsp", "tsp-hamiltonian"):
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
        Return indices of entities within a small neighborhood of the agent.
        The distance between entities and the agent is hardcoded as 2 since we are only admitting [-1,0,1] steps per coordinate.
        :param entities: ``(E, 2)`` array of entity positions.
        :type entities: numpy.ndarray
        :return: Indices of entities within the neighborhood.
        :rtype: numpy.ndarray
        """
        entities = np.abs(entities - np.array([self.x, self.y]))
        masks = (entities <= 0.5).all(axis=1)
        idx = np.flatnonzero(masks)
        return idx

    def feedNearby(self, agents: List[Agent],t) -> List["FeedingBout"]:
        """
        Attempt to feed nearby agents according to ``chanceOfFeeding`` and their hunger.

        :param agents: Candidate agents to consider for feeding.
        :type agents: List[Agent]
        :return: Successful forager-to-feeder transfer bouts.
        :rtype: List[FeedingBout]
        """
        # Self is not a receiver and must not consume a feeding trial.
        agents = [agent for agent in agents if agent is not self]
        if not agents:
            return []
        # If there are agents that can be feed, obtain their positions
        agents_position = np.array([agent.getPosition() for agent in agents])
        # Filter agents by distance and keep those that are nearby
        idx = self.nearbyEntity(agents_position)
        feeding_bouts = []
        if idx.shape[0] > 0:
            for id in idx:
                # If the wasp is not full and a random draw is less than the chance of foraging realize the feed
                if random.random() < self.chanceOfFeeding and agents[id].hunger > 1.2 and self.food > 0:
                    amount = self.feed(agents[id])
                    if (amount > 0 and self.role == WaspRole.FORAGER
                            and isinstance(agents[id], Wasp)
                            and agents[id].role == WaspRole.FEEDER):
                        bout = FeedingBout(self, agents[id], t, amount)
                        feeding_bouts.append(bout)
        return feeding_bouts
    def step(self, t: int = None, agents: List[Agent] = None, forage=None) -> List["FeedingBout"]:
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
        :return: Successful forager-to-feeder transfer bouts.
        :rtype: List[FeedingBout]
        """
        # Collect all wasps, all feeders and all larvae
        agents = agents or []
        wasps = [agent for agent in agents if agent.type == AgentType.WASP]
        wasps_feeders = [wasp for wasp in wasps if wasp.role == WaspRole.FEEDER]
        larvaes = [agent for agent in agents if agent.type == AgentType.LARVAE]
        net_receivers = wasps_feeders + larvaes
        feed_events = []
        
        # If agent has food
        if self.food > 0:
            if self.role == WaspRole.FORAGER:
                # If agent is a forager and hunger cues are high only feed feeders
                if self.hungerCuesHigh():
                    feed_events = self.feedNearby(wasps_feeders,t)
                # If agent is a forager and hunger cues are not high feed feeders and larvae
                else:
                    feed_events = self.feedNearby(net_receivers,t)
            # If agent is a feeder feeders and larvae
            elif self.role == WaspRole.FEEDER:
                feed_events = self.feedNearby(net_receivers,t)
        else:
            feed_events = []
        # If agent has no food and is a forager, forage
        if self.role == WaspRole.FORAGER and forage is not None and self.food < self.maxFood:
            self.forage(forage)
        # Update local hunger cue with a decay, the decay can be updated in the configuration file
        self.hungerCue = self.hungerCue * self.hungerCueDecay
        return feed_events
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
    :param hungerMultiplier: Multiplier applied to configured ``hunger_rate``.
                             The effective rate is ``hungerMultiplier * hunger_rate``.
    :type hungerMultiplier: float, optional
    :param hunger_rate: Base per-step metabolic/hunger rate. Defaults to ``0.01``.
    :type hunger_rate: float, optional

    **Key attributes**

    :ivar hunger_rate: Per-step hunger increase rate stored on this instance.
    :vartype hunger_rate: float
    :ivar hungerMultiplier: Scalar applied to the configured base rate.
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
        if (not np.isfinite([hunger_rate, hungerMultiplier]).all()
                or hunger_rate < 0 or hungerMultiplier < 0):
            raise ValueError("Larval hunger rate and multiplier must be finite and nonnegative")
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger, food)
        self.hunger_rate = hunger_rate
        self.hungerMultiplier = hungerMultiplier
        self.hungerRate = self.hungerMultiplier * self.hunger_rate
        if not np.isfinite(self.hungerRate):
            raise ValueError("Effective larval hunger rate must be finite")

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
        self.storedEvents.append(f"{self.id} asked for food at time {t}")

class FeedingBout():
    def __init__(self,feeder,receiver,t, amount=0.0):
        self.feeder = feeder
        self.receiver = receiver
        self.timestamp = t
        self.amount = amount
