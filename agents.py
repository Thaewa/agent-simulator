# agents.py
# Defines Agent (abstract), Wasp, and Larvae classes based on UML

from abc import ABC, abstractmethod
from typing import List
from enum import Enum


# ---------------------------
# Enums
# ---------------------------
class AgentType(Enum):
    """
    Enum to classify agent types in the simulation.
    - WASP: Represents a wasp agent.
    - LARVAE: Represents a larvae agent.
    """
    WASP = "Wasp"
    LARVAE = "Larvae"

class WaspRole(Enum):
    """
    Enum to classify the role of a wasp agent.
    - FEEDER: Wasp responsible for feeding larvae.
    - FORAGER: Wasp responsible for collecting food.
    """
    FEEDER = "Feeder"
    FORAGER = "Forager"


# ---------------------------
# Abstract Base Class
# ---------------------------
class Agent(ABC):
    """
    Abstract base class representing a generic agent in the simulation.
    Attributes:
        id (str): Unique identifier for the agent.
        x (int): X-coordinate of the agent in the environment.
        y (int): Y-coordinate of the agent in the environment.
        type (AgentType): The type of agent (Wasp or Larvae).
        hunger (int): Hunger level of the agent.
        storedEvents (List[str]): Log of events performed by the agent.
    """
    def __init__(self, agent_id: str, x: int, y: int, agent_type: AgentType, hunger: int = 0):
        self.id: str = agent_id
        self.x: int = x
        self.y: int = y
        self.type: AgentType = agent_type
        self.hunger: int = hunger
        self.storedEvents: List[str] = []

    def getPosition(self) -> List[int]:
        """
        Return the current position of the agent.
        Returns:
            List[int]: A list [x, y] representing the position.
        """
        return [self.x, self.y]

    @abstractmethod
    def generateEvent(self) -> str:
        """
        Generate a new event string.
        Abstract method: must be implemented by subclasses.
        """
        pass

    def getType(self) -> AgentType:
        """
        Return the type of the agent.
        Returns:
            AgentType: Type of the agent (Wasp or Larvae).
        """
        return self.type

    @abstractmethod
    def step(self, t: int) -> None:
        """
        Perform one simulation step for this agent.
        Abstract method: must be implemented by subclasses.
        Args:
            t (int): Current simulation time.
        """
        pass

    def askForFood(self) -> None:
        """
        Increase hunger level and log a request for food.
        Called every step for larvae and optionally for wasps.
        """
        self.hunger += 1
        self.storedEvents.append("Asked for food")


# ---------------------------
# Subclass: Wasp
# ---------------------------
class Wasp(Agent):
    """
    Represents a wasp agent.
    Attributes:
        food (int): Amount of food the wasp currently holds.
        role (WaspRole): Role of the wasp (Feeder or Forager).
    """
    def __init__(self, agent_id: str, x: int, y: int, role: WaspRole, hunger: int = 0, food: int = 0):
        super().__init__(agent_id, x, y, AgentType.WASP, hunger)
        self.food: int = food
        self.role: WaspRole = role

    # Extra methods
    def feed(self, larvae: "Larvae") -> None:
        """
        Feed a larvae by giving it one unit of food (if available).
        Args:
            larvae (Larvae): The larvae to feed.
        """
        if self.food > 0:
            self.food -= 1
            self.storedEvents.append(f"{self.id} fed larvae {larvae.id}")

    def forage(self) -> None:
        """
        Collect food from the environment.
        (Placeholder: simply increases food by 1.)
        """
        self.food += 1
        self.storedEvents.append(f"{self.id} foraged food")

    def move(self) -> None:
        """
        Move to a new position in the environment.
        (Placeholder: increments x and y by 1.)
        """
        self.x += 1  # placeholder logic
        self.y += 1
        self.storedEvents.append(f"{self.id} moved to {self.getPosition()}")

    def feelGradient(self, gradientField: List[tuple[int, int]]) -> None:
        """
        Sense gradient field (e.g., pheromone trail or resource gradient).
        Args:
            gradientField (List[tuple[int, int]]): Placeholder gradient data.
        """
        self.storedEvents.append(f"{self.id} sensed gradient {gradientField}")

    def decideAction(self) -> None:
        """
        Decide what to do in this step.
        Currently placeholder: logs the decision event.
        """
        self.storedEvents.append(f"{self.id} decided action")

    # Implement abstract methods
    def generateEvent(self) -> str:
        """
        Generate a wasp-specific event string.
        Returns:
            str: Event description.
        """
        return f"Wasp {self.id} event"

    def step(self, t: int) -> None:
        """
        Perform one step of simulation for the wasp.
        Calls decideAction and optionally other actions.
        Args:
            t (int): Current simulation time.
        """
        self.decideAction()


# ---------------------------
# Subclass: Larvae
# ---------------------------
class Larvae(Agent):
    """
    Represents a larvae agent.
    Larvae do not move or forage; they only request food.
    """
    def __init__(self, agent_id: str, x: int, y: int, hunger: int = 0):
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger)

    # Implement abstract methods
    def generateEvent(self) -> str:
        """
        Generate a larvae-specific event string.
        Returns:
            str: Event description.
        """
        return f"Larvae {self.id} event"

    def step(self, t: int) -> None:
        """
        Perform one step of simulation for the larvae.
        Larvae always ask for food.
        Args:
            t (int): Current simulation time.
        """
        self.askForFood()
