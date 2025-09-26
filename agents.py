# agents.py
# Defines Agent (abstract), Wasp, and Larvae classes based on UML

from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
from utils import gaussian_attraction,estimate_gradient
import numpy as np
import random
from simulator import Simulator
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
    def __init__(self, agent_id: str, x: int, y: int, agent_type: AgentType, hunger: int = 0, food: int=0, radius: int =5):
        self.id: str = agent_id
        self.x: int = x
        self.y: int = y
        self.type: AgentType = agent_type
        self.hunger: int = hunger
        self.storedEvents: List[str] = []
        self.radius: int = radius
        self.food: int = food
        self.hungerRate: float = 0.2
        
        

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
        super().__init__(agent_id, x, y, AgentType.WASP, hunger, food)
        self.role: WaspRole = role
        self.smellRadius: int = 5
        self.smellIntensity: float = 1.0
        self.next_step = {'x': 0, 'y': 0}
        self.chanceOfFeeding: float = 0.3
        self.forageIncrease: int = 10
        self.maxFood: int = 20
        self.chanceOfForaging: float = 0.3
    # Extra methods
    def feed(self, target:Agent) -> None:
        """
        Feed an agent by giving it one unit of food (if available).
        """
        if self.food > 0:
            self.food -= 1
            target.food += 1
            if isinstance(target, Larvae):
                self.storedEvents.append(f"{self.id} fed {target.id}")
            elif isinstance(target, Wasp) and target.role == WaspRole.FEEDER:
                self.storedEvents.append(f"{self.id} fed {target.id} (transfer)")
        else:
            self.storedEvents.append(f"{self.id} tried to feed {target.id} but had no food")

    def forage(self,forage: List[np.ndarray]) -> None:
        """
        Collect food from the environment.
        (Placeholder: simply increases food by 1.)
        """
        forage=np.array(forage)
        idx = self.nearbyEntity(forage)
        if idx.shape[0]>0 :
            for id in idx:
                if random.random()>self.chanceOfForaging and self.food<self.maxFood:
                    self.food += self.forageIncrease
        self.storedEvents.append(f"{self.id} foraged food")

    def move(self, simulator:Simulator=None) -> None:
        """
        Move to a new position in the environment.
        (Placeholder: increments x and y by 1.)
        """
        old_pos = self.getPosition()
        self.x += self.next_step['x']
        self.y += self.next_step['y']
        new_pos = self.getPosition()
        self.next_step = {'x': 0, 'y': 0}
        if simulator and new_pos != old_pos:
            simulator.movementHistory[self.id].append(new_pos)
        self.storedEvents.append(f"{self.id} moved to {new_pos}")

    def feelGradient(self, grid: np.ndarray, gradientField: Dict[WaspRole, np.ndarray], forage: List[tuple[float, float]]=None) -> None:
        """
        Sense gradient field (e.g., pheromone trail or resource gradient).
        Args:
            gradientField (List[tuple[int, int]]): Placeholder gradient data.
        """
        feltGradient = gradientField[self.role].copy()
        if self.role == WaspRole.FORAGER:
            for forage_position in forage:
                x0,y0 = forage_position
                spread = self.smellRadius
                peak = (self.hunger/max(self.food,0.1))/self.smellIntensity
                gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
                feltGradient = feltGradient + gradient
        dZdx, dZdy = estimate_gradient(grid, feltGradient)
        dZ = np.column_stack((dZdx, dZdy))
        mask = np.all(grid == np.array([self.x,self.y]).ravel(), axis=1)
        idx = np.flatnonzero(mask)
        leading_displacement_index = np.argmax(dZ[idx,:])
        sign_displacement = np.sign(dZ[idx,leading_displacement_index])
        self.next_step[list(self.next_step.keys())[leading_displacement_index]] += sign_displacement.item()
        self.storedEvents.append(f"{self.id} sensed gradient {feltGradient}")

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

    def nearbyEntity(self,entities:np.ndarray) -> np.ndarray:
        """
        Return indices of entities within one unit of the wasp.

        Args:
            entities (numpy.ndarray): 2D array of entity positions.

        Returns:
            numpy.ndarray: Indices of entities within one unit of the wasp.
        """
        entities = np.abs(entities-np.array([self.x,self.y]))
        masks = (entities<=1).all(axis=1)
        idx = np.flatnonzero(masks)
        return idx
    
    def feedNearby(self, agents) -> None:
        """
        Feed nearby agents.
        """
        agents = np.array([agent.getPosition() for agent in agents])
        idx = self.nearbyEntity(agents)
        if idx.shape[0]>0:
            for id in idx:
                if random.random()>self.chanceOfFeeding:
                    self.feed(agents[id])
        

    def step(self,agents,forage) -> None:
        """
        Perform one step of simulation for the wasp.
        Calls decideAction and optionally other actions.
        """
        # Case 1: Forager
        wasps = [agent for agent in agents if agent.type == AgentType.WASP]
        wasps_feeders =[wasp for wasp in wasps if wasp.role == WaspRole.FEEDER]
        larvaes = [agent for agent in agents if agent.type == AgentType.LARVAE]
        wasps_foragers = [wasp for wasp in wasps if wasp.role == WaspRole.FORAGER]
        net_receivers = wasps_feeders + larvaes
        if self.food > 0:
            if self.role == WaspRole.FORAGER:
                self.feedNearby(net_receivers)        
            elif self.role == WaspRole.FEEDER:
                self.feedNearby(larvaes)
        if self.role == WaspRole.FORAGER:
                self.forage(forage)
        self.move()
        self.hunger += self.hungerRate
        self.food -= self.hungerRate

    def transfer_food(self, feeder: "Wasp") -> None:
        """
        Transfer one unit of food to a feeder wasp.
        Used when this wasp is a Forager.
        """
        if self.food > 0 and feeder.role == WaspRole.FEEDER:
            self.food -= 1
            feeder.food += 1
            self.storedEvents.append(f"{self.id} transferred food to {feeder.id}")
        else:
            self.storedEvents.append(f"{self.id} tried to transfer food but failed")

    def move_towards(self, target: Agent, simulator=None) -> None:
        """
        Move one step towards the target agent.
        """
        if self.x < target.x:
            self.x += 1
        elif self.x > target.x:
            self.x -= 1

        if self.y < target.y:
            self.y += 1
        elif self.y > target.y:
            self.y -= 1

        # ถ้ามี simulator ให้ log เฉพาะตอนที่มีการเคลื่อนจริง
        if simulator:
            simulator.movementHistory[self.id].append(self.getPosition())

        self.storedEvents.append(
            f"{self.id} moved towards {target.id} → {self.getPosition()}"
        )


# ---------------------------
# Subclass: Larvae
# ---------------------------
class Larvae(Agent):
    """
    Represents a larvae agent.
    Larvae do not move or forage; they only request food.
    """
    def __init__(self, agent_id: str, x: int, y: int, hunger: int = 0, food: int = 0):
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger, food)

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
        # Set hunger threshold as 3
        if self.hunger > 3:
            self.askForFood()
        else:
            # Log that Larvae is full / not hungry
            self.storedEvents.append(f"{self.id} is full at time {t}")
