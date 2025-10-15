# agents.py
# Defines Agent (abstract), Wasp, and Larvae classes based on UML

from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
# from utils import gaussian_attraction,estimate_gradient
import numpy as np
import random
import matplotlib.pyplot as plt
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
        radius (int): The radius of the agent.
        food (int): The amount of food currently stored by the agent.
        hungerRate (float): The rate at which the agent's hunger level increases.
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
    
    def step(self, t: int) -> None:
        """
        Perform one simulation step for this agent.
        Optional abstract method: may be implemented by subclasses.
        Args:
            t (int): Current simulation time.
        """
        pass

# ---------------------------
# Subclass: Wasp
# ---------------------------
class Wasp(Agent):
    """
    Represents a wasp agent in the simulation.

    Attributes:
        food (int): Amount of food the wasp currently holds.
        role (WaspRole): Role of the wasp (Feeder or Forager).
        smellRadius (int): Radius of the smell perception of the wasp.
        smellIntensity (float): Intensity of the smell perception of the wasp.
        next_step (dict): Next step of the wasp in the simulation.
        chanceOfFeeding (float): Chance of feeding the larvae.
        forageIncrease (int): Amount of food the wasp gains from foraging.
        maxFood (int): Maximum amount of food the wasp can hold.
        chanceOfForaging (float): Chance of foraging.
        foodTransferToWasp (int): Amount of food transferred to the wasp.
        foodTransferToLarvae (int): Amount of food transferred to the larvae.

    Methods:
        feed(target: Agent): Feed an agent by giving it one unit of food (if available).
        forage(forage: List[np.ndarray]): Move towards foraging points.
        step(t: int): Perform one simulation step for this wasp.
        generateEvent(): Generate a new event string.
        move(simulator=None): Move to a new position in the environment.
        feelGradient(grid: np.ndarray, gradientField: Dict[WaspRole, np.ndarray]): Sense gradient field (e.g., pheromone trail or resource gradient).
    """

    def __init__(self, agent_id: str, x: int, y: int, role: WaspRole, hunger: int = 0, food: int = 0):
        super().__init__(agent_id, x, y, AgentType.WASP, hunger, food)
        self.role: WaspRole = role
        self.smellRadius: int = 10
        self.smellIntensity: float = 5.0
        self.next_step = {'x': 0, 'y': 0}
        self.chanceOfFeeding: float = 0.3
        self.forageIncrease: int = 10
        self.maxFood: int = 20
        self.chanceOfForaging: float = 0.3
        self.foodTransferToWasp: int = 5
        self.foodTransferToLarvae: int = 1

    # Extra methods
    def feed(self, target:Agent) -> None:
        """
        Feed an agent by giving it one unit of food (if available).
        If the target is a larvae, the wasp will transfer food to the larvae.
        If the target is a wasp, the wasp will transfer food to the target wasp.
        Args:
            target (Agent): The agent to feed.
        """
        passed_food = self.foodTransferToLarvae if target.type == AgentType.LARVAE else self.foodTransferToWasp
        if self.food > passed_food:
            self.food -= passed_food
            target.food += passed_food
            target.hunger = 1
            if isinstance(target, Larvae):
                self.storedEvents.append(f"{self.id} fed {target.id}")
            elif isinstance(target, Wasp) and target.role == WaspRole.FEEDER:
                self.storedEvents.append(f"{self.id} fed {target.id} (transfer)")
        else:
            self.storedEvents.append(f"{self.id} tried to feed {target.id} but had no food")

    def forage(self,forage: List[np.ndarray]) -> None:
        
        """
        Move towards foraging points.

        Args:
            forage (List[np.ndarray]): List of foraging points in the environment.

        If the wasp is within the smell radius of a foraging point and the chance of foraging is greater than a random number, 
        the wasp will move towards the foraging point and gain food. The amount of food gained is equal to the foraging increase attribute of the wasp.

        The wasp will also generate an event string indicating that it foraged food at the given location.

        """

        forage=np.array(forage)
        idx = self.nearbyEntity(forage)
        if idx.shape[0]>0 :
            for id in idx:
                if random.random()>self.chanceOfForaging and self.food+self.forageIncrease<self.maxFood:
                    self.food += self.forageIncrease
                    self.hunger = 1
                    self.storedEvents.append(f"{self.id} is in location {self.getPosition()} and foraged food at location {forage[id]}")

    def move(self) -> None:
        
        """
        Move the agent by one step in the direction specified by the next_step attribute.

        
        The agent will also generate an event string indicating that it moved to a new location.

        """
        self.x += self.next_step['x']
        self.y += self.next_step['y']
        new_pos = self.getPosition()
        self.next_step = {'x': 0, 'y': 0}
        self.storedEvents.append(f"{self.id} moved to {new_pos} current hunger level is {format(self.hunger, '.2f')} and food stored is {format(self.food, '.2f')}")

    def feelGradient(self, grid: np.ndarray, gradientField: Dict[WaspRole, np.ndarray], forage: List[tuple[float, float]]=None,\
                      foragersPositions:List[np.ndarray]=None) -> None:
        
        """
        Feel the gradient of the environment.

        If the wasp has not enough food (i.e., less than the hunger rate times 10), it will ignore the larvae gradient field.
        If the wasp is a forager and has not enough food, it will also add the pheromone trail of the foraging points to the gradient field.
        If the wasp is a feeder and has not enough food, it will also add the pheromone trail of the forager wasps to the gradient field.

        The wasp will then move in the direction of the gradient of the modified gradient field.

        Args:
            grid (np.ndarray): 2D NumPy array representing the grid.
            gradientField (Dict[WaspRole, np.ndarray]): Dictionary mapping each WaspRole to its gradient values.
            forage (List[tuple[float, float]], optional): List of foraging points in the environment. Defaults to None.
            foragersPositions (List[np.ndarray], optional): List of forager wasps' positions in the environment. Defaults to None.

        Returns:
            None
        """

        # Initialize the felt gradient to zero if the wasp does not have enough food
        feltGradient = np.zeros_like(gradientField[self.role]) if self.food<(self.hungerRate*10) else gradientField[self.role].copy()
        
        # If the wasp is a forager and does not have enough food, add the pheromone trail of the foraging points to the gradient field
        if self.role == WaspRole.FORAGER and self.food<(self.hungerRate*10):
            for forage_position in forage:
                x0,y0 = forage_position
                spread = self.smellRadius
                peak = (self.hunger/max(self.food,0.1))/self.smellIntensity
                gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
                feltGradient = feltGradient + gradient
        
        # If the wasp is a feeder and does not have enough food, add the pheromone trail of the forager wasps to the gradient field
        if self.role == WaspRole.FEEDER and self.food<(self.hungerRate*10):
            for forager_position in foragersPositions:
                x0,y0 = forager_position
                spread = self.smellRadius
                peak = (self.hunger/max(self.food,0.1))/self.smellIntensity
                gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
                feltGradient = feltGradient + gradient
            
        # Estimate the gradient of the modified gradient field
        dZdx, dZdy = estimate_gradient(grid, feltGradient)
        dZ = np.column_stack((dZdx, dZdy))
        mask = np.all(grid == np.array([self.x,self.y]).ravel(), axis=1)
        idx = np.flatnonzero(mask)
        sign_displacement = np.sign(dZ[idx])
        
        # Update the next step of the wasp based on the sign of the displacement
        self.next_step[list(self.next_step.keys())[0]] += sign_displacement[0,0].item()
        self.next_step[list(self.next_step.keys())[1]] += sign_displacement[0,1].item()
        
        # Log the event of sensing the gradient
        self.storedEvents.append(f"{self.id} sensed gradient ")

   
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
        Feed nearby agents if the chance of feeding is greater than a random number.

        Args:
            agents (List[Agent]): List of agents to check for feeding.

        Returns:
            None
        """
        if not agents:   # <-- Prevent empty list bug
            return
    
        agents_position = np.array([agent.getPosition() for agent in agents])
        idx = self.nearbyEntity(agents_position)
        if idx.shape[0]>0:
            for id in idx:
                if random.random()>self.chanceOfFeeding:
                    self.feed(agents[id])
        

    def step(self, t: int = None, agents: List[Agent] = None, forage=None) -> None:
        """
        Advance the wasp's state by one time step.

        Args:
            t (int, optional): Current simulation time (not used directly).
            agents (List[Agent], optional): List of all agents in the simulation.
            forage (List[tuple[float, float]], optional): List of foraging points in the simulation.

        Returns:
            None
        """
        # Default to empty list if agents not provided
        if agents is None:
            agents = []

        # Split agents into groups
        wasps = [agent for agent in agents if agent.type == AgentType.WASP]
        wasps_feeders = [wasp for wasp in wasps if wasp.role == WaspRole.FEEDER]
        larvaes = [agent for agent in agents if agent.type == AgentType.LARVAE]
        net_receivers = wasps_feeders + larvaes

        # Feeding logic
        if self.food > 0:
            if self.role == WaspRole.FORAGER:
                self.feedNearby(net_receivers)
            elif self.role == WaspRole.FEEDER:
                self.feedNearby(larvaes)

        # Foraging logic
        if self.role == WaspRole.FORAGER and forage is not None:
            self.forage(forage)

        # Movement
        self.move()

# ---------------------------
# Subclass: Larvae
# ---------------------------
class Larvae(Agent):
    """
    Represents a larvae agent.
    Larvae do not move or forage; they only request food.
    """
    def __init__(self, agent_id: str, x: int, y: int, hunger: int = 0, food: int = 0, hungerMultiplier: float = 1.0):
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger, food)
        self.hungerMultiplier = hungerMultiplier
        self.hungerRate = self.hungerMultiplier * self.hungerRate
    # Implement abstract methods
    def generateEvent(self) -> str:
        """
        Generate a larvae-specific event string.
        Returns:
            str: Event description.
        """
        return f"Larvae {self.id} event"
    
    def step(self, t: int = None, agents: List[Agent] = None, forage=None) -> None:
        """
        Advance the larvae's state by one time step.
        Larvae cannot move or forage; they only increase hunger and request food.
        """
        self.hunger += self.hungerRate
        self.storedEvents.append(f"{self.id} asked for food at time {t}")
