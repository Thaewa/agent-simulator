# agents.py
# Defines Agent (abstract), Wasp, and Larvae classes based on UML

from abc import ABC, abstractmethod
from typing import List, Dict
from enum import Enum
from utils import gaussian_attraction,estimate_gradient,m_closest_rows,grid_graph_from_array
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
        self.hungerRate: float = 0.01
        

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

    def __init__(self, agent_id: str, x: int, y: int, path_finding: str="greedy", hunger: int = 0, food: int = 0, max_food: int = 10, outer_nest_radius: int = 10,hunger_rate: float = 0.01):
        super().__init__(agent_id, x, y, AgentType.WASP, hunger, food, outer_nest_radius)
        self.path_finding: str = path_finding
        self.smellRadius: int = 2
        self.smellIntensity: float = 5.0
        self.next_step = {'x': 0, 'y': 0}
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
        self.repulsionRadius = 1
        self.rolePersistence = 50
        self.hungerCueDecay = 0.9
        self.prevStep = None if path_finding != 'random_walk' else self.getPosition()
        self.path = None 
    def updateRolePersistence(self):
        self.rolePersistence += 50
    #Assuming that agents can memorize how much is too much and how little is too little
    def updateHungerCueRange(self):
        updates = self.hungerCue
        if self.minHungerCue == None:
            self.minHungerCue = updates
        if self.maxHungerCue == None:
            self.maxHungerCue = updates
        self.minHungerCue = updates if self.minHungerCue>updates else self.minHungerCue
        self.maxHungerCue = updates if self.maxHungerCue<updates else self.maxHungerCue
    # Extra methods
    def feedWasp(self,wasp:Agent, passed_food:int):
        if wasp.hungerCue > 0.0:
            self.hungerCue = (self.hungerCue+wasp.hungerCue)/2
            self.food -= passed_food
            wasp.food += passed_food
            wasp.hunger = 1
            print(f"{self.id} fed {wasp.id} (transfer)")
            self.storedEvents.append(f"{self.id} fed {wasp.id} (transfer)")
    def feedLarvae(self, larvae:Agent, passed_food:int):
        self.food -= passed_food
        larvae.food += passed_food
        larvae.hunger = 1
        print(f"{self.id} fed {larvae.id}")
        self.storedEvents.append(f"{self.id} fed {larvae.id}")
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
            if self.hungerCuesHigh() and self.role == WaspRole.FORAGER:
                if isinstance(target, Larvae):
                    if np.random.rand() < 0.8:
                        self.feedLarvae(target, passed_food)
                else:
                    self.feedWasp(target, passed_food)
            else:
                if isinstance(target, Larvae):
                    self.feedLarvae(target, passed_food)
                else:
                    self.feedWasp(target, passed_food)
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
                    self.food += self.maxFood
                    self.hunger = 1
                    self.storedEvents.append(f"{self.id} is in location {self.getPosition()} and foraged food at location {forage[id]}")
    def inOuterNest(self):
        return ((self.x+self.next_step['x'])**2 + (self.y+self.next_step['y'])**2) < self.outerNestRadius**2
    def hungerCuesLow(self):
        return (self.hungerCue<((self.maxHungerCue+self.minHungerCue)/4)) 
    def foragerMovesCondition(self):    
        return not self.hungerCuesLow()
    def positionInGrid(self,grid:np.ndarray):
        mask = np.all(grid == np.array([self.x+self.next_step['x'],self.y+self.next_step['y']]).T, axis=1)
        idx = np.flatnonzero(mask)
        return len(idx)>1
    
    def move(self, next_step_array: np.ndarray, grid: np.ndarray) -> None:
        
        """
        Move the agent by one step in the direction specified by the next_step attribute.
        The agent will also generate an event string indicating that it moved to a new location.

        """
        if self.next_step['x'] == 0 and self.next_step['y'] == 0:
            self.next_step['x'] = np.random.choice([-1.0, 0.0, 1.0],1)[0]
            self.next_step['y'] = np.random.choice([-1.0, 0.0, 1.0],1)[0]
        if next_step_array.shape[0] != 0 or (not self.positionInGrid(grid)):
            next_step_array_condition = next_step_array==np.array([self.x+self.next_step['x'],self.y+self.next_step['y']]).T
            count = 0
            while (np.any(np.all(next_step_array_condition,axis=1)) and (not self.positionInGrid(grid))) or (self.role==WaspRole.FEEDER and (not self.inOuterNest())):
                self.next_step['x'] = np.random.choice([-1.0, 1.0],1)[0]
                self.next_step['y'] = np.random.choice([-1.0, 1.0],1)[0]
                next_step_array_condition = next_step_array==np.array([self.x+self.next_step['x'],self.y+self.next_step['y']]).T
                count += 1
                if count > 10:
                    self.next_step['x'] = 0.0
                    self.next_step['y'] = 0.0
                    break
        if self.path_finding == "random_walk":
            self.prevStep = self.getPosition()
        self.x += self.next_step['x']
        self.y += self.next_step['y']
        new_pos = self.getPosition()
        self.storedEvents.append(f"{self.id} moved to {new_pos} current hunger level is {format(self.hunger, '.2f')} and food stored is {format(self.food, '.2f')}")
        self.next_step = {'x': 0, 'y': 0}
        
    def hungerCuesHigh(self):
        return (self.hungerCue>(2*((self.maxHungerCue+self.minHungerCue)/3)))
    def estimateLocalHungerCue(self,gradientField,grid):
        index = np.where((grid[:,0]-self.x)**2+(grid[:,1]-self.y)**2<=self.smellRadius**2)[0]
        return sum(gradientField[index])
    def updateHungerCue(self,hungerCue):
        self.hungerCue=hungerCue
    def feelForageGradient(self,gradientField,forage,grid):
        feltGradient = np.zeros_like(gradientField[self.role])
        for forage_position in forage:
            x0,y0 = forage_position
            spread = self.smellRadius
            peak = (self.hunger/max(self.food,0.1))/self.smellIntensity
            gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
            feltGradient = feltGradient + gradient 
        return feltGradient
    def feelForagersGradient(self,gradientField,foragersPositions,grid):
        foragers_index = np.where((foragersPositions[:,0]**2+foragersPositions[:,1]**2)<self.outerNestRadius**2)[0]
        foragersPositions=foragersPositions[foragers_index]
        feltGradient = np.zeros_like(gradientField[self.role])
        for forage_position in foragersPositions:
            x0,y0 = forage_position
            spread = self.smellRadius
            peak = (self.hunger/max(self.food,0.1))/self.smellIntensity
            gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
            feltGradient = feltGradient + gradient 
        return feltGradient
    def addRepulsionGradient(self, feltGradient, waspPositions, grid):
        wasp_index = np.where((((waspPositions[:,0]-self.x)**2+(waspPositions[:,1]-self.y)**2)<=2))[0]
        for index in wasp_index:
            x0,y0 = waspPositions[index]
            spread = self.smellRadius
            peak = self.repulsionRadius
            gradient = gaussian_attraction(grid[:,0],grid[:,1],x0,y0,spread,peak)
            feltGradient = feltGradient - gradient 
        return feltGradient

    def greedy_path_finding(self, grid: np.ndarray, gradientField: Dict[WaspRole, np.ndarray], forage: List[tuple[float, float]]=None,\
                      foragersPositions:np.ndarray=None, waspPositions:np.ndarray=None)-> None:
        # If the wasp is a forager and does not have enough food, add the pheromone trail of the foraging points to the gradient field
        if self.role == WaspRole.FORAGER:
            if self.food<1:
                feltGradient = self.feelForageGradient(gradientField,forage,grid)
            else:
                feltGradient = gradientField[self.role]

        # If the wasp is a feeder and does not have enough food, add the pheromone trail of the forager wasps to the gradient field or 
        # add the pheromone trail of the forager wasp to the gradient field if the hunger cues are higher
        if self.role == WaspRole.FEEDER:
            if self.food<1:
                feltGradient = self.feelForagersGradient(gradientField,foragersPositions, grid)
            else:
                feltGradient = gradientField[self.role]
                feltGradient = self.addRepulsionGradient(feltGradient, waspPositions, grid)
        # Estimate the gradient of the modified gradient field
        dZdx, dZdy = estimate_gradient(grid, feltGradient)
        dZ = np.column_stack((dZdx, dZdy))
        mask = np.all(grid == np.array([self.x,self.y]).T, axis=1)
        idx = np.flatnonzero(mask)
        sign_displacement = np.sign(dZ[idx])
        # Update the next step of the wasp based on the sign of the displacement
        self.next_step[list(self.next_step.keys())[0]] += sign_displacement[0,0].item()
        self.next_step[list(self.next_step.keys())[1]] += sign_displacement[0,1].item()

    def random_walk_path_finding(self):
        new_x = np.random.choice([-1.0, 0.0, 1.0],1)[0]
        new_y = np.random.choice([-1.0, 0.0, 1.0],1)[0]
        if "biased" in self.path_finding:
            while self.prevStep[0]==(self.x+new_x) and self.prevStep[1]==(self.y+new_y):
                new_x = np.random.choice([-1.0, 0.0, 1.0],1)[0]
                new_y = np.random.choice([-1.0, 0.0, 1.0],1)[0]

    def estimate_path(self,x,y,relevant_larvae_position,X,Y):
        arr_full = np.ones_like(X,dtype=bool)
        G = grid_graph_from_array(arr_full,X,Y)
        int_position = [int(pos) for pos in self.getPosition()]
        nodes = np.array(relevant_larvae_position)
        nodes = (nodes.astype(int)).tolist()
        nodes_list = [tuple(node) for node in nodes]
        tsp = nx.approximation.traveling_salesman_problem(G, nodes = nodes_list, cycle=True if "Hamiltonian" in self.path_finding else False)
        int_position = [int(pos) for pos in self.getPosition()]
        print(G.nodes())
        print(tuple(int_position))
        added_path = nx.shortest_path(G,tuple(int_position),tsp[0])
        return added_path+tsp
    def shortest_path_finding(self,larvaePositions):
        if self.path == None:
            index = np.where(((larvaePositions[:,0]-self.x)**2+(larvaePositions[:,1]-self.y)**2<=(self.smellRadius+2)**2) & (((larvaePositions[:,0]-self.x)**2+(larvaePositions[:,1]-self.y)**2)!=0))[0]    
            if len(index) == 0:
                pass
            else:
                min_x = larvaePositions[index][:,0].min()
                max_x = larvaePositions[index][:,0].max()
                min_y = larvaePositions[index][:,1].min()
                max_y = larvaePositions[index][:,1].max()
                x = np.arange(min(min_x-2,self.x-2),max(max_x+2,self.x+2))
                y = np.arange(min(min_y-2,self.y-2),max(max_y+2,self.y+2))
                X, Y = np.meshgrid(x, y)
                m = int(self.food/self.foodTransferToLarvae)
                _,relevant_larvae_position = m_closest_rows(larvaePositions[index],np.array(self.getPosition()),min(len(index),m))
                if relevant_larvae_position.shape[0] == 1:
                    pass
                else:
                    self.path = self.estimate_path(x,y,relevant_larvae_position,X,Y)
                    print(self.path)
                    self.next_step['x'] = self.path[0][0]-self.x 
                    self.next_step['y'] = self.path[0][1]-self.y
                    self.path = self.path[1:]
        else:
            if abs(self.path[0][0]-self.x) >1 or abs(self.path[0][1]-self.y)>1:
                self.path = None
            else:
                self.next_step['x'] = self.path[0][0]-self.x
                self.next_step['y'] = self.path[0][1]-self.y
                if len(self.path)==1:
                    self.path = None
                else:
                    print(self.path[0],self.x,self.y)
                    self.path = self.path[1:]
                
    def feelGradient(self, grid: np.ndarray, gradientField: Dict[WaspRole, np.ndarray], forage: List[tuple[float, float]]=None,\
                      foragersPositions:np.ndarray=None, waspPositions:np.ndarray=None, larvaePositions:np.ndarray=None) -> None:
        
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
        hungerCue = sum(gradientField[self.role])/gradientField[self.role].shape[0]
        self.updateHungerCue(hungerCue)
        self.updateHungerCueRange()

        if self.food<1.1:
            self.greedy_path_finding(grid, gradientField, forage,foragersPositions, waspPositions)
        elif self.food<self.foodTransferToWasp and self.role == WaspRole.FORAGER:
            self.greedy_path_finding(grid, gradientField, forage,foragersPositions, waspPositions)
        else:
            # Initialize the felt gradient to zero if the wasp does not have enough food
            if self.path_finding == "greedy":
                self.greedy_path_finding(grid, gradientField, forage,foragersPositions, waspPositions)
            elif "random_walk" in self.path_finding:
                self.random_walk_path_finding()
            elif "TSP" in self.path_finding:
                self.greedy_path_finding(grid, gradientField, forage,foragersPositions, waspPositions)
                self.shortest_path_finding(larvaePositions)
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
        masks = (entities<=2).all(axis=1)
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
                if random.random()>self.chanceOfFeeding and agents[id].hunger>1:
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
        
        # Split agents into groups
        wasps = [agent for agent in agents if agent.type == AgentType.WASP]
        wasps_feeders = [wasp for wasp in wasps if wasp.role == WaspRole.FEEDER]
        larvaes = [agent for agent in agents if agent.type == AgentType.LARVAE]
        net_receivers = wasps_feeders + larvaes

        # Feeding logic
        if self.food > 0:
            if self.role == WaspRole.FORAGER :
                if self.hungerCuesHigh():
                    self.feedNearby(wasps_feeders)
                else: 
                    self.feedNearby(net_receivers)
            elif self.role == WaspRole.FEEDER:
                self.feedNearby(larvaes)
                
        # Foraging logic
        if self.role == WaspRole.FORAGER and forage is not None:
            self.forage(forage)
        self.hungerCue=self.hungerCue*self.hungerCueDecay

# ---------------------------
# Subclass: Larvae
# ---------------------------
class Larvae(Agent):
    """
    Represents a larvae agent.
    Larvae do not move or forage; they only request food.
    """
    def __init__(self, agent_id: str, x: int, y: int, hunger: int = 0, food: int = 0, hungerMultiplier: float = 1.0,hunger_rate: float = 0.01):
        super().__init__(agent_id, x, y, AgentType.LARVAE, hunger, food)
        self.hunger_rate = hunger_rate
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
