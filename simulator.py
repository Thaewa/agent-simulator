# simulator.py
# Defines the Simulator class that manages agents and runs the simulation

from typing import List, Dict
from agents import Agent, Wasp, Larvae
from agents import AgentType, WaspRole
import numpy as np
from utils import gaussian_attraction

class instanceGenerator():
    def __init__(self,larvae_to_wasps_ratio:float=0.3,percentage_foragers:float =0.1,min_number_of_cells:int = 100, max_number_of_cells:int = 300, \
                 nest_fill_percentage:float = 0.3, forage_fill_percentage:float = 0.1, larvae_hunger_multiplier:float = 3.0, mean_food_capacity:float = 10.0, std_food_capacity:float = 3.0, 
                 forage_distance:int = 10,hunger_rate: float = 0.2):
        
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
        self.agent_hunger_ratio=hunger_rate

    def generateLarvae(self,x,y,hunger_multiplier=1.0):
        random_food = np.random.choice([0.9,3,5],1)[0]
        random_hunger = 1 if random_food > 1 else 2
        agent_id = "L"+str(x)+str(y)
        random_hunger_multiplier = np.random.choice([1.0,hunger_multiplier],1)[0]
        return Larvae(agent_id, x=x, y=y, hunger=random_hunger, food=random_food,hungerMultiplier=random_hunger_multiplier,hunger_rate=self.agent_hunger_ratio)
    def generateGrid(self,radius):
        grid_x, grid_y = np.mgrid[-radius:radius+1, -radius:radius+1]
        grid = np.concatenate((grid_x.reshape(-1,1), grid_y.reshape(-1,1)), axis=1)
        return grid
    def addForaging(self, chosen_forage_indices, grid, simulator):
        for row in grid[chosen_forage_indices,:]:
            simulator.addForage(row[0],row[1])
        return simulator
    def addLarvaes(self, chosen_nest_indices, chosen_inner_nest_indices,grid, simulator):
        for row in grid[chosen_inner_nest_indices,:]:
            simulator.addAgent(self.generateLarvae(row[0],row[1],self.larvae_hunger_multiplier))
        for row in grid[chosen_nest_indices,:]:
            simulator.addAgent(self.generateLarvae(row[0],row[1]))
        return simulator
    def addWasps(self, chosen_nest_indices_feeders, chosen_inner_nest_indices_foragers, grid, simulator,path_finding,outer_nest_radius):
        for row in chosen_nest_indices_feeders:
            wasp = self.generateWasp(grid[row,0],grid[row,1],path_finding,outer_nest_radius)
            wasp.role = WaspRole.FEEDER
            simulator.addAgent(wasp)
        for row in chosen_inner_nest_indices_foragers: 
            wasp = self.generateWasp(grid[row,0],grid[row,1],path_finding,outer_nest_radius)
            wasp.role = WaspRole.FORAGER
            simulator.addAgent(wasp)
        return simulator
    def generateWasp(self,x,y,path_finding,outer_nest_radius=None):
        max_food = np.random.normal(self.mean_food_capacity, self.std_food_capacity, 1)[0]
        if outer_nest_radius is not None:
            wasp = Wasp(agent_id="W"+str(x)+str(y), x=x, y=y, food=1,hunger= 1,path_finding=path_finding, max_food = max_food, outer_nest_radius=outer_nest_radius,hunger_rate=self.agent_hunger_ratio)
        else:
            wasp = Wasp(agent_id="W"+str(x)+str(y), x=x, y=y, food=1,hunger= 1,path_finding=path_finding, max_food = max_food, hunger_rate=self.agent_hunger_ratio)
        return wasp
    def generateSimulator(self,path_finding):
        simulator = Simulator()
        number_of_cells = np.random.randint(self.min_number_of_cells, self.max_number_of_cells+1)
        radius_nest = int(number_of_cells**(1/2))+1
        radius_inner_nest = int(radius_nest/2)
        radius_forage = radius_nest+self.forage_distance
        grid = self.generateGrid(radius_forage+10)
        simulator.grid = grid
        inner_nest_indices = np.where(grid[:,0]**2+grid[:,1]**2<radius_inner_nest**2)[0]
        inner_chosen_nest_indices_larvaes = np.random.choice(inner_nest_indices, int(self.nest_fill_percentage/2*inner_nest_indices.shape[0]), replace=False)
        nest_indices = np.where((grid[:,0]**2+grid[:,1]**2<radius_nest**2) & (grid[:,0]**2+grid[:,1]**2>radius_inner_nest**2))[0]
        chosen_nest_indices_larvaes = np.random.choice(nest_indices, int(self.nest_fill_percentage/2*nest_indices.shape[0]), replace=False)

        forage_indices = np.where((grid[:,0]**2+grid[:,1]**2<radius_forage**2) & (grid[:,0]**2+grid[:,1]**2>((radius_nest+4)**2)))[0]
        chosen_forage_indices_forage = np.random.choice(forage_indices, int(self.forage_fill_percentage*forage_indices.shape[0]), replace=False)
        
        total_wasps = int((len(chosen_nest_indices_larvaes)+len(inner_chosen_nest_indices_larvaes))*self.larvae_to_wasps_ratio)
        total_foragers = max(1,int(total_wasps*self.percentage_foragers))
        total_feeders = total_wasps - total_foragers
        
        chosen_nest_indices_feeders = np.random.choice(inner_nest_indices, total_feeders, replace=False)
        chosen_nest_indices_foragers = np.random.choice(nest_indices, total_foragers, replace=False)
        
        simulator = self.addForaging(chosen_forage_indices_forage, grid, simulator)
        simulator = self.addLarvaes(chosen_nest_indices_larvaes, inner_chosen_nest_indices_larvaes,grid, simulator)
        simulator = self.addWasps(chosen_nest_indices_feeders, chosen_nest_indices_foragers, grid, simulator, path_finding,radius_nest)
        
        return simulator
    
    def generateSimulationInstance(self,path_finding:str="greedy"):
        simulator = self.generateSimulator(path_finding)
        return simulator


class Simulator:
    """
    Simulator class that manages a collection of agents and coordinates the simulation.
    Responsible for advancing time, handling agents, and aggregating results.
    """
    def __init__(self):
        """
        Initialize the simulator.

        Attributes:
            currentTime (int): Current time step of the simulation.
            agents (List[Agent]): List of all agents participating in the simulation.
            movementHistory (Dict[str, List[List[int]]]): Dictionary mapping each agent's ID to its movement history.
            gradients (Dict[WaspRole, List[List[float]]]): Dictionary mapping each WaspRole to its gradient values.
            grid (numpy.ndarray): 2D NumPy array representing the grid.
            forage (List[List[int]]): List of all forage points in the simulation.
        """
        self.currentTime: int = 0
        self.agents: List[Agent] = []
        self.movementHistory: Dict[str, List[List[int]]] = {}
        self.gradients = {WaspRole.FEEDER:[],WaspRole.FORAGER:[]}
        self.grid = None
        self.forage = []
        self.forager_ratio = 0.10
        self.potential_feeder_to_forager = 0.25
        self.max_role_changes = 2
                
    # ---------------------------
    # Core methods
    # ---------------------------

    def step(self) -> None:
        """
        Advance the simulation by one time unit.
        Calls step(t) on each agent.
        """
        for agent in self.agents:
            agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
        self.currentTime += 1

    def addAgent(self, agent: Agent) -> None:
        """Add a new agent to the simulation."""
        self.agents.append(agent)
        
        # initialize movement history with starting position
        self.movementHistory[agent.id] = [agent.getPosition()]

    def addForage(self, x: int, y: int) -> None:
        """
        Add a foraging location (placeholder).
        """
        self.forage.append(np.array([x,y]))
        

    def removeAgent(self, agent: Agent) -> None:
        """Remove an agent from the simulation."""
        if agent in self.agents:
            self.agents.remove(agent)

    def accumulateGradients(self) -> None:
        """
        Accumulate gradients generated by the larvae for all agents.
        """
        agents = self.agents
        # get larvae
        larvae = [agent for agent in agents if agent.type==AgentType.LARVAE]
        if len(self.gradients[WaspRole.FEEDER])==0 :
            self.gradients[WaspRole.FEEDER]=np.zeros_like(self.grid[:,0])
        if len(self.gradients[WaspRole.FORAGER])==0:
            self.gradients[WaspRole.FORAGER]=np.zeros_like(self.grid[:,0])
        for agent in larvae:
            if agent.hunger>1:
                x0,y0 = agent.getPosition()
                spread = agent.radius
                peak = max(agent.hunger/min(agent.food+0.1,0.1),0.1)
                # calculate gradient for larvae
                gradient = gaussian_attraction(self.grid[:,0],self.grid[:,1],x0,y0,spread,peak)
                # accumulate gradients for all type of wasps
               
                self.gradients[WaspRole.FEEDER]=self.gradients[WaspRole.FEEDER]+gradient
                self.gradients[WaspRole.FORAGER]=self.gradients[WaspRole.FORAGER]+gradient
               
    def aggregateMovements(self) -> Dict[int, List[tuple[int, int]]]:
        """
        Collect movement data for all agents.
        Returns {agent_id: [(x, y), ...]}
        """
        #for agent in self.agents:
        #    self.movementHistory[agent.id].append(agent.getPosition())
        return self.movementHistory

    def aggregateFeedLarvae(self) -> dict:
        """
        Collect all feeding events performed by wasps.
        Returns:
            dict: {wasp_id: [list of larvae fed]}
        """
        result = {}
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

    def aggregateHungerLarvae(self) -> Dict[int, List[int]]:
        """Collect hunger values for larvae agents."""
        return {a.id: [a.hunger] for a in self.agents if isinstance(a, Larvae)}

    def aggregateHungerWasp(self) -> Dict[int, List[int]]:
        """Collect hunger values for wasp agents."""
        return {a.id: [a.hunger] for a in self.agents if isinstance(a, Wasp)}


    def verifyNumberAgents(self, min_feeders: int = 1, min_foragers: int = 1, min_larvae: int = 1) -> bool:
        
        """
        Verify if the number of agents meets the minimum requirements
        for the simulation to start.

        Args:
            min_feeders (int, optional): Minimum number of feeder wasps. Defaults to 1.
            min_foragers (int, optional): Minimum number of forager wasps. Defaults to 1.
            min_larvae (int, optional): Minimum number of larvae. Defaults to 1.

        Returns:
            bool: True if the number of agents meets the minimum requirements, False otherwise.
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
        return count_feeders >= min_feeders and count_foragers >= min_foragers and count_larvae >= min_larvae
    
    def createGrid(self, padding: int = 3):
        
        """
        Create a grid based on agent positions.

        The grid is a 2D NumPy array that spans the range of x and y
        coordinates of all agents in the simulation.

        Initiates the gradients dictionary for each WaspRole as an empty list.

        Args:
            padding (int, optional): Padding value for the grid. Defaults to 3.
            
        """
        positions_dict = {'x': [agent.x for agent in self.agents], 'y': [agent.y for agent in self.agents]}
        xmin = min(positions_dict['x'])
        xmax = max(positions_dict['x'])
        ymin = min(positions_dict['y'])
        ymax = max(positions_dict['y'])
        
        x, y = np.meshgrid(np.arange(xmin-padding, xmax+padding+1), np.arange(ymin-padding, ymax+padding+1))
        self.grid = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        
        for role in self.gradients:
            self.gradients[role] = np.zeros(self.grid.shape[0])

    def verifyGrid(self):
        """
        Verify if the grid has been initialized correctly for the simulation. The conditions are as follows:

        1. The grid matrix has at least one row.
        2. The grid matrix has two columns.

        Returns:
            bool: True if the grid has been initialized correctly, False otherwise.
        """
        return self.grid.shape[0]>0 and self.grid.shape[1]==2
    
    def verifyForage(self):
        """
        Verify if there are any foraging points in the simulation.

        Returns:
            bool: True if there are any foraging points, False otherwise.
        """
        return len(self.forage)>0

    def verifySimulationConditions(self) -> bool:
        
        """
        Verify if the simulation conditions are valid before running.

        The conditions are as follows:

        1. The number of agents meets the minimum requirements.
        2. The grid has been initialized correctly for the simulation.
        3. There are any foraging points in the simulation.

        Returns:
            bool: True if the simulation conditions are valid, False otherwise.
        """
        num_agents = self.verifyNumberAgents()
        if self.grid is None:
            self.createGrid()
        grid_verification = self.verifyGrid()
        forage_verification = self.verifyForage()
        return num_agents and grid_verification and forage_verification

    def runSimulation(self, t: int) -> List[Dict]:
        """
        Run the simulation for t steps.

        The simulation conditions are as follows:

        1. The number of agents meets the minimum requirements.
        2. The grid has been initialized correctly for the simulation.
        3. There are any foraging points in the simulation.

        Returns a report dictionary containing the following information:

        - movements: a dictionary mapping each agent's ID to its movement history.
        - feedLarvae: a dictionary mapping each larvae's ID to its feeding history.
        - hungerLarvae: a dictionary mapping each larvae's ID to its hunger history.
        - hungerWasp: a dictionary mapping each wasp's ID to its hunger history.

        Raises:
            ValueError: If the simulation conditions are not met.
        """
        if not self.verifySimulationConditions():
            raise ValueError("Simulation conditions not met")
        num_hunger_larvae = []
        i = 0
        larvae = [agent for agent in self.agents if isinstance(agent, Larvae)]
        larvae_position = [agent.getPosition() for agent in larvae]
        total_larvae = len(larvae)
        wasps = [agent for agent in self.agents if isinstance(agent, Wasp)]
        total_wasp = len(wasps)
        total_foragers = sum([1 for agent in wasps if isinstance(agent, Wasp) and agent.role == WaspRole.FORAGER])
        num_foragers = [total_foragers]
        hunger_cues = []
        total_wasp_position = []
        while i < t:
            # Accumulate gradients (placeholder)
            self.accumulateGradients()
            count_roles = 0
            j = 0
            while j < len(self.agents):
                agent = self.agents[j]
                # If the agent is a wasp, perform the following steps
                if isinstance(agent, Wasp):
                    # Update the hunger cue
                    if agent.inOuterNest():
                        local_hunger_cue = agent.estimateLocalHungerCue(self.gradients[agent.role],self.grid)
                        agent.updateHungerCue(local_hunger_cue/self.gradients[agent.role].shape[0])
                    # Get the positions of all wasp agents with the role of FORAGER
                    position_foragers = [agent_.getPosition() for agent_ in wasps if agent_.role == WaspRole.FORAGER]

                    # Get the positions of all wasp agents (excluding the current agent)
                    position_wasp = [wasp.getPosition() for wasp in wasps if agent.id != wasp.id]
                    
                    # If the wasp agent is a FORAGER, feel the gradient of the foraging points
                    if agent.role == WaspRole.FORAGER:
                        agent.feelGradient(self.grid,self.gradients,self.forage,larvaePositions = np.array(larvae_position))
                        # Move the wasp agent based on the gradient
                        agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
                    # If the wasp agent is a FEEDER, feel the gradient of the larvae and wasp agents
                    elif agent.role == WaspRole.FEEDER:
                        agent.feelGradient(self.grid,self.gradients,foragersPositions=np.array(position_foragers),waspPositions = np.array(position_wasp), larvaePositions = np.array(larvae_position))
                        # Move the wasp agent based on the gradient
                        agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
                    else:
                        agent.step(t=self.currentTime)
                j += 1
            total_wasp_position.append([wasp.getPosition() for wasp in wasps])
            total_hunger = [1 for agent in self.agents if isinstance(agent, Larvae) and agent.hunger>1]
            num_hunger_larvae.append(sum(total_hunger)/(total_larvae))
            j = 0
            
            while j < len(self.agents):
                agent = self.agents[j]
                # If the agent is a wasp, perform the following steps
                if isinstance(agent, Wasp):
                    wasp_agents = [agent for agent in self.agents if isinstance(agent, Wasp)]
                    agent_next_positions = np.array([[wasp.x+wasp.next_step['x'],wasp.y+wasp.next_step['y']] for wasp in wasp_agents if wasp != agent])
                    agent.move(agent_next_positions,self.grid)
                    
                # Get the current position of the wasp agent
                    current_pos = agent.getPosition()
                    # If the current position is different from the previous position, add it to the movement history
                    if self.movementHistory[agent.id][-1] != current_pos:
                        self.movementHistory[agent.id].append(current_pos)

                # If the agent's food is less than 1, increase its hunger by the hunger rate
                if agent.food<1:
                    agent.hunger += agent.hungerRate
                
                # If the agent's food minus its hunger rate is greater than 0, decrease its food by the hunger rate
                if agent.food-agent.hungerRate>0:
                    agent.food -= agent.hungerRate
                # Otherwise, set its food to 0
                else:
                    agent.food = 0
                if  isinstance(agent, Wasp):
                    if i % 100 == 0 and count_roles <= self.max_role_changes:
                        if agent.role==WaspRole.FEEDER and agent.hungerCuesHigh() and ((total_foragers+1)/total_wasp)<(self.forager_ratio+self.potential_feeder_to_forager):
                            if np.random.random()>0.8 and agent.rolePersistence == 0 and agent.inOuterNest():
                                agent.role=WaspRole.FORAGER
                                total_foragers+=1
                                agent.updateRolePersistence()
                                count_roles += 1
                                        
                        if agent.role==WaspRole.FORAGER and agent.hungerCuesLow() and ((total_foragers-1)/total_wasp)>max((1/total_wasp),(self.forager_ratio)):
                            if np.random.random()>0.2 and agent.rolePersistence == 0 and agent.inOuterNest():
                                agent.role=WaspRole.FEEDER
                                total_foragers-=1
                                agent.updateRolePersistence()
                                count_roles += 1
                    
                    if agent.rolePersistence>0:
                        agent.rolePersistence-=1            
                j += 1
            # Advance time
            num_foragers.append(total_foragers)
            self.currentTime += 1
            i += 1
            self.clearGradients()
            import matplotlib.pyplot as plt
            hunger_cues.append(agent.hungerCue)
            print(i,agent.hungerCue)
            # if i >200:
            #     wasp_agents = [agent for agent in self.agents if isinstance(agent, Wasp)]
            #     position_wasp = [wasp.getPosition() for wasp in wasp_agents]
            #     position_foragers_ = [wasp.getPosition() for wasp in wasp_agents if wasp.role == WaspRole.FORAGER]
            #     position_wasp = np.array(position_wasp)
            #     position_foragers_ = np.array(position_foragers) 
            #     forage_ = np.array(self.forage)
            #     larvae_positions = np.array(larvae_position)
            #     # print(position_wasp.shape,position_foragers_.shape,forage_.shape)
            #     # plt.scatter(position_wasp[:,0],position_wasp[:,1],c='g')
            #     # plt.scatter(position_foragers_[:,0],position_foragers_[:,1],c='b')
            #     # plt.scatter(forage_[:,0],forage_[:,1],c='r')
            #     # plt.scatter(larvae_positions[:,0],larvae_positions[:,1],c='black',marker='x')
            #     plt.show()
        total_wasp_position=np.array(total_wasp_position)
        # for i in range(total_wasp_position.shape[1]):
            # plt.plot(total_wasp_position[:,i,0],total_wasp_position[:,i,1])
        plt.plot(num_hunger_larvae)
        # plt.plot(num_foragers)
        # plt.plot(hunger_cues)
        plt.show()        
        # Build report dictionary
        report: Dict = {}
        report["movements"] = self.aggregateMovements()
        report["feedLarvae"] = self.aggregateFeedLarvae()
        report["hungerLarvae"] = self.aggregateHungerLarvae()
        report["hungerWasp"] = self.aggregateHungerWasp()

        return report

    def clearGradients(self):
        self.gradients[WaspRole.FEEDER]=[]
        self.gradients[WaspRole.FORAGER]=[]