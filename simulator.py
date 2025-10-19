# simulator.py
# Defines the Simulator class that manages agents and runs the simulation
from __future__ import annotations

from typing import List, Optional, Dict, Tuple
from agents import Agent, Wasp, Larvae
from agents import AgentType, WaspRole
import numpy as np
from utils import gaussian_attraction, in_circle, outside_circle

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

    def generateLarvae(self, x: int, y: int, hunger_multiplier: float = 1.0) -> "Larvae":
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
        random_hunger_multiplier = np.random.choice([1.0, hunger_multiplier], 1)[0]
        return Larvae(
            agent_id,
            x=x,
            y=y,
            hunger=random_hunger,
            food=random_food,
            hungerMultiplier=random_hunger_multiplier,
            hunger_rate=self.agent_hunger_ratio,
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
            simulator.addAgent(self.generateLarvae(row[0], row[1], self.larvae_hunger_multiplier))
        for row in grid[chosen_nest_indices, :]:
            simulator.addAgent(self.generateLarvae(row[0], row[1]))
        return simulator

    def addWasps(
        self,
        chosen_nest_indices_feeders: np.ndarray,
        chosen_inner_nest_indices_foragers: np.ndarray,
        grid: np.ndarray,
        simulator: "Simulator",
        path_finding: str,
        outer_nest_radius: int,
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
        :param path_finding: Path-finding mode to assign to generated wasps.
        :type path_finding: str
        :param outer_nest_radius: Nest radius used by wasps for constraints.
        :type outer_nest_radius: int
        :return: The same simulator, for chaining.
        :rtype: Simulator
        """
        for row in chosen_nest_indices_feeders:
            wasp = self.generateWasp(grid[row, 0], grid[row, 1], path_finding, outer_nest_radius)
            wasp.role = WaspRole.FEEDER
            simulator.addAgent(wasp)
        for row in chosen_inner_nest_indices_foragers:
            wasp = self.generateWasp(grid[row, 0], grid[row, 1], path_finding, outer_nest_radius)
            wasp.role = WaspRole.FORAGER
            simulator.addAgent(wasp)
        return simulator

    def generateWasp(self, x: int, y: int, path_finding: str, outer_nest_radius: Optional[int] = None) -> "Wasp":
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
        if outer_nest_radius is not None:
            wasp = Wasp(
                agent_id="W" + str(x) + str(y),
                x=x,
                y=y,
                food=1,
                hunger=1,
                path_finding=path_finding,
                max_food=max_food,
                outer_nest_radius=outer_nest_radius,
                hunger_rate=self.agent_hunger_ratio,
            )
        else:
            wasp = Wasp(
                agent_id="W" + str(x) + str(y),
                x=x,
                y=y,
                food=1,
                hunger=1,
                path_finding=path_finding,
                max_food=max_food,
                hunger_rate=self.agent_hunger_ratio,
            )
        return wasp

    # -------------------------
    # High-level instance maker
    # -------------------------

    def generateSimulator(self, path_finding: str) -> "Simulator":
        """
        Create and populate a :class:`Simulator` with grid, larvae, wasps, and forage.

        Steps:
        1. Draw number of cells and derive radii (nest, inner nest, forage).
        2. Build grid with padding.
        3. Sample inner/outer larvae placements and forage ring cells.
        4. Compute wasp counts and split into foragers/feeders.
        5. Populate the simulator accordingly.

        :param path_finding: Path-finding mode for generated wasps.
        :type path_finding: str
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
            chosen_nest_indices_feeders, chosen_nest_indices_foragers, grid, simulator, path_finding, radius_nest
        )

        return simulator

    def generateSimulationInstance(self, path_finding: str = "greedy") -> "Simulator":
        """
        Convenience wrapper for :meth:`generateSimulator`.

        :param path_finding: Path-finding mode for created wasps.
        :type path_finding: str
        :return: Populated simulator instance.
        :rtype: Simulator
        """
        simulator = self.generateSimulator(path_finding)
        return simulator

# class Simulator:
#     """
#     Simulator class that manages a collection of agents and coordinates the simulation.
#     Responsible for advancing time, handling agents, and aggregating results.
#     """
#     def __init__(self):
#         """
#         Initialize the simulator.

#         Attributes:
#             currentTime (int): Current time step of the simulation.
#             agents (List[Agent]): List of all agents participating in the simulation.
#             movementHistory (Dict[str, List[List[int]]]): Dictionary mapping each agent's ID to its movement history.
#             gradients (Dict[WaspRole, List[List[float]]]): Dictionary mapping each WaspRole to its gradient values.
#             grid (numpy.ndarray): 2D NumPy array representing the grid.
#             forage (List[List[int]]): List of all forage points in the simulation.
#         """
#         self.currentTime: int = 0
#         self.agents: List[Agent] = []
#         self.movementHistory: Dict[str, List[List[int]]] = {}
#         self.gradients = {WaspRole.FEEDER:[],WaspRole.FORAGER:[]}
#         self.grid = None
#         self.forage = []
#         self.forager_ratio = 0.10
#         self.potential_feeder_to_forager = 0.25
#         self.max_role_changes = 2
#         self.role_changes_frequency = 100
                
#     # ---------------------------
#     # Core methods
#     # ---------------------------

#     def step(self) -> None:
#         """
#         Advance the simulation by one time unit.
#         Calls step(t) on each agent.
#         """
#         for agent in self.agents:
#             agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
#         self.currentTime += 1

#     def addAgent(self, agent: Agent) -> None:
#         """Add a new agent to the simulation."""
#         self.agents.append(agent)
        
#         # initialize movement history with starting position
#         self.movementHistory[agent.id] = [agent.getPosition()]

#     def addForage(self, x: int, y: int) -> None:
#         """
#         Add a foraging location (placeholder).
#         """
#         self.forage.append(np.array([x,y]))
        

#     def removeAgent(self, agent: Agent) -> None:
#         """Remove an agent from the simulation."""
#         if agent in self.agents:
#             self.agents.remove(agent)

#     def accumulateGradients(self) -> None:
#         """
#         Accumulate gradients generated by the larvae for all agents.
#         """
#         agents = self.agents
#         # get larvae
#         larvae = [agent for agent in agents if agent.type==AgentType.LARVAE]
#         if len(self.gradients[WaspRole.FEEDER])==0 :
#             self.gradients[WaspRole.FEEDER]=np.zeros_like(self.grid[:,0])
#         if len(self.gradients[WaspRole.FORAGER])==0:
#             self.gradients[WaspRole.FORAGER]=np.zeros_like(self.grid[:,0])
#         for agent in larvae:
#             if agent.hunger>agent.noHunger:
#                 x0,y0 = agent.getPosition()
#                 spread = agent.radius
#                 peak = max(agent.hunger/min(agent.food+0.1,0.1),0.1)
#                 # calculate gradient for larvae
#                 gradient = gaussian_attraction(self.grid[:,0],self.grid[:,1],x0,y0,spread,peak)
#                 # accumulate gradients for all type of wasps
               
#                 self.gradients[WaspRole.FEEDER]=self.gradients[WaspRole.FEEDER]+gradient
#                 self.gradients[WaspRole.FORAGER]=self.gradients[WaspRole.FORAGER]+gradient
               
#     def aggregateMovements(self) -> Dict[int, List[tuple[int, int]]]:
#         """
#         Collect movement data for all agents.
#         Returns {agent_id: [(x, y), ...]}
#         """
#         #for agent in self.agents:
#         #    self.movementHistory[agent.id].append(agent.getPosition())
#         return self.movementHistory

#     def aggregateFeedLarvae(self) -> dict:
#         """
#         Collect all feeding events performed by wasps.
#         Returns:
#             dict: {wasp_id: [list of larvae fed]}
#         """
#         result = {}
#         for agent in self.agents:
#             if isinstance(agent, Wasp):
#                 result[agent.id] = {}
#                 for event in agent.storedEvents:
#                     if "fed" in event:
#                         parts = event.split()
#                         if len(parts) >= 3:
#                             target = parts[2]
#                             result[agent.id][target] = result[agent.id].get(target, 0) + 1
#         return result

#     def aggregateHungerLarvae(self) -> Dict[int, List[int]]:
#         """Collect hunger values for larvae agents."""
#         return {a.id: [a.hunger] for a in self.agents if isinstance(a, Larvae)}

#     def aggregateHungerWasp(self) -> Dict[int, List[int]]:
#         """Collect hunger values for wasp agents."""
#         return {a.id: [a.hunger] for a in self.agents if isinstance(a, Wasp)}


#     def verifyNumberAgents(self, min_feeders: int = 1, min_foragers: int = 1, min_larvae: int = 1) -> bool:
        
#         """
#         Verify if the number of agents meets the minimum requirements
#         for the simulation to start.

#         Args:
#             min_feeders (int, optional): Minimum number of feeder wasps. Defaults to 1.
#             min_foragers (int, optional): Minimum number of forager wasps. Defaults to 1.
#             min_larvae (int, optional): Minimum number of larvae. Defaults to 1.

#         Returns:
#             bool: True if the number of agents meets the minimum requirements, False otherwise.
#         """

#         count_feeders = 0
#         count_larvae = 0
#         count_foragers = 0
#         for agent in self.agents:
#             if isinstance(agent, Wasp):
#                 if agent.role == WaspRole.FEEDER:
#                     count_feeders += 1
#                 elif agent.role == WaspRole.FORAGER:
#                     count_foragers += 1
#             elif isinstance(agent, Larvae):
#                 count_larvae += 1
#         return count_feeders >= min_feeders and count_foragers >= min_foragers and count_larvae >= min_larvae
    
#     def createGrid(self, padding: int = 3):
        
#         """
#         Create a grid based on agent positions.

#         The grid is a 2D NumPy array that spans the range of x and y
#         coordinates of all agents in the simulation.

#         Initiates the gradients dictionary for each WaspRole as an empty list.

#         Args:
#             padding (int, optional): Padding value for the grid. Defaults to 3.
            
#         """
#         positions_dict = {'x': [agent.x for agent in self.agents], 'y': [agent.y for agent in self.agents]}
#         xmin = min(positions_dict['x'])
#         xmax = max(positions_dict['x'])
#         ymin = min(positions_dict['y'])
#         ymax = max(positions_dict['y'])
        
#         x, y = np.meshgrid(np.arange(xmin-padding, xmax+padding+1), np.arange(ymin-padding, ymax+padding+1))
#         self.grid = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        
#         for role in self.gradients:
#             self.gradients[role] = np.zeros(self.grid.shape[0])

#     def verifyGrid(self):
#         """
#         Verify if the grid has been initialized correctly for the simulation. The conditions are as follows:

#         1. The grid matrix has at least one row.
#         2. The grid matrix has two columns.

#         Returns:
#             bool: True if the grid has been initialized correctly, False otherwise.
#         """
#         return self.grid.shape[0]>0 and self.grid.shape[1]==2
    
#     def verifyForage(self):
#         """
#         Verify if there are any foraging points in the simulation.

#         Returns:
#             bool: True if there are any foraging points, False otherwise.
#         """
#         return len(self.forage)>0

#     def verifySimulationConditions(self) -> bool:
        
#         """
#         Verify if the simulation conditions are valid before running.

#         The conditions are as follows:

#         1. The number of agents meets the minimum requirements.
#         2. The grid has been initialized correctly for the simulation.
#         3. There are any foraging points in the simulation.

#         Returns:
#             bool: True if the simulation conditions are valid, False otherwise.
#         """
#         num_agents = self.verifyNumberAgents()
#         if self.grid is None:
#             self.createGrid()
#         grid_verification = self.verifyGrid()
#         forage_verification = self.verifyForage()
#         return num_agents and grid_verification and forage_verification

    
#     def runSimulation(self, t: int) -> List[Dict]:
#         """
#         Run the simulation for t steps.

#         The simulation conditions are as follows:

#         1. The number of agents meets the minimum requirements.
#         2. The grid has been initialized correctly for the simulation.
#         3. There are any foraging points in the simulation.

#         Returns a report dictionary containing the following information:

#         - movements: a dictionary mapping each agent's ID to its movement history.
#         - feedLarvae: a dictionary mapping each larvae's ID to its feeding history.
#         - hungerLarvae: a dictionary mapping each larvae's ID to its hunger history.
#         - hungerWasp: a dictionary mapping each wasp's ID to its hunger history.

#         Raises:
#             ValueError: If the simulation conditions are not met.
#         """
#         if not self.verifySimulationConditions():
#             raise ValueError("Simulation conditions not met")
#         i = 0
#         larvae = [agent for agent in self.agents if isinstance(agent, Larvae)]
#         larvaes_position = [agent.getPosition() for agent in larvae]
#         wasps = [agent for agent in self.agents if isinstance(agent, Wasp)]
#         total_wasp = len(wasps)
#         total_foragers = sum([1 for agent in wasps if isinstance(agent, Wasp) and agent.role == WaspRole.FORAGER])
#         while i < t:
#             # Accumulate gradients (placeholder)
#             self.accumulateGradients()
#             count_roles = 0
#             j = 0
#             while j < len(self.agents):
#                 agent = self.agents[j]
#                 # If the agent is a wasp, perform the following steps
#                 if isinstance(agent, Wasp):
#                     self.stepAgent(agent,wasps,larvaes_position)
#                 j += 1
#             j = 0
            
#             while j < len(self.agents):
#                 agent = self.agents[j]
#                 # If the agent is a wasp, perform the following steps
#                 if isinstance(agent, Wasp):
#                     self.moveAgent(agent)
#                 self.updateAgent(agent)
#                 if  isinstance(agent, Wasp):
#                     count_roles,total_foragers = self.updateRoles(t,agent,count_roles,total_foragers,total_wasp)   
#                 j += 1
#             # Advance time
#             self.currentTime += 1
#             if i % 10 == 0:
#                 print(f"Step {i}")
#             i += 1
#             self.clearGradients()
            
#         # Build report dictionary
#         report: Dict = {}
#         report["movements"] = self.aggregateMovements()
#         report["feedLarvae"] = self.aggregateFeedLarvae()
#         report["hungerLarvae"] = self.aggregateHungerLarvae()
#         report["hungerWasp"] = self.aggregateHungerWasp()

#         return report
#     def updateAgent(self,agent):
#         # If the agent's food is less than 1, increase its hunger by the hunger rate
#         if agent.food<1:
#             agent.hunger += agent.hungerRate
        
#         # If the agent's food minus its hunger rate is greater than 0, decrease its food by the hunger rate
#         if agent.food-agent.hungerRate>0:
#             agent.food -= agent.hungerRate
#         # Otherwise, set its food to 0
#         else:
#             agent.food = 0        
#     def stepAgent(self,agent,wasps,larvaes_position):
#         # Update the hunger cue
#         if agent.inOuterNest():
#             local_hunger_cue = agent.estimateLocalHungerCue(self.gradients[agent.role],self.grid)
#             agent.updateHungerCue(local_hunger_cue/self.gradients[agent.role].shape[0])
#         # Get the positions of all wasp agents with the role of FORAGER
#         position_foragers = [agent_.getPosition() for agent_ in wasps if agent_.role == WaspRole.FORAGER]

#         # Get the positions of all wasp agents (excluding the current agent)
#         position_wasp = [wasp.getPosition() for wasp in wasps if agent.id != wasp.id]
        
#         # If the wasp agent is a FORAGER, feel the gradient of the foraging points
#         if agent.role == WaspRole.FORAGER:
#             agent.feelGradient(self.grid,self.gradients,self.forage,larvaePositions = np.array(larvaes_position))
#             # Move the wasp agent based on the gradient
#             agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
#         # If the wasp agent is a FEEDER, feel the gradient of the larvae and wasp agents
#         elif agent.role == WaspRole.FEEDER:
#             agent.feelGradient(self.grid,self.gradients,foragersPositions=np.array(position_foragers),waspPositions = np.array(position_wasp), larvaePositions = np.array(larvaes_position))
#             # Move the wasp agent based on the gradient
#             agent.step(t=self.currentTime, agents=self.agents, forage=self.forage)  # <-- fixed
#         else:
#             agent.step(t=self.currentTime)

#     def moveAgent(self,agent):
#         wasp_agents = [agent for agent in self.agents if isinstance(agent, Wasp)]
#         agent_next_positions = np.array([[wasp.x+wasp.next_step['x'],wasp.y+wasp.next_step['y']] for wasp in wasp_agents if wasp != agent])
#         agent.move(agent_next_positions,self.grid)
        
#         # Get the current position of the wasp agent
#         current_pos = agent.getPosition()
#         # If the current position is different from the previous position, add it to the movement history
#         if self.movementHistory[agent.id][-1] != current_pos:
#             self.movementHistory[agent.id].append(current_pos)
#     def updateRoles(self,t,agent,count_roles,total_foragers,total_wasp):
#         if t % self.role_changes_frequency == 0 and count_roles <= self.max_role_changes:
#             if agent.role==WaspRole.FEEDER and agent.hungerCuesHigh() and ((total_foragers+1)/total_wasp)<(self.forager_ratio+self.potential_feeder_to_forager):
#                 if np.random.random()>0.8 and agent.rolePersistence == 0 and agent.inOuterNest():
#                     agent.role=WaspRole.FORAGER
#                     total_foragers+=1
#                     agent.updateRolePersistence()
#                     count_roles += 1
                            
#             if agent.role==WaspRole.FORAGER and agent.hungerCuesLow() and ((total_foragers-1)/total_wasp)>max((1/total_wasp),(self.forager_ratio)):
#                 if np.random.random()>0.2 and agent.rolePersistence == 0 and agent.inOuterNest():
#                     agent.role=WaspRole.FEEDER
#                     total_foragers-=1
#                     agent.updateRolePersistence()
#                     count_roles += 1
        
#         if agent.rolePersistence>0:
#             agent.rolePersistence-=1   
#         return count_roles,total_foragers
#     def clearGradients(self):

        # self.gradients[WaspRole.FEEDER]=[]
        # self.gradients[WaspRole.FORAGER]=[]

class Simulator:
    r"""
    Manage a collection of agents and coordinate the simulation.

    The simulator advances time, maintains a grid and forage sites, accumulates
    gradient fields, and aggregates results from agents' actions.

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

    def __init__(self) -> None:
        """Initialize the simulator and default state."""
        self.currentTime: int = 0
        self.agents: List[Agent] = []
        self.movementHistory: Dict[str, List[List[int]]] = {}
        self.gradients: Dict[WaspRole, List[float] | np.ndarray] = {WaspRole.FEEDER: [], WaspRole.FORAGER: []}
        self.grid: Optional[np.ndarray] = None
        self.forage: List[np.ndarray] = []
        self.forager_ratio: float = 0.10
        self.potential_feeder_to_forager: float = 0.25
        self.max_role_changes: int = 2
        self.role_changes_frequency: int = 100

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
                self.updateAgent(agent)
                if isinstance(agent, Wasp):
                    count_roles, total_foragers = self.updateRoles(
                        t, agent, count_roles, total_foragers, total_wasp
                    )
                j += 1

            self.currentTime += 1
            if i % 10 == 0:
                print(f"Step {i}")
            i += 1
            self.clearGradients()

        report: Dict[str, Dict] = {}
        report["movements"] = self.aggregateMovements()
        report["feedLarvae"] = self.aggregateFeedLarvae()
        report["hungerLarvae"] = self.aggregateHungerLarvae()
        report["hungerWasp"] = self.aggregateHungerWasp()
        return report

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
                if np.random.random() > 0.8 and agent.rolePersistence == 0 and agent.inOuterNest():
                    agent.role = WaspRole.FORAGER
                    total_foragers += 1
                    agent.updateRolePersistence()
                    count_roles += 1

            if (
                agent.role == WaspRole.FORAGER
                and agent.hungerCuesLow()
                and ((total_foragers - 1) / total_wasp) > max((1 / total_wasp), (self.forager_ratio))
            ):
                if np.random.random() > 0.2 and agent.rolePersistence == 0 and agent.inOuterNest():
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