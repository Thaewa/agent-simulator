# simulation_loop.py
# Contains the main runSimulation logic based on UML

from typing import List, Dict
from simulator import Simulator
from agents import Wasp, Larvae, WaspRole
from agent_decide import decide_action
from agent_step import execute_action
import numpy as np

class SimulationError(Exception):
    """Custom exception for simulation errors."""
    pass


class SimulationLoop(Simulator):
    """
    Extends the base Simulator with a detailed runSimulation loop
    defined by the UML activity diagram.
    
    Responsibilities:
        - Verify if simulation conditions are valid before running.
        - Execute simulation steps across multiple time units.
        - Handle agent interactions (e.g., wasps feeling gradients).
        - Produce a report that aggregates movements, feeding,
          and hunger data after simulation.
    """
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

        Attributes:
            grid (numpy.ndarray): A 2D NumPy array representing the grid.
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
        return self.grid.shape[0]>0 and self.grid.shape[1]==2

    def verifySimulationConditions(self) -> bool:
        """
        Verify if the simulation can start.
        """
        num_agents = self.verifyNumberAgents()
        self.createGrid()
        grid_verification = self.verifyGrid()
        
        return num_agents and grid_verification

    def runSimulation(self, t: int) -> List[Dict]:
        """
        Run the simulation for t steps.
        Based directly on UML activity diagram.
        """
        if not self.verifySimulationConditions():
            raise SimulationError("Simulation conditions not met")

        i = 0
        while i < t:
            # Accumulate gradients (placeholder)
            self.accumulateGradients()
            j = 0
            while j < len(self.agents):
                agent = self.agents[j]

                # UML: agent feels gradient (placeholder: pass None)
                if isinstance(agent, Wasp):
                    if agent.role == WaspRole.FORAGER:
                        agent.feelGradient(self.grid,self.gradients,self.forage)
                    elif agent.role == WaspRole.FEEDER:
                        agent.feelGradient(self.grid,self.gradients)
                
                    agent.step(self.agents,self.forage)

                    current_pos = agent.getPosition()
                    if self.movementHistory[agent.id][-1] != current_pos:
                        self.movementHistory[agent.id].append(current_pos)
                    j += 1

            # Advance time
            self.currentTime += 1
            i += 1

        # Build report dictionary
        report: Dict = {}
        report["movements"] = self.aggregateMovements()
        report["feedLarvae"] = self.aggregateFeedLarvae()
        report["hungerLarvae"] = self.aggregateHungerLarvae()
        report["hungerWasp"] = self.aggregateHungerWasp()

        return report
