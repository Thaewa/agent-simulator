# simulator.py
# Defines the Simulator class that manages agents and runs the simulation

from typing import List, Dict
from agents import Agent, Wasp, Larvae
from agents import AgentType, WaspRole
import numpy as np
from utils import gaussian_attraction

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
        """
        self.currentTime: int = 0
        self.agents: List[Agent] = []
        self.movementHistory: Dict[str, List[List[int]]] = {}
        self.gradients = {WaspRole.FEEDER:[],WaspRole.FORAGER:[]}
        self.grid = None
        self.forage = []
    # ---------------------------
    # Core methods
    # ---------------------------

    def step(self) -> None:
        """
        Advance the simulation by one time unit.
        Calls step(t) on each agent.
        """
        for agent in self.agents:
            agent.step(self.currentTime)
        self.currentTime += 1

    def addAgent(self, agent: Agent) -> None:
        """Add a new agent to the simulation."""
        self.agents.append(agent)
        
        # initialize movement history with starting position
        self.movementHistory[agent.id] = [agent.getPosition()]

    def addForage(self, x: int, y: int) -> None:
        """
        Add a foraging location (placeholder).
        UML shows this, but details depend on environment model.
        """
        self.forage.append(np.array(x,y))
        

    def removeAgent(self, agent: Agent) -> None:
        """Remove an agent from the simulation."""
        if agent in self.agents:
            self.agents.remove(agent)

    def accumulateGradients(self) -> None:
        
        agents = self.agents
        for agent in agents:
            x0,y0 = agent.getPosition()
            spread = agent.radius
            peak = agent.hunger/agent.food
            gradient = gaussian_attraction(self.grid[:,0],self.grid[:,1],x0,y0,spread,peak)
            type = agent.type
            if type == AgentType.WASP:
                role = agent.role
            if type == AgentType.LARVAE:
                self.gradients[WaspRole.FEEDER]=self.gradients[WaspRole.FEEDER]+gradient
                self.gradients[WaspRole.FORAGER]=self.gradients[WaspRole.FORAGER]+gradient
            if type == AgentType.WASP:
                if role == WaspRole.FORAGER:
                    self.gradients[WaspRole.FEEDER]= self.gradients[WaspRole.FEEDER]+gradient
                else:
                    # Modify if feeders also attract forager but for now only foragers attract feeders
                    # self.gradients[WaspRole.FORAGER]= self.gradients[WaspRole.FORAGER]+gradient
                    pass

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

    def runSimulation(self, t: int) -> List[Dict]:
        """
        Run the simulation for t steps.
        Returns a list of snapshots (dictionary per step).
        """
        results: List[Dict] = []
        for _ in range(t):
            self.step()
            snapshot = {
                "time": self.currentTime,
                "movements": self.aggregateMovements(),
                "hungerLarvae": self.aggregateHungerLarvae(),
                "hungerWasp": self.aggregateHungerWasp(),
            }
            results.append(snapshot)
        return results
