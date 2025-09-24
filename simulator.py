# simulator.py
# Defines the Simulator class that manages agents and runs the simulation

from typing import List, Dict
from agents import Agent, Wasp, Larvae

class Simulator:
    def __init__(self):
        self.currentTime: int = 0
        self.agents: List[Agent] = []

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

    def addForage(self, x: int, y: int) -> None:
        """
        Add a foraging location (placeholder).
        UML shows this, but details depend on environment model.
        """
        # TODO: implement environment/foraging logic
        pass

    def removeAgent(self, agent: Agent) -> None:
        """Remove an agent from the simulation."""
        if agent in self.agents:
            self.agents.remove(agent)

    def accumulateGradients(self) -> None:
        """Placeholder: accumulate gradient fields (not specified in UML)."""
        pass

    def aggregateMovements(self) -> Dict[int, List[tuple[int, int]]]:
        """
        Collect movement data for all agents.
        Returns {agent_id: [(x, y), ...]}
        """
        return {a.id: [a.getPosition()] for a in self.agents}

    def aggregateFeedLarvae(self) -> Dict[int, List[int]]:
        """
        Collect feeding data.
        Placeholder: return empty structure.
        """
        return {a.id: [] for a in self.agents if isinstance(a, Wasp)}

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
