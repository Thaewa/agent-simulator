# simulation_loop.py
# Contains the main runSimulation logic based on UML

from typing import List, Dict
from simulator import Simulator
from agents import Wasp, Larvae

class SimulationError(Exception):
    """Custom exception for simulation errors."""
    pass


class SimulationLoop(Simulator):
    def verifySimulationConditions(self) -> bool:
        """
        Verify if the simulation can start.
        Placeholder: always True for now.
        """
        return True

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
                    agent.feelGradient([])  # attractorGradients not defined yet
                # Step simulation for this agent
                agent.step(self.currentTime)
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
