# simulation_loop.py
# Contains the main runSimulation logic based on UML

from typing import List, Dict
from simulator import Simulator
from agents import Wasp, Larvae
from agent_decide import decide_action
from agent_step import execute_action

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
                     agent.feelGradient([])

                # NEW: Decide and take action
                action = decide_action(agent, {"agents": self.agents})
                execute_action(agent, action, self)
                # track movement only if changed
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
