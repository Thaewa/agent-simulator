# simulator.py
# Defines the Simulator class that manages agents and the environment

class Simulator:
    def __init__(self, agents, environment):
        # List of all agents in the simulation
        self.agents = agents
        # Representation of the environment (grid, graph, etc.)
        self.environment = environment

    def run_step(self):
        """
        Run a single step of the simulation.
        Calls each agent's step() method.
        """
        for agent in self.agents:
            agent.step(self.environment)
