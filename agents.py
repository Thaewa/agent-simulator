# agents.py
# Defines the Agent class and its basic properties

class Agent:
    def __init__(self, name):
        # Unique identifier for the agent
        self.name = name
        # Internal state or memory of the agent
        self.state = {}

    def decide_action(self, environment):
        """
        Decide what action to take based on the current environment.
        This will call functions defined in agent_decide.py.
        """
        pass

    def step(self, environment):
        """
        Perform one step of simulation.
        This will call functions defined in agent_step.py.
        """
        pass
