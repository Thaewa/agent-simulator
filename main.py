# main.py
# Entry point of the simulation project

from agents import Agent
from simulator import Simulator
from simulation_loop import run_simulation

if __name__ == "__main__":
    # Example: create a few agents
    agents = [Agent("A1"), Agent("A2")]

    # Example: simple environment (can be replaced with real data)
    environment = {"type": "grid", "size": (10, 10)}

    # Create the simulator
    sim = Simulator(agents, environment)

    # Run the simulation
    run_simulation(sim, steps=5)
