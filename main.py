# main.py
# Entry point for running the agent simulation

# Import agent classes (Wasp, Larvae, and WaspRole enum)
from agents import Wasp, Larvae, WaspRole
# Import SimulationLoop which extends Simulator
from simulation_loop import SimulationLoop


def main():
    # Create the simulator (controller of the whole simulation process)
    sim = SimulationLoop()

    # Create agents (example initialization)
    
    # Wasp W1 at position (0,0), role=Feeder, starts with 3 food
    w1 = Wasp(agent_id="W1", x=0, y=0, role=WaspRole.FEEDER, food=3)
    
    # Wasp W2 at position (2,2), role=Forager, starts with 1 food
    w2 = Wasp(agent_id="W2", x=2, y=2, role=WaspRole.FORAGER, food=1)
    
    # Larvae L1 at position (5,5)
    l1 = Larvae(agent_id="L1", x=5, y=5)
    
    # Larvae L2 at position (6,6)
    l2 = Larvae(agent_id="L2", x=6, y=6)

    # Add all agents to the simulator
    sim.addAgent(w1)
    sim.addAgent(w2)
    sim.addAgent(l1)
    sim.addAgent(l2)

    # Run simulation for T steps
    T = 5
    report = sim.runSimulation(T)

    # Print results
    print("Simulation finished.")
    print("Report summary:")
    
    # Iterate through the dictionary (movements, hunger, etc.)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
