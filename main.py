# main.py
# Entry point for running the agent simulation

from agents import Wasp, Larvae, WaspRole
from simulation_loop import SimulationLoop


def main():
    # Create simulator (with loop behavior)
    sim = SimulationLoop()

    # Add agents (example initialization)
    w1 = Wasp(agent_id="W1", x=0, y=0, role=WaspRole.FEEDER, food=3)
    w2 = Wasp(agent_id="W2", x=2, y=2, role=WaspRole.FORAGER, food=1)
    l1 = Larvae(agent_id="L1", x=5, y=5)
    l2 = Larvae(agent_id="L2", x=6, y=6)

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
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
