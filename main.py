# main.py
# Entry point for running the agent simulation

from simulator import instanceGenerator

def main():
    generator = instanceGenerator()
    simulator = generator.generateSimulationInstance("TSP-Hamiltonian")
    # # Run simulation for T steps
    T = 1000
    report = simulator.runSimulation(T)
    
    # Print results
    print("Simulation finished.")
    print("Report summary:")
    
    # Iterate through the dictionary (movements, hunger, etc.)
    for key, value in report.items():
        print(f"{key}: {value}")

    # Debug logs
    print("\nEvent logs:")
    for agent in simulator.agents:
        print(f"{agent.id} ({agent.type.value}):")
        for e in agent.storedEvents:
            print("  ", e)

if __name__ == "__main__":
    main()
