# main.py
# Entry point for running the agent simulation

from simulator import instanceGenerator, Simulator
from logger import DataLogger
import yaml
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Agent Simulation Runner")
    parser.add_argument("--mode", type=str, required=True,
                        help="Pathfinding mode (greedy, pheromone, random, etc.)")
    parser.add_argument("--sim-id", type=str, default=None,
                        help="Unique simulation ID for batch runs")
    parser.add_argument("--append-logs", action="store_true")
    return parser.parse_args()

def main():

    cli_args = parse_args()

    # Load config
    config_path = os.path.join("config", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    # Override mode from CLI
    args["simulator"]["pathfinding_mode"] = cli_args.mode

    # Generate simulator  
    generator = instanceGenerator(**args['instance_generator'])
    generator.waspDictionary(args['wasp'])
    generator.larvaeDictionary(args['larvae'])
    generator.simulatorDictionary(args['simulator'])
    
    simulator = generator.generateSimulator()

    # Attach config to simulator (so simulator.py can log it safely)
    simulator.config = args

    #  Replace simulator.logger (the old one created in Simulator.__init__)
    simulator.logger = DataLogger(
        pathfinding_mode=cli_args.mode,
        simulation_id=cli_args.sim_id,
        reset_logs=not cli_args.append_logs
    )

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
