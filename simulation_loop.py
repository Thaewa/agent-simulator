# simulation_loop.py
# Implements the main simulation loop logic

def run_simulation(simulator, steps=10):
    """
    Run the simulation for a given number of steps.
    """
    for step in range(steps):
        print(f"Step {step+1}")
        simulator.run_step()
