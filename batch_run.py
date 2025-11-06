"""
Batch runner for multiple pathfinding modes.
Supports two run strategies:
  1. per_algorithm → run all repeats of one mode before switching
  2. per_round     → run each mode once per round, repeat for num_runs rounds
"""

import subprocess
import pandas as pd
import glob
import os
import yaml

# -------------------------------
# Load configuration
# -------------------------------
config_path = os.path.join("config", "config.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Cannot find config file at: {config_path}")

with open(config_path) as f:
    cfg = yaml.safe_load(f)

num_runs = cfg["simulation"].get("num_runs", 1)
run_mode = cfg["simulation"].get("run_mode", "per_algorithm").lower()  # new config param

# Define modes to run
MODES = ["greedy", "random_walk", "random_walk-biased", "tsp", "tsp-hamiltonian"]

sim_counter = 0  # global counter for all simulations

# -------------------------------
# Helper function
# -------------------------------
def run_simulation(mode, run_index):
    global sim_counter
    sim_counter += 1
    sim_id = sim_counter
    print(f"[RUN {run_index+1}] Mode: {mode} | Simulation ID: {sim_id}")
    subprocess.run(["python", "main.py", "--mode", mode, "--sim-id", str(sim_id), "--append-logs"], check=True)

# -------------------------------
# Run strategy
# -------------------------------
if run_mode == "per_algorithm":
    print(f"\n[INFO] Running {num_runs} times per mode (grouped by algorithm)...\n")
    for mode in MODES:
        for run in range(num_runs):
            simulation_id = run + 1  # Each mode has its own independent ID sequence
            print(f"[RUN] Mode: {mode} | Simulation ID: {simulation_id}")
            subprocess.run([
                "python", "main.py",
                "--mode", mode,
                "--sim-id", str(simulation_id),
                "--append-logs"
            ], check=True)

elif run_mode == "per_round":
    print(f"\n[INFO] Running {num_runs} rounds (each round covers all modes)...\n")
    for run in range(num_runs):
        simulation_id = run + 1  # All modes in this round share the same simulation ID
        print(f"\n=== Round {simulation_id}/{num_runs} ===")
        for mode in MODES:
            print(f"[RUN] Mode: {mode} | Simulation ID: {simulation_id}")
            subprocess.run([
                "python", "main.py",
                "--mode", mode,
                "--sim-id", str(simulation_id),
                "--append-logs"
            ], check=True)

else:
    print(f"[ERROR] Unknown run_mode: {run_mode}")
    exit(1)


# -------------------------------
# Combine aggregate CSVs
# -------------------------------
output_dir = "output_logs"
aggregate_files = sorted(glob.glob(os.path.join(output_dir, "*", "aggregate_results_*.csv")))

if not aggregate_files:
    print("[WARN] No aggregate result files found.")
else:
    combined_path = os.path.join(output_dir, "aggregate_combined.csv")
    combined_df = pd.concat([pd.read_csv(f) for f in aggregate_files], ignore_index=True)
    combined_df.to_csv(combined_path, index=False)

    print(f"\n[INFO] Combined all aggregate results into: {combined_path}")
    print(f"  Total runs combined: {len(aggregate_files)}")

    # STEP 1: Compute mean and standard deviation for each pathfinding mode
    mode_col = "pathfinding_mode" if "pathfinding_mode" in combined_df.columns else "__mode__"
    numeric_cols = combined_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Remove simulation_id since it’s not a metric to average
    if "simulation_id" in numeric_cols:
        numeric_cols.remove("simulation_id")
    combined_df.drop(columns=["simulation_id"], inplace=True, errors="ignore")

    mean_df = combined_df.groupby(mode_col)[numeric_cols].mean().reset_index()
    std_df  = combined_df.groupby(mode_col)[numeric_cols].std().reset_index()

    summary_df = mean_df.merge(std_df, on=mode_col, suffixes=("_mean", "_std"))

    # STEP 2: Save summary (mean/std) to a new file
    summary_path = os.path.join(output_dir, "aggregate_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[INFO] Saved per-mode summary (mean/std) → {summary_path}")

