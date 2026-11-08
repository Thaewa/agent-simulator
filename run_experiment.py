"""Seeded five-mode experiment with isolated outputs: python run_experiment.py."""

import argparse
import contextlib
import copy
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from agents import Wasp, WaspRole
from logger import DataLogger
from pathfinding import PATHFINDING_MODES
from simulator import Simulator, instanceGenerator


ROOT = Path(__file__).resolve().parent


def json_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {str(key.value if isinstance(key, Enum) else key): safe(val)
                    for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(val) for val in item]
        if isinstance(item, (Enum, np.ndarray, np.generic)):
            return safe(json_value(item))
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_instance(config, seed):
    random.seed(seed)
    np.random.seed(seed)
    generator = instanceGenerator(**config["instance_generator"])
    generator.waspDictionary(config["wasp"])
    generator.larvaeDictionary(config["larvae"])
    generator.simulatorDictionary({**config["simulator"], "pathfinding_mode": "greedy"})
    return generator.generateSimulator()


def snapshot(simulator):
    return {"grid": simulator.grid, "forage": simulator.forage,
            "agents": [agent.__dict__ for agent in simulator.agents],
            "simulator": {key: value for key, value in simulator.__dict__.items()
                          if key not in {"agents", "grid", "forage", "logger"}}}


def run_mode(base, config, mode, seed, steps, output, console):
    simulator = copy.deepcopy(base)
    grid_cells = set(map(tuple, simulator.grid))
    for agent in simulator.agents:
        if isinstance(agent, Wasp):
            agent.path_finding = mode
    simulator.config = copy.deepcopy(config)
    simulator.config["simulator"]["pathfinding_mode"] = mode
    simulator.config["experiment"] = {"seed": seed, "steps": steps}
    # All modes receive identical initial agents and maps. Reset both random
    # streams immediately before dynamics; policies can consume draws differently.
    random.seed(seed)
    np.random.seed(seed)
    counters = {"route_requests": 0, "routes_built": 0}
    choose, estimate, clear = Wasp.shortest_path_finding, Wasp.estimate_path, Simulator.clearGradients

    def record_choose(wasp, *args, **kwargs):
        counters["route_requests"] += 1
        return choose(wasp, *args, **kwargs)

    def record_estimate(wasp, *args, **kwargs):
        route = estimate(wasp, *args, **kwargs)
        if any(node != tuple(wasp.getPosition()) for node in route):
            counters["routes_built"] += 1
        return route

    def progress(current):
        result = clear(current)
        adults = [a for a in current.agents if isinstance(a, Wasp)]
        positions = [tuple(a.getPosition()) for a in adults]
        assert len(positions) == len(set(positions)), "Adults share a cell"
        assert all(position in grid_cells for position in positions), "Adult left grid"
        assert all(a.x**2 + a.y**2 <= a.outerNestRadius**2
                   for a in adults if a.role == WaspRole.FEEDER), "Feeder left nest"
        if current.currentTime % 50 == 0 or current.currentTime == steps:
            print(f"  {mode}: {current.currentTime}/{steps} steps", file=console, flush=True)
        return result

    with (output / f"{mode}.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), \
             patch.object(Wasp, "shortest_path_finding", record_choose), \
             patch.object(Wasp, "estimate_path", record_estimate), \
             patch.object(Simulator, "clearGradients", progress):
            simulator.logger = DataLogger(mode, simulation_id=1, output_dir=output / "output_logs")
            simulator.runSimulation(steps)
    assert simulator.currentTime == steps
    nest = pd.read_csv(output / "output_logs" / mode / f"nest_log_{mode}.csv")
    assert nest.timestamp.tolist() == list(range(steps + 1)), "Invalid nest timestamps"
    for agent in simulator.agents:
        assert np.isfinite([agent.x, agent.y, agent.food, agent.hunger]).all()
        assert agent.food >= 0
        if isinstance(agent, Wasp):
            assert agent.food <= agent.maxFood + 1e-9
    # Check all recorded states, not only the final state.
    agents = pd.read_csv(output / "output_logs" / mode / f"agent_log_{mode}.csv")
    for wasp in (a for a in simulator.agents if isinstance(a, Wasp)):
        rows = agents[agents.agent_id == wasp.id]
        assert rows.food_stored.between(0, wasp.maxFood + 1e-9).all(), f"Food outside capacity for {wasp.id}"
    aggregate = pd.read_csv(output / "output_logs" / mode / f"aggregate_results_{mode}.csv")
    for column, total in (("feeding_events", "total_larval_feeding_events"),
                          ("transfer_events", "total_adult_transfer_events"),
                          ("foraging_events", "total_foraging_events")):
        assert nest[column].sum() == aggregate.iloc[0][total], f"Event totals disagree: {column}"
    return {**counters, **aggregate.iloc[0].to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32 or args.steps <= 0:
        parser.error("Seed must be 0..2**32-1; steps must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "exports" / "experiments" / f"seed_{args.seed}_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    write_json(output / "config.json", config)
    source = output / "source"
    source.mkdir()
    for path in sorted(ROOT.glob("*.py")):
        shutil.copy2(path, source / path.name)
    shutil.copytree(ROOT / "config", source / "config")
    source_hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(source.rglob("*")) if p.is_file()}
    with (output / "initialization.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            base = prepare_instance(config, args.seed)
    # JSON is portable and records all generated instance state for later replay.
    write_json(output / "initial_instance.json", snapshot(base))
    manifest = {
        "status": "running", "seed": args.seed, "steps": args.steps,
        "modes": list(PATHFINDING_MODES), "runs_per_mode": 1,
        "initialization": "deep copies of the same generated instance; both RNGs reset per mode",
        "movement_model": "sequential_exclusive_adult_cells_v1",
        "nest_regions": "inner_boundary_inclusive_outer_strictly_exterior",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "source_sha256": source_hashes, "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "networkx", "pandas", "matplotlib", "PyYAML")},
        "limitations": ["One seed: exploratory run, not a statistical comparison",
                        "Feeding, collision, role scheduling and metric semantics still need review",
                        "Matching seeds do not guarantee identical draws across different policies"],
        "results": {},
    }
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output}", flush=True)
    try:
        for mode in PATHFINDING_MODES:
            print(f"Running {mode}...", flush=True)
            manifest["results"][mode] = run_mode(base, config, mode, args.seed, args.steps, output, sys.stdout)
            routes = manifest["results"][mode]["routes_built"]
            print(f"  Finished {mode}: {routes} routes built", flush=True)
            if mode.startswith("tsp") and routes == 0:
                print("  WARNING: No TSP route built; this run does not exercise routing.", flush=True)
            write_json(output / "manifest.json", manifest)
        frames = [pd.read_csv(p) for p in sorted((output / "output_logs").glob("*/aggregate_results_*.csv"))]
        pd.concat(frames, ignore_index=True).to_csv(output / "aggregate_combined.csv", index=False)
        print("Creating plots...", flush=True)
        with (output / "plotting.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(source / "plot_v3_results_v3_1.py")],
                           cwd=output, env={**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    print(f"Done. Plots: {output / 'output_plots_v3_1'}", flush=True)


if __name__ == "__main__":
    main()
