"""Run small seeded simulations in isolation: python smoke_test.py."""

import contextlib
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
from unittest.mock import patch

import numpy as np
import pandas as pd

from agents import Larvae, Wasp, WaspRole
from logger import DataLogger, validate_food_event_log
from pathfinding import PATHFINDING_MODES
from simulator import Simulator


ROOT = Path(__file__).resolve().parent
SEED = 20261108
STEPS = 20


def fingerprints(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def scenario(mode, output):
    sim = Simulator()
    X, Y = np.meshgrid(np.arange(-6, 6.5, 0.5), np.arange(-6, 6.5, 0.5))
    sim.grid = np.column_stack((X.ravel(), Y.ravel()))
    for name, pos, role, food in (
        ("F1", (-0.5, -0.5), WaspRole.FEEDER, 6),
        ("F2", (0.5, -0.5), WaspRole.FEEDER, 6),
        ("R1", (3, 0), WaspRole.FORAGER, 1),
    ):
        wasp = Wasp(name, outer_nest_radius=4, x=pos[0], y=pos[1],
                    path_finding=mode, food=food, hunger=1)
        wasp.role = role
        sim.addAgent(wasp)
    for i, pos in enumerate(((-1, 0.5), (0.5, 0.5), (1, 1), (-0.5, 1.5))):
        sim.addAgent(Larvae(f"L{i}", x=pos[0], y=pos[1], hunger=2, food=0))
    sim.addForage(3.5, 0)
    sim.config = {"simulator": {"pathfinding_mode": mode}}
    sim.logger = DataLogger(mode, simulation_id=1, output_dir=output)
    return sim


def run_case(mode, output):
    random.seed(SEED)
    np.random.seed(SEED)
    sim = scenario(mode, output)
    stats = {"route_requests": 0, "routes_built": 0,
             "closed_routes_built": 0, "planned_moves_followed": 0,
             "planned_moves_altered": 0}
    planned = {}
    estimate = Wasp.estimate_path
    choose = Wasp.shortest_path_finding
    move = Simulator.moveAgent
    solver = __import__("networkx").approximation.traveling_salesman_problem

    def record_solver(*args, **kwargs):
        route = solver(*args, **kwargs)
        if kwargs["cycle"]:
            assert route[0] == route[-1], "Closed solver tour is not closed"
            stats["closed_routes_built"] += 1
        return route

    def record_estimate(wasp, *args, **kwargs):
        route = estimate(wasp, *args, **kwargs)
        if any(node != tuple(wasp.getPosition()) for node in route):
            stats["routes_built"] += 1
        return route

    def record_choice(wasp, *args, **kwargs):
        stats["route_requests"] += 1
        cached = bool(wasp.path) and max(
            abs(wasp.path[0][0] - wasp.x), abs(wasp.path[0][1] - wasp.y)
        ) <= 0.5
        before = stats["routes_built"]
        result = choose(wasp, *args, **kwargs)
        if (cached or stats["routes_built"] > before) and any(wasp.next_step.values()):
            planned[wasp.id] = (wasp.x + wasp.next_step["x"],
                                wasp.y + wasp.next_step["y"])
        return result

    def record_move(current_sim, wasp):
        result = move(current_sim, wasp)
        positions = [tuple(a.getPosition()) for a in current_sim.agents if isinstance(a, Wasp)]
        assert len(positions) == len(set(positions)), "Adults share a cell"
        assert np.any(np.all(current_sim.grid == wasp.getPosition(), axis=1)), "Adult left grid"
        if wasp.role == WaspRole.FEEDER:
            assert wasp.x**2 + wasp.y**2 <= wasp.outerNestRadius**2
        if wasp.id in planned:
            matched = tuple(wasp.getPosition()) == planned.pop(wasp.id)
            stats["planned_moves_followed" if matched else "planned_moves_altered"] += 1
        values = np.array([wasp.x, wasp.y, wasp.food, wasp.hunger], dtype=float)
        assert np.isfinite(values).all(), "Non-finite wasp state"
        assert wasp.food >= 0, "Negative food"
        assert wasp.food <= wasp.maxFood + 1e-9, "Food exceeds wasp capacity"
        return result

    with patch.object(Wasp, "estimate_path", record_estimate), \
         patch.object(Wasp, "shortest_path_finding", record_choice), \
         patch.object(Simulator, "moveAgent", record_move), \
         patch("agents.nx.approximation.traveling_salesman_problem", record_solver):
        sim.runSimulation(STEPS)
    assert sim.currentTime == STEPS
    for agent in sim.agents:
        assert np.isfinite([agent.x, agent.y, agent.food, agent.hunger]).all()
        assert agent.food >= 0
        if isinstance(agent, Wasp):
            assert agent.food <= agent.maxFood + 1e-9
    stats["larvae_feeds"] = sum(getattr(a, "feed_count_larvae", 0) for a in sim.agents)
    if mode.startswith("tsp"):
        assert stats["routes_built"] > 0, "TSP was never exercised"
        assert stats["planned_moves_followed"] > 0, "No planned step was followed"
    else:
        assert stats["route_requests"] == 0
    if mode == "tsp-hamiltonian":
        assert stats["closed_routes_built"] > 0
    # Compare state and food-event logs, excluding aggregate wall-clock timestamps.
    stats["state_log_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((output / mode).glob("*_log_*.csv"))
    }
    nest = pd.read_csv(output / mode / f"nest_log_{mode}.csv")
    stats["duplicate_timestamps"] = int(nest.timestamp.duplicated().sum())
    assert nest.timestamp.tolist() == list(range(STEPS + 1))
    assert stats["duplicate_timestamps"] == 0
    validate_food_event_log(sim.logger)
    aggregate = pd.read_csv(output / mode / f"aggregate_results_{mode}.csv").iloc[0]
    for column, total in (("feeding_events", "total_larval_feeding_events"),
                          ("transfer_events", "total_adult_transfer_events"),
                          ("foraging_events", "total_foraging_events")):
        assert nest[column].sum() == aggregate[total]
    larvae = pd.read_csv(output / mode / f"larvae_log_{mode}.csv")
    for _, rows in larvae.groupby("larva_id"):
        assert rows.timestamp.tolist() == list(range(STEPS + 1))
    agent_log = pd.read_csv(output / mode / f"agent_log_{mode}.csv")
    for agent in (a for a in sim.agents if isinstance(a, Wasp)):
        rows = agent_log[agent_log.agent_id == agent.id]
        assert rows.timestamp.tolist() == list(range(STEPS + 1))
        assert rows.food_stored.between(0, agent.maxFood + 1e-9).all()
    return stats


def main():
    baseline_before = fingerprints(ROOT / "output_logs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = ROOT / "exports" / "smoke_checks" / stamp
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for mode in PATHFINDING_MODES:
        repeats = []
        for repeat in (1, 2):
            destination = output / f"repeat_{repeat}"
            with (output / f"{mode}_repeat_{repeat}.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log):
                    repeats.append(run_case(mode, destination))
        assert repeats[0] == repeats[1], f"Seeded replay differs for {mode}"
        results[mode] = repeats[0]
        print(f"PASS {mode}: routes={repeats[0]['routes_built']}, "
              f"planned moves followed={repeats[0]['planned_moves_followed']}, "
              f"larvae feeds={repeats[0]['larvae_feeds']}, replay identical")
    # A stochastic mode can legitimately have no feeds in twenty steps.
    # Require feeding to be exercised across the suite, not by every mode.
    assert any(case["larvae_feeds"] > 0 for case in results.values())
    assert fingerprints(ROOT / "output_logs") == baseline_before, "Baseline logs changed"
    manifest = {
        "purpose": "Small hand-built integration check, not a performance experiment",
        "seed": SEED, "steps_per_run": STEPS, "repeats_per_mode": 2,
        "scenario": "2 feeders, 1 forager, 4 hungry larvae, one forage site; see smoke_test.py",
        "results": results,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("agents.py", "simulator.py", "logger.py", "utils.py",
                                       "pathfinding.py", "metrics.py", "smoke_test.py")},
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "networkx", "pandas")},
        "known_limitations": ["Movement is sequential with adult-cell exclusion; diagonal segment crossings are allowed",
                              "Biological bouts are unavailable; the legacy bout count is a food-threshold proxy",
                              "Planned steps can be altered by collision handling",
                              "No performance or biological validation claims"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Baseline logs unchanged. All state-log timestamps are unique per agent/run: 0..20.")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
