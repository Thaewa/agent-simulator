# Social Wasp Feeding Simulator

A Python agent-based simulator for studying food distribution among foragers, feeders and stationary larvae. Five hand-crafted movement policies provide a computational baseline for future comparisons with field observations and an RL optimizer comparator.

The actively validated work is on **`fix/tsp-validation`**. RL training and empirical validation are not implemented. Current experiments use synthetic nest geometry and uncalibrated model parameters.

## Validated checkpoint

The baseline shared with the advisor is [commit `3882633`](https://github.com/Thaewa/agent-simulator/commit/3882633a168637e6e460a2a320dcb776c1ea101e). That commit remains the source reference for the completed 500-step run; later README changes do not replace or modify it.

| Check | Checkpoint result |
| --- | --- |
| Unit/integration tests | 118 passed |
| Seeded smoke check | All five modes passed, with identical repeated logs |
| Full generated-nest run | Five modes, 500 steps each, seed `20261108` |
| Initial population | 30 adults and 29 larvae |
| Plotting | 19 PNG figures generated |
| Food-event journal | Counts reconciled with per-step nest counts and aggregate totals |

Full-run output directory, relative to the repository:

```text
exports/experiments/seed_20261108_20261108T081742_362756Z/
```

Generated results are retained locally and excluded from Git. A fresh clone will not contain this directory. This is a software-validation checkpoint, not evidence of biological validity or a general policy ranking.

## Setup

Python 3.12 was used for the validated checkpoint. From a Python environment with pip:

```powershell
git clone --branch fix/tsp-validation https://github.com/Thaewa/agent-simulator.git
cd agent-simulator
python -m pip install -r requirements.txt networkx PyYAML
```

`networkx` and `PyYAML` are required by the simulator but are not currently listed in `requirements.txt`; the command installs them explicitly. Existing checkouts can run the commands below from the repository directory.

## Validate and run

Run the regression suite:

```powershell
python -B -m unittest discover -s tests
```

Run the short five-mode integration check:

```powershell
python -B smoke_test.py
```

The smoke check runs a hand-built scene for 20 steps twice per mode, records actual route execution, compares repeated logs and verifies historical baseline logs remain unchanged. Results go into a new directory under `exports/smoke_checks/`.

Run all five policies and generate plots automatically:

```powershell
python -B run_experiment.py --steps 500 --seed 20261108
```

For a shorter generated-nest check:

```powershell
python -B run_experiment.py --steps 50 --seed 20261108
```

No separate plotting command is needed. Short runs may not exercise multi-target routing or scheduled role changes and are not substitutes for the route tests. To reproduce the historical checkpoint, use source revision `3882633`, its configuration and the dependency versions recorded in its manifest; rerunning the current branch does not automatically select that revision.

## Output isolation and provenance

Every experiment invocation creates a new timestamped directory. Old results are preserved; plots read only that invocation's logs. Each policy starts from a deep copy of the same generated instance, with Python and NumPy RNGs reset to the supplied seed.

```text
exports/experiments/seed_<seed>_<UTC timestamp>/
    manifest.json             # Status, source revision/hashes, seed, dependencies, metrics
    config.json               # Saved configuration
    initial_instance.json     # Shared starting instance
    source/                   # Source and configuration snapshot
    output_logs/<mode>/       # State, food-event and aggregate CSVs
    aggregate_combined.csv    # Combined per-mode aggregates
    output_plots_v3_1/        # Generated PNG figures
    plotting.log              # Plotting output
```

Use the directory printed after `Results:` and `Done. Plots:`. A completed run has `status: "complete"` in its manifest. Do not treat partial/failed runs as completed comparisons.

The food-event journal records successful collections, adult transfers and larval feeds with actual quantities and event-time identities, positions and roles. It does not reconstruct biological feeding bouts. State timestamps run from initialization at zero through the completed step horizon; multiple event rows within one step are intentional.

## Movement policies

| Mode | Behavior |
| --- | --- |
| `greedy` | Local gradient-guided movement |
| `random_walk` | Random local movement subject to movement constraints |
| `random_walk-biased` | Random movement with rejection of immediate backtracking |
| `tsp` | Eligible feeders plan approximate open routes over nearby targets |
| `tsp-hamiltonian` | Eligible feeders plan approximate closed target tours |

TSP target selection uses nearby positions and available food; it does not directly minimize colony hunger. Collision handling can alter planned moves. An open or closed target tour is not a guarantee of a globally optimal full route from the adult's origin.

`routes built` counts nontrivial constructed paths, including single-target shortest paths. It is not a count of multi-target TSP solver calls. Zero routes for the three non-TSP modes is expected. Real solver calls, open/closed tours and route following are checked in [tests/test_tsp_routes.py](tests/test_tsp_routes.py).

## Interpretation and remaining work

- Current parameters are in model units. Larval rates of 0.3/0.6/0.9 per step are now applied as configured, but are not calibrated to physical time or measured demand. Hunger remains high in the full checkpoint.
- One synthetic nest and seed cannot establish algorithm superiority. Feeding timing, resource supply, target coverage and role allocation affect hunger in addition to route length. Random walk is not required to have the highest hunger.
- Several corrections changed dynamics and initialization. Old and current graphs do not isolate the original TSP case-mismatch effect; equal seeds across changed model versions do not guarantee identical initial maps or random draws.
- The retained bout counter is explicitly labeled `legacy_food_threshold_proxy_v1`; biological bout reconstruction is unavailable. A transfer event or collection is not automatically a complete biological bout.
- The next research phase requires agreement on the biological comparison, available real nest maps, field endpoints and calibration. RL is intended as an optimizer comparator, not an assumed mechanism of wasp behavior.

## Code and technical notes

| File | Purpose |
| --- | --- |
| `agents.py` | Adult/larval behavior, food transfers and local route planning |
| `simulator.py` | Instance generation, simulation loop, movement and role updates |
| `pathfinding.py` | Canonical policy names and validation |
| `logger.py` / `metrics.py` | State/event journals, aggregate metrics and reconciliation |
| `run_experiment.py` | Seeded five-mode experiments, snapshots and automatic plotting |
| `smoke_test.py` | Isolated repeated integration checks |
| `plot_v3_results_v3_1.py` | Schema-aware plots |
| `config/config.yaml` | Model configuration; see active/inactive-field audit |
| `tests/` | Regression and integration tests |

Further documentation:

- [Correction and validation history](docs/bugfix_progress.md)
- [Metric definitions](docs/metrics.md)
- [Movement assumptions](docs/movement.md)
- [Effective configuration](docs/configuration.md)
- [Food-event schema](docs/food_event_log.md)
- [Bout definitions and limits](docs/bout_definition.md)
- [Research alignment](docs/research_alignment.md)
- [Local empirical-data inventory](docs/empirical_data_inventory.md)
- [Field measurement versus current logs](docs/field_measurement_audit.md)

The legacy `main.py` and batch entry points use different output/reset conventions. Use `run_experiment.py` for the isolated checkpoint workflow described here. Manually running the plot script uses the working directory's `output_logs/` and overwrites matching figure names; it does not clear unrelated old figures first.
