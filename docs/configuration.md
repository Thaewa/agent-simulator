# Effective configuration audit

## Metabolism correction

Larval `hungerRate` now equals `larvae.hunger_rate * sampled_multiplier`. The previous constructor multiplied the inherited base-class default 0.01 instead of the supplied base rate. The factory also repeated L2 in its sampling choices; it now samples uniformly from `[1, L2, L3]`.

With the unchanged current YAML values, possible larval rates are **0.3, 0.6 and 0.9 per step**, whereas previous runs effectively used 0.01 and 0.02. These are rates in model units, not calibrated biological measurements.

`Simulator.updateAgent` uses this effective rate for food depletion each step and hunger increase when pre-update food is below 1. This checkpoint fixes assignment and sampling; it does not change that metabolism rule. Zero rates/multipliers are allowed; negative or non-finite values are rejected. The effective product must also be finite.

Wasp metabolism already used its configured rate and continues to do so, with added finite/nonnegative validation.

## Active and inactive fields

| Configuration | Actual behavior |
| --- | --- |
| instance_generator.number_of_cells | Fixed adult count and nominal nest area used to calculate radii |
| instance_generator.nest_fill_percentage | Sets larval count as ceil(number_of_cells * fill) |
| instance_generator.proportion_inner_larvae | Sets inner larval population fraction and nominal inner area before rounding |
| wasp.hunger_rate | Adult effective metabolic/hunger rate |
| larvae.hunger_rate | Base rate multiplied by each larva's sampled multiplier |
| larvae.hunger_rate_multiplier_l2 / l3 | Sampled with 1 as three equally likely choices, independent of nest region |
| instance_generator.larvae_to_wasps_ratio | Legacy stored value, not used to set current populations |
| instance_generator.larvae_hunger_multiplier / hunger_rate | Legacy stored values; use the wasp/larvae sections instead |
| simulator.grid_buffer / forage_buffer | Legacy; generation uses the instance_generator buffers |
| simulator.number_of_bouts | Inactive stopping option; current loop stops by step horizon |
| simulation.pathfinding_mode | Inactive legacy value; main requires --mode and run_experiment runs all modes |
| simulation.num_runs / run_mode | Used by batch_run; run_experiment intentionally runs one generated instance per CLI seed |

Legacy keys remain accepted for compatibility, with comments marking their status. They are not silently reinterpreted as new biological mechanisms. The experiment manifest now reports actual populations, effective sampled rates/multipliers, total larval metabolic demand, role scheduling values, ignored legacy keys and runner overrides.

For seed 20261108 and the current config, the audited generated instance has 30 adults and 29 larvae, adult rate 0.1, larval rates 0.3/0.6/0.9, and total larval metabolic demand 17.7 model food units per step. This demand does not imply that the colony can supply it.

## Interpretation

The YAML numerical values were not tuned to preserve the appearance of historical graphs. Enforcing them can substantially increase hunger. Per-mode hunger plots now autoscale their upper y limit instead of clipping observations at 3.

Start with a short generated-nest run to inspect effective dynamics. Treat it as software validation until rates and food supply are calibrated against field observations and the updated manuscript. Previous plots cannot establish that these newly effective parameters are biologically appropriate.
