# Metric definitions: activity_roles_v2

These definitions apply to newly generated experiments whose aggregates contain `metrics_schema=activity_roles_v2`. Historical CSV files and plots are not rewritten. The updated plotting script rejects historical/mixed schemas rather than relabeling their values. Use the source snapshot saved with a historical experiment to reproduce its original figures.

## Scope and units

Metrics assume a fresh simulator instance for each run. One step is a discrete simulation tick, and distance is Euclidean distance in grid-coordinate units. No claim about physical centimeters or energy is implied. A successful feeding event can transfer less than the nominal ration when adult recipient capacity limits it; event counts are not units of food delivered.

| Field | Definition |
| --- | --- |
| total_larval_feeding_events | Successful larval feeds by all adults during the run |
| total_adult_transfer_events | Successful adult-to-adult food transfers |
| total_foraging_events | Successful collection events; can include a small refill |
| mean_feed_count_larvae / wasp | Mean lifetime event count per adult, including zero-event adults |
| std_feed_count_larvae / wasp | Population SD of adult event counts within that run |
| mean_feed_rate_larvae / wasp | Mean event count per adult divided by completed run steps |
| std_feed_rate_larvae / wasp | Within-run adult count SD divided by completed run steps |
| mean_distance_per_feed_larvae / wasp | Total distance of all adults divided by total successful events of that type; includes travel by adults with no events |
| final_low_hunger_fraction | Fraction of larvae with hunger strictly below 1.5 at the final state |
| feeding_efficiency | Compatibility alias for final_low_hunger_fraction; not an energetic or route-efficiency measure |
| mean_feed_freq_* / std_feed_freq_* | Compatibility aliases for the corresponding rate fields in this schema; historical schema values were counts |

The distance-per-event metric is a colony-level travel cost proxy. It does not claim every movement was caused by that event type. A zero event denominator is undefined, not zero travel cost. Undefined quantities are NaN/empty numerical CSV values and JSON null in experiment manifests.

## Role attribution

Each adult now records feeder step exposure, feeder travel distance and larval feed events at the time those activities occur. Changes to its final role do not remove its earlier feeder activity or add its forager activity to feeder totals.

| Field | Definition |
| --- | --- |
| num_feeder_participants | Adults with at least one feeder-role move/step call or successful feeder larval feed |
| sum_distance_traveled_feeders | Distance traveled while acting as a feeder |
| sum_step_count_feeders | Number of adult step exposures while acting as a feeder, including stationary steps |
| sum_feed_count_larvae | Successful larval feeds made while acting as a feeder; use total_larval_feeding_events for all adults |
| avg_*_feeders / avg_feed_count_larvae | Corresponding feeder activity sum divided by participating adults |
| efficiency_walk_feed_ratio | Feeder larval feed events / feeder travel distance |
| efficiency_step_feed_ratio | Feeder larval feed events / feeder step exposures |

These replace final-role selection with lifetime counters. An adult can contribute both feeder and forager activities during a run.

## Plot normalization and uncertainty

- Distance per feeder-role step is feeder distance / feeder step exposures. The old formula divided by adult population again even though the denominator already summed exposures across adults.
- Feeder feeds per larva is feeder larval feed count / number of larvae.
- Feeder steps per participating adult is feeder exposure / participating adults.
- Feeds per distance and feeds per exposure use their raw event/denominator ratios; they do not divide differently normalized populations and call the result feeds per distance.
- Bar/error plots aggregate one metric value per independent run. Error bars are sample SD across run-level values, not the SD of within-run standard deviations. With one run, between-run SD is undefined and omitted. These bars are not confidence intervals.
- Duplicate mode/run IDs are rejected rather than silently averaged together. Each independent run must have a unique ID within the combined data.
- The spatial plot retains run ID and larva position in grouping so equal larva labels in different runs are not merged.

Validation covers pooled distance costs, zero denominators, count-versus-rate units, role changes, a real action sequence and logged aggregates after role changes, population normalization, between-run SD, missing single-run uncertainty and schema/identity rejection. This schema change affects observation and reporting; it does not change movement or feeding decisions.
