# Food event journal: food_events_v1

Fresh runs write `food_event_log_<mode>.csv` alongside the existing state logs. Old outputs are not rewritten. Each row represents one successful collection, adult transfer or larval feed with a positive actual quantity. Failed attempts, initial food, metabolism, movement and role changes are not food events.

## Fields and ordering

| Fields | Meaning |
| --- | --- |
| event_schema | `food_events_v1` |
| simulation_id, pathfinding_mode | Run identity within that output; retain experiment-directory identity when combining experiments |
| event_id | One-based sequence within the logger's run |
| timestamp, step_event_order | Simulation step and one-based order of successful food events across all adults in that step |
| event_type | `collection`, `adult_transfer` or `larval_feed` |
| actor_id/type/role/x/y | Acting adult's identity, type, role and position at event time |
| recipient_id/type/role/x/y | Food recipient at event time; absent for collection, and larval role is empty |
| amount, amount_unit | Actual amount gained/transferred, including partial refills/transfers; unit is `model_food` |
| source_x, source_y | Forage-site position for collection; empty for transfers/feeds |

Positions use model grid-coordinate units. Roles use enum names such as `FEEDER` and `FORAGER`. Feeding occurs before movement in the current simulator; use event coordinates/roles for feeding analysis, not later state-row coordinates/roles. No row is emitted at initialization or during an idle step. Multiple events at one timestamp are intentional. Rows are written immediately, not reconstructed from legacy event strings.

The old larval state column `food_received` continues to mean current food storage. The new journal supplies delivered quantities without silently redefining that historical column. Existing state and aggregate schemas remain unchanged.

## Verification and limits

`smoke_test.py` and `run_experiment.py` reconcile event counts against nest counts at each step. Their existing nest-to-aggregate checks supply a second reconciliation. Journal IDs, within-step ordering, units and positive finite amounts are checked for their fresh single-run outputs. Seeded smoke replay hashes include the journal as well as state logs.

Tests cover multi-recipient events, no-event steps, no-op attempts, partial quantities, collection sources, event-time snapshots, sink cleanup on exceptions, and missing-event detection. A five-mode comparison with journal logging enabled versus disabled checks identical modeled states, state CSV bytes, counters and Python/NumPy RNG states. `DataLogger(..., enable_event_log=False)` is an internal diagnostic option, not a simulation policy change.

The journal deliberately has no biological bout ID. It does not track food provenance through onward transfers, forager arrivals, inspected/skipped cells, physical time/distance calibration or developmental-stage observations. Consequently it prepares route/event analysis but does not establish equivalence to the manuscript's field protocol. A collection is not automatically a new biological bout.

Event IDs are scoped to a logger run, not globally unique. Normal runners use fresh output folders; when combining appended logs, preserve simulation ID and output identity. An interrupted run may contain action records before that step's state row and must not be treated as a completed experiment.
