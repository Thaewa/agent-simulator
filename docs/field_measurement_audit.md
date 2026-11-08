# Field measurement versus simulator logging

Audit: 2026-11-08. Scope: read-only comparison of `logger.py`, `agents.py`, `simulator.py`, current metric definitions, and methods recovered from the local biological manuscript described in empirical_data_inventory.md. The manuscript revision is unverified. This is not an empirical validation or an advisor-approved endpoint selection.

## Main finding

Current logs support model-state diagnostics and aggregate successful-event counts. They do not yet support a faithful reconstruction of each adult's ordered larval-feeding route within an empirically defined feeding bout. More plots or longer runs cannot recover information that was not logged.

## Measurement mapping

| Field quantity | Current evidence | Readiness and limitation |
| --- | --- | --- |
| Colony/map identity | Experiment seed, initial instance, simulation ID and coordinates | Suitable for synthetic replay; simulation ID is not an empirical colony ID. No validated real-map import or physical coordinate scale. |
| Adult paths | End-of-step adult positions and IDs | Grid-space trajectories can be reconstructed. These are not the ordered larval feeding stops used for the local manuscript's route analysis. |
| Colony feeding activity | Per-step successful event deltas and adult lifetime totals | Usable in model units. No physical time conversion or matched field observation exposure. |
| Individual feeding participation | Adult lifetime feeding/transfer counters and aggregate summaries | Some counters exist in memory, but aggregate mean/SD loses individual distributions. Adult state CSV has no cumulative individual feeding counters. Cannot substitute larval feeds for food-receiving counts. |
| Individual feeding sequence | `storedEvents` strings and adult CSV `action`/`target_id` | Insufficient. CSV records only the last stored event, which can repeat on later steps without a new event. Multiple same-step feeds are collapsed. Strings lack event time, quantity and event-time positions. |
| Larval food delivery | Larval CSV `food_received` | The value is current larval food storage, not delivered food during that step or cumulative intake. Metabolism prevents treating storage changes alone as a feeding-event record. |
| Biological feeding bouts | `FeedingBout`, `active_bouts`, `count_bouts` | `FeedingBout` represents a successful forager-to-feeder transfer. Completion uses receiver/forager food thresholds and does not establish the manuscript's arrival-to-exhaustion food episode. No exported event-to-biological-bout table. |
| Larval spatial feeding distribution | Larval IDs/coordinates and distance from coordinate origin | Hunger-by-distance is available, but exact per-larva feeding frequency and feeder identity are not losslessly exported. Need the empirical nest-centre definition. |
| Developmental stages | Sampled hunger multipliers | A multiplier is not a validated larval stage label. No explicit empirically linked stage field in state logs. |
| Within-bout cell sharing / feeding-area overlap | Positions and aggregate counters | Needs complete feeder-to-cell events and valid bout membership; cannot infer from end-of-step proximity. |
| Inspected but skipped cells | No explicit inspection/skip observation | Unsupported. Being nearby does not establish an inspection. Do not manufacture these events from movement alone. |

## Logging semantics to preserve explicitly

- In `Simulator.run`, feeding actions occur before movement; adult state rows are written after movement, metabolism and role changes. State-row coordinates/roles cannot be assumed to be event-time coordinates/roles.
- `feedNearby` can feed multiple recipients during one step. A separate sequence within each step is needed to preserve order.
- `Wasp.feedLarvae` and `Wasp.feedWasp` already know the donor, recipient and actual amount at success time. `storedEvents` is a legacy string history, not a complete timestamped event journal.
- `FeedingBout.timestamp` and `amount` exist for a subset of transfers but do not constitute a journal of all larval feeds, adult transfers and collections.
- Current event totals can remain useful reconciliation checks. Existing schemas and historical plots must not silently acquire new meanings.

## Next bounded implementation: lossless successful-event journal

Proposed engineering step, not a direct instruction in the advisor's email: add a separate versioned event CSV, without changing behavioral decisions or random draws. This provides an auditable foundation for later field comparisons while raw empirical data and revised endpoints are being confirmed.

Proposed minimum fields:

- Run identity and mode; schema version; event ID; simulation step and deterministic within-step order.
- Event type: collection, adult transfer, larval feed.
- Donor/actor and recipient identity/type as applicable; actor and recipient roles and positions at event time.
- Actual food quantity, in explicitly labeled model units.

Record only successful events with positive quantities. Use empty recipient identity for external collection, rather than inventing a recipient. Do not assign a biological bout ID until the bout definition and food-origin tracking have been resolved. Do not rename the existing larval `food_received` column silently; document its storage semantics or version any replacement.

Validation required for that step:

1. Multiple feeds in one step retain every recipient and its order; no-event steps do not duplicate the previous event.
2. Failed/no-op attempts yield no successful-event row; partial adult transfers use the actual quantity.
3. Event-time roles/positions are preserved across subsequent movement and role changes.
4. Journal counts reconcile with existing per-step and lifetime totals for each event type.
5. With the same seed and instance, adding the journal leaves modeled states and behavior unchanged; repeat runs produce identical event records after excluding output-directory metadata.

At the time of this audit commit no journal code was implemented. The subsequent implementation is documented in food_event_log.md. Real-map import, biological bout reconstruction, field-unit calibration, null-route analysis and RL training remain separate steps. Collection records alone will not resolve food provenance through onward transfers or establish a biological bout.
