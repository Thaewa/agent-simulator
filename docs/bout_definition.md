# Feeding bouts: definition and implementation boundary

Assessment: 2026-11-08. The biological reference is the local, revision-unverified Sharma/Gadagkar manuscript at `../manuscripts/[32] .pdf`, methods p. 14. The advisor's latest email takes precedence for the research framing. This document does not assert approval of that manuscript's older optimization interpretation.

## Three different objects

| Object | Definition | Current implementation |
| --- | --- | --- |
| Successful food event | One collection, adult transfer or larval feed | Exported losslessly in `food_events_v1`; not a whole bout. |
| Legacy threshold proxy | Groups forager-to-feeder transfer recipients and closes groups using adult food thresholds | Existing `active_bouts`/`finished_bouts` logic is retained for compatibility. Its threshold output is explicitly labeled as a proxy. |
| Biological feeding bout | In the local manuscript, a food-distribution episode from a forager's arrival through exhaustion of brought food, extending across overlapping arrivals | Not reconstructed; no biological bout IDs or valid biological bout count are available. |

The Python class now called `FoodTransferEvent` represents only one successful forager-to-feeder transfer. `FeedingBout` remains an alias for old imports and constructor calls. Its historical `feeder` field is the donor; `donor` is an explicit readable alias. This naming change does not alter feeding decisions, transfers, random draws or thresholds.

New aggregate rows include:

- `bout_count_semantics=legacy_food_threshold_proxy_v1`.
- `biological_bout_status=unavailable`.
- `legacy_bout_proxy_count`: the existing `finished_bouts` value, never relabeled as an empirical bout count.

The console also identifies completed proxy episodes by name. `number_of_bouts` remains an inactive stopping option; runs end at the step horizon. A zero proxy count does not mean no food distribution occurred. A positive proxy count does not prove that any complete biological bout was observed.

## Why the journal is not enough to assign biological IDs

1. Collection at an external site is not a recorded arrival/alighting on the nest. Adults can refill and exchange food before/after movement under current rules.
2. Initial food is unassigned to an observed arrival. Its contribution cannot simply be called the first field bout.
3. Adult food stores mix multiple sources and can be passed onward. Event quantities alone do not establish when the food associated with an arrival has been exhausted.
4. Adult metabolism depletes stores; larval storage persists after feeding. Neither total nest food nor a recipient's food threshold is automatically the field episode's end rule.
5. Multiple arrivals can overlap, episodes can cross the observation boundary, and the within-step ordering of actions/movement is coarser than field timestamps.

Do not substitute a fixed time window, one transfer, one forager, one collection or a food threshold for the biological definition without explicitly defining a different analysis and justifying it.

## Inputs and decisions needed before an episode implementation

- Confirm the latest bout definition and any pre-existing empirical bout IDs with the advisor.
- Obtain event/observation tables with colony ID, adult identity, recorded arrival time/location, feeding/transfer identities and times, and observation start/end where available.
- Determine whether food quantity/exhaustion is directly observed or annotated, and how ambiguous/overlapping arrivals and incomplete episodes were handled.
- Agree how model arrivals, initial food, source mixing, onward transfers, adult consumption and incomplete final episodes map onto that observation protocol. If provenance is necessary, specify a conservation rule before adding source tracking.

If empirical bout labels already exist, preserve them rather than re-deriving them from an invented threshold. Future model bout reconstruction should be a separate tested change after these decisions; current code does not claim this work is done.

## Proposed acceptance cases for that future change

Use agreed observed/synthetic examples for one arrival, overlapping arrivals, repeated arrival by the same adult, onward food transfers, mixed initial/new food, and observation-end truncation. Check that event membership is complete and unambiguous where the agreed definition permits it. Any provenance ledger must conserve food including consumption, and enabling annotation must leave movement/feeding decisions and RNG streams unchanged.

Next scientific dependency is the advisor's current methods and usable empirical data, rather than another synthetic parameter adjustment. No data-request message has been sent.
