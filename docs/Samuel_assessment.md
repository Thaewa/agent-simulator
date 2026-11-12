# Samuel's assessment: biological bouts, a food provenance ledger, and a modular refactor

Author: Samuel Cornejo. Date: 2026-11-12. Written against branch `fix/tsp-validation`, building on commit `3882633` ("fix: distinguish transfer events from biological bouts") and `docs/bout_definition.md`.

**Status: proposal only. No code is changed by this document.** It sets out what I think is missing, a candidate architecture, and the decisions we need, mostly from the advisor, before anyone implements it.

---

## 1. Three objects that must stay distinct

| Object | What it is | Status in the simulator |
| --- | --- | --- |
| **Food transfer** (`FoodTransferEvent`) | One handoff: donor, receiver, timestamp, amount | Logged exactly (`food_events_v1`) |
| **Legacy proxy** (`finished_bouts`) | Groups transfers by forager and closes a group when food thresholds are reached | Kept, labelled `legacy_food_threshold_proxy_v1` |
| **Biological feeding bout** | From a forager's *arrival* at the nest until the food it brought is used up. Overlapping arrivals extend the same bout | **Not reconstructed** |

## 2. Why the simulator can't produce real bouts yet

The reasons, from `docs/bout_definition.md`:

1. **No arrival event.** The simulator records when food is collected at a foraging site, not when a forager lands on the nest, and a bout starts at that landing.
2. **Food has no origin tag.** Food from different sources mixes in each adult's store and gets passed on. The simulator can't tell when a particular arrival's food is used up.
3. **Initial food** at the start of a run comes from no arrival.
4. **Food leaves stores in other ways.** Adults burn food through metabolism, and larvae keep what they're fed. Neither of these marks the end of a bout.
5. **Overlapping arrivals, runs that end mid-episode, and coarse time steps** make the boundaries ambiguous.

### The legacy proxy

The old `finished_bouts` counter sits between a transfer and a bout. It groups transfers by forager and closes a group when food thresholds are reached. That is a made-up rule, not the field definition, so commit `3882633` labels it `legacy_food_threshold_proxy_v1`. The count must not be read as a number of biological bouts in either direction. A count of zero doesn't mean no food was distributed, and a positive count doesn't mean a real bout happened.

---

## 3. Proposal: a food provenance ledger

A bout ID by itself isn't enough. To know when a bout ends, the simulator has to know at every step how much of each bout's food is still held, and by which agent across the whole nest. That calls for a **ledger** with a static part (IDs and records) and a dynamic part (who holds what, step by step).

### 3.1 Identifiers (the static part)

- **Bout ID:** one per biological bout, opened by an arrival and closed when its food is used up.
- **Load ID (arrival ID):** one per forager arrival carrying food. A bout is made of one or more loads.
- **Feeding / transfer ID:** the existing `event_id` for each transfer and larval feed. Each one points to the bout(s) its food came from.

Because a donor can hold food from several bouts, one transfer can carry food from more than one bout. So the link from transfers to bouts is **many-to-many**, stored as a small allocation table:

| event_id | bout_id | load_id | amount |
| --- | --- | --- | --- |

rather than a single `bout_id` column on the event.

### 3.2 Agents holding tagged food (the dynamic part)

- **Per agent:** instead of one `food` number, each agent holds parcels such as `{bout_7: 1.5, bout_9: 0.5, initial: 2.0}`. The existing `food` value stays equal to their sum, so simulation logic keeps working unchanged.
- **Nest-wide:** a table of open bouts, each with food still held by adults, food delivered to larvae, and food eaten through metabolism.
- **Each step:** a bout closes when no adult holds any of its food anymore.
- **Each transfer:** writes its allocation rows (above).

This dynamic part is the core of the proposal: tracking, across the whole nest, which agents still carry food from which bout, so the end of every bout can be detected at the step it happens.

### 3.3 Decisions the code forces

1. **What opens a bout.** Food currently enters the simulation at `collection` (`Wasp.forage`, `agents.py:494`), at the foraging site, but a bout starts when the forager *arrives at the nest*. There is no arrival event yet. Candidate: the first step a forager carrying food is back in the nest, using `inOuterNest()` as a possible test. Food stays untagged from collection until arrival.
2. **Mixed stores: whose food leaves first?** When an adult holding food from several bouts gives 1 unit away, which bout does it come from: oldest first, proportional, or newest first? The choice changes *when each bout ends*. Suggested default: **proportional**, because it is neutral, kept configurable.
3. **Food that leaves circulation in other ways:**
   - **Metabolism** (`simulator.py:982`): adults burn food every step, so their bout parcels must shrink by the same allocation rule.
   - **Larvae** keep food in storage. If larval food counted as "still held", no bout would ever end. Suggested: food delivered to a larva counts as **delivered** (used up for that bout). Only food held by adults is tracked as outstanding.
   - **A forager flying back out** with leftover food: does that food stay in the bout, or count as exported? Open question.
   - **Initial food** gets an `initial` tag and never belongs to a bout.
4. **Overlapping arrivals.** The manuscript says a bout extends when a new forager arrives before the earlier food is gone. Suggested: track each **arrival load** as the basic unit, and define a bout as a chain of overlapping loads, so arrival 2 joins bout 7 if bout 7 is still open. This keeps per-forager detail and matches the field definition. Bouts still open when the run ends are marked **`censored`**, not complete.

### 3.4 Guardrails for any implementation

- **Food is conserved:** at every step, food held + delivered + eaten = food brought in (plus initial food).
- **No behavior change:** the ledger only annotates. Movement, feeding and RNG draws must stay byte-identical, checked against the existing smoke-test hashes.
- **Tests for:** a single arrival, overlapping arrivals, the same forager arriving twice, onward transfers, mixed initial and new food, and a run ending mid-bout.

### 3.5 Caveat

`docs/bout_definition.md` asks that the field protocol be confirmed with the advisor first, especially decisions 2 and 3. Their answers could change those rules, so the rules should be configuration, not hard-coded.

---

## 4. Advocacy: a general refactor toward modular planning and action

I am advocating for a **general refactor** of the agent step, not only a ledger bolted onto the current code. The aim is to separate **perception**, **planning (decision)** and **action (state change)** into explicit modules. That makes the simulator compatible with future changes, in particular **Deep RL** policies.

### 4.1 What is entangled today

- `Wasp.step` (`agents.py:1055`) decides whether to feed, draws random numbers, mutates food stores, and calls `forage` in one method.
- `feedNearby` / `feed` / `feedWasp` / `feedLarvae` (`agents.py:399`–`1026`) mix the *decision* (chance draws, hunger cues, role checks) with the *effect* (moving food between agents and logging).
- `feelGradient` (`agents.py:956`) mixes perception (gradient sensing) with choosing among the path-finding modes (greedy, random walk, TSP variants).
- `Simulator.stepAgent` (`simulator.py:986`) branches by role and assembles each role's perception arguments inline.
- Metabolism is applied separately in the simulator (`simulator.py:982`), outside the agent's action path.

Because of this, swapping in a learned policy, or adding a ledger, means editing the same methods that hold the biological rules.

### 4.2 Proposed shape

```
Observation  ->  Policy (plan)  ->  Action  ->  Environment / Ledger (apply + record)
```

- **Observation:** a per-agent snapshot: position, role, food (and parcels), hunger cues, local gradients, nearby agents.
- **Policy:** a pluggable interface that returns *intended actions* only (move target, feed target and amount, forage, idle). The current rule-based behaviors and path-finding modes become policy implementations. A Deep RL policy would be another one, with no changes elsewhere.
- **Action application:** a single place that validates and applies actions, such as capacity limits and collision handling, and makes every food movement through the ledger.
- **Ledger:** the static + dynamic structure from section 3. It is the one source of truth for food movement, so transfers, larval feeds, collection, arrival and metabolism are all recorded the same way, and bout open/close is computed there.

### 4.3 Why this matters

- **Deep RL:** a clean observation → action → reward loop (Gym-style `reset`/`step`) is what RL training needs. Rewards and evaluation can then be defined against field statistics such as bout durations and feeds per bout. Those come straight from the ledger.
- **Comparability:** rule-based, TSP and RL policies run in the same environment with the same ledger and metrics, so comparisons are fair.
- **Reproducibility:** separating decision from effect makes it easier to guarantee that annotations such as the ledger never change RNG streams or behavior.
- **Extensibility:** future biological rules, such as new roles or a different bout definition from the advisor, change one module instead of several interleaved methods.

### 4.4 Suggested order

1. Agree the bout definition and the decisions in 3.3 with the advisor.
2. Refactor step into observation / policy / action with **byte-identical** outputs, using the existing smoke-test hashes as the regression gate.
3. Add the ledger as the action-application layer, with the conservation and scenario tests from 3.4.
4. Add the RL environment wrapper and policy interface on top.

---

## 5. Summary

- Transfers are logged. Biological bouts are not, and the legacy count is only a proxy.
- Real bouts need a **provenance ledger**: Bout IDs, Load IDs and Feeding/transfer IDs as the static part, and **agents holding food tagged by bout ID** as the dynamic part, tracked nest-wide every step.
- I recommend building it as part of a **modular refactor** (observation → policy → action → ledger) so the simulator is ready for Deep RL and future rule changes.
- Nothing is implemented yet. The next step is confirming the protocol decisions with the advisor.
