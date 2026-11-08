# Bug-fix progress

Base: `f64d70c9a02782ab2840e74708b0f4a7f624b0d2`, branch `baseline/report-era`.

## Step 1: mode selection

Implemented on 2026-11-08:

- Normalize supported mode names to lowercase at the CLI and agent property, including the generator's post-construction override.
- Reject unknown modes instead of silently using greedy under an incorrect label.
- Select TSP routes for `tsp` and `tsp-hamiltonian`; select a closed solver tour only for `tsp-hamiltonian`.
- Preserve the existing feeder-only, food-greater-than-one restriction and greedy fallback.
- Preserve constructor-selected policy when the generator has no simulator-level mode override.

Validation:

```powershell
python -B -m unittest discover -s tests -p test_pathfinding_modes.py -v
```

All 11 new regression tests pass. Tests isolate dispatch and the cycle flag with mocks; they do not establish successful end-to-end TSP routing. No batch experiment or plot regeneration was performed, and historical outputs were not replaced.

Full discovery runs 22 tests: 13 pass, 8 error, and 1 fails. Running the 11 original tests in the unchanged baseline worktree reproduces the same 8 errors and 1 failure. Those original tests use obsolete Wasp constructor arguments/positions and reference a removed Simulator.step method. This is pre-existing test debt, not a passing full suite.

## Step 2: half-cell routes and target capacity

Implemented on 2026-11-08 after the user confirmed the mode tests passed:

- Preserve exact target and agent coordinates instead of truncating to integers.
- Build the rectangular route graph on the same half-cell lattice as sampled agent movement. Bound it by the selected targets and current position; there are no obstacles in this local graph.
- Select up to `min(floor(food / transfer), candidate_count)` nearest larvae, including when capacity covers every candidate. Handle one target using a shortest path and keep greedy fallback for no targets/no capacity.
- Join the connector to the tour without a duplicated seam node, and skip the stationary start node that movement would otherwise randomize.
- Discard a cached route if the next node is more than one sampled half-cell step away after a deviation.

Validation command:

```powershell
python -B -m unittest discover -s tests -p test_tsp_routes.py -v
```

Nine new route tests use the real graph and NetworkX solver. They cover integer, half-cell and negative coordinates; enough food for all candidates; single and zero targets; open and closed solver tours; real collision-free `move()` execution; and route invalidation after deviation. No solver stubs are used in these route tests; one wrapper records the actual solver tour. The existing 11 mode-selection tests remain separate.

The closed target tour returns to its first target; the connector from the wasp's starting position is not itself required to be a closed tour. The open connector can incidentally pass another target before reaching the solver's tour start. These heuristics are not claims of globally optimal routes.

No full simulation or baseline plot regeneration has been performed during this step.

## Step 3: isolated seeded simulation check

Implemented on 2026-11-08. Run with:

```powershell
python smoke_test.py
```

This creates a new directory under `exports/smoke_checks/` on every invocation. DataLogger now accepts an optional keyword-only `output_dir`; its normal output location is unchanged when omitted. No simulator or planner is replaced with a stub in this check: wrappers record calls to the real methods and solver.

Each of the five modes runs for 20 steps in a small hand-built scene with two feeders, one forager and four larvae, twice with Python and NumPy seed 20261108. Both repeats must have identical state-log hashes and counters. The check requires finite states, nonnegative food, observed feeding somewhere in the suite, actual TSP route construction and at least one followed route step in both TSP modes. A random mode can legitimately have zero feeds in this short scene; the check does not assert a performance ranking. Existing collision handling may alter a planned move and is counted separately.

Observed result: greedy 0 routes/4 larval feeds; random walk and biased random walk 0 routes/0 feeds; TSP 11 routes/26 followed planned moves/4 feeds; Hamiltonian TSP 6 routes/35 followed planned moves/3 feeds. Both repeats matched exactly for all five modes. The script verifies baseline log hashes are unchanged and writes source hashes, dependency versions, logs and a result manifest into its output directory.

This is an integration smoke check, not a full 500-step experiment, statistical comparison, or validation against field data. Duplicate timestamp zero remains present and is reported in the manifest.

## Step 4: unique timestamps and current-API tests

Implemented on 2026-11-08 after the user confirmed the smoke check:

- Keep initialization at timestamp zero, advance time before each step's actions, and log the resulting states at timestamps 1..T. Final `currentTime` remains T. No rows are removed or deduplicated after the fact.
- Add regression checks for zero-step initialization, nest/larva/wasp timestamp sequences, initial values, and final logged positions.
- Update the legacy tests to the actual Wasp constructor, explicit role assignment, initialized cue ranges, move arguments, and `runSimulation` entry point. Tests use temporary log directories and preserve caller random state; the removed Simulator.step API was not reintroduced just to satisfy old tests.
- Strengthen the feeding test to check transferred food and counts, and the movement test to check exact position and distance. The foraging test checks that collection occurs at a nearby site under a controlled draw; food-capacity semantics still require their own audit.
- Require the smoke test's nest, larva and wasp state logs to contain exactly timestamps 0..20 with no duplicates per identity. Repeated seeded runs still match and baseline log fingerprints remain unchanged.

Validation: all 33 tests pass with `python -B -m unittest discover -s tests -v`; the five-mode smoke check also passes after this change. The earlier record of nine failing legacy tests describes the pre-update state, not the current suite.

## Step 5: food capacity and conservation

Implemented on 2026-11-08 after capacity overflow was confirmed in the first 500-step run:

- Forage adds `min(forageIncrease, maxFood - food)` on success instead of adding a whole maxFood. Multiple nearby sites cannot overfill the agent.
- Wasp transfers are limited by the recipient's free capacity. A recipient with 0.25 space receives 0.25, and the donor loses exactly 0.25. This rule applies to both normal and high-cue forager transfers. The donor must hold at least the requested ration before attempting a transfer; an undersupplied donor does not send a partial ration.
- Direct transfer helpers ignore zero/negative requests, insufficient donor food and full recipients; wasp self-transfer is ignored. No food-transfer count or hunger reset is generated for these no-op cases.
- Exact donor equality is allowed: the final complete ration can be transferred, including through the normal nearby-feeding loop.
- Larvae still have no storage-capacity model. Their transfers consume the specified ration with a donor-sufficiency check; no new larval capacity assumption was introduced.
- Smoke and full experiment runners now check upper food bounds as well as nonnegativity, including all recorded adult food states. Existing logs and experiment snapshots were not modified.

Validation: ten new food-boundary tests reproduce overflow, wrong forage increment, capacity-limited transfer, insufficient donor resources and self/zero-transfer counter defects before the fix. After the fix, all 43 tests pass and the seeded five-mode smoke check passes, with identical replay logs and baseline fingerprints unchanged.

The previously reviewed experiment `seed_20261108_20261108T021313_345591Z` is a pre-food-fix result and remains intact for inspection. A fresh `python run_experiment.py` invocation writes a new experiment folder and must be used for post-food-fix results. No full 500-step rerun was performed during this step.

## Step 6: role-switch scheduling and consistent state logs

Implemented after the user pushed checkpoint `43417c9` to `fix/tsp-validation`:

- Pass the current completed step to updateRoles, rather than the simulation horizon. A 500-step horizon previously made the frequency-100 gate true on every step; a nondivisible horizon made it false on every step.
- Apply the colony-wide change quota with `< max_role_changes`, preventing an extra transition after the quota is exhausted. The counter is shared across agents and resets for each step, with eligibility only at scheduled reassessments.
- Treat the two configured role-change probabilities as probabilities of changing: `draw < probability`. Previously `draw > probability` inverted their meaning. Eligibility constraints still apply.
- Restore the configured role-persistence interval after a successful switch. Decrement an existing cooldown before considering a switch so consecutive changes must be separated by at least the configured number of steps; do not decrement a newly restored interval that tick.
- Treat the configured forager-fraction boundaries as inclusive hard bounds, retaining at least one forager. This is an explicit boundary interpretation, replacing strict inequalities that excluded equality.
- Define the default upper forager threshold in Simulator itself; generator configuration can still override it. A directly constructed simulator previously lacked this attribute.
- Emit adult state logs in a separate pass after all movement, metabolism and role updates, so adult role counts and nest role counts refer to the same completed step. Adult food/hunger fields now also represent post-metabolism state. Historical logs are not rewritten.

Validation: 12 new role tests reproduce clock, probability, quota, boundary, cooldown and role-log inconsistencies before the fixes. All 55 tests pass after the changes; the five-mode 20-step seeded smoke check passes with identical repeat logs and unchanged baseline files. No 500-step rerun was performed for this step. The short smoke check itself does not reach the default step-100 reassessment; dedicated integration tests use a two-step frequency to exercise the actual simulation call and state logging.

## Step 7: successful feeding events and per-step counts

Implemented after role-switch checkpoint `ef74858`:

- Interpret `chanceOfFeeding` as the probability of an attempted feed for a nearby hungry recipient (`draw < probability`), replacing the inverted comparison. Actual success still depends on food, capacity and the existing unloading rule.
- Feeding methods return the actual quantity transferred, or zero for a failed/no-op attempt. Feeding bouts are created only for successful forager-to-feeder adult transfers and include that quantity. Larval feeds and failed transfers do not enter adult-transfer bout accounting.
- Exclude the donor from nearby recipients before consuming random draws. Return an empty list for no receivers, preventing None from reaching the simulator's feeding-event list extension. Empty forage-site lists are also harmless.
- Track successful collection events with a numeric counter. Nest log foraging/larval-feeding/adult-transfer fields are counter deltas for the current step, rather than cumulative string searches over all history. Initialization records zero events.
- Add all-adult lifetime totals under `total_larval_feeding_events`, `total_adult_transfer_events`, and `total_foraging_events`. The aggregate marker `nest_event_count_semantics=per_step_successes` distinguishes new nest logs from historical cumulative event counts.
- Smoke and experiment runners assert that summed per-step nest event counts agree with aggregate lifetime totals.

Validation: 12 new tests cover probability endpoints/thresholds, empty recipients, self exclusion, failed/partial transfers, rejected unloading, adult-versus-larval bout classification and all three step-count totals. All 67 tests pass, and the five-mode seeded smoke check replays identically with unchanged baseline logs. No full 500-step rerun was performed for this step.

Probability correction and exclusion of self trials change random-stream consumption and feeding dynamics. Historical graphs must not be relabeled as post-fix results. The existing feeding-efficiency and final-role aggregate formulas have not yet been redefined in this step.

## Step 8: metric units, activity-time roles and run uncertainty

- Add the versioned `activity_roles_v2` schema and definitions in `docs/metrics.md`.
- Track feeder exposure, movement and larval feeds at action time, rather than selecting final-role agents and using their whole lifetime activity.
- Keep explicit event counts and calculate actual rates per adult per completed step. Distance cost is a ratio of all-adult total travel to successful events, including adults with zero feeds. Undefined denominators produce missing values, not favorable zero costs.
- Label the final hunger-threshold fraction as a state measure. It is not a measure of route or energetic efficiency.
- Correct feeder distance per step normalization and use run-level means for between-run SD; omit SD error bars for a single run. Preserve run ID in spatial grouping and reject duplicate mode/run IDs.
- Reject historical metric schemas in the updated plotting pipeline. Old results remain intact with their saved source versions. New experiment manifests serialize undefined metric values as JSON null.

Validation: all 78 tests pass and the five-mode seeded smoke check passes. A separate two-step generated-nest run checks the complete plotting pipeline; this is not a new 500-step comparison. No movement, feeding, role-switch or random-draw rule was changed in this step.

## Step 9: movement validity, collision exclusion and disjoint nest regions

- Correct grid membership, preserve explicit zero actions, validate allowed displacements, and bound fallback attempts with a safe stay action.
- Use actual adult occupancy during sequential movement, preventing transient overlap and swaps; document fixed-order priority and the allowance of diagonal segment crossings.
- Align feeder movement and TSP graph masks with the closed nest boundary, exclude inactive graph nodes, and ignore exterior feeder targets.
- Create default grids at movement resolution covering forage sites as well as agents. Reject invalid origins and reused agent IDs.
- Make generated inner/outer sampling regions disjoint to prevent boundary duplicates. This changes initialization sampling; old/new seed equality alone does not establish map equality.
- Add movement-model metadata to experiment manifests and runtime geometry assertions to smoke and generated-nest runners. Definitions and modeling limits are in docs/movement.md.

Validation: all 98 tests pass. New cases cover membership, stay, bounds, retries, non-finite actions, occupied origins, swaps, contested cells, half-cell default grids, masked TSP routes, exterior targets and unique generated populations across 20 seeds. The five-mode seeded smoke check replays identically with geometry assertions and unchanged baseline logs. A two-step generated-nest runner check also passes in a temporary output directory. No new 500-step run was performed for this checkpoint.

## Step 10: effective larval metabolism and explicit legacy configuration

- Apply configured larval base rate before multiplying, rather than inheriting an unrelated 0.01 default; use L3 in sampling instead of repeating L2.
- Reject negative/non-finite metabolic rates or multipliers, including invalid multiplier choices before random selection.
- Mark inactive legacy YAML keys without assigning them invented biological behavior or changing numerical parameter values. Document actual population and radius controls.
- Record realized populations, effective rates/multipliers, role scheduling, larval demand and ignored keys in experiment manifests. Remove the fixed hunger-plot upper bound so corrected dynamics are not hidden.

Validation: all 106 tests pass; seeded smoke checks pass; a three-step generated TSP run passes geometry, capacity, finite-state and event-total checks in isolated output. The current seeded generated nest realizes rates 0.3/0.6/0.9 and total larval demand 17.7 per step. This is a configuration-behavior correction, not empirical calibration, and may change hunger strongly. Details are in docs/configuration.md. No new 500-step experiment was run for this checkpoint.

## Required next steps before interpreting algorithm performance

1. Run a short generated-nest checkpoint with the newly effective configuration and inspect hunger versus food supply. Do not tune parameters just to recover historical graph shapes.
2. Establish empirical parameter calibration and the real-nest/field-statistics comparison with the advisor. `run_experiment.py` records a seed, shared initial instance, source snapshot, dependencies and separate output folders; it currently runs one seed per invocation. Previous 500-step plots predate these audits.
3. Use multiple independent seeds and controlled historical replays for performance comparisons. Current single-seed plots are exploratory.

## Step 11: successful-food-event journal

- Add a separate `food_events_v1` CSV for every successful collection, adult transfer and larval feed. Preserve actual amounts, actor/recipient identities, event-time roles/positions, collection source and deterministic within-step ordering.
- Capture actions immediately rather than recovering them from the last legacy event string or end-of-step state. Do not alter existing state/aggregate column meanings; larval `food_received` remains food storage.
- Scope event capture to adult action execution and restore the prior sink even on exceptions. No decision rule or RNG call was added. The new inactive sink field is null in initial-instance snapshots.
- Reconcile journal counts against every step's nest counters in smoke and experiment runners; existing checks reconcile nest counts with aggregate totals. Seeded smoke replay also hashes event CSVs.
- Document schema and interpretation limits in food_event_log.md. Biological bouts, food provenance, real nest maps, physical units and field validation are not implemented by this change.

Validation: all 115 tests pass. Nine new tests cover lossless multi-recipient logging, idle/no-op steps, event-time snapshots, partial transfer/refill quantities, within-step ordering, sink restoration, missing-event detection and five-mode logging-on/off state/counter/RNG equivalence. All five seeded smoke modes pass with identical replay logs and unchanged baseline logs.

A direct comparison against pre-journal commit `54555c3` in isolated source directories, with seed 20261108 and PYTHONHASHSEED=0, produced byte-identical adult, larval and nest CSVs for all five modes over 20 steps. This checks the journal addition, not the historical TSP bug.

The generated-nest 50-step checkpoint completed all five modes and generated 19 PNG figures at `exports/experiments/seed_20261108_20261108T075309_742555Z`. All 15 state CSVs are byte-identical to the pre-journal 50-step run `seed_20261108_20261108T051007_371772Z`. Initial JSON differs only by 30 added inactive `_food_event_sink: null` adult fields. The checkpoint's source snapshot records uncommitted implementation files, so its captured hashes, rather than its pre-change HEAD alone, identify the code used.

The short single-seed experiment remains software validation. It does not establish policy rankings, empirical calibration or equivalence to the revised biological manuscript. The outstanding research inputs and measurement gaps are in research_alignment.md, empirical_data_inventory.md and field_measurement_audit.md.
