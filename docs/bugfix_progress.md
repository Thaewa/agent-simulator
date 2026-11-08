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

## Required next steps before interpreting algorithm performance

1. Review feeding-event accounting/probability, collision and metric semantics independently of route construction. A no-recipient feedNearby call currently returns None, which can break feeding-event list extension if an edge-case scene has no feeders; this is pending a separate feeding audit.
2. Rerun generated nests after completing the next audit checkpoint. `run_experiment.py` records a seed, shared initial instance, source snapshot, dependencies and separate output folders; it currently runs one seed per invocation. Previous 500-step plots predate the role fixes.
3. Use multiple independent seeds and controlled historical replays for performance comparisons. Current single-seed plots are exploratory.
