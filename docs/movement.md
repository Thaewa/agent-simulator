# Movement and geometry checkpoint

The movement model is a discrete, sequential lattice update, not continuous simultaneous motion. Adults are processed in their existing list order. These choices are explicit model assumptions and must accompany comparisons with other policies.

- Each axis displacement must be in `oneStepList` (currently -0.5, 0, 0.5). Greedy already uses half-cell gradient-sign displacements; random walk and TSP use the same lattice.
- A proposed destination must exist in the grid. The old membership method incorrectly required more than one matching row.
- Adult destinations cannot be occupied by another adult at that point in the sequential update. This prevents direct swaps and ensures the first arrival wins a contested free destination. No optimistic reservation of a future-vacated cell is made.
- Adults may occupy a larval cell to feed; larvae are not movement obstacles. Crossing diagonal line segments are allowed: exclusion applies to cells at discrete updates, not to continuous swept paths.
- An explicit zero step stays in place without random draws. Invalid actions receive up to maxStepTrials random fallback proposals. If no proposal is valid, the agent stays at its valid, unoccupied origin.
- Non-finite actions and invalid/overlapping adult starting cells raise errors instead of silently propagating invalid state.
- Feeder positions lie in the closed nest disk, matching `in_circle` and larval boundary placement. Foragers can leave this disk but must remain on the grid.
- TSP feeder graphs contain only valid nest cells. Masked graph cells are excluded as nodes, and exterior larval targets are not planned for by feeders.
- Default grids cover agents and forage sites at movement resolution. Previously a manually constructed Simulator could create an integer grid while wasps moved in half cells.
- Generated inner and outer nest regions are disjoint. The inner boundary belongs to the inner region; outer samples lie strictly outside it. Independent sampling within overlapping regions previously allowed duplicate larvae or adults at a boundary coordinate.
- Agent IDs cannot be reused to overwrite movement history, even after an agent is removed. Adult-larva co-location remains valid because their identities and roles differ.

Smoke checks validate after each adult move that adult positions are unique and on-grid and that feeders stay within the nest. Generated-nest runs validate these properties after each completed step. Neither check is a proof for all possible configurations.

The disjoint region correction changes the sampling pool. Reusing a seed across the old and new initialization rules does not establish identical initial maps; use and compare saved initial instances for controlled replays. Historical source snapshots and outputs remain unchanged.

Fixed sequential priority can bias access to crowded cells. Evaluating alternative update orders is a future robustness experiment, not an assertion that this update order is biologically correct.
