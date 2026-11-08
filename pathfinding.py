"""Canonical names shared by CLI configuration and agent policy selection."""

PATHFINDING_MODES = (
    "greedy",
    "random_walk",
    "random_walk-biased",
    "tsp",
    "tsp-hamiltonian",
)


def normalize_pathfinding_mode(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Pathfinding mode must be a string")
    mode = value.strip().lower()
    if mode not in PATHFINDING_MODES:
        raise ValueError(f"Unknown pathfinding mode {value!r}; choose from {PATHFINDING_MODES}")
    return mode
