"""Compare a run against a stored baseline and fail on regressions."""
from __future__ import annotations

# metric -> (direction, allowed drop). "up" = higher is better.
GATES = {
    "correctness": ("up", 0.02),
    "refusal_accuracy": ("up", 0.02),
    "groundedness": ("up", 0.02),
    "retrieval_hit_rate": ("up", 0.02),
    "hallucination_rate": ("down", 0.02),
}


def compare(current: dict, baseline: dict, gates: dict = GATES) -> list[str]:
    failures = []
    for metric, (direction, tol) in gates.items():
        cur, base = current.get(metric), baseline.get(metric)
        if cur is None or base is None:
            continue
        if direction == "up" and cur < base - tol:
            failures.append(f"{metric} dropped: {base:.3f} -> {cur:.3f}")
        if direction == "down" and cur > base + tol:
            failures.append(f"{metric} rose: {base:.3f} -> {cur:.3f}")
    return failures
