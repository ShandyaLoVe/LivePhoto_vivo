"""Assign source videos to deterministic train, validation, and test splits."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Sequence


def assign_splits(
    source_names: Sequence[str],
    ratios: Mapping[str, float],
    seed: int,
) -> Dict[str, str]:
    """Assign whole source videos using deterministic largest-remainder counts."""
    names = sorted(source_names)
    random.Random(seed).shuffle(names)
    order = ["train", "val", "test"]
    exact = {name: len(names) * float(ratios[name]) for name in order}
    counts = {name: int(exact[name]) for name in order}
    remainder = len(names) - sum(counts.values())
    priority = sorted(
        order,
        key=lambda name: (-(exact[name] - counts[name]), order.index(name)),
    )
    for index in range(remainder):
        counts[priority[index]] += 1

    assignments: Dict[str, str] = {}
    cursor = 0
    for split in order:
        for source in names[cursor:cursor + counts[split]]:
            assignments[source] = split
        cursor += counts[split]
    return assignments
