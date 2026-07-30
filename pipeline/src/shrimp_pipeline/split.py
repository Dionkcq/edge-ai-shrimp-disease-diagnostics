# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import random
from dataclasses import dataclass

SpecimenKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class GroupedSplit:
    train: list[SpecimenKey]
    validation: list[SpecimenKey]
    test: list[SpecimenKey]


def grouped_split(
    keys: list[SpecimenKey],
    seed: int = 20260730,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> GroupedSplit:
    unique = sorted(set(keys))
    rng = random.Random(seed)
    by_class: dict[str, list[SpecimenKey]] = {}
    for key in unique:
        by_class.setdefault(key[0], []).append(key)
    train: list[SpecimenKey] = []
    validation: list[SpecimenKey] = []
    test: list[SpecimenKey] = []
    for values in by_class.values():
        rng.shuffle(values)
        first = round(len(values) * train_fraction)
        second = first + round(len(values) * validation_fraction)
        train.extend(values[:first])
        validation.extend(values[first:second])
        test.extend(values[second:])
    return GroupedSplit(sorted(train), sorted(validation), sorted(test))
