"""Immutable hand observations shared by recognition and gesture estimation."""

from __future__ import annotations

from dataclasses import dataclass

PALM_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Classification:
    label: str
    score: float


@dataclass(frozen=True)
class HandObservation:
    """Both coordinate representations and classifications of one detected hand."""

    image_landmarks: tuple[Landmark, ...]
    world_landmarks: tuple[Landmark, ...]
    poses: tuple[Classification, ...]
    handedness: Classification | None
