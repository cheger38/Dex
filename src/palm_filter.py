"""Replaceable temporal filtering of the palm landmarks in a hand observation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot, isfinite, pi
from typing import Protocol

from hand_observation import PALM_LANDMARKS, HandObservation, Landmark


class PalmLandmarkFilter(Protocol):
    def apply(self, timestamp_ms: int, hand: HandObservation) -> HandObservation: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class OneEuroSettings:
    min_cutoff_hz: float = 3.0
    derivative_cutoff_hz: float = 1.0
    speed_gain: float = 1.0  # Hz per palm width per second.

    def __post_init__(self) -> None:
        if not isfinite(self.min_cutoff_hz) or self.min_cutoff_hz <= 0:
            raise ValueError("min_cutoff_hz must be finite and positive")
        if not isfinite(self.derivative_cutoff_hz) or self.derivative_cutoff_hz <= 0:
            raise ValueError("derivative_cutoff_hz must be finite and positive")
        if not isfinite(self.speed_gain) or self.speed_gain < 0:
            raise ValueError("speed_gain must be finite and nonnegative")


def _alpha(cutoff_hz: float, seconds: float) -> float:
    angle = 2 * pi * cutoff_hz * seconds
    return angle / (1 + angle)


def _coordinates(point: Landmark) -> tuple[float, float, float]:
    return point.x, point.y, point.z


def _palm_width(points: tuple[Landmark, ...], *, image_space: bool) -> float:
    # Points are stored in PALM_LANDMARKS order: wrist, index, middle, ring, pinky.
    index = points[1]
    pinky = points[4]
    difference = (index.x - pinky.x, index.y - pinky.y, index.z - pinky.z)
    return hypot(*difference[:2]) if image_space else hypot(*difference)


class OneEuroPalmFilter:
    """Smooth palm points at rest and admit faster movement with less lag.

    The speed term uses palm widths per second, so the same settings work for
    normalized image points and world points with different coordinate scales.
    """

    def __init__(self, settings: OneEuroSettings | None = None) -> None:
        self.settings: OneEuroSettings = settings or OneEuroSettings()
        self.reset()

    def reset(self) -> None:
        self._last_timestamp_ms: int | None = None
        self._raw_image: tuple[Landmark, ...] = ()
        self._raw_world: tuple[Landmark, ...] = ()
        self._filtered_image: tuple[Landmark, ...] = ()
        self._filtered_world: tuple[Landmark, ...] = ()
        self._image_derivatives: tuple[tuple[float, float, float], ...] = ()
        self._world_derivatives: tuple[tuple[float, float, float], ...] = ()

    def apply(self, timestamp_ms: int, hand: HandObservation) -> HandObservation:
        image = tuple(hand.image_landmarks[index] for index in PALM_LANDMARKS)
        world = tuple(hand.world_landmarks[index] for index in PALM_LANDMARKS)
        previous_ms = self._last_timestamp_ms
        if previous_ms is None or timestamp_ms <= previous_ms:
            self.reset()
            filtered_image, filtered_world = image, world
            self._image_derivatives = self._world_derivatives = (
                (0.0, 0.0, 0.0),
            ) * len(PALM_LANDMARKS)
        else:
            seconds = (timestamp_ms - previous_ms) / 1_000
            filtered_image, self._image_derivatives = self._filter_points(
                image,
                self._raw_image,
                self._filtered_image,
                self._image_derivatives,
                seconds,
                image_space=True,
            )
            filtered_world, self._world_derivatives = self._filter_points(
                world,
                self._raw_world,
                self._filtered_world,
                self._world_derivatives,
                seconds,
                image_space=False,
            )

        self._last_timestamp_ms = timestamp_ms
        self._raw_image, self._raw_world = image, world
        self._filtered_image, self._filtered_world = filtered_image, filtered_world
        image_points = list(hand.image_landmarks)
        world_points = list(hand.world_landmarks)
        for index, image_point, world_point in zip(
            PALM_LANDMARKS, filtered_image, filtered_world, strict=True
        ):
            image_points[index] = image_point
            world_points[index] = world_point
        return replace(
            hand,
            image_landmarks=tuple(image_points),
            world_landmarks=tuple(world_points),
        )

    def _filter_points(
        self,
        points: tuple[Landmark, ...],
        previous_raw: tuple[Landmark, ...],
        previous_filtered: tuple[Landmark, ...],
        previous_derivatives: tuple[tuple[float, float, float], ...],
        seconds: float,
        *,
        image_space: bool,
    ) -> tuple[tuple[Landmark, ...], tuple[tuple[float, float, float], ...]]:
        derivative_alpha = _alpha(self.settings.derivative_cutoff_hz, seconds)
        width = _palm_width(points, image_space=image_space)
        filtered: list[Landmark] = []
        derivatives: list[tuple[float, float, float]] = []
        for point, raw, prior, prior_derivative in zip(
            points, previous_raw, previous_filtered, previous_derivatives, strict=True
        ):
            current = _coordinates(point)
            old_raw = _coordinates(raw)
            raw_derivative = (
                (current[0] - old_raw[0]) / seconds,
                (current[1] - old_raw[1]) / seconds,
                (current[2] - old_raw[2]) / seconds,
            )
            derivative = (
                prior_derivative[0]
                + derivative_alpha * (raw_derivative[0] - prior_derivative[0]),
                prior_derivative[1]
                + derivative_alpha * (raw_derivative[1] - prior_derivative[1]),
                prior_derivative[2]
                + derivative_alpha * (raw_derivative[2] - prior_derivative[2]),
            )
            speed = hypot(*derivative[:2]) if image_space else hypot(*derivative)
            relative_speed = speed / width if isfinite(width) and width > 1e-6 else 0.0
            cutoff = (
                self.settings.min_cutoff_hz + self.settings.speed_gain * relative_speed
            )
            alpha = _alpha(cutoff, seconds)
            old_filtered = _coordinates(prior)
            filtered.append(
                Landmark(
                    current[0] * alpha + old_filtered[0] * (1 - alpha),
                    current[1] * alpha + old_filtered[1] * (1 - alpha),
                    current[2] * alpha + old_filtered[2] * (1 - alpha),
                )
            )
            derivatives.append(derivative)
        return tuple(filtered), tuple(derivatives)
