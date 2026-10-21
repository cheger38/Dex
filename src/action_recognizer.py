"""Interpret palm metrics as semantic action events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from math import hypot, isfinite
from typing import Protocol

from palm_metrics_estimator import MAX_FRAME_GAP_MS, PalmMetrics

SWIPE_START_SPEED = 0.6
SWIPE_STOP_SPEED = 0.15
SWIPE_PEAK_DROP = 0.10
SWIPE_MAX_DURATION_MS = 700
PINCH_START_GAP = 0.45
PINCH_END_GAP = 0.60
PINCH_CONFIRM_MS = 50


class ActionEvent(Protocol):
    """Shared shape for concrete action events added by recognizers."""

    @property
    def timestamp_ms(self) -> int: ...


@dataclass(frozen=True)
class Swipe:
    """A palm speed peak in normalized image coordinates per second."""

    timestamp_ms: int
    peak_velocity: tuple[float, float]


@dataclass(frozen=True)
class PinchStart:
    """The thumb and index finger have closed into a pinch."""

    timestamp_ms: int
    point: tuple[float, float]


@dataclass(frozen=True)
class PinchEnd:
    """The thumb and index finger have visibly separated."""

    timestamp_ms: int
    point: tuple[float, float]


@dataclass(frozen=True)
class PinchLost:
    """Tracking ended before a pinched hand visibly opened."""

    timestamp_ms: int
    point: tuple[float, float]


@dataclass(frozen=True)
class ActivePinch:
    """Current location of a confirmed pinch."""

    timestamp_ms: int
    point: tuple[float, float]


@dataclass
class _SwipePeak:
    start_ms: int
    velocity: tuple[float, float]
    speed: float


class _SwipePhase(Enum):
    UNARMED = auto()
    READY = auto()
    TRACKING_PEAK = auto()
    WAITING_FOR_REST = auto()


class ActionRecognizer:
    """Own action recognition state and expose events from the latest feed."""

    def __init__(self) -> None:
        self._latest_metrics: PalmMetrics | None = None
        self._latest_events: tuple[ActionEvent, ...] = ()
        self._swipe_peak: _SwipePeak | None = None
        self._swipe_phase = _SwipePhase.UNARMED
        self._active_pinch: ActivePinch | None = None
        self._pinch_candidate_ms: int | None = None

    def feed(self, metrics: PalmMetrics) -> None:
        previous = self._latest_metrics
        self._latest_metrics = metrics
        events: list[ActionEvent] = []
        swipe = self._recognize_swipe(metrics, previous)
        if swipe is not None:
            events.append(swipe)
        pinch = self._recognize_pinch(metrics, previous)
        if pinch is not None:
            events.append(pinch)
        self._latest_events = tuple(events)

    def _recognize_swipe(
        self, metrics: PalmMetrics, previous: PalmMetrics | None
    ) -> Swipe | None:
        timestamp_ms = metrics.last_frame_ms
        velocity = metrics.palm_velocity
        if (
            timestamp_ms is None
            or metrics.detected_hands != 1
            or velocity is None
            or not all(isfinite(value) for value in velocity)
            or (
                previous is not None
                and previous.last_frame_ms is not None
                and (
                    timestamp_ms <= previous.last_frame_ms
                    or timestamp_ms - previous.last_frame_ms > MAX_FRAME_GAP_MS
                )
            )
        ):
            self._swipe_peak = None
            self._swipe_phase = _SwipePhase.UNARMED
            return None

        speed = hypot(*velocity)
        if not isfinite(speed):
            self._swipe_peak = None
            self._swipe_phase = _SwipePhase.UNARMED
            return None

        if self._swipe_phase is _SwipePhase.UNARMED:
            if speed < SWIPE_START_SPEED:
                self._swipe_phase = _SwipePhase.READY
            return None

        if self._swipe_phase is _SwipePhase.WAITING_FOR_REST:
            if speed <= SWIPE_STOP_SPEED:
                self._swipe_phase = _SwipePhase.READY
            return None

        if self._swipe_phase is _SwipePhase.READY:
            if speed >= SWIPE_START_SPEED:
                self._swipe_peak = _SwipePeak(timestamp_ms, velocity, speed)
                self._swipe_phase = _SwipePhase.TRACKING_PEAK
            return None

        peak = self._swipe_peak
        if peak is None:
            raise RuntimeError("Tracking a swipe without a peak")
        if timestamp_ms - peak.start_ms > SWIPE_MAX_DURATION_MS:
            self._swipe_peak = None
            self._swipe_phase = (
                _SwipePhase.READY
                if speed <= SWIPE_STOP_SPEED
                else _SwipePhase.WAITING_FOR_REST
            )
            return None

        if speed > peak.speed:
            peak.velocity = velocity
            peak.speed = speed
        elif peak.speed - speed >= SWIPE_PEAK_DROP:
            self._swipe_peak = None
            self._swipe_phase = (
                _SwipePhase.READY
                if speed <= SWIPE_STOP_SPEED
                else _SwipePhase.WAITING_FOR_REST
            )
            return Swipe(timestamp_ms, peak.velocity)
        return None

    def _recognize_pinch(
        self, metrics: PalmMetrics, previous: PalmMetrics | None
    ) -> PinchStart | PinchEnd | PinchLost | None:
        timestamp_ms = metrics.last_frame_ms
        point = metrics.pinch_point
        gap = metrics.pinch_gap
        discontinuous = (
            previous is not None
            and previous.last_frame_ms is not None
            and timestamp_ms is not None
            and (
                timestamp_ms <= previous.last_frame_ms
                or timestamp_ms - previous.last_frame_ms > MAX_FRAME_GAP_MS
                or (
                    previous.track_id is not None
                    and metrics.track_id is not None
                    and previous.track_id != metrics.track_id
                )
            )
        )
        if (
            timestamp_ms is None
            or metrics.detected_hands != 1
            or gap is None
            or not isfinite(gap)
            or gap < 0
            or point is None
            or not all(isfinite(value) for value in point)
            or discontinuous
        ):
            self._pinch_candidate_ms = None
            lost: PinchLost | None = None
            lost_timestamp_ms = timestamp_ms
            if lost_timestamp_ms is None and previous is not None:
                lost_timestamp_ms = previous.last_frame_ms
            if self._active_pinch is not None and lost_timestamp_ms is not None:
                lost = PinchLost(lost_timestamp_ms, self._active_pinch.point)
            self._active_pinch = None
            return lost

        if self._active_pinch is not None:
            self._active_pinch = ActivePinch(timestamp_ms, point)
            if gap < PINCH_END_GAP:
                self._pinch_candidate_ms = None
                return None
            if self._pinch_candidate_ms is None:
                self._pinch_candidate_ms = timestamp_ms
            elif timestamp_ms - self._pinch_candidate_ms >= PINCH_CONFIRM_MS:
                self._active_pinch = None
                self._pinch_candidate_ms = None
                return PinchEnd(timestamp_ms, point)
            return None

        if gap > PINCH_START_GAP:
            self._pinch_candidate_ms = None
            return None
        if self._pinch_candidate_ms is None:
            self._pinch_candidate_ms = timestamp_ms
        elif timestamp_ms - self._pinch_candidate_ms >= PINCH_CONFIRM_MS:
            self._active_pinch = ActivePinch(timestamp_ms, point)
            self._pinch_candidate_ms = None
            return PinchStart(timestamp_ms, point)
        return None

    def events(self) -> tuple[ActionEvent, ...]:
        return self._latest_events

    def active_pinch(self) -> ActivePinch | None:
        return self._active_pinch
