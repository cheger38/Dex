"""Pinch measurements and lifecycle event behavior."""

from __future__ import annotations

import unittest
from dataclasses import replace

from action_recognizer import (
    PINCH_END_GAP,
    PINCH_START_GAP,
    ActionRecognizer,
    ActivePinch,
    PinchEnd,
    PinchLost,
    PinchStart,
    Swipe,
)
from hand_observation import HandObservation, Landmark
from palm_metrics_estimator import PalmMetrics, PalmMetricsEstimator

CLOSED_GAP = PINCH_START_GAP * 0.8
BETWEEN_GAPS = (PINCH_START_GAP + PINCH_END_GAP) / 2
OPEN_GAP = PINCH_END_GAP + 0.05


def hand_with_gap(gap: float, *, offset: float = 0.0) -> HandObservation:
    points = [Landmark(0.1 + offset, 0.1, 0.0) for _ in range(21)]
    points[0] = Landmark(0.5 + offset, 0.7, 0.0)
    points[5] = Landmark(0.4 + offset, 0.5, 0.0)
    points[9] = Landmark(0.5 + offset, 0.45, 0.0)
    points[13] = Landmark(0.55 + offset, 0.5, 0.0)
    points[17] = Landmark(0.6 + offset, 0.5, 0.0)
    points[4] = Landmark(0.5 + offset, 0.3, 0.0)
    points[8] = Landmark(0.5 + offset + 0.2 * gap, 0.3, 0.0)
    landmarks = tuple(points)
    return HandObservation(landmarks, landmarks, (), None)


def metrics(
    timestamp_ms: int,
    gap: float | None,
    *,
    velocity: tuple[float, float] | None = None,
    hands: int = 1,
    track_id: int = 1,
) -> PalmMetrics:
    return PalmMetrics(
        last_frame_ms=timestamp_ms,
        detected_hands=hands,
        palm_velocity=velocity,
        palm_normal=None,
        pinch_gap=gap,
        pinch_point=(0.52, 0.3) if gap is not None else None,
        track_id=track_id if hands == 1 else None,
    )


class PinchMetricTests(unittest.TestCase):
    def test_gap_and_midpoint_use_image_landmarks_and_palm_width(self) -> None:
        estimator = PalmMetricsEstimator()
        estimator.feed(0, (hand_with_gap(0.2),))
        measured = estimator.metrics()
        self.assertAlmostEqual(measured.pinch_gap or 0, 0.2)
        self.assertIsNotNone(measured.pinch_point)
        if measured.pinch_point is not None:
            self.assertAlmostEqual(measured.pinch_point[0], 0.52)
            self.assertAlmostEqual(measured.pinch_point[1], 0.3)

        estimator.feed(33, ())
        self.assertIsNone(estimator.metrics().pinch_gap)
        self.assertIsNone(estimator.metrics().pinch_point)

    def test_new_hand_track_gets_new_identifier(self) -> None:
        estimator = PalmMetricsEstimator()
        estimator.feed(0, (hand_with_gap(0.2),))
        original_id = estimator.metrics().track_id
        estimator.feed(33, (hand_with_gap(0.2, offset=0.4),))
        self.assertNotEqual(estimator.metrics().track_id, original_id)


class PinchActionTests(unittest.TestCase):
    def test_active_pinch_snapshot_tracks_location_only_while_confirmed(self) -> None:
        recognizer = ActionRecognizer()
        recognizer.feed(metrics(0, CLOSED_GAP))
        self.assertIsNone(recognizer.active_pinch())
        recognizer.feed(metrics(50, CLOSED_GAP))
        self.assertEqual(recognizer.active_pinch(), ActivePinch(50, (0.52, 0.3)))

        recognizer.feed(replace(metrics(80, CLOSED_GAP), pinch_point=(0.7, 0.4)))
        self.assertEqual(recognizer.events(), ())
        self.assertEqual(recognizer.active_pinch(), ActivePinch(80, (0.7, 0.4)))
        recognizer.feed(metrics(100, OPEN_GAP))
        self.assertIsNotNone(recognizer.active_pinch())
        recognizer.feed(metrics(150, OPEN_GAP))
        self.assertEqual(recognizer.events(), (PinchEnd(150, (0.52, 0.3)),))
        self.assertIsNone(recognizer.active_pinch())

    def test_initial_closed_hand_starts_and_visible_opening_ends(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, gap in (
            (0, CLOSED_GAP),
            (30, CLOSED_GAP * 0.9),
            (50, CLOSED_GAP),
        ):
            recognizer.feed(metrics(timestamp_ms, gap))
        self.assertEqual(recognizer.events(), (PinchStart(50, (0.52, 0.3)),))

        for timestamp_ms, gap in (
            (80, BETWEEN_GAPS),
            (100, OPEN_GAP),
            (130, BETWEEN_GAPS),
            (160, OPEN_GAP),
        ):
            recognizer.feed(metrics(timestamp_ms, gap))
            self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(210, OPEN_GAP))
        self.assertEqual(recognizer.events(), (PinchEnd(210, (0.52, 0.3)),))

    def test_start_candidate_resets_when_gap_reopens(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, gap in (
            (0, CLOSED_GAP),
            (30, PINCH_START_GAP + 0.01),
            (60, CLOSED_GAP),
            (100, CLOSED_GAP),
        ):
            recognizer.feed(metrics(timestamp_ms, gap))
            self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(110, CLOSED_GAP))
        self.assertEqual(recognizer.events(), (PinchStart(110, (0.52, 0.3)),))

    def test_tracking_loss_emits_lost_and_reacquisition_can_start_again(self) -> None:
        recognizer = ActionRecognizer()
        recognizer.feed(metrics(0, CLOSED_GAP))
        recognizer.feed(metrics(50, CLOSED_GAP))
        recognizer.feed(metrics(80, None, hands=0))
        self.assertEqual(recognizer.events(), (PinchLost(80, (0.52, 0.3)),))
        self.assertIsNone(recognizer.active_pinch())
        recognizer.feed(metrics(110, CLOSED_GAP, track_id=2))
        self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(160, CLOSED_GAP, track_id=2))
        self.assertEqual(recognizer.events(), (PinchStart(160, (0.52, 0.3)),))

    def test_track_change_and_long_gap_lose_active_pinch(self) -> None:
        for interruption in (
            replace(metrics(80, CLOSED_GAP, track_id=2), pinch_point=(0.8, 0.3)),
            metrics(400, CLOSED_GAP),
            metrics(80, None, hands=2),
        ):
            with self.subTest(interruption=interruption):
                recognizer = ActionRecognizer()
                recognizer.feed(metrics(0, CLOSED_GAP))
                recognizer.feed(metrics(50, CLOSED_GAP))
                recognizer.feed(interruption)
                self.assertEqual(
                    recognizer.events(),
                    (PinchLost(interruption.last_frame_ms or 0, (0.52, 0.3)),),
                )

    def test_pinch_does_not_require_velocity_and_can_coincide_with_swipe(self) -> None:
        recognizer = ActionRecognizer()
        recognizer.feed(metrics(0, CLOSED_GAP))
        recognizer.feed(metrics(50, CLOSED_GAP))
        self.assertEqual(recognizer.events(), (PinchStart(50, (0.52, 0.3)),))

        recognizer = ActionRecognizer()
        recognizer.feed(metrics(0, OPEN_GAP, velocity=(0.0, 0.0)))
        recognizer.feed(metrics(50, CLOSED_GAP, velocity=(0.8, 0.0)))
        recognizer.feed(metrics(100, CLOSED_GAP, velocity=(0.6, 0.0)))
        self.assertEqual(
            recognizer.events(),
            (Swipe(100, (0.8, 0.0)), PinchStart(100, (0.52, 0.3))),
        )


if __name__ == "__main__":
    unittest.main()
