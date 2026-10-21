"""Behavioral checks for swipe recognition from palm velocity peaks."""

from __future__ import annotations

import unittest

from action_recognizer import ActionRecognizer, Swipe
from palm_metrics_estimator import PalmMetrics


def metrics(
    timestamp_ms: int,
    velocity: tuple[float, float] | None,
    *,
    detected_hands: int = 1,
) -> PalmMetrics:
    return PalmMetrics(
        last_frame_ms=timestamp_ms,
        detected_hands=detected_hands,
        palm_velocity=velocity,
        palm_normal=None,
        palm_center=None,
    )


class SwipeTests(unittest.TestCase):
    def test_emits_peak_on_first_clear_decline(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, speed in (
            (0, 0.0),
            (50, 0.5),
            (100, 0.7),
            (150, 1.0),
            (200, 0.96),  # A small dip does not finish the rising burst.
            (250, 1.2),
            (300, 1.15),
        ):
            recognizer.feed(metrics(timestamp_ms, (speed, 0.0)))
            self.assertEqual(recognizer.events(), ())

        recognizer.feed(metrics(350, (1.08, 0.0)))
        expected = (Swipe(timestamp_ms=350, peak_velocity=(1.2, 0.0)),)
        self.assertEqual(recognizer.events(), expected)
        self.assertEqual(recognizer.events(), expected)

        recognizer.feed(metrics(400, (0.9, 0.0)))
        self.assertEqual(recognizer.events(), ())

    def test_diagonal_peak_preserves_its_vector_without_palm_center(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, velocity in (
            (0, (0.0, 0.0)),
            (50, (0.6, 0.8)),
            (100, (0.9, 1.2)),
        ):
            recognizer.feed(metrics(timestamp_ms, velocity))
            self.assertEqual(recognizer.events(), ())

        recognizer.feed(metrics(150, (0.7, 0.8)))
        self.assertEqual(recognizer.events(), (Swipe(150, (0.9, 1.2)),))

    def test_initial_high_speed_needs_an_observed_rising_edge(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, speed in ((0, 0.8), (50, 1.0), (100, 0.4)):
            recognizer.feed(metrics(timestamp_ms, (speed, 0.0)))
            self.assertEqual(recognizer.events(), ())

        recognizer.feed(metrics(150, (0.8, 0.0)))
        self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(200, (0.65, 0.0)))
        self.assertEqual(recognizer.events(), (Swipe(200, (0.8, 0.0)),))

    def test_motion_below_threshold_does_not_swipe(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, speed in (
            (0, 0.0),
            (50, 0.2),
            (100, 0.5),
            (150, 0.59),
            (200, 0.2),
            (250, 0.0),
        ):
            recognizer.feed(metrics(timestamp_ms, (speed, 0.0)))
            self.assertEqual(recognizer.events(), ())

    def test_second_swipe_requires_rest(self) -> None:
        recognizer = ActionRecognizer()
        for timestamp_ms, speed in (
            (0, 0.0),
            (50, 0.8),
            (100, 1.0),
            (150, 0.8),
        ):
            recognizer.feed(metrics(timestamp_ms, (speed, 0.0)))
        self.assertEqual(recognizer.events(), (Swipe(150, (1.0, 0.0)),))

        for timestamp_ms, speed in ((200, 0.4), (250, 0.9), (300, 0.6)):
            recognizer.feed(metrics(timestamp_ms, (speed, 0.0)))
            self.assertEqual(recognizer.events(), ())

        recognizer.feed(metrics(350, (0.1, 0.0)))
        recognizer.feed(metrics(400, (-0.8, 0.0)))
        self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(450, (-0.6, 0.0)))
        self.assertEqual(recognizer.events(), (Swipe(450, (-0.8, 0.0)),))

    def test_tracking_interruption_discards_unconfirmed_peak(self) -> None:
        interruptions = (
            metrics(150, None),
            metrics(150, None, detected_hands=0),
            metrics(150, None, detected_hands=2),
            metrics(80, (0.5, 0.0)),
            metrics(400, (0.5, 0.0)),
            metrics(150, (float("nan"), 0.0)),
        )
        for interruption in interruptions:
            with self.subTest(interruption=interruption):
                recognizer = ActionRecognizer()
                recognizer.feed(metrics(0, (0.0, 0.0)))
                recognizer.feed(metrics(50, (0.8, 0.0)))
                recognizer.feed(metrics(100, (1.0, 0.0)))
                recognizer.feed(interruption)
                self.assertEqual(recognizer.events(), ())

                assert interruption.last_frame_ms is not None
                next_ms = max(200, interruption.last_frame_ms + 50)
                recognizer.feed(metrics(next_ms, (0.7, 0.0)))
                self.assertEqual(recognizer.events(), ())

    def test_timeout_waits_for_rest_before_another_swipe(self) -> None:
        recognizer = ActionRecognizer()
        recognizer.feed(metrics(0, (0.0, 0.0)))
        for timestamp_ms in range(50, 851, 100):
            recognizer.feed(metrics(timestamp_ms, (0.8, 0.0)))
            self.assertEqual(recognizer.events(), ())

        recognizer.feed(metrics(900, (0.9, 0.0)))
        self.assertEqual(recognizer.events(), ())
        recognizer.feed(metrics(950, (0.1, 0.0)))
        recognizer.feed(metrics(1000, (0.8, 0.0)))
        recognizer.feed(metrics(1050, (0.6, 0.0)))
        self.assertEqual(recognizer.events(), (Swipe(1050, (0.8, 0.0)),))


if __name__ == "__main__":
    unittest.main()
