"""Behavioral checks for palm smoothing and the estimator's filter boundary."""

from __future__ import annotations

import unittest
from dataclasses import replace
from itertools import pairwise

from hand_observation import PALM_LANDMARKS, Classification, HandObservation, Landmark
from palm_filter import OneEuroPalmFilter, OneEuroSettings
from palm_metrics_estimator import PalmMetricsEstimator


def hand_at(x: float = 0.5, *, label: str = "Right") -> HandObservation:
    image = [Landmark(0.1, 0.1, 0.0) for _ in range(21)]
    world = [Landmark(0.1, 0.1, 0.0) for _ in range(21)]
    for index, offset_x, y in (
        (0, 0.0, 0.7),
        (5, -0.07, 0.5),
        (9, -0.01, 0.46),
        (13, 0.05, 0.48),
        (17, 0.11, 0.53),
    ):
        image[index] = Landmark(x + offset_x, y, 0.01 * index)
        world[index] = Landmark(x + offset_x, y, 0.01 * index)
    return HandObservation(tuple(image), tuple(world), (), Classification(label, 0.99))


def resize_palm(hand: HandObservation, scale: float) -> HandObservation:
    image = list(hand.image_landmarks)
    for index in PALM_LANDMARKS:
        point = image[index]
        image[index] = replace(
            point,
            x=0.5 + scale * (point.x - 0.5),
            y=0.5 + scale * (point.y - 0.5),
        )
    return replace(hand, image_landmarks=tuple(image))


class FreezePalmFilter:
    """Test replacement that holds palm x at the first observed position."""

    def __init__(self) -> None:
        self.resets = 0
        self.first_x: float | None = None

    def reset(self) -> None:
        self.resets += 1
        self.first_x = None

    def apply(self, timestamp_ms: int, hand: HandObservation) -> HandObservation:
        del timestamp_ms
        if self.first_x is None:
            self.first_x = hand.image_landmarks[0].x
        shift = self.first_x - hand.image_landmarks[0].x
        image = list(hand.image_landmarks)
        for index in PALM_LANDMARKS:
            image[index] = replace(image[index], x=image[index].x + shift)
        return replace(hand, image_landmarks=tuple(image))


class PalmFilterTests(unittest.TestCase):
    def test_only_palm_points_change_and_input_is_immutable(self) -> None:
        palm_filter = OneEuroPalmFilter()
        first = hand_at()
        second = hand_at(0.54)
        palm_filter.apply(0, first)
        result = palm_filter.apply(33, second)

        self.assertLess(result.image_landmarks[0].x, second.image_landmarks[0].x)
        self.assertLess(result.world_landmarks[0].x, second.world_landmarks[0].x)
        self.assertEqual(result.image_landmarks[8], second.image_landmarks[8])
        self.assertEqual(result.world_landmarks[8], second.world_landmarks[8])
        self.assertEqual(second.image_landmarks[0].x, 0.54)
        self.assertIs(result.handedness, second.handedness)

    def test_still_hand_jitter_is_reduced(self) -> None:
        palm_filter = OneEuroPalmFilter()
        filtered = [palm_filter.apply(0, hand_at()).image_landmarks[0].x]
        raw = [0.5]
        for frame in range(1, 21):
            x = 0.5 + (0.01 if frame % 2 else -0.01)
            raw.append(x)
            filtered.append(
                palm_filter.apply(frame * 33, hand_at(x)).image_landmarks[0].x
            )

        raw_travel = sum(abs(b - a) for a, b in pairwise(raw))
        filtered_travel = sum(abs(b - a) for a, b in pairwise(filtered))
        self.assertLess(filtered_travel, raw_travel * 0.6)

    def test_adaptive_filter_responds_faster_than_fixed_cutoff(self) -> None:
        adaptive = OneEuroPalmFilter()
        fixed = OneEuroPalmFilter(OneEuroSettings(speed_gain=0.0))
        for palm_filter in (adaptive, fixed):
            palm_filter.apply(0, hand_at())
        moving = hand_at(0.55)
        adaptive_x = adaptive.apply(33, moving).image_landmarks[0].x
        fixed_x = fixed.apply(33, moving).image_landmarks[0].x
        self.assertGreater(adaptive_x, fixed_x)
        self.assertLess(adaptive_x, moving.image_landmarks[0].x)

    def test_settings_reject_invalid_cutoffs(self) -> None:
        with self.assertRaises(ValueError):
            OneEuroSettings(min_cutoff_hz=0)
        with self.assertRaises(ValueError):
            OneEuroSettings(derivative_cutoff_hz=float("nan"))
        with self.assertRaises(ValueError):
            OneEuroSettings(speed_gain=-1)


class EstimatorFilterTests(unittest.TestCase):
    def test_palm_center_uses_filtered_points_and_requires_one_hand(self) -> None:
        filtered = PalmMetricsEstimator(palm_filter=FreezePalmFilter())
        raw = PalmMetricsEstimator()
        for timestamp_ms, hand in ((0, hand_at()), (33, hand_at(0.54))):
            filtered.feed(timestamp_ms, (hand,))
            raw.feed(timestamp_ms, (hand,))

        filtered_center = filtered.metrics().palm_center
        raw_center = raw.metrics().palm_center
        self.assertIsNotNone(filtered_center)
        self.assertIsNotNone(raw_center)
        if filtered_center is not None and raw_center is not None:
            self.assertLess(filtered_center[0], raw_center[0])
            self.assertAlmostEqual(filtered_center[1], raw_center[1])

        filtered.feed(66, ())
        self.assertIsNone(filtered.metrics().palm_center)

    def test_injected_filter_changes_estimate_and_none_bypasses_it(self) -> None:
        frozen = PalmMetricsEstimator(palm_filter=FreezePalmFilter())
        raw = PalmMetricsEstimator(palm_filter=None)
        for timestamp_ms, x in ((0, 0.5), (60, 0.52), (120, 0.54)):
            hand = hand_at(x)
            frozen.feed(timestamp_ms, (hand,))
            raw.feed(timestamp_ms, (hand,))

        self.assertEqual(frozen.metrics().palm_velocity, (0.0, 0.0))
        raw_velocity = raw.metrics().palm_velocity
        self.assertIsNotNone(raw_velocity)
        if raw_velocity is not None:
            self.assertGreater(raw_velocity[0], 0)

    def test_filtered_palm_size_feeds_approach_rate(self) -> None:
        filtered = PalmMetricsEstimator(palm_filter=OneEuroPalmFilter())
        raw = PalmMetricsEstimator()
        base = hand_at()
        for timestamp_ms, scale in ((0, 1.0), (60, 1.05), (120, 1.1)):
            hand = resize_palm(base, scale)
            filtered.feed(timestamp_ms, (hand,))
            raw.feed(timestamp_ms, (hand,))

        filtered_rate = filtered.metrics().approach_rate
        raw_rate = raw.metrics().approach_rate
        self.assertIsNotNone(filtered_rate)
        self.assertIsNotNone(raw_rate)
        if filtered_rate is not None and raw_rate is not None:
            self.assertLess(raw_rate, filtered_rate)
            self.assertLess(filtered_rate, 0)

    def test_filtered_world_points_feed_palm_normal(self) -> None:
        filtered = PalmMetricsEstimator(palm_filter=OneEuroPalmFilter())
        raw = PalmMetricsEstimator()
        base = hand_at()
        for estimator in (filtered, raw):
            estimator.feed(0, (base,))

        world = list(base.world_landmarks)
        world[5] = replace(world[5], z=world[5].z + 0.08)
        tilted = replace(base, world_landmarks=tuple(world))
        for estimator in (filtered, raw):
            estimator.feed(33, (tilted,))

        original_normal = PalmMetricsEstimator()
        original_normal.feed(0, (base,))
        base_value = original_normal.metrics().palm_normal
        filtered_value = filtered.metrics().palm_normal
        raw_value = raw.metrics().palm_normal
        self.assertIsNotNone(base_value)
        self.assertIsNotNone(filtered_value)
        self.assertIsNotNone(raw_value)
        if (
            base_value is not None
            and filtered_value is not None
            and raw_value is not None
        ):
            self.assertLess(
                abs(filtered_value[0] - base_value[0]),
                abs(raw_value[0] - base_value[0]),
            )

    def test_tracking_breaks_reset_filter_and_motion(self) -> None:
        cases = (
            ("no hand", 130, ()),
            ("two hands", 130, (hand_at(0.54), hand_at(0.6))),
            ("backward time", 90, (hand_at(0.54),)),
            ("long gap", 400, (hand_at(0.54),)),
            ("large jump", 130, (hand_at(0.9),)),
            ("handedness change", 130, (hand_at(0.54, label="Left"),)),
        )
        for name, timestamp_ms, hands in cases:
            with self.subTest(name=name):
                palm_filter = FreezePalmFilter()
                estimator = PalmMetricsEstimator(palm_filter=palm_filter)
                estimator.feed(0, (hand_at(),))
                estimator.feed(100, (hand_at(0.52),))
                resets_before = palm_filter.resets
                estimator.feed(timestamp_ms, hands)
                self.assertEqual(palm_filter.resets, resets_before + 1)
                self.assertIsNone(estimator.metrics().palm_velocity)


if __name__ == "__main__":
    unittest.main()
