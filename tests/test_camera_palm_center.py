"""Camera-space palm center estimation from image landmarks."""

from __future__ import annotations

import unittest
from math import sqrt

from camera_geometry import CameraGeometry
from hand_observation import HandObservation, Landmark
from palm_metrics_estimator import PalmMetricsEstimator


def offset_hand() -> HandObservation:
    image = [Landmark(0.6, 0.4, 0.0) for _ in range(21)]
    image[5] = Landmark(0.5, 0.4, 0.0)
    image[17] = Landmark(0.7, 0.4, 0.0)
    landmarks = tuple(image)
    return HandObservation(landmarks, landmarks, (), None)


def rotated_hand() -> HandObservation:
    image = list(offset_hand().image_landmarks)
    image[5] = Landmark(0.55, 0.4, 0.0)
    image[17] = Landmark(0.65, 0.4, 0.0)
    world = list(image)
    world[5] = Landmark(0.0, 0.0, 0.0)
    world[17] = Landmark(1.0, 0.0, sqrt(3.0))
    return HandObservation(tuple(image), tuple(world), (), None)


class CameraPalmCenterTests(unittest.TestCase):
    def test_calibrated_intrinsics_set_depth_and_off_center_position(self) -> None:
        geometry = CameraGeometry((1000, 1000), 700, 800, 400, 450)
        estimator = PalmMetricsEstimator(camera_geometry=geometry)
        estimator.feed(0, (offset_hand(),))

        center = estimator.metrics().palm_center_m
        self.assertIsNotNone(center)
        if center is not None:
            self.assertAlmostEqual(center[0], 0.08)
            self.assertAlmostEqual(center[1], -0.0175)
            self.assertAlmostEqual(center[2], 0.28)

    def test_back_projects_palm_center_using_assumed_width_and_fov(self) -> None:
        for image_size, expected_y in (((1000, 1000), -0.04), ((640, 480), -0.03)):
            with self.subTest(image_size=image_size):
                estimator = PalmMetricsEstimator(
                    image_size=image_size, palm_width_m=0.08, horizontal_fov_deg=90
                )
                estimator.feed(0, (offset_hand(),))

                center = estimator.metrics().palm_center_m
                self.assertIsNotNone(center)
                if center is not None:
                    self.assertAlmostEqual(center[0], 0.04)
                    self.assertAlmostEqual(center[1], expected_y)
                    self.assertAlmostEqual(center[2], 0.2)

    def test_rotation_adjustment_cancels_knuckle_foreshortening(self) -> None:
        estimator = PalmMetricsEstimator(
            image_size=(1000, 1000), palm_width_m=0.08, horizontal_fov_deg=90
        )
        estimator.feed(0, (rotated_hand(),))

        center = estimator.metrics().palm_center_m
        self.assertIsNotNone(center)
        if center is not None:
            self.assertAlmostEqual(center[0], 0.04)
            self.assertAlmostEqual(center[1], -0.04)
            self.assertAlmostEqual(center[2], 0.2)

    def test_nearly_edge_on_knuckle_span_is_unavailable(self) -> None:
        hand = rotated_hand()
        world = list(hand.world_landmarks)
        world[17] = Landmark(0.1, 0.0, 1.0)
        edge_on = HandObservation(hand.image_landmarks, tuple(world), (), None)
        estimator = PalmMetricsEstimator(image_size=(1000, 1000))
        estimator.feed(0, (edge_on,))

        self.assertIsNone(estimator.metrics().palm_center_m)

    def test_camera_center_requires_image_size_and_single_hand(self) -> None:
        estimator = PalmMetricsEstimator()
        estimator.feed(0, (offset_hand(),))
        self.assertIsNone(estimator.metrics().palm_center_m)

        sized_estimator = PalmMetricsEstimator(image_size=(640, 480))
        sized_estimator.feed(33, (offset_hand(),))
        self.assertIsNotNone(sized_estimator.metrics().palm_center_m)

        sized_estimator.feed(66, ())
        self.assertIsNone(sized_estimator.metrics().palm_center_m)


if __name__ == "__main__":
    unittest.main()
