"""Offline checks for the interactive single-camera calibrator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from calibrate_camera import (
    BOARD_SIZE,
    DEFAULT_CAPTURE_INTERVAL,
    MIN_CALIBRATION_VIEWS,
    calibration_path,
    calibration_result,
    create_board_points,
    parse_arguments,
    run_session,
    save_calibration,
)
from camera_calibration import CameraCalibration
from camera_capture import CAMERA_SIZE, read_frame


class CalibrationDataTests(unittest.TestCase):
    def test_capture_interval_defaults_to_two_seconds_and_must_be_positive(
        self,
    ) -> None:
        arguments = parse_arguments(["--camera-index", "0", "--camera-name", "desk"])
        self.assertEqual(arguments.capture_interval, DEFAULT_CAPTURE_INTERVAL)
        arguments = parse_arguments(
            [
                "--camera-index",
                "0",
                "--camera-name",
                "desk",
                "--capture-interval",
                "0.5",
            ]
        )
        self.assertEqual(arguments.capture_interval, 0.5)
        for value in ("0", "-1", "nan", "inf"):
            with (
                self.subTest(value=value),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parse_arguments(
                    [
                        "--camera-index",
                        "0",
                        "--camera-name",
                        "desk",
                        "--capture-interval",
                        value,
                    ]
                )

    def test_board_points_follow_fixed_checkerboard(self) -> None:
        points = create_board_points()
        self.assertEqual(points.shape, (BOARD_SIZE[0] * BOARD_SIZE[1], 3))
        np.testing.assert_allclose(points[0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(points[1], (0.023, 0.0, 0.0))
        np.testing.assert_allclose(points[BOARD_SIZE[0]], (0.0, 0.023, 0.0))
        np.testing.assert_allclose(points[-1], (0.184, 0.115, 0.0))

    def test_result_requires_fifteen_views_before_calling_opencv(self) -> None:
        with patch("calibrate_camera.cv2.calibrateCamera") as calibrate:
            with self.assertRaisesRegex(ValueError, "at least 15 views"):
                calibration_result([], (640, 480))
            calibrate.assert_not_called()

            corners = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 1, 2), np.float32)
            matrix = np.eye(3, dtype=np.float64)
            distortion = np.zeros((1, 5), dtype=np.float64)
            calibrate.return_value = (0.2, matrix, distortion, None, None)
            rms, actual_matrix, actual_distortion = calibration_result(
                [corners] * MIN_CALIBRATION_VIEWS, (640, 480)
            )
            self.assertEqual(rms, 0.2)
            self.assertIs(actual_matrix, matrix)
            self.assertIs(actual_distortion, distortion)
            self.assertEqual(len(calibrate.call_args.args[0]), MIN_CALIBRATION_VIEWS)

    def test_yaml_round_trip_and_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrations" / "desk-640x480.yaml"
            matrix = np.array(
                [[600.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]]
            )
            distortion = np.array([[0.1, -0.02, 0.0, 0.0, 0.001]])
            save_calibration(path, "desk", (640, 480), matrix, distortion, 0.23, 15)
            self.assertTrue(path.is_file())
            saved = CameraCalibration.load(path)
            self.assertEqual(saved.image_size, (640, 480))
            np.testing.assert_allclose(saved.camera_matrix, matrix)
            np.testing.assert_allclose(saved.distortion, distortion)
            self.assertAlmostEqual(saved.rms_error, 0.23)

            with self.assertRaises(FileExistsError):
                save_calibration(path, "desk", (640, 480), matrix, distortion, 0.5, 15)
            self.assertAlmostEqual(CameraCalibration.load(path).rms_error, 0.23)
            save_calibration(
                path, "desk", (640, 480), matrix, distortion, 0.5, 15, force=True
            )
            self.assertAlmostEqual(CameraCalibration.load(path).rms_error, 0.5)

    def test_path_includes_project_frame_size(self) -> None:
        self.assertEqual(
            calibration_path("desk", CAMERA_SIZE).name, "desk-1280x720.yaml"
        )


class SessionTests(unittest.TestCase):
    def test_timed_attempts_skip_frames_without_a_board(self) -> None:
        camera = Mock()
        frame = np.zeros((720, 1280, 3), np.uint8)
        camera.read.return_value = (True, frame)
        corners = np.ones((BOARD_SIZE[0] * BOARD_SIZE[1], 1, 2), np.float32)
        detected = [None, corners, None, corners, corners, corners, corners]
        with (
            patch(
                "calibrate_camera.time.monotonic", side_effect=[0, 0, 1, 2, 3, 4, 5, 6]
            ),
            patch("calibrate_camera.capture_preview") as preview,
            patch("calibrate_camera.cv2.imshow"),
            patch("calibrate_camera.cv2.waitKey", side_effect=[255] * 6 + [ord("q")]),
        ):
            preview.side_effect = [(frame, board) for board in detected]
            run_session(
                camera,
                frame,
                Path("unused.yaml"),
                "desk",
                force=False,
                capture_interval=2,
            )
        self.assertEqual(
            [call.args[1] for call in preview.call_args_list], [0, 0, 0, 0, 0, 1, 1]
        )

    def test_pause_resume_and_reset_restart_the_timer(self) -> None:
        camera = Mock()
        frame = np.zeros((720, 1280, 3), np.uint8)
        camera.read.return_value = (True, frame)
        corners = np.ones((BOARD_SIZE[0] * BOARD_SIZE[1], 1, 2), np.float32)
        keys = [ord(" "), 255, ord(" "), 255, 255, ord("r"), 255, 255, ord("q")]
        with (
            patch(
                "calibrate_camera.time.monotonic",
                side_effect=[0, 0, 10, 11, 12, 13, 14, 15, 16, 17],
            ),
            patch(
                "calibrate_camera.capture_preview", return_value=(frame, corners)
            ) as preview,
            patch("calibrate_camera.cv2.imshow"),
            patch("calibrate_camera.cv2.waitKey", side_effect=keys),
        ):
            run_session(
                camera,
                frame,
                Path("unused.yaml"),
                "desk",
                force=False,
                capture_interval=2,
            )
        self.assertEqual(
            [call.args[1] for call in preview.call_args_list],
            [0, 0, 0, 0, 0, 1, 0, 0, 1],
        )
        self.assertEqual(
            [call.kwargs["paused"] for call in preview.call_args_list],
            [False, True, True, False, False, False, False, False, False],
        )

    def test_read_frame_rejects_resolution_changes(self) -> None:
        camera = Mock()
        camera.read.return_value = (True, np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(read_frame(camera).shape, (720, 1280, 3))
        camera.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        with self.assertRaisesRegex(RuntimeError, "changed from 1280x720 to 640x480"):
            read_frame(camera)

    def test_insufficient_views_never_save(self) -> None:
        camera = Mock()
        frame = np.zeros((720, 1280, 3), np.uint8)
        camera.read.return_value = (True, frame)
        with (
            patch("calibrate_camera.capture_preview", return_value=(frame, None)),
            patch("calibrate_camera.cv2.imshow"),
            patch("calibrate_camera.cv2.waitKey", side_effect=[ord("c"), ord("q")]),
            patch("calibrate_camera.calibration_result") as calibrate,
            patch("calibrate_camera.save_calibration") as save,
        ):
            run_session(camera, frame, Path("unused.yaml"), "desk", force=False)
        calibrate.assert_not_called()
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
