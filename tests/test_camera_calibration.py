"""Calibration loading and frame geometry for the live preview."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np

from calibrate_camera import save_calibration
from camera_calibration import CameraCalibration, CameraCorrection
from camera_capture import CAMERA_SIZE
from preview_hand_observation import load_camera_correction, prepare_camera_frame


class CameraCalibrationTests(unittest.TestCase):
    def test_loaded_intrinsics_match_the_corrected_mirrored_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "webcam-1280x720.yaml"
            matrix = np.array(
                [[800.0, 0.0, 600.0], [0.0, 810.0, 365.0], [0.0, 0.0, 1.0]]
            )
            distortion = np.zeros((1, 5), dtype=np.float64)
            save_calibration(path, "webcam", CAMERA_SIZE, matrix, distortion, 0.4, 20)

            calibration = CameraCalibration.load(path, expected_size=CAMERA_SIZE)
            self.assertEqual(calibration.captured_views, 20)
            with redirect_stdout(io.StringIO()):
                correction = load_camera_correction(path, CAMERA_SIZE)
            self.assertIsNotNone(correction)
            if correction is None:
                return
            geometry = correction.mirrored_geometry
            self.assertEqual(geometry.focal_x_px, 800)
            self.assertEqual(geometry.focal_y_px, 810)
            self.assertEqual(geometry.principal_x_px, 679)
            self.assertEqual(geometry.principal_y_px, 365)
            point = (0.75, 0.4)
            camera_point = geometry.back_project(point, 0.8)
            projected = geometry.project(camera_point)
            self.assertAlmostEqual(projected[0], point[0])
            self.assertAlmostEqual(projected[1], point[1])

            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[100:120, 200:220] = (10, 20, 30)
            np.testing.assert_array_equal(correction.apply(frame), frame)

    def test_correction_precedes_horizontal_flip(self) -> None:
        raw = np.zeros((2, 3, 3), dtype=np.uint8)
        corrected = raw.copy()
        corrected[0, 0] = (1, 2, 3)
        correction = Mock(spec=CameraCorrection)
        correction.apply.return_value = corrected

        result = prepare_camera_frame(raw, correction)

        correction.apply.assert_called_once_with(raw)
        np.testing.assert_array_equal(result[0, 2], (1, 2, 3))
        np.testing.assert_array_equal(prepare_camera_frame(raw, None), cv2.flip(raw, 1))

    def test_other_resolution_and_invalid_matrix_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.yaml"
            matrix = np.eye(3, dtype=np.float64)
            distortion = np.zeros((1, 5), dtype=np.float64)
            save_calibration(path, "webcam", (640, 480), matrix, distortion, 0.4, 15)
            with self.assertRaisesRegex(ValueError, "calibration is 640x480"):
                CameraCalibration.load(path, expected_size=CAMERA_SIZE)

            with redirect_stderr(io.StringIO()) as warning:
                self.assertIsNone(load_camera_correction(path, CAMERA_SIZE))
            self.assertIn("using uncorrected video", warning.getvalue())

            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
            storage.write("format_version", 1)
            storage.write("camera_name", "webcam")
            storage.write("image_width", 1280)
            storage.write("image_height", 720)
            storage.write("captured_views", 15)
            storage.write("rms_reprojection_error", 0.4)
            storage.write("camera_matrix", np.eye(2, dtype=np.float64))
            storage.write("distortion_coefficients", distortion)
            storage.release()
            with self.assertRaisesRegex(ValueError, "matrix or distortion is invalid"):
                CameraCalibration.load(path, expected_size=CAMERA_SIZE)
            path.write_text(
                path.read_text().replace("format_version: 1", "format_version: 2")
            )
            with self.assertRaisesRegex(ValueError, "unsupported calibration version"):
                CameraCalibration.load(path, expected_size=CAMERA_SIZE)

    def test_missing_file_uses_warning_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.yaml"
            with redirect_stderr(io.StringIO()) as warning:
                self.assertIsNone(load_camera_correction(path, CAMERA_SIZE))
            self.assertIn(str(path), warning.getvalue())
            self.assertIn("assumed camera field of view", warning.getvalue())


if __name__ == "__main__":
    unittest.main()
