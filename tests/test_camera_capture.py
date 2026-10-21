"""The calibration and preview camera mode must match."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, call, patch

import cv2

from camera_capture import CAMERA_FOURCC_VALUE, CAMERA_FPS, CAMERA_SIZE, open_camera


class CameraCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = Mock()
        self.camera.isOpened.return_value = True
        self.mode = {
            cv2.CAP_PROP_FRAME_WIDTH: CAMERA_SIZE[0],
            cv2.CAP_PROP_FRAME_HEIGHT: CAMERA_SIZE[1],
            cv2.CAP_PROP_FPS: CAMERA_FPS,
            cv2.CAP_PROP_FOURCC: CAMERA_FOURCC_VALUE,
        }
        self.camera.get.side_effect = self.mode.__getitem__

    def test_requests_and_accepts_project_mode(self) -> None:
        with patch("camera_capture.cv2.VideoCapture", return_value=self.camera):
            self.assertIs(open_camera(0), self.camera)
        self.camera.set.assert_has_calls(
            [
                call(cv2.CAP_PROP_FOURCC, self.mode[cv2.CAP_PROP_FOURCC]),
                call(cv2.CAP_PROP_FRAME_WIDTH, 1280),
                call(cv2.CAP_PROP_FRAME_HEIGHT, 720),
                call(cv2.CAP_PROP_FPS, 30.0),
            ]
        )
        self.camera.release.assert_not_called()

    def test_rejects_any_different_negotiated_mode(self) -> None:
        expected_mode = self.mode.copy()
        for property_id, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, 640),
            (cv2.CAP_PROP_FRAME_HEIGHT, 480),
            (cv2.CAP_PROP_FPS, 15),
            (cv2.CAP_PROP_FOURCC, int.from_bytes(b"YUYV", "little")),
        ):
            with self.subTest(property_id=property_id):
                self.mode[property_id] = value
                self.camera.release.reset_mock()
                with (
                    patch("camera_capture.cv2.VideoCapture", return_value=self.camera),
                    self.assertRaisesRegex(
                        RuntimeError, "camera must support MJPG 1280x720"
                    ),
                ):
                    open_camera(0)
                self.camera.release.assert_called_once_with()
                self.mode[property_id] = expected_mode[property_id]


if __name__ == "__main__":
    unittest.main()
