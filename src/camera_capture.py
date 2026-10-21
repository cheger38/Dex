"""Open and read the camera mode shared by calibration and live preview."""

from __future__ import annotations

import sys
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

CAMERA_SIZE = (1280, 720)
CAMERA_FPS = 30.0
CAMERA_FPS_TOLERANCE = 0.5
CAMERA_FOURCC = "MJPG"
CAMERA_FOURCC_VALUE = int.from_bytes(CAMERA_FOURCC.encode("ascii"), "little")

Frame = NDArray[np.uint8]


def _fourcc_text(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))


def open_camera(index: int) -> cv2.VideoCapture:
    """Request the project mode and reject a different negotiated mode."""
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"could not open camera index {index}")

    camera.set(cv2.CAP_PROP_FOURCC, CAMERA_FOURCC_VALUE)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    actual_size = (
        round(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    actual_fps = float(camera.get(cv2.CAP_PROP_FPS))
    actual_fourcc = round(camera.get(cv2.CAP_PROP_FOURCC))
    print(
        f"Camera {index}: {actual_size[0]}x{actual_size[1]} @ {actual_fps:g} FPS, "
        f"format {_fourcc_text(actual_fourcc)}"
    )

    if (
        actual_size != CAMERA_SIZE
        or abs(actual_fps - CAMERA_FPS) > CAMERA_FPS_TOLERANCE
        or actual_fourcc != CAMERA_FOURCC_VALUE
    ):
        camera.release()
        raise RuntimeError(
            f"camera must support {CAMERA_FOURCC} {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]} "
            f"at {CAMERA_FPS:g} FPS"
        )
    return camera


def read_frame(camera: cv2.VideoCapture) -> Frame:
    """Read a color frame and ensure the camera has kept the project mode."""
    success, frame = camera.read()
    if not success or frame is None:
        raise RuntimeError("could not read a camera frame")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise RuntimeError("camera did not return a three-channel 8-bit color frame")
    actual_size = (frame.shape[1], frame.shape[0])
    if actual_size != CAMERA_SIZE:
        raise RuntimeError(
            f"camera frame changed from {CAMERA_SIZE[0]}x{CAMERA_SIZE[1]} "
            f"to {actual_size[0]}x{actual_size[1]}"
        )
    return cast(Frame, frame)
