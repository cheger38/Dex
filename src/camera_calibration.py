"""Load intrinsic calibration and correct frames for the hand pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from camera_capture import CAMERA_SIZE, Frame
from camera_geometry import CameraGeometry

FORMAT_VERSION = 1
CALIBRATION_DIRECTORY = Path(__file__).resolve().parent.parent / "data" / "calibrations"
DEFAULT_CALIBRATION_PATH = (
    CALIBRATION_DIRECTORY / f"webcam-{CAMERA_SIZE[0]}x{CAMERA_SIZE[1]}.yaml"
)


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    image_size: tuple[int, int]
    camera_matrix: NDArray[np.float64]
    distortion: NDArray[np.float64]
    rms_error: float
    captured_views: int

    @classmethod
    def load(
        cls, path: Path, *, expected_size: tuple[int, int] | None = None
    ) -> CameraCalibration:
        if not path.is_file():
            raise FileNotFoundError(f"calibration file does not exist: {path}")
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise ValueError(f"could not read calibration file: {path}")
        try:
            version = int(storage.getNode("format_version").real())
            name = storage.getNode("camera_name").string()
            image_size = (
                int(storage.getNode("image_width").real()),
                int(storage.getNode("image_height").real()),
            )
            captured_views = int(storage.getNode("captured_views").real())
            rms_error = float(storage.getNode("rms_reprojection_error").real())
            camera_matrix = storage.getNode("camera_matrix").mat()
            distortion = storage.getNode("distortion_coefficients").mat()
        finally:
            storage.release()

        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported calibration version {version}: {path}")
        if expected_size is not None and image_size != expected_size:
            raise ValueError(
                f"calibration is {image_size[0]}x{image_size[1]}, "
                f"but camera mode is {expected_size[0]}x{expected_size[1]}: {path}"
            )
        if (
            not name
            or min(image_size) <= 0
            or captured_views <= 0
            or not isfinite(rms_error)
            or rms_error < 0
            or camera_matrix is None
            or distortion is None
        ):
            raise ValueError(f"calibration metadata or matrices are invalid: {path}")
        if (
            camera_matrix.shape != (3, 3)
            or distortion.size != 5
            or not np.isfinite(camera_matrix).all()
            or not np.isfinite(distortion).all()
            or camera_matrix[0, 0] <= 0
            or camera_matrix[1, 1] <= 0
            or not np.isclose(camera_matrix[0, 1], 0.0)
            or not np.isclose(camera_matrix[1, 0], 0.0)
            or not np.allclose(camera_matrix[2], (0.0, 0.0, 1.0))
        ):
            raise ValueError(f"calibration matrix or distortion is invalid: {path}")
        return cls(
            name,
            image_size,
            cast(NDArray[np.float64], camera_matrix),
            cast(NDArray[np.float64], distortion),
            rms_error,
            captured_views,
        )


class CameraCorrection:
    """Apply a saved undistortion map and expose geometry for mirrored frames."""

    def __init__(self, calibration: CameraCalibration) -> None:
        self.image_size = calibration.image_size
        matrix = calibration.camera_matrix
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            matrix,
            calibration.distortion,
            None,
            matrix,
            self.image_size,
            cv2.CV_32FC1,
        )
        width, _ = self.image_size
        self.mirrored_geometry = CameraGeometry(
            self.image_size,
            float(matrix[0, 0]),
            float(matrix[1, 1]),
            float(width - 1 - matrix[0, 2]),
            float(matrix[1, 2]),
        )

    def apply(self, frame: Frame) -> Frame:
        if (frame.shape[1], frame.shape[0]) != self.image_size:
            raise ValueError("frame size does not match calibration")
        return cast(
            Frame,
            cv2.remap(frame, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR),
        )
