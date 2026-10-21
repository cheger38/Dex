"""Interactively calibrate one camera with a checkerboard."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import cast

# OpenCV's Qt build needs XWayland when both Wayland and an X display exist.
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from numpy.typing import NDArray

from camera_calibration import CALIBRATION_DIRECTORY, FORMAT_VERSION, CameraCalibration
from camera_capture import CAMERA_SIZE, Frame, open_camera, read_frame

BOARD_SIZE = (9, 6)  # Inner corners of a 10-by-7-square checkerboard.
SQUARE_SIZE_M = 0.023
MIN_CALIBRATION_VIEWS = 15
DEFAULT_CAPTURE_INTERVAL = 2.0
CAMERA_WARMUP_FRAMES = 20
WINDOW_NAME = "Dex camera calibration"

Corners = NDArray[np.float32]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    def camera_index(value: str) -> int:
        try:
            index = int(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "camera index must be an integer"
            ) from error
        if index < 0:
            raise argparse.ArgumentTypeError("camera index must be non-negative")
        return index

    def camera_name(value: str) -> str:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value) is None:
            raise argparse.ArgumentTypeError(
                "camera name must use lowercase letters, numbers, hyphens, or underscores"
            )
        return value

    def capture_interval(value: str) -> float:
        try:
            seconds = float(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "capture interval must be a positive number of seconds"
            ) from error
        if not math.isfinite(seconds) or seconds <= 0:
            raise argparse.ArgumentTypeError(
                "capture interval must be a positive number of seconds"
            )
        return seconds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=camera_index, required=True)
    parser.add_argument("--camera-name", type=camera_name, required=True)
    parser.add_argument(
        "--capture-interval",
        type=capture_interval,
        default=DEFAULT_CAPTURE_INTERVAL,
        metavar="SECONDS",
        help=f"seconds between automatic capture attempts (default: {DEFAULT_CAPTURE_INTERVAL:g})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow replacement of an existing calibration",
    )
    return parser.parse_args(argv)


def calibration_path(name: str, image_size: tuple[int, int]) -> Path:
    width, height = image_size
    return CALIBRATION_DIRECTORY / f"{name}-{width}x{height}.yaml"


def create_board_points() -> Corners:
    columns, rows = BOARD_SIZE
    return np.asarray(
        [
            (column * SQUARE_SIZE_M, row * SQUARE_SIZE_M, 0.0)
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float32,
    )


def calibration_result(
    image_points: list[Corners], image_size: tuple[int, int]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    if len(image_points) < MIN_CALIBRATION_VIEWS:
        raise ValueError(f"at least {MIN_CALIBRATION_VIEWS} views are required")
    board_points = create_board_points()
    object_points = [board_points.copy() for _ in image_points]
    rms_error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return (
        float(rms_error),
        cast(NDArray[np.float64], camera_matrix),
        cast(NDArray[np.float64], distortion),
    )


def save_calibration(
    path: Path,
    name: str,
    image_size: tuple[int, int],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    rms_error: float,
    captured_views: int,
    *,
    force: bool = False,
) -> None:
    if path.exists() and not force:
        raise FileExistsError(
            f"calibration already exists: {path}; use --force to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"could not open calibration file for writing: {path}")
    try:
        storage.write("format_version", FORMAT_VERSION)
        storage.write("camera_name", name)
        storage.write("image_width", image_size[0])
        storage.write("image_height", image_size[1])
        storage.write("board_cols", BOARD_SIZE[0])
        storage.write("board_rows", BOARD_SIZE[1])
        storage.write("square_size_m", SQUARE_SIZE_M)
        storage.write("captured_views", captured_views)
        storage.write("rms_reprojection_error", rms_error)
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", distortion)
    finally:
        storage.release()

    saved = CameraCalibration.load(path)
    if (
        saved.image_size != image_size
        or not np.allclose(saved.camera_matrix, camera_matrix)
        or not np.allclose(saved.distortion, distortion)
        or not np.isclose(saved.rms_error, rms_error)
    ):
        raise RuntimeError(f"calibration file verification failed: {path}")


def put_status(
    image: Frame,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    scale: float = 0.65,
) -> None:
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def capture_preview(
    frame: Frame,
    captured_views: int,
    message: str,
    *,
    paused: bool,
    seconds_until_capture: float,
) -> tuple[Frame, Corners | None]:
    preview = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        BOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if found and corners is not None:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(preview, BOARD_SIZE, corners, found)

    put_status(
        preview,
        f"Views: {captured_views}/{MIN_CALIBRATION_VIEWS} | Board: "
        + ("detected" if found else "not detected"),
        (20, 35),
        (0, 255, 0) if found else (0, 0, 255),
    )
    timing = "Paused" if paused else f"Next attempt: {seconds_until_capture:.1f}s"
    put_status(preview, timing, (20, 65), (0, 255, 255), scale=0.5)
    put_status(
        preview,
        "Space: pause/resume | R: reset | C: calibrate | Q/Esc: quit",
        (20, 95),
        scale=0.5,
    )
    if message:
        put_status(preview, message, (20, 125), (0, 255, 255), scale=0.5)
    return preview, cast(Corners, corners) if found else None


def review_preview(
    frame: Frame,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    rms_error: float,
) -> Frame:
    height, width = frame.shape[:2]
    half_size = (max(1, width // 2), max(1, height // 2))
    raw = cv2.resize(frame, half_size)
    corrected = cv2.resize(cv2.undistort(frame, camera_matrix, distortion), half_size)
    preview = cast(Frame, np.hstack((raw, corrected)))
    put_status(preview, "Raw", (20, 30))
    put_status(preview, "Undistorted", (half_size[0] + 20, 30))
    put_status(
        preview,
        f"RMS: {rms_error:.4f} px",
        (20, max(60, half_size[1] - 45)),
        (0, 255, 255),
    )
    put_status(
        preview,
        "S: save | R: restart | Q/Esc: quit",
        (20, max(85, half_size[1] - 15)),
        (0, 255, 255),
        scale=0.5,
    )
    return preview


def run_session(
    camera: cv2.VideoCapture,
    first_frame: Frame,
    output_path: Path,
    name: str,
    *,
    force: bool,
    capture_interval: float = DEFAULT_CAPTURE_INTERVAL,
) -> None:
    image_size = (first_frame.shape[1], first_frame.shape[0])
    image_points: list[Corners] = []
    frame = first_frame
    mode = "capture"
    message = ""
    paused = False
    next_capture_at = time.monotonic() + capture_interval
    result: tuple[float, NDArray[np.float64], NDArray[np.float64]] | None = None
    print(
        f"Capturing every {capture_interval:g} seconds when the board is detected; "
        "Space pauses/resumes; R resets; C calibrates; Q or Escape quits."
    )

    while True:
        if mode == "capture":
            now = time.monotonic()
            preview, corners = capture_preview(
                frame,
                len(image_points),
                message,
                paused=paused,
                seconds_until_capture=max(0.0, next_capture_at - now),
            )
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                paused = not paused
                if not paused:
                    next_capture_at = now + capture_interval
                message = "Capture paused" if paused else "Capture resumed"
                print(message)
            elif key in (ord("r"), ord("R")):
                image_points.clear()
                next_capture_at = now + capture_interval
                message = "Captured views reset"
                print(message)
            elif key in (ord("c"), ord("C")):
                if len(image_points) < MIN_CALIBRATION_VIEWS:
                    message = (
                        f"Capture {MIN_CALIBRATION_VIEWS - len(image_points)} "
                        "more view(s) before calibrating"
                    )
                else:
                    result = calibration_result(image_points, image_size)
                    rms_error, camera_matrix, distortion = result
                    print(f"RMS reprojection error: {rms_error}")
                    print(f"Camera matrix:\n{camera_matrix}")
                    print(f"Distortion coefficients:\n{distortion}")
                    mode = "review"
                    message = ""
            elif key in (ord("q"), ord("Q"), 27):
                print("Calibration aborted; no file was written.")
                return
            elif not paused and now >= next_capture_at:
                next_capture_at = now + capture_interval
                if corners is None:
                    message = "No board detected; skipped this attempt"
                else:
                    image_points.append(corners.copy())
                    message = f"Captured view {len(image_points)}"
                    print(message)
        else:
            assert result is not None
            rms_error, camera_matrix, distortion = result
            preview = review_preview(frame, camera_matrix, distortion, rms_error)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                save_calibration(
                    output_path,
                    name,
                    image_size,
                    camera_matrix,
                    distortion,
                    rms_error,
                    len(image_points),
                    force=force,
                )
                print(f"Saved calibration to {output_path}")
                return
            if key in (ord("r"), ord("R")):
                image_points.clear()
                result = None
                mode = "capture"
                paused = False
                next_capture_at = time.monotonic() + capture_interval
                message = "Calibration discarded; capture new views"
                print(message)
            elif key in (ord("q"), ord("Q"), 27):
                print("Calibration aborted; no file was written.")
                return
        frame = read_frame(camera)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    camera: cv2.VideoCapture | None = None
    try:
        camera = open_camera(arguments.camera_index)
        print("Warming up camera...")
        for _ in range(CAMERA_WARMUP_FRAMES):
            read_frame(camera)
        first_frame = read_frame(camera)
        output_path = calibration_path(arguments.camera_name, CAMERA_SIZE)
        if output_path.exists() and not arguments.force:
            raise FileExistsError(
                f"calibration already exists: {output_path}; use --force to replace it"
            )
        run_session(
            camera,
            first_frame,
            output_path,
            arguments.camera_name,
            force=arguments.force,
            capture_interval=arguments.capture_interval,
        )
        return 0
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Calibration error: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
