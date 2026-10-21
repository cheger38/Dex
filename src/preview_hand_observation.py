"""Preview the hand-controlled cube playground and hand diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from math import hypot
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

# OpenCV's bundled Qt has an XCB plugin but no Wayland plugin. Select XWayland
# before importing cv2 when both a Wayland session and X display are available.
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import numpy as np
from numpy.typing import NDArray

from action_playground import ActionPlayground
from action_recognizer import (
    ActionEvent,
    ActionRecognizer,
    PinchEnd,
    PinchLost,
    PinchStart,
    Swipe,
)
from camera_calibration import (
    DEFAULT_CALIBRATION_PATH,
    CameraCalibration,
    CameraCorrection,
)
from camera_capture import open_camera, read_frame
from camera_geometry import CameraGeometry
from cube_overlay import draw_cube_scene
from hand_observation import Classification, HandObservation, Landmark
from palm_filter import OneEuroPalmFilter
from palm_metrics_estimator import PalmMetrics, PalmMetricsEstimator

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision.gesture_recognizer_result import (
        GestureRecognizerResult,
    )

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "gesture_recognizer.task"
)
WINDOW_NAME = "Hand Observation"
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 900
PANEL_LEFT = 980
ORIENTATION_PANEL_LEFT = 1600
VELOCITY_ORIGIN = (1430, 307)
VELOCITY_MAX_LENGTH = 65
VELOCITY_PIXELS_PER_UNIT = 90
NORMAL_ORIGIN = (1760, 330)
NORMAL_RADIUS = 90
APPROACH_METER_CENTER = (1760, 730)
APPROACH_METER_HALF_WIDTH = 105
APPROACH_DISPLAY_LIMIT = 2.0  # Log palm-size change per second.
APPROACH_DISPLAY_DEADZONE = 0.08
ACTION_DISPLAY_MS = 2_000
PreviewView = Literal["diagnostics", "playground"]
HAND_CONNECTIONS = (
    (0, 1),  # Wrist to thumb base (CMC)
    (1, 2),  # Thumb base to knuckle (MCP)
    (2, 3),  # Thumb knuckle to joint (IP)
    (3, 4),  # Thumb joint to tip
    (0, 5),  # Wrist to index knuckle (MCP)
    (5, 6),  # Index knuckle to first joint (PIP)
    (6, 7),  # Index first joint to second joint (DIP)
    (7, 8),  # Index second joint to tip
    (5, 9),  # Index knuckle to middle knuckle across palm
    (9, 10),  # Middle knuckle to first joint
    (10, 11),  # Middle first joint to second joint
    (11, 12),  # Middle second joint to tip
    (9, 13),  # Middle knuckle to ring knuckle across palm
    (13, 14),  # Ring knuckle to first joint
    (14, 15),  # Ring first joint to second joint
    (15, 16),  # Ring second joint to tip
    (13, 17),  # Ring knuckle to pinky knuckle across palm
    (17, 18),  # Pinky knuckle to first joint
    (18, 19),  # Pinky first joint to second joint
    (19, 20),  # Pinky second joint to tip
    (0, 17),  # Wrist to pinky knuckle along palm edge
)

Frame = NDArray[np.uint8]


def observe_hands(
    result: GestureRecognizerResult,
) -> tuple[HandObservation, ...]:
    """Copy one recognition result into plain, immutable application data."""
    hands = []
    for index, image_points in enumerate(result.hand_landmarks):
        world_points = result.hand_world_landmarks[index]
        poses = result.gestures[index] if index < len(result.gestures) else ()
        handedness = result.handedness[index] if index < len(result.handedness) else ()
        hands.append(
            HandObservation(
                image_landmarks=tuple(
                    Landmark(point.x, point.y, point.z) for point in image_points
                ),
                world_landmarks=tuple(
                    Landmark(point.x, point.y, point.z) for point in world_points
                ),
                poses=tuple(
                    Classification(category.category_name, category.score)
                    for category in poses
                ),
                handedness=(
                    Classification(handedness[0].category_name, handedness[0].score)
                    if handedness
                    else None
                ),
            )
        )
    return tuple(hands)


def draw_preview(
    frame: Frame,
    hands: tuple[HandObservation, ...],
    selected_hand: int,
    coordinate_space: str,
    metrics: PalmMetrics,
    action: ActionEvent | None,
    view: PreviewView = "diagnostics",
    playground: ActionPlayground | None = None,
    camera_geometry: CameraGeometry | None = None,
) -> Frame:
    """Draw either the clean cube scene or the detailed hand diagnostics."""
    if view == "playground":
        if playground is None or camera_geometry is None:
            raise ValueError("Playground view requires a scene and camera geometry")
        preview = frame.copy()
        draw_cube_scene(
            preview,
            playground.cube_positions_m,
            camera_geometry,
            playground.highlighted_position_m,
            playground.selected_position_m,
        )
        return preview

    height, width = frame.shape[:2]
    # Reserve the right side for values so it cannot cover landmark numbers.
    scale = min(PANEL_LEFT / width, DISPLAY_HEIGHT / height)
    shown_width = round(width * scale)
    shown_height = round(height * scale)
    left = (PANEL_LEFT - shown_width) // 2
    top = (DISPLAY_HEIGHT - shown_height) // 2
    preview = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    preview[top : top + shown_height, left : left + shown_width] = cv2.resize(
        frame, (shown_width, shown_height)
    )
    for index, hand in enumerate(hands):
        points = [
            (
                left + round(point.x * (shown_width - 1)),
                top + round(point.y * (shown_height - 1)),
            )
            for point in hand.image_landmarks
        ]
        color = (0, 230, 255) if index == selected_hand else (0, 180, 0)
        for start, end in HAND_CONNECTIONS:
            cv2.line(preview, points[start], points[end], color, 2, cv2.LINE_AA)
        for landmark_index, point in enumerate(points):
            cv2.circle(preview, point, 4, color, -1, cv2.LINE_AA)
            number_position = (
                min(point[0] + 6, PANEL_LEFT - 28),
                max(point[1] - 6, 16),
            )
            number = str(landmark_index)
            cv2.putText(
                preview,
                number,
                number_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                number,
                number_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        label = f"Hand {index + 1}: {_pose_label(hand)}"
        label_x = max(8, min(point[0] for point in points))
        label_y = max(30, min(point[1] for point in points) - 12)
        _write(preview, label, (label_x, label_y), 0.6)

    panel = preview.copy()
    cv2.rectangle(
        panel, (PANEL_LEFT, 0), (DISPLAY_WIDTH, DISPLAY_HEIGHT), (0, 0, 0), -1
    )
    cv2.addWeighted(panel, 0.78, preview, 0.22, 0, preview)
    cv2.line(
        preview,
        (ORIENTATION_PANEL_LEFT, 0),
        (ORIENTATION_PANEL_LEFT, DISPLAY_HEIGHT),
        (90, 90, 90),
        1,
    )
    text_x = PANEL_LEFT + 22
    _write(preview, "HAND OBSERVATION", (text_x, 36), 0.8)
    _write(
        preview, "TAB: playground  I/W: space  1/2: hand  Q: quit", (text_x, 69), 0.48
    )
    _write(preview, f"Detected hands: {len(hands)}", (text_x, 106), 0.64)
    _draw_palm_velocity(preview, metrics.palm_velocity)
    _draw_camera_palm_center(preview, metrics.palm_center_m)
    _draw_palm_normal(preview, metrics.palm_normal)
    _draw_approach_rate(preview, metrics.approach_rate)
    _draw_action(preview, action)

    if not hands:
        _write(preview, "No hands detected", (text_x, 156), 0.72)
        return preview
    if selected_hand >= len(hands):
        _write(preview, f"Hand {selected_hand + 1} not detected", (text_x, 156), 0.72)
        return preview

    hand = hands[selected_hand]
    handedness = (
        f"{hand.handedness.label} {hand.handedness.score:.1%}"
        if hand.handedness
        else "Unknown"
    )
    _write(preview, f"Hand {selected_hand + 1}  {handedness}", (text_x, 143), 0.68)
    _write(preview, f"Pose: {_pose_label(hand)}", (text_x, 174), 0.65)
    _write(
        preview, f"{coordinate_space.title()} landmarks (x, y, z)", (text_x, 425), 0.62
    )

    landmarks = (
        hand.image_landmarks if coordinate_space == "image" else hand.world_landmarks
    )
    for index, point in enumerate(landmarks):
        value = f"{index:02d}   {point.x:+.4f}   {point.y:+.4f}   {point.z:+.4f}"
        _write(preview, value, (text_x, 455 + index * 20), 0.56)
    return preview


def _draw_palm_velocity(preview: Frame, velocity: tuple[float, float] | None) -> None:
    text_x = PANEL_LEFT + 22
    _write(preview, "PALM VELOCITY (XY)", (text_x, 215), 0.65)
    _write(preview, "Normalized image / second", (text_x, 242), 0.49)
    cv2.line(
        preview, (text_x, 397), (ORIENTATION_PANEL_LEFT - 22, 397), (90, 90, 90), 1
    )

    origin_x, origin_y = VELOCITY_ORIGIN
    axis_color = (95, 95, 95)
    cv2.line(
        preview, (origin_x - 72, origin_y), (origin_x + 72, origin_y), axis_color, 1
    )
    cv2.line(
        preview, (origin_x, origin_y - 72), (origin_x, origin_y + 72), axis_color, 1
    )
    _write(preview, "+x", (origin_x + 76, origin_y + 6), 0.43)
    _write(preview, "+y", (origin_x + 7, origin_y + 84), 0.43)

    if velocity is None:
        _write(preview, "Unavailable", (text_x, 290), 0.57)
        return

    vx, vy = velocity
    _write(preview, f"vx: {vx:+.3f}", (text_x, 281), 0.59)
    _write(preview, f"vy: {vy:+.3f}", (text_x, 313), 0.59)
    speed = hypot(vx, vy)
    if speed == 0:
        cv2.circle(preview, VELOCITY_ORIGIN, 5, (0, 220, 255), -1, cv2.LINE_AA)
        return

    length = min(speed * VELOCITY_PIXELS_PER_UNIT, VELOCITY_MAX_LENGTH)
    endpoint = (
        origin_x + round(vx / speed * length),
        origin_y + round(vy / speed * length),
    )
    cv2.arrowedLine(
        preview,
        VELOCITY_ORIGIN,
        endpoint,
        (0, 220, 255),
        3,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def _draw_camera_palm_center(
    preview: Frame, center_m: tuple[float, float, float] | None
) -> None:
    text_x = 1300
    cv2.line(preview, (1280, 425), (1280, 650), (90, 90, 90), 1)
    _write(preview, "CAMERA PALM CENTER", (text_x, 455), 0.62)
    _write(preview, "Rotation-adjusted estimate (m)", (text_x, 482), 0.46)
    if center_m is None:
        _write(preview, "Unavailable", (text_x, 528), 0.57)
    else:
        for axis, value, y in zip("xyz", center_m, (528, 560, 592), strict=True):
            _write(preview, f"{axis}: {value:+.3f} m", (text_x, y), 0.59)
    _write(preview, "+x right  +y down  +z away", (text_x, 635), 0.43)


def _draw_palm_normal(
    preview: Frame, normal: tuple[float, float, float] | None
) -> None:
    text_x = ORIENTATION_PANEL_LEFT + 22
    _write(preview, "PALM ORIENTATION", (text_x, 36), 0.69)
    _write(preview, "Palm-facing normal (XYZ)", (text_x, 68), 0.51)
    _write(preview, "Tilt shown in XY plane", (text_x, 92), 0.46)

    origin_x, origin_y = NORMAL_ORIGIN
    axis_color = (95, 95, 95)
    cv2.circle(preview, NORMAL_ORIGIN, NORMAL_RADIUS, axis_color, 1, cv2.LINE_AA)
    cv2.line(
        preview,
        (origin_x - NORMAL_RADIUS, origin_y),
        (origin_x + NORMAL_RADIUS, origin_y),
        axis_color,
        1,
    )
    cv2.line(
        preview,
        (origin_x, origin_y - NORMAL_RADIUS),
        (origin_x, origin_y + NORMAL_RADIUS),
        axis_color,
        1,
    )
    _write(preview, "+x", (origin_x + NORMAL_RADIUS + 7, origin_y + 5), 0.43)
    _write(preview, "+y", (origin_x + 8, origin_y + NORMAL_RADIUS + 20), 0.43)

    if normal is None:
        _write(preview, "Unavailable", (text_x, 163), 0.57)
        return

    nx, ny, nz = normal
    _write(preview, f"x: {nx:+.3f}", (text_x, 148), 0.54)
    _write(preview, f"y: {ny:+.3f}", (text_x, 177), 0.54)
    _write(preview, f"z: {nz:+.3f}", (text_x, 206), 0.54)

    # MediaPipe uses negative z toward the camera and positive z away.
    color = (
        (255, 220, 0) if nz < -0.1 else (0, 150, 255) if nz > 0.1 else (255, 255, 255)
    )
    projected_length = hypot(nx, ny) * NORMAL_RADIUS
    if projected_length < 4:
        cv2.circle(preview, NORMAL_ORIGIN, 7, color, -1, cv2.LINE_AA)
    else:
        endpoint = (
            origin_x + round(nx * NORMAL_RADIUS),
            origin_y + round(ny * NORMAL_RADIUS),
        )
        cv2.arrowedLine(
            preview, NORMAL_ORIGIN, endpoint, color, 3, cv2.LINE_AA, tipLength=0.2
        )

    facing = (
        "Toward camera" if nz < -0.1 else "Away from camera" if nz > 0.1 else "Edge-on"
    )
    _write(preview, facing, (text_x, 476), 0.57)
    cv2.circle(preview, (text_x + 7, 515), 6, (255, 220, 0), -1, cv2.LINE_AA)
    _write(preview, "-z  toward camera", (text_x + 23, 521), 0.43)
    cv2.circle(preview, (text_x + 7, 546), 6, (0, 150, 255), -1, cv2.LINE_AA)
    _write(preview, "+z  away from camera", (text_x + 23, 552), 0.43)


def _draw_approach_rate(preview: Frame, rate: float | None) -> None:
    text_x = ORIENTATION_PANEL_LEFT + 22
    cv2.line(preview, (text_x, 585), (DISPLAY_WIDTH - 22, 585), (90, 90, 90), 1)
    _write(preview, "APPROACH RATE", (text_x, 625), 0.69)
    _write(preview, "Estimated from palm size", (text_x, 654), 0.48)

    center_x, center_y = APPROACH_METER_CENTER
    left = center_x - APPROACH_METER_HALF_WIDTH
    right = center_x + APPROACH_METER_HALF_WIDTH
    _write(preview, "TOWARD", (left - 8, 695), 0.5)
    _write(preview, "AWAY", (right - 54, 695), 0.5)
    cv2.line(preview, (left, center_y), (right, center_y), (75, 75, 75), 12)
    cv2.line(
        preview,
        (center_x, center_y - 15),
        (center_x, center_y + 15),
        (160, 160, 160),
        2,
    )
    _write(preview, "-2/s", (left - 10, 770), 0.45)
    _write(preview, "0", (center_x - 5, 770), 0.45)
    _write(preview, "+2/s", (right - 28, 770), 0.45)

    if rate is None:
        _write(preview, "Unavailable", (text_x, 815), 0.57)
        return
    if abs(rate) < APPROACH_DISPLAY_DEADZONE:
        cv2.circle(preview, APPROACH_METER_CENTER, 7, (255, 255, 255), -1, cv2.LINE_AA)
        _write(preview, "Steady", (text_x, 815), 0.57)
        return

    distance = round(
        min(abs(rate) / APPROACH_DISPLAY_LIMIT, 1) * APPROACH_METER_HALF_WIDTH
    )
    endpoint = (center_x - distance if rate < 0 else center_x + distance, center_y)
    color = (255, 220, 0) if rate < 0 else (0, 150, 255)
    cv2.line(preview, APPROACH_METER_CENTER, endpoint, color, 12, cv2.LINE_AA)
    cv2.circle(preview, endpoint, 7, color, -1, cv2.LINE_AA)
    _write(preview, "Approaching" if rate < 0 else "Moving away", (text_x, 815), 0.57)


def _draw_action(preview: Frame, action: ActionEvent | None) -> None:
    text_x = ORIENTATION_PANEL_LEFT + 22
    cv2.line(preview, (text_x, 832), (DISPLAY_WIDTH - 22, 832), (90, 90, 90), 1)
    if action is None:
        _write(preview, "No recent action", (text_x, 870), 0.55)
        return

    _write(preview, f"ACTION: {type(action).__name__}", (text_x, 852), 0.57)
    if isinstance(action, Swipe):
        vx, vy = action.peak_velocity
        _write(preview, f"peak vx {vx:+.3f}  vy {vy:+.3f}", (text_x, 873), 0.46)
        _write(
            preview,
            f"speed {hypot(vx, vy):.3f}/s  t {action.timestamp_ms} ms",
            (text_x, 892),
            0.42,
        )
    elif isinstance(action, (PinchStart, PinchEnd, PinchLost)):
        x, y = action.point
        _write(preview, f"point {x:+.3f}, {y:+.3f}", (text_x, 873), 0.46)
        _write(preview, f"t {action.timestamp_ms} ms", (text_x, 892), 0.42)
    else:
        _write(preview, f"t {action.timestamp_ms} ms", (text_x, 879), 0.47)


def _pose_label(hand: HandObservation) -> str:
    if not hand.poses or hand.poses[0].label == "None":
        return "Unrecognized"
    pose = hand.poses[0]
    return f"{pose.label} {pose.score:.1%}"


def _write(frame: Frame, value: str, position: tuple[int, int], size: float) -> None:
    cv2.putText(
        frame,
        value,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def load_camera_correction(
    path: Path, image_size: tuple[int, int]
) -> CameraCorrection | None:
    try:
        calibration = CameraCalibration.load(path, expected_size=image_size)
        correction = CameraCorrection(calibration)
    except (OSError, ValueError, cv2.error) as error:
        print(
            f"Warning: {error}; using uncorrected video and assumed camera field of view",
            file=sys.stderr,
        )
        return None
    print(f"Loaded camera calibration: {path}")
    return correction


def prepare_camera_frame(frame: Frame, correction: CameraCorrection | None) -> Frame:
    """Correct camera pixels before mirroring for MediaPipe and the preview."""
    if correction is not None:
        frame = correction.apply(frame)
    return cast(Frame, cv2.flip(frame, 1))


def main() -> None:
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="OpenCV YAML intrinsic calibration for this camera",
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH, help="MediaPipe .task path"
    )
    parser.add_argument(
        "--no-palm-filter",
        action="store_true",
        help="Show estimates calculated from raw palm landmarks",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Start in the hand diagnostics view instead of the playground",
    )
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model file does not exist: {args.model}")

    camera = open_camera(args.camera)

    try:
        # Keep the heavyweight model import out of --help and offline inspection.
        import mediapipe as mp

        options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with mp.tasks.vision.GestureRecognizer.create_from_options(
            options
        ) as recognizer:
            # Read a first frame to init the estimator
            frame = read_frame(camera)
            image_size = (frame.shape[1], frame.shape[0])
            correction = load_camera_correction(args.calibration, image_size)

            estimator = PalmMetricsEstimator(
                palm_filter=None if args.no_palm_filter else OneEuroPalmFilter(),
                image_size=image_size,
                camera_geometry=(
                    correction.mirrored_geometry if correction is not None else None
                ),
            )
            action_recognizer = ActionRecognizer()
            camera_geometry = estimator.camera_geometry
            if camera_geometry is None:
                raise RuntimeError("Camera geometry requires a valid image size")
            playground = ActionPlayground(camera_geometry)
            displayed_action: ActionEvent | None = None
            view: PreviewView = "diagnostics" if args.diagnostics else "playground"

            selected_hand = 0
            coordinate_space = "image"
            last_timestamp_ms = -1
            while True:
                frame = read_frame(camera)

                # Timestamp the frame read from the camera.
                timestamp_ms = max(
                    time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1
                )
                last_timestamp_ms = timestamp_ms

                # Convert frame for model
                frame = prepare_camera_frame(frame, correction)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Get labeled results
                result = recognizer.recognize_for_video(image, timestamp_ms)
                hands = observe_hands(result)

                # Pass into metrics calculator
                estimator.feed(timestamp_ms, hands)
                metrics = estimator.metrics()

                # Feed and get action events
                action_recognizer.feed(metrics)
                actions = action_recognizer.events()
                playground.feed(
                    actions,
                    action_recognizer.active_pinch(),
                    metrics.palm_center_m,
                    metrics.pinch_point,
                )

                if actions:
                    displayed_action = actions[-1]
                if (
                    displayed_action is not None
                    and timestamp_ms - displayed_action.timestamp_ms > ACTION_DISPLAY_MS
                ):
                    displayed_action = None

                preview = draw_preview(
                    frame,
                    hands,
                    selected_hand,
                    coordinate_space,
                    metrics,
                    displayed_action,
                    view,
                    playground,
                    camera_geometry,
                )

                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == 9:  # Tab
                    view = "playground" if view == "diagnostics" else "diagnostics"
                elif key == ord("i"):
                    coordinate_space = "image"
                elif key == ord("w"):
                    coordinate_space = "world"
                elif key in (ord("1"), ord("2")):
                    selected_hand = key - ord("1")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
