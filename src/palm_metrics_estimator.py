"""Estimate palm metrics from timestamped hand observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite, log, sqrt
from typing import TYPE_CHECKING

from camera_geometry import CameraGeometry
from hand_observation import PALM_LANDMARKS, HandObservation

if TYPE_CHECKING:
    from palm_filter import PalmLandmarkFilter

MAX_HISTORY_FRAMES = 120
MOTION_WINDOW_MS = 120
MIN_MOTION_WINDOW_MS = 80
MAX_FRAME_GAP_MS = 250
MAX_PALM_STEP = 0.25
STILL_SPEED = 0.05
MIN_HANDEDNESS_SCORE = 0.7
MIN_PALM_AREA = 1e-8
ROTATION_CORRECTION_START_ABS_Z = 0.3
ROTATION_CORRECTION_FULL_ABS_Z = 0.6
ROTATION_CORRECTION_GAIN = 1.25
DEFAULT_PALM_WIDTH_M = 0.08
DEFAULT_HORIZONTAL_FOV_DEG = 60.0
MIN_WIDTH_PROJECTION = 0.25


@dataclass(frozen=True)
class ObservationFrame:
    timestamp_ms: int
    hands: tuple[HandObservation, ...]


@dataclass(frozen=True)
class PalmMetrics:
    last_frame_ms: int | None
    detected_hands: int
    # Normalized image width/height per second; None when motion is unavailable.
    palm_velocity: tuple[float, float] | None
    # Unit vector in world-landmark axes, pointing out of the palm side.
    palm_normal: tuple[float, float, float] | None
    # Rotation-corrected log scale change per second (1/s); negative is approaching.
    approach_rate: float | None = None
    # Center of the filtered palm in normalized image coordinates.
    palm_center: tuple[float, float] | None = None
    # Rotation-adjusted camera-space center in meters: x right, y down, z away.
    palm_center_m: tuple[float, float, float] | None = None
    # Thumb-index tip gap in image-space palm widths and its image-space midpoint.
    pinch_gap: float | None = None
    pinch_point: tuple[float, float] | None = None
    # Changes whenever the estimator begins a new single-hand track.
    track_id: int | None = None


class PalmMetricsEstimator:
    def __init__(
        self,
        palm_filter: PalmLandmarkFilter | None = None,
        *,
        image_size: tuple[int, int] | None = None,
        camera_geometry: CameraGeometry | None = None,
        palm_width_m: float = DEFAULT_PALM_WIDTH_M,
        horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    ) -> None:
        if not isfinite(palm_width_m) or palm_width_m <= 0:
            raise ValueError("palm_width_m must be finite and positive")
        if not isfinite(horizontal_fov_deg) or not 0 < horizontal_fov_deg < 180:
            raise ValueError("horizontal_fov_deg must be between 0 and 180")
        if (
            camera_geometry is not None
            and image_size is not None
            and camera_geometry.image_size != image_size
        ):
            raise ValueError("camera geometry and image size must match")
        self._observation_history: deque[ObservationFrame] = deque()
        self._palm_filter: PalmLandmarkFilter | None = palm_filter
        self._camera_geometry = camera_geometry
        if (
            self._camera_geometry is None
            and image_size is not None
            and all(dimension > 0 for dimension in image_size)
        ):
            self._camera_geometry = CameraGeometry.from_horizontal_fov(
                image_size, horizontal_fov_deg
            )
        self._palm_width_m = palm_width_m
        self._last_raw_hand: HandObservation | None = None
        self._last_raw_timestamp_ms: int | None = None
        self._track_id = 0

    def feed(self, timestamp_ms: int, hands: tuple[HandObservation, ...]) -> None:
        if not self._continues_single_hand_track(timestamp_ms, hands):
            self._observation_history.clear()
            self._track_id += 1
            if self._palm_filter is not None:
                self._palm_filter.reset()

        observed_hands = hands
        if len(hands) == 1:
            self._last_raw_hand = hands[0]
            self._last_raw_timestamp_ms = timestamp_ms
            if self._palm_filter is not None:
                observed_hands = (self._palm_filter.apply(timestamp_ms, hands[0]),)
        else:
            self._last_raw_hand = None
            self._last_raw_timestamp_ms = None

        self._observation_history.append(ObservationFrame(timestamp_ms, observed_hands))
        if len(self._observation_history) > MAX_HISTORY_FRAMES:
            self._observation_history.popleft()

    def _continues_single_hand_track(
        self, timestamp_ms: int, hands: tuple[HandObservation, ...]
    ) -> bool:
        previous = self._last_raw_hand
        previous_ms = self._last_raw_timestamp_ms
        if len(hands) != 1 or previous is None or previous_ms is None:
            return False
        elapsed_ms = timestamp_ms - previous_ms
        if elapsed_ms <= 0 or elapsed_ms > MAX_FRAME_GAP_MS:
            return False

        current = hands[0]
        previous_center = _palm_center(previous)
        current_center = _palm_center(current)
        distance = hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        if not isfinite(distance) or distance > MAX_PALM_STEP:
            return False

        previous_hand = previous.handedness
        current_hand = current.handedness
        return not (
            previous_hand is not None
            and current_hand is not None
            and previous_hand.score >= MIN_HANDEDNESS_SCORE
            and current_hand.score >= MIN_HANDEDNESS_SCORE
            and previous_hand.label != current_hand.label
        )

    def metrics(self) -> PalmMetrics:
        if not self._observation_history:
            return PalmMetrics(None, 0, None, None)

        latest = self._observation_history[-1]
        palm_center = _palm_center(latest.hands[0]) if len(latest.hands) == 1 else None
        palm_center_m = (
            _palm_center_m(
                latest.hands[0],
                self._camera_geometry,
                self._palm_width_m,
            )
            if len(latest.hands) == 1
            else None
        )
        palm_normal = (
            self._palm_normal(latest.hands[0]) if len(latest.hands) == 1 else None
        )
        pinch_gap, pinch_point = (
            _pinch_measurement(latest.hands[0])
            if len(latest.hands) == 1
            else (None, None)
        )
        return PalmMetrics(
            last_frame_ms=latest.timestamp_ms,
            detected_hands=len(latest.hands),
            palm_velocity=self._palm_velocity(),
            palm_normal=palm_normal,
            approach_rate=self._approach_rate(),
            palm_center=palm_center,
            palm_center_m=palm_center_m,
            pinch_gap=pinch_gap,
            pinch_point=pinch_point,
            track_id=self._track_id if len(latest.hands) == 1 else None,
        )

    @property
    def camera_geometry(self) -> CameraGeometry | None:
        return self._camera_geometry

    def _palm_velocity(self) -> tuple[float, float] | None:
        window = self._motion_window()
        if window is None:
            return None

        anchor, latest = window
        anchor_center = _palm_center(anchor.hands[0])
        latest_center = _palm_center(latest.hands[0])
        seconds = (latest.timestamp_ms - anchor.timestamp_ms) / 1_000
        vx = (latest_center[0] - anchor_center[0]) / seconds
        vy = (latest_center[1] - anchor_center[1]) / seconds
        if hypot(vx, vy) < STILL_SPEED:
            return (0.0, 0.0)
        return (vx, vy)

    def _approach_rate(self) -> float | None:
        """Estimate depth motion while accounting for apparent size from palm tilt."""
        window = self._motion_window()
        if window is None:
            return None

        anchor, latest = window
        anchor_hand = anchor.hands[0]
        latest_hand = latest.hands[0]
        anchor_size = _palm_size(anchor_hand)
        latest_size = _palm_size(latest_hand)
        if (
            not isfinite(anchor_size)
            or not isfinite(latest_size)
            or anchor_size <= 0
            or latest_size <= 0
        ):
            return None

        seconds = (latest.timestamp_ms - anchor.timestamp_ms) / 1_000
        rate = -log(latest_size / anchor_size) / seconds

        anchor_normal = self._palm_normal(anchor_hand)
        latest_normal = self._palm_normal(latest_hand)
        if anchor_normal is None or latest_normal is None:
            return rate

        # A planar, roughly symmetric palm looks smaller when tilted away from
        # face-on. The projection factor is bounded between 1/sqrt(2) and 1.
        anchor_projection = sqrt((1 + anchor_normal[2] ** 2) / 2)
        latest_projection = sqrt((1 + latest_normal[2] ** 2) / 2)

        # Depth estimates are least reliable near edge-on. Fade the correction
        # in smoothly so crossing the threshold cannot abruptly change the rate.
        facing = min(abs(anchor_normal[2]), abs(latest_normal[2]))
        correction_weight = min(
            1.0,
            max(
                0.0,
                (facing - ROTATION_CORRECTION_START_ABS_Z)
                / (ROTATION_CORRECTION_FULL_ABS_Z - ROTATION_CORRECTION_START_ABS_Z),
            ),
        )
        rotation_effect = log(latest_projection / anchor_projection) / seconds
        return rate + ROTATION_CORRECTION_GAIN * correction_weight * rotation_effect

    def _motion_window(self) -> tuple[ObservationFrame, ObservationFrame] | None:
        """Find two frames from one continuous hand track about 120 ms apart."""
        frames = self._observation_history
        if len(frames) < 2 or len(frames[-1].hands) != 1:
            return None

        latest = frames[-1]
        anchor = latest
        for frame in reversed(frames):
            anchor = frame
            if latest.timestamp_ms - anchor.timestamp_ms >= MOTION_WINDOW_MS:
                break

        duration_ms = latest.timestamp_ms - anchor.timestamp_ms
        if duration_ms < MIN_MOTION_WINDOW_MS:
            return None
        return anchor, latest

    def _palm_normal(self, hand: HandObservation) -> tuple[float, float, float] | None:
        """Estimate a unit vector pointing out of the palm side.

        In live preview, externally rotated palms still facing the camera have
        sometimes shown the wrong horizontal direction. The index/pinky depth
        estimates can dominate that component, so use it cautiously in this pose.
        """
        handedness = hand.handedness
        if (
            handedness is None
            or handedness.label not in ("Left", "Right")
            or handedness.score < MIN_HANDEDNESS_SCORE
        ):
            return None

        points = hand.world_landmarks
        if len(points) <= 17:
            return None

        wrist = points[0]
        middle_knuckle = points[9]
        across = (
            points[5].x - points[17].x,
            points[5].y - points[17].y,
            points[5].z - points[17].z,
        )
        toward_fingers = (
            middle_knuckle.x - wrist.x,
            middle_knuckle.y - wrist.y,
            middle_knuckle.z - wrist.z,
        )

        # The input frame is mirrored before recognition. With the resulting
        # handedness labels, this cross product points out of a Right palm and
        # into a Left palm; reverse it for Left.
        normal = (
            across[1] * toward_fingers[2] - across[2] * toward_fingers[1],
            across[2] * toward_fingers[0] - across[0] * toward_fingers[2],
            across[0] * toward_fingers[1] - across[1] * toward_fingers[0],
        )
        magnitude = hypot(*normal)
        if not isfinite(magnitude) or magnitude < MIN_PALM_AREA:
            return None

        sign = 1 if handedness.label == "Right" else -1
        return (
            sign * normal[0] / magnitude,
            sign * normal[1] / magnitude,
            sign * normal[2] / magnitude,
        )


def _palm_center(hand: HandObservation) -> tuple[float, float]:
    points = hand.image_landmarks
    return (
        sum(points[index].x for index in PALM_LANDMARKS) / len(PALM_LANDMARKS),
        sum(points[index].y for index in PALM_LANDMARKS) / len(PALM_LANDMARKS),
    )


def _palm_center_m(
    hand: HandObservation,
    camera_geometry: CameraGeometry | None,
    palm_width_m: float,
) -> tuple[float, float, float] | None:
    """Estimate a rotation-adjusted palm center from the knuckle span."""
    if camera_geometry is None:
        return None
    image_width, image_height = camera_geometry.image_size
    if len(hand.image_landmarks) <= 17 or len(hand.world_landmarks) <= 17:
        return None

    points = hand.image_landmarks
    index_knuckle = points[5]
    pinky_knuckle = points[17]
    knuckle_span_camera_units = hypot(
        (index_knuckle.x - pinky_knuckle.x) * image_width / camera_geometry.focal_x_px,
        (index_knuckle.y - pinky_knuckle.y) * image_height / camera_geometry.focal_y_px,
    )
    if not isfinite(knuckle_span_camera_units) or knuckle_span_camera_units <= 1e-6:
        return None

    world_index = hand.world_landmarks[5]
    world_pinky = hand.world_landmarks[17]
    width_dx = world_index.x - world_pinky.x
    width_dy = world_index.y - world_pinky.y
    width_dz = world_index.z - world_pinky.z
    world_width = hypot(width_dx, width_dy, width_dz)
    if not isfinite(world_width) or world_width <= 1e-6:
        return None

    # Only the part of the knuckle span parallel to the image plane contributes
    # to its apparent width. A near edge-on span is too unstable to correct.
    width_projection = hypot(width_dx, width_dy) / world_width
    if not isfinite(width_projection) or width_projection < MIN_WIDTH_PROJECTION:
        return None

    depth_m = palm_width_m * width_projection / knuckle_span_camera_units
    camera_center = camera_geometry.back_project(_palm_center(hand), depth_m)
    return camera_center if all(isfinite(value) for value in camera_center) else None


def _pinch_measurement(
    hand: HandObservation,
) -> tuple[float | None, tuple[float, float] | None]:
    points = hand.image_landmarks
    if len(points) <= 17:
        return None, None
    thumb, index = points[4], points[8]
    palm_width = hypot(points[5].x - points[17].x, points[5].y - points[17].y)
    gap = hypot(thumb.x - index.x, thumb.y - index.y)
    midpoint = ((thumb.x + index.x) / 2, (thumb.y + index.y) / 2)
    if (
        not isfinite(palm_width)
        or palm_width <= 1e-6
        or not isfinite(gap)
        or not all(isfinite(value) for value in midpoint)
    ):
        return None, None
    return gap / palm_width, midpoint


def _palm_size(hand: HandObservation) -> float:
    """RMS radius of the wrist and knuckles in normalized image coordinates."""
    points = hand.image_landmarks
    center_x, center_y = _palm_center(hand)
    squared_radii = (
        (points[index].x - center_x) ** 2 + (points[index].y - center_y) ** 2
        for index in PALM_LANDMARKS
    )
    return sqrt(sum(squared_radii) / len(PALM_LANDMARKS))
