"""Edit a camera-anchored grid of cubes with a pinch and a straight drag."""

from __future__ import annotations

from math import floor, isfinite

from action_recognizer import ActionEvent, ActivePinch, PinchEnd, PinchLost, PinchStart
from camera_geometry import CameraGeometry
from cube_geometry import CUBE_SIDE_M, Point3D, cube_vertices

INITIAL_DEPTH_M = 0.5
MAX_DRAG_STEPS = 200
PICK_MARGIN_PX = 12
PICK_DEPTH_TOLERANCE_M = 0.10
Cell = tuple[int, int, int]
Point2D = tuple[float, float]


class ActionPlayground:
    """Own the persistent cube grid and one provisional pinch edit."""

    def __init__(self, camera_geometry: CameraGeometry) -> None:
        self._camera_geometry = camera_geometry
        self._origin_m = camera_geometry.back_project((0.5, 0.5), INITIAL_DEPTH_M)
        self._cubes: set[Cell] = {(0, 0, 0)}
        self._preview: set[Cell] = self._cubes.copy()
        self._highlighted: Cell | None = None
        self._selected: Cell | None = None
        self._drag_start_m: Point3D | None = None
        self._direction: Cell | None = None
        self._edit_mode: str | None = None

    def feed(
        self,
        events: tuple[ActionEvent, ...],
        active_pinch: ActivePinch | None,
        palm_center_m: Point3D | None,
        pinch_point: Point2D | None,
    ) -> None:
        """Update hover, preview, and commit state from one observation frame."""
        depth_m = palm_center_m[2] if palm_center_m is not None else None
        if depth_m is None or not isfinite(depth_m) or depth_m <= 0:
            depth_m = None

        start = next((event for event in events if isinstance(event, PinchStart)), None)
        end = next((event for event in events if isinstance(event, PinchEnd)), None)
        lost = any(isinstance(event, PinchLost) for event in events)

        if self._selected is None:
            self._highlighted = self._pick(pinch_point, depth_m)
            if start is not None and depth_m is not None:
                selected = self._pick(start.point, depth_m)
                if selected is not None:
                    self._selected = selected
                    self._highlighted = selected
                    self._drag_start_m = self._camera_geometry.back_project(
                        start.point, depth_m
                    )
                    self._direction = None
                    self._edit_mode = None

        if self._selected is None:
            return

        if lost:
            self._finish_drag(commit=False)
            return

        point = (
            end.point
            if end is not None
            else (active_pinch.point if active_pinch is not None else None)
        )
        if point is not None and depth_m is not None:
            self._update_preview(self._camera_geometry.back_project(point, depth_m))

        if end is not None:
            self._finish_drag(commit=True)
        elif active_pinch is None:
            self._finish_drag(commit=False)

    def _pick(self, point: Point2D | None, depth_m: float | None) -> Cell | None:
        if point is None or depth_m is None or not all(map(isfinite, point)):
            return None
        width, height = self._camera_geometry.image_size
        px, py = point[0] * width, point[1] * height
        candidates: list[tuple[float, float, Cell]] = []
        for cell in self._cubes:
            center = self._center_m(cell)
            if center[2] <= CUBE_SIDE_M / 2:
                continue
            depth_distance = max(0.0, abs(depth_m - center[2]) - CUBE_SIDE_M / 2)
            if depth_distance > PICK_DEPTH_TOLERANCE_M:
                continue
            projected = [
                self._camera_geometry.project(v) for v in cube_vertices(center)
            ]
            xs = [p[0] * width for p in projected]
            ys = [p[1] * height for p in projected]
            if not (
                min(xs) - PICK_MARGIN_PX <= px <= max(xs) + PICK_MARGIN_PX
                and min(ys) - PICK_MARGIN_PX <= py <= max(ys) + PICK_MARGIN_PX
            ):
                continue
            center_px = self._camera_geometry.project(center)
            screen_distance = (px - center_px[0] * width) ** 2 + (
                py - center_px[1] * height
            ) ** 2
            candidates.append((depth_distance, screen_distance, cell))
        return min(candidates)[2] if candidates else None

    def _update_preview(self, hand_m: Point3D) -> None:
        assert self._drag_start_m is not None and self._selected is not None
        displacement = tuple(hand_m[i] - self._drag_start_m[i] for i in range(3))
        if self._direction is None:
            axis = max(range(3), key=lambda index: abs(displacement[index]))
            if abs(displacement[axis]) < CUBE_SIDE_M / 2:
                return
            direction = [0, 0, 0]
            direction[axis] = 1 if displacement[axis] > 0 else -1
            self._direction = (direction[0], direction[1], direction[2])
            neighbor = self._offset(self._selected, self._direction, 1)
            self._edit_mode = "remove" if neighbor in self._cubes else "add"

        axis = next(i for i, value in enumerate(self._direction) if value)
        distance = displacement[axis] * self._direction[axis]
        requested_steps = min(
            MAX_DRAG_STEPS, max(0, floor((distance + CUBE_SIDE_M / 2) / CUBE_SIDE_M))
        )
        preview = self._cubes.copy()
        for step in range(1, requested_steps + 1):
            cell = self._offset(self._selected, self._direction, step)
            if self._edit_mode == "remove":
                if cell not in self._cubes:
                    break
                preview.remove(cell)
            else:
                if cell in self._cubes or self._center_m(cell)[2] <= CUBE_SIDE_M / 2:
                    break
                preview.add(cell)
        self._preview = preview

    def _finish_drag(self, *, commit: bool) -> None:
        if commit:
            self._cubes = self._preview
        self._preview = self._cubes.copy()
        self._selected = None
        self._drag_start_m = None
        self._direction = None
        self._edit_mode = None
        self._highlighted = None

    @staticmethod
    def _offset(cell: Cell, direction: Cell, steps: int) -> Cell:
        return (
            cell[0] + direction[0] * steps,
            cell[1] + direction[1] * steps,
            cell[2] + direction[2] * steps,
        )

    def _center_m(self, cell: Cell) -> Point3D:
        return (
            self._origin_m[0] + cell[0] * CUBE_SIDE_M,
            self._origin_m[1] + cell[1] * CUBE_SIDE_M,
            self._origin_m[2] + cell[2] * CUBE_SIDE_M,
        )

    @property
    def cube_positions_m(self) -> tuple[Point3D, ...]:
        return tuple(self._center_m(cell) for cell in sorted(self._preview))

    @property
    def highlighted_position_m(self) -> Point3D | None:
        return (
            self._center_m(self._highlighted) if self._highlighted is not None else None
        )

    @property
    def selected_position_m(self) -> Point3D | None:
        return self._center_m(self._selected) if self._selected is not None else None

    def is_dragging(self) -> bool:
        return self._selected is not None
