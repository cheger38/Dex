"""Draw the exposed faces of a camera-aligned cube grid."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from camera_geometry import CameraGeometry
from cube_geometry import CUBE_SIDE_M, Point3D, cube_vertices

Frame = NDArray[np.uint8]
Cell = tuple[int, int, int]
Color = tuple[int, int, int]

# OpenCV colors are BGR. Face shading preserves the shape in every state.
_IDLE_COLOR: Color = (242, 214, 155)  # Light blue
_HOVER_COLOR: Color = (255, 240, 112)  # Cyan
_ACTIVE_COLOR: Color = (102, 170, 255)  # Orange

# Corner order matches cube_vertices. Each outward normal also identifies the
# neighboring grid cell that hides this face when occupied.
_FACES: tuple[tuple[tuple[int, int, int, int], Cell, float], ...] = (
    ((0, 2, 3, 1), (-1, 0, 0), 0.78),
    ((4, 5, 7, 6), (1, 0, 0), 0.95),
    ((0, 1, 5, 4), (0, -1, 0), 0.86),
    ((2, 6, 7, 3), (0, 1, 0), 0.82),
    ((0, 4, 6, 2), (0, 0, -1), 1.0),
    ((1, 3, 7, 5), (0, 0, 1), 0.72),
)


def draw_cube_scene(
    camera_image: Frame,
    positions_m: tuple[Point3D, ...],
    camera_geometry: CameraGeometry,
    highlighted_position_m: Point3D | None = None,
    selected_position_m: Point3D | None = None,
) -> None:
    """Project exposed faces and paint them from farthest to nearest.

    Ordering individual faces matters: two cubes at the same center depth can
    have different face depths, so painting whole cubes can put a distant side
    over a nearer front face.
    """
    if not positions_m:
        return

    anchor = positions_m[0]
    cells = {
        tuple(round((position[axis] - anchor[axis]) / CUBE_SIDE_M) for axis in range(3))
        for position in positions_m
    }
    height, width = camera_image.shape[:2]
    visible_faces: list[tuple[float, NDArray[np.int32], Color, Color, int]] = []

    for position in positions_m:
        vertices = cube_vertices(position)
        if any(vertex[2] <= 0 for vertex in vertices):
            continue
        cell = tuple(
            round((position[axis] - anchor[axis]) / CUBE_SIDE_M) for axis in range(3)
        )
        image_points = [
            (round(u * width), round(v * height))
            for vertex in vertices
            for u, v in (camera_geometry.project(vertex),)
        ]
        if position == selected_position_m:
            base_color = _ACTIVE_COLOR
        elif position == highlighted_position_m:
            base_color = _HOVER_COLOR
        else:
            base_color = _IDLE_COLOR
        emphasized = position in (selected_position_m, highlighted_position_m)
        for corners, normal, shade in _FACES:
            neighbor = tuple(cell[axis] + normal[axis] for axis in range(3))
            if neighbor in cells:
                continue
            # The camera is at the origin. A face is visible when its outward
            # normal points toward that origin.
            if sum(normal[axis] * position[axis] for axis in range(3)) >= 0:
                continue
            polygon = np.array([image_points[index] for index in corners], np.int32)
            depth = sum(vertices[index][2] for index in corners) / 4
            fill = (
                round(base_color[0] * shade),
                round(base_color[1] * shade),
                round(base_color[2] * shade),
            )
            outline = (255, 255, 255)
            visible_faces.append(
                (depth, polygon, fill, outline, 2 if emphasized else 1)
            )

    for _, polygon, fill, outline, thickness in sorted(
        visible_faces, key=lambda face: face[0], reverse=True
    ):
        cv2.fillConvexPoly(camera_image, polygon, fill, cv2.LINE_AA)
        cv2.polylines(camera_image, [polygon], True, outline, thickness, cv2.LINE_AA)
