"""Draw the exposed faces of a camera-aligned cube grid."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from camera_geometry import CameraGeometry
from cube_geometry import CUBE_SIDE_M, Point3D, cube_vertices

Frame = NDArray[np.uint8]
Cell = tuple[int, int, int]

# Corner order matches cube_vertices. Each outward normal also identifies the
# neighboring grid cell that hides this face when occupied.
_FACES: tuple[tuple[tuple[int, int, int, int], Cell, tuple[int, int, int]], ...] = (
    ((0, 2, 3, 1), (-1, 0, 0), (0, 135, 190)),
    ((4, 5, 7, 6), (1, 0, 0), (0, 200, 245)),
    ((0, 1, 5, 4), (0, -1, 0), (0, 165, 215)),
    ((2, 6, 7, 3), (0, 1, 0), (0, 150, 205)),
    ((0, 4, 6, 2), (0, 0, -1), (0, 180, 230)),
    ((1, 3, 7, 5), (0, 0, 1), (0, 120, 175)),
)


def draw_cube_scene(
    camera_image: Frame,
    positions_m: tuple[Point3D, ...],
    camera_geometry: CameraGeometry,
    highlighted_position_m: Point3D | None = None,
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
    visible_faces: list[
        tuple[float, NDArray[np.int32], tuple[int, int, int], tuple[int, int, int], int]
    ] = []

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
        highlighted = position == highlighted_position_m
        for corners, normal, color in _FACES:
            neighbor = tuple(cell[axis] + normal[axis] for axis in range(3))
            if neighbor in cells:
                continue
            # The camera is at the origin. A face is visible when its outward
            # normal points toward that origin.
            if sum(normal[axis] * position[axis] for axis in range(3)) >= 0:
                continue
            polygon = np.array([image_points[index] for index in corners], np.int32)
            depth = sum(vertices[index][2] for index in corners) / 4
            fill = (90, 240, 120) if highlighted else color
            outline = (40, 255, 70) if highlighted else (255, 255, 255)
            visible_faces.append(
                (depth, polygon, fill, outline, 2 if highlighted else 1)
            )

    for _, polygon, fill, outline, thickness in sorted(
        visible_faces, key=lambda face: face[0], reverse=True
    ):
        cv2.fillConvexPoly(camera_image, polygon, fill, cv2.LINE_AA)
        cv2.polylines(camera_image, [polygon], True, outline, thickness, cv2.LINE_AA)
