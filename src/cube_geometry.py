"""Camera-aligned cube dimensions and vertices shared by editing and drawing."""

from __future__ import annotations

Point3D = tuple[float, float, float]
CUBE_SIDE_M = 0.05


def cube_vertices(center_m: Point3D) -> tuple[Point3D, ...]:
    """Return corners with the front face parallel to the camera sensor."""
    half = CUBE_SIDE_M / 2
    return tuple(
        (center_m[0] + x, center_m[1] + y, center_m[2] + z)
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
    )
