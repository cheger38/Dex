"""Pinhole projection shared by camera-space estimates and the playground."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, radians, tan


@dataclass(frozen=True)
class CameraGeometry:
    image_size: tuple[int, int]
    focal_x_px: float
    focal_y_px: float
    principal_x_px: float
    principal_y_px: float

    @classmethod
    def from_horizontal_fov(
        cls, image_size: tuple[int, int], horizontal_fov_deg: float
    ) -> CameraGeometry:
        width, height = image_size
        focal_px = width / (2 * tan(radians(horizontal_fov_deg) / 2))
        return cls(image_size, focal_px, focal_px, width / 2, height / 2)

    @property
    def focal_px(self) -> float:
        """Horizontal focal length kept for callers that use a scalar focal length."""
        return self.focal_x_px

    @property
    def horizontal_fov_deg(self) -> float:
        width, _ = self.image_size
        return 2 * degrees(atan(width / (2 * self.focal_x_px)))

    def back_project(
        self, point: tuple[float, float], depth_m: float
    ) -> tuple[float, float, float]:
        """Place normalized image coordinates on a plane at positive depth."""
        width, height = self.image_size
        return (
            (point[0] * width - self.principal_x_px) * depth_m / self.focal_x_px,
            (point[1] * height - self.principal_y_px) * depth_m / self.focal_y_px,
            depth_m,
        )

    def project(self, point_m: tuple[float, float, float]) -> tuple[float, float]:
        """Project a camera-space point into normalized image coordinates."""
        x, y, z = point_m
        width, height = self.image_size
        return (
            (self.principal_x_px + x * self.focal_x_px / z) / width,
            (self.principal_y_px + y * self.focal_y_px / z) / height,
        )
