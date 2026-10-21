"""Camera-aligned cube editing and perspective rendering."""

import unittest

import numpy as np

from action_playground import ActionPlayground
from action_recognizer import ActivePinch, PinchEnd, PinchLost, PinchStart, Swipe
from camera_geometry import CameraGeometry
from cube_geometry import CUBE_SIDE_M, cube_vertices
from cube_overlay import draw_cube_scene
from palm_metrics_estimator import PalmMetrics
from preview_hand_observation import draw_preview

CAMERA = CameraGeometry.from_horizontal_fov((1000, 800), 90)
ORIGIN = (0.0, 0.0, 0.5)


class ActionPlaygroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.playground = ActionPlayground(CAMERA)

    def feed_at(
        self,
        position: tuple[float, float, float],
        events: tuple = (),
        *,
        pinching: bool = False,
    ) -> None:
        point = CAMERA.project(position)
        active = ActivePinch(0, point) if pinching else None
        self.playground.feed(events, active, (0, 0, position[2]), point)

    def start_at(self, position: tuple[float, float, float]) -> None:
        point = CAMERA.project(position)
        self.feed_at(position)
        self.assertEqual(self.playground.highlighted_position_m, position)
        self.feed_at(position, (PinchStart(0, point),), pinching=True)
        self.assertTrue(self.playground.is_dragging())

    def release_at(self, position: tuple[float, float, float]) -> None:
        self.feed_at(position, (PinchEnd(1, CAMERA.project(position)),))
        self.assertFalse(self.playground.is_dragging())

    def test_initial_cube_is_centered_and_camera_aligned(self) -> None:
        self.assertEqual(self.playground.cube_positions_m, (ORIGIN,))
        self.assertEqual(CAMERA.project(ORIGIN), (0.5, 0.5))
        vertices = cube_vertices(ORIGIN)
        self.assertEqual({v[2] for v in vertices}, {0.475, 0.525})
        self.assertEqual(len({(v[0], v[1]) for v in vertices}), 4)
        self.playground.feed((Swipe(0, (1, 0)),), None, None, None)
        self.assertEqual(self.playground.cube_positions_m, (ORIGIN,))

    def test_hover_requires_screen_and_depth_proximity(self) -> None:
        self.feed_at(ORIGIN)
        self.assertEqual(self.playground.highlighted_position_m, ORIGIN)
        self.feed_at((0.2, 0, 0.5))
        self.assertIsNone(self.playground.highlighted_position_m)
        self.feed_at((0, 0, 0.8))
        self.assertIsNone(self.playground.highlighted_position_m)
        point = CAMERA.project((0.2, 0, 0.5))
        self.feed_at((0.2, 0, 0.5), (PinchStart(0, point),), pinching=True)
        self.feed_at(ORIGIN, pinching=True)
        self.assertFalse(self.playground.is_dragging())

    def test_all_six_face_directions_add_one_cube(self) -> None:
        for axis in range(3):
            for sign in (-1, 1):
                with self.subTest(axis=axis, sign=sign):
                    self.playground = ActionPlayground(CAMERA)
                    self.start_at(ORIGIN)
                    endpoint = list(ORIGIN)
                    endpoint[axis] += sign * CUBE_SIDE_M
                    position = (endpoint[0], endpoint[1], endpoint[2])
                    self.feed_at(position, pinching=True)
                    self.assertEqual(len(self.playground.cube_positions_m), 2)
                    self.assertIn(ORIGIN, self.playground.cube_positions_m)
                    self.assertIn(position, self.playground.cube_positions_m)
                    self.release_at(position)
                    self.assertEqual(len(self.playground.cube_positions_m), 2)

    def test_drag_locks_axis_and_pullback_shortens_preview(self) -> None:
        self.start_at(ORIGIN)
        self.feed_at((0.15, 0, 0.5), pinching=True)
        self.assertEqual(len(self.playground.cube_positions_m), 4)
        self.feed_at((0.05, 0.3, 0.5), pinching=True)
        self.assertEqual(len(self.playground.cube_positions_m), 2)
        self.assertIn((0.05, 0, 0.5), self.playground.cube_positions_m)
        self.release_at((0.05, 0.3, 0.5))
        self.assertEqual(len(self.playground.cube_positions_m), 2)

    def test_drag_toward_occupied_cells_removes_only_crossed_row(self) -> None:
        self.start_at(ORIGIN)
        self.release_at((0.10, 0, 0.5))
        self.assertEqual(len(self.playground.cube_positions_m), 3)
        self.start_at(ORIGIN)
        self.feed_at((0.20, 0, 0.5), pinching=True)
        self.assertEqual(self.playground.cube_positions_m, (ORIGIN,))
        self.feed_at((0.05, 0, 0.5), pinching=True)
        self.assertEqual(len(self.playground.cube_positions_m), 2)
        self.release_at((0.05, 0, 0.5))
        self.assertIn((0.10, 0, 0.5), self.playground.cube_positions_m)

    def test_added_cube_can_be_selected_for_a_branch(self) -> None:
        self.start_at(ORIGIN)
        self.release_at((0.05, 0, 0.5))
        second = (0.05, 0, 0.5)
        self.start_at(second)
        self.release_at((0.05, 0.05, 0.5))
        self.assertEqual(len(self.playground.cube_positions_m), 3)
        self.assertIn((0.05, 0.05, 0.5), self.playground.cube_positions_m)

    def test_original_cube_can_be_removed_from_an_added_cube(self) -> None:
        self.start_at(ORIGIN)
        self.release_at((0.05, 0, 0.5))
        self.start_at((0.05, 0, 0.5))
        self.release_at(ORIGIN)
        self.assertEqual(self.playground.cube_positions_m, ((0.05, 0, 0.5),))

    def test_depth_selects_between_overlapping_cubes(self) -> None:
        self.start_at(ORIGIN)
        self.release_at((0, 0, 0.45))
        near = (0, 0, 0.45)
        self.start_at(near)
        self.assertEqual(self.playground.selected_position_m, near)
        self.feed_at((0.05, 0, 0.45), pinching=True)
        self.assertIn((0.05, 0, 0.45), self.playground.cube_positions_m)

    def test_lost_tracking_cancels_preview(self) -> None:
        self.start_at(ORIGIN)
        self.feed_at((0.10, 0, 0.5), pinching=True)
        self.assertEqual(len(self.playground.cube_positions_m), 3)
        self.feed_at((0.10, 0, 0.5), (PinchLost(1, (0.6, 0.5)),))
        self.assertEqual(self.playground.cube_positions_m, (ORIGIN,))
        self.assertFalse(self.playground.is_dragging())


class CameraOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.playground = ActionPlayground(CAMERA)
        self.frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.metrics = PalmMetrics(None, 0, None, None)

    def render(self, view: str = "playground") -> np.ndarray:
        return draw_preview(
            self.frame,
            (),
            0,
            "image",
            self.metrics,
            None,
            view,  # type: ignore[arg-type]
            self.playground,
            CAMERA,
        )

    def test_cube_is_only_in_playground_and_highlight_changes_color(self) -> None:
        normal = self.render()
        self.playground.feed((), None, ORIGIN, (0.5, 0.5))
        highlighted = self.render()
        self.assertFalse(np.array_equal(normal[:, :980], highlighted[:, :980]))
        self.assertEqual(normal[450, 490, 1] < highlighted[450, 490, 1], True)
        self.assertEqual(np.count_nonzero(self.render("diagnostics")[:, :980]), 0)

    def test_additional_cube_is_projected_in_camera_view(self) -> None:
        point = CAMERA.project(ORIGIN)
        self.playground.feed(
            (PinchStart(0, point),), ActivePinch(0, point), ORIGIN, point
        )
        endpoint = (0.10, 0, 0.5)
        self.playground.feed(
            (PinchEnd(1, CAMERA.project(endpoint)),),
            None,
            endpoint,
            CAMERA.project(endpoint),
        )
        rendered = self.render()
        camera_image = rendered[:, :980]
        colored = (camera_image[:, :, 2] > 100) & (camera_image[:, :, 0] < 30)
        self.assertGreater(int(np.nonzero(colored)[1].max()), 550)

    def test_joined_cubes_do_not_draw_their_shared_side_face(self) -> None:
        left = np.zeros((800, 1000, 3), dtype=np.uint8)
        joined = left.copy()
        draw_cube_scene(left, (ORIGIN,), CAMERA)
        draw_cube_scene(joined, (ORIGIN, (0.05, 0, 0.5)), CAMERA)
        # The right cube's left side projects into the left cube's front face.
        # Its shared face must not darken that region.
        np.testing.assert_array_equal(joined[350:450, 524], left[350:450, 524])


if __name__ == "__main__":
    unittest.main()
