import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hand_retarget_wuji2 import (MODES, PALM_ALIGNMENT_POINTS, PARENTS,
                                 Wuji2StreamingRetargeter, pose_error)


def root_pose():
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", [0.25, -0.18, 0.4]).as_matrix()
    value[:3, 3] = [0.31, -0.12, 0.72]
    return value


def camera_points(solver, q, root):
    return solver.hand.fk(q) @ root[:3, :3].T + root[:3, 3]


class HandRetargetTests(unittest.TestCase):
    def test_urdf_has_only_one_truly_root_fixed_keypoint(self):
        solver = Wuji2StreamingRetargeter("left")
        _, jacobian = solver.hand.fk(solver.hand.neutral, True)
        moving = np.max(np.abs(jacobian), axis=(1, 2)) > 1e-10
        self.assertEqual(np.flatnonzero(~moving).tolist(), [0])
        self.assertFalse(moving[PALM_ALIGNMENT_POINTS[0]])
        self.assertTrue(moving[PALM_ALIGNMENT_POINTS[1:]].all())

    def exact_case(self, side, mode):
        solver = Wuji2StreamingRetargeter(side, mode, thresholds={
            "max_joint_step_rad": 1.0,
            "max_root_translation_step_m": 1.0,
            "max_root_rotation_step_rad": 2.0,
        })
        rng = np.random.default_rng(4 if side == "left" else 8)
        q = solver.hand.lo * 0.2 + solver.hand.hi * 0.2
        q += rng.uniform(-0.08, 0.08, 20)
        q = np.clip(q, solver.hand.lo + 0.1, solver.hand.hi - 0.1)
        root = root_pose()
        points = camera_points(solver, q, root)
        solver.begin_chunk(q, None)
        result = solver.retarget_frame(points, 1_000_000_000, normalize=False)
        self.assertTrue(result["accepted"], result["diagnostic"])
        translation, angle = pose_error(result["cam_T_root"], root)
        self.assertLess(translation, 1e-4)
        self.assertLess(angle, np.deg2rad(0.05))
        self.assertLess(result["diagnostic"]["all_rmse_m"], 1e-4)
        np.testing.assert_allclose(result["q20"], q, atol=1e-4)
        nominal = np.linalg.norm(solver.hand.fk(q)[0])
        self.assertAlmostEqual(result["diagnostic"]["root_to_keypoint0_distance_m"],
                               nominal, places=9)

    def test_exact_wuji_fk_recovers_random_root_and_joints(self):
        for side in ("left", "right"):
            for mode in MODES:
                with self.subTest(side=side, mode=mode):
                    self.exact_case(side, mode)

    def test_human_bone_lengths_normalize_and_both_modes_converge(self):
        for mode in MODES:
            solver = Wuji2StreamingRetargeter("left", mode, thresholds={
                "max_joint_step_rad": 1.0,
                "max_root_translation_step_m": 1.0,
                "max_root_rotation_step_rad": 2.0,
            })
            q = np.clip(np.linspace(-0.2, 0.35, 20), solver.hand.lo + 0.05,
                        solver.hand.hi - 0.05)
            root = root_pose()
            exact = camera_points(solver, q, root)
            human = exact.copy()
            for index in range(1, 21):
                parent = PARENTS[index]
                human[index] = human[parent] + 1.18 * (exact[index] - exact[parent])
            solver.begin_chunk(q, root)
            result = solver.retarget_frame(human, 1_000_000_000, normalize=True)
            self.assertTrue(result["accepted"], result["diagnostic"])
            self.assertLess(result["diagnostic"]["tip_rmse_m"], 0.003)

    def test_chunk_boundary_resets_time_and_rejected_frame_does_not_raise(self):
        solver = Wuji2StreamingRetargeter("right", "free_root")
        q = solver.hand.neutral.copy()
        root = root_pose()
        points = camera_points(solver, q, root)
        solver.begin_chunk(q, root)
        self.assertTrue(solver.retarget_frame(points, 100, normalize=False)["accepted"])
        rejected = solver.retarget_frame(np.zeros((21, 3)), 101)
        self.assertFalse(rejected["accepted"])
        self.assertIn("degenerate", rejected["diagnostic"]["reason"])
        solver.begin_chunk(q, root)
        self.assertTrue(solver.retarget_frame(points, 1, normalize=False)["accepted"])

    def test_rejected_timestamp_is_still_consumed_within_chunk(self):
        solver = Wuji2StreamingRetargeter("right", "free_root")
        q = solver.hand.neutral.copy(); root = root_pose()
        solver.begin_chunk(q, root)
        self.assertFalse(solver.retarget_frame(np.zeros((21, 3)), 200)["accepted"])
        answer = solver.retarget_frame(camera_points(solver, q, root), 150,
                                       normalize=False)
        self.assertFalse(answer["accepted"])
        self.assertIn("strictly increasing", answer["diagnostic"]["reason"])

    def test_runtime_and_model_joint_names_are_explicitly_separate(self):
        solver = Wuji2StreamingRetargeter("left")
        self.assertEqual(solver.runtime_joint_names[0], "left_finger1_joint1")
        self.assertEqual(solver.runtime_joint_names[-1], "left_finger5_joint4")
        self.assertEqual(solver.model_joint_names[0], "l_thumb_cmc_flex")
        self.assertNotEqual(solver.runtime_joint_names, solver.model_joint_names)

    def test_neutral_kabsch_regression_is_not_mistaken_for_rigid_points(self):
        from hand_retarget_wuji2 import FREE_ROOT_INITIAL_POINTS, rigid_align
        solver = Wuji2StreamingRetargeter("left")
        q = np.clip(np.linspace(-0.5, 0.7, 20), solver.hand.lo + 0.05,
                    solver.hand.hi - 0.05)
        root = root_pose(); goal = camera_points(solver, q, root)
        rotation, translation = rigid_align(
            solver.neutral_points[FREE_ROOT_INITIAL_POINTS],
            goal[FREE_ROOT_INITIAL_POINTS])
        neutral_root = np.eye(4); neutral_root[:3,:3] = rotation
        neutral_root[:3,3] = translation
        translation_error, angle_error = pose_error(neutral_root, root)
        self.assertTrue(translation_error > 1e-3 or angle_error > np.deg2rad(1))

    def test_threshold_rejection_preserves_previous_state(self):
        solver = Wuji2StreamingRetargeter("left", "free_root",
                                          thresholds={"max_joint_step_rad": 1e-5})
        q = solver.hand.neutral.copy()
        root = root_pose()
        solver.begin_chunk(q, root)
        first = camera_points(solver, q, root)
        self.assertTrue(solver.retarget_frame(first, 1, normalize=False)["accepted"])
        moved = np.clip(q + 0.15, solver.hand.lo + 0.01, solver.hand.hi - 0.01)
        answer = solver.retarget_frame(camera_points(solver, moved, root), 2, normalize=False)
        self.assertFalse(answer["accepted"])
        np.testing.assert_allclose(solver.q, q, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

