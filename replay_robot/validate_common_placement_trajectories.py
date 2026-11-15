"""Validate common-placement trajectories offline with no device interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose_errors(solver, q, target):
    position, rotation = solver.fk(q)
    return (np.linalg.norm(position - target[:, :3], axis=1),
            (Rotation.from_matrix(rotation).inv()
             * Rotation.from_quat(target[:, 3:])).magnitude())


def solve_path(root, t, targets, optimizer, initial_path=None):
    sys.path.insert(0, str(root / "Pinocchio"))
    from pin_model import PinArms
    import pinocchio as pin

    solver = PinArms()
    sample_t = np.asarray(optimizer["sample_times_s"], dtype=float)
    sample_q = np.asarray(optimizer["sample_q16"], dtype=float)
    interpolated = np.column_stack([
        np.interp(t, sample_t, sample_q[:, joint]) for joint in range(16)
    ])
    interpolated[:, 14:] = sample_q[0, 14:]
    if initial_path is not None:
        with np.load(initial_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["t"], t)
            if "q16" in archive:
                interpolated = archive["q16"].copy()
            elif "diagnostic_q16" in archive:
                interpolated = archive["diagnostic_q16"].copy()
            else:
                raise ValueError("Initial path has no q16 array")
    path = np.empty((len(t), 16))
    position_error = np.empty((len(t), 2))
    rotation_error = np.empty((len(t), 2))
    fallback_frames = []
    failed_frames = []
    for frame, target in enumerate(targets):
        reference = (interpolated[frame] if frame == 0 else path[frame - 1]).copy()
        reference[14:] = sample_q[0, 14:]
        fast_seed = interpolated[frame].copy()
        if frame and np.max(np.abs(fast_seed[:14] - reference[:14])) > .2:
            fast_seed = reference.copy()
        fast_seed[14:] = sample_q[0, 14:]
        q, _ = solver.solve(fast_seed, target, branch=fast_seed, rot_weight=.5,
                            max_iter=20, tol=(.015, np.deg2rad(1.5)))
        q[14:] = sample_q[0, 14:]
        fast_position, fast_rotation = pose_errors(solver, q, target)
        for arm in range(2):
            sl = slice(7 * arm, 7 * arm + 7)
            fast_pass = (fast_position[arm] <= .020
                         and fast_rotation[arm] <= np.deg2rad(2.))
            fast_step = float(np.max(np.abs(q[sl] - reference[sl])))
            if fast_pass and (frame == 0 or fast_step <= .20):
                continue
            # First search only the neighborhood of the preceding solution.
            # This prevents a redundant 7-DoF arm from switching branches just
            # because an unconstrained least-squares solution is more exact.
            target_rotation = Rotation.from_quat(target[arm, 3:]).as_matrix()
            work = q.copy()
            local_best = None
            for cap in (.03, .08, .19):
                cap_candidates = []
                lower = np.maximum(solver.lower[sl] + 1e-5, reference[sl] - cap)
                upper = np.minimum(solver.upper[sl] - 1e-5, reference[sl] + cap)
                initial = np.clip(reference[sl], lower + 1e-10, upper - 1e-10)
                for weight in (.1, .2, .3, .4, .5):
                    def residual(values):
                        work[sl] = values
                        pin.framesForwardKinematics(
                            solver.model, solver.data, solver.full(work))
                        placement = solver.data.oMf[solver.frames[arm]]
                        return np.r_[
                            placement.translation - target[arm, :3],
                            weight * pin.log3(
                                placement.rotation @ target_rotation.T),
                        ]
                    trial = q.copy()
                    trial[sl] = least_squares(
                        residual, initial, bounds=(lower, upper), max_nfev=500,
                        ftol=1e-11, xtol=1e-11, gtol=1e-11).x
                    pe, re = pose_errors(solver, trial, target)
                    passing = pe[arm] <= .020 and re[arm] <= np.deg2rad(2.)
                    step = float(np.max(np.abs(trial[sl] - reference[sl])))
                    normalized = max(pe[arm] / .020,
                                     re[arm] / np.deg2rad(2.))
                    cap_candidates.append((not passing,
                                           step if passing else normalized,
                                           normalized,
                                           f"continuous_step_limited_{cap:.2f}",
                                           weight, trial, pe, re))
                passing_at_cap = [row for row in cap_candidates if not row[0]]
                if passing_at_cap:
                    local_best = min(passing_at_cap, key=lambda row: row[:3])
                    break
            if local_best is not None:
                _, _, _, label, weight, q, pe, re = local_best
                fallback_frames.append(dict(frame=frame, arm=arm,
                                            time_s=float(t[frame]), weight=weight,
                                            method=label))
                continue

            candidates = []
            for label, seed in (("continuous", reference),
                                ("sample_interpolation", interpolated[frame])):
                # Around joint bounds the feasible point can occupy a narrow
                # position/orientation tradeoff band.  Include intermediate
                # weights before considering a different redundant branch.
                for weight in (.3, .35, .4, .425, .45, .5, .573):
                    trial = q.copy()
                    trial[sl] = solver._bounded(seed.copy(), arm, target,
                                                rot_weight=weight)
                    pe, re = pose_errors(solver, trial, target)
                    passing = pe[arm] <= .020 and re[arm] <= np.deg2rad(2.)
                    step = float(np.max(np.abs(trial[sl] - reference[sl])))
                    normalized = max(pe[arm] / .020, re[arm] / np.deg2rad(2.))
                    candidates.append((not passing, step if passing else normalized,
                                       normalized, label, weight, trial, pe, re))
            best = min(candidates, key=lambda row: row[:3])
            _, _, _, label, weight, q, pe, re = best
            if label != "continuous":
                fallback_frames.append(dict(frame=frame, arm=arm,
                                            time_s=float(t[frame]), weight=weight,
                                            method=label))
            if pe[arm] > .020 or re[arm] > np.deg2rad(2.):
                failed_frames.append(dict(frame=frame, arm=arm,
                                          time_s=float(t[frame]),
                                          position_m=float(pe[arm]),
                                          rotation_rad=float(re[arm])))
        path[frame] = q
        position_error[frame], rotation_error[frame] = pose_errors(solver, q, target)
        if frame % 200 == 0:
            print("IK", frame, "/", len(t), flush=True)
    return path, position_error, rotation_error, fallback_frames, failed_frames


def validate(root, episode, source, optimizer_path, output, existing_seed=None,
             initial_path=None):
    began = time.monotonic()
    root = root.resolve()
    source = source.resolve()
    optimizer_path = optimizer_path.resolve()
    output = output.resolve()
    existing_seed = existing_seed.resolve() if existing_seed else None
    initial_path = initial_path.resolve() if initial_path else None
    sys.path[:0] = [str(root / "replay_hand"), str(root / "replay"),
                    str(root / "tools"), str(root / "Pinocchio")]
    from episode_cross import load_cross_episode
    from pin_model import PinArms
    from transport import replay_backend
    import replay_thresholds
    import mujoco

    t, eef, hands, metadata = load_cross_episode(source, 50.)
    optimizer = json.loads(optimizer_path.read_text())
    if existing_seed:
        with np.load(existing_seed, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["t"], t)
            qs = archive["q16"].copy()
        solver = PinArms()
        values = [pose_errors(solver, q, target) for q, target in zip(qs, eef)]
        position_error = np.asarray([value[0] for value in values])
        rotation_error = np.asarray([value[1] for value in values])
        fallback_frames, failed_frames = [], []
    else:
        qs, position_error, rotation_error, fallback_frames, failed_frames = solve_path(
            root, t, eef, optimizer, initial_path)
    pose_pass = ((position_error <= .020)
                 & (rotation_error <= np.deg2rad(2.)))

    # Check joint-linear interpolation against xyz linear + quaternion SLERP.
    solver = PinArms()
    dense_position = []
    dense_rotation = []
    for frame in range(len(t) - 1):
        alpha = np.arange(1, 6) / 6.
        target_rotation = [
            Slerp([0., 1.], Rotation.from_quat(eef[frame:frame + 2, arm, 3:]))(alpha)
            for arm in range(2)
        ]
        for index, fraction in enumerate(alpha):
            q = qs[frame] + fraction * (qs[frame + 1] - qs[frame])
            position, rotation = solver.fk(q)
            xyz = (eef[frame, :, :3]
                   + fraction * (eef[frame + 1, :, :3] - eef[frame, :, :3]))
            dense_position.append(np.linalg.norm(position - xyz, axis=1))
            dense_rotation.append([
                (Rotation.from_matrix(rotation[arm]).inv()
                 * target_rotation[arm][index]).magnitude() for arm in range(2)
            ])
    dense_position = np.asarray(dense_position)
    dense_rotation = np.asarray(dense_rotation)

    # Compute both the currently installed normalized policy and raw 20 mm.
    _, Model, _ = replay_backend()
    model = Model()
    policy_clearance = np.empty(len(t))
    raw_clearance = np.empty(len(t))
    raw_pairs = []
    for frame, q in enumerate(qs):
        model.set_q(q)
        policy_clearance[frame], _ = model.clearance()
        best, pair = float("inf"), None
        for geom_a, geom_b in model.pairs:
            distance = float(mujoco.mj_geomDistance(
                model.model, model.data, geom_a, geom_b, .5, None))
            if distance < best:
                best = distance
                pair = [model.body_name(model.model.geom_bodyid[index])
                        for index in (geom_a, geom_b)]
        raw_clearance[frame] = best
        raw_pairs.append(pair)
        if frame % 300 == 0:
            print("COLLISION", frame, "/", len(t), flush=True)
    worst = int(np.argmin(raw_clearance))
    steps = np.max(np.abs(np.diff(qs[:, :14], axis=0)), axis=1)
    servop_box = np.array([[[.250, .732], [-.213, .900], [-.673, .5]],
                           [[.250, .732], [-.900, .213], [-.673, .5]]])
    box_pass = np.all((eef[:, :, :3] >= servop_box[..., 0])
                      & (eef[:, :, :3] <= servop_box[..., 1]), axis=(1, 2))
    report = dict(
        episode=episode,
        source=str(source),
        common_placement=optimizer["T_target_base_original_base"],
        rotation_xyz_deg=optimizer["rotation_xyz_deg"],
        first_eef_midpoint_m=optimizer["first_eef_midpoint_m"],
        frames=len(t), duration_s=float(t[-1]),
        motion_commands_sent=0, execution_ready=False,
        original_targets_changed_by_one_common_rigid_transform=True,
        bimanual_relative_pose_preserved=(
            metadata["task_placement"]["bimanual_relative_matrix_error_max"] < 1e-9),
        pose=dict(
            passing_frames_per_arm=pose_pass.sum(axis=0).tolist(),
            both_passing_frames=int(pose_pass.all(axis=1).sum()),
            all_frames_pass=bool(pose_pass.all()),
            max_position_mm=(position_error.max(axis=0) * 1000).tolist(),
            max_rotation_deg=np.rad2deg(rotation_error.max(axis=0)).tolist(),
            failed_frames=failed_frames,
            sample_interpolation_fallbacks=fallback_frames,
        ),
        dense_300hz=dict(
            all_pass=bool(np.all((dense_position <= .020)
                                 & (dense_rotation <= np.deg2rad(3.)))),
            max_position_mm=(dense_position.max(axis=0) * 1000).tolist(),
            max_rotation_deg=np.rad2deg(dense_rotation.max(axis=0)).tolist(),
        ),
        continuity=dict(
            max_joint_step_rad=float(steps.max()),
            max_joint_step_deg=float(np.rad2deg(steps.max())),
            steps_over_0p2rad=int((steps > .2).sum()),
        ),
        collision=dict(
            installed_policy=dict(
                model_clearance_m=replay_thresholds.MODEL_CLEARANCE_M,
                internal_wrist_clearance_m=getattr(
                    replay_thresholds, "INTERNAL_WRIST_CLEARANCE_M",
                    replay_thresholds.MODEL_CLEARANCE_M),
                passing_frames=int((policy_clearance
                                    >= replay_thresholds.MODEL_CLEARANCE_M).sum()),
                minimum_score_mm=float(policy_clearance.min() * 1000),
            ),
            uniform_20mm=dict(
                passing_frames=int((raw_clearance >= .020).sum()),
                minimum_raw_distance_mm=float(raw_clearance[worst] * 1000),
                minimum_pair=raw_pairs[worst],
                minimum_time_s=float(t[worst]),
            ),
            scope="Sampled arm/body/head model pairs; no swept/hand/environment/cable check",
        ),
        servop_position_box=dict(passing_frames=int(box_pass.sum()),
                                 all_frames_pass=bool(box_pass.all())),
        source_sha256={
            str(source.relative_to(root)): sha(source),
            str(optimizer_path.relative_to(root)): sha(optimizer_path),
            "replay_robot/apply_common_placement_task_transfer.py": sha(
                root / "replay_robot/apply_common_placement_task_transfer.py"),
            "replay_robot/validate_common_placement_trajectories.py": sha(Path(__file__)),
            "Pinocchio/pin_model.py": sha(root / "Pinocchio/pin_model.py"),
            "record/config/model/robot.urdf": sha(root / "record/config/model/robot.urdf"),
        },
        elapsed_s=time.monotonic() - began,
        not_checked=["hardware/ServoP envelope and tracking", "MoveJ entry and takeover",
                     "swept collision", "hands/body/environment/cables", "dynamics/retiming"],
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "diagnostics_not_for_execution.npz",
                        t=t, q16=qs, position_error_m=position_error,
                        rotation_error_rad=rotation_error, pose_pass=pose_pass,
                        raw_model_clearance_m=raw_clearance,
                        installed_policy_clearance_score_m=policy_clearance)
    (output / "report.json").write_text(json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("RESULT", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-seed", type=Path)
    parser.add_argument("--initial-path", type=Path)
    args = parser.parse_args()
    validate(args.root, args.episode, args.source, args.optimizer,
             args.output, args.existing_seed, args.initial_path)
