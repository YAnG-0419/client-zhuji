# replay_hand

Cross-embodiment replay for episode395: Franka/Sharpa EEF and 21-keypoint hand motion to Tron2/Wuji Hand 2.

A parallel Pinocchio IK + ServoJ-only entry is available as `replay_hand_pinocchio.sh`.
See [README_PINOCCHIO.md](README_PINOCCHIO.md). The original entry below retains its ServoP source stream.

The local `execution_cross.py` adapts `replay/replay_servop.py` orchestration and reuses its source streaming,
bounded IK verification, continuous swept collision checks, Wuji targets and feedback supervision.
`servo_stages.py` implements the measured-start ServoJ approach and ServoP takeover. Because no recorded
Tron2 joints exist, `anchor_cross.py` places frame zero at a measured reachable Tron2 pose and preserves the source
relative mount translation and wrist rotation. `planner_cross.py` then generates a continuous previous-solution IK
seed path. The absolute cross-workcell camera pose is therefore not preserved. The ServoP EEF hard stop is 4 cm,
matching the validated `test_ik/servo_test.py` test.

Offline by default:

```bash
bash replay_hand/replay_hand.sh
bash replay_hand/replay_hand.sh --end-seconds 2
bash replay_hand/replay_hand.sh --end-seconds 2 --check
```

Use `replay_hand.sh` rather than invoking the Python entry point directly. The
wrapper sources the installed ROS 2 workspace so an execute run can resolve
the disabled-by-default Wuji driver and `wuji_control` packages.

`--execute` is the only mode that may command motion. Do not use it until the complete offline plan and read-only
check pass, the right-hand mount rotation is independently verified, and the physical scene is cleared. The first
hardware trial should use `--end-seconds 2 --speed 0.05 --rate 300` with an operator on the physical e-stop.

The new preparation flow, authorized on 2026-11-10, follows test_ik's measured-start principle:

1. Prepare the hands at frame zero through wuji_control.
2. Certify the latest measured joint pose to the original frame-zero target. Send a 300 Hz
   quintic ServoJ approach, at least 20 seconds, with 16 joint values, filter_ratio=1 and
   measured head-angle pass-through. First command equals measured start; planned arm
   speed is at most 0.30 rad/s. The operator's 0.1 rad approach tolerance remains in effect.
3. Re-read the measured pose and certify its short connection to frame zero. Start ServoP
   at FK of this actual pose, hold for 1 second, then smoothly connect to the original
   first-frame pose over at least 3 seconds. Hold hands/source phase at frame zero.
4. Require stationary feedback within the existing 0.02 rad ServoP start gate for 0.3 seconds,
   then start the original source stream. Both preparation stages allow up to 3 seconds of
   additional target streaming to settle. No whole-episode reanchoring is performed here.

For supervised preparation-only validation (does move arms and hands):

```bash
bash replay_hand/replay_hand.sh --end-seconds 0.5 --speed 0.05 --rate 300 --prepare-only --execute
```

Only after this succeeds, remove `--prepare-only` for a 0.5-second source replay. The logs
`servoj_approach.jsonl` and `servop_takeover.jsonl` include desired/measured joints and EEFs,
feedback age, and elapsed time. Result summaries include actual sending rate and interval
p99/max. Hardware acceptance of the new preparation flow is not established by offline tests.

Known boundaries:

- The operator requested a 1.0 rad/s measured arm speed ceiling on 2026-11-10. This applies
  to replay_hand feedback supervision; planned joint trajectory speed remains capped at
  0.30 rad/s, and `--speed` still controls source-time progression. Position deviation,
  head holding, EEF tracking and hand guards are unchanged. ServoP exceptions after motion
  commands explicitly send a stop request even if the inherited stream cleared `armed`;
  request delivery is recorded separately from physical stop confirmation.

- The former MoveJ approach and its zero extra-wait policy are retained only as legacy code
  in `approach_cross.py`; the active execution path uses the ServoJ/ServoP stages above.

- Franka cam0 and Tron2 top-camera frame identity is an operator assumption.
- Frame zero is anchored to the stationary pose physically reached in the first supervised approach trial because
  both the absolute episode395 pose and the earlier idealized Wanchi anchor were not reached by live MoveJ; source
  motion relative to frame zero is preserved. The source trial sent one MoveJ and never started ServoP.
- Left `R_hand_eef` comes from `wanchi/replay_axis_20261104`; right currently uses the same matrix under a symmetric
  installation assumption and is explicitly marked unvalidated.
- The per-frame free-root geometry adapter is not applied to the arm EEF.
- Arm collision certification excludes Wuji hand bodies, adapters, cables, table and manipulated objects.
- ServoP tests were repeatable but showed roughly 28 mm maximum EEF error on a 30 mm right-arm circle.
- The ServoP entry's hand first-frame tolerance is 0.20 rad, authorized on 2026-11-10 after
  a trial plateaued at 0.180 rad. The runtime hand tracking hard stop remains 0.20 rad.
  `hand_preparation.jsonl` records per-hand q20 feedback and the worst joint; this acceptance
  threshold does not establish accurate physical hand tracking. The parallel Pinocchio
  entry retains its previously configured 0.15 rad first-frame tolerance.
