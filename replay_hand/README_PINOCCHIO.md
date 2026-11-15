# Pinocchio IK + ServoJ replay

Parallel entry: `replay_hand_pinocchio.sh`. The original `replay_hand.sh` continues
to use the ServoP source stream and remains available.

The new path is episode395 EEF -> Pinocchio IK -> bounded, continuous joint path
-> ServoJ q16 at 300 Hz, with the existing Wuji Hand 2 q20 synchronized to source
time. It retains the same original frame-zero anchor and EEF conversion; it does
not change the coordinate mapping or retarget the fingers again.

Pinocchio IK reuses `/home/guest/pengyang/Pinocchio/pin_model.py`: damped Jacobian
iterations with a seed nullspace preference, plus bounded optimization using
Pinocchio FK when needed. No MuJoCo IK fallback is used. The existing MuJoCo model
is retained for independent FK verification and swept collision certification.

The runtime transport rejects ServoP, MoveJ and MoveP requests. Preparation uses
the measured-start ServoJ approach, followed by a certified ServoJ alignment to
the original first frame. The source clock remains stopped and the hands hold
frame zero until alignment finishes. Head angles pass through live feedback as
in test_ik; unintended head displacement remains guarded.

Offline and read-only checks:

```bash
cd /home/guest/pengyang
bash replay_hand/replay_hand_pinocchio.sh --speed 0.05
bash replay_hand/replay_hand_pinocchio.sh --end-seconds 0.5 --speed 0.05 --check
```

Supervised preparation-only test (moves arms and hands, no source replay):

```bash
bash replay_hand/replay_hand_pinocchio.sh \
  --end-seconds 0.5 --speed 0.05 --rate 300 --prepare-only --execute
```

After preparation succeeds, remove `--prepare-only` for a 0.5-second replay,
then extend to 2 seconds and 30 seconds as hardware behavior is confirmed.
Omitting `--end-seconds` selects the full 37.850-second source recording.

Runs are separate: `replay_hand/runs/pinocchio_servoj_<timestamp>_<pid>/`.
`plan_servoj.npz` contains joint references and planned times; `pinocchio_ik.json`
records solver provenance and residuals. `servoj_approach.jsonl`,
`servoj_alignment.jsonl` and `servoj_stream.jsonl` contain preparation/source
feedback. Result summaries include actual command rate and interval p99/max.

Planned arm speed remains at most 0.30 rad/s. The operator-selected measured arm
speed ceiling is 1.0 rad/s; the existing 3-degree joint deviation, 4 cm / 0.20 rad
EEF tracking and Wuji feedback guards remain. ServoJ alignment and final settling
require less than 0.02 rad joint error and stationary feedback. Exceptions retain
the armed state for executor stop cleanup, and any failed run after arm commands
attempts a stop request even if another component cleared that state.

Offline verification on 2026-11-10: the complete 1894-frame source passed Pinocchio
IK, limits, continuity and dense swept collision checks; maximum cross-seed joint
step was 0.024190 rad, conservative clearance score was 0.010604 m. 38 hardware-free
tests passed. This does not establish physical replay acceptance. Existing limits
still apply: assumed camera alignment, unverified right mount rotation, and a
collision model that excludes Wuji bodies, adapters, cables and external objects.
