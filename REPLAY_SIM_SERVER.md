# Episode 2 XFlow replay simulator

This server implements the same inference-only `info/reset/infer` wire_v4 API
as the GPU XFlow service. It does not import ROS or any robot transport.

It converts Episode 2's recorded SDK EEF poses and Wuji q20 values to
MediaPipe21 points with the live client's frozen FK/calibration. Each `infer`
returns the next 32 time-sampled points. The episode runs once; subsequent
calls hold the final frame.

Start it on zhuji without disturbing the real-model tunnel on port 28765:

```bash
cd /home/guest/pengyang
bash client/run_replay_sim_server.sh \
  --port 28769 \
  --endpoint-file client/runs/replay-sim-episode2-endpoint.json
```

Use the live client with an explicit local port:

```bash
cd /home/guest/pengyang
bash client/run_xflow_live.sh \
  --relay-port 28769 \
  --task tidying \
  --speed-ratio 2 \
  --anchor-mode constant \
  --sag-comp on \
  --cycles 100
```

Add `--execute` only after the shadow run has produced accepted, collision-free
plans and the robot is supervised by the operator.
