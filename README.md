# TRON2 + Wuji2 XFlow kp21 client（第二轮 dry-run）

本目录已完成“在线数据链 + 离线 dry-run”，但**没有硬件发送入口**。代码不会启动驱动、
发布控制话题、调用 ServoP/ServoJ/Wuji command 或紧急停止。所有报告都固定记录
`execution_ready=false`、`hardware_commands_sent=0`。

主要模块：

- `hand_retarget_wuji2.py`：逐帧 free-root 与 q-conditioned Kabsch fixed-root Wuji2 IK；
- `proprio.py`：实测 SDK EEF + Wuji q20 → 相机系 kp21，以及反向 EEF 工具链；
- `image_preprocess.py`：NumPy `virtual_cam0` / `stretch_undistort` → `rgb320`；
- `xflow_rpc.py`：只暴露 `info/reset/infer` 的严格 wire_v4 client；
- `observation_buffer.py`、`ros_readonly_observer.py`：fail-closed 的只读 ROS 环形缓冲；
- `bag_observation_source.py`：sqlite `mode=ro` 的离线 bag 数据源；
- `timing_policy.py`：默认 32 点 × 30 Hz = 1.0667 s，不存在 0.25 倍速入口；
- `execution_postprocess.py`：局部右乘锚定、窗口平滑、静止检查、含首跳的速度门；
- `online_dryrun.py`：fake-gt/remote/cache replay、多快照、50 Hz 重采样、Pinocchio IK 和
  1 rad/s 速度审计；
- `live_dryrun.py`：系统 ROS 只读快照进程 + pixi 规划进程；缺话题时在 infer 前退出；
- `relay_mac.py`：Mac 上两条仅绑定 loopback 的 SSH 中继；
- `docs/execution_layer_design.md`：未来发送层设计，不含发送代码。

时间约定按用户最新决定：`hh_*` 返回的第 1…32 点分别占一个 30 Hz 执行帧，deadline 为
`dispatch + reserve + (1…32)/30 s`。服务端原生 1/120…32/120 s 仅保留为诊断模式。
默认同步保持推理期间的采样姿态，并按执行时钟取 16 帧（0.533 s）前的历史图。

常用离线命令：

```bash
cd /home/guest/pengyang/client

/home/guest/.pixi/bin/pixi run --locked python -m pytest -q

/home/guest/.pixi/bin/pixi run --locked python online_dryrun.py \
  --source testdata/episode023_i800_state.npz \
  --server fake-gt --root-mode both \
  --time-scale frame_per_step_30hz --steps 32

# 影子模式：只订阅、推理、规划和保存；无 publisher/sender。
/home/guest/.pixi/bin/pixi run --locked python live_dryrun.py \
  --task flower --cycles 1
```

旧 Orbbec/replay camera chunk 必须显式选 `board_transfer`；实时 zhuji 图像/关键点必须选
`live_zhuji_camera`（`M=I`）。例如旧编译器现在要求明确传：

```bash
/home/guest/.pixi/bin/pixi run --locked python online_client.py \
  --camera-mapping board_transfer \
  --chunk ../replay_robot/camera_trajectory_adc619f1_20261114/inputs/wuji2_camera_replay_trajectory.npz
```

第一轮报告见 `docs/IMPLEMENTATION_REPORT_R1.md`，第二轮结果见
`docs/XFLOW_CLIENT_REPORT_R2.md`。当前不能进入硬件阶段：实时
`/wuji/*/joint_states` 没有 publisher；真实多快照仍有 1 rad/s 速度门失败。

