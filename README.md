# TRON2 + Wuji2 XFlow kp21 client（阶段一）

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
- `online_dryrun.py`：fake-gt/remote、双根模式、50 Hz 重采样、EEF_BOX、Pinocchio IK 和
  1 rad/s 速度审计；
- `relay_mac.py`：Mac 上两条仅绑定 loopback 的 SSH 中继；
- `docs/execution_layer_design.md`：未来发送层设计，不含发送代码。

时间约定按用户最新决定：`hh_*` 返回的第 1…32 点分别占一个 30 Hz 发送帧，deadline 为
`exec_start + (1…32)/30 s`。服务端原生 1/120…32/120 s 仅保留为诊断模式。历史图仍严格
按 info 的 4 个相机帧选取。

常用离线命令：

```bash
cd /home/guest/pengyang/client

/home/guest/.pixi/bin/pixi run --locked python -m pytest -q

/home/guest/.pixi/bin/pixi run --locked python online_dryrun.py \
  --source testdata/episode023_i800_state.npz \
  --server fake-gt --root-mode both \
  --time-scale frame_per_step_30hz --steps 32
```

旧 Orbbec/replay camera chunk 必须显式选 `board_transfer`；实时 zhuji 图像/关键点必须选
`live_zhuji_camera`（`M=I`）。例如旧编译器现在要求明确传：

```bash
/home/guest/.pixi/bin/pixi run --locked python online_client.py \
  --camera-mapping board_transfer \
  --chunk ../replay_robot/camera_trajectory_adc619f1_20261114/inputs/wuji2_camera_replay_trajectory.npz
```

完整实现、测试、benchmark、通信和已知阻塞项见 `IMPLEMENTATION_REPORT.md`。当前不能进入
硬件阶段：实时 `/wuji/*/joint_states` 没有 publisher；41.5° 外参仍标记未物理验收；两种
远端图像路径的预测均出现 1 rad/s 速度门失败。

