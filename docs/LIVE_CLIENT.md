# XFlow 在线 client（zhuji：TRON2 + Wuji Hand 2）

## 启动

1. Mac 上启动中继（Mac 能同时连 b300 和 zhuji；zhuji 不能上外网）：

   ```bash
   cd /Users/lpy/Documents/ws/Xflow/zhuji-client
   ./mac_relay.sh aw-hha12-tidy-5090-1114     # 参数是 5090 上的任务名，脚本自动读 endpoint
   ```

   看到 `relay ready on zhuji 127.0.0.1:28765` 后保持这个终端开着。

2. zhuji 上（机器人 High-Level Dev、碰撞保护开、柔顺关、VR 头部跟随关；先退出遥操/回放/home；建议先 `pixi run --locked home --execute` 回 home）：

   ```bash
   cd /home/guest/pengyang
   bash client/run_xflow_live.sh --task flower --cycles 3             # 影子模式：不发任何命令
   bash client/run_xflow_live.sh --task flower --execute --step       # 真机，每个 chunk 前按 Enter
   bash client/run_xflow_live.sh --task flower --execute              # 真机连续运行，Ctrl-C 停
   ```

   没有在运行的 Wuji 驱动时，client 会自己启动（电机默认不使能）；`--execute` 才会使能手、发 ServoP。
   Ctrl-C：停止发送，确认控制器保持在当前位姿后关闭手部输出。故障：急停。

## 每个周期

保持当前指令位姿 → 静止 0.2 s 后取头部无损图像（`/camera/top/color/image_raw`）、SDK EEF、Wuji q20 →
规划子进程：当前关键点（SDK EEF + 标定工具链 + Wuji FK）→ 虚拟相机重投影到训练相机（`virtual_cam0`）→
XFlow 推理 → free-root Wuji2 重定向（骨长归一化）→ 平滑（窗口 5）→ 手根局部右乘锚定到当前指令位姿
（衰减步数按速度预算自动加长）→ SDK EEF 目标 → Pinocchio IK + 双臂/手-身体扫掠碰撞检查 →
32 步每步 1/30 s，只在速度上限（臂 0.8 rad/s、手 0.9 rad/s、EEF 0.25 m/s / 1.0 rad/s）约束时局部拉长 →
300 Hz ServoP + 50 Hz Wuji 执行 → 回到保持。历史帧取执行时间轴上 16 帧前（停顿不计）。

监督沿用 replay_hand：臂反馈速度 1.2 rad/s、实时碰撞间隙、关节范围、头部不动、EEF 跟踪 4 cm / 0.2 rad、
手部限位；单步指令跳变 6 mm / 0.03 rad 以上视为程序错误并急停。规划被拒（重定向失败、IK 不可达、
碰撞、超出 ServoP 工作空间）时继续保持并重新采样，连续 3 次被拒则停止。

## 常用参数

`--model hh_a12|rb_a12`（需要中继到对应服务）、`--virtual-preset cam0_newk|hh_435|hh_335`、
`--image-mode stretch_undistort`、`--steps N`（每个 chunk 只执行前 N 步）、`--denoise-steps 5`、
`--arm-cap/--hand-cap/--linear-cap/--angular-cap`（只能调低到 replay 的上限以内）。

## 产物

`client/runs/live-{shadow,exec}-*/`：`result.json`、`cycles.jsonl`（每周期往返、各阶段耗时、拉长比例、
锚定偏移、历史帧）、`cycleNNN.npz`（预测关键点、EEF/q20/q16 目标、时间）、`arm_feedback.jsonl`、
`hand_feedback.jsonl`、`robot_wire.jsonl`、`stack.log`。
