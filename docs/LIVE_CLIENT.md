# XFlow 在线 client（zhuji：TRON2 + Wuji Hand 2）

## 启动

1. Mac 上启动中继（Mac 能同时连 b300 和 zhuji；zhuji 不能上外网）：

   ```bash
   cd /Users/lpy/Documents/ws/Xflow/zhuji-client
   ./mac_relay.sh xflow-4srv-hh15k-rb30k-5090-1115/flower-hh   # <任务>/<服务>；单服务任务只写任务名
   # 脚本从该服务的 endpoint.json 读 host 和 port
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
   任何停止（正常结束、Ctrl-C、安全检查触发、看门狗超时）都**不发 `request_emgy_stop`**：只停止发送 ServoP，
   TRON2 控制器在 ServoP 超时后保持当前位姿、保持高级开发者模式；client 确认静止和 working_mode 后关闭手部输出。
   未能确认时打印 WARNING（同样不发急停）。现场物理急停是最后一道保护。

## 每个周期

保持当前指令位姿 → 静止 0.2 s 后取头部无损图像（`/camera/top/color/image_raw`）、SDK EEF、Wuji q20 →
规划子进程：当前关键点（SDK EEF + 标定工具链 + Wuji FK）→ 虚拟相机重投影到训练相机（`virtual_cam0`）→
XFlow 推理 → free-root Wuji2 重定向（骨长归一化）→ 平滑（窗口 5）→ 手根局部右乘锚定到当前指令位姿
（衰减步数按速度预算自动加长）→ SDK EEF 目标 → Pinocchio IK + 双臂/手-身体扫掠碰撞检查 →
32 步每步 1/30 s，只在速度上限（臂 0.8 rad/s、手 0.9 rad/s、EEF 0.25 m/s / 1.0 rad/s）约束时局部拉长 →
300 Hz ServoP + 50 Hz Wuji 执行 → 回到保持。历史帧取执行时间轴上 16 帧前（停顿不计）。

启动时：若某个手臂关节实测略超出 SDK 指令范围（≤0.05 rad，通常是 ServoP 在关节上限处的过冲），先用 1.5 s 把它缓慢收回范围内 0.02 rad 再开始；超出更多则直接停。规划时手臂关节也保持在范围内 0.02 rad。
保持阶段：两臂负载下静止时会比指令低 1–2 cm；每段结束后要等实测回到指令 2 cm / 3° 以内才采下一帧，等待超过 3 s 才停。
监督沿用 replay_hand：臂反馈速度 1.2 rad/s、实时碰撞间隙、关节范围、头部不动、EEF 跟踪 4 cm / 0.2 rad、
手部限位；单步指令跳变 6 mm / 0.03 rad 以上视为程序错误。触发后一律停发、由控制器保持（见上）。规划被拒（重定向失败、IK 不可达、
碰撞、超出 ServoP 工作空间）时继续保持并重新采样，连续 3 次被拒则停止。

## 常用参数

`--model <名字>`（默认用服务的 default_model，一个服务一个模型时不用填）、`--speed-ratio R`（默认取服务端：hh 4、rb 1；同时决定历史帧间隔 round(训练历史帧×R) 和每步执行时间，范围 0.25–8）、`--virtual-preset cam0_newk|hh_435|hh_335`、
`--image-mode stretch_undistort`、`--steps N`（每个 chunk 只执行前 N 步）、`--denoise-steps 5`、
`--arm-cap/--hand-cap/--linear-cap/--angular-cap`（只能调低到 replay 的上限以内）。

## 产物

`client/runs/live-{shadow,exec}-*/` 中每次推理用唯一 `plan_id` 贯穿模型、规划、发送和反馈：

- `audit_plans/planNNN.npz`：实际送给模型的 320×320 head/history 图、当前 kp21、服务原始
  kp21 输出与 `keypoint_time_s`、采样时的 EEF/q16/q20，以及最终规划 EEF/q16/q20；
- `audit_plans/planNNN.json`：task/model/deployment/request/session、RTT、服务各阶段耗时、规划报告；
  即使重定向、IK 或碰撞门拒绝，也保留已经产生的模型输入/输出和拒绝原因；
- `audit_executions/execNNN.npz`：300 Hz 实际 ServoP 目标及发送时间，50 Hz 目标与实测
  SDK EEF、q16/dq16、Wuji q20，和逐样本 EEF/q20/q16 误差；
- `audit_executions/execNNN.json`：命令/反馈频率，跟踪误差 mean/p50/p95/max、左右臂最大误差、
  是否触发运行时 EEF 门。q16 只作参考，因为 ServoP 可能选择不同的冗余 IK 分支；
- `audit_summary.json`：按 `plan_id` 配对后的快速索引。若规划目标正确但反馈误差大，优先查真机
  执行；若反馈紧跟目标但动作语义不对，优先查模型输入/输出与重定向。

原有 `result.json`、`cycles.jsonl`、`planNNN_targets.npz`、`arm_feedback.jsonl`、
`hand_feedback.jsonl`、`robot_wire.jsonl` 和 `stack.log` 继续保留。影子模式也生成 plan/feedback
记录，但明确标记 `tracking_comparison_valid=false`，不能用它评价真机跟踪性能。相机帧和数组由
单独的后台线程以不压缩 NPZ 写盘，避免压缩或磁盘延迟阻塞 300 Hz ServoP 发送循环。
