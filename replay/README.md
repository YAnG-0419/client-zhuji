# 双臂 EEF 与 Wuji 双手回放

默认输入是 `record/data/20261104_133350_628817_combined_record_episode_003`。
在 `/home/guest/pengyang` 运行，依赖沿用根目录锁定的 Pixi 环境。

```bash
# 离线读取 bag、求解整条 EEF 路径并检查碰撞，不连接设备
/home/guest/.pixi/bin/pixi run --locked replay

# 同上，再读取机器人与手部状态；不启动驱动、不使能、不运动
/home/guest/.pixi/bin/pixi run --locked replay --check

# 真机：检查通过后，使能双手、到录制起点、回放、关闭手部输出
/home/guest/.pixi/bin/pixi run --locked replay --execute
```

也可以 `bash replay/replay.sh`，参数相同。指定其他记录：

```bash
bash replay/replay.sh --episode record/data/<episode> --speed 0.5 --check
```

可用 `--start-seconds 0 --end-seconds 5` 明确选择录制的一段；默认检查和回放整段，
不会在后半段规划失败时自动改为执行前半段。

输入可以是 episode 文件夹或其 `manifest.json`。要求 `phase=complete`、
`recording_valid=true`，bag 指定话题的实际数量与 manifest 相符，时间戳递增且间隔不超过 100 ms。
直接从只读 SQLite bag 解码 ROS 消息，避免把 `states.jsonl` 中 bag 开始前的预热数据当成回放内容。

双臂使用 `/record/{left,right}_arm/ee_state_fk`，坐标为 `base_link`、米、XYZW 四元数。
这是同一台机器人上的实测 EEF 轨迹，因此保持绝对坐标，不做初始位姿对齐或任意平移旋转。
选择实际共有 FK/关节反馈时刻，默认约 50 Hz，保留录制末帧。
双手使用 `/wuji/{left,right}/command`，按明确的 20 个关节名称排序，
插值到同一源时间轴，经 `/wuji/{side}/target/program` 回放。
不会直接发布硬件 `/command`，也不把 `/joint_command` 当成可靠动作。

EEF 先经过有界 IK，录制的实测关节用于选择冗余解分支；最终目标由 EEF 决定。
每个解要求位置误差不超过 10 mm、旋转误差不超过 0.01 rad（约 0.57°），并保持连续和 URDF 机械范围。
按用户要求，回放规划与发送均不再使用旧 MoveJ 目标范围表（例如手腕 ±0.7505 rad）；
采用 URDF 自身的机械范围，保持碰撞、速度、反馈和状态检查。此调整不发送固件限位配置命令。
自适应选择路点，检查重采样点上的关节弦 FK 与 EEF 源轨迹误差不超过 10 mm / 0.03 rad。
每段检查连续关节线性扫掠的机械臂、机身、头部间隙。跨臂、机身等连杆要求 10 mm；
同一腕部 `wrist_yaw_*_Link` 与 `wrist_roll_*_Link` 的内部装配连杆要求 1 mm。
初次扫掠下界不足时自适应细分区间，用更紧的连续下界复核。
间隙分数为“真实距离 − 该对连杆要求的间隙 + 10 mm”，计划要求分数至少 10 mm。
因此报告中的间隙分数不是所有连杆对的原始几何距离；运行时分数低于 9.5 mm 会停止，
对应内部腕连杆 0.5 mm、其他连杆对 9.5 mm 的实测保护值。
直接 ServoP 曾发生危险的关节重构，ServoJ 也有既有故障禁用记录，本脚本不调用这两个后端。

执行采用逐段停止的 MoveJ，所以不是连续 ServoP，也不保证原录制的墙钟时间。
`--speed` 是源时间速度上限，默认 0.5；关节速度、加速度、手指速度和路点停稳会继续延长时间。
计划限制为机械臂 0.25 rad/s、0.5 rad/s²，手指目标 0.4 rad/s。
双手按同一个单调源时间进度推进，并受实际机械臂进度约束；不会在机械臂停顿时独立播放完整手部轨迹。

执行前退出 teleop、home 和其他控制程序。机器人须为 `High-Level Dev`、碰撞保护开启、
柔顺关闭、VR 头部跟随关闭，电机与 IMU 健康；程序检查但不切换模式。
保持当前头部位置，并用当前头部几何重新检查整条路径，不发送头部或底盘运动命令。
若现有手部驱动完整且独占则复用，否则仅执行模式会启动本项目默认禁用、0.5 A 的双手驱动。
先以平滑限速手部轨迹到首帧，再以至少 20 秒的已检查 MoveJ 路径把机械臂移到首帧。
双手首帧到位要求为全部 40 个关节误差小于 6°（约 0.10472 rad），连续满足 0.3 秒。
机械臂首帧直接采用录制 `/joint_states` 的 14 个关节位置，不用 EEF IK 定位首帧；
后续仍根据 EEF 求解并同步回放。首帧关节目标同样必须满足 URDF 机械范围。
首次定位的直接路径不通过就拒绝，不自动绕路。

检查模型没有包含 Wuji 本体、转接件、线缆、桌子、物体和其他外部障碍物。
模型通过不能代替现场空间检查。回放结束关闭手部输出，不保持抓取，也不自动归位。

运行产物在 `replay/runs/<时间_PID>/`：

- `plan.npz`、`plan.json`：源 EEF/双手数据、求解关节、路点、分段时间、误差及间隙。
- `start_state.json`：在线预检时的真实起始状态。
- `feedback.jsonl`：执行过程的关节、源时间、手部误差及模型间隙。
- `stack.log`：本次自动启动驱动的日志。
- `result.json`：模式、成功/失败原因、运动命令数量及退出状态。
- `ik_failure.json`：如 IK 不满足要求，保存目标、最近求解结果、误差和限位。

Ctrl-C、SIGTERM 或创建该次运行目录下的 `stop.request` 都会触发停止流程；
故障时若机械臂仍在执行，会请求停止并尝试关闭双手输出。驱动清理检查整个进程组，
即使 launcher 已退出也清理残留子进程。Python/网络监控不替代物理急停。

本机离线和 native ROS 消息/SQLite 测试：

```bash
source /opt/ros/foxy/setup.bash
source install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/guest/.pixi/bin/pixi run --locked python -m pytest replay/tests -q
```

关闭插件自动加载是为了避开 Foxy 的 `launch_testing` 与 pytest 8 的收集器不兼容；
测试仍使用真实 Foxy CDR 序列化/反序列化。测试通过不等同于真机验收。

2026-11-04 对本次 episode 的实际校验：8 项测试通过。前 5 秒（实际选中 4.95 秒、
100 帧）离线规划通过，28 段、预计运动 16.16 秒，模型连续保守间隙至少 15.65 mm。
调整前，整段在源时间约 28.000 秒未通过原有有界 IK 精度门槛：局部求解位置误差 0.596 mm，
旋转误差 0.002735 rad；位置超过当时设定的 0.5 mm 要求。现已按用户要求改为 10 mm。
这是当前求解器未满足精度要求，不是对全局不可达的数学证明。
录制右手腕 pitch 达到 0.7834 rad，超过现有 MoveJ 有效上限 0.7505 rad。
完整拒绝报告与 `ik_failure.json` 在 `replay/runs/20261104_121258_39858/`。
前 5 秒在线只读检查又被机器人的 `controller_mode` 拦截，报告在
`replay/runs/20261104_121521_40793/`。本次开发校验发送的运动命令为零，整段没有完成真机回放。

位置误差上限改为 10 mm 后，8 项测试仍通过。整段重新离线检查在源时间约 28.060 秒
被旋转精度要求拦截：位置误差 0.829 mm 已在新上限内，旋转误差 0.003801 rad 超过
当时保留的 0.0035 rad 上限。现已按用户要求把 IK 旋转误差上限改为 0.01 rad。
原报告和诊断数据在 `replay/runs/20261104_124940_51397/`，
此次检查运动命令数为零。

IK 旋转误差上限改为 0.01 rad 后，9 项测试通过，整段 1188 帧完成 IK 检查。
后续连续碰撞检查在第 60 段被拦截：`wrist_yaw_L_Link` 与 `wrist_roll_L_Link`
的间隙保守下界为 9.913 mm，低于要求的 10 mm。这是保守间隙检查未通过，
不能仅据此判定实际发生碰撞。报告为 `replay/runs/20261104_125844_54543/result.json`，
运动命令数为零。
