# 分开验证机械臂与手

两套独立入口部署在 `/home/guest/pengyang/replay_hand`，本地副本在 `/Users/lpy/Documents/ws/Xflow/replay_hand`。原有同步回放和 Pinocchio 入口保持原样。

| 入口 | 控制范围 | 不使用的设备 |
| --- | --- | --- |
| `replay_arm_only.sh` | Tron2 双臂；MoveJ 首帧准备、ServoP 接管与回放；不使用 ServoJ | 不创建 Hands、不初始化 ROS、不启停手驱动或发送手指目标 |
| `replay_hands_only.sh` | 左右 Wuji Hand 2，经现有 wuji_control | 不连接 Tron2、不加载臂模型、不做 IK、不发送或停止机械臂命令 |

所有命令在 `/home/guest/pengyang` 下执行。默认只离线规划；`--check` 仅连接对应设备读取状态，不启动驱动或使能；只有 `--execute` 会实际控制设备。两个测试共用既有执行锁，应顺序运行。另一部分不会被本程序主动锁住、停止或监视，测试前需自行保持其静止并退出其他控制程序。

## 机械臂

2026-11-10 已对齐 test_ik 的 MoveJ→ServoP 接管顺序。MoveJ 仍采用至少20秒的慢速准备，位置误差须小于0.02 rad、反馈速度小于0.01 rad/s并稳定0.3秒，最多额外等待4秒；没有继续采用先前0.1 rad的宽松验收。随后从实测FK接管ServoP并平滑对齐原首帧，不整体搬移源轨迹。

保持模式现在也是 ServoP，仅适用于已在该工作空间内的姿态。自然下垂姿态可能不在范围内，会在发送命令前拒绝；此时应先使用下方经过碰撞检查的首帧准备流程。保持模式预检：

```bash
bash replay_hand/replay_arm_only.sh --hold-seconds 2 --check
```

预检通过后，由操作者现场启动 ServoP 保持测试：

```bash
bash replay_hand/replay_arm_only.sh --hold-seconds 2 --rate 300 --execute
```

该测试以刚测得的姿态FK作为目标，不加载 episode、不做 IK；会真实发送 ServoP 命令，监视头部位移。正常持续约1秒接管保持、指定时长和至少0.3秒稳定判定，未稳定最多再等待3秒。只测臂入口不再开放ServoJ。

原 test_ik 准备姿态对照可加 `--ready-arm right`、`left` 或 `both`（必须同时使用 `--hold-seconds`）。这会先用至少20秒的MoveJ将指定臂移至 `[0,0,0,-1.6,0,0,0]` rad，确认静止后再保持实测末端目标；非指定臂及头部目标沿用当前位置。ServoP发送左右两臂目标，两侧均须满足当前采用的工作空间检查。若非指定臂处于范围外，原方案会被拒绝，不能自动扩大成双臂移动。`--ready-arm both` 会实际移动双臂，须由操作者明确选择。该对照不画圈、不回放episode、不自动返回原姿态。

用户授权的精度观察实验可添加 `--record-accuracy-only`，支持0.2–2秒保持，或明确指定最多0.5秒的源轨迹窗口（名义播放时长最多15秒）。例如保持实验：

```bash
bash replay_hand/replay_arm_only.sh --ready-arm both --hold-seconds 2 --rate 300 --record-accuracy-only --execute
```

该模式中，MoveJ完成名义时长且实测静止后，记录名义准备误差而不据此拒绝；再对实测姿态重新检查。保持实验固定发送其实测FK目标；短轨迹实验先保持1秒、沿经过认证的关节段平滑过渡到原首帧，再按原时间轴播放，最后保持末帧0.3秒。不搬移原轨迹。

```bash
bash replay_hand/replay_arm_only.sh --end-seconds 0.5 --speed 0.05 --rate 300 --record-accuracy-only --execute
```

关节路径偏差、EEF位置与旋转误差只记录，不设精度退出阈值；保留设备状态、有限值、机械限位、模型碰撞、1rad/s臂速度、头部保护、反馈新鲜度、调度超时和操作员停止。实验严格按时限结束，正常结束也发送停止命令，可能需要重新在UI进入开发者模式。`accuracy_hold.jsonl`或`accuracy_takeover.jsonl`、`accuracy_source_replay.jsonl`及result相应字段提供偏差与发送计时；completed只表示实验采样完成，不表示精度合格。原默认回放的验收条件保持不变。

保持测试稳定后，单独验证到首帧（会移动双臂）：

```bash
bash replay_hand/replay_arm_only.sh --end-seconds 0.5 --speed 0.05 --rate 300 --prepare-only --execute
```

仅臂回放前 0.5 秒：

```bash
bash replay_hand/replay_arm_only.sh --end-seconds 0.5 --speed 0.05 --rate 300 --execute
```

普通回放仍计算现有完整 IK 种子路径，启动耗时不会因隔离而消失；只有保持模式跳过该计算。保留关节限位、碰撞范围、反馈新鲜度、状态、速度 1 rad/s、关节偏差及 ServoP EEF 保护，并检查test_ik所用的ServoP工作空间。继承原有相机/安装标定假设与手、适配器、线缆、桌面不在碰撞模型中的限制。改走MoveJ→ServoP不等于已查明或修复ServoJ控制器内部异常。

## 手

离线检查（不做 IK）：

```bash
bash replay_hand/replay_hands_only.sh --end-seconds 0.5 --speed 0.05
```

仅手到首帧，便于观察不到位的关节：

```bash
bash replay_hand/replay_hands_only.sh --end-seconds 0.5 --speed 0.05 --prepare-only --execute
```

仅手回放前 0.5 秒：

```bash
bash replay_hand/replay_hands_only.sh --end-seconds 0.5 --speed 0.05 --execute
```

`--check` 需要已有手驱动及 wuji_control，且不创建驱动；`--execute` 在没有驱动时可启动原有双手 launch，结束后停手并清理本次启动的驱动。手部首帧至少 10 秒，最多额外等待 8 秒；验收与运行跟踪阈值保留 0.20 rad，不代表消除了约 0.18 rad 残余误差。位置目标速度最多 1 rad/s；源时间也会按该限制减速。

默认手部监督/目标更新频率 50 Hz，可用 `--rate 20..50` 调整。继承的 ROS program 发布器名义 50 Hz，与机械臂 300 Hz 的 WebSocket 发送不同；这两个数字都不是电机内部控制频率。手部时间独立，不承诺与单独运行的臂部时间一致。

## 日志及停止

- 日志目录分别为 `runs/replay_arm_only_*` 和 `runs/replay_hands_only_*`；`result.json` 写明 `component`，两类命令计数分开。
- 臂：`arm_feedback.jsonl` 在保护检查前保存 q/dq；`robot_wire.jsonl` 保存有界原始收发尾部；各准备阶段另有日志。
- 手：`hand_feedback.jsonl` 保存每个关节的期望、程序目标、实测位置、最大偏差关节，以及被动订阅到的 `/wuji/{side}/command`。反馈包含速度、effort 和采样时间；驱动没有提供时记为 null，effort 单位须以驱动定义为准。订阅到 command 不能单独证明电机执行了该命令。
- Ctrl-C 或在当前运行目录创建 `stop.request` 会停止本次被控组件。不会为了停止手测试向臂发送 stop，反之亦然。

离线单元测试覆盖无另一设备依赖、异常清理、故障帧记录、手部范围与速度限制。离线通过不能代替真机接管验证；本次编写和部署不包含真机运动。
