# 完整双臂、双手同步回放

> 2026-11-10 最新：新规则的 episode395 已完成一次整段手臂同步真机回放。默认输入为 `data/episode395_board_placed_rx10.npz`，两臂共用任务放置，保持双手相对位姿；准备路径增加展开肩部的中间姿态。播放源37.850秒、实际47.754秒，无保护中止，双臂和双手已停止。最大EEF位置误差左33.08/右41.62mm，仅记录，未完成精度验收。详见 `diagnostics/board_aligned_full_20261110_222525/README.md`。下文约42秒等历史数据指旧轨迹。

新入口 `replay_full.sh` 使用已经测试的 MoveJ 准备和实测姿态 ServoP 接管，再回放整段 episode395。它不经过旧联合入口中的 ServoJ 准备流程。

```bash
cd /home/guest/pengyang
# 离线规划，无设备控制
bash replay_hand/replay_full.sh --speed 1 --rate 300
# 只读现场检查；不会启动缺失的手驱动
bash replay_hand/replay_full.sh --speed 1 --rate 300 --check
# 现场准备好后真实执行，控制双臂和双手
bash replay_hand/replay_full.sh --speed 1 --rate 300 --execute
```

`--speed 1` 是目标源速度上限，不保证按原始37.85秒完成。臂目标速度上限1.0 rad/s、手目标速度上限1.0 rad/s，任一侧超限时共用源时钟同步放慢；按已保存的新规则episode395路径重算，预计播放约41.35秒（此前0.30 rad/s时约47.75秒），新速度尚未进行真机回放；另有双手至少10秒准备、双臂至少20秒准备和约4秒ServoP接管。手部目标名义50 Hz发布，臂请求300 Hz发送，实际发送频率记录在结果中；均不代表电机内部控制频率。

臂关节及EEF精度偏差只记录；保留模式、反馈新鲜度、关节范围、模型碰撞、1 rad/s臂反馈速度、头部运动、调度超时和操作员停止检查。手在准备和MoveJ阶段保留0.20 rad跟踪硬保护；正式`synchronized_replay`阶段为夹取物体把该阈值改为日志告警，继续记录误差、最坏关节及驱动velocity/effort。`wuji_control`链路、实测关节范围和驱动硬件限制不变。此入口限制源时长最多60秒、名义播放时间最多120秒；单独臂实验的0.5秒限制不变。设备须在developer_mode、碰撞保护开启、柔顺关闭状态。

运行目录为`/home/guest/pengyang/replay_hand/runs/replay_full_*`。`accuracy_source_replay.jsonl`和`hand_feedback.jsonl`记录共用源时间及各自目标/反馈。正常结束停止发送ServoP，让控制器命令超时后锁存末帧位置；程序确认双臂静止且仍为`developer_mode`，不发送`request_emgy_stop`。双手仍关闭目标并禁用软硬件使能；不自动返回。可Ctrl-C，或在当前运行目录创建`stop.request`请求停止。

2026-11-14：正常完成时停止发送ServoP，关闭本地急停租约，让TRON2控制器在约0.2秒命令超时后锁存当前位置。程序继续核对双臂静止0.2秒且仍为`developer_mode`，成功时不发送`request_emgy_stop`，结果记录`arm_stop=controller_hold`，原因为`synchronized replay complete`。保持未建立、模式变化、MoveJ仍在执行或其他设备保护触发时，继续使用原急停清理。双手仍关闭程序目标并禁用软硬件使能。共享的独立机械臂精度实验入口保留原来的正常完成急停行为；只有`replay_full`显式选择控制器保持。

同日加入预期物体接触策略：仅在`replay_full`的`synchronized_replay`阶段，手指目标—反馈偏差超过0.20 rad时不再抛出`HandTrackingError`，而是在`hand_feedback.jsonl`逐帧写入`tracking_warning`，并在`result.json.hand_tracking_warnings`汇总次数、时间、最坏关节、最大误差及对应velocity/effort。双手准备、MoveJ保持阶段和独立手回放入口仍为硬停止。机器人固件碰撞保护必须开启；Wuji驱动`effort_limit=0.5`、反馈超时、发布者唯一性、目标与实测关节范围、手/机器人本体5 mm间距及所有臂保护不变。外部物体本来就不在模型碰撞范围内。

模型碰撞检查包含臂、躯干和头部，仍未包含Wuji手、适配器、线缆及环境障碍。共同源时钟不保证机械臂和手实际响应延迟相同；应结合反馈评估。完成流程不代表精度验收。

2026-11-10：56项无硬件测试通过，包括共同限速时钟、手部采样相位、只读预检不使能设备、异常时双侧停止，以及实测反馈机械范围检查。

第一次完整联合尝试运行到源时间27.044秒，右腕俯仰实测0.751500 rad超过SDK目标上限0.750500 rad而中止；当时模型碰撞余量20.18 mm，手误差0.0723 rad，双侧停止完成。这不是机械越界：该关节URDF范围为±0.785398 rad。已经修正联合会话：发送目标仍使用SDK与URDF交集，实际反馈改为检查URDF机械范围，不以SDK目标边界验收实测位置。原单独臂入口默认范围检查未更改。修正后的真机结果以新的运行目录`result.json`为准。

修正后第二次运行已通过27.04秒位置，但在32.7827秒因右手小指第二关节跟踪偏差0.2251 rad停止。目标与控制输出约0.1496 rad，反馈向反方向移动至−0.0755 rad。双侧停止确认；未放宽手部保护或自动重试。完整轨迹仍未通过真机验证，需先排查小指接触/外力及控制执行问题，不能直接把本入口视为已验收的全程回放。

随后现场授权复现中，小指在MoveJ准备阶段再次超出0.20 rad跟踪阈值，尚未开始源轨迹。用户确认没有接触；该关节反馈effort约−0.4983，接近启动配置0.5限幅，仍需排查手部参数与执行链路。没有提高effort限幅。另已修复手准备后预检起点变旧的问题：重新确认实际臂姿态静止0.3秒再认证接近路径，58项测试通过。

之后右手一度SDK连接超时；连接恢复时报告右手固件2.11.0，左手2.10.0。沿用原配置再测，首帧准备通过，但在源时间32.7727秒又因右小指第二关节偏差0.2005 rad停止，接近旧记录32.7827秒。双侧停止确认。完整回放仍未验收通过，详见`diagnostics/full_sync_fw211_20261110_203142/README.md`。

## 机械臂目标速度更新

用户将播放段机械臂目标关节速度上限改为1.0 rad/s；手指仍为1.0 rad/s，手臂同步时钟保留。规划报告和计时使用同一常量。机械臂实测速度保护仍为1 rad/s；MoveJ准备和实测ServoP接管仍按原慢速流程执行。90项离线测试通过。基于最近成功运行的plan_full.npz重算预计播放41.352501秒，本次未启动设备。

当前执行入口：replay_full.sh → replay_full.py → full_sync.py。默认数据episode395_board_placed_rx10.npz。离线IK与FK检查后，MoveJ进入准备姿态，ServoP发送SDK末端位姿；双手通过wuji_control发送q20目标。此记录中的两次完整真机成功均发生在0.30 rad/s目标速度上限下。

## 2026-11-11 operator threshold update

Canonical values: `replay/replay_thresholds.py`. IK position tolerance is 20 mm;
IK orientation tolerance is 2 degrees; joint-chord orientation tolerance is
3 degrees (position tolerance remains the same 20 mm). Checked model
collision pairs require 5 mm except same-wrist yaw/roll at 1 mm, following
the latest operator adjustment on 2026-11-11. Offline planning, adaptive MoveJ
preparation and online arm clearance share this pair-specific requirement;
hand/body clearance is 5 mm. Arm clearance scores are normalized to
the common 5 mm gate, not raw distances for the internal wrist pairs. Joint limits,
FK consistency checks, speed limits and device protocols are unchanged.
This update does not certify a previously blocked trajectory for execution.
