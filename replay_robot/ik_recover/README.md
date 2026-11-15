# 回放 IK 断口恢复

恢复模式使用已映射到 `tron2_base_link` 的双臂双手轨迹。输入为现有的
`replay_hand_cross_v1` NPZ，且 `placement_policy` 为 `preserve_common_frame_v1`。
完整回放还需要现有的 `task_placement` 摆位信息。

在主机上做离线检查：

```bash
cd /home/guest/pengyang
bash replay_hand/replay_full.sh --episode /absolute/path/replay_input.npz --ik-recover
```

指定过渡段的偏移上限和前视范围：

```bash
bash replay_hand/replay_full.sh --episode /absolute/path/replay_input.npz \
  --ik-recover --recover-position-mm 35 --recover-rotation-deg 4 --recover-max-gap-s 1.5
```

默认分别为 **20 mm、3°、1.5 秒原始轨迹时间**。带参数的示例允许过渡段偏离原轨迹，
适用于该动作允许这些偏移的情况。正常段按现有回放误差设置检查。

求解从首帧合格姿态开始。遇到超差、分支跳变或帧间超差时，向前找最近能接上的
合格双臂姿态，先试关节插值，再尝试修正中间姿态。过渡的误差检查包含两臂、原始
轨迹的每个时间节点和加密的帧间采样。候选通过现有的关节范围、ServoP 工作区、
双臂扫掠和手体扫掠检查后，交给原回放入口统一重定时。手指与双臂使用同一条时间轴。
缺少合格首帧或终点，或所有候选都超过误差或碰撞上限时，离线规划报错。

默认命令为离线模式。现有入口也接受同样参数和 `--execute`，启动时先完成整条
轨迹的恢复与验证，再按现有流程实时回放。恢复目前在开始运动前计算，
因此仅支持播放时源轨迹已经给定的场景。模型扫掠覆盖现有臂、手和本体模型，
实物场景中的接触、障碍物和转接件仍取决于现场模型与标定。

运行目录的 `result.json` 中，`ik_recovery` 保存过渡时间、偏移峰值、验证结果及
拒绝原因。规划成功后，`ik_recovery_path.npz` 包含对应时刻的原目标、规划末端、
关节角和手指目标，可对照检查改动范围。
