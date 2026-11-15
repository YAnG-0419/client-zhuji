# Wuji2 相机轨迹：不额外调整摆位的回放检查

结论：通过当前 replay_full 完整离线检查。150 个原始帧、14.91072 秒；仅做对应标定板场景映射、B 点到 SDK 工具转换和 50 Hz 格式适配，没有额外整体摆位搜索或平移旋转修正。未执行实机。

## 输入与坐标

原轨迹已是 Wuji Hand 2 左右 base_link 位姿和每手 20 个关节目标，源流为 final_handpose.npz:joints_camera_m。原点采用用户确认的 B 点，左右安装关系继续使用冻结配置。没有再做人手重定向、局部 Z 修正，也未叠加 28.5 mm 或约 50 mm 偏移。

新 observation 的相机是 orbbec_gemini_335。与既有 zhuji 观测的板指纹、尺寸、轴定义、单位一致；新映射为 target_camera_T_source_camera = target_camera_T_board @ inverse(source_camera_T_board)，板位姿闭合最大差 1.12e-16。不是沿用原 franka-sharpa 的 task_mapping，也没有把轨迹当作 lychee_root。

从源相机到目标相机的平移为 [-12.855,29.203,-55.076] mm，旋转角约 1.738°。这是场景坐标转换，不是额外选择可达摆位。目标相机到基座继续使用用户确认约 41.5° 对应的现有估计。

按照用户此次提供关系，将该观测视为轨迹对应相机且安装未变；去畸变 optical 与原 color optical 轴一致、不含额外整流旋转。这些前提和原目标相机外参的物理准确性，仍不是离线 IK 通过所能证明的。

## 输入适配

原始采样约 9.993 Hz，间隔 99.84–100.24 ms，其中 96 个间隔略大于现有加载器要求的 100 ms。原文件未修改；converted_source_samples.npz 保存 150 帧完成坐标转换后的原始采样，因间隔规则不能原样交给现有入口。

replay_input_50hz.npz 为 747 点适配输入，保持 14.91072 秒总时长，按现有规则对 SDK 位置/手指线性插值、姿态 Slerp；保留 original_time_s、original_timestamp_ns、original_hands 供追溯。原 timestamp_ns 为源设备时钟，未当作 Unix 时间解释。指令使用 eef 的基座坐标 xyz+xyzw；手指使用 hands 的左右固件顺序。

## 离线结果

- 连续逆解 747/747 通过，关节范围采用 MoveJ 与 URDF 交集。
- 最大逆解残差约 0.0200 mm、0.00492°；最大相邻关节变化 0.03438 rad。
- 28 个规划路点及连续扫掠检查通过；普通受检对要求 5 mm，同腕 yaw–roll 1 mm。
- 插值偏差最大 19.834 mm、2.727°，满足当前 20 mm / 3° 插值门限。这与上面的逆解残差是不同指标，不代表实际轨迹精度达到 0.02 mm。
- 手–手、手–臂/身体/头部扫掠检查 1064 次通过，保守最小间隙 66.579 mm。
- 源手指速度峰值左 0.793、右 0.666 rad/s，所有目标在当前安装限位内。
- 目标 1 倍速、臂手 1 rad/s 同步限速后，预计正式播放 16.492 秒，准备时间另计。
- 模型未覆盖环境、线缆、实测转接件几何及同一只手内部接触；实机当前状态及进入首帧的准备路径尚未检查。

首次直接启动 Python 缺少 ROS 包环境，已保存失败日志；随后使用项目正式 replay_full.sh 设置依赖环境，完成全部离线检查。没有修改门限或回放代码。

## 复现

在主机 /home/guest/pengyang 下：

```bash
bash replay_hand/replay_full.sh --episode replay_robot/camera_trajectory_1987e125_20261111/replay_input_50hz.npz --speed 1 --rate 300
```

命令不含 --check 或 --execute，属于离线检查。

validation_summary.json 汇总数值，offline_replay/ 保存正式入口的完整结果和计划。旧加载器报告内部分 arm_source/hand_source/first_frame_arm_source 是固定历史文案；本次实际来源以 inputs 原文件、conversion_report.json 为准：人手相机拟合的 Wuji 目标，机器人关节是离线 IK 生成，没有读取源 Franka 指令或实机 joint_states。
