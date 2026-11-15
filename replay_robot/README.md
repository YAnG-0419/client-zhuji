# 新轨迹导入及 TRON2 / Wuji 回放接入检查

当前已确认模型：[CONFIRMED_MODEL.md](CONFIRMED_MODEL.md)。冻结版本：`models/tron2_wuji_board_v1_20261110/`。后续转换以该记录为准。


检查日期：2026-11-10。此次只导入文件并做离线检查，未连接设备控制接口，运动命令为零。尚未生成机械臂执行轨迹。

## 已导入

源文件：`/Users/lpy/Downloads/sharpa_to_wuji2_beta_594_615_616.zip`。
远端原包：`/home/guest/pengyang/replay_robot/traj/sharpa_to_wuji2_beta_594_615_616.zip`。
解压目录：`/home/guest/pengyang/replay_robot/traj/sharpa_to_wuji2_beta/`。
压缩包 SHA256：`a8170fbb107483c9c73cbb3d0d096b6a07c6b62179797dbe0e6c720f19c2137e`。
源包和远端哈希一致，包内 MANIFEST 的 181 个文件全部通过 SHA256 校验；原说明、许可证、模型、视频和轨迹完整保留。

| Episode | 帧数 | 首末帧跨度 | 采样频率 | 手指原速峰值 |
| --- | ---: | ---: | ---: | ---: |
| 594 | 965 | 32.13 s | 30 Hz | 4.066 rad/s |
| 615 | 909 | 30.27 s | 30 Hz | 3.246 rad/s |
| 616 | 1263 | 42.07 s | 30 Hz | 2.857 rad/s |

检查报告：`import_validation_20261110.json`；导入记录：`traj/import_20261110.json`。
时间戳严格递增，time_s 与 timestamp_ns 对齐，轨迹数值有限。40 维与 52 维数组和 NPZ 分侧数据逐项一致。
左右手 URDF、MJCF 与当前 `teleop/wuji_teleop/models/hand2_beta` 逐字节相同。
借助已部署 MJCF actuator 的名称到关节映射确认，每侧 20 维顺序与 firmware finger1..5/joint1..4 一致；全部手指目标在已安装 wuji_control 关节限位内。
局部根变换与场景变换组合一致；左 +90°、右 −90° 的局部 Z 修正已经存在，不能重复施加。
这些检查不代表实机跟踪、抓取或碰撞验收。

## 接下来需要做什么

### 1. 接入数据格式；手指不必重新重定向

现有 `replay_hand/episode_cross.py` 不能直接读取新包：缺少 `eef`、`hands`、`t`、固件关节名和 `metadata_json`。
应新增转换器，保留原包不变，输出派生文件：

- `t = time_s`；保留原 `timestamp_ns`。
- `hands[N,2,20]` 为左、右手；按已核实的 MJCF 映射写入固件名，不能混用 40 维的左→右顺序与 52 维的右→左顺序。
- 仅手验证可以复用 `replay_hand/hand_only.py` / `replay_hands_only.sh`，其加载器不依赖机械臂 EEF，但仍要求对应格式的元数据与关节名。当前尚未生成此派生输入。
- 联动回放需要下述正确的 `eef[N,2,7]`，坐标为 TRON2 基座、米、XYZW 四元数；不得伪造 EEF 来让完整回放加载器通过。
- 旧 `build_episode395.py` 是另一种输入格式和坐标链，不能直接用于本包。

### 2. 明确工位坐标与两侧真实安装关系

52 维中的 xyz/rotvec 是 Wuji base 相对于各侧 Sharpa 掌根的局部变换，不是相机坐标，也不是 SDK 末端。
`wuji_motion_scene.npz` 已包含组合后的 Wuji base 位姿，但位于源装配模型 `lychee_root`；包说明明确该装配布局尚未与录制工位配置哈希对应。

若要保持任务在工作台中的位置，需建立并验证源场景到 TRON2 基座的变换，以及左右 `T_sdk_eef_wuji_base`。列向量约定下：

`T_tron_base_sdk_eef(t) = T_tron_base_scene @ T_scene_wuji_base(t) @ inverse(T_sdk_eef_wuji_base)`

若通过相机/标定板建立跨工位对应，还需要源模型场景到源相机的关系、真实双相机/板观测和目标相机到基座的关系。现有 `calibration/camera_board_alignment` 只有工具及 synthetic 示例，检查时没有真实 `results/`；不能使用 synthetic 映射作为实测结果。历史目标相机到基座外参也仍标记未独立验证。

若只验证相对动作，可以另行明确首帧锚定策略；这会改变绝对任务位置。现有 `anchor_cross.py` 按两臂分别重锚，不能据此声称保留双手相对摆放或物体接触关系。其历史 mount_center 偏移和 R_HAND_EEF 也不能自动当成本包 Wuji base 的固定安装变换，右侧旧朝向仍是未独立验证假设。

本包含每帧变化的手根拟合和已确认朝向修正；需要把完整根变换纳入末端转换，不能沿用 episode395 忽略 free-root adapter 的做法。源 FR3 的 7 关节只能用于源 FK，不能作为 TRON2 关节目标。

### 3. 全段离线规划、限速和碰撞检查

得到目标 EEF 后，再生成连续 TRON2 IK，检查机械范围、解跳变、EEF 误差、连续扫掠碰撞和 ServoP 工作空间。接入时可复用现有规划/执行组件，但应逐项核对其 episode395 专属的锚点与安装参数。

仅手入口现有目标速度限制为 1 rad/s。按原采样线性插值，若全段采用统一源速度倍率，单看双手速度的上限分别为 594: 0.246、615: 0.308、616: 0.350；这不是推荐真机速度，现有仅手 CLI 还限制倍率至多 0.25。
也可使用逐段重定时；联动时必须与机械臂共享时间进度，不能让双手单独减速。机械臂速度/加速度、准备阶段和实测跟踪还可能要求进一步降速。

现有机械臂碰撞模型不包含手本体、转接件、线缆、桌面或操作物体；包内转换也未验证手指自碰撞或接触效果。包报告显示朝向修正后相对于未旋转 Sharpa 目标的平均指尖距离约 85–100 mm，因此修正前 1–2 mm 拟合误差不代表当前输出的抓取位置精度。

### 4. 离线通过后再验证实际回放

依次做已有驱动状态下的只读预检、首帧准备、短片段低速回放，再扩大范围。Wuji 沿用 wuji_control 的 program 通道。
本次没有执行 --execute、启动驱动、调整模式或移动机器人。

建议下一步先完成仅手格式转换和全段离线检查；整机回放优先补齐坐标映射与左右实际安装变换。

## 2026-11-10 用户补充后的坐标核对

用户确认左右手手腕末端中心与标定板中心重合。安装中心位置已有依据，不再作为缺失标定项。按各侧 `T_eef_board @ [0.14,0.09,0,1]` 计算，SDK 末端坐标下中心偏移为左 [-9.963, 0.982, 72.118] mm、右 [-2.984, 0.831, 68.944] mm，与旧 replay 保存的中心偏移一致。注意 T_eef_board 的平移是板原点的位置，不能直接作为板中心。

仍需区分：板坐标轴与 Wuji 模型轴的对应；轨迹 base_link 原点与物理手腕中心的对应；源 lychee_root 场景与 TRON2 基座的对应。新包 root 姿态使用 l/r_base_link，URDF 内 l/r_wrist 相对 base_link 还有约 28.5 mm 的 Z 偏移（以及 X/Y 小偏移），左侧另有约 -90° 的固定旋转。不能因名字含 wrist 就自动把两个原点当成同一点。

左右标定的板轴旋转不同，不能把左侧矩阵直接复制到右侧；旧 replay 的 R_HAND_EEF 属于旧 hand_coordinates 约定，新包尚未套用标定完成 SDK EEF 转换。上述检查不改变任何实机参数。具体标定矩阵和来源见 coordinate_alignment_review_20261110.json。

## 2026-11-10 板中心坐标对齐（本次确认后采用）

用户确认：左右轨迹手根中心按各自标定板中心对齐；左手轴与板中心轴一致，右手轨迹 +X 对应板 +Y，轨迹 +Y 对应板 -X，右手系下 +Z 对应板 +Z。此确认解决前文待确认的安装轴向；它是操作者确认的关系，不是新增实测标定。

配置：`board_frame_alignment.json`。这里的轨迹手根指新包已修正过朝向的 `l_base_link/r_base_link`，不额外加 URDF 内部 `base_link -> wrist` 偏移。各侧使用自己的 `T_eef_board` 和板中心 [0.14,0.09,0] m。

用 C 表示板中心坐标、H 表示轨迹手根、E 表示 SDK 末端、S 表示源场景：

- 左侧 `R_C_H = I`；右侧 `R_C_H = [[0,-1,0],[1,0,0],[0,0,1]]`。
- `T_S_C = T_S_H @ inverse(T_C_H)`。右侧姿态因此右乘 Rz(-90°)，这是更换工具轴定义，不是重新执行包内朝向修正。
- `T_E_C = T_E_board @ Translation([0.14,0.09,0])`。
- `T_S_E = T_S_C @ inverse(T_E_C)`。

运行入口 `align_board_frames.py` 仅依赖文件、NumPy 和 SciPy，不包含设备接口。原包不修改，派生输出保存于：

`/home/guest/pengyang/replay_robot/derived/board_aligned_20261110/episode{594,615,616}/`

每段包含 `board_center_source_scene.npz`、`alignment_report.json`。输出保存左右 `scene_T_board_center`、`scene_T_sdk_eef`、固定工具变换、原时间戳、原手指关节与名称。坐标仍为源场景 `lychee_root`，没有冒充 TRON2 基座的通用 `eef` 字段；`execution_ready=false`。

5 项离线测试通过，覆盖右手轴符号/逆变换、板角点到中心的旋转补偿、任意姿态回代及反射拒绝。三段共 3,137 帧的中心位置变化为 0，手指角保持一致，完整变换回代的矩阵元素最大误差 <=3.34e-16。原包 MANIFEST 的 181 个文件重新校验不变。未发送运动命令。

仍需建立源场景到 TRON2 基座的任务映射，再做整段 IK、限速和碰撞检查。此次工具坐标对齐不代替跨工位场景对齐。

## 转接法兰补充说明

用户说明：手和臂之间图中空出的部分是转接法兰，长度大概 5 cm。已记录于 `adapter_observation_20261110.json`。它表示连接结构的近似尺寸，不额外叠加到标定得到的 SDK 末端到手腕中心偏移。

按当前模型原厂 grasper_Joint 连接原点到标定板中心计算，左右参考点间距分别为 60.4 mm、62.6 mm；这些参考点尚未证明与实物测量的两个表面一致，因此不使用约 50 mm 的口述值覆盖标定矩阵。

可视图在上述既有端点间补了转接件示意外形，示意半径 18 mm 未经测量。该外形不加入执行端碰撞模型，也不表示转接件已完成尺寸验收。已确认的手中心、左右轴向和派生轨迹均不变。

## 2026-11-10 跨场景任务映射与回放检查更新

真实 task_mapping 与用户确认的 41.5° 头部姿态已接入。三段已生成 TRON2 基座下的 replay 输入并通过加载；现有及独立 IK 均未获得合格完整双臂路径。当前最新结果见 [任务映射与离线检查](task_transfer_20261110/README.md)，三段仍为 execution_ready=false。原包与冻结模型不变。
