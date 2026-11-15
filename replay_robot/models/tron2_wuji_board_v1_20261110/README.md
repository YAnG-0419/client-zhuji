# 已确认回放模型：TRON2 + Wuji 双手

版本：`tron2_wuji_board_v1_20261110`。确认日期：2026-11-10。

主机归档目录：`/home/guest/pengyang/replay_robot/models/tron2_wuji_board_v1_20261110/`。
本记录作为该批新轨迹后续转换的安装关系依据；旧聊天中的“轴向待确认”已由本次用户确认取代。

## 已确认的关系

- 本体使用 DACH_TRON2A 厂商 URDF，双手使用当前已部署的 Wuji Hand 2 beta 模型。
- 左右手分别以各自标定板中心作为回放工具原点，使用各侧已有手眼结果计算中心偏移。
- 左手轨迹轴与板中心轴一致。
- 右手轨迹 +X = 板 +Y，轨迹 +Y = 板 -X，右手系下轨迹 +Z = 板 +Z。
- 新包导出的 `l_base_link/r_base_link` 是此次指定的轨迹手根坐标；不额外叠加模型内部 `base_link -> wrist` 偏移。
- 手与臂之间存在转接法兰，操作者估计长约 5 cm。它不作为新的平移量重复加入标定。
- 转接件示意外形连接既有参考点，左右长度约 60.4/62.6 mm；示意半径 18 mm 未经测量，未加入碰撞几何。不能用它代替转接件实测 CAD。

## 归档内容

- `robot.urdf`：整机装配模型，含本体、左右手、SDK 末端、板中心坐标与转接件示意；原厂夹爪已从此派生模型移除。
- `meshes/`：68 个模型网格，URDF 使用包内相对路径。
- `alignment_config.json`：冻结的左右轴向和原点定义，指向本版本的标定快照。
- `calibration/`：左右标定原始数值快照。
- `model_record.json`：来源、固定变换、使用范围及验证结果。
- `sources/`、`licenses/`、`tools/`：原始模型、许可证、原配置和转换程序快照。
- `view.html`：可交互的坐标/装配标注片段，`view_geometry.json` 为其简化几何数据。
- `SHA256SUMS.json`：归档文件哈希。

## 后续如何使用

**末端轨迹转换会用到。** 当前 `align_board_frames.py` 已使用这套轴向和标定中心；后续为保证复现，显式指定冻结配置：

```bash
cd /home/guest/pengyang
/home/guest/.pixi/bin/pixi run --locked python replay_robot/align_board_frames.py \
  --config replay_robot/models/tron2_wuji_board_v1_20261110/alignment_config.json \
  --episode replay_robot/traj/sharpa_to_wuji2_beta/episode594 \
  --output replay_robot/derived/<新的输出目录>
```

转换顺序为：源场景中的手根位姿 → 板中心工具位姿 → SDK 末端位姿。原包的左 +90°/右 -90° 修正已包含在输入中，不重复执行。手指关节角保持原样。

转换目前仍输出源场景 `lychee_root` 中的位姿。接下来需要建立源场景到 TRON2 基座的任务映射，再做 IK、限速和碰撞检查。这个归档不表示该任务映射已经完成。

**整机 URDF 可以作为后续模型展示和 FK/IK 接入的依据。** 它已经通过离线加载与坐标链校验，但没有自动替换现有 replay/Pinocchio/实机执行入口的模型。接入整机 IK 时需明确 56 维关节映射（双臂与头部 16 维、双手 40 维），不能直接替换旧 16 维模型参数。

原手模型的碰撞几何随 URDF 保留；这不等于完成整机碰撞模型验收。转接件、线缆及外部环境仍需另行处理。

## 已完成的验证

- 56 个可动关节，68 个网格引用全部可解析。
- 零位及两组非零关节值，共 3 组 FK 检查：SDK 末端、板中心和手根与解析变换链一致，矩阵元素最大误差小于 1e-14。
- 轨迹转换已有 5 项离线测试通过；594/615/616 共 3,137 帧完成转换并回代验证，原包未修改。
- 本次无设备控制调用，无实机运动。`execution_ready=false`。

若以后更换手、转接件、安装方向或重新标定，应建立新版本，不覆盖本版原始记录。
