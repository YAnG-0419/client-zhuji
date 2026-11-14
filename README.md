# 在线 EEF 客户端：2026-11-14 实现状态

本目录现已加入**纯离线的部署核心**，仍未连接模型 RPC 或机器人硬件：

- `online_core.py`：最近 Orbbec→zhuji board-correspondence 坐标转换、Wuji 手基座到
  TRON2 SDK EEF 的工具变换、带时间戳 chunk、最短弧 SLERP、100 ms 五次平滑交接，
  以及“一个异步规划任务、规划线程绝不发送”的状态机。
- `deployment_config.json`：记录手与臂之间物理转接法兰约 **0.05 m**，但明确
  `apply_as_transform=false`。5 cm 是物理端点/间隙的估计，不是 SDK EEF 与 Wuji
  `base_link` 两个坐标原点之间的距离。在线转换与 replay 一样，原样使用完整标定链；
  不额外增加、也不缩放 5 cm。标定坐标原点距离约为左 7.28 cm、右 6.90 cm，二者
  不能与转接件可见长度混为一谈。
- `online_client.py`：严格 dry-run 编译器；没有 `--execute`，不导入 robot/ROS/Wuji
  传输。它把模型或录制的 camera-frame chunk 编译为 50 Hz 的 base-frame EEF＋q20，
  输出 `accepted_chunk.npz` 和审计报告，始终记录 `execution_ready=false`、
  `hardware_commands_sent=0`。
- `test_online_core.py`：验证 5 cm 注释不覆盖标定、刚体闭环、双臂相对位姿、交接端点、
  四元数单位长度和异步规划边界。

参考了 AgiBot `g2_rtc.py` 的职责划分，但没有复制它的 GDK 发送器。来源和哈希见
`client/SOURCES.json`。当前 replay/replay_hand 文件未改动；11 月 14 日已有的
ServoP controller-hold 行为应由未来硬件发送适配器复用，不能在本离线核心中模拟为验收。

离线编译示例：

```bash
cd /home/guest/pengyang
/home/guest/.pixi/bin/pixi run --locked python client/online_client.py \
  --chunk replay_robot/camera_trajectory_adc619f1_20261114/inputs/wuji2_camera_replay_trajectory.npz

/home/guest/.pixi/bin/pixi run --locked python -m pytest client/test_online_core.py -q
```

尚缺少的两项是具体模型 RPC 协议和硬件发送适配器。定义这两个接口前，本目录不会把
离线 chunk 标记成可执行，也不会把 `replay_full.sh` 循环调用伪装成在线部署。

---

# replay_full 到在线 EEF 客户端：2026-11-11 复核

工作位置：`/home/guest/pengyang/client`。本目录目前只包含审查说明和离线性能测量，不是已经可控制机器人的在线客户端。现有 `replay_hand`、`replay` 代码未改动；本轮没有发送任何机器人指令。

**结论：可以复用现有 ServoP 通信和手部控制作为模型部署基础，但不能原样反复调用 replay_full.sh 执行模型 chunk。直接 EEF → ServoP 在接口上不要求本地逐帧 IK；当前代码中的 IK/FK 则参与了真正的目标生成、速度规划和检查，不能按“纯验证”整段删除后仍认为行为等价。**

## 最新状态取代旧结论

- 入口为 `replay_full.sh → replay_full.py → full_sync.run → accuracy_replay.record_replay/run_window`。
- 全程回放已经有成功记录：`replay_full_20261110_222525_181732` 和 `replay_full_20261110_223807_187642`。源 37.850 s，播放约 47.75 s，两次均为当时的 0.30 rad/s 臂目标速度设置。
- 该入口 `record_accuracy_only=True`：臂部 EEF/关节偏差记录但不据此中止。两次完整运行左/右最大 EEF 误差分别约 33.08/41.62 mm、33.82/41.73 mm。因此“未完整回放”已过时，“任务跟踪精度已验收”仍不能由完成状态推导。
- 当前臂目标速度上限 1.0 rad/s，手目标上限 1.0 rad/s；最新实测臂速度保护为 1.2 rad/s。不能用旧的 0.30 rad/s 完成记录证明新上限已经全程验证。
- 默认 `episode395_board_placed_rx10.npz` 采用共同任务放置并保留双臂相对关系；不能继续描述成默认对每只手单独重设 anchor。
- 新路径有额外的手/机身与双手之间的网格碰撞检查；环境、线缆、适配器几何和手内接触仍有边界。

## 真实执行链路

```text
读取完整 episode，重采样到 50 Hz
  → cross_loader / solve_seed_path：生成全段连续 IK seed
  → base.make_plan：再次逐帧 IK，计算旧 MoveJ 路点/弦误差/检查
  → 全段密集碰撞检查，按关节和手部速度上限生成共同时间轴 T
  → 手准备、MoveJ 准备、实测姿态 ServoP 接管
  → 按 T 插值 q_ref
  → FK(q_ref) 生成左右 EEF
  → request_servop(left_pos, right_pos)
```

播放阶段不运行逐帧 IK。IK 在开始发送前完成；FK 则在发送循环中实际生成目标。

当前 full 入口默认调用 `replay.solve_eef` 和现有 MuJoCo 模型求解，失败时使用 SciPy bounded least squares。旁边的 `diagnose_full_pinocchio.py` 是独立离线诊断入口，不会自动把正式 full 入口改成 Pinocchio。

本地关节路径至少有四项作用：首帧 MoveJ 目标；共同时间轴中的关节速度约束；规划几何检查；每次 ServoP 真正发送的末端目标。`base.make_plan` 返回的分段 MoveJ 路点和分段时长在 full 播放中被丢弃，但仍花时间计算；这是可以优先重构的重复工作。

关键代码位置：`full_sync.py:128,135,139,246`；`accuracy_replay.py:39,40,43,66,119`；`planner_cross.py` 的 `solve_seed_path`；`replay/replay.py` 的 `make_plan`。

## 频率与延迟

| 已有运行 | 请求发送频率 | 源播放实际均频 | 全源播放最大间隔 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 11/10 22:25 | 300 Hz | 227.47 Hz | 22.93 ms | 完整播放 |
| 11/10 22:38 | 300 Hz | 200.28 Hz | 33.14 ms | 完整播放 |
| 11/11 12:07 | 300 Hz | 207.21 Hz | 23.15 ms | 旧 1.0 rad/s 实测速度保护触发；此后保护更新为 1.2 |

两次完整运行保留的通信日志尾部（各约 13.9 s）中，ServoP 发送间隔 p99 约 15 ms；不是全程 p99。对应 WebSocket send 调用自身的尾部 p99 小于 0.8 ms，也不是控制器接收/执行延迟。现有平均频率不能当作稳定 300 Hz 的证据。

源轨迹 50 Hz、手部名义发布 50 Hz、臂目标请求 300 Hz、机械臂运动速度 1 rad/s 是四个不同概念。模型推理频率也独立，模型可以输出包含多个未来动作的 chunk，发送线程按固定周期消费；模型不必每 3.33 ms 完成一次推理。

今天另外两次运行的文件时间显示：12:28 的运行约 586.9 s 后写出 cross_seed，约 644.9 s 后中断；14:03 的运行约 767.8 s 后写出 cross_seed，约 1066.0 s 后中断。二者运动指令数与手部目标更新数均为 0。这里的耗时包含读取、模型初始化和第一遍求解，不是精确的 IK 函数剖析；但足以说明整段准备不能放进在线 chunk 更新的关键路径。

## 本轮离线计时

`offline_benchmark.py` 读取 `replay_full_20261111_120729_44426` 保存的 episode102 计划，禁止 socket 连接，不构造 Robot/Hands，不调用执行入口；记录在 `benchmark_20261111.json`。同机有另一轮离线规划负载，这些是量级参考，不是实时性保证。

| 纯计算部分 | 样本 | 平均 | p99 |
| --- | ---: | ---: | ---: |
| 关节插值 + FK + 工作空间/四元数处理 + JSON | 200 | 0.192 ms | 0.298 ms |
| EEF 位置插值 + 旋转 Slerp + 同样的消息处理 | 200 | 0.109 ms | 0.160 ms |
| 臂模型更新 + 几何距离检查 | 50 | 2.828 ms | 3.163 ms |
| 手模型更新 + 几何距离检查 | 50 | 1.049 ms | 1.482 ms |

三个抽测帧使用前一帧已保存关节作为 seed，本地 IK 分别耗时约 0.80、1061.71、871.02 ms；三个均满足当前残差验收。样本仅三帧，不能据此估计全路径 IK p99。内部迭代仍追求 20 µm / 0.0002 rad 的收敛精度，而外部验收允许 20 mm / 2°；近限位时可能长时间迭代、回退后才被较宽外部门槛接纳。后续应检查限时求解和收敛/验收的一致性。

微测量不包含 ROS、日志、WebSocket、真实反馈与完整调度。它说明：直接 EEF 生成目标每次平均只省约 0.083 ms，不能据此宣称发送会从 200 Hz 飞跃到更高频率。两个几何检查平均合计约 3.9 ms，目前在约每 20 ms 的监督 tick 内与发送共享线程，本身就可跨过一个 300 Hz 的 3.33 ms 周期；另有 ROS 检查及同步日志。完整 tick 的各部分仍需进一步剖析，不能把全部发送间隔归因于某一项。

## 直接 EEF 部署该改什么

可以使用以下边界：

```text
模型输出带坐标/TCP/时间定义的 EEF chunk
       ↓
短段检查与时间衔接 → 有界目标缓冲
       ↓
独立固定周期发送：取当前有效 EEF，插值、转换四元数、发 ServoP
       ↓
机器人反馈 → 独立监督与下一轮模型观测
```

1. 去掉“每轮先求完整 episode 的 IK”这一前提，保留一次性任务初始化。不要每次模型返回都重新 MoveJ。
2. 若直接发送模型 EEF，可不做完整本地 EEF→IK→FK 往返；但需要重建当前依赖 q 路径的速度约束、首帧准备和可达/碰撞检查。厂家内部如何解出关节由机器人控制器处理；本地没有 q 命令时不能保证其冗余分支与本地假想解一致。
3. 验证与日志放到有明确时限/数据新鲜度要求的工作线程或进程，执行线程只消费已接受的短段。不能把“后台处理”变成未通过检查就提前执行。
4. 如果保留本地 IK，采用短 horizon、前一解 warm start、超时/不收敛处理，避免没有时间预算的 150 次迭代加最多 500 次优化评估进入高频循环。优化内部容差要以任务要求和误差验证为依据。
5. 需要处理观测时间与推理返回时间、旧段已提交前缀、过期目标、chunk 衔接和缓冲耗尽。不能将“平均能发 200 多 Hz”直接等同于闭环部署完成。
6. 目前控制台上的 `completed=true` 表示流程结束。部署评价还要单独看任务所需的 EEF 误差、动作时效、发送抖动和实际成功率。

因此，跳过全段 IK 会显著减少计算等待；跳过发送时的 FK 只带来较小的单帧收益。当前最有价值的调整是拆开完整回放规划器与在线发送器，并消除重复规划，而不是笼统关掉全部验证。
