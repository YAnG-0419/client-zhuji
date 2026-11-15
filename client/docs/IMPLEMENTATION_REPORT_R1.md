# XFlow kp21 → TRON2/Wuji2 阶段一实现报告（R1）

日期：2026-11-14　分支：`xflow-client`　范围：在线数据链、离线 dry-run、只读 ROS/服务核验。

本阶段没有启动机器人或 Wuji 驱动，没有发布控制话题，没有调用 ServoP/ServoJ/MoveJ、
`request_emgy_stop` 或任何 admin API。远端模型共调用 `infer` 4 次（两种图像模式各两次，
后两次生成最终带录制未来对照的报告）；所有产物均为 `execution_ready=false`、
`hardware_commands_sent=0`。

## 1. 代码与用途

| 文件 | 用途 |
|---|---|
| `vendor/keypoints21_to_wujihand2/` | 原样保留 Wuji2 URDF/MJCF/mesh、转换源码和许可证 |
| `hand_retarget_wuji2.py` | 流式 free-root 与 q-conditioned Kabsch fixed-root；拒绝帧事务化 |
| `proprio.py` | 实测 SDK EEF + q20 的 kp21 FK；Wuji hand-root → SDK EEF 逆工具链 |
| `online_core.py` | `live_zhuji_camera` / `board_transfer` 分离、TimedChunk、50 Hz 重采样和交接 |
| `image_preprocess.py` | NumPy virtual_cam0、stretch_undistort 和 camera-info 启动门 |
| `xflow_rpc.py` | 严格 info/reset/infer；deployment pin；无 admin/token/load API |
| `timing_policy.py` | 默认 32 点逐帧 30 Hz；native 时间只作诊断 |
| `observation_buffer.py` | 1.5 s bounded buffer、不可变快照、header/monotonic 双时钟、fail-closed |
| `ros_readonly_observer.py` | 只创建 subscription；实时图像样例和完整快照诊断 |
| `bag_observation_source.py` | sqlite `mode=ro` 抽取 JPEG、状态和 32 步未来实测值；不播放 bag |
| `relay_mac.py` | 两条仅绑定 127.0.0.1 的 SSH 中继及生命周期清理 |
| `online_dryrun.py` | fake/remote → 重定向 → EEF → 50 Hz → EEF_BOX/限位/Pinocchio IK/速度 |
| `docs/execution_layer_design.md` | 第二阶段唯一发送器、startup、handoff、underrun、hold 设计；无发送代码 |
| `test_*.py`、`benchmark_hand_retarget.py` | 单元/集成测试和纯计算计时 |

原有 replay、replay_hand、teleop 未修改。`client` 在改动前备份到
`backups/pre_xflow_20261114_1842`；该目录不入 Git。Git 从 baseline `7393eac` 建立，所有
实现均为 client-only 小提交。

## 2. 来源、版本和完整性

- Wuji vendor：65 文件，tree SHA256
  `f3fb5e0ec0b088ddce5669b45b47a8ae24f3b62b2492864b016232cc34366485`；
  `retarget.py` 为 `08a6e024…b759c0`，`convert.py` 为 `fce6ec4b…61bf3`。两份 MIT
  许可证均原样保留。vendor 左右 URDF 与最近 replay 的 `wuji_*_original.urdf` 字节一致。
- XFlow `wire_v4.py` 原样复制，SHA256
  `6850b72c2e13ba396a201d8c0bd166120130bed91015895038619e93401c5c80`。
- 当前服务：protocol `xflow_kp21_chunk_v1` / v4，model `hh_a12`，deployment
  `34ca15a63ef03aa8`，任务 endpoint `10.0.0.142:18765`，启动于 17:24:54+08:00。
- 服务指南、server、core 的核验哈希分别为 `9117c0d7…5427`、`7829a3b4…a8d`、
  `e3d5db23…c2e5b`；全部来源及完整哈希见 `SOURCES.json`。
- 录制状态 fixture 来自 episode 023 的 bag image index 800，bag 用 sqlite URI
  `mode=ro` 打开；fixture SHA256
  `7edec50df3142d83cc95176016ab37500e4256912008c42d22f4fbba3a4d7d86`。

## 3. 测试结果

主测试命令：

```bash
cd /home/guest/pengyang/client
/home/guest/.pixi/bin/pixi run --locked python -m pytest -q
```

覆盖原有 3 项及新增坐标、vendor、两种手根、时间表、RPC fake server、图像映射、观测缓冲、
relay 和 32 步 recorded-state dry-run。端到端测试单独实测 `1 passed in 22.42s`；最终全量
结果为 `34 passed in 22.46s`。

首次直接运行全量命令时发生过一次测试收集失败（不是算法失败）：pytest 递归进入了
`backups/`，三个同名 `test_online_core.py` 导致 `import file mismatch`，最终输出为
`2 errors during collection`。随后加入 `pytest.ini`，明确排除 `backups/runs/vendor`，再运行
得到上述 34/34 通过；备份内容没有删除或修改。

额外核验：

- `/usr/bin/python3 image_preprocess_cv2_reference.py`：OpenCV 4.2.0，map x/y 最大差均
  `0.0 px`（要求 ≤0.05 px）。
- vendor 完整性与 installed limit：`2 passed`；左右 20×2 limit 数值 exact。
- 在线 read-only observer：成功订阅/保存图像；完整 snapshot 因 live Wuji topic 无样本而
  fail-closed，未启动驱动。
- 离线 episode 023 快照：历史帧目标误差 0.081 ms；arm joint/EEF/Wuji 最近对齐最大
  2.971 ms（均小于 10 ms）。

## 4. 手部重定向 benchmark

zhuji、双手合计；每种模式均 64 次（32×左右）或 16 次（每 4 步×左右）：

| 模式 | 取样 | 单手帧 p50 / p95 | 双手 chunk 求解总和 | wall |
|---|---|---:|---:|---:|
| free_root | 32 步全算 | 1.686 / 1.960 ms | 106.44 ms | 122.70 ms |
| free_root | 每 4 步取 1 | 1.742 / 2.297 ms | 28.11 ms | 34.36 ms |
| kabsch_fixed_root | 32 步全算 | 2.835 / 3.195 ms | 178.09 ms | 185.92 ms |
| kabsch_fixed_root | 每 4 步取 1 | 3.274 / 3.788 ms | 48.79 ms | 54.93 ms |

全部帧接受。fixed 模式报告四次外迭代的累计 nfev，不再只报最后一次。

Wuji URDF 的关键发现：21 点中只有 point 0 对所有手指关节真正固定；
`[1,5,9,13,17]` 均随对应 flex 关节移动。neutral-FK Kabsch 在随机姿态可有 36.1 mm / 27.7°
最大偏差，因此实现改为当前实测 q 的 `[0,1,5,9,13,17]` Kabsch 与固定根 q 求解交替。
精确 FK 自动测试两种模式均满足 <0.1 mm / 0.05°。

## 5. 通信

中继为 Mac local forward + zhuji reverse forward，二者都只绑定 127.0.0.1；测试后已关闭。

| 路径 | info RTT p50 | 备注 |
|---|---:|---|
| b300 任务侧直达 | 4.80 ms | 只读 info |
| Mac → local SSH forward | 26.27 ms | 5 次；首次 59.98 ms |
| zhuji → reverse → Mac → b300 | 27.18 ms | 5 次；首次 59.27 ms，reset 25.87 ms |

最终 remote inference：virtual_cam0 RTT 917.9 ms，server total 810.7 ms，denoise
655.6 ms；stretch RTT 901.3 ms，server total 798.6 ms，denoise 649.2 ms。协议层严格核验
protocol/version/request/session/sample/model/deployment/shape/time/finiteness。服务的一个重要
细节是 `_check_session` 在后续字段检查前已经消耗 request_id/sample_ns；客户端因此在错误
路径也推进 attempt clock，STALE_SESSION 只允许 reset 后用新观测重试。

## 6. 图像

实时样例：

- `runs/image_samples_live_20261114_stage1_v2/raw_rgb.png`
- `runs/image_samples_live_20261114_stage1_v2/virtual_cam0_rgb320.png`
- `runs/image_samples_live_20261114_stage1_v2/stretch_undistort_rgb320.png`

实时 camera_info 相对 11-10 存档 fx/fy 差 0.0817%，低于 0.5% 门槛；D exact。virtual_cam0
黑边 `25227/102400 = 24.635742%`，人工查看与映射预期一致。`K_v` 按任务指定的
`diag(0.5,0.8,1) @ new_K`，并明确记录它与严格 cv2.resize pixel-center convention 的
0.25 px(x)/0.10 px(y) 主点差。

远端预测与该 bag 未来实测的对照仅是链路回归：bag 图像是有损 JPEG，且本次请求 task
固定为 `flower`，不能把误差当成任务模型精度。virtual_cam0 的 free-root SDK EEF 最大差
24.18 mm / 6.34°；stretch 为 31.85 mm / 30.69°。virtual_cam0 明显更接近该录制未来，
但仍需真实任务数据评估。

## 7. dry-run 与时间策略

用户最新时间策略已作为默认且唯一执行候选：第 `i` 点为
`exec_start + (i+1)/30 s`，32 点占 32 个发送周期、1.066666667 s，默认 `--steps 32`。
没有 `0.25` speed-scale 参数。`native_time` 仍可显式选择，但报告标记 diagnostic-only。
历史图保持 info 指定的 4 个相机帧（0.1333 s），不取 16 帧。

recorded-state fake-gt（30 Hz × 32，50 Hz 后 53 点）：

| 模式 | root 最大平移/转角 | SDK EEF 最大平移/转角 | 手速峰值 | Pin IK | 臂速峰值 |
|---|---:|---:|---:|---:|---:|
| free_root | 0.0134 mm / 0.0400° | 0.0133 mm / 0.0400° | 0.172 rad/s | 53/53 | 0.427 rad/s |
| fixed_root | 0.0038 mm / 0.0082° | 0.0026 mm / 0.0082° | 0.191 rad/s | 53/53 | 0.403 rad/s |

两者均 0 个 EEF_BOX/限位违规，四元数单位误差 ≤2.3e-16。native-time 诊断使用 0.2667 s
覆盖，几何误差相同；它不是当前默认。

远端预测的离线安全门全部失败，原因是速度而非可达性/工作空间：

| 图像 / 根模式 | 手 q 速度峰值 | 臂 IK 速度峰值 | IK/EEF_BOX | 总安全门 |
|---|---:|---:|---|---|
| virtual / free | 1.068 rad/s | 0.961 rad/s | 53/53，0 违规 | fail（手速） |
| virtual / fixed | 1.399 rad/s | 1.291 rad/s | 53/53，0 违规 | fail |
| stretch / free | 1.156 rad/s | 1.392 rad/s | 53/53，0 违规 | fail |
| stretch / fixed | 1.654 rad/s | 1.770 rad/s | 53/53，0 违规 | fail |

因此 remote dry-run “流程完成”不等于可执行；所有报告仍为 execution-ready false。

## 8. 关节/EEF 契约

- runtime q20 名为 `left/right_finger1_joint1 … finger5_joint4`；vendor anatomical 名仅顺序
  对应，API 已分别暴露 `runtime_joint_names` / `model_joint_names`。
- 历史 bag 左/右 Wuji 4359/4348 条的名称顺序 mismatch 为 0、非有限为 0、限位违规为 0。
- episode 023 index 800 的 SDK EEF vs 录制 FK：左 1.178 mm / 0.0999°，右
  0.0668 mm / 0.0129°。这说明直接 SDK EEF 是当前 proprio 主输入，joint-FK 适合作交叉诊断。
- 5 cm 是物理转接件近似，不是 SDK EEF 与 Wuji base_link 坐标原点距离。代码继续使用完整
  标定链；原点距离左 72.810 mm、右 69.014 mm，不额外加 5 cm。

## 9. 未决问题与进入第二阶段的门槛

1. live `/wuji/left|right/joint_states` 当前 0 publisher；完整实时 snapshot 必须继续拒绝，
   不能为了测试启动驱动。
2. `zhuji_camera_41p5.json` 的头部 41.5° 外参仍为运动学推算、`validation_passed=false`；
   需要物理复核。
3. remote 预测在用户指定的 30 Hz×32 时间表下触发 1 rad/s 手/臂速度门。需要决定是拒绝
   整段、只执行安全前缀、加入有界平滑，还是重新训练/调整输出；本阶段不擅自降速。
4. virtual_cam0 有 24.6% 黑边；stretch 虽无同类黑边，但本次录制未来对照更差。需要模型
   团队用任务匹配、无损在线数据决定生产模式。
5. 普通跟踪误差/队列耗尽的设计是受控停止 ServoP + 验证 controller hold，不发
   `request_emgy_stop`；“保持高级开发者模式”仍需厂商控制器台架验证。
6. 当前只完成发送层设计。实时硬件 sender、启动接近、hold 故障注入和分阶段真机认证均未
   获得本阶段授权，也没有被实现。

