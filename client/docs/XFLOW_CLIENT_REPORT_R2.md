# XFlow TRON2 + Wuji Hand 2 client 第二轮报告

日期：2026-11-14　分支：`xflow-client`　基线：`r1-review-baseline` / `021e1c1`

结论：同步保持、16 帧执行历史、局部右乘锚定、平滑、首跳速度门、缓存重放和只订阅影子
模式均已实现。首个执行点现在与实测 EEF/q20 数值一致；但新的 4 个真实 hh_a12 快照在默认
`decay/12/window5/reject` 下只有 1/4 同时通过手和臂的 1 rad/s 门，因此所有产物仍为
`execution_ready=false`，不能接硬件发送层。

## 提交

从 `r1-review-baseline` 起：

| commit | 内容 |
|---|---|
| `0210ad1` | sync-hold 时间策略、16 帧执行历史、4 s buffer |
| `6efae7a` | 局部锚定、平滑、首跳速度门、多快照/cache replay |
| `b4083b9` | subscription-only `live_dryrun` 与可省略历史图的 RPC |
| `a5393d3` | 显式系统 ROS Python/DDS 环境 |
| `2d5a75e` | clamp_report 同时覆盖 hand q20 与 arm q16 |
| `53b27b5` | 影子模式 dispatch 前二次实测静止检查 |
| 本报告所在提交 | 文档、R1 报告归档、最终审计 |

## 2.1–2.8 完成情况

1. **sync_hold：完成。** `frame_per_step_30hz` 只允许 `sync_hold`，32 点完整保留；
   `exec_start=dispatch+reserve`。`native_time+realtime_cutoff` 仅作诊断。静止门默认 2 mm / 1° /
   0.05 rad；离线明确记录 `unverified_offline`，影子模式会采第二份实测再检查。
2. **锚定：完成。** 使用 `E=inv(T_fit0)@T_meas` 和 `T_fit@E_k`；旋转用 rotvec
   exp/log 缩放，覆盖接近 180° 的测试；q20 同权重衰减并限位。逐步权重、局部偏移和首点
   前后误差均保存。
3. **平滑/速度门：完成。** 根平移、相对首帧旋转切空间、q20 在锚定前做 SG 平滑；手、
   Pinocchio q16、SDK EEF 均含“实测→step0”。`reject` 为默认；`clamp_report` 只报告裁剪，
   永不标记可执行。真实多快照仍有速度拒绝，见下表。
4. **root auto：完成。** `humanhand→free_root`，`realrobot→kabsch_fixed_root`；CLI 可显式覆盖。
5. **执行历史：完成。** 默认执行时钟 16 周期 / 0.533 s；hold 不推进；离线 wall=execution；
   shadow 没有执行历史时省略 `history_rgb` 并记录原因。buffer=4 s，动态容差为近期中位周期
   一半并夹到 16.7–50 ms。
6. **多快照：完成最低项。** episode023 manifest 的任务为 `flower`；在 20/40/60/80% 四个
   快照、`virtual_cam0` 上调用 4 次并缓存。后续全部调参为缓存重放，未加 stretch 新请求或
   5/10 step 对照。第一轮 virtual/stretch 已以等价缓存重放。
7. **live_dryrun：代码和测试完成，实测周期被启动门阻止。** ROS 子进程只有 subscription；
   `/wuji/left/joint_states`、`/wuji/right/joint_states` 无 publisher，故在 infer 前退出，0 次
   infer。没有启动驱动。
8. **文档：完成。** `execution_layer_design.md` 已更新同步周期、锚定、速度拒绝、50 Hz EEF
   和 q20 输出；仍无 sender。R1 报告移到 `docs/IMPLEMENTATION_REPORT_R1.md`，根文件留链接。

## 测试

```bash
cd /home/guest/pengyang
/home/guest/.pixi/bin/pixi run --locked pytest -q client
```

结果：`47 passed in 21.43s`。原 34 项加 13 项第二轮测试全部通过。测试覆盖 180° 附近旋转、
三种锚定模式、首跳速度、动态历史容差、hold 跳过、无历史 shadow、可省略 history RPC、
缺 Wuji 话题 fail-closed，以及源码中无 `create_publisher`/急停调用。

fake-gt 最终结果：

| root mode | root max mm / deg | SDK EEF max mm / deg | hand peak | arm peak |
|---|---:|---:|---:|---:|
| free_root | 0.0142 / 0.0399 | 0.0133 / 0.0399 | 0.178 | 0.703 |
| kabsch_fixed_root | 0.0038 / 0.0082 | 0.0026 / 0.0082 | 0.194 | 0.494 |

fake-gt 的 decay/constant/none 三种模式 EEF 最大误差相同（0.0133 mm / 0.0399°），首点 q
误差均为 0，说明近零锚定偏移不会改变几何回归。

## 第一轮产物缓存重放

配置均为 sync_hold、window5、free_root；速度单位 rad/s。锚定模式 decay/constant 的首点
EEF/q20 误差为数值零；none 保留 reviewer 指出的真实首跳。

| R1 响应 | anchor | hand peak | arm peak | step0 EEF mm / deg | step0 q max |
|---|---|---:|---:|---:|---:|
| virtual | decay | 0.945 | 0.956 | 0 / 0 | 0 |
| virtual | constant | 0.762 | 0.480 | 0 / 0 | 0 |
| virtual | none | 4.136 | 0.808 | 0.992 / 3.235 | 0.138 |
| stretch | decay | 0.923 | 1.073 | 0 / 0 | 0 |
| stretch | constant | 0.923 | 1.040 | 0 / 0 | 0 |
| stretch | none | 5.107 | 1.046 | 1.010 / 3.933 | 0.170 |

这证明首跳遗漏已经修复；virtual 可通过，stretch 的 arm 瓶颈仍比门高 4–7%。

## 新的 4 快照真实推理

服务：`aw-hha12-5090-1114-c7d1`，endpoint 在运行前由 `squeue --pool 5090 --me` 和
`endpoint.json` 刷新为 `10.0.0.142:18765`，deployment `34ca15a63ef03aa8`。任务名来自
episode023 manifest：`flower`。下表为默认 decay/12/window5/reject。

| snapshot | RTT ms | plan ms | L offset mm/deg/q | R offset mm/deg/q | hand peak | arm peak | tip max mm | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| frac020 | 1042 | 269 | 2.86/1.56/0.185 | 3.25/1.98/0.148 | 2.704 | 1.940 | 0.326 | no |
| frac040 | 942 | 199 | 4.92/2.45/0.288 | 1.74/1.54/0.156 | 1.190 | 0.534 | 0.351 | no |
| frac060 | 941 | 211 | 2.00/2.36/0.092 | 0.90/1.62/0.245 | 0.878 | 0.977 | 0.365 | yes |
| frac080 | 918 | 422 | 2.24/1.93/0.114 | 2.92/2.21/0.276 | 1.702 | 1.342 | 0.328 | no |

四个快照都无 retarget 拒绝帧。首个执行点最大 SDK EEF 误差 < `2e-13 mm / 8e-15°`，
q20 误差为 0，远小于 0.5 mm / 0.2° / 0.01 rad。post-inference 规划耗时 p95=398.9 ms，
通过 0.5 s 门。

### 平滑前后抖动（四快照均值）

二阶差分为逐执行帧离散量，不除以 dt²。

| side | phase | root translation mm | root rotation rad | q20 rad |
|---|---|---:|---:|---:|
| left | fit | 1.739 | 0.0195 | 0.0484 |
| left | window5 | 0.634 | 0.00727 | 0.0196 |
| right | fit | 1.331 | 0.0143 | 0.0372 |
| right | window5 | 0.479 | 0.00487 | 0.0139 |

window5 把二阶差分降低约 60–66%，但速度不全通过。缓存参数对比（4 快照峰值/均值）：

| 参数 | hand max/mean | arm max/mean | 全门通过数 |
|---|---:|---:|---:|
| decay, window1, d12 | 3.172 / 2.007 | 2.697 / 1.557 | 0/4 |
| decay, window5, d12 | 2.704 / 1.619 | 1.940 / 1.199 | 1/4 |
| decay, window9, d12 | 1.407 / 1.189 | 1.405 / 0.899 | 1/4 |
| constant, window5 | 2.735 / 1.346 | 1.802 / 1.142 | 2/4 |
| none, window5 | 8.636 / 7.452 | 5.251 / 2.044 | 0/4 |
| decay, window5, d24 | 2.727 / 1.404 | 1.918 / 1.181 | 2/4 |

加大窗口或延长 decay 不能稳定消除瓶颈；window15/constant 能让四条手轨迹均 ≤1，但
frac020 arm 仍约 1.35 rad/s，而且永久偏移和过强平滑会显著改变模型轨迹，所以未改默认值。
`reject` 正确地把这些段判为不可执行；没有靠降速、静默裁剪或漏掉首跳来制造通过结果。
最终 `clamp_report` 诊断中，frac020 的 hand/arm 分别裁 9/29 步（最大 0.073/0.093 rad），
frac040 为 6/0 步，frac060 为 0/0 步，frac080 为 10/4 步；全部仍明确
`execution_ready=false`。

### 偏移稳定性与“人手化输入”

| side | root translation mm | root rotation deg | q offset max rad | kp21 local mean mm | wrist t0 residual mm |
|---|---:|---:|---:|---:|---:|
| left | 3.00 ± 1.15 | 2.07 ± 0.36 | 0.170 ± 0.076 | 3.00 ± 0.43 | 3.85 ± 1.48 |
| right | 2.20 ± 0.94 | 1.83 ± 0.27 | 0.206 ± 0.055 | 3.27 ± 0.50 | 1.79 ± 0.84 |

旋转偏移方向/幅值有一定稳定性，但 q 与平移偏移跨快照变化明显。第一预测帧相对当前 kp21
有稳定的约 3 mm 人手形状偏差，值得做“人手化输入”实验；应使用缓存或新预算单独 A/B，
不能把当前 4 个样本的均值硬编码为补偿。

## 产物

- `runs/r2-multisnapshot-episode023-v2/`：只读 bag 的 4 个执行历史快照；
- `runs/r2-response-cache-episode023-virtual/`：4 个请求元数据和原始响应；
- `runs/r2-remote-episode023-virtual/`：默认真实推理规划结果；
- `runs/r2-final-cache-default/`：最终代码的默认缓存重放，含 `unverified_offline`；
- `runs/r2-final-cache-clamp-report/`：最终 hand/arm 裁剪诊断；
- `runs/r2-r1-replay-{virtual,stretch}-{decay,constant,none}/`：第一轮响应等价缓存重放；
- `runs/r2-cache-eval-*/`：窗口、anchor、decay、clamp_report 缓存对比；
- `runs/r2-fake-gt-*`：fake-gt 验收；
- `runs/r2-live-preflight-final-20261114/`：缺 Wuji 状态话题的 fail-closed 证据。

## infer 计数与安全审计

| 项目 | infer |
|---|---:|
| episode023 × 4 fractions × virtual_cam0 | 4 |
| 第一轮 virtual/stretch 重放 | 0 |
| 全部参数调优/cache replay | 0 |
| fake-gt | 0 |
| live_dryrun preflight | 0 |
| **第二轮总计** | **4（≤30）** |

未读取 token，未提交/停止/修改集群任务；中继在 4 次推理后关闭。未启动 Wuji 驱动或任何
服务，未创建 publisher，未调用 ServoP/ServoJ/MoveJ/MoveH/`request_emgy_stop`，未发送命令。

## 建议默认值与待用户决定

- anchor：继续 `decay`；永久 `constant` 虽较慢，但会保留整段姿态偏置；
- `decay_steps=12`：保留用户指定默认，延长到 24 对最坏快照无实质改善；
- smoothing window：继续 5；9 可作诊断，15+ 过强且仍不能解决 arm 瓶颈；
- speed policy：必须 `reject`；`clamp_report` 仅用于看裁剪代价；
- `num_inference_steps=10`：本轮未做 5-step A/B，不应擅自降；
- 人手化输入：建议做，优先级高于继续加大平滑窗口；
- 硬件：在 4/4 多快照通过速度门、Wuji state 恢复、外参物理复核前，不进入发送阶段。

与 prompt 的唯一未完成实测项是 live shadow 周期：缺少两个 Wuji JointState publisher，启动门
按设计在 infer 前退出。可选 stretch 新推理和 5/10 denoise 对照未做；它们不是最低验收项，
且不会改变当前默认速度门已失败的结论。
