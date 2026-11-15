"""Collect measured offline results and their hashes; no control interfaces."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'replay_robot/task_transfer_20261110'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


rows=[]
summary=dict(status='conversion_complete_full_replay_blocked_by_ik',execution_ready=False,motion_commands_sent=0,
    target_head_pitch_deg=41.5,tool_origin='B: board center = l_base_link/r_base_link origin',
    source_mapping_policy='per-arm nominal source base removed, corresponding measured camera extrinsic applied',
    target_mapping_policy='common left-derived 41.5-degree camera extrinsic for both arms; side-specific EEF-to-B mounting transforms',
    coordinate_regression_tests_passed=9,episodes={})
for ep in ['episode594','episode615','episode616']:
    d=WORK/'episodes'/ep;c=read(d/'conversion_report.json');r=read(d/'offline_check.json');h=read(d/'target_hand_collision.json')
    assert sha(d/'replay_input.npz')==c['output_sha256']
    first=r['independent_first_frame_best'];passed=r['all_original_frame_ik']
    summary['episodes'][ep]=dict(frames=c['frames'],replay_input=str(d/'replay_input.npz'),input_sha256=c['output_sha256'],
        replay_loader_passed=r['current_loader_passed'],current_replay_offline_completed=False,
        first_frame_position_residual_mm=[v*1000 for v in first['position_m']],
        first_frame_rotation_residual_rad=first['rotation_rad'],
        original_frames_both_arms_passed=passed['both_arms_passing_frames'],
        hand_target_sampled_intersections=h['sampled_intersections_detected'],hand_target_min_separation_bound_m=h['minimum_separation_lower_bound_m'],
        hand_timing=r['hand_timing'],execution_ready=False)
    rows.append('| %s | %d | %.1f / %.1f | %.1f / %.1f | 未通过 |' %
                (ep.replace('episode',''),c['frames'],*(v*1000 for v in first['position_m']),*(v*180/3.141592653589793 for v in first['rotation_rad'])))
summary['pending']=['Valid continuous dual-arm IK under current executable joint limits',
    'Valid-path arm/hand/body swept collision, acceleration-aware synchronized timing and measured-start approach',
    'Environment, measured adapter/cable geometry, workcell/physical accuracy and live tracking validation']
summary['provenance']={str(p.relative_to(ROOT)):sha(p) for p in [
    ROOT/'replay_robot/prepare_task_replay.py',ROOT/'replay_robot/check_task_replay.py',
    ROOT/'replay_robot/check_target_hand_collision.py',ROOT/'replay_robot/test_task_transfer.py',
    ROOT/'Pinocchio/pin_model.py',ROOT/'replay_hand/episode_cross.py',ROOT/'replay_hand/planner_cross.py',
    ROOT/'replay_hand/full_sync.py',ROOT/'replay_hand/replay_full.py',ROOT/'replay/transport.py',WORK/'config.json',
    WORK/'inputs/task_mapping.json']}
save(WORK/'summary.json',summary)
text='''# 三条新轨迹的任务映射与离线回放检查

2026-11-10：用户确认本次 zhuji 拍板时头部俯仰约 41.5°。

**基座坐标转换和 replay 输入适配已完成；三条轨迹均被当前双臂 IK 检查阻止，不能直接实机回放。所有检查均为离线，运动命令为零。**

## 已固定的转换关系

- 手部参考点是 B：各侧板中心与 `l_base_link/r_base_link` 原点重合。
- 左手与板轴一致，右手 `R_board_hand = Rz(+90°)`；转换工具位姿时使用逆变换，不重复原包局部轴向修正，不增加 28.5 mm 或约 50 mm 平移。
- 源场景先移除各臂 URDF 的名义基座安装，再接入该臂实测 `camera_T_base`。两侧独立源 FK 对整批轨迹复核通过。
- 使用实拍 `task_mapping.json`，方向为源 RGB optical → 目标 RGB optical；闭合通过。
- 两手共用已有左臂标定推算的目标 41.5° `T_base_camera`。右臂估计只用于敏感性比较，不能把它当成另一个目标世界。采用另一估计时，本批 B 点最大差异约 18.3 mm。
- 41.5° 是本次操作者确认；偏航沿用归档外参值 -0.0401°，未在本次观测中独立记录。

列向量下每臂使用：

`T_tron_H = T_tron_targetCamera @ T_targetCamera_sourceCamera @ T_sourceCamera_sourceBase @ inverse(T_scene_sourceBase) @ T_scene_H`

`T_tron_B = T_tron_H @ inverse(T_B_H)`；`T_tron_E = T_tron_B @ inverse(T_E_B)`。

其中 E 为 SDK 末端。源安装/相机外参对三次录制的适用性仍来自现有模型和记录，未新增物理验收。

## 实际检查结果

- 9 项坐标回归检查通过；原包和冻结模型未改动。
- 3,137 个原始帧全部完成转换，B 点定义保持，手指关节值和时间戳保持。
- 三个新输入通过现有 replay 严格加载器及 50 Hz 重采样；手指顺序、限位、数值和 ServoP 位置盒检查通过。
- 现有完整 replay 默认离线入口均在首帧 IK 失败。用独立 Pinocchio、12 组首帧初值及逐帧有界求解复核后，仍没有合格完整双臂路径。这是求解检查结果，不声称数学上排除了所有可能分支。

| Episode | 原始帧 | 首帧位置残差，左/右 mm | 首帧角度残差，左/右 ° | 完整双臂路径 |
| --- | ---: | ---: | ---: | --- |
'''+ '\n'.join(rows)+'''

当前门限为位置 10 mm、旋转 0.015 rad（约 0.86°）。最佳首帧解触及腕 yaw/roll 限位；机械范围诊断也未解决目标位姿残差。没有放宽执行限位或误差门限。

双手间已用每侧 22 个碰撞网格的保守包围球筛查全部原始采样点，间距下界分别约 283 / 291 / 300 mm，因此采样点没有双手互撞；无需进入 FCL 窄相查询。这不覆盖手内自碰撞、插值扫掠、手臂/身体、环境、转接件或线缆。

手指原速峰值约 4.066 / 3.246 / 2.857 rad/s，超过当前 1 rad/s 目标上限。已计算限速需求，不能按原速直接播放；最终手臂与手的共同时间轴和加速度检查必须基于有效机械臂路径，目前没有生成可执行时序。

## 文件与复现

每段目录 `episodes/episode{594,615,616}/`：

- `replay_input.npz`：格式兼容的目标输入，`execution_ready=false`；包含 SDK EEF、B 点、手根和手指序列。
- `conversion_report.json`：各段变换链、原点保持、数据哈希和外参差异。
- `existing_result.json`、`existing_cross_seed_failure.json`：现有 replay 的实际失败报告。
- `offline_check.json`：独立逐帧源 FK、目标 IK、位置盒和手指限速诊断。
- `ik_diagnostics_not_for_execution.npz`：含残差超限的诊断解，**不是执行计划**。
- `target_hand_collision.json`：采样双手间碰撞筛查及覆盖限制。

从 `/home/guest/pengyang` 运行以下检查，不加 `--execute` 或 `--check`：

```bash
/home/guest/.pixi/bin/pixi run --locked python replay_hand/replay_full.py --episode replay_robot/task_transfer_20261110/episodes/episode594/replay_input.npz
/home/guest/.pixi/bin/pixi run --locked python replay_robot/check_task_replay.py --episode episode594
/home/guest/.pixi/bin/pixi run --locked python replay_robot/check_target_hand_collision.py --episode episode594
```

另外两段替换 episode 编号即可。`prepare_task_replay.py` 拒绝覆盖已有输出。新的转换应使用新工作目录并保留原始记录。

## 阻碍与下一步

当前不能生成合格的首帧目标，因此不能认证从实际姿态到首帧的过渡、完整整机扫掠碰撞和执行时序。继续推进需要核实目标位姿/物理安装或调整整体任务摆位，再重新规划；没有自动移动、旋转或缩放这批任务来绕过失败。

任何后续任务摆位调整都应对双手使用同一个变换，并同步考虑工作台物体位置。标定板、相机和机器人基座关系不应为求解通过而被改写。
'''
(WORK/'README.md').write_text(text)
save(WORK/'SHA256SUMS.json',{str(p.relative_to(WORK)):sha(p) for p in sorted(WORK.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
print(json.dumps(summary['episodes'],ensure_ascii=False,indent=2))
