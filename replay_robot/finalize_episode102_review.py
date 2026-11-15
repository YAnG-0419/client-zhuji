"""Record completed episode102 offline inspection without device access."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'replay_robot/episode102_review_20261111'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def describe(rows):
    return '；'.join('%.2f–%.2f%s' % (r['start_s'], r['end_s'], '（仅 %d 帧）'%r['frames'] if r['frames'] < 3 else '') for r in rows) or '未找到'


def main():
    names = ['movej', 'urdf', 'movej_reseeded']
    reports = {name:json.loads((WORK/name/'report.json').read_text()) for name in names}
    conversion = json.loads((WORK/'conversion_report.json').read_text())
    assert sha(WORK/'replay_input.npz') == conversion['output_sha256']
    for path, digest in conversion['source_files'].items():
        assert sha(ROOT/path) == digest, path
    verification = []
    with np.load(WORK/'replay_input.npz', allow_pickle=False) as z:
        t, hands = z['t'].copy(), z['hands'].copy()
    with np.load(ROOT/'replay_robot/traj/episode102/episode102_replay_trajectory.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(t, z['time_s'])
        for a, side in enumerate(['left', 'right']):
            np.testing.assert_array_equal(hands[:, a], z[side+'_joint_positions_rad'])
    for name, report in reports.items():
        for path, digest in report['source_sha256'].items():
            assert sha(ROOT/path) == digest, path
        with np.load(WORK/name/'ik_diagnostics_not_for_execution.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(t, z['t'])
            mask = (z['position_error_m'] <= .020) & (z['rotation_error_rad'] <= np.deg2rad(2))
            np.testing.assert_array_equal(mask, z['pose_pass'])
            assert mask.sum(axis=0).tolist() == report['pose_passing_frames']
            assert int(mask.all(axis=1).sum()) == report['both_pose_passing_frames']
        assert report['independent_pinocchio_mujoco_residual_difference_max'] < 1e-7
        verification.append(dict(result=name, original_time_verified=True, pose_classification_verified=True,
                                 independent_fk_difference=report['independent_pinocchio_mujoco_residual_difference_max']))
    current, urdf = reports['movej_reseeded'], reports['urdf']
    assert current['both_pose_passing_frames'] == 0 and urdf['both_pose_passing_frames'] == 0
    run = ROOT/'replay_hand/runs/replay_full_20261111_112139_24557'
    actual = json.loads((run/'result.json').read_text())
    assert not actual['execute_requested'] and actual.get('arm_motion_commands_sent', 0) == 0
    assert actual.get('hand_target_updates', 0) == 0
    dest = WORK/'existing_replay'; dest.mkdir(exist_ok=True)
    for name in ['result.json', 'cross_seed_failure.json']:
        shutil.copy2(run/name, dest/name)
    summary = dict(episode='episode102', original_file_sha256='1988f7dd7b35dc61094ee16fd895dceb13a038179f23fa106c93b17c402c00c7',
                   frames=len(t), duration_s=float(t[-1]), thresholds=current['thresholds'],
                   current_movej_result=current, urdf_comparison=urdf,
                   initial_movej_result_superseded_by='movej_reseeded/report.json',
                   source_conversion=conversion, existing_replay_result=actual,
                   independent_verification=verification, motion_commands_sent=0, execution_ready=False)
    save(WORK/'summary.json', summary)
    text = ['# Episode 102 离线检查（2026-11-11）', '',
        '输入：`/Users/lpy/Downloads/episode102_replay_trajectory.npz`。已导入主机 `replay_robot/traj/episode102/`。',
        '863 帧、30 Hz、28.733333334 秒。原始手指关节值和时间戳未改变。', '',
        '文件声明源场景为 `lychee_root`，目标是 Wuji `l_base_link/r_base_link`，采用已有局部轴对齐版本。',
        '沿用用户确认的 B 点定义、左右安装关系、task_mapping.json 和共同的 41.5° 目标相机外参转换到 TRON2 基座。没有重新定位场景或增加腕内/适配器偏移。',
        '原始源臂关节未包含在这个导出中，因此无法独立复核源机器人 FK；源装配与手眼适用性沿用此前假设，目标相机运动学外参仍未独立实测验证。', '',
        '当前检查：位置误差 20 mm，朝向误差 2°，插值朝向偏差 3°；受检普通部件间距 5 mm，同腕 yaw–roll 间距 1 mm。', '',
        '## 位姿通过时间段', '',
        '以下是原始轨迹时间，只代表模型位姿误差通过，不能当作已验收的实机回放段。', '',
        '| 关节范围 | 左臂 | 右臂 | 双臂同时通过 |',
        '| --- | --- | --- | --- |',
        '| 当前 MoveJ∩URDF | %s | %s | %d 帧 |' % (describe(current['left_pose_intervals']), describe(current['right_pose_intervals']), current['both_pose_passing_frames']),
        '| URDF-only 对照 | %s | %s | %d 帧 |' % (describe(urdf['left_pose_intervals']), describe(urdf['right_pose_intervals']), urdf['both_pose_passing_frames']), '',
        '初次从首帧连续求解的 MoveJ 左臂没有通过点；随后用 URDF 解、home 和原回放首帧分支独立复核，在 8 秒与 10 秒处找到有效解。最终 MoveJ 结果采用从 8 秒有效解向前/向后追踪的复核，并保留初次结果作诊断。不能把单一局部逆解失败当成全局不可达证明。', '',
        '## 碰撞与连续性', '']
    for label, r in [('当前 MoveJ 范围', current), ('URDF 对照', urdf)]:
        c, ch = r['collision'], r['chord']
        text.append('- %s：模型采样碰撞通过 %d / %d 帧；左右臂插值偏差失败区间 %s；位姿通过段内超过 0.20 rad 的相邻关节步长数量 %s。' %
                    (label, c['all_model_passing_frames'], r['frames'], ch['failed_intervals_per_arm'], ch['large_joint_steps_in_passing_intervals']))
    text += ['', 'URDF 左臂原候选在 11.10–11.1333 秒、11.1667–11.20 秒有约 0.502 rad 和 1.047 rad 的关节跳变，未通过现有连续性门限。URDF 关节范围不是已验证的 ServoP 范围。',
        '碰撞结果针对计算出的候选关节姿态，包括已有机械臂/躯干/头部受检对及装配模型手与手/身体的受检对。归一化腕部间距分数不是实际几何距离。环境、线缆、适配件及同手内部碰撞不在认证范围。',
        '双臂位姿同时通过的区间为空，未进行整段双臂扫掠认证，也未做当前实测起点的 MoveJ 入场检查。', '',
        '## 数据与运行条件', '',
        '所有原始末端位置均处于当前 ServoP 位置盒内；位置盒通过不保证同时满足朝向与关节范围。',
        '手指峰值速度左 %.3f、右 %.3f rad/s，高于当前 1 rad/s 目标上限，需要共同时间轴重定时。只考虑手指速度的时间下界约 %.3f 秒，尚不是可执行的整段时长。' %
        (*conversion['hand_peak_speed_rad_s'], conversion['hand_only_retimed_lower_bound_s']),
        '现有 replay_full 默认离线入口在首帧逆解阶段拒绝；本次没有使用 --check/--execute，没有读取或控制实机，运动命令为零。',
        '三组诊断共 2589 个双臂姿态，均以独立 MuJoCo FK 复核 Pinocchio 残差，最大差小于 1e-7；目标、原时间与手指数组、来源哈希已验证。', '',
        '主要文件：`replay_input.npz` 为基座坐标下的转换数据；`movej_reseeded/report.json` 为最终当前范围结果；`urdf/report.json` 为模型范围对照；`summary.json` 为完整摘要。所有诊断文件都不是可直接执行的关节轨迹。', '']
    (WORK/'README.md').write_text('\n'.join(text))
    code = [ROOT/'replay_robot'/name for name in ['prepare_episode102_review.py', 'check_episode102_review.py',
                                               'check_episode102_reseeded.py', 'finalize_episode102_review.py']]
    entry = dict(record='replay_robot/episode102_review_20261111/README.md',
                 summary='replay_robot/episode102_review_20261111/summary.json',
                 original_sha256=summary['original_file_sha256'], thresholds=current['thresholds'],
                 code_hashes={str(p.relative_to(ROOT)):sha(p) for p in code},
                 reused=['replay_hand/build_exported_trajectory.py', 'replay_robot/prepare_task_replay.py',
                         'Pinocchio/pin_model.py', 'replay/transport.py', 'replay_hand/hand_body_clearance.py'],
                 original_source_licenses_retained=True, motion_commands_sent=0, execution_ready=False)
    with open(ROOT/'replay_robot/.sources_update.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX); path = ROOT/'SOURCES.json'
        sources = json.loads(path.read_text()); key = 'replay_robot_episode102_review_20261111'
        assert key not in sources; sources[key] = entry
        fd, temp = tempfile.mkstemp(prefix='.episode102_sources_', dir=str(ROOT))
        with os.fdopen(fd, 'w') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2); f.write('\n')
        os.chmod(temp, path.stat().st_mode & 0o777); os.replace(temp, str(path))
    save(WORK/'source_record.json', entry)
    save(WORK/'SHA256SUMS.json', {str(p.relative_to(WORK)):sha(p) for p in sorted(WORK.rglob('*'))
                                if p.is_file() and p.name != 'SHA256SUMS.json'})
    print('Verified episode102 reports, unchanged source arrays, independent FK, and zero device commands')


if __name__ == '__main__':
    main()
