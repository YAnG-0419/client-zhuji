"""Verify and record the completed offline threshold update; no device I/O."""
from pathlib import Path
import difflib
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
UPDATE = ROOT/'replay_robot/threshold_update_20261111'
RESULT = ROOT/'replay_robot/threshold_recalculation_20261111'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def describe(rows):
    return '；'.join('%.2f–%.2f' % (r['start_s'], r['end_s']) for r in rows) or '无'


def main():
    reports = [RESULT/('episode'+str(e))/policy/'report.json'
               for e in (594, 615, 616) for policy in ('movej', 'urdf')]
    assert all(p.is_file() for p in reports), 'Recomputation is not complete'
    sys.path[:0] = [str(ROOT/'replay'), str(ROOT/'tools')]
    from transport import replay_backend
    from replay_thresholds import IK_POSITION_TOL, IK_ROTATION_TOL, CHORD_ROTATION_TOL, MODEL_CLEARANCE_M
    _, Model, _ = replay_backend(); model = Model()
    verified = []; rows = []
    for p in reports:
        r = json.loads(p.read_text())
        for name, digest in r['source_sha256'].items():
            assert sha(ROOT/name) == digest, name
        data = np.load(p.parent/'diagnostics_not_for_execution.npz', allow_pickle=False)
        src = np.load(ROOT/'replay_robot/task_transfer_20261110/episodes'/r['episode']/'replay_input.npz', allow_pickle=False)
        np.testing.assert_array_equal(src['t'], data['t'])
        pe, re = [], []
        for q, target in zip(data['diagnostic_q16'], src['eef']):
            model.set_q(q); actual = model.poses()
            pe.append(np.linalg.norm(actual[:, :3]-target[:, :3], axis=1))
            re.append((Rotation.from_quat(actual[:, 3:]).inv()*Rotation.from_quat(target[:, 3:])).magnitude())
        pe, re = np.asarray(pe), np.asarray(re)
        np.testing.assert_allclose(pe, data['position_error_m'], atol=1e-7, rtol=0)
        np.testing.assert_allclose(re, data['rotation_error_rad'], atol=1e-7, rtol=0)
        valid = (pe <= IK_POSITION_TOL) & (re <= IK_ROTATION_TOL)
        np.testing.assert_array_equal(valid, data['pose_pass'])
        assert int(valid.all(axis=1).sum()) == r['both_pose_passing_frames']
        assert (data['internal_wrist_distance_m'][:, 0] < MODEL_CLEARANCE_M).all()
        assert not data['collision_pass'].any() and not data['combined_pass'].any()
        verified.append(dict(episode=r['episode'], policy=r['policy'], frames=len(pe),
                             independent_fk='MuJoCo vs saved Pinocchio residuals within 1e-7',
                             original_time_and_input_hash_verified=True))
        rows.append(r)

    # Preserve evidence from the unmodified CLI flow (offline mode only).
    runs = {594: 'replay_full_20261111_105509_11571',
            615: 'replay_full_20261111_105509_11537',
            616: 'replay_full_20261111_105508_11399'}
    integration = []
    for e, run in runs.items():
        src = ROOT/'replay_hand/runs'/run
        result = json.loads((src/'result.json').read_text())
        assert not result['execute_requested'] and result.get('arm_motion_commands_sent', 0) == 0
        assert 'no first-frame cross IK candidate passed' in result['error']
        dest = RESULT/('episode'+str(e))/'existing_replay'; dest.mkdir(exist_ok=True)
        for name in ['result.json', 'cross_seed_failure.json']:
            shutil.copy2(src/name, dest/name)
        integration.append(dict(episode=e, run=str(src), error=result['error'], motion_commands_sent=0))

    changes = json.loads((UPDATE/'changes.json').read_text()); diffs = []
    for name, item in changes.items():
        assert sha(ROOT/name) == item['after'], name
        before = UPDATE/'before'/name
        if item['before'] is not None:
            assert sha(before) == item['before']
        assert sha(UPDATE/'after'/name) == item['after']
        diffs.extend(difflib.unified_diff(before.read_text().splitlines(True) if before.exists() else [],
                    (ROOT/name).read_text().splitlines(True), fromfile='before/'+name, tofile='after/'+name))
    (UPDATE/'changes.diff').write_text(''.join(diffs))
    tests = '\n'.join((UPDATE/name).read_text() for name in ['tests.log', 'planner_tests.log', 'anchor_tests.log'])
    assert '75 passed' in tests and tests.count('4 passed') == 2
    summary = dict(thresholds=dict(position_m=IK_POSITION_TOL, orientation_deg=2., chord_orientation_deg=3.,
                    model_clearance_m=MODEL_CLEARANCE_M, internal_wrist_clearance_m=MODEL_CLEARANCE_M),
                   offline_tests_passed=83, changed_files=len(changes), motion_commands_sent=0,
                   execution_ready=False, results=rows, independent_verification=verified,
                   existing_replay_checks=integration,
                   preserved=['Original goals and timestamps', 'B-point calibration', 'Joint limits',
                              'Speed limits', 'First-frame FK consistency thresholds', 'Historical results'])
    write_json(RESULT/'summary.json', summary)
    write_json(RESULT/'independent_verification.json', verified)
    text = ['# 2026-11-11 新阈值离线重算', '',
            '位置误差 20 mm，单帧朝向误差 2°，插值朝向偏差 3°（插值位置误差同为 20 mm）。',
            '所有受检模型部件对的碰撞间距均为 20 mm；按用户明确要求，取消同腕 yaw–roll 的 1 mm 例外。', '',
            '**三条轨迹在两种关节范围下均无双臂同时满足位姿要求的帧；本次候选姿态也全部未通过 20 mm 碰撞要求。没有执行机器人。**', '',
            '以下时间为原始轨迹时间，单臂区间只表示位姿逆解通过，不是可执行回放区间。', '',
            '| 关节范围 | Episode | 左臂位姿通过段（秒） | 右臂位姿通过段（秒） | 双臂位姿同时通过 |',
            '| --- | --- | --- | --- | --- |']
    for r in rows:
        text.append('| %s | %s | %s | %s | %d 帧 |' % (
            '当前 MoveJ∩URDF' if r['policy']=='movej' else 'URDF-only 诊断', r['episode'],
            describe(r['left_pose_intervals']), describe(r['right_pose_intervals']), r['both_pose_passing_frames']))
    text += ['', '## 碰撞结果', '',
             '| 关节范围 | Episode | 左腕内部间距范围（mm） | 低于 20 mm 的帧数 | 全部受检模型通过 |',
             '| --- | --- | --- | --- | --- |']
    for r in rows:
        c = r['collision']; low, high = c['left_internal_wrist_distance_range_mm']
        text.append('| %s | %s | %.2f–%.2f | %d / %d | %d 帧 |' %
            (r['policy'], r['episode'], low, high, c['internal_wrist_under_20mm_frames'][0],
             r['source_frames'], c['all_model_passing_frames']))
    text += ['', '受检机械臂/躯干/头部部件对已因左腕内部距离不足而拒绝，因此没有继续计算手与躯干的距离，也没有进行整段扫掠认证。',
             '这说明本次得到的候选关节解不满足新间距，不构成其他关节解全局不存在的证明。环境、线缆、适配件和手内碰撞不在本次认证范围。', '',
             '## 计算与验证', '',
             '保留满足新阈值的旧解，对超差帧按新位置/朝向容差比例重新作有界逆解；必要时从前一通过帧重试，目标和时间均未改变。',
             '对相邻位姿通过帧，按原 30 Hz 时间轴在两帧间增加 9 个检查点，约为 300 Hz，检查关节插值对应末端位置/朝向偏差。逐策略结果见 report.json。',
             '全部 6274 组双臂候选姿态由独立 MuJoCo FK 复核，与 Pinocchio 残差相差不超过 1e-7。源输入及代码哈希已核验。',
             '83 项离线测试通过；另将三个原始转换输入交给现有 replay_full 默认离线入口，均在首帧逆解阶段拒绝，运动命令为零。',
             'URDF-only 只是模型范围对照，不是已验证的 ServoP 范围。原关节限位、速度限制、首帧 FK 一致性检查未调整。',
             '源码阈值统一在 `replay/replay_thresholds.py`；修改备份与差异位于 `replay_robot/threshold_update_20261111`。', '']
    (RESULT/'README.md').write_text('\n'.join(text))
    (UPDATE/'README.md').write_text('''# 回放阈值更新记录（2026-11-11）

按用户要求：位置误差 20 mm、朝向误差 2°、插值朝向偏差 3°、所有受检模型碰撞间距 20 mm，包含原本例外的同腕 yaw–roll 部件对。

统一设置位于 `replay/replay_thresholds.py`。修改已覆盖主规划器、跨机器人与 Pinocchio 入口、模型距离、离线扫掠、实测起点的 MoveJ 入场认证、在线臂/手体间距和放置诊断。自适应入场实现从原 `wanchi/replay_human_20261102/movej_path.py` 复制到回放目录，保留算法与原许可证记录，使用新的 20 mm 约束；未修改 Wanchi 的共享文件。

`before/` 保存原文件，`after/` 保存应用内容，`changes.json` 保存前后哈希，`changes.diff` 保存差异。83 项离线测试通过；测试日志同目录。旧测试文件导入的 `test_ik_ready` 是准备姿态辅助函数，不是 pytest 测试，因此显式排除该误收集项；另 6 项与本次修改无关的 bag 测试未纳入定向规划器检查。

未发设备命令，未重启设备或控制进程。原轨迹、B 点标定、关节限位、速度设置和历史结果未修改。新参数用于后续加载该代码的进程。完整重算报告位于 `../threshold_recalculation_20261111/README.md`。
''')
    entry = dict(record='replay_robot/threshold_update_20261111/README.md',
                 changes='replay_robot/threshold_update_20261111/changes.json',
                 result='replay_robot/threshold_recalculation_20261111/summary.json',
                 thresholds=summary['thresholds'], offline_tests_passed=83,
                 motion_commands_sent=0, execution_ready=False,
                 adapted_source='/home/guest/wanchi/replay_human_20261102/movej_path.py',
                 adapted_source_sha256=sha(Path('/home/guest/wanchi/replay_human_20261102/movej_path.py')),
                 existing_source_licenses_retained=True,
                 code_hashes={name:item['after'] for name,item in changes.items()})
    for name in ['recompute_thresholds_20261111.py', 'finalize_threshold_recalculation_20261111.py']:
        entry['code_hashes']['replay_robot/'+name] = sha(ROOT/'replay_robot'/name)
    with open(ROOT/'replay_robot/.sources_update.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ROOT/'SOURCES.json'; sources = json.loads(path.read_text())
        key = 'replay_robot_threshold_update_20261111'; assert key not in sources
        shutil.copy2(path, UPDATE/'SOURCES_before.json'); sources[key] = entry
        fd, temp = tempfile.mkstemp(prefix='.threshold_sources_', dir=str(ROOT))
        with os.fdopen(fd, 'w') as f:
            json.dump(sources, f, ensure_ascii=False, indent=2); f.write('\n')
        os.chmod(temp, path.stat().st_mode & 0o777); os.replace(temp, str(path))
    write_json(UPDATE/'source_record.json', entry)
    for directory in [UPDATE, RESULT]:
        files = {str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*'))
                 if p.is_file() and p.name != 'SHA256SUMS.json'}
        write_json(directory/'SHA256SUMS.json', files)
    print('VERIFIED: 83 tests; 6274 independently checked candidates; zero passing bimanual frames; no motion')


if __name__ == '__main__':
    main()
