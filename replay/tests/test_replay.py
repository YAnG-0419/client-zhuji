"""Native Foxy bag decoding and offline trajectory invariants; no hardware."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from episode import ROBOT_NAMES, SIDES, hand_names, load_episode, ordered, pose_error, sample

spec = importlib.util.spec_from_file_location('pengyang_replay', Path(__file__).resolve().parents[1]/'replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_slerp_shortest_quaternion_path_and_synchronized_hands():
    t = np.array([0., 2.])
    poses = np.zeros((2, 2, 7)); poses[..., 6] = 1
    poses[1, :, 6] = -1  # Same orientation, opposite quaternion sign.
    poses[1, :, 0] = 2
    h = np.zeros((2, 2, 20)); h[1] = 2
    p, hands = sample(t, poses, h, .5)
    assert np.allclose(p[:, 0], .5)
    assert np.allclose(hands, .5)
    assert np.max(pose_error(p, np.tile([.5, 0, 0, 0, 0, 0, 1], (2, 1)))[1]) < 1e-10


def test_hand_names_reordered_and_duplicate_rejected():
    from sensor_msgs.msg import JointState
    msg = JointState(); names = hand_names('left')
    msg.name = names[::-1]; msg.position = list(map(float, range(20)))
    assert np.array_equal(ordered(msg, names), np.arange(20)[::-1])
    msg.name[-1] = msg.name[0]
    with pytest.raises(ValueError, match='joint names'):
        ordered(msg, names)


def test_timing_enforces_arm_and_hand_limits():
    q = np.zeros(16); goal = q.copy(); goal[0] = .8
    duration = replay.timing(q, goal, 1., 2., .5)
    assert duration >= 2
    assert 1.875*.8/duration <= replay.ARM_SPEED_LIMIT
    assert 5.774*.8/duration**2 <= replay.ARM_ACCEL_LIMIT+1e-12
    assert 1.875*2/duration <= replay.HAND_SPEED_LIMIT


def test_session_accepts_integer_zero_velocity_from_robot_sdk(tmp_path):
    from types import SimpleNamespace
    info = {'accid': 'DACH_TRON2A_447', 'data': {'motor': 'OK', 'imu': 'OK'}}
    for key, value in (('working_mode', 'developer_mode'), ('collision_enable', True),
                       ('compliant_enable', False), ('enable_vr_follow_head', False)):
        info['data'][key] = value; info['data'][key+'_valid'] = True
    q = np.zeros(16, dtype=float)
    dq = np.zeros(16, dtype=np.int64)
    robot = SimpleNamespace(info=info, healthy=lambda: (q.copy(), dq.copy()))
    hands = SimpleNamespace(feedback=lambda: {s: [0.]*20 for s in SIDES},
                            node=SimpleNamespace(count_publishers=lambda topic: 1),
                            set_targets=lambda values: None)
    model = SimpleNamespace(lower=np.full(16, -2.), upper=np.full(16, 2.),
                            set_q=lambda value: None, clearance=lambda: (.02, ['a', 'b']))
    eef = np.zeros((2, 2, 7)); eef[..., 6] = 1
    session = replay.Session(robot, hands, model, np.array([0., 1.]), eef, np.zeros((2, 2, 20)), tmp_path)
    try:
        actual, velocity, _, _ = session.tick(np.zeros((2, 20)), q, q)
        assert np.array_equal(actual, q) and np.array_equal(velocity, dq)
    finally:
        session.feedback_log.close()


@pytest.mark.parametrize('position_error,rotation_error',
                         [(replay.IK_POSITION_TOL*2, 0.), (0., replay.IK_ROTATION_TOL*1.5)])
def test_unsolved_eef_is_rejected_with_diagnostics(tmp_path, monkeypatch, position_error, rotation_error):
    from scipy.spatial.transform import Rotation
    root = tmp_path/'project'
    (root/'record/config/model').mkdir(parents=True)
    (root/'record/config/model/robot.urdf').write_bytes(b'model')
    (root/'teleop/wuji_control/config').mkdir(parents=True)
    (root/'teleop/wuji_control/config/joint_limits.json').write_text(json.dumps({s: [[-1, 1]]*20 for s in SIDES}))
    monkeypatch.setattr(replay, 'ROOT', root)
    poses = np.zeros((2, 2, 7)); poses[..., 6] = 1
    class Unsolved:
        lower = np.full(16, -1.); upper = np.full(16, 1.)
        solved = False
        def set_q(self, q):
            pass
        def poses(self):
            p = poses[0].copy()
            if self.solved:
                p[1, 0] = position_error
                p[1, 3:] = Rotation.from_rotvec([rotation_error, 0., 0.]).as_quat()
            return p
        def solve(self, seed, target):
            self.solved = True
            return seed
        def check_path(self, *args, **kwargs):
            pytest.fail('Inaccurate IK must be rejected before execution planning')
    meta = {'fk': {'frame_id': 'base_link', 'quaternion_order': 'xyzw',
                   'sdk_eef_offset_from_wrist_roll_m': [-.0338, 0., -.2125],
                   'urdf_sha256': hashlib.sha256(b'model').hexdigest()}}
    with pytest.raises(ValueError, match='residual'):
        replay.make_plan(Unsolved(), np.array([0., 1.]), poses, np.zeros((2, 2, 20)),
                         np.zeros((2, 16)), meta, .5, tmp_path)
    diagnostic = json.loads((tmp_path/'ik_failure.json').read_text())
    assert diagnostic['position_error_m'][1] == position_error
    assert diagnostic['rotation_error_rad'][1] == pytest.approx(rotation_error)
    assert not (tmp_path/'plan.npz').exists()


@pytest.fixture
def bag_episode(tmp_path):
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    from rclpy.serialization import serialize_message
    folder = tmp_path/'bag'; folder.mkdir()
    db = sqlite3.connect(str(folder/'bag_0.db3'))
    db.executescript('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY,topic_id INTEGER,timestamp INTEGER,data BLOB);')
    topics = [('/joint_states', JointState, ROBOT_NAMES)]
    for s in SIDES:
        topics.extend([('/record/'+s+'_arm/ee_state_fk', PoseStamped, None),
                       ('/wuji/'+s+'/command', JointState, hand_names(s))])
    counts = {}
    for tid, (topic, cls, names) in enumerate(topics, 1):
        typ = 'sensor_msgs/msg/JointState' if names else 'geometry_msgs/msg/PoseStamped'
        db.execute('INSERT INTO topics VALUES(?,?,?)', (tid, topic, typ))
        for i in range(30):
            msg = cls()
            offset = tid*1000000 if names and names != ROBOT_NAMES else 0
            stamp = 1700000000000000000+i*50000000+offset
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
            if names:
                msg.name = names; msg.position = [i*.01]*len(names)
            else:
                msg.header.frame_id = 'base_link'; msg.pose.orientation.w = 1.
                msg.pose.position.x = i*.01
            db.execute('INSERT INTO messages(topic_id,timestamp,data) VALUES(?,?,?)',
                       (tid, stamp+200000000, bytes(serialize_message(msg))))
        counts[topic] = 30
    db.commit(); db.close()
    manifest = {'phase': 'complete', 'recording_valid': True, 'quality_errors': [],
                'bag_validation': {'valid': True, 'database_files': ['bag_0.db3'], 'database_counts': counts},
                'fk': {'frame_id': 'base_link'}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def test_real_rosbag_decode_uses_common_source_time(bag_episode):
    t, eef, hands, q, meta = load_episode(bag_episode)
    assert t[0] == 0 and 1.3 < t[-1] < 1.5
    assert np.all(np.diff(t) > 0)
    assert eef.shape == (len(t), 2, 7) and hands.shape == (len(t), 2, 20)
    assert q.shape == (len(t), 16)
    assert np.isclose(q[0, 0], .01)  # First actual FK instant after all streams begin.
    assert len(meta['selected_bag_payload_sha256']) == 64


def test_incomplete_episode_rejected(bag_episode):
    p = bag_episode/'manifest.json'; data = json.loads(p.read_text()); data['phase'] = 'recording'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='phase=complete'):
        load_episode(bag_episode)


def test_missing_bag_message_rejected(bag_episode):
    with sqlite3.connect(str(bag_episode/'bag/bag_0.db3')) as db:
        db.execute('DELETE FROM messages WHERE id=(SELECT max(id) FROM messages)')
    with pytest.raises(ValueError, match='count differs'):
        load_episode(bag_episode)


def test_source_time_gap_rejected(bag_episode):
    with sqlite3.connect(str(bag_episode/'bag/bag_0.db3')) as db:
        db.execute('DELETE FROM messages WHERE topic_id=1 AND id BETWEEN 10 AND 14')
    p = bag_episode/'manifest.json'; data = json.loads(p.read_text())
    data['bag_validation']['database_counts']['/joint_states'] -= 5
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='source gap'):
        load_episode(bag_episode)
