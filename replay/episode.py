"""Read completed Pengyang rosbag episodes without importing hardware SDKs."""
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SIDES = ('left', 'right')
ROBOT_NAMES = [p+'_'+s+'_Joint' for s in ('L', 'R') for p in
               ('abad', 'hip', 'yaw', 'knee', 'wrist_yaw', 'wrist_pitch', 'wrist_roll')]
ROBOT_NAMES += ['head_pitch_Joint', 'head_yaw_Joint']


def hand_names(side):
    return [side+'_finger%d_joint%d' % (f, j) for f in range(1, 6) for j in range(1, 5)]


def vector(values, size):
    a = np.asarray(values, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError('Expected %d finite values' % size)
    return a


def ordered(msg, names):
    if len(msg.name) != len(names) or set(msg.name) != set(names):
        raise ValueError('Missing, duplicate or unexpected joint names')
    values = vector(msg.position, len(names))
    by_name = dict(zip(msg.name, values))
    return np.array([by_name[n] for n in names])


def pose(msg):
    if msg.header.frame_id != 'base_link':
        raise ValueError('EEF must be in the recorded base_link frame')
    p, q = msg.pose.position, msg.pose.orientation
    result = vector([p.x, p.y, p.z, q.x, q.y, q.z, q.w], 7)
    norm = np.linalg.norm(result[3:])
    if abs(norm-1) > .001:
        raise ValueError('EEF quaternion is not unit length')
    result[3:] /= norm
    return result


def interpolate(t, values, query, quaternion=False):
    result = np.stack([np.interp(query, t, values[:, j]) for j in range(values.shape[1])], axis=-1)
    if quaternion:
        result[:, 3:] = Slerp(t, Rotation.from_quat(values[:, 3:]))(query).as_quat()
    return result


def load_episode(path, hz=20.):
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import JointState
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent
    manifest_raw = (path/'manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get('phase') != 'complete' or manifest.get('recording_valid') is not True:
        raise ValueError('Episode must have phase=complete and recording_valid=true')
    if not manifest.get('bag_validation', {}).get('valid') or manifest.get('quality_errors'):
        raise ValueError('Recording validation failed')
    if not np.isfinite(hz) or not 5 <= hz <= 100:
        raise ValueError('Sample rate must be within 5..100 Hz')
    topics = {'/joint_states': (JointState, lambda msg: ordered(msg, ROBOT_NAMES))}
    for s in SIDES:
        topics['/record/'+s+'_arm/ee_state_fk'] = (PoseStamped, pose)
        topics['/wuji/'+s+'/command'] = (JointState, lambda msg, s=s: ordered(msg, hand_names(s)))
    rows = {topic: [] for topic in topics}
    digest = hashlib.sha256()
    files = manifest['bag_validation']['database_files']
    if not files or len(set(files)) != len(files):
        raise ValueError('Invalid bag database file list')
    for name in files:
        bag = (path/'bag'/name).resolve()
        if bag.parent != (path/'bag').resolve() or not bag.is_file():
            raise ValueError('Invalid/missing bag database path')
        with sqlite3.connect(bag.as_uri()+'?mode=ro', uri=True) as db:
            table = db.execute('SELECT id,name,type FROM topics').fetchall()
            for topic_id, topic, ros_type in table:
                if topic not in topics:
                    continue
                cls, decode = topics[topic]
                expected_type = 'geometry_msgs/msg/PoseStamped' if cls is PoseStamped else 'sensor_msgs/msg/JointState'
                if ros_type != expected_type:
                    raise ValueError('Wrong ROS type for '+topic)
                for bag_stamp, data in db.execute('SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topic_id,)):
                    msg = deserialize_message(data, cls)
                    stamp = msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
                    if stamp <= 0:
                        raise ValueError('Missing source stamp: '+topic)
                    rows[topic].append((stamp, decode(msg)))
                    digest.update(topic.encode()+str(bag_stamp).encode()+bytes(data))
    stats = {}
    for topic, records in rows.items():
        expected = manifest['bag_validation']['database_counts'].get(topic)
        if len(records) < 2 or len(records) != expected:
            raise ValueError('Bag count differs from manifest for '+topic)
        stamps = np.array([r[0] for r in records], dtype=np.int64)
        gaps = np.diff(stamps)/1e9
        if np.any(gaps <= 0) or np.max(gaps) > .1:
            raise ValueError('Nonmonotonic stamps or >100 ms source gap: '+topic)
        stats[topic] = {'messages': len(records), 'max_gap_s': float(gaps.max())}
    begin = max(r[0][0] for r in rows.values())
    end = min(r[-1][0] for r in rows.values())
    # Keep actual common FK/joint-state instants. Interpolating Cartesian
    # targets near full extension can create a point outside the workspace.
    common = np.array([r[0] for r in rows['/joint_states']], dtype=np.int64)
    for s in SIDES:
        common = np.intersect1d(common, np.array([r[0] for r in rows['/record/'+s+'_arm/ee_state_fk']], dtype=np.int64))
    common = common[(common >= begin) & (common <= end)]
    if len(common) < 2 or (common[-1]-common[0])/1e9 < 1:
        raise ValueError('Insufficient common source-time interval')
    begin = int(common[0])
    available = (common-begin)/1e9
    selected = np.unique(np.r_[np.searchsorted(available, np.arange(0, available[-1], 1/hz)), len(common)-1])
    t = available[selected]
    span = float(t[-1])
    aligned = {}
    for topic, records in rows.items():
        axis = (np.array([r[0] for r in records], dtype=np.int64)-begin)/1e9
        values = np.array([r[1] for r in records])
        aligned[topic] = interpolate(axis, values, t, topic.endswith('ee_state_fk'))
    eef = np.stack([aligned['/record/'+s+'_arm/ee_state_fk'] for s in SIDES], axis=1)
    hands = np.stack([aligned['/wuji/'+s+'/command'] for s in SIDES], axis=1)
    return t, eef, hands, aligned['/joint_states'], {
        'episode': str(path), 'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
        'selected_bag_payload_sha256': digest.hexdigest(), 'source_start_ns': begin,
        'source_duration_s': span, 'topics': stats, 'fk': manifest['fk'],
        'resampling': 'Actual shared FK/joint source instants, downsampled; hand commands interpolated on that axis',
        'arm_source': 'measured FK EEF in base_link; XYZ metres, quaternion XYZW',
        'hand_source': 'recorded /wuji/{side}/command, replayed through target/program',
        'joint_state_use': 'measured joints seed the recorded IK branch; EEF determines each solved target; never joint_command replay'}


def pose_error(actual, target):
    p = np.linalg.norm(actual[..., :3]-target[..., :3], axis=-1)
    shape = actual.shape[:-1]
    a = Rotation.from_quat(actual[..., 3:].reshape(-1, 4))
    b = Rotation.from_quat(target[..., 3:].reshape(-1, 4))
    return p, (a.inv()*b).magnitude().reshape(shape)


def sample(t, poses, hands, source_time):
    value = float(np.clip(source_time, t[0], t[-1]))
    index = int(np.clip(np.searchsorted(t, value, side='right')-1, 0, len(t)-2))
    sl = slice(index, index+2)
    query = np.array([value])
    eef = np.stack([interpolate(t[sl], poses[sl, i], query, True)[0] for i in range(2)])
    h = np.stack([interpolate(t[sl], hands[sl, i], query)[0] for i in range(2)])
    return eef, h
