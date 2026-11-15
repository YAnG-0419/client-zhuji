"""Validated IK reuse for full replay; never caches collision/live acceptance."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np

SCHEMA = 'full_replay_ik_cache_v1'


def array_hash(*arrays):
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def signature(root, model, t, eef, hands, meta, hz):
    files = [root / name for name in (
        'replay_hand/ik_cache.py', 'replay_hand/planner_cross.py',
        'replay_hand/episode_cross.py', 'replay_hand/anchor_cross.py',
        'replay_hand/replay_hand.py', 'replay_hand/full_sync.py',
        'replay/replay.py', 'replay/transport.py', 'replay/replay_thresholds.py',
        'tools/robot_home.py', 'poses/home.json')]
    for cls in type(model).__mro__:
        if cls is not object:
            files.append(Path(inspect.getfile(cls)).resolve())
    return dict(schema=SCHEMA, episode_sha256=meta['episode_sha256'], hz=float(hz),
                fk=meta['fk'], placement_policy=meta['placement_policy'],
                arrays_sha256=array_hash(t, eef, hands),
                bounds_sha256=array_hash(model.lower, model.upper),
                model_dependencies=model.dependencies,
                code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(set(files))})


def cache_path(root, sig):
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()
    return root / 'replay_hand/cache/ik' / (key + '.npz')


def validate_seed(model, t, eef, qs, position_tol, rotation_tol):
    from episode import pose_error
    t, eef, qs = map(np.asarray, (t, eef, qs))
    if (len(t) < 2 or t.ndim != 1 or eef.shape != (len(t), 2, 7)
            or qs.shape != (len(t), 16)
            or not all(np.isfinite(a).all() for a in (t, eef, qs))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid cached IK arrays')
    if np.any(qs < model.lower + 1e-5) or np.any(qs > model.upper - 1e-5):
        raise ValueError('Cached IK outside executable joint bounds')
    step = float(np.max(abs(np.diff(qs[:, :14], axis=0))))
    if step > .20 or not np.array_equal(qs[:, 14:], np.broadcast_to(qs[0, 14:], qs[:, 14:].shape)):
        raise ValueError('Cached IK discontinuity or moving head')
    max_p = max_r = 0.
    for i, (q, target) in enumerate(zip(qs, eef)):
        model.set_q(q)
        p, r = pose_error(model.poses(), target)
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError('Nonfinite cached IK FK')
        if max(p) > position_tol or max(r) > rotation_tol:
            raise ValueError('Cached IK residual at frame %d' % i)
        if i == 0 and (max(p) > .001 or max(r) > .0035):
            raise ValueError('Cached first-frame FK consistency')
        max_p, max_r = max(max_p, float(max(p))), max(max_r, float(max(r)))
    return dict(frames=len(t), max_position_error_m=max_p,
                max_rotation_error_rad=max_r, max_joint_step_rad=step,
                head_policy='Fixed offline head; live head is replaced and recertified')


def write_cache(path, sig, t, eef, qs, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(signature=sig, arrays_sha256=array_hash(t, eef, qs),
                    provenance=provenance)
    fd, temporary = tempfile.mkstemp(prefix='.ik-', suffix='.npz', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, t=t, eef=eef, q16=qs,
                                manifest_json=json.dumps(manifest, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cache(path, sig, t, eef, model, position_tol, rotation_tol):
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(str(z['manifest_json']))
        ct, ce, qs = z['t'], z['eef'], z['q16'].copy()
    if (manifest['signature'] != sig or not np.array_equal(ct, t)
            or not np.array_equal(ce, eef)
            or manifest['arrays_sha256'] != array_hash(ct, ce, qs)):
        raise ValueError('IK cache signature/time/target/digest mismatch')
    report = validate_seed(model, t, eef, qs, position_tol, rotation_tol)
    report.update(selected_first_seed='verified_cache', cache_provenance=manifest['provenance'])
    return qs, report


def load_full_episode(path, hz, out):
    import replay as base
    from episode_cross import load_cross_episode
    from isolated_cli import ROOT
    from replay_hand import cross_loader
    from transport import replay_backend
    began = time.monotonic()
    t, eef, hands, meta = load_cross_episode(path, hz)
    if meta.get('placement_policy') != 'preserve_common_frame_v1':
        return cross_loader(path, hz, out)
    _, Model, _ = replay_backend()
    model = Model()
    sig = signature(ROOT, model, t, eef, hands, meta, hz)
    cached = cache_path(ROOT, sig)
    try:
        qs, report = read_cache(cached, sig, t, eef, model,
                                base.IK_POSITION_TOL, base.IK_ROTATION_TOL)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        print('IK CACHE MISS: %s; computing continuous IK' % type(exc).__name__, flush=True)
        t, eef, hands, qs, meta = cross_loader(path, hz, out)
        meta['ik_cache'] = dict(hit=False, reason=str(exc), path=str(cached),
                                load_seconds=time.monotonic()-began)
        return t, eef, hands, qs, meta
    meta['cross_anchor'] = dict(policy='preserve_common_frame_v1',
                               absolute_camera_pose_preserved=True,
                               bimanual_relative_pose_preserved=True,
                               per_arm_reanchoring_applied=False)
    meta['cross_seed'] = report
    meta['ik_cache'] = dict(hit=True, path=str(cached), load_seconds=time.monotonic()-began,
                            policy='Joint solutions only; all path/live checks run again')
    np.savez_compressed(out/'cross_seed.npz', t=t, q16=qs)
    (out/'cross_seed.json').write_text(json.dumps(report, indent=2)+'\n')
    print('VERIFIED IK CACHE %d frames in %.2fs' % (len(t), time.monotonic()-began), flush=True)
    return t, eef, hands, qs, meta


def save_validated_plan(root, model, t, eef, hands, qs, meta, hz, out):
    if meta.get('placement_policy') != 'preserve_common_frame_v1':
        return
    import replay as base
    validate_seed(model, t, eef, qs, base.IK_POSITION_TOL, base.IK_ROTATION_TOL)
    sig = signature(root, model, t, eef, hands, meta, hz)
    write_cache(cache_path(root, sig), sig, t, eef, qs,
                dict(run=str(out), acceptance='Full offline path checks passed'))


def main():
    """Import an earlier completed offline plan after current-model FK checks."""
    import argparse
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root/'replay'), str(root/'tools')]
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--import-run', type=Path, required=True)
    args = p.parse_args()
    import replay as base
    from episode_cross import load_cross_episode
    from transport import replay_backend
    t, eef, hands, meta = load_cross_episode(args.episode, 50.)
    previous = json.loads((args.import_run/'result.json').read_text())
    _, Model, _ = replay_backend()
    model = Model()
    if (not previous.get('completed') or not previous.get('offline_validated')
            or previous.get('execute_requested')
            or meta.get('placement_policy') != 'preserve_common_frame_v1'
            or previous['source']['episode_sha256'] != meta['episode_sha256']
            or previous['source']['fk'] != meta['fk']
            or previous['plan']['model_dependencies'] != model.dependencies):
        raise ValueError('Import requires a matching completed offline plan/model')
    with np.load(args.import_run/'plan.npz', allow_pickle=False) as z:
        if not all(np.array_equal(z[k], v) for k, v in [('t', t), ('eef', eef), ('hands', hands)]):
            raise ValueError('Import plan arrays differ from current episode')
        qs = z['q16'].copy()
    report = validate_seed(model, t, eef, qs, base.IK_POSITION_TOL, base.IK_ROTATION_TOL)
    sig = signature(root, model, t, eef, hands, meta, 50.)
    dest = cache_path(root, sig)
    write_cache(dest, sig, t, eef, qs, dict(imported_run=str(args.import_run),
                acceptance='Previous offline plan; FK/bounds/continuity rechecked; collision checks still required'))
    print(json.dumps(dict(cache=str(dest), validation=report), indent=2))


if __name__ == '__main__':
    main()
