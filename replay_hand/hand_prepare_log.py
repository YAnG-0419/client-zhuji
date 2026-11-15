"""Record per-joint hand preparation feedback without changing runtime guards."""
import json
import time

import numpy as np


def prepare_hands(session, duration, stationary_q, out, result, tolerance):
    original_tick = session.tick
    began, last_log = time.monotonic(), -float('inf')
    report = dict(duration_s=duration, first_frame_tolerance_rad=tolerance,
                  runtime_tracking_limit_rad=.20)
    result['hand_preparation'] = report
    with (out/'hand_preparation.jsonl').open('w') as log:
        def tick(*args, **kwargs):
            nonlocal last_log
            feedback = original_tick(*args, **kwargs)
            measured = np.asarray(feedback[3])
            error = abs(measured-session.hand_data[0])
            side, joint = np.unravel_index(int(np.argmax(error)), error.shape)
            row = dict(elapsed_s=time.monotonic()-began, measured_q20=measured.tolist(),
                       target_q20=session.hand_data[0].tolist(),
                       commanded_q20=session.last_hand.tolist(),
                       per_side_max_error_rad=np.max(error, axis=1).tolist(),
                       worst_joint='%s_finger%d_joint%d' %
                       (('left', 'right')[side], joint//4+1, joint%4+1),
                       max_error_rad=float(error[side, joint]))
            report['last_feedback'] = row
            if time.monotonic()-last_log >= .1:
                log.write(json.dumps(row)+'\n'); log.flush()
                last_log = time.monotonic()
            return feedback
        session.tick = tick
        try:
            session.approach_hands(duration, stationary_q)
            report['completed'] = True
        except BaseException as exc:
            report['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            session.tick = original_tick
            log.write(json.dumps(dict(event='end', **report))+'\n')
