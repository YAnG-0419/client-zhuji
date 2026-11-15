"""Refine conservative swept bounds while retaining the 5 mm clearance gate."""
import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M


def certify_path(model, qs, refine=None):
    qs = np.asarray(qs, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != 16 or not len(qs) or not np.isfinite(qs).all():
        raise ValueError('Expected a finite nonempty (N,16) joint path')
    try:
        return model.check_path(qs, clearance=MODEL_CLEARANCE_M)
    except ValueError as coarse_error:
        if len(qs) == 1:
            raise
        if refine is None:
            from replay_movej_path import check_segment
            refine = check_segment
        minimum, refined = float('inf'), []
        for index, (start, goal) in enumerate(zip(qs[:-1], qs[1:]), 1):
            try:
                record = model.check_path(np.array([start, goal]), clearance=MODEL_CLEARANCE_M)
                bound = record['conservative_min_clearance_m']
            except ValueError:
                try:
                    bound = refine(model, start, goal)
                except ValueError as exc:
                    raise ValueError('Adaptive clearance failed at waypoint segment %d: %s' % (index, exc)) from exc
                if not np.isfinite(bound) or bound < MODEL_CLEARANCE_M:
                    raise ValueError('Adaptive clearance below 5 mm at waypoint segment '+str(index))
                refined.append(index)
            minimum = min(minimum, bound)
        return {'conservative_min_clearance_m': minimum, 'required_clearance_m': MODEL_CLEARANCE_M,
                'segments': len(qs)-1, 'refined_segments': refined,
                'method': 'joint-linear continuous swept bounds with adaptive subdivision',
                'initial_coarse_rejection': str(coarse_error)}
