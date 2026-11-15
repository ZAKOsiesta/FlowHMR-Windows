"""Track-relative validity; all internal ranges are half open."""
import numpy as np


def preview_shot_ids(meta, job):
    """Expose source-indexed cut boundaries without changing cached geometry."""
    tracks = (job.get('shot_linking_report') or {}).get('tracks', [])
    if not tracks:
        return None
    first, count = int(meta.get('track_start', 0)), int(meta['num_frames'])
    ids = np.full(count, -1, dtype=np.int32)
    for shot in tracks:
        a, b = max(0, int(shot['start'])-first), min(count, int(shot['end'])-first)
        if a < b:
            ids[a:b] = int(shot['id'])
    return ids.tolist()


def validity(values, frames):
    if values is None:
        return np.ones(frames, dtype=bool)
    values = np.asarray(values)
    if values.shape != (frames,) or values.dtype.kind != 'b':
        raise ValueError('valid_frames must be a boolean value for every motion frame')
    return values


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def held_indices(mask):
    mask = np.asarray(mask, dtype=bool)
    found = np.flatnonzero(mask)
    if not len(found):
        raise ValueError('Motion has no valid frames')
    # Before entry, hold the first valid pose; later gaps hold the preceding pose.
    return np.maximum.accumulate(np.where(mask, np.arange(len(mask)), found[0]))


def display_keys(mask, start=0, scene_end=None):
    mask = np.asarray(mask, dtype=bool)
    keys = {0: False, start - 1: False} if start else {}
    keys[start] = bool(mask[0])
    for i in np.flatnonzero(mask[1:] != mask[:-1]) + 1:
        keys[start + int(i) - 1] = bool(mask[i - 1])
        keys[start + int(i)] = bool(mask[i])
    end = start + len(mask) - 1
    if scene_end is not None and end < scene_end:
        keys[end] = bool(mask[-1])
        keys[end + 1] = False
    return dict(sorted(keys.items()))


def gap_report(motion, start=0):
    mask = validity(motion.get('valid_frames'), len(motion['trans']))
    return {'fps': 30, 'track_start': int(start), 'interval_convention': 'half_open_source_frames',
            'gaps': [[a + start, b + start] for a, b in intervals(~mask)],
            'gap_pose': 'previous_valid; before entry: first_valid',
            'visibility': 'VMD: display keys; FBX/VRMA: held pose, no portable visibility track'}
