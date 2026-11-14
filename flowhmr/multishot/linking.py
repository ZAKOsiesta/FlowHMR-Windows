"""Select one observed person per shot; never infer an identity across a cut."""
import numpy as np


def shot_candidates(result, shots):
    rows = []
    for shot in shots:
        choices = []
        minimum = min(15, (shot['end']-shot['start'])//2+1)
        for i, track in enumerate(result.tracks):
            use = (track.frames >= shot['start']) & (track.frames < shot['end'])
            if not use.any():
                continue
            boxes = track.bboxes[use]
            area = float(np.mean(np.prod(boxes[:, 2:]-boxes[:, :2], axis=1)))
            choices.append(dict(track_index=i, count=int(use.sum()), area=area))
        # Tiny fragments do not force a person-selection dialogue. If all are
        # short, retain the best observed track rather than inventing a box.
        long = [c for c in choices if c['count'] >= minimum]
        rows.append(sorted(long or choices, key=lambda c: (c['area'], c['count']), reverse=True))
    return rows


def link_tracks(result, shots, selected=None):
    first, stop = shots[0]['start'], shots[-1]['end']
    boxes = np.zeros((stop-first, 4), np.float32)
    valid = np.zeros(stop-first, bool)
    report = []
    choices = shot_candidates(result, shots)
    for shot, candidates in zip(shots, choices):
        preferred = next((c for c in candidates if c['track_index'] == selected), None)
        chosen = preferred or (candidates[0] if candidates else None)
        if chosen:
            track = result.tracks[chosen['track_index']]
            use = (track.frames >= shot['start']) & (track.frames < shot['end'])
            frames = track.frames[use].astype(int)
            order = np.argsort(frames)
            frames, observed = frames[order], track.bboxes[use][order]
            # Match upstream interpolation of one track, confined to this shot.
            # ByteTrack activation can miss the first two frames after a cut;
            # use this shot's nearest box only at that short boundary, never a
            # box from the previous camera. Longer absences stay invalid.
            a,b = int(frames[0]),int(frames[-1])+1
            if a-shot['start'] <= 2:a=shot['start']
            if shot['end']-b <= 2:b=shot['end']
            timeline = np.arange(a,b)
            for j in range(4):
                boxes[a-first:b-first,j] = np.interp(timeline,frames,observed[:,j])
            valid[a-first:b-first] = True
            for left,right in zip(frames[:-1],frames[1:]):
                if right-left>3:
                    valid[left+1-first:right-first]=False
        report.append(dict(**shot, track_index=chosen['track_index'] if chosen else None,
                           candidates=len(candidates), user_selected=preferred is not None))
    valid &= np.isfinite(boxes).all(1) & (boxes[:, 2:] > boxes[:, :2]).all(1)
    if not valid.any():
        raise ValueError('处理区间里没有检测到人物，请调整时间轴的起止帧。')
    return boxes, valid, report


def continuity_breaks(points, shots):
    """Compare upper-body pose in its own torso basis; ignore camera cut angle."""
    points = np.asarray(points, float)
    torso = (points[:, 5]+points[:, 6])/2
    hip = (points[:, 9]+points[:, 10])/2
    def unit(a):
        return a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    up = unit(torso-hip)
    right = points[:, 6]-points[:, 5]
    right = unit(right-np.sum(right*up, -1, keepdims=True)*up)
    forward = unit(np.cross(right, up))
    basis = np.stack([right, up, forward], -1)
    local = np.einsum('tji,tkj->tki', basis, points[:, [5,6,7,8,62,41]]-torso[:, None])
    delta = np.linalg.norm(np.diff(local, axis=0), axis=-1).mean(-1)
    within = np.ones(len(delta), bool)
    for shot in shots[1:]:
        within[shot['start']-1] = False
    threshold = 3*max(float(np.percentile(delta[within], 99)) if within.any() else 0, 1e-6)
    return [dict(frame=s['start'], jump_m=float(delta[s['start']-1]), threshold_m=threshold,
                 broken=bool(delta[s['start']-1] > threshold)) for s in shots[1:]]
