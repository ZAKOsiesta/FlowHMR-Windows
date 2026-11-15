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


def filter_short_segments(valid, breaks=(), *, first=0, minimum=15):
    """Reject short generation spans, including spans delimited by true jumps."""
    from .timeline import intervals
    valid = np.array(valid, dtype=bool, copy=True)
    discarded = []
    for a, b in intervals(valid):
        bounds = [a, *sorted({int(c) for c in breaks if a < c < b}), b]
        for left, right in zip(bounds, bounds[1:]):
            if right-left < minimum:
                valid[left:right] = False
                discarded.append(dict(source_frames=[first+left, first+right],
                    frames=right-left, reason='below_minimum_generation_length', minimum_frames=minimum))
    return valid, discarded


def continuity_breaks(points, shots, *, K, image_size_hw):
    """Compare only jointly in-frame articulation in a trusted torso basis.

    Image inclusion is evidence of observability, not an occlusion/confidence
    detector. The four torso anchors must also be in frame on both sides; an
    unseen anchor must never rotate otherwise visible joints into a false jump.
    """
    points = np.asarray(points, float)
    K = np.asarray(K, float)
    h, w = image_size_hw
    projected = np.einsum('tij,tkj->tki', K, points)
    uv = projected[..., :2]/np.maximum(points[..., 2:3], 1e-8)
    visible = (np.isfinite(points).all(-1) & np.isfinite(uv).all(-1) & (points[..., 2] > .05)
               & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h))
    torso = (points[:, 5]+points[:, 6])/2
    hip = (points[:, 9]+points[:, 10])/2
    def unit(a):
        return a/np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    up = unit(torso-hip)
    right = points[:, 6]-points[:, 5]
    right = unit(right-np.sum(right*up, -1, keepdims=True)*up)
    forward = unit(np.cross(right, up))
    basis = np.stack([right, up, forward], -1)
    joints = np.array([5,6,7,8,62,41])
    local = np.einsum('tji,tkj->tki', basis, points[:, joints]-torso[:, None])
    delta = np.linalg.norm(np.diff(local, axis=0), axis=-1)
    anchors = (visible[:, [5,6,9,10]].all(-1) & np.isfinite(basis).all((1,2))
               & (np.linalg.det(basis) > .99))
    trusted = anchors[1:] & anchors[:-1]
    common = visible[1:, joints] & visible[:-1, joints]
    within = np.ones(len(delta), bool)
    for shot in shots[1:]:
        within[shot['start']-1] = False
    report = []
    for shot in shots[1:]:
        edge = shot['start']-1
        use = common[edge]
        row = dict(frame=shot['start'], jump_m=None, threshold_m=None, broken=False,
                   visible_joints=joints[use].tolist(), baseline_pairs=0, reason='insufficient_visible_joints')
        if trusted[edge] and use.sum() >= 4:
            # Compare the same joint subset at the cut and within shots, so a
            # changing visibility set cannot lower the baseline artificially.
            baseline = within & trusted & common[:, use].all(-1)
            row['baseline_pairs'] = int(baseline.sum())
            if baseline.any():
                jump = float(delta[edge, use].mean())
                threshold = 3*max(float(np.percentile(delta[baseline][:, use].mean(-1), 99)), 1e-6)
                row.update(jump_m=jump, threshold_m=threshold, broken=jump > threshold,
                           reason='visible_pose_jump' if jump > threshold else 'continuous_visible_pose')
            else:
                row['reason'] = 'insufficient_visible_baseline'
        report.append(row)
    return report
