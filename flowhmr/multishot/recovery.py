"""Camera-only recovery. The boundary accepts joints, never pose/model parameters.

Image observations and gravity are supplied by the runtime adapter. Each valid
interval has one fixed similarity; the reviewed V3 shared fast-motion gate is
then applied. A failed interval does not invalidate other people/intervals.
"""
import numpy as np

from .alignment import SM, MH, body_facing_rotation, fit_shot
from .camera import Camera
from .gate import gate_camera
from .similarity import align_camera
from .timeline import intervals

LEGACY_SM = [16, 17, 1, 2, 4, 5, 7, 8]
LEGACY_MH = [5, 6, 9, 10, 11, 12, 13, 14]


class CameraCancelled(Exception):
    pass


def recover_camera(rt, K, joints, points, valid, shots, image_size_hw, *,
                   gravity=None, track_start=0, check=lambda: None,
                   notify=lambda message, progress: None):
    """Return camera and diagnostic report without mutating any caller array.

Shot bounds are track-relative, half-open. Approved archived gravity is an
immutable input of a reviewed recipe; new high-dispersion measurements fall
back to the long-window body orientation and ground constraint.
"""
    rt, K, joints, points = map(np.asarray, (rt, K, joints, points))
    valid = np.asarray(valid, dtype=bool)
    n = len(joints)
    if (joints.shape != (n, 52, 3) or points.shape != (n, 70, 3) or valid.shape != (n,)
            or rt.shape != (n, 4, 4) or K.shape != (n, 3, 3)):
        raise ValueError('相机输入与人物时间线不一致。')
    if not np.isfinite(joints[valid]).all():
        raise ValueError('冻结人物关节含无效值。')
    # Missing observations are camera gaps; the motion's validity is untouched.
    usable = valid & np.isfinite(points).all((1, 2))
    rows, segments, taken = [], [], np.zeros(n, bool)
    ids = np.full(n, -1, np.int32)
    for shot in shots:
        a, b, sid = int(shot['start']), int(shot['end']), int(shot['id'])
        if not 0 <= a < b <= n or taken[a:b].any():
            raise ValueError('相机镜头范围重叠或越界。')
        taken[a:b] = True
        ids[a:b] = sid
        for x, y in intervals(usable[a:b]):
            segments.append((a+x, a+y, sid))
        if not usable[a:b].any():
            rows.append(dict(shot=sid, source_frames=[a, b], labels=['no_observation']))
    if np.any(valid & ~taken):
        raise ValueError('相机镜头范围未覆盖有效人物帧。')
    # Reviewed V3 stores fixed similarities at float64 precision.
    aligned, intrinsics = rt.astype(float), K.copy()
    scales = {}
    for i, (a, b, sid) in enumerate(segments):
        check()
        notify(f'摄像机整体尺度 · {i+1}/{len(segments)}', 55 + 10*i/max(1, len(segments)))
        sample = np.unique(np.linspace(0, b-a-1, min(80, b-a), dtype=int))
        try:
            legacy = align_camera(rt[a:b], K[a:b], sample, points[a+sample][:, LEGACY_MH],
                                  joints[a+sample][:, LEGACY_SM], check=check)
            scales[a, b] = float(legacy['world_scale'])
        except ValueError:
            # Static/short/weakly observed shots still have body-facing evidence.
            pass
    reliable = [s for s in scales.values() if .1 < s < 1000]
    prior = float(np.median(reliable)) if reliable else 1.
    initials = {}
    for a,b,_ in segments:
        check()
        try:
            initials[a,b] = body_facing_rotation(rt[a:b], points[a:b][:, MH], joints[a:b][:, SM])
        except (ValueError, np.linalg.LinAlgError):
            pass
    from .leveling import body_gravity
    fallback_up, leveling_report = body_gravity(rt, points, segments, initials)
    fitted, camera_valid = [], np.zeros(n, bool)
    for i, (a, b, sid) in enumerate(segments):
        check()
        notify(f'对齐原片机位 · {i+1}/{len(segments)}', 65 + 25*i/max(1, len(segments)))
        try:
            if (a,b) not in initials:
                raise ValueError('人物朝向观测不可用。')
            initial = initials[a,b]
            g = (gravity or {}).get(sid, {})
            up = np.asarray(g.get('mean_world_up', [0, 0, 0]), dtype=float)
            gravity_ok = (up.shape == (3,) and np.isfinite(up).all() and np.linalg.norm(up) > .5
                          and (float(g.get('dispersion_max_deg', 0)) <= 8 or g.get('reviewed_input', False)))
            if not gravity_ok:
                up = fallback_up[a,b]
            rawscale = scales.get((a, b), prior)
            scale_prior = rawscale if .1 < rawscale < 1000 else prior
            result = fit_shot(rt[a:b], K[a:b], points[a:b][:, MH], joints[a:b][:, SM],
                              up, initial, scale_prior, image_size_hw=image_size_hw,
                              check=check, ground_min=None if gravity_ok else 0.)
            report = result['report']
            if report['visible_min_depth_m'] <= .05:
                raise ValueError('对齐后人物位于摄像机后方。')
            aligned[a:b] = result['RT']
            camera_valid[a:b] = True
            report.update(shot=sid, source_frames=[a, b], labels=[],
                          gravity_source='image' if gravity_ok else 'body_long_window_and_ground',
                          gravity_dispersion_deg=g.get('dispersion_max_deg'),
                          reviewed_gravity=bool(g.get('reviewed_input', False)))
            if not gravity_ok:
                report['labels'].append('gravity_fallback')
            if b-a < 12:
                report['labels'].append('short_shot')
            fitted.append(report)
        except (ValueError, np.linalg.LinAlgError) as exc:
            rows.append(dict(shot=sid, source_frames=[a, b], labels=['failed'], error=str(exc)))
    check()
    if not camera_valid.any():
        raise ValueError('未得到有效摄像机；人物动作已保留。请检查人物可见范围与素材。')
    notify('稳定摄像机并整理逐镜头说明', 94)
    (final_rt, final_K), _, gate = gate_camera(aligned, intrinsics, joints.astype(float), points.astype(float), fitted,
                                               camera_valid, image_size_hw)
    decisions = {tuple(row['source_frames']): row for row in gate['shots']}
    for row in fitted:
        decision = decisions.get(tuple(row['source_frames']), {})
        labels = row['labels'] + decision.get('labels', [])
        rows.append({**row, **decision, 'labels': labels})
    check()
    camera = Camera(final_rt, final_K, camera_valid, ids, np.asarray(image_size_hw), track_start)
    return camera, dict(version=1, recipe='reviewed_v3_with_product_fallbacks',
                        track_start=int(track_start), fps=30, interval_convention='half_open_track_relative',
                        shots=sorted(rows, key=lambda r: r['source_frames'][0]), gate=gate, leveling=leveling_report,
                        generation_calls=0, motion_changed=False,
                        physical_calibration_guaranteed=False)
