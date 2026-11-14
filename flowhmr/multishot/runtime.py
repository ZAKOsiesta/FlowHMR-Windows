"""Lazy I/O adapter: existing VGGT/SAM lifecycle, frozen joints, atomic output.

No generation pipeline is constructed here. The caller lends its SAM context
and inference device; CPU/XPU fallback policy remains with the existing app.
"""
import hashlib
import json
from pathlib import Path
from zipfile import BadZipFile
import numpy as np

from .recovery import recover_camera
from .timeline import validity


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.part')
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Unsupported camera report value: {type(value).__name__}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar), encoding='utf8')
    tmp.replace(path)


def save_npz(path, **values):
    path = Path(path)
    tmp = path.with_suffix('.npz.part')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **values)
    tmp.replace(path)


def camera_directory(directory, job):
    root = Path(directory)
    return root / 'scene' if job.get('person_batch') == job['id'] else root


def _raw_camera(directory, job, first, count, hw, video, device, check, notify, release_cuda):
    from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera_isolated, make_static_camera
    mode = 'static' if job['params'].get('static_camera') else 'vggt'
    source = directory / f'input/input_{mode}_camera.npz'
    try:
        with np.load(source, allow_pickle=False) as z:
            a, b = map(int, z['frame_range'])
            if a <= first <= first+count-1 <= b and str(z['k_space']) == 'original_frame':
                s = slice(first-a, first-a+count)
                rt, K = z['camera_RT'][s], z['K'][s]
                if rt.shape == (count, 4, 4) and K.shape == (count, 3, 3) and np.isfinite(rt).all() and np.isfinite(K).all():
                    notify('已复用任务的原始相机观测', 18)
                    return rt, K, 'task_cache'
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    output = directory / 'camera' / f'raw_{first}_{first+count-1}.npz'
    if mode == 'static':
        result = make_static_camera(hw, frame_range=(first, first+count-1), output_npz=str(output))
    else:
        result = extract_vggt_camera_isolated(str(video), output_npz=str(output),
            frame_range=(first, first+count-1), expected_len=count, device=device,
            frame_interval=job['params'].get('vggt_interval', 0), cancel_check=check,
            on_progress=lambda *args: notify(str(args[-1]), 15), release_cuda=release_cuda)
    check()
    return result.camera_RT, result.K, mode


def _observations(folder, video, boxes, K, valid, first, acquire_sam, identity, check, notify):
    import cv2
    target, proof = folder / 'observations.npz', folder / 'observations.json'
    try:
        meta = json.loads(proof.read_text(encoding='utf8'))
        if meta['identity'] == identity and meta['sha256'] == digest(target):
            with np.load(target, allow_pickle=False) as z:
                points = z['points']
                if points.shape == (len(valid), 70, 3) and np.isfinite(points[valid]).all():
                    notify('已复用人物观测，动作保持不变', 45)
                    return points, True
    except (OSError, ValueError, KeyError, BadZipFile):
        pass
    points = np.full((len(valid), 70, 3), np.nan)
    selected = np.flatnonzero(valid)
    if not np.isfinite(boxes[valid]).all():
        raise ValueError('旧任务缺少有效的人物检测框，无法补算相机观测。')
    cap = cv2.VideoCapture(str(video))
    previous = -2
    try:
        check()
        with acquire_sam() as extractor:
            for i, frame in enumerate(selected):
                check()
                if frame != previous+1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first+frame))
                ok, image = cap.read()
                if not ok:
                    raise ValueError(f'无法读取原视频第 {first+frame+1} 帧。')
                points[frame] = extractor.scene_keypoints(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    boxes[frame:frame+1], K[frame], list(range(70)))[0]
                previous = frame
                if i % 5 == 0 or i+1 == len(selected):
                    notify(f'补算人物观测 · {i+1}/{len(selected)} 帧', 20 + 25*(i+1)/len(selected))
    finally:
        cap.release()
    check()
    save_npz(target, points=points)
    save_json(proof, dict(identity=identity, sha256=digest(target), generation_calls=0))
    return points, False


def build_camera(directory, job, inspection, acquire_sam, device, *, check=lambda: None,
                 notify=lambda message, progress: None, release_cuda=None):
    from contextlib import nullcontext
    from .gravity import image_gravity
    from .shared import build_shared_camera
    directory = Path(directory)
    if job.get('person_batch') == job['id']:
        return build_shared_camera(directory / 'scene', check=check, notify=notify)
    folder = directory / 'camera'
    folder.mkdir(exist_ok=True)
    motion_hash = digest(directory / 'motion.npz')
    check()
    notify('读取已完成的人物动作', 2)
    path, info = inspection.load(directory, body_reference=job.get('body_reference') or 'neutral')
    joints = np.fromfile(path, dtype='<f4').reshape(info['frames'], 52, 3)
    with np.load(directory / 'motion.npz', allow_pickle=False) as z:
        valid = validity(z.get('valid_frames'), len(joints))
    first = int((job.get('track_range') or [0])[0])
    # Explicit, source-identified archived input contract. It contains raw camera,
    # SAM evidence and reviewed gravity, never fitted/gated output cameras.
    archive = folder / 'inputs.npz'
    meta_path = folder / 'inputs.json'
    linked = directory / 'input/linked/camera_inputs.npz'
    linked_meta = linked.with_suffix('.json')
    if linked.exists() and linked_meta.exists():
        meta = json.loads(linked_meta.read_text(encoding='utf8'))
        if meta['motion_sha256'] != motion_hash or meta['sha256'] != digest(linked):
            raise ValueError('跨镜头观测与当前动作不一致，请重跑任务。')
        with np.load(linked, allow_pickle=False) as z:
            if not np.array_equal(z['valid'], valid):
                raise ValueError('跨镜头观测有效区间不一致。')
            rt, K, points, hw = (z[k] for k in ('RT', 'K', 'points', 'image_size_hw'))
            first = int(z['track_start'])
        shots = meta['shots']
        gravity = {} if job['params'].get('static_camera') else image_gravity(
            directory/'input/input_30fps.mp4', rt, K, shots, first, folder, device, check, notify)
        source = dict(raw_camera='linked_shot_cache', observations_reused=True, generation_calls=0)
    elif archive.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf8'))
        if meta['motion_sha256'] != motion_hash or meta['sha256'] != digest(archive):
            raise ValueError('相机输入档案与冻结人物不一致。')
        with np.load(archive, allow_pickle=False) as z:
            if not np.array_equal(z['joints'][valid], joints[valid]) or not np.array_equal(z['valid'], valid):
                raise ValueError('相机输入档案的人物关节或有效范围不一致。')
            rt, K, points, hw = (z[k] for k in ('RT', 'K', 'points', 'image_size_hw'))
        shots = meta['shots']
        gravity = {int(k): v for k, v in meta['gravity'].items()}
        source = dict(raw_camera='archived_inputs', observations_reused=True, archive_sha256=meta['sha256'])
    else:
        import cv2
        video = directory / 'input/input_30fps.mp4'
        cap = cv2.VideoCapture(str(video))
        hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if min(hw) < 1 or abs(fps-30) > .01:
            raise ValueError('此旧任务缺少可读取的 30fps 原视频。')
        boxes_path = directory / 'input/input_bbox.npz'
        if not boxes_path.is_file():
            raise ValueError('此旧任务缺少人物检测资料，请恢复原任务的检测缓存后重试；已有动作仍可导出。')
        with np.load(boxes_path, allow_pickle=False) as z:
            first, last = map(int, z['start_end'])
            boxes = z['bbox']
        if last-first+1 != len(joints) or len(boxes) != len(joints) or last >= frames:
            raise ValueError('人物动作与原视频检测范围不一致。')
        rt, K, mode = _raw_camera(directory, job, first, len(joints), hw, video, device, check,
                                  notify, release_cuda or nullcontext)
        identity = dict(video_sha256=digest(video), boxes=array_digest(boxes), K=array_digest(K),
                        valid=array_digest(valid), first=first, method='upstream_scene_keypoints_body_70_v1')
        sidecar = directory / ('input/input_sam3d_feat' + ('_static' if job['params'].get('static_camera') else '') + '.pt.observations.npz')
        points, reused = None, False
        try:
            with np.load(sidecar, allow_pickle=False) as z:
                if (np.array_equal(z['K'], K) and np.array_equal(z['boxes'], boxes)
                    and np.array_equal(z['source_frames'], np.arange(first,first+len(joints)))
                    and z['points'].shape == (len(joints),70,3) and np.isfinite(z['points']).all()):
                    points, reused = z['points'], True
        except (OSError, ValueError, KeyError, BadZipFile):
            pass
        if points is None:
            points, reused = _observations(folder, video, boxes, K, valid, first, acquire_sam, identity, check, notify)
        shots = [dict(id=0, start=0, end=len(joints))]
        gravity = {} if job['params'].get('static_camera') else image_gravity(
            video, rt, K, shots, first, folder, device, check, notify)
        source = dict(raw_camera=mode, observations_reused=reused, observations_identity=identity)
    camera, report = recover_camera(rt, K, joints, points, valid, shots, hw,
        gravity=gravity, track_start=first, check=check, notify=notify)
    check()
    if digest(directory / 'motion.npz') != motion_hash:
        raise ValueError('相机计算期间人物动作已变化，请重新计算。')
    report.update(source=source, motion_sha256=motion_hash, body_reference=job.get('body_reference') or 'neutral')
    save_json(folder / 'report.json', report)
    check()
    save_npz(folder / 'camera.npz', **camera.arrays())
    return report
