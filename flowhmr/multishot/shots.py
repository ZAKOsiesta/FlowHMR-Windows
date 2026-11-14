"""Automatic cuts on the canonical 30 fps source, using half-open ranges."""
import json
from pathlib import Path


def detect_shots(video, first, stop, cache, check=lambda: None):
    from .runtime import save_json
    video, cache = Path(video), Path(cache)
    stat = video.stat()
    identity = dict(size=stat.st_size, modified=stat.st_mtime_ns, first=first, stop=stop,
                    detector='AdaptiveDetector-default-v2-complete-read')
    try:
        data = json.loads(cache.read_text(encoding='utf8'))
        if data['identity'] == identity:
            return data['shots']
    except (OSError, ValueError, KeyError):
        pass
    from scenedetect import open_video, SceneManager, AdaptiveDetector
    stream = open_video(str(video))
    # Cancellation is checked during decoding, including long uncut footage.
    read = stream.read
    def checked_read(*args, **kwargs):
        check()
        return read(*args, **kwargs)
    stream.read = checked_read
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    stream.seek(first)
    try:
        manager.detect_scenes(stream, end_time=stop, show_progress=False)
        if stream.frame_number < stop:
            raise ValueError('视频未能完整读取所选区间，请释放内存后重试；未保存不完整的镜头检测。')
        scenes = manager.get_scene_list(start_in_scene=True)
    finally:
        capture = getattr(stream, '_cap', None)
        if capture is not None:
            capture.release()
    cuts = sorted({a.get_frames() for a, _ in scenes if first < a.get_frames() < stop})
    bounds = [first, *cuts, stop]
    shots = [dict(id=i, start=a, end=b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]
    check()
    save_json(cache, dict(identity=identity, shots=shots))
    return shots


def clip_shots(shots, first, stop):
    return [dict(id=s['id'], start=max(first, s['start'])-first,
                 end=min(stop, s['end'])-first, source_start=max(first, s['start']),
                 source_end=min(stop, s['end'])) for s in shots
            if max(first, s['start']) < min(stop, s['end'])]
