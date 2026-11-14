"""Optional GeoCalib measurements. Missing/unsupported models use body fallback."""
import json
import sys
from pathlib import Path
import numpy as np


def image_gravity(video, rt, K, shots, first, folder, device, check, notify):
    from .runtime import digest, array_digest, save_json
    from .recovery import CameraCancelled
    root = Path(__file__).resolve().parents[2]
    weights = root / 'models/geocalib/pinhole.tar'
    source = root / 'third_party/GeoCalib'
    cache = folder / 'gravity.json'
    identity = dict(video=digest(video), RT=array_digest(rt), K=array_digest(K), first=first,
                    shots=shots, weights=digest(weights) if weights.is_file() else None)
    try:
        old = json.loads(cache.read_text(encoding='utf8'))
        if old['identity'] == identity:
            return {int(k): v for k, v in old['shots'].items()}
    except (OSError, ValueError, KeyError):
        pass
    if not weights.is_file() or not source.is_dir() or str(device).startswith('xpu'):
        notify('采用人物长窗口与地面估计水平', 50)
        return {}
    model, cap = None, None
    result = {}
    try:
        import cv2
        import torch
        for path in (source, root / 'optional_packages/geocalib'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from geocalib import GeoCalib
        check()
        notify('估计画面重力方向', 48)
        model = GeoCalib(weights=str(weights)).to(device).eval()
        cap = cv2.VideoCapture(str(video))
        with torch.inference_mode():
            for shot in shots:
                a, b = shot['start'], shot['end']
                frames = np.unique(np.linspace(a+min(2,b-a-1), b-1-min(2,b-a-1), 5, dtype=int))
                directions = []
                for frame in frames:
                    check()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first+frame))
                    ok, image = cap.read()
                    if not ok:
                        raise ValueError('Gravity source decode failed')
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2,0,1).float().to(device)/255
                    pred = model.calibrate(tensor, priors={'focal': torch.tensor([K[frame,1,1]], device=device)})
                    up = pred['gravity'].vec3d.detach().cpu().numpy().reshape(-1,3)[0]
                    world = rt[frame,:3,:3].T @ up
                    directions.append(world/np.linalg.norm(world))
                directions = np.asarray(directions)
                mean = np.median(directions, axis=0)
                mean /= np.linalg.norm(mean)
                dispersion = np.rad2deg(np.arccos(np.clip(directions @ mean, -1, 1)))
                result[int(shot['id'])] = dict(mean_world_up=mean.tolist(), dispersion_max_deg=float(dispersion.max()))
        save_json(cache, dict(identity=identity, shots=result))
    except CameraCancelled:
        raise
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        notify(f'画面重力不可用，采用人物与地面估计：{exc}', 50)
        result = {}
    finally:
        if cap is not None:
            cap.release()
        del model
    return result
