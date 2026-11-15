"""GeoCalib measurements with explicit, persistent fallback diagnostics."""
import json
import numpy as np


def image_gravity(video, rt, K, shots, first, folder, device, check, notify, *, diagnostics=None):
    from .runtime import digest, array_digest, save_json
    from .recovery import CameraCancelled
    from .. import geocalib_setup as setup, model_files, model_sources
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.update(status='unavailable', source_revision=setup.SOURCE_REVISION)
    check()
    reason = setup.readiness(device, verify=True)
    if reason:
        diagnostic['reason'] = reason
        notify('画面重力不可用，采用人物与地面估计：' + reason, 50)
        return {}
    weights = model_files.resolve_model(model_files.GEOCALIB)
    diagnostic['model_sha256'] = model_sources.PUBLIC_ASSETS['geocalib'][0].sha256
    cache = folder / 'gravity.json'
    identity = dict(video=digest(video), RT=array_digest(rt), K=array_digest(K), first=first,
                    shots=shots, weights=diagnostic['model_sha256'], source_revision=setup.SOURCE_REVISION)
    try:
        old = json.loads(cache.read_text(encoding='utf8'))
        if old['identity'] == identity:
            diagnostic.update(status='measured', cache_reused=True)
            return {int(k): v for k, v in old['shots'].items()}
    except (OSError, ValueError, KeyError):
        pass
    model, cap = None, None
    result = {}
    try:
        import cv2
        import torch
        GeoCalib = setup.load_class()
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
        diagnostic.update(status='measured', cache_reused=False)
    except CameraCancelled:
        raise
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        diagnostic.update(status='unavailable', reason='runtime_missing' if isinstance(exc, ImportError) else 'inference_failed',
                          detail=f'{type(exc).__name__}: {exc}')
        notify(f'画面重力不可用，采用人物与地面估计：{exc}', 50)
        result = {}
    finally:
        if cap is not None:
            cap.release()
        del model
    return result
