"""One published ground frame for linked motion and all of its consumers."""
import copy
import json
from pathlib import Path
import numpy as np
from .runtime import digest,save_json


def align_segment_ground(parts,reports,first=0,*,fixed_floor=None):
    """Translate each whole segment once; never change relative foot heights."""
    if len(parts)!=len(reports) or not parts:raise ValueError('Missing segment ground reports')
    floors=[float(r['floor']) for r in reports]
    if not np.isfinite(floors).all():raise ValueError('Non-finite segment ground')
    common=floors[0] if len(parts)==1 else (0. if fixed_floor is None else float(fixed_floor))
    if not np.isfinite(common):raise ValueError('Non-finite common ground')
    normalized=[];segments=[]
    for (a,b,motion,broken),floor,report in zip(parts,floors,reports):
        shift=common-floor
        if shift:
            motion={**motion,'trans':motion['trans'].copy()};motion['trans'][:,1]+=shift
        normalized.append((a,b,motion,broken))
        segments.append(dict(source_frames=[first+a,first+b],input_floor_m=floor,translation_m=[0.,shift,0.],
                             correction=copy.deepcopy(report)))
    ground=dict(height_m=common,frame='final_motion_world',method='constant_translation_per_segment',segments=segments)
    return normalized,ground


def aggregate_correction(ground):
    reports=[s['correction'] for s in ground['segments']]
    if len(reports)==1:return {**reports[0],'ground':ground}
    presets={r['preset'] for r in reports}
    if len(presets)!=1:raise ValueError('Segment correction presets do not agree')
    combined=dict(floor=ground['height_m'],preset=reports[0]['preset'],contacts=sum(int(r['contacts']) for r in reports),
                  elapsed_seconds=sum(float(r['elapsed_seconds']) for r in reports),
                  warning=' '.join(dict.fromkeys(r.get('warning','') for r in reports if r.get('warning'))),ground=ground)
    fits=[r['sole_fit'] for r in reports if 'sole_fit' in r]
    if fits:combined['sole_fit']=dict(strength=fits[0]['strength'],adjusted=sum(int(f['adjusted']) for f in fits))
    return combined


def write_correction(motion_path,report):
    path=Path(motion_path)
    save_json(path.with_suffix('.correction.json'),dict(version=1,motion_sha256=digest(path),report=report))


def correction_summary(motion_path,postprocess,params):
    if postprocess is None:return None
    path=Path(motion_path);sidecar=path.with_suffix('.correction.json')
    if sidecar.exists():
        saved=json.loads(sidecar.read_text(encoding='utf8'))
        if saved['version']!=1 or saved['motion_sha256']!=digest(path):raise ValueError('Correction metadata does not match motion')
        report=saved['report']
    else:
        # Compatibility for pre-sidecar single-interval callers. A normalized
        # linked result must never fall back to the last interval's report.
        with np.load(path,allow_pickle=False) as z:
            if 'ground_height_m' in z:raise ValueError('Linked motion is missing its ground report')
        report=postprocess.last_report
    result=dict(floor_m=float(report['floor']),automatic=params['correction_floor_m'] is None,preset=report['preset'],
                contacts=int(report['contacts']),seconds=float(report['elapsed_seconds']),warning=report.get('warning',''))
    for key in ('sole_fit','ground'):
        if key in report:result[key]=report[key]
    return result


def published_floor(directory,correction,fallback):
    """The final motion's coordinate declaration outranks legacy job metadata."""
    with np.load(Path(directory)/'motion.npz',allow_pickle=False) as z:
        if 'ground_height_m' in z:
            value=float(z['ground_height_m'])
            if not np.isfinite(value):raise ValueError('Invalid published ground')
            return value
    return float((correction or {}).get('floor_m',fallback))


def stored_floor(directory):
    """Read the matching world declaration for a persisted job or scene."""
    root=Path(directory)
    if (root/'scene.json').is_file():
        return float(json.loads((root/'scene.json').read_text(encoding='utf8')).get('ground_m',0.))
    job=json.loads((root/'job.json').read_text(encoding='utf8')) if (root/'job.json').is_file() else {}
    return published_floor(root,job.get('correction'),0.)
