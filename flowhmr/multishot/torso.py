"""Compose torso motion only on observed, source-indexed generation intervals."""
import numpy as np

from .timeline import held_indices
from flowhmr.utils.runtime_torso_optimization import compose_torso_optimization


def compose_linked(standard, spans, generate, model_path):
    source = np.asarray(standard["source_frames"])
    valid = np.asarray(standard["valid_frames"])
    count = len(source)
    if (not count or source.ndim != 1 or source.dtype.kind not in "iu" or
            valid.shape != (count,) or valid.dtype != np.bool_ or not np.all(np.diff(source) == 1)):
        raise ValueError("Invalid linked source timeline")
    first = int(source[0])
    covered = np.zeros(count, dtype=bool)
    result = dict(standard)
    for key in ("poses", "trans", "Rh"):
        result[key] = np.array(standard.get(key, standard["poses"][:, 0]), copy=True)
    metrics = []
    for start, stop in spans:
        section = slice(start-first, stop-first)
        if not first <= start < stop <= first+count or not valid[section].all() or covered[section].any():
            raise ValueError(f"Invalid linked interval {start}:{stop}")
        base = generate(start, stop)
        if (not np.array_equal(base["source_frames"], source[section]) or
                len(base["poses"]) != stop-start or len(base["trans"]) != stop-start):
            raise ValueError(f"Base pass source mismatch at {start}:{stop}")
        reference = {key: standard[key] for key in ("betas", "mocap_framerate", "body_reference") if key in standard}
        reference.update(poses=standard["poses"][section], trans=standard["trans"][section])
        composed, report = compose_torso_optimization(base, reference, model_path)
        for key in ("poses", "trans", "Rh"):
            result[key][section] = composed[key]
        covered[section] = True
        metrics.append(dict(source_frames=[start, stop], **report))
    if not np.array_equal(covered, valid):
        raise ValueError("Base pass does not cover exactly the valid frames")
    hold = held_indices(valid)
    for key in ("poses", "trans", "Rh"):
        result[key] = result[key][hold]
    return result, dict(mode="linked", frames=int(valid.sum()), intervals=metrics)
