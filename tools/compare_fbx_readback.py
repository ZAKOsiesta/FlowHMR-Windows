"""Diagnose skeleton FBX differences; strict byte equality still controls exit status."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import fbx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowhmr.core.export.fbx_utils import create_manager, load_scene, collect_all_nodes


def snapshot(path):
    manager = create_manager()
    try:
        scene = load_scene(manager, str(path))
        stack = scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        if stack is None:
            raise ValueError("FBX has no animation stack")
        scene.SetCurrentAnimationStack(stack)
        span = stack.GetLocalTimeSpan()
        first = int(round(span.GetStart().GetSecondDouble() * 30))
        last = int(round(span.GetStop().GetSecondDouble() * 30))
        nodes = collect_all_nodes(scene.GetRootNode())
        if isinstance(nodes, dict):
            nodes = list(nodes.values())
        nodes = sorted(nodes, key=lambda n: n.GetName())
        hierarchy = [(n.GetName(), n.GetParent().GetName() if n.GetParent() else None) for n in nodes]
        transforms = np.empty((last - first + 1, len(nodes), 4, 4), dtype=np.float64)
        for i, frame in enumerate(range(first, last + 1)):
            time = fbx.FbxTime()
            time.SetSecondDouble(frame / 30)
            for j, node in enumerate(nodes):
                matrix = node.EvaluateGlobalTransform(time)
                transforms[i, j] = [[matrix[r][c] for c in range(4)] for r in range(4)]
        settings = scene.GetGlobalSettings()
        meta = {'hierarchy': hierarchy, 'first_frame': first, 'last_frame': last,
                'unit_scale_cm': settings.GetSystemUnit().GetScaleFactor(),
                'coordinate_system': str(settings.GetAxisSystem().GetCoorSystem()),
                'up_vector': str(settings.GetAxisSystem().GetUpVector()),
                'nodes': len(nodes), 'sampled_frames': len(transforms)}
        return meta, transforms
    finally:
        manager.Destroy()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    left, a = snapshot(args.left)
    right, b = snapshot(args.right)
    same_shape = a.shape == b.shape
    report = dict(left=str(args.left.resolve()), right=str(args.right.resolve()),
                  byte_equal=digest(args.left) == digest(args.right),
                  metadata_equal=left == right,
                  transforms_bit_exact=bool(same_shape and np.array_equal(a, b)),
                  maximum_matrix_difference=float(np.max(np.abs(a-b))) if same_shape else None,
                  readback_left=left, readback_right=right,
                  scope='Skeleton hierarchy, units, axes, time range, integer-frame global matrices only; '
                        'not skinned geometry, materials, subframe interpolation or rendered output.')
    report['strict_pass'] = report['byte_equal']
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.report:
        args.report.write_text(text, encoding='utf-8')
    print(text)
    return 0 if report['strict_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
