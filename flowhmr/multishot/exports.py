"""Paired camera bundles. The stored motion and camera are never modified."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from .camera import Camera, transform_parameters
from .timeline import gap_report


def camera_bundle(directory, format_name, joints, transform=None):
    if format_name not in ('vmd', 'fbx', 'npz'):
        raise ValueError('Camera export supports VMD, FlowHMR FBX or NPZ')
    root = Path(directory)
    original = Camera.load(root / 'camera/camera.npz')
    scale, translation = transform_parameters(transform)
    camera = original.transformed(scale, translation)
    from .ground import stored_floor
    ground=stored_floor(root)*scale+float(translation[1])
    if joints is not None:
        joints = np.asarray(joints) * scale + translation
    with np.load(root / 'motion.npz', allow_pickle=False) as motion:
        if 'trans' in motion:
            if len(motion['trans']) != len(camera.RT):
                raise ValueError('Paired camera and motion lengths differ')
            gaps = gap_report(motion, camera.track_start)
        else:
            from .timeline import intervals
            scene = json.loads((root / 'scene.json').read_text(encoding='utf8'))
            if scene['num_frames'] != len(camera.RT):
                raise ValueError('Shared camera and scene lengths differ')
            gaps = dict(fps=30, track_start=camera.track_start, interval_convention='half_open_source_frames',
                        people=scene['people'], gaps=[[a+camera.track_start, b+camera.track_start]
                        for a,b in intervals(~camera.camera_valid)])
    report = {'camera_transform': {'scale': scale, 'translation': translation.tolist()},
              'motion': gaps, 'camera': {'format': format_name},
              'source_world': 'paired_motion_y_up_metres',
              'required_aspect_ratio': float(camera.image_size_hw[1] / camera.image_size_hw[0])}
    from .quality import read_quality
    report['quality'] = read_quality(root)
    if (root / 'scene.json').exists():
        report['character_scale_policy'] = 'Camera uses the first selected character scale; different body sizes cannot all preserve the same projection.'
    detail = root / 'camera/report.json'
    if detail.exists():
        report['recovery'] = json.loads(detail.read_text(encoding='utf8'))
    folder = root / 'camera/exports'
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='paired-', dir=folder) as temp:
        temp = Path(temp)
        camera.save(temp / 'camera.npz')
        (temp / 'camera.json').write_text(json.dumps({**camera.json(ground), 'quality': report['quality']}, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        if format_name == 'vmd':
            from .camera_vmd import export_camera_vmd
            report['camera'] = export_camera_vmd(camera, joints, temp / 'camera.vmd')
        elif format_name == 'fbx':
            from .camera_fbx import export_camera_fbx
            report['camera'] = export_camera_fbx(camera, temp / 'camera.fbx')
        (temp / 'export-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        (temp / 'READ_ME.txt').write_text(
            'FlowHMR paired motion and camera\n'
            '30 FPS; source frame 0 is frame 0. Use the source image aspect ratio in export-report.json.\n'
            'VMD: load the character motion on the same PMX, and camera.vmd on the camera.\n'
            'Blender mmd_tools: import the PMX and both VMD files at scale 0.08, current frame 0, frame margin 0.\n'
            'Enable the model root Use Property Driver option for display and IK animation.\n'
            'VMD uses one integer vertical FOV per shot plus optical-axis dolly. FBX averages fx/fy.\n'
            'These are approximations. camera.npz and camera.json retain full intrinsics and independent camera validity.\n'
            'VMD hides motion gaps. FBX/VRMA hold the preceding pose; before entry they hold the first valid pose.\n'
            'Consult the gap ranges in the export report. Camera gaps contain no keys; players may interpolate across them.\n'
            'A reference pose at FBX frame -1 is outside the paired camera timeline.\n'
            'The character model and textures are not included.\n', encoding='utf-8')
        stream = BytesIO()
        with ZipFile(stream, 'w', ZIP_DEFLATED) as archive:
            for path in sorted(temp.iterdir()):
                archive.write(path, path.name)
        stream.seek(0)
        return stream


def motion_report_bundle(path, source, filename, start=None):
    """Only motions carrying actual gaps change their old download format."""
    with np.load(source, allow_pickle=False) as motion:
        if 'valid_frames' not in motion:
            return None
        report = gap_report(motion, start)
    if not report['gaps']:
        return None
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.write(path, filename)
        archive.writestr('motion-report.json', json.dumps(report, ensure_ascii=False, indent=2))
    output.seek(0)
    return output
