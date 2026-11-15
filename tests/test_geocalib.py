import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from flask import Flask

from flowhmr import geocalib_setup as setup, model_files as files, model_sources as sources
from flowhmr import model_import, model_integrity
from flowhmr.model_inventory import model_inventory
from flowhmr.multishot.camera import Camera
from flowhmr.multishot.gravity import image_gravity
from flowhmr.multishot.quality import camera_quality
from flowhmr.multishot.recovery import CameraCancelled
from flowhmr.multishot.tasks import CameraTasks
from flowhmr.multishot.web import register_camera_routes
from flowhmr.multishot.exports import camera_bundle


class GeoCalibTests(unittest.TestCase):
    def test_bundled_source_matches_manifest(self):
        manifest = json.loads((setup.SOURCE_ROOT / 'manifest.json').read_text())
        self.assertEqual(manifest['revision'], setup.SOURCE_REVISION)
        for relative, expected in manifest['files'].items():
            self.assertEqual(hashlib.sha256((setup.SOURCE_ROOT / relative).read_bytes()).hexdigest(), expected)

    def test_missing_model_is_discoverable_but_does_not_block_motion(self):
        with TemporaryDirectory() as temp, patch.object(files, 'MODEL_ROOT', Path(temp)):
            geo = next(row for row in model_inventory('cpu')['items'] if row['id'] == 'geocalib')
            self.assertEqual(geo['status'], 'missing')
            self.assertFalse(geo['blocking'])
            self.assertEqual(geo['role'], 'source_camera')
            self.assertEqual(sources.PUBLIC_ASSETS['geocalib'][0].relative, files.GEOCALIB)
            self.assertNotIn(files.resolve_model(files.GEOCALIB), files.core_model_paths())

    def test_command_line_perception_download_includes_verified_geocalib(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location('geocalib_download_test', files.PROJECT_ROOT / 'tools/windows/download_models.py')
        download_models = module_from_spec(spec)
        spec.loader.exec_module(download_models)
        with patch('sys.argv', ['download_models.py', '--perception-only']), \
                patch.object(files, 'ensure_model_folders'), patch.object(files, 'smplh_reference', return_value='neutral'), \
                patch.object(download_models, 'download_url') as fetch:
            self.assertEqual(download_models.main(), 0)
            asset = sources.PUBLIC_ASSETS['geocalib'][0]
            fetch.assert_any_call(asset.urls[0], asset.relative, asset.size, asset.sha256)

    def test_weight_integrity_including_same_size_corruption(self):
        content = b'valid model'
        asset = sources.Asset(files.GEOCALIB, len(content), hashlib.sha256(content).hexdigest())
        with TemporaryDirectory() as temp, patch.object(files, 'MODEL_ROOT', Path(temp)), \
                patch.object(model_integrity, '_CACHE_PATH', Path(temp) / 'integrity.json'), \
                patch.dict(sources.PUBLIC_ASSETS, geocalib=(asset,)), \
                patch.object(model_integrity, 'pinned_asset', return_value=asset), \
                patch.object(setup, 'runtime_issue', return_value=None):
            path = files.resolve_model(files.GEOCALIB)
            self.assertEqual(setup.readiness('cpu', verify=True), 'weights_missing')
            path.parent.mkdir()
            path.write_bytes(b'bad')
            self.assertEqual(setup.readiness('cpu', verify=True), 'weights_corrupt')
            path.write_bytes(b'!' * len(content))
            self.assertEqual(setup.readiness('cpu', verify=True), 'weights_corrupt')
            path.write_bytes(content)
            self.assertIsNone(setup.readiness('cpu', verify=True))

    def test_official_filename_import_and_archive_checkpoint(self):
        content = b'known model fixture'
        asset = sources.Asset(files.GEOCALIB, len(content), hashlib.sha256(content).hexdigest())
        with TemporaryDirectory() as temp, patch.dict(model_import._ASSETS, {files.GEOCALIB: asset}):
            folder = Path(temp)
            (folder / 'geocalib-pinhole.tar').write_bytes(content)
            found = model_import._discover(folder)
            self.assertEqual(found[files.GEOCALIB][0].path.name, 'geocalib-pinhole.tar')
            # This .tar is a PyTorch checkpoint, not a nested archive to expand.
            member = model_import._ArchiveIndex()
            self.assertEqual(member.add('models/geocalib-pinhole.tar', len(content), False), files.GEOCALIB)

    def test_unsupported_device_and_missing_runtime_are_explicit(self):
        self.assertEqual(setup.runtime_issue('xpu:0'), 'unsupported_device')
        with patch.object(setup.util, 'find_spec', return_value=None):
            self.assertEqual(setup.runtime_issue('cpu'), 'runtime_missing')

    def test_missing_corrupt_and_unsupported_return_diagnostics_before_cache(self):
        for reason in ('weights_missing', 'weights_corrupt', 'unsupported_device', 'runtime_missing'):
            diagnostic = {}
            with patch.object(setup, 'readiness', return_value=reason), patch.object(setup, 'load_class') as load:
                result = image_gravity(None, None, None, None, 0, None, 'cpu', lambda: None,
                                       lambda *args: None, diagnostics=diagnostic)
                self.assertEqual(result, {})
                self.assertEqual(diagnostic['reason'], reason)
                load.assert_not_called()

    def test_load_failure_is_persistable_and_cancel_does_not_degrade(self):
        with TemporaryDirectory() as temp, patch.object(setup, 'readiness', return_value=None):
            folder = Path(temp)
            video = folder / 'video'
            video.write_bytes(b'identity only')
            diagnostic = {}
            with patch.object(setup, 'load_class', side_effect=ImportError('missing dependency')):
                result = image_gravity(video, np.eye(4)[None], np.eye(3)[None], [], 0, folder, 'cpu',
                                       lambda: None, lambda *args: None, diagnostics=diagnostic)
            self.assertEqual(result, {})
            self.assertEqual(diagnostic['reason'], 'runtime_missing')
            self.assertIn('missing dependency', diagnostic['detail'])
            with self.assertRaises(CameraCancelled):
                image_gravity(None, None, None, None, 0, None, 'cpu',
                              lambda: (_ for _ in ()).throw(CameraCancelled()), lambda *args: None)

    def test_quality_reports_accepted_rejected_and_legacy_gravity(self):
        report = dict(shots=[dict(shot=0, gravity_source='image'),
                             dict(shot=1, gravity_source='body_long_window_and_ground')],
                      gravity=dict(status='measured'))
        quality = camera_quality(report)
        self.assertEqual((quality['status'], quality['reason'], quality['fallback_shots']), ('partial', 'unstable', [1]))
        del report['gravity']
        self.assertEqual(camera_quality(report)['reason'], 'legacy_fallback')
        report['shots'].pop()
        self.assertFalse(camera_quality(report)['degraded'])
        self.assertFalse(camera_quality(dict(shared_world=True))['degraded'])

    def test_ready_http_export_and_legacy_reopen_keep_warning_without_mutating_history(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'camera').mkdir()
            np.savez(root / 'motion.npz', trans=np.zeros((3, 3)))
            frozen = (root / 'motion.npz').read_bytes()
            rt = np.tile(np.eye(4), (3, 1, 1))
            K = np.tile([[500., 0, 320], [0, 500, 240], [0, 0, 1]], (3, 1, 1))
            job = dict(id='sample', status='done')
            report = dict(shots=[dict(shot=0, gravity_source='body_long_window_and_ground')],
                          gravity=dict(status='unavailable', reason='weights_missing'))
            def build(*args, **kwargs):
                Camera(rt, K, np.ones(3, bool), np.zeros(3, np.int32), np.array([480, 640])).save(root / 'camera/camera.npz')
                (root / 'camera/report.json').write_text(json.dumps(report))
            service = CameraTasks(lambda _: copy.deepcopy(job), lambda _: root, lambda _, **kw: job.update(kw), lambda _: None, build)
            service.run('sample')
            self.assertTrue(job['camera_quality']['degraded'])
            self.assertIn('降级', job['camera_message'])
            app = Flask(__name__)
            register_camera_routes(app, lambda _: job, lambda _: root, None, service)
            client = app.test_client()
            for path in ('camera', 'camera/status'):
                response = client.get('/api/jobs/sample/' + path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json['quality']['reason'], 'weights_missing')
            with ZipFile(camera_bundle(root, 'npz', None)) as archive:
                self.assertEqual(json.loads(archive.read('export-report.json'))['quality']['reason'], 'weights_missing')
            del report['gravity']
            (root / 'camera/report.json').write_text(json.dumps(report))
            saved = copy.deepcopy(job)
            self.assertEqual(service.status('sample')['quality']['reason'], 'legacy_fallback')
            self.assertEqual(job, saved)
            self.assertEqual((root / 'motion.npz').read_bytes(), frozen)


if __name__ == '__main__':
    unittest.main()
