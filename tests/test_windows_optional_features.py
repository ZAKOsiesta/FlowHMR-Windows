"""Optional features must not make a usable installation or download fail."""
from contextlib import ExitStack, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from flowhmr import geocalib_setup, model_files, model_inventory


def script(name):
    spec = spec_from_file_location(name, model_files.PROJECT_ROOT / f'tools/windows/{name}.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OptionalFeatureCommandsTest(unittest.TestCase):
    def check_install(self, *, missing=None, geo_error=None, flags=()):
        check = script('check_install')
        torch = MagicMock(__version__='test')
        torch.version = SimpleNamespace(cuda='12.8', hip=None)
        torch.device.return_value = SimpleNamespace(type='cuda')
        torch.cuda.get_device_name.return_value = 'test device'
        torch.cuda.get_device_capability.return_value = (8, 0)
        vision = MagicMock()
        vision.ops.nms.return_value.tolist.return_value = [0]

        def import_dependency(name):
            if name == missing:
                raise ImportError(f'{name} hidden for test')
            if name == 'rgbmocap_correction':
                raise ImportError('optional native correction not installed')
            return SimpleNamespace()

        with TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(check, 'ROOT', Path(folder)))
            stack.enter_context(patch.object(check.os, 'chdir'))
            runtime = stack.enter_context(patch.object(check, 'RUNTIME'))
            runtime.summary.return_value = {'ready': True}
            runtime.default_device.return_value = 'cuda'
            runtime.cuda_variant.return_value = None
            stack.enter_context(patch.object(check.importlib, 'import_module', side_effect=import_dependency))
            stack.enter_context(patch.dict(sys.modules, {
                'torch': torch, 'torchvision': vision,
                'flowhmr.utils.runtime_video_processing': SimpleNamespace(find_ffmpeg=lambda: 'ffmpeg'),
            }))
            stack.enter_context(patch('flowhmr.runtime_rocm.configure_windows_hip'))
            stack.enter_context(patch.object(geocalib_setup, 'load_class', side_effect=geo_error))
            stack.enter_context(patch.object(geocalib_setup, 'readiness', return_value=None))
            stack.enter_context(patch.object(check, 'missing_model_paths', return_value=[]))
            stack.enter_context(patch.object(check, 'core_model_paths', return_value=[]))
            stack.enter_context(patch.object(model_inventory, 'model_inventory', return_value={'items': []}))
            stack.enter_context(patch.object(sys, 'argv', ['check_install.py', '--json', *flags]))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            code = check.main()
            return code, json.loads(output.getvalue())

    def test_optional_import_failure_keeps_installation_ready_and_reports_degradation(self):
        for missing in ('scenedetect', 'kornia', 'kornia_rs'):
            with self.subTest(missing=missing):
                code, report = self.check_install(missing=missing)
                self.assertEqual(code, 0, report)
                self.assertTrue(report['ready'])
                self.assertEqual(report['errors'], [])
                self.assertIn('unavailable', report['optional'][missing])
                feature = 'shot_linking' if missing == 'scenedetect' else 'source_camera_gravity'
                self.assertEqual(report[feature]['status'], 'degraded')

    def test_geocalib_import_error_is_degraded_even_when_readiness_probe_passes(self):
        for flags in ((), ('--environment-only',)):
            with self.subTest(flags=flags):
                code, report = self.check_install(geo_error=ImportError('broken optional runtime'), flags=flags)
                self.assertEqual(code, 0, report)
                self.assertEqual(report['source_camera_gravity']['status'], 'degraded')
                self.assertEqual(report['source_camera_gravity']['reason'], 'runtime_missing')
                self.assertIn('broken optional runtime', report['optional']['geocalib'])

    def test_required_import_failure_still_blocks_installation(self):
        code, report = self.check_install(missing='yolox')
        self.assertEqual(code, 2)
        self.assertFalse(report['ready'])
        self.assertTrue(any('yolox' in error for error in report['errors']))

    def run_download(self, failed_target):
        download = script('download_models')
        def fetch(url, target, *checks):
            if target == failed_target:
                raise OSError('simulated network failure')
        with patch.object(sys, 'argv', ['download_models.py', '--perception-only']), \
                patch.object(model_files, 'ensure_model_folders'), \
                patch.object(model_files, 'smplh_reference', return_value='neutral'), \
                patch.object(download, 'download_url', side_effect=fetch), redirect_stdout(io.StringIO()) as output:
            return download.main(), output

    def test_geocalib_download_failure_warns_but_required_downloads_succeed(self):
        code, output = self.run_download(model_files.GEOCALIB)
        self.assertEqual(code, 0, output.getvalue())
        self.assertIn('Warning', output.getvalue())
        self.assertIn('GeoCalib', output.getvalue())
        self.assertNotIn('Unfinished downloads:', output.getvalue())

    def test_required_perception_download_failure_still_fails(self):
        code, output = self.run_download(model_files.VGGT)
        self.assertEqual(code, 1)
        self.assertIn('Unfinished downloads:', output.getvalue())


if __name__ == '__main__':
    unittest.main()
