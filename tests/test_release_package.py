"""A staged portable package must be accepted by the existing updater."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from zipfile import ZipFile

with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'tools/windows'), *sys.path]):
    import desktop_update
    import package_release


class PortableUpdateTest(TestCase):
    def test_staged_portable_omits_tests_and_passes_update_extraction(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / 'source'
            root.mkdir()
            files = ['app.py', 'tests/test_example.py', 'tools/web/index.html',
                     'tools/windows/desktop_launcher.pyw', 'tools/windows/desktop_update.py',
                     'tools/web/static/js/update.js', 'flowhmr/multishot/runtime.py',
                     'third_party/geocalib_inference/geocalib/extractor.py']
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture', encoding='utf8')
            (root / 'third_party/hamer_inference').mkdir()
            stage = Path(temporary) / 'stage'
            def runtime(target, *args, **kwargs):
                target.mkdir()
                for name in ('python.exe', 'pythonw.exe'):
                    (target / name).write_bytes(b'MZ fixture')
            def launcher(*args, **kwargs):
                (stage / 'FlowHMR.exe').write_bytes(b'MZ fixture')
            with patch.object(package_release, 'ROOT', root), \
                 patch.object(package_release, 'source_files', return_value=list(map(Path, files))), \
                 patch.object(package_release, 'SUBMODULES', ()), \
                 patch.object(package_release, 'copy_dinov3'), \
                 patch.object(package_release, 'adapt_sam'), \
                 patch('flowhmr.hamer_setup.runtime_installed', return_value=True), \
                 patch.object(package_release, 'copy_runtime', side_effect=runtime), \
                 patch.object(package_release.subprocess, 'run', side_effect=launcher):
                package_release.stage_release(stage, root, root, version='9.9.901')
            archive_path = Path(temporary) / 'portable.zip'
            with ZipFile(archive_path, 'w') as archive:
                for path in stage.rglob('*'):
                    if path.is_file():
                        archive.write(path, Path('FlowHMR') / path.relative_to(stage))
            target = Path(temporary) / 'update'
            target.mkdir()
            desktop_update.stage_package(archive_path, target, '9.9.901', lambda: None)
            self.assertFalse((stage / 'tests').exists())
            self.assertTrue((root / 'tests/test_example.py').is_file())
            self.assertTrue((target / 'files/third_party/geocalib_inference/geocalib/extractor.py').is_file())
            self.assertIn('runtime/python.exe', (target / 'targets.txt').read_text())
