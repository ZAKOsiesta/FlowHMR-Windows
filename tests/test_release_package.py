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
            files = ['app.py', 'tests/test_example.py', '.github/workflows/pr-size.yml', desktop_update.EDITION_FILE, 'tools/web/index.html',
                     'tools/windows/desktop_launcher.pyw', 'tools/windows/desktop_update.py',
                     'tools/web/static/js/update.js', 'flowhmr/multishot/runtime.py',
                     'third_party/geocalib_inference/geocalib/extractor.py']
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(desktop_update.EDITION if name == desktop_update.EDITION_FILE else 'fixture', encoding='utf8')
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
            self.assertFalse((stage / '.github').exists())
            self.assertTrue((root / '.github/workflows/pr-size.yml').is_file())
            self.assertTrue((root / 'tests/test_example.py').is_file())
            self.assertTrue((target / 'files/third_party/geocalib_inference/geocalib/extractor.py').is_file())
            self.assertIn('runtime/python.exe', (target / 'targets.txt').read_text())

    def test_different_editions_rejected_even_at_same_version(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / desktop_update.PACKAGE
            for marker in (None, 'another edition', desktop_update.EDITION):
                with ZipFile(path, 'w') as archive:
                    archive.writestr('FlowHMR/FlowHMR-Version.txt', '0.1.0')
                    if marker:
                        archive.writestr('FlowHMR/' + desktop_update.EDITION_FILE, marker)
                if marker == desktop_update.EDITION:
                    self.assertEqual(desktop_update._archive_version(path), '0.1.0')
                else:
                    with self.assertRaisesRegex(desktop_update.LocalUpdateError, 'wrong_edition'):
                        desktop_update._archive_version(path)
                    with self.assertRaisesRegex(desktop_update.LocalUpdateError, 'wrong_edition'):
                        desktop_update.stage_package(path, Path(temporary), '0.1.0', lambda: None)

    def test_only_published_ryki_releases_from_the_fork_are_offered(self):
        self.assertEqual(desktop_update.RELEASES_API, 'https://api.github.com/repos/ZAKOsiesta/FlowHMR-Windows/releases?per_page=100')
        tag = 'ryki-v0.1.1'
        expected = f'{desktop_update.RELEASES}/download/{tag}/{desktop_update.PACKAGE}'
        binary = dict(name=desktop_update.PACKAGE, state='uploaded', browser_download_url=expected, size=1, digest='sha256:' + 'a' * 64)
        entry = dict(tag_name=tag, assets=[binary, dict(name=desktop_update.UPDATE_INFO, state='uploaded')])
        self.assertTrue(desktop_update._portable_release(entry).installable)
        for change in (dict(tag_name='windows-v0.1.1'), dict(draft=True), dict(prerelease=True)):
            self.assertIsNone(desktop_update._portable_release({**entry, **change}))
        binary['browser_download_url'] = expected.replace('ZAKOsiesta/', 'xiaoerofficial021-collab/')
        with self.assertRaises(ValueError):
            desktop_update._portable_release(entry)
