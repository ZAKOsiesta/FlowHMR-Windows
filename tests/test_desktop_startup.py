"""Workspace health is acknowledged only by the local, initialized desktop."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest import TestCase, skipUnless
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class FrontendStartupTest(TestCase):
    @skipUnless(shutil.which('node'), 'Node is required for the JavaScript startup regression')
    def test_actual_page_controller_and_update_bridge(self):
        result = subprocess.run(['node', '--experimental-vm-modules',
                                 str(ROOT / 'tools/check_desktop_startup.mjs')],
                                capture_output=True, text=True, encoding='utf8', timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@skipUnless(os.name == 'nt', 'Requires Windows desktop imports')
class DesktopReadinessTest(TestCase):
    def test_health_acknowledgement_requires_local_origin_and_open_backend(self):
        with patch.object(sys, 'path', [str(ROOT / 'tools/windows'), *sys.path]):
            desktop = importlib.import_module('desktop')
        with TemporaryDirectory() as directory:
            work = Path(directory)
            (work / 'applying').mkdir()
            exposed = {}
            window = SimpleNamespace(get_current_url=lambda: 'about:blank',
                                     expose=lambda *functions: exposed.update({f.__name__: f for f in functions}))
            backend, ready = SimpleNamespace(closed=Event()), Event()
            with patch.object(desktop, 'DesktopUpdate', return_value=SimpleNamespace(work=work)):
                desktop.expose_updates(window, 60566, backend, ready, None)
                confirm = exposed['confirm_update_boot']
                for url in ('about:blank', 'https://example.org', 'http://127.0.0.1:60567/'):
                    window.get_current_url = lambda: url
                    self.assertFalse(confirm())
                    self.assertFalse(ready.is_set())
                    self.assertFalse((work / 'applying/healthy').exists())
                window.get_current_url = lambda: 'http://127.0.0.1:60566/'
                self.assertTrue(confirm())
                self.assertTrue(ready.is_set())
                self.assertTrue((work / 'applying/healthy').exists())
                self.assertTrue(confirm())  # duplicate notification is idempotent
                backend.closed.set()
                self.assertFalse(confirm())

    def test_non_trial_acknowledges_without_creating_health_marker(self):
        with patch.object(sys, 'path', [str(ROOT / 'tools/windows'), *sys.path]):
            desktop = importlib.import_module('desktop')
        with TemporaryDirectory() as directory:
            exposed = {}
            window = SimpleNamespace(get_current_url=lambda: 'http://127.0.0.1:60566/',
                                     expose=lambda *functions: exposed.update({f.__name__: f for f in functions}))
            with patch.object(desktop, 'DesktopUpdate', return_value=SimpleNamespace(work=Path(directory))):
                desktop.expose_updates(window, 60566, SimpleNamespace(closed=Event()), None, None)
                self.assertTrue(exposed['confirm_update_boot']())
                self.assertFalse((Path(directory) / 'applying').exists())
