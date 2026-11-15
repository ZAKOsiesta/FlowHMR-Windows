import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from flowhmr.utils.runtime_hamer import worker_runtime_arguments

ROOT = Path(__file__).resolve().parents[1]


class HamerWorkerRuntimeTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32' and sys.version_info[:2] == (3, 10),
                         'Windows inference runtime')
    def test_isolated_worker_receives_system_runtime_from_external_site(self):
        with TemporaryDirectory() as temp:
            site = Path(temp)/'site-packages'
            for name, version in [('torch', '2.8.0+cu128'), ('torchvision', '0.23.0+cu128')]:
                package = site/name
                package.mkdir(parents=True)
                (package/'__init__.py').write_text('')
                metadata = site/f'{name}-{version}.dist-info'
                metadata.mkdir()
                (metadata/'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
            args = worker_runtime_arguments(SimpleNamespace(active='system'),
                                            SimpleNamespace(__file__=str(site/'torch/__init__.py')))
            code = (
                'import sys,json; sys.path.insert(0,sys.argv[1]); '
                'from flowhmr.utils.runtime_hamer_worker import configure_runtime; '
                'configure_runtime(sys.argv[2:]); '
                'from flowhmr.runtime_environments import RUNTIME; '
                'print(json.dumps(dict(active=RUNTIME.active,variant=RUNTIME.system,path=sys.path[0])))'
            )
            # -I must remain enabled; inheriting PYTHONPATH is not a valid fix.
            env = dict(os.environ, PYTHONPATH=str(site))
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(ROOT), *args],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record['variant'], 'cu128')
            self.assertEqual(record['active'], 'system')
            self.assertEqual(Path(record['path']).resolve(), site.resolve())

    def test_named_slots_and_default_keep_existing_arguments(self):
        self.assertEqual(worker_runtime_arguments(SimpleNamespace(active='cu128')), ['cu128'])
        self.assertEqual(worker_runtime_arguments(SimpleNamespace(active=None)), [])
