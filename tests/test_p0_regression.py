from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'tools/p0_regression.py'
spec = spec_from_file_location('p0_regression', SCRIPT)
reg = module_from_spec(spec)
spec.loader.exec_module(reg)


class P0RegressionTest(unittest.TestCase):
    def test_byte_comparison_catches_numeric_and_missing_file_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.npz'
            np.savez(path, poses=np.zeros((2, 52, 3), dtype=np.float32))
            baseline = {'files': {'motion': reg.describe(path)}}
            self.assertTrue(reg.compare(baseline, {'motion': path})['ok'])
            np.savez(path, poses=np.ones((2, 52, 3), dtype=np.float32))
            changed = reg.compare(baseline, {'motion': path})
            self.assertFalse(changed['ok'])
            self.assertEqual(changed['files']['motion']['changed_arrays'], ['poses'])
            path.unlink()
            self.assertFalse(reg.compare(baseline, {'motion': path})['ok'])

    def test_cli_refuses_to_overwrite_an_existing_oracle(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, paths, baseline = root / 'motion.bin', root / 'paths.json', root / 'baseline.json'
            data.write_bytes(b'accepted motion')
            paths.write_text(json.dumps({'motion': str(data)}), encoding='utf-8')
            args = [sys.executable, str(SCRIPT), 'record', str(paths), str(baseline)]
            first = subprocess.run(args, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            frozen = baseline.read_bytes()
            data.write_bytes(b'changed motion')
            self.assertNotEqual(subprocess.run(args, capture_output=True).returncode, 0)
            self.assertEqual(baseline.read_bytes(), frozen)


if __name__ == '__main__':
    unittest.main()
