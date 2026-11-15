"""Exercise the Windows junction bug from the superproject, not just the child."""
import importlib.util
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'Windows directory junction regression')
class WorktreeDependenciesTest(unittest.TestCase):
    def test_sam_resolved_location_keeps_its_local_dinov3_sibling(self):
        spec = importlib.util.spec_from_file_location('dependency_setup', ROOT/'tools/prepare_worktree_dependencies.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seed, old, new, storage = [root/name for name in ('seed', 'old', 'new', 'storage')]
            def git(path, *args):
                return subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                       '-C', str(path), *args], capture_output=True, text=True, check=True)
            for path in (seed, old):
                path.mkdir(); git(path, 'init', '-q')
            relative = Path('sam_3d_body/models/backbones/dinov3.py')
            backbone = seed/relative
            backbone.parent.mkdir(parents=True)
            # Match the installed SAM call site: resolve the junction before locating its sibling.
            backbone.write_text('from pathlib import Path\n'
                                'def load():\n'
                                '    return (Path(__file__).resolve().parents[4]/"dinov3/hubconf.py").read_text()\n')
            git(seed, 'add', '.'); git(seed, 'commit', '-qm', 'local SAM backbone')
            git(old, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(seed), 'third_party/sam-3d-body')
            git(old, 'commit', '-qam', 'submodule')
            git(old, 'worktree', 'add', '--detach', str(new), 'HEAD')
            dino = old/'third_party/dinov3'
            dino.mkdir(); (dino/'hubconf.py').write_text('original local DINO\n')
            module.junction(new/'third_party/dinov3', dino)
            old_git = (old/'third_party/sam-3d-body/.git').read_bytes()
            module.prepare(old, new, storage)
            installed = new/'third_party/sam-3d-body'/relative
            loader = importlib.util.spec_from_file_location('installed_backbone', installed)
            installed_module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(installed_module)
            self.assertEqual(installed_module.load(), 'original local DINO\n')
            target = storage/'third_party/dinov3'
            self.assertEqual((new/'third_party/dinov3').resolve(), target.resolve())
            self.assertNotEqual(target.resolve(), dino.resolve())
            (target/'hubconf.py').write_text('new local DINO\n')
            module.prepare(old, new, storage)
            self.assertEqual(installed_module.load(), 'new local DINO\n')
            self.assertEqual((dino/'hubconf.py').read_text(), 'original local DINO\n')
            self.assertEqual((old/'third_party/sam-3d-body/.git').read_bytes(), old_git)
            self.assertEqual(git(new, 'status', '--short').returncode, 0)

    def test_independent_repository_preserves_p4_patches_and_top_level_git(self):
        spec = importlib.util.spec_from_file_location('dependency_setup', ROOT/'tools/prepare_worktree_dependencies.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            seed, old, new, storage = [root/name for name in ('seed', 'old', 'new', 'storage')]
            def git(path, *args, check=True):
                return subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                                       '-C', str(path), *args], capture_output=True, text=True, check=check)
            for path in (seed, old):
                path.mkdir(); git(path, 'init', '-q')
            (seed/'dependency.py').write_text('original\n')
            git(seed, 'add', '.'); git(seed, 'commit', '-qm', 'dependency')
            git(old, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(seed), 'third_party/model')
            git(old, 'commit', '-qam', 'submodule')
            git(old, 'worktree', 'add', '--detach', str(new), 'HEAD')
            child = old/'third_party/model'
            (child/'dependency.py').write_text('installed local patch\n')
            (child/'local_runtime.txt').write_text('untracked installed runtime\n')
            old_git = (child/'.git').read_bytes()
            before = git(old, 'status', '--short').stdout
            (new/'third_party/model').rmdir()
            module.junction(new/'third_party/model', child)
            self.assertNotEqual(git(new, 'status', '--short', check=False).returncode, 0)
            result = module.prepare(old, new, storage)
            self.assertEqual(len(result), 1)
            self.assertEqual(git(new, 'status', '--short').stdout, before)
            self.assertEqual(git(new, 'submodule', 'status').returncode, 0)
            self.assertEqual((child/'.git').read_bytes(), old_git)
            self.assertEqual(git(old, 'status', '--short').stdout, before)
            target = new/'third_party/model'
            self.assertNotEqual(git(target, 'rev-parse', '--absolute-git-dir').stdout,
                                git(child, 'rev-parse', '--absolute-git-dir').stdout)
            self.assertEqual((target/'dependency.py').read_text(), 'installed local patch\n')
            self.assertEqual((target/'local_runtime.txt').read_text(), 'untracked installed runtime\n')
            (target/'dependency.py').write_text('new version only\n')
            module.prepare(old, new, storage)
            self.assertEqual((target/'dependency.py').read_text(), 'new version only\n')
            self.assertEqual((child/'dependency.py').read_text(), 'installed local patch\n')
