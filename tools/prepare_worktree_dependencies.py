"""Independent submodule repositories for a local Windows product worktree.

Keep large dependency copies in storage. Their own .git directories are copied
by Git (no shared objects), so junctions do not inherit another root's gitdir.
Preserve installed source patches without changing the source repository.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def junction(path, target):
    path, target = Path(path), Path(target).resolve()
    if os.name == 'nt':
        quote = lambda p: "'" + str(p).replace("'", "''") + "'"
        subprocess.run(['powershell', '-NoProfile', '-Command',
                        f'New-Item -ItemType Junction -Path {quote(path)} -Target {quote(target)} | Out-Null'],
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        path.symlink_to(target, target_is_directory=True)


def gitlinks(root):
    result = {}
    for record in git(root, 'ls-files', '--stage', '-z').split('\0'):
        if not record:
            continue
        info, name = record.split('\t', 1)
        mode, commit, stage = info.split()
        if mode == '160000':
            if stage != '0' or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'Unresolved or unsafe submodule: {name}')
            result[name] = commit
    return result


def replace_link(path, old_target, new_target=None):
    """Remove only a checked junction or empty directory, never its contents."""
    if os.path.lexists(path):
        if path.resolve() == Path(old_target).resolve() and path.resolve() != path.absolute():
            os.rmdir(path)  # Windows removes the junction itself, not its target.
        elif path.is_dir() and path.resolve() == path.absolute() and not any(path.iterdir()):
            path.rmdir()
        else:
            raise ValueError(f'Refusing to replace an unexpected directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if new_target is None:
        path.mkdir()
    else:
        junction(path, new_target)


def prepare(source, destination, storage):
    source, destination, storage = [Path(p).absolute() for p in (source, destination, storage)]
    if source.resolve() == destination.resolve() or storage.resolve().is_relative_to(source.resolve()):
        raise ValueError('Source, worktree and independent storage must be separate')
    expected = gitlinks(destination)
    if expected != gitlinks(source):
        raise ValueError('Dependency commits differ; explicit version adaptation is required')
    report = []
    for name, commit in expected.items():
        original, link, target = source/name, destination/name, storage/name
        link.absolute().relative_to(destination)
        target.absolute().relative_to(storage)
        if not (original/'.git').exists():
            if original.exists() and any(original.iterdir()):
                raise ValueError(f'Uninitialized dependency contains files: {original}')
            if link.resolve() != link.absolute() or not link.exists():
                replace_link(link, original)
            elif any(link.iterdir()):
                raise ValueError(f'Uninitialized destination is not empty: {link}')
            report.append(dict(path=name, state='uninitialized', commit=commit))
            continue
        if git(original, 'rev-parse', 'HEAD') != commit:
            raise ValueError(f'Installed dependency is on an unexpected commit: {original}')
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'clone', '--local', '--no-hardlinks', '--no-checkout',
                            str(original), str(target)], check=True)
            git(target, 'checkout', '--detach', commit)
            remote = git(original, 'config', '--get', 'remote.origin.url')
            git(target, 'remote', 'set-url', 'origin', remote)
            shutil.copytree(original, target, dirs_exist_ok=True, symlinks=True,
                            ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
            # Preserve tracked local deletions as well as modified/untracked files.
            for relative in git(original, 'ls-files', '-z').split('\0'):
                if relative and not (original/relative).exists():
                    removed = target/relative
                    removed.resolve().relative_to(target.resolve())
                    removed.unlink(missing_ok=True)
        git_dir = Path(git(target, 'rev-parse', '--absolute-git-dir')).resolve()
        if git_dir != (target/'.git').resolve() or not (target/'.git').is_dir():
            raise ValueError(f'Dependency does not own its Git metadata: {target}')
        if (git_dir/'objects/info/alternates').exists() or git(target, 'rev-parse', 'HEAD') != commit:
            raise ValueError(f'Dependency is shared or on an unexpected commit: {target}')
        if link.resolve() != target.resolve():
            replace_link(link, original, target)
        report.append(dict(path=name, state='independent', commit=commit,
                           storage=str(target), status=git(link, 'status', '--short')))
    # The installed offline SAM patch resolves __file__ before looking for DINOv3
    # next to sam-3d-body. A junction in the superproject alone cannot satisfy it.
    companion = 'third_party/dinov3'
    original, link, target = source/companion, destination/companion, storage/companion
    if ((source/'third_party/sam-3d-body/.git').exists()
            and original.is_dir() and companion not in expected):
        if not (original/'hubconf.py').is_file():
            raise ValueError(f'Installed local DINOv3 is incomplete: {original}')
        if not target.exists():
            shutil.copytree(original, target, symlinks=True,
                            ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        if target.resolve() != target.parent.resolve()/target.name or not (target/'hubconf.py').is_file():
            raise ValueError(f'Independent local DINOv3 is incomplete or shared: {target}')
        if link.resolve() != target.resolve():
            replace_link(link, original, target)
        report.append(dict(path=companion, state='independent-source', storage=str(target),
                           required_by='third_party/sam-3d-body', entry='hubconf.py'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--storage', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination, args.storage), ensure_ascii=False, indent=2))
