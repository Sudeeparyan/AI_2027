"""Package the instructor project; exclude credentials, environments and caches.

This complete instructor archive includes answer guides. Distribute the weekly
student files separately rather than sharing this archive with students.
"""
from pathlib import Path
import argparse
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent


def package(output, include_history=False, root=ROOT):
    root = Path(root).resolve()
    out = (root / output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    example = root / 'config/providers.example.json'
    if not example.is_file():
        raise ValueError('Missing clean provider configuration example')
    excluded = {'.venv', 'venv', '__pycache__', '.git', 'drafts', 'backups',
                '.pytest_cache', '.mypy_cache', '.ruff_cache', '.ipynb_checkpoints'}
    with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.stem + '-', suffix='.tmp', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for f in sorted(root.rglob('*')):
                if not f.is_file() or f.is_symlink() or f.resolve() in (out, temporary_path):
                    continue
                rel = f.relative_to(root)
                if any(part in excluded or part.startswith('.venv-') for part in rel.parts):
                    continue
                if f.suffix.lower() in ('.pyc', '.key', '.pem', '.zip') or f.name.startswith('.env') or f.name.endswith('.local.json'):
                    continue
                if not include_history and rel.parts[:2] == ('outputs', 'runs'):
                    continue
                if rel.as_posix() == 'config/providers.json':
                    continue
                archive.write(f, (Path(root.name) / rel).as_posix())
            archive.write(example, (Path(root.name) / 'config/providers.json').as_posix())
        with zipfile.ZipFile(temporary_path) as archive:
            if archive.testzip():
                raise RuntimeError('ZIP integrity check failed')
        temporary_path.replace(out)
    finally:
        temporary_path.unlink(missing_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='../Generative_AI_2027_Teaching_Pack.zip')
    parser.add_argument('--include-history', action='store_true')
    args = parser.parse_args()
    out = package(args.output, args.include_history)
    print(out, round(out.stat().st_size / 1048576, 2), 'MiB')


if __name__ == '__main__':
    main()
