#!/usr/bin/env python3
"""
Clean up local artifacts before publishing the repo.

Removes heavy data outputs, caches, and local environments while keeping
minimal placeholders for a smooth first run.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def safe_rmtree(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def main():
    # 1) Remove local virtualenv
    safe_rmtree(ROOT / '.venv')
    safe_rmtree(ROOT / '.venv-3.12')

    # 2) Remove pycaches
    for d in ROOT.rglob('__pycache__'):
        safe_rmtree(d)

    # 3) Prune data directory, keep placeholders
    if DATA.exists():
        for p in DATA.iterdir():
            if p.name in {'README_DATA.md', 'predictions_20240101_000000.csv'}:
                continue
            if p.is_dir():
                safe_rmtree(p)
            else:
                try:
                    p.unlink()
                except Exception:
                    pass

    # 4) Keep logs directory but empty
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    for f in logs.glob('*'):
        if f.name == '.gitkeep':
            continue
        try:
            f.unlink()
        except Exception:
            pass

    print('Cleanup complete. Repo is ready for publishing.')


if __name__ == '__main__':
    main()

