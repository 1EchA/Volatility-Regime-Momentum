#!/usr/bin/env python3
"""
Report packager: bundle key outputs (predictions, execution metrics/timeseries,
turnover grids, robustness summaries, and generated visuals) into a zip with a
machine-readable summary and a human-readable Markdown README.

Usage example:
  python3 analysis/report_packager.py \
    --predictions data/predictions_*.csv \
    --execution-metrics data/pipeline_execution_*_metrics.json \
    --execution-timeseries data/pipeline_execution_*_timeseries.csv \
    --turnover-grid data/turnover_strategy_grid_*.csv \
    --robustness data/robustness_summary_*.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import shutil


def _copy_if_exists(src: str | None, out_dir: Path) -> list[Path]:
    out_paths: list[Path] = []
    if not src:
        return out_paths
    for pattern in str(src).split(','):
        for path in Path().glob(pattern.strip()):
            dest = out_dir / path.name
            shutil.copy2(path, dest)
            out_paths.append(dest)
    return out_paths


def main():
    p = argparse.ArgumentParser(description='Bundle outputs into a portable zip report')
    p.add_argument('--predictions', type=str, default=None, help='Glob(s) for predictions CSV')
    p.add_argument('--execution-metrics', type=str, default=None, help='Glob(s) for execution *_metrics.json')
    p.add_argument('--execution-timeseries', type=str, default=None, help='Glob(s) for execution *_timeseries.csv')
    p.add_argument('--turnover-grid', type=str, default=None, help='Glob(s) for turnover_strategy_grid_*.csv')
    p.add_argument('--robustness', type=str, default=None, help='Glob(s) for robustness_summary_*.csv')
    p.add_argument('--visuals', type=str, default='data/turnover_*.png,data/exec_profile_*_profile.png', help='Additional visuals to include (comma-separated globs)')
    args = p.parse_args()

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path('data') / f'report_package_{ts}'
    out_dir.mkdir(parents=True, exist_ok=True)

    included: dict[str, list[str]] = {}
    def add(label: str, items: list[Path]):
        if items:
            included[label] = [str(p) for p in items]

    add('predictions', _copy_if_exists(args.predictions, out_dir))
    add('execution_metrics', _copy_if_exists(args.execution_metrics, out_dir))
    add('execution_timeseries', _copy_if_exists(args.execution_timeseries, out_dir))
    add('turnover_grid', _copy_if_exists(args.turnover_grid, out_dir))
    add('robustness', _copy_if_exists(args.robustness, out_dir))
    add('visuals', _copy_if_exists(args.visuals, out_dir))

    # Aggregate a light summary from execution metrics if available
    summary = {
        'generated_at': ts,
        'contents': included,
        'highlights': [],
    }
    for p in out_dir.glob('*_metrics.json'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            summary['highlights'].append({
                'file': p.name,
                'strategy': meta.get('strategy'),
                'param': meta.get('param'),
                'ls_ann': meta.get('metrics', {}).get('ls_ann'),
                'ls_ir': meta.get('metrics', {}).get('ls_ir'),
                'avg_turnover': meta.get('metrics', {}).get('avg_turnover'),
                'max_drawdown': meta.get('metrics', {}).get('max_drawdown'),
                'execution_profile': meta.get('execution_profile', {}),
            })
        except Exception:
            continue

    # Write summary JSON & README.md
    with open(out_dir / 'SUMMARY.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write('# 策略报告包\n\n')
        f.write(f'- 生成时间: {ts}\n')
        f.write('- 包含文件:\n')
        for k, vs in included.items():
            f.write(f'  - {k}: {len(vs)} 个文件\n')
        if summary['highlights']:
            f.write('\n## 执行层摘要\n')
            for h in summary['highlights']:
                f.write(f"- {h['file']}: 策略={h.get('strategy')}, 参数={h.get('param')}, 年化={h.get('ls_ann'):.2%} IR={h.get('ls_ir'):.3f}, 换手={h.get('avg_turnover'):.2%}, 回撤={h.get('max_drawdown'):.2%}\n")

    # Zip folder
    zip_path = shutil.make_archive(str(out_dir), 'zip', root_dir=str(out_dir))
    print('✅ Report package created:', zip_path)


if __name__ == '__main__':
    main()
