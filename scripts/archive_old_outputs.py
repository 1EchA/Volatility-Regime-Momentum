#!/usr/bin/env python3
"""
数据归档脚本 - 安全地将历史数据文件移动到archive目录

用法:
    python scripts/archive_old_outputs.py [--dry-run] [--days-old 30]

功能:
- 移动旧的分析结果文件到data/archive/
- 保留最新的N个文件不被归档
- 支持dry-run模式预览操作
- 自动创建archive目录结构
"""

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 项目根目录和数据目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
ARCHIVE_DIR = DATA_DIR / 'archive'

# 需要归档的文件模式和保留策略
ARCHIVE_PATTERNS = {
    'predictions_*.csv': {'keep_latest': 3, 'days_old': 7},
    'pipeline_execution_*.json': {'keep_latest': 5, 'days_old': 14},
    'pipeline_execution_*.csv': {'keep_latest': 5, 'days_old': 14},
    'turnover_strategy_grid_*.csv': {'keep_latest': 2, 'days_old': 30},
    'cost_sensitivity_grid_*.csv': {'keep_latest': 2, 'days_old': 30},
    'performance_report_*.json': {'keep_latest': 3, 'days_old': 14},
    'performance_report_*.csv': {'keep_latest': 3, 'days_old': 14},
    'performance_report_*.png': {'keep_latest': 3, 'days_old': 14},
    'robustness_summary_*.csv': {'keep_latest': 2, 'days_old': 21},
    'factor_selection_*.csv': {'keep_latest': 2, 'days_old': 30},
    'model_grid_results_*.csv': {'keep_latest': 2, 'days_old': 30},
    '*_20250918_*.csv': {'keep_latest': 0, 'days_old': 0},  # 立即归档特定日期
    '*_20250922_*.csv': {'keep_latest': 0, 'days_old': 0},  # 立即归档特定日期
    '*_20250923_*.csv': {'keep_latest': 0, 'days_old': 0},  # 立即归档特定日期
}


def create_archive_structure():
    """创建archive目录结构"""
    ARCHIVE_DIR.mkdir(exist_ok=True)

    # 创建子目录按类型组织
    subdirs = [
        'predictions', 'pipeline_execution', 'grids',
        'performance_reports', 'robustness', 'factors', 'models'
    ]

    for subdir in subdirs:
        (ARCHIVE_DIR / subdir).mkdir(exist_ok=True)


def get_target_archive_dir(file_path: Path) -> Path:
    """根据文件类型确定归档目标目录"""
    filename = file_path.name.lower()

    if 'predictions' in filename:
        return ARCHIVE_DIR / 'predictions'
    elif 'pipeline_execution' in filename:
        return ARCHIVE_DIR / 'pipeline_execution'
    elif any(pattern in filename for pattern in ['grid', 'cost_sensitivity', 'turnover']):
        return ARCHIVE_DIR / 'grids'
    elif 'performance_report' in filename:
        return ARCHIVE_DIR / 'performance_reports'
    elif 'robustness' in filename:
        return ARCHIVE_DIR / 'robustness'
    elif 'factor' in filename:
        return ARCHIVE_DIR / 'factors'
    elif 'model' in filename:
        return ARCHIVE_DIR / 'models'
    else:
        return ARCHIVE_DIR


def should_archive_file(file_path: Path, pattern_config: dict, cutoff_date: datetime) -> bool:
    """判断文件是否应该被归档"""
    file_stat = file_path.stat()
    file_age = datetime.fromtimestamp(file_stat.st_mtime)

    # 检查文件年龄
    days_old_threshold = pattern_config.get('days_old', 30)
    if days_old_threshold == 0:  # 立即归档
        return True

    age_threshold = cutoff_date - timedelta(days=days_old_threshold)
    if file_age > age_threshold:
        return False  # 文件太新

    return True


def archive_files_by_pattern(pattern: str, config: dict, cutoff_date: datetime, dry_run: bool = True) -> tuple:
    """按模式归档文件"""
    files = list(DATA_DIR.glob(pattern))

    # 排除已经在archive目录中的文件
    files = [f for f in files if 'archive' not in f.parts]

    if not files:
        logger.info(f"No files found for pattern: {pattern}")
        return 0, 0

    # 按修改时间排序，最新的在前
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # 保留最新的N个文件
    keep_latest = config.get('keep_latest', 2)
    files_to_consider = files[keep_latest:]

    archived_count = 0
    kept_count = len(files) - len(files_to_consider)

    logger.info(f"Pattern {pattern}: {len(files)} total files, keeping {kept_count} latest")

    for file_path in files_to_consider:
        if should_archive_file(file_path, config, cutoff_date):
            target_dir = get_target_archive_dir(file_path)
            target_path = target_dir / file_path.name

            if dry_run:
                logger.info(f"[DRY RUN] Would move: {file_path} -> {target_path}")
            else:
                try:
                    # 检查目标文件是否已存在
                    if target_path.exists():
                        logger.warning(f"Target file exists, skipping: {target_path}")
                        continue

                    shutil.move(str(file_path), str(target_path))
                    logger.info(f"Moved: {file_path.name} -> {target_dir.name}/{target_path.name}")
                    archived_count += 1
                except Exception as e:
                    logger.error(f"Failed to move {file_path}: {e}")
        else:
            logger.info(f"Keeping (too recent): {file_path.name}")
            kept_count += 1

    return archived_count, kept_count


def main():
    parser = argparse.ArgumentParser(description='Archive old analysis output files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview operations without actually moving files')
    parser.add_argument('--days-old', type=int, default=30,
                       help='Additional age threshold for all files (days)')
    parser.add_argument('--force-archive-date', type=str,
                       help='Force archive files from specific date (YYYYMMDD format)')

    args = parser.parse_args()

    # 确保目录存在
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        return 1

    # 创建归档目录结构
    create_archive_structure()

    cutoff_date = datetime.now()
    total_archived = 0
    total_kept = 0

    logger.info(f"Starting archive process (dry_run={args.dry_run})")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Archive directory: {ARCHIVE_DIR}")

    # 处理每个模式
    for pattern, config in ARCHIVE_PATTERNS.items():
        logger.info(f"\n--- Processing pattern: {pattern} ---")

        # 如果指定了强制归档日期，特殊处理
        if args.force_archive_date and args.force_archive_date in pattern:
            config = config.copy()
            config['keep_latest'] = 0
            config['days_old'] = 0

        archived, kept = archive_files_by_pattern(
            pattern, config, cutoff_date, args.dry_run
        )
        total_archived += archived
        total_kept += kept

    # 汇总报告
    logger.info(f"\n=== Archive Summary ===")
    logger.info(f"Total files archived: {total_archived}")
    logger.info(f"Total files kept: {total_kept}")

    if args.dry_run:
        logger.info("This was a DRY RUN. No files were actually moved.")
        logger.info("Run without --dry-run to perform actual archiving.")
    else:
        logger.info("Archive operation completed successfully.")

    return 0


if __name__ == '__main__':
    exit(main())