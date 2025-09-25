#!/usr/bin/env python3
"""
A股数据采集模块 - 50只股票扩展版本
支持7年历史数据、50只精选股票、质量优先的数据收集

重要更新:
- 时间范围: 2018-2024 (7年)
- 股票数量: 50只精选股票 (科学分层抽样)
- 质量控制: 更严格的数据验证和完整性检查
- 性能优化: 适配小规模高质量采集
"""

import akshare as ak
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self, data_dir='data', cache_dir='cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)

        # 创建目录
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # 线程锁，用于API限频
        self.api_lock = threading.Lock()
        self.last_api_call = 0
        self.min_interval = 0.1  # API调用间隔(秒)

        # 缓存配置
        self.cache_file = self.cache_dir / 'download_status.json'
        self.load_cache_status()

        # 50只精选股票代码
        self.SELECTED_STOCKS = [
            # 超大盘股 (10只)
            '601138', '688981', '000333', '601601', '002415',
            '000651', '601766', '302132', '300033', '601881',

            # 大盘股 (12只)
            '600031', '605499', '300014', '601888', '601800',
            '000617', '601390', '000166', '600989', '600905',
            '600886', '600415',

            # 中大盘股 (15只)
            '000895', '001979', '600703', '002466', '000807',
            '601872', '601607', '002648', '300251', '600426',
            '002202', '300207', '601058', '000999', '002555',

            # 中盘股 (10只)
            '300316', '002252', '601598', '002601', '603087',
            '000786', '603129', '002078', '002032', '600392'
        ]

        print(f"📁 数据目录: {self.data_dir}")
        print(f"💾 缓存目录: {self.cache_dir}")
        print(f"📊 股票数量: {len(self.SELECTED_STOCKS)} 只")

    def load_cache_status(self):
        """加载下载状态缓存"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache_status = json.load(f)
        else:
            self.cache_status = {}

        print(f"📋 已加载缓存状态: {len(self.cache_status)} 条记录")

    def save_cache_status(self):
        """保存下载状态缓存"""
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_status, f, ensure_ascii=False, indent=2)

    def rate_limit(self):
        """API调用限频控制"""
        with self.api_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_api_call
            if time_since_last < self.min_interval:
                time.sleep(self.min_interval - time_since_last)
            self.last_api_call = time.time()

    def download_single_stock(self, stock_code, start_date='20180101', end_date='20241231'):
        """下载单只股票的7年历史数据"""

        # 检查缓存
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if cache_key in self.cache_status and self.cache_status[cache_key]['status'] == 'success':
            return self.load_stock_data(stock_code)

        try:
            # API限频
            self.rate_limit()

            print(f"   📥 正在下载 {stock_code} 数据...")

            # 获取日度行情数据
            daily_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""  # 不复权，后续自己处理
            )

            if daily_data.empty:
                raise ValueError(f"股票 {stock_code} 无数据")

            # 数据预处理
            daily_data = self.preprocess_daily_data(daily_data, stock_code)

            # 数据质量检查
            self.validate_single_stock_data(daily_data, stock_code)

            # 保存数据
            output_file = self.data_dir / f"{stock_code}.csv"
            daily_data.to_csv(output_file, index=False, encoding='utf-8-sig')

            # 更新缓存状态
            self.cache_status[cache_key] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'rows': len(daily_data),
                'date_range': f"{daily_data['date'].min()} - {daily_data['date'].max()}"
            }

            print(f"   ✅ {stock_code} 下载成功: {len(daily_data)} 条记录")
            return daily_data

        except Exception as e:
            error_msg = f"下载 {stock_code} 失败: {str(e)}"
            print(f"   ❌ {error_msg}")

            # 记录失败状态
            self.cache_status[cache_key] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

            return None

    def preprocess_daily_data(self, data, stock_code):
        """预处理日度数据"""
        # 重命名列名为英文
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover_rate'
        }

        # 应用列名映射
        for old_name, new_name in column_mapping.items():
            if old_name in data.columns:
                data = data.rename(columns={old_name: new_name})

        # 添加股票代码
        data['stock_code'] = stock_code

        # 确保日期列是datetime格式
        data['date'] = pd.to_datetime(data['date'])

        # 按日期排序
        data = data.sort_values('date').reset_index(drop=True)

        # 数据类型转换
        numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def validate_single_stock_data(self, data, stock_code):
        """验证单只股票数据质量"""
        if len(data) < 1200:  # 7年约1750个交易日，要求至少1200天
            raise ValueError(f"数据量不足: {len(data)} < 1200 天")

        # 检查必要字段
        required_fields = ['date', 'open', 'close', 'high', 'low', 'volume']
        missing_fields = [field for field in required_fields if field not in data.columns]
        if missing_fields:
            raise ValueError(f"缺少必要字段: {missing_fields}")

        # 检查空值
        for field in required_fields:
            null_count = data[field].isnull().sum()
            if null_count > len(data) * 0.05:  # 空值不超过5%
                raise ValueError(f"字段 {field} 空值过多: {null_count}/{len(data)}")

    def load_stock_data(self, stock_code):
        """加载单只股票数据"""
        try:
            file_path = self.data_dir / f"{stock_code}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path, encoding='utf-8-sig')
                data['date'] = pd.to_datetime(data['date'])
                return data
            else:
                return None
        except Exception as e:
            print(f"❌ 加载 {stock_code} 数据失败: {e}")
            return None

    def download_all_stocks(self, start_date='20180101', end_date='20241231', max_workers=3):
        """下载所有50只股票数据"""

        print(f"\n🚀 开始下载 {len(self.SELECTED_STOCKS)} 只股票的7年历史数据")
        print(f"   时间范围: {start_date} - {end_date}")
        print(f"   并发数: {max_workers}")
        print("="*60)

        successful_downloads = []
        failed_downloads = []

        # 使用线程池下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有下载任务
            future_to_code = {
                executor.submit(self.download_single_stock, code, start_date, end_date): code
                for code in self.SELECTED_STOCKS
            }

            # 收集结果
            for future in as_completed(future_to_code):
                stock_code = future_to_code[future]
                try:
                    result = future.result()
                    if result is not None:
                        successful_downloads.append(stock_code)
                    else:
                        failed_downloads.append(stock_code)
                except Exception as e:
                    print(f"   ❌ {stock_code} 线程异常: {e}")
                    failed_downloads.append(stock_code)

        # 保存缓存状态
        self.save_cache_status()

        # 输出结果统计
        print("\n" + "="*60)
        print(f"📊 下载完成统计:")
        print(f"   ✅ 成功: {len(successful_downloads)} 只")
        print(f"   ❌ 失败: {len(failed_downloads)} 只")

        if failed_downloads:
            print(f"   失败股票: {failed_downloads}")

        return successful_downloads, failed_downloads

    def validate_data_quality(self, stock_codes=None):
        """验证数据质量"""
        if stock_codes is None:
            stock_codes = self.SELECTED_STOCKS

        print("\n🔍 验证数据质量...")

        quality_report = {
            'total_stocks': len(stock_codes),
            'valid_stocks': 0,
            'insufficient_data': [],
            'missing_files': [],
            'date_range_stats': {},
            'quality_score': 0
        }

        for code in stock_codes:
            data = self.load_stock_data(code)
            if data is None:
                quality_report['missing_files'].append(code)
                continue

            if len(data) < 1200:  # 7年数据要求至少1200个交易日
                quality_report['insufficient_data'].append(code)
                continue

            quality_report['valid_stocks'] += 1

            # 统计日期范围
            min_date = data['date'].min()
            max_date = data['date'].max()
            quality_report['date_range_stats'][code] = {
                'start_date': str(min_date),
                'end_date': str(max_date),
                'trading_days': len(data),
                'completeness': len(data) / 1750  # 7年理论交易日数
            }

        # 计算质量评分
        quality_report['quality_score'] = quality_report['valid_stocks'] / quality_report['total_stocks']

        # 输出质量报告
        print(f"   📊 质量统计:")
        print(f"   - 总股票数: {quality_report['total_stocks']}")
        print(f"   - 数据有效: {quality_report['valid_stocks']}")
        print(f"   - 数据缺失: {len(quality_report['missing_files'])}")
        print(f"   - 数据不足: {len(quality_report['insufficient_data'])}")
        print(f"   - 质量评分: {quality_report['quality_score']:.1%}")

        # 保存质量报告
        report_file = self.data_dir / 'data_quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2, default=str)

        print(f"   💾 质量报告已保存到: {report_file}")

        return quality_report

    def generate_summary_statistics(self):
        """生成汇总统计"""
        print("\n📈 生成汇总统计...")

        all_data = []
        for code in self.SELECTED_STOCKS:
            data = self.load_stock_data(code)
            if data is not None:
                all_data.append(data)

        if not all_data:
            print("   ❌ 没有有效数据")
            return

        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)

        # 计算基础统计
        stats = {
            'total_observations': len(combined_data),
            'total_stocks': len(all_data),
            'date_range': {
                'start': str(combined_data['date'].min()),
                'end': str(combined_data['date'].max())
            },
            'avg_trading_days_per_stock': len(combined_data) / len(all_data),
            'price_statistics': {
                'avg_price': combined_data['close'].mean(),
                'median_price': combined_data['close'].median(),
                'price_std': combined_data['close'].std()
            }
        }

        print(f"   📊 数据汇总:")
        print(f"   - 总观测值: {stats['total_observations']:,}")
        print(f"   - 有效股票: {stats['total_stocks']}")
        print(f"   - 日期范围: {stats['date_range']['start']} ~ {stats['date_range']['end']}")
        print(f"   - 平均交易日/股票: {stats['avg_trading_days_per_stock']:.0f}")

        # 保存统计报告
        stats_file = self.data_dir / 'summary_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

        print(f"   💾 统计报告已保存到: {stats_file}")

        return stats

def main():
    """主函数 - 执行50只股票7年数据采集"""

    import argparse
    parser = argparse.ArgumentParser(description='50只股票数据采集')
    parser.add_argument('--start', type=str, default='20180101')
    parser.add_argument('--end', type=str, default='20241231')
    parser.add_argument('--max-workers', type=int, default=3)
    args = parser.parse_args()

    print("🚀 启动50只股票扩展数据采集")
    print("="*70)
    print("📊 样本设计: 科学分层抽样，兼顾代表性与质量")
    print(f"🗓️ 时间范围: {args.start} ~ {args.end}")

    collector = DataCollector()

    # 1. 下载所有股票数据
    print(f"\n🎯 开始下载 {len(collector.SELECTED_STOCKS)} 只股票的数据...")
    successful_stocks, failed_stocks = collector.download_all_stocks(
        start_date=args.start,
        end_date=args.end,
        max_workers=args.max_workers
    )

    # 2. 数据质量验证
    quality_report = collector.validate_data_quality()

    # 3. 生成汇总统计
    summary_stats = collector.generate_summary_statistics()

    # 4. 重试失败股票（如果有）
    if failed_stocks:
        print(f"\n🔄 重试 {len(failed_stocks)} 只失败股票...")
        retry_successful, still_failed = collector.download_all_stocks(
            start_date='20180101',
            end_date='20241231',
            max_workers=1  # 单线程重试
        )

        if still_failed:
            print(f"   ⚠️ 仍有 {len(still_failed)} 只股票下载失败: {still_failed}")
        else:
            print("   ✅ 所有股票重试成功!")

    # 5. 最终统计
    print("\n" + "="*70)
    print("🎉 50只股票数据采集完成!")
    print(f"✅ 成功率: {quality_report['quality_score']:.1%}")

    if quality_report['quality_score'] >= 0.9:
        print("🏆 数据质量优秀，可以进行下一步分析!")
    elif quality_report['quality_score'] >= 0.8:
        print("👍 数据质量良好，建议检查缺失数据后继续!")
    else:
        print("⚠️ 数据质量需要改进，建议检查网络和API状态!")

    print("="*70)

if __name__ == "__main__":
    main()
