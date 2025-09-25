#!/usr/bin/env python3
"""
A股数据采集模块 - 扩展版本
支持5年历史数据、500只股票、高并发下载、严格质量控制

重要升级:
- 时间范围: 2020-2024 (5年 vs 原2年)
- 股票数量: 支持500只 (vs 原100只)
- 并发优化: max_workers=5, batch_size=20
- 质量控制: 更严格的数据验证和流动性筛选
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
        
        # 线程锁，用于API限频 (优化为更高并发)
        self.api_lock = threading.Lock()
        self.last_api_call = 0
        self.min_interval = 0.08  # 最小API调用间隔(秒) - 从0.1优化到0.08
        
        # 缓存配置
        self.cache_file = self.cache_dir / 'download_status.json'
        self.load_cache_status()
        
        print(f"📁 数据目录: {self.data_dir}")
        print(f"💾 缓存目录: {self.cache_dir}")
    
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
    
    def download_single_stock(self, stock_code, start_date='20200101', end_date='20241231'):
        """下载单只股票的日度数据"""
        
        # 检查缓存
        cache_key = f"{stock_code}_{start_date}_{end_date}"
        if cache_key in self.cache_status and self.cache_status[cache_key]['status'] == 'success':
            return self.load_stock_data(stock_code)
        
        try:
            # API限频
            self.rate_limit()
            
            # 获取日度行情数据
            daily_data = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=""
            )
            
            if daily_data.empty:
                raise ValueError(f"股票 {stock_code} 无数据")
            
            # 数据预处理
            daily_data = self.preprocess_daily_data(daily_data, stock_code)
            
            # 保存数据
            output_file = self.data_dir / f"{stock_code}.csv"
            daily_data.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 更新缓存状态
            self.cache_status[cache_key] = {
                'status': 'success',
                'download_time': datetime.now().isoformat(),
                'records': len(daily_data),
                'file_path': str(output_file)
            }
            
            return daily_data
            
        except Exception as e:
            # 记录失败状态
            self.cache_status[cache_key] = {
                'status': 'failed',
                'error': str(e),
                'download_time': datetime.now().isoformat()
            }
            print(f"❌ {stock_code} 下载失败: {e}")
            return None
    
    def preprocess_daily_data(self, df, stock_code):
        """预处理日度数据"""
        df = df.copy()
        
        # 标准化列名
        column_mapping = {
            '日期': 'date',
            '股票代码': 'code', 
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保有股票代码
        if 'code' not in df.columns or df['code'].isna().all():
            df['code'] = stock_code
        
        # 转换数据类型
        df['date'] = pd.to_datetime(df['date'])
        
        numeric_columns = ['open', 'close', 'high', 'low', 'volume', 'amount', 
                          'amplitude', 'pct_change', 'change', 'turnover']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 数据质量检查 (提高标准)
        if len(df) < 1000:  # 少于1000个交易日 (5年约1250个交易日)
            print(f"⚠️  {stock_code} 数据不足: {len(df)} 个交易日 (需要≥1000天，5年数据)")
        
        return df
    
    def load_stock_data(self, stock_code):
        """从本地文件加载股票数据"""
        file_path = self.data_dir / f"{stock_code}.csv"
        if file_path.exists():
            try:
                return pd.read_csv(file_path, encoding='utf-8-sig')
            except Exception as e:
                print(f"❌ 加载 {stock_code} 数据失败: {e}")
                return None
        return None
    
    def download_stock_batch(self, stock_codes, start_date='20200101', end_date='20241231', 
                           max_workers=5, batch_size=20):
        """批量下载股票数据"""
        
        print(f"🚀 开始批量下载 {len(stock_codes)} 只股票数据")
        print(f"   时间范围: {start_date} - {end_date}")
        print(f"   并发数: {max_workers}, 批次大小: {batch_size}")
        print("="*60)
        
        results = {}
        failed_stocks = []
        
        # 分批处理
        for i in range(0, len(stock_codes), batch_size):
            batch = stock_codes[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(stock_codes) + batch_size - 1) // batch_size
            
            print(f"\n📦 处理批次 {batch_num}/{total_batches} ({len(batch)} 只股票)")
            
            # 多线程下载当前批次
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务
                future_to_stock = {
                    executor.submit(self.download_single_stock, code, start_date, end_date): code 
                    for code in batch
                }
                
                # 收集结果
                for future in as_completed(future_to_stock):
                    stock_code = future_to_stock[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results[stock_code] = result
                            print(f"   ✅ {stock_code}: {len(result)} 条记录")
                        else:
                            failed_stocks.append(stock_code)
                            print(f"   ❌ {stock_code}: 下载失败")
                    except Exception as e:
                        failed_stocks.append(stock_code)
                        print(f"   ❌ {stock_code}: 异常 - {e}")
            
            # 保存缓存状态
            self.save_cache_status()
            
            # 批次间休息
            if i + batch_size < len(stock_codes):
                print(f"   ⏸️  批次休息 2 秒...")
                time.sleep(2)
        
        # 统计结果
        success_count = len(results)
        fail_count = len(failed_stocks)
        
        print("\n" + "="*60)
        print(f"📊 下载完成统计:")
        print(f"   成功: {success_count} 只")
        print(f"   失败: {fail_count} 只")
        print(f"   成功率: {success_count/(success_count+fail_count)*100:.1f}%")
        
        if failed_stocks:
            print(f"\n❌ 失败股票列表: {failed_stocks[:10]}")
            if len(failed_stocks) > 10:
                print(f"   ... 另外 {len(failed_stocks)-10} 只")
        
        return results, failed_stocks
    
    def load_stock_universe(self, universe_file='stock_universe.csv'):
        """加载股票池"""
        try:
            # 将code列作为字符串读取，保持前导零
            universe_df = pd.read_csv(universe_file, encoding='utf-8-sig', dtype={'code': str})
            
            # 确保股票代码格式正确（6位数字，不足补零）
            universe_df['code'] = universe_df['code'].str.zfill(6)
            
            stock_codes = universe_df['code'].tolist()
            print(f"📋 从 {universe_file} 加载 {len(stock_codes)} 只股票")
            print(f"   示例代码: {stock_codes[:5]}")
            return stock_codes
        except Exception as e:
            print(f"❌ 加载股票池失败: {e}")
            return []
    
    def validate_data_quality(self, stock_codes):
        """验证数据质量"""
        print("\n🔍 验证数据质量...")
        
        quality_report = {
            'total_stocks': len(stock_codes),
            'valid_stocks': 0,
            'insufficient_data': [],
            'missing_files': [],
            'date_range_stats': {}
        }
        
        for code in stock_codes:
            data = self.load_stock_data(code)
            if data is None:
                quality_report['missing_files'].append(code)
                continue
            
            if len(data) < 1000:  # 提高数据量要求至1000个交易日
                quality_report['insufficient_data'].append(code)
                continue
            
            quality_report['valid_stocks'] += 1
            
            # 统计日期范围
            min_date = data['date'].min()
            max_date = data['date'].max()
            quality_report['date_range_stats'][code] = {
                'start_date': str(min_date),
                'end_date': str(max_date),
                'trading_days': len(data)
            }
        
        # 输出质量报告
        print(f"   📊 质量统计:")
        print(f"   - 总股票数: {quality_report['total_stocks']}")
        print(f"   - 数据有效: {quality_report['valid_stocks']}")
        print(f"   - 数据不足: {len(quality_report['insufficient_data'])}")
        print(f"   - 文件缺失: {len(quality_report['missing_files'])}")
        
        # 保存质量报告
        report_file = self.cache_dir / 'data_quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        print(f"   💾 质量报告已保存到: {report_file}")
        
        return quality_report
    
    def validate_liquidity_requirements(self, stock_codes, min_trading_days=200, 
                                      lookback_days=250):
        """验证流动性要求 - 新增功能"""
        print(f"\n💧 验证流动性要求 (近{lookback_days}天中≥{min_trading_days}天有交易)...")
        
        liquidity_report = {
            'total_stocks': len(stock_codes),
            'liquid_stocks': [],
            'illiquid_stocks': [],
            'missing_data': []
        }
        
        # 计算截止日期 (近250个自然日对应约200个交易日)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for code in stock_codes:
            data = self.load_stock_data(code)
            if data is None:
                liquidity_report['missing_data'].append(code)
                continue
            
            # 转换日期列
            data['date'] = pd.to_datetime(data['date'])
            
            # 筛选近期数据
            recent_data = data[data['date'] >= start_date]
            
            if len(recent_data) >= min_trading_days:
                liquidity_report['liquid_stocks'].append({
                    'code': code,
                    'trading_days': len(recent_data)
                })
            else:
                liquidity_report['illiquid_stocks'].append({
                    'code': code,
                    'trading_days': len(recent_data)
                })
        
        print(f"   📊 流动性统计:")
        print(f"   - 总股票数: {liquidity_report['total_stocks']}")
        print(f"   - 流动性充足: {len(liquidity_report['liquid_stocks'])}")
        print(f"   - 流动性不足: {len(liquidity_report['illiquid_stocks'])}")
        print(f"   - 数据缺失: {len(liquidity_report['missing_data'])}")
        
        # 保存流动性报告
        liquidity_file = self.cache_dir / 'liquidity_report.json'
        with open(liquidity_file, 'w', encoding='utf-8') as f:
            json.dump(liquidity_report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"   💾 流动性报告已保存到: {liquidity_file}")
        
        return liquidity_report
    
    def collect_market_data_batch(self, stock_codes):
        """批量获取市场数据 (市值、PE、PB等)"""
        print("\n📈 获取市场数据...")
        
        try:
            # 获取实时行情数据
            market_data = ak.stock_zh_a_spot_em()
            
            # 筛选目标股票
            target_data = market_data[market_data['代码'].isin(stock_codes)].copy()
            
            # 标准化列名
            column_mapping = {
                '代码': 'code',
                '名称': 'name',
                '总市值': 'total_market_cap',
                '流通市值': 'float_market_cap',
                '市盈率-动态': 'pe_ttm',
                '市净率': 'pb'
            }
            
            target_data = target_data.rename(columns=column_mapping)
            
            # 选择需要的列
            keep_columns = ['code', 'name', 'total_market_cap', 'float_market_cap', 'pe_ttm', 'pb']
            target_data = target_data[keep_columns]
            
            # 保存市场数据
            market_file = self.data_dir / 'market_data.csv'
            target_data.to_csv(market_file, index=False, encoding='utf-8-sig')
            
            print(f"   ✅ 成功获取 {len(target_data)} 只股票的市场数据")
            print(f"   💾 已保存到: {market_file}")
            
            return target_data
            
        except Exception as e:
            print(f"   ❌ 获取市场数据失败: {e}")
            return None

def main():
    """主函数 - 执行扩展数据采集 (Phase 1.2)"""
    import argparse
    parser = argparse.ArgumentParser(description='扩展数据采集')
    parser.add_argument('--universe', type=str, default='stock_universe.csv')
    parser.add_argument('--start', type=str, default='20200101')
    parser.add_argument('--end', type=str, default='20241231')
    parser.add_argument('--max-workers', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()

    print("🚀 启动Phase 1.2: 扩展数据采集")
    print("="*70)
    print(f"🎯 股票池: {args.universe}")
    print(f"🗓️ 时间范围: {args.start} ~ {args.end}")
    
    collector = DataCollector()
    
    # 1. 加载扩展股票池
    stock_codes = collector.load_stock_universe(args.universe)
    if not stock_codes:
        print("❌ 无法加载股票池")
        return
    
    # 2. 批量下载5年历史数据 (升级参数)
    print(f"\n🎯 开始下载 {len(stock_codes)} 只股票的数据...")
    results, failed_stocks = collector.download_stock_batch(
        stock_codes=stock_codes,
        start_date=args.start,
        end_date=args.end,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # 3. 获取市场数据
    market_data = collector.collect_market_data_batch(stock_codes)
    
    # 4. 数据质量验证 (更严格标准)
    quality_report = collector.validate_data_quality(stock_codes)
    
    # 5. 新增：流动性验证
    liquidity_report = collector.validate_liquidity_requirements(stock_codes)
    
    # 6. 自动重试失败股票
    if failed_stocks:
        print(f"\n🔄 自动重试 {len(failed_stocks)} 只失败股票...")
        retry_results, still_failed = collector.download_stock_batch(
            stock_codes=failed_stocks,
            start_date=args.start,
            end_date=args.end,
            max_workers=2,
            batch_size=5
        )
        print(f"   重试成功: {len(retry_results)} 只")
        print(f"   仍然失败: {len(still_failed)} 只")
    
    # 7. Phase 1.2 最终统计
    print("\n" + "="*70)
    print("✅ Phase 1.2 完成: 扩展数据采集!")
    print(f"📊 采集统计:")
    print(f"   - 目标股票: {len(stock_codes)} 只")
    print(f"   - 成功下载: {len(results)} 只")
    print(f"   - 数据有效: {quality_report['valid_stocks']} 只 (≥1000交易日)")
    print(f"   - 流动性充足: {len(liquidity_report['liquid_stocks'])} 只")
    print(f"   - 时间范围: 2020-2024 (5年)")
    print(f"   - 预期观测量: 375k+ (vs 原47k)")
    
    if still_failed if 'still_failed' in locals() else failed_stocks:
        remaining_failed = still_failed if 'still_failed' in locals() else failed_stocks
        print(f"   ⚠️ 仍需关注: {len(remaining_failed)} 只股票数据获取失败")
    
    print("✅ 下一步: Phase 1.3 精简因子至6个核心因子")

if __name__ == "__main__":
    main()
