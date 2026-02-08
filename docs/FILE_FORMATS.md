# 📋 文件格式规范

> A股量化策略研究平台数据文件格式标准

## 📊 数据文件类型总览

本平台处理以下几类数据文件，每种都有严格的格式要求：

### 1️⃣ **股票价格数据** (Individual Stock CSV)
### 2️⃣ **预测结果文件** (Predictions CSV)
### 3️⃣ **波动率制度数据** (Volatility Regime CSV)
### 4️⃣ **策略执行结果** (Strategy Performance CSV)
### 5️⃣ **参数网格结果** (Parameter Grid CSV)
### 6️⃣ **因子数据** (Factor Data CSV)
### 7️⃣ **股票池信息** (Stock Universe CSV)

## 📈 1. 股票价格数据格式

**文件命名**: `{6位股票代码}.csv` (如: `000001.csv`, `600519.csv`)
**存放目录**: `data/`
**编码格式**: UTF-8

### 🔧 必需列 (Required Columns)
```csv
date,open,high,low,close,volume,amount,股票代码,股票名称
2020-01-02,11.85,11.94,11.83,11.92,486384,579736640.0,000001,平安银行
2020-01-03,11.90,12.05,11.88,12.02,712484,854442752.0,000001,平安银行
```

| 列名 | 数据类型 | 说明 | 示例 |
|-----|---------|------|------|
| `date` | String/Date | 交易日期 (YYYY-MM-DD) | `2020-01-02` |
| `open` | Float | 开盘价 | `11.85` |
| `high` | Float | 最高价 | `11.94` |
| `low` | Float | 最低价 | `11.83` |
| `close` | Float | 收盘价 | `11.92` |
| `volume` | Integer | 成交量(股) | `486384` |
| `amount` | Float | 成交额(元) | `579736640.0` |
| `股票代码` | String | 6位股票代码 | `000001` |
| `股票名称` | String | 股票中文名称 | `平安银行` |

### ⚠️ 注意事项
- 日期必须连续，缺失交易日会影响计算
- 价格数据建议使用前复权价格
- 成交量/成交额用于计算流动性指标

---

## 🎯 2. 预测结果文件格式

**文件命名**: `predictions_{YYYYMMDD}_{HHMMSS}.csv`
**存放目录**: `data/`
**文件大小**: 通常50-100MB+

### 🔧 必需列
```csv
date,stock_code,y_pred,y_true,regime,industry
2020-01-02,000001,0.0234,-0.0156,正常,银行
2020-01-02,000002,0.0145,0.0089,正常,房地产
```

| 列名 | 数据类型 | 说明 | 示例 |
|-----|---------|------|------|
| `date` | String/Date | 预测日期 | `2020-01-02` |
| `stock_code` | String | 6位股票代码(补齐前导0) | `000001` |
| `y_pred` | Float | 预测收益率 | `0.0234` |
| `y_true` | Float | 实际收益率 | `-0.0156` |
| `regime` | String | 波动率制度 | `正常`/`高波动`/`极高波动` |
| `industry` | String | 行业分类(可选) | `银行` |

### 📊 性能优化
- 文件按日期排序提高读取效率
- 支持分块读取(chunksize=300000)
- 股票代码统一6位格式(前导零补齐)

---

## 📊 3. 波动率制度数据格式

**文件命名**: `volatility_regime_data_{YYYYMMDD}_{HHMMSS}.csv`
**存放目录**: `data/`

### 🔧 必需列
```csv
date,stock_code,returns,volatility,regime,regime_prob
2020-01-02,000001,-0.0156,0.0234,正常,0.85
2020-01-02,000002,0.0089,0.0145,高波动,0.72
```

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| `date` | Date | 日期 |
| `stock_code` | String | 股票代码 |
| `returns` | Float | 日收益率 |
| `volatility` | Float | 波动率估计值 |
| `regime` | String | 制度分类 |
| `regime_prob` | Float | 制度概率(0-1) |

---

## ⚙️ 4. 策略执行结果格式

**文件命名**: `performance_report_{YYYYMMDD}_{HHMMSS}_timeseries.csv`

### 🔧 核心列
```csv
date,long_ret,short_ret,ls_ret,cumulative_ret,drawdown,turnover
2020-01-02,0.0123,-0.0089,0.0212,0.0212,0.0,0.23
2020-01-03,0.0156,-0.0034,0.0190,0.0406,-0.01,0.18
```

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| `date` | Date | 日期 |
| `long_ret` | Float | 多头收益率 |
| `short_ret` | Float | 空头收益率 |
| `ls_ret` | Float | 多空收益率 |
| `cumulative_ret` | Float | 累积收益率 |
| `drawdown` | Float | 回撤 |
| `turnover` | Float | 换手率 |

---

## 🔍 5. 参数网格结果格式

**文件命名**: `turnover_strategy_grid_{YYYYMMDD}_{HHMMSS}.csv`

### 🔧 核心列
```csv
strategy,param,param_name,ema_span,delta,top_n,bottom_n,cost_bps,n_days,ic_mean,ic_ir,ls_mean,ls_ann,ls_ir,max_drawdown,avg_turnover
E_combo,3_12,ema_span_delta,3,12,35,35,0.0003,515,0.0115,0.0577,0.0001,0.0158,0.0887,-0.1922,0.2438
```

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| `strategy` | String | 策略名称 |
| `top_n` | Integer | 做多股票数量 |
| `bottom_n` | Integer | 做空股票数量 |
| `cost_bps` | Float | 交易成本(基点) |
| `ic_mean` | Float | 平均IC |
| `ic_ir` | Float | 信息比率 |
| `ls_ir` | Float | 多空信息比率 |
| `max_drawdown` | Float | 最大回撤 |
| `avg_turnover` | Float | 平均换手率 |

---

## 📊 6. 因子数据格式

**文件命名**: `factors_data_redesigned.csv`

### 🔧 基础列
```csv
date,stock_code,ret_1d,ret_5d,volatility_20d,momentum_12d,rsi_14d
2020-01-02,000001,-0.0156,0.0234,0.0189,0.0456,0.45
```

**因子分类**:
- **收益率类**: `ret_1d`, `ret_5d`, `ret_20d`
- **动量类**: `momentum_12d`, `momentum_60d`
- **技术指标**: `rsi_14d`, `macd_signal`
- **波动率类**: `volatility_20d`, `volatility_60d`

---

## 📋 7. 股票池信息格式

**文件名**: `stock_universe.csv`
**位置**: 项目根目录

### 🔧 必需列
```csv
code,name,市值排名,总市值_亿元,流通市值_亿元,市盈率-动态,市净率,行业,样本标识,更新时间
601138,工业富联,1,12157.97,12157.18,50.18,8.02,未分类,扩展500只,2025-09-12 10:45:33
```

| 列名 | 数据类型 | 说明 |
|-----|---------|------|
| `code` | String | 6位股票代码 |
| `name` | String | 股票名称 |
| `市值排名` | Integer | 市值排名 |
| `行业` | String | 行业分类 |

---

## 🔧 文件处理最佳实践

### 📁 文件组织
```
data/
├── 000001.csv              # 股票价格
├── 000002.csv
├── ...
├── predictions_latest.csv   # 最新预测结果
├── volatility_regime_data_latest.csv
└── archive/                # 历史文件归档
    ├── old_predictions/
    └── old_results/
```

### 🔄 数据更新流程
1. **增量更新**: 新数据追加到现有文件
2. **全量替换**: 重新生成完整数据文件
3. **文件归档**: 定期将旧文件移至archive目录
4. **缓存清理**: 数据更新后清除相关缓存

### ⚡ 性能优化建议
- **分块读取**: 大文件使用pandas chunksize参数
- **数据类型优化**: 合理设置dtype减少内存占用
- **索引优化**: 对频繁查询的列建立索引
- **压缩存储**: 考虑使用parquet格式替代CSV

### ❗ 常见错误避免
- **编码问题**: 统一使用UTF-8编码
- **日期格式**: 统一使用YYYY-MM-DD格式
- **缺失值**: 明确缺失值的表示方式(NaN/NULL)
- **数据类型**: 确保数值列不包含字符串
- **文件大小**: 监控文件大小，及时分割或压缩

---

## 🛠️ 验证工具

平台提供以下验证功能确保数据质量:

### 📊 自动验证
- 文件格式检查
- 必需列验证
- 数据类型校验
- 缺失值检测

### 🔍 手动验证
- 数据统计摘要
- 异常值识别
- 时间序列连续性检查
- 跨文件一致性验证

使用这些标准格式可以确保平台的正常运行和最佳性能表现。
