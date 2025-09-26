# Volatility-Regime-Momentum

> A股量化策略研究平台 - 基于波动率制度的条件动量效应研究

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目简介

本项目是一个量化金融研究平台，专注于中国A股市场在不同波动率制度下的动量效应研究。通过Streamlit交互式界面，提供从数据处理、因子计算、制度识别到策略回测与可视化分析的完整工作流。

### 核心功能

- 数据处理: 300+只A股历史数据（2020-2024），支持增量更新
- 因子计算: 动量、波动率、技术指标等多维因子
- 制度识别: 基于统计/机器学习的波动率制度分类（正常/高波动/极高波动）
- 策略回测: Fama-MacBeth回归预测，多空策略构建，换手率优化
- 可视化分析: 实时监控面板，参数网格搜索，个股深度分析

## ⚡ 快速开始

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/1EchA/Volatility-Regime-Momentum.git
cd Volatility-Regime-Momentum

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动应用
streamlit run app/streamlit_app.py
```

访问 http://localhost:8501 即可使用。

### 首次使用

1. 生成数据: 在侧边栏点击“🚀 运行流水线”，自动生成因子与制度数据
2. 查看结果: 在“总览”页面查看策略表现与净值曲线
3. 个股分析: 使用“个股查询”模块进行单票深度分析

> 📊 数据说明：大体量文件不会随仓库分发，请按下文“数据与设置”生成

## 📦 数据与设置（已合并）

为了避免GitHub 100MB限制，本项目采用“智能数据分层”策略：

- 已包含（可直接使用）
  - 股票价格示例：`data/000001.csv` 等
  - 预测结果：`data/predictions_*.csv`
  - 执行指标与报告：`pipeline_execution_*.json`、`performance_report_*.png/.csv`
  - 汇总/示例：`*summary*.csv`
- 需本地生成（通常 >50MB）
  - 因子数据：`*factor_data_zscore_*.csv`
  - 制度数据：`*regime_data_*.csv`、`*market_volatility_data_*.csv`

快速设置

```bash
# 1) 启动UI一键生成（推荐）
streamlit run app/streamlit_app.py  # 侧边栏点击“🚀 运行流水线”

# 2) 命令行生成（可选）
python3 run_full_pipeline.py --recompute-factors --recompute-regime
```

验证生成

- 在“总览”页应看到最新策略指标与图表
- `data/` 下出现对应的 `*_zscore_*.csv`、`regime_data_*.csv` 等大文件

期望目录结构（示意）

```
data/
├── simple_factor_data_zscore_YYYYMMDD_HHMMSS.csv
├── volatility_regime_data_YYYYMMDD_HHMMSS.csv
├── predictions_*.csv
├── pipeline_execution_*.json
└── performance_report_*.png
```

常见问题

- 为什么仓库里没有大文件？→ 超过限制，需本地生成
- 生成耗时多久？→ 约3–10分钟，取决于机器性能
- 生成失败？→ 确认依赖安装完整，或在 Issues 反馈
- 可否用自有数据？→ 可以，按 docs/FILE_FORMATS.md 放入 `data/`

（说明：原 DATA_SETUP.md 与 data/README_DATA.md 已合并至本节）

## 🏗️ 项目结构

```
Volatility-Regime-Momentum/
├── app/                     # Streamlit Web 应用
├── analysis/                # 研究/分析脚本
│   └── model_enhancement_runner.py  # 模型增强对比工具
├── data/                    # 数据与产物（大文件需本地生成）
├── docs/                    # 详细文档（用户手册/技术/研究/格式）
├── tests/                   # 测试文件
├── run_full_pipeline.py     # 主流水线脚本
└── requirements.txt         # 依赖包
```

## 🎯 主要功能模块

### 1. 总览面板
- 策略关键指标：IC、IR、年化收益、最大回撤
- 参数敏感性热力图
- 策略净值曲线

### 2. 网格分析
- 多维参数优化
- 换手率-收益权衡
- 成本敏感性测试

### 3. 个股查询
- 500+ 股票池
- 价格与信号叠加显示
- 波动率制度时序
- 行业内排名追踪

### 4. 执行分析
- 换手率分布
- 持仓重叠度
- 交易成本影响

## 📊 关键指标说明

| 指标 | 说明 |
|------|------|
| IC | 信息系数，预测与实际收益的秩相关 |
| IR | 信息比率，风险调整后收益（年化收益/年化波动） |
| 换手率 | 持仓更换比例，反映交易频率 |
| 最大回撤 | 净值峰值到谷值的最大跌幅 |
| TopN/BottomN | 做多最高分 N 只，做空最低分 N 只 |

## 🛠️ 高级用法

命令行

```bash
# 运行完整流水线
python run_full_pipeline.py --top-n 30 --bottom-n 30

# 参数网格搜索
python model_grid_search.py --top_n 30,35,40 --cost_bps 0.0003,0.0005

# 模型增强对比（弹性网/交互项/GBoost）
python analysis/model_enhancement_runner.py --start-oos 2022-01-01 --top-n 30 --bottom-n 30

# 数据收集
python data_collector.py
```

Docker 部署

```bash
docker build -t volatility-regime-momentum .
docker run -d -p 8501:8501 volatility-regime-momentum
```

## 📚 文档

- 用户手册：docs/USER_GUIDE.md
- 技术架构：docs/TECHNICAL.md
- 研究设计：docs/RESEARCH.md
- 文件格式：docs/FILE_FORMATS.md

## 📈 研究成果（概览）

- 样本规模：300+只A股，240,000+日度观测值
- 统计方法：Fama-MacBeth 回归，Newey-West 标准误校正
- 核心发现：不同波动率制度下动量效应存在显著差异
- 策略表现：IC=3.8%，IR=0.21，年化收益 22.98%

## ❓ 常见问题

- 大数据文件在哪里？→ 通过“🚀 运行流水线”自动生成
- 个股查询无数据？→ 确认已生成最新预测文件
- 页面不更新？→ 侧边栏点击“🧹 清除缓存并刷新”

## 📄 许可证

MIT License（详见 LICENSE）

## 📞 联系方式

- 维护者: pingtianhechuan@gmail.com
- Issues: https://github.com/1EchA/Volatility-Regime-Momentum/issues

---

⭐ 如果这个项目对您有帮助，请给一个 Star 支持！
