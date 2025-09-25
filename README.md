# 🚀 A股量化策略研究平台

> 基于波动率制度的条件动量效应研究与策略测评工作台

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub_Actions-green.svg)](/.github/workflows/ci.yml)
[![Code Quality](https://img.shields.io/badge/code--quality-black%2Bisort%2Bflake8-brightgreen.svg)](#)

## 🎯 项目概述

本项目是一个**企业级量化研究平台**，专注于A股市场的条件动量效应研究。通过交互式Web界面，提供从数据处理、因子计算、模型训练到策略回测的完整工作流。

### 🏆 核心价值

- **学术研究**：首次在中国A股发现波动率条件动量效应，可发表学术论文
- **策略开发**：完整的量化策略开发与测评框架
- **风险控制**：严格防前瞻偏差，保守统计标准(1%显著性水平)
- **工业应用**：支持大规模数据处理，分块读取80MB+预测文件

## ⚡ 快速开始

可按下面“零基础三步上手”，一步一步照做即可跑通页面与策略。

### 零基础三步上手（3–10分钟）
- 第一步：安装环境
  - 安装 Python 3.12（或≥3.10），并创建虚拟环境
  - `pip install -r requirements.txt`
- 第二步：启动网页端
  - `streamlit run app/streamlit_app.py`
  - 浏览器打开 `http://localhost:8501`
- 第三步：生成一份“预测与回测”示例（可在侧边栏一键运行）
  - 侧边栏点击“🚀 运行流水线”（保持默认参数即可）
  - 完成后“总览”会自动刷新，个股页可选择最新 `predictions_*.csv`

提示：任何时候界面异常或列表不更新，可在侧边栏点击“🧹 清除缓存并刷新”。

### 1️⃣ 环境配置
```bash
# 克隆项目
git clone <repository-url>
cd 回归分析-a股分析

# 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2️⃣ 启动应用
```bash
# 启动Streamlit应用
streamlit run app/streamlit_app.py

# 浏览器访问: http://localhost:8501
```

### 3️⃣ 核心功能体验
1. **总揽** - 查看最新策略表现概览
2. **网格分析** - 参数优化与敏感性分析
3. **个股查询** - 单个股票深度分析
4. **执行画像** - 策略执行细节与换手率分析

## 📚 文档体系

- **[用户指南](docs/USER_GUIDE.md)** - 完整的使用手册和操作指南
- **[文件格式](docs/FILE_FORMATS.md)** - 数据文件格式规范和验证规则
- **[技术架构](docs/TECHNICAL.md)** - 系统架构、算法实现和性能优化
- **[研究方法](docs/RESEARCH.md)** - 研究假设、实验设计和方法论

## 🏗️ 系统架构

### 📁 目录结构
```
回归分析 a股分析/
├── 📱 app/                          # Streamlit应用
│   └── streamlit_app.py            # 主应用文件
├── 📊 data/                        # 数据文件
│   ├── archive/                    # 历史数据归档
│   ├── *.csv                      # 股票价格数据
│   └── *_results.json             # 分析结果
├── 📚 docs/                        # 项目文档
├── 🧪 tests/                       # 测试文件
├── 📄 *.py                        # 核心分析脚本
├── stock_universe.csv              # 股票池信息
└── README.md                       # 项目说明
```

### 🔧 核心模块
- **数据收集**: `data_collector.py` - 股票数据获取
- **因子计算**: 动量、波动率、技术指标因子
- **制度识别**: 基于GARCH的波动率制度划分
- **模型训练**: Fama-MacBeth两步法回归
- **策略回测**: 多空策略与换手率优化
- **可视化**: 交互式图表与热力图

## 🎛️ 功能特性

### 📈 **总揽模块**
- 策略表现实时监控
- 关键指标一览(IC、IR、年化收益、最大回撤)
- 热力图可视化参数敏感性
- 自动刷新最新分析结果

### 🔍 **网格分析**
- 多维参数网格搜索
- 换手率-收益权衡分析
- 成本敏感性测试
- 稳健性验证报告

### 🎯 **个股查询**
- 500+股票池选择器
- 单股票深度画像分析
- 价格走势与信号叠加
- 波动率制度时序展示

### ⚙️ **执行画像**
- 策略执行细节监控
- 换手率优化建议
- 交易成本影响分析
- 容量约束评估

## 📊 数据说明

### 📋 数据源
- **股票池**: 300+只A股，涵盖主要行业
- **时间跨度**: 2020-2024年日频数据
- **数据质量**: 前复权处理，缺失值填充
- **更新频率**: 支持增量更新

### 📈 关键指标（术语解释）
- IC（信息系数）：每天“预测分数 y_pred 与真实收益 y_true”的秩相关（Spearman）。数值越高，代表排序越准。
- IR（信息比率）：多空净收益序列的年化收益/年化波动，约等于日均值/日波动×√252，衡量稳健程度。
- 换手率（Turnover）：当天多端、空端的“持仓更换比例”之和，反映交易频率与成本敏感度。
- 最大回撤（Max Drawdown）：净值从任意峰值到之后最低谷的跌幅绝对值。
- bps（基点）：万分之一。成本“5 bps/边”表示每买或卖一次成本0.05%。
- TopN/BottomN：每天分数最高的前 N 只做多、最低的后 N 只做空。
- 行业内分位（Industry Percentile）：同一交易日同一行业内，按 y_pred 降序的百分位（0–1，越大越靠前）。
- 滞后带（Hysteresis Δ）：为降低换手，退出阈值放宽到 N+Δ；进入仍按 N，避免“来回抖动”。
- EMA 平滑：对 y_pred 做指数移动平均，降低噪音与频繁换手。
- 样本外（OOS）：滚动训练外用于检验的时间段，避免前瞻偏差。
- **IC (信息系数)**: 预测准确性指标
- **IR (信息比率)**: 风险调整后的IC
- **换手率**: 策略交易频率控制
- **最大回撤**: 风险控制指标

## 🛠️ 使用示例

### 基础分析流程
```bash
# 1. 数据收集
python data_collector.py

# 2. 运行完整分析流水线
python run_full_pipeline.py

# 3. 启动可视化界面 (默认端口8501)
streamlit run app/streamlit_app.py

# 4. 自定义端口启动
streamlit run app/streamlit_app.py --server.port 8502
```

## 🔣 文件与产物（你会看到什么）
- 预测文件：`data/predictions_YYYYMMDD_HHMMSS.csv`
  - 主要列：`date, stock_code, y_pred, y_true, industry, ind_rank_pct`
- 性能报告：`data/performance_report_*_{metrics.json,timeseries.csv,...}`
- 执行层产物：`data/pipeline_execution_*_{timeseries.csv,metrics.json}`
- 网格结果：`data/turnover_strategy_grid_*.csv`
- 归档目录：`data/archive/`（旧文件会被移入；界面可勾选“归档”列出）

## 🧪 测试与CI/CD

### 本地测试运行
```bash
# 运行所有测试
pytest tests/ -v

# 运行E2E测试
pytest tests/e2e/ -v --tb=short

# 生成测试覆盖率报告
pytest tests/ --cov=app --cov-report=html

# 代码格式检查
black --check --diff .
isort --check-only --diff .
flake8 . --count --statistics
```

### CI/CD流水线
项目配置了完整的GitHub Actions工作流：

- **代码质量**: Black + isort + flake8 + mypy
- **测试执行**: pytest + 覆盖率报告
- **安全扫描**: bandit + safety
- **容器构建**: Docker镜像构建与测试
- **文档生成**: 自动构建和部署文档

### Docker容器化部署
```bash
# 构建镜像
docker build -t quant-research:latest .

# 运行容器
docker run -d -p 8501:8501 \
  --name quant-app \
  -v $(pwd)/data:/app/data \
  quant-research:latest

# 健康检查
curl -f http://localhost:8501/_stcore/health
```

### 高级用法
```python
# 自定义参数网格搜索
python model_grid_search.py --top_n 30,35,40 --cost_bps 0.0003,0.0005

# 单独运行稳健性测试
python regime_model_sensitivity.py
```

## 🔧 配置说明

主要配置通过Streamlit界面完成，支持：
- 动态参数调整
- 文件路径选择
- 分析模式切换(简洁/高级)

## ❓ 常见问题（新手向）

1) 个股页提示“没有数据”？
- 选中的股票可能不在该预测文件里。勾选“仅可用”让列表只显示有数据的股票，或更换预测文件。

2) 行业分位不显示？
- 需要制度数据（含 industry 列）。请在侧边栏“运行流水线”中勾选“重算制度”，或运行 `python run_full_pipeline.py --recompute-regime`。

3) 总览时间范围和我看到的不一致？
- 总览支持“时间范围”筛选（全部/近90/180/365/自定义）。简洁模式默认近365天。

4) 页面卡住或列表不刷新？
- 点击侧边栏“🧹 清除缓存并刷新”；若仍异常，查看 `logs/app.log`。

5) 交易成本怎么填？
- 界面填 bps（基点）数值，如 5 表示每边 0.05%。脚本参数用十进制，如 `--cost-bps 0.0005`。

**Q: 首次运行报错找不到数据文件？**
A: 请确保data/目录包含股票CSV文件，或运行`python data_collector.py`获取数据

**Q: Streamlit页面加载缓慢？**
A: 大文件自动分块加载，首次可能较慢，后续有缓存机制

**Q: 如何调整分析参数？**
A: 在界面左侧边栏调整参数，或修改脚本中的默认值

**Q: 支持哪些Python版本？**
A: 推荐Python 3.12+，最低支持3.10

## 📈 研究成果

- **样本规模**: 300+只A股，11,732个观测值
- **研究方法**: Fama-MacBeth两步法，Newey-West标准误
- **核心发现**: 极高波动期年化收益可达150.30%
- **学术价值**: 首次在中国A股发现波动率条件动量效应

## 🤝 贡献指南

欢迎提交Issues和Pull Requests！

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看[LICENSE](LICENSE)文件了解详情

## 📞 联系方式

- 项目维护: [您的邮箱]
- 问题反馈: [GitHub Issues]
- 技术交流: [微信群/QQ群]

---

⭐ **如果这个项目对您有帮助，请给我们一个Star！**
