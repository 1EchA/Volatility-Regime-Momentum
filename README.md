# Volatility-Regime-Momentum (VRM)

> A股量化策略研究平台 - 基于波动率制度的条件动量效应研究

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目概述

**VRM (Volatility-Regime-Momentum)** 是一个专业的量化金融研究平台，致力于探索中国 A 股市场在不同**波动率制度**（正常/高波动/极高波动）下的动量效应差异。

![image-20260208140043219](/Users/xujia/Library/Application Support/typora-user-images/image-20260208140043219.png)

平台通过 GARCH 模型动态识别市场状态，并在不同制度下自适应地选择有效因子，构建具备**制度适应性**的量化投资策略。配合全新的 **Streamlit 交互式仪表盘**，用户可以从数据清洗、因子计算、模型训练到策略回测，实现全流程的可视化研究与实盘模拟。

### 核心特色

- **🔬 制度化归因**: 深度剖析策略收益来源，量化不同波动率制度对净值的贡献。
- **🖥️ 交互式沙箱**: 实时调整策略参数（TopN、换手率上限、滞后带宽度），即时对比基线与优化策略表现。
- **📊 专业级可视化**: 集成 Plotly 联动图表，支持净值/回撤同步缩放，个股 K 线深度悬浮分析（包含预测分、行业分位、制度背景）。
- **🛡️ 策略健康度监控**: 自动评估 IR、回撤、换手率等核心指标，提供红绿灯式的风险预警。
- **🚀 完整工程流**: 支持 500+ 股票并行处理，严格防范前瞻偏差，集成 Newey-West 统计校正。

## 🚀 快速开始

### 环境要求

- Python 3.12+ (推荐)
- 内存 ≥ 8GB (推荐 16GB)
- 磁盘空间 ≥ 5GB

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/1EchA/Volatility-Regime-Momentum.git
cd Volatility-Regime-Momentum

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 启动Web界面
streamlit run app/streamlit_app.py
```

访问浏览器 http://localhost:8501 即可开始使用。

### 首次使用指南

1.  **数据初始化**:
    - 进入 Web 界面，点击侧边栏 **"🚀 运行流水线"** 按钮。
    - 建议勾选 "重算因子" 和 "重算制度" 以生成初始数据（耗时约 3-5 分钟）。

2.  **探索核心功能**:
    - **📊 总览**: 查看多空年化、IR、最大回撤等核心指标，以及制度贡献环形图。
    - **⚡ 执行测试 (沙箱)**: 在 "沙箱演练" 区域调整参数，实时回测并对比策略效果。
    - **📈 个股查询**: 输入代码查看特定股票的预测分、行业排名及进出场时点。
    - **🔥 网格分析**: 探索 "成本 x 持仓数" 的参数敏感性曲面。

> 💡 **提示**: 因子数据文件通常较大 (>100MB)，建议通过流水线在本地生成，详见 [DATA_SETUP.md](DATA_SETUP.md)。

## 🏗️ 项目架构

```
Volatility-Regime-Momentum/
├── app/                           # Streamlit Web 应用
│   └── streamlit_app.py          # 交互式仪表板主程序
├── analysis/                      # 分析模块集
│   ├── execution_strategies.py   # 交易执行策略（滞后带、EMA平滑等）
│   ├── performance_reporter.py   # 绩效指标计算与报告生成
│   ├── robustness_validator.py   # 稳健性检验模块
│   └── ...
├── data/                         # 数据存储目录 (自动忽略大文件)
│   ├── *.csv                    # 股票日频行情数据
│   ├── predictions_*.csv        # 模型预测输出
│   └── pipeline_execution_*     # 回测执行结果
├── run_full_pipeline.py          # 全流程主控脚本
├── simple_factor_calculator.py   # 因子计算引擎
├── volatility_regime_analyzer.py # 波动率制度识别 (GARCH)
├── predictive_model.py           # 预测模型 (Fama-MacBeth)
└── requirements.txt              # 项目依赖清单
```

## 💡 核心功能模块

### 1. 动态制度识别
- **GARCH(1,1) 建模**: 精确捕捉市场波动率的时变特征。
- **三级制度划分**:
  - 🟢 **正常制度**: 波动率 < 75 分位
  - 🟡 **高波动制度**: 75 分位 ≤ 波动率 < 90 分位
  - 🔴 **极高波动制度**: 波动率 ≥ 90 分位

![image-20260208141413432](/Users/xujia/Library/Application Support/typora-user-images/image-20260208141413432.png)

### 2. 制度条件预测
- **条件 Fama-MacBeth 回归**: 在不同波动率制度下，分别训练因子权重。
- **自适应因子选择**: 正常市看重基本面与反转，高波市看重动量与情绪。
- **严格防前瞻**: 采用滚动窗口（Rolling Window）训练，确保无未来数据泄漏。

### 3. 策略执行与优化
- **多空组合**: 做多 Top N / 做空 Bottom N。
- **执行优化策略**:
  - **滞后带 (Hysteresis)**: 引入缓冲区，降低非必要换手。
  - **EMA 平滑**: 对预测信号进行指数平滑，减少噪音交易。
  - **换手率约束**: 强制限制每日最大换手比例。

![image-20260208140355965](/Users/xujia/Library/Application Support/typora-user-images/image-20260208140355965.png)

### 4. 全能可视化平台
- **策略总览**: 净值曲线与回撤曲线联动展示，制度收益贡献一目了然。
- **个股显微镜**: 悬浮查看每日预测分、行业内排名百分位及当时的市场制度。

![image-20260208140430462](/Users/xujia/Library/Application Support/typora-user-images/image-20260208140430462.png)

- **参数敏感性**: 热力图展示不同回测窗口和成本设定下的 IR 稳定性。

## 📊 关键指标说明

| 指标 (中文) | 指标 (英文) | 定义 | 参考标准 |
|---|---|---|---|
| **多空年化收益** | Annualised LS | 策略多空对冲后的年化收益率 | > 15% 优秀 |
| **信息比率** | IR | 超额收益均值 / 超额收益标准差 × √252 | > 2.0 优秀 |
| **IC 均值** | IC Mean | 预测值与下期收益的秩相关系数均值 | > 0.03 优秀 |
| **最大回撤** | Max Drawdown | 净值曲线从峰值回落的最大幅度 | < 15% 稳健 |
| **平均换手率** | Avg Turnover | 双边换手率的日均值 | < 50% 理想 |

## 🛠️ 进阶：命令行操作

除了 Web 界面，本项目完全支持命令行自动化运行：

```bash
# 1. 运行完整流水线 (指定参数)
python run_full_pipeline.py \
    --start-oos 2022-01-01 \
    --train-window 756 \
    --top-n 30 \
    --cost-bps 0.0005 \
    --execution-strategy hysteresis

# 2. 运行参数网格搜索
python analysis/cost_sensitivity_grid.py \
    --data-file data/regime_data_latest.csv \
    --top-ns 20,30,40

# 3. 运行稳健性验证 (多起点 x 多窗口)
python analysis/robustness_validator.py \
    --strategy hysteresis \
    --start-oos 2021-01-01,2022-01-01
```

## 🤝 贡献与反馈

欢迎提交 Pull Request 或 Issue！

- **Bug 反馈**: 请提供复现步骤和日志截图。
- **功能建议**: 欢迎提出新的因子思路或执行策略。

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

⭐ **喜欢这个项目？请点亮 Star 支持！**

📢 **免责声明**: 本项目仅供学术研究和量化策略开发参考，不构成任何投资建议。市场有风险，实盘需谨慎。