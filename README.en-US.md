# Volatility-Regime-Momentum (VRM)

> A-share Quantitative Strategy Research Platform - Research on Conditional Momentum Effects Based on Volatility Regimes

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Project Overview

**VRM (Volatility-Regime-Momentum)** is a professional quantitative finance research platform dedicated to exploring the differences in momentum effects within the Chinese A-share market under different **volatility regimes** (Normal / High Volatility / Extreme Volatility).

<img width="1920" height="899" alt="image" src="https://github.com/user-attachments/assets/b09f5a45-b29e-4bbf-9e5e-a10b3150f8c1" />

The platform dynamically identifies market states using GARCH models and adaptively selects effective factors under different regimes to build quantitative investment strategies with **regime adaptability**. Paired with a brand new **Streamlit interactive dashboard**, users can perform full-process visual research and live simulation—from data cleaning and factor calculation to model training and strategy backtesting.

### Core Features

- **🔬 Regime Attribution**: Deeply analyze the sources of strategy returns and quantify the contribution of different volatility regimes to the net asset value.
- **🖥️ Interactive Sandbox**: Adjust strategy parameters (TopN, turnover cap, hysteresis band width) in real-time to instantly compare baseline and optimized strategy performance.
- **📊 Professional Visualization**: Integrated Plotly linked charts supporting synchronized zooming of net value/drawdown, and deep hover analysis of individual stock K-lines (including prediction scores, industry percentiles, and regime background).
- **🛡️ Strategy Health Monitoring**: Automatically evaluate core metrics such as IR, Drawdown, and Turnover, providing traffic-light style risk warnings.
- **🚀 Complete Engineering Workflow**: Supports parallel processing of 500+ stocks, strictly prevents look-ahead bias, and integrates Newey-West statistical correction.

## 🚀 Quick Start

### Environment Requirements

- Python 3.12+ (Recommended)
- RAM ≥ 8GB (16GB Recommended)
- Disk Space ≥ 5GB

### Installation Steps

```bash
# 1. Clone the project
git clone https://github.com/1EchA/Volatility-Regime-Momentum.git
cd Volatility-Regime-Momentum

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Start Web interface
streamlit run app/streamlit_app.py
```

Visit http://localhost:8501 in your browser to begin.

### First-Time User Guide

1.  **Data Initialization**:
    - Enter the Web interface and click the **"🚀 Run Pipeline"** button in the sidebar.
    - It is recommended to check "Recalculate Factors" and "Recalculate Regimes" to generate initial data (takes approximately 3-5 minutes).

2.  **Explore Core Functions**:
    - **📊 Overview**: View core metrics such as Long-Short Annualized Return, IR, Maximum Drawdown, and the regime contribution donut chart.
    - **⚡ Execution Test (Sandbox)**: Adjust parameters in the "Sandbox Drill" area to backtest and compare strategy effects in real-time.
    - **📈 Stock Query**: Enter a stock code to view its prediction score, industry ranking, and entry/exit timing.
    - **🔥 Grid Analysis**: Explore the parameter sensitivity surface for "Cost x Position Size".

> 💡 **Tip**: Factor data files are usually large (>100MB); it is recommended to generate them locally via the pipeline. See [DATA_SETUP.md](DATA_SETUP.md) for details.

## 🏗️ Project Architecture

```
Volatility-Regime-Momentum/
├── app/                           # Streamlit Web Application
│   └── streamlit_app.py          # Main program for interactive dashboard
├── analysis/                      # Analysis module collection
│   ├── execution_strategies.py   # Trading execution strategies (Hysteresis, EMA smoothing, etc.)
│   ├── performance_reporter.py   # Performance metric calculation and report generation
│   ├── robustness_validator.py   # Robustness testing module
│   └── ...
├── data/                         # Data storage directory (Large files ignored automatically)
│   ├── *.csv                    # Daily stock price data
│   ├── predictions_*.csv        # Model prediction output
│   └── pipeline_execution_*     # Backtest execution results
├── run_full_pipeline.py          # Full-process control script
├── simple_factor_calculator.py   # Factor calculation engine
├── volatility_regime_analyzer.py # Volatility regime identification (GARCH)
├── predictive_model.py           # Predictive model (Fama-MacBeth)
└── requirements.txt              # Project dependency list
```

## 💡 Core Functional Modules

### 1. Dynamic Regime Identification
- **GARCH(1,1) Modeling**: Precisely capture the time-varying characteristics of market volatility.
- **Three-Tier Regime Classification**:
  - 🟢 **Normal Regime**: Volatility < 75th percentile
  - 🟡 **High Volatility Regime**: 75th percentile ≤ Volatility < 90th percentile
  - 🔴 **Extreme Volatility Regime**: Volatility ≥ 90th percentile

<img width="1432" height="369" alt="image" src="https://github.com/user-attachments/assets/cf01eab4-106d-4350-ad4f-cd991acd8570" />

### 2. Regime-Conditional Prediction
- **Conditional Fama-MacBeth Regression**: Train factor weights separately for different volatility regimes.
- **Adaptive Factor Selection**: Prioritize fundamentals and reversal in normal markets; prioritize momentum and sentiment in high-volatility markets.
- **Strict Anti-Lookahead**: Use rolling window training to ensure no future data leakage.

### 3. Strategy Execution & Optimization
- **Long-Short Portfolio**: Long Top N / Short Bottom N.
- **Execution Optimization Strategies**:
  - **Hysteresis**: Introduce a buffer zone to reduce unnecessary turnover.
  - **EMA Smoothing**: Apply exponential smoothing to prediction signals to reduce noise trading.
  - **Turnover Constraints**: Enforce a maximum daily turnover ratio.

<img width="1920" height="949" alt="image" src="https://github.com/user-attachments/assets/169b7a17-2488-4660-9e88-4272430a49cd" />

### 4. All-in-One Visualization Platform
- **Strategy Overview**: Linked display of net value and drawdown curves, with clear visibility of regime-based return contributions.
- **Stock Microscope**: Hover to view daily prediction scores, industry ranking percentiles, and the market regime at that time.

<img width="1398" height="701" alt="image" src="https://github.com/user-attachments/assets/8632d38f-323b-4a18-83b1-55b6243f477e" />

- **Parameter Sensitivity**: Heatmaps showing IR stability under different backtest windows and cost settings.

## 📊 Key Metrics Explanation

| Metric | Definition | Reference Standard |
|---|---|---|
| **Annualised LS** | Annualized return rate after long-short hedging | > 15% Excellent |
| **IR (Information Ratio)** | Mean excess return / Std dev of excess return × √252 | > 2.0 Excellent |
| **IC Mean** | Mean of rank correlation between predictions and next-period returns | > 0.03 Excellent |
| **Max Drawdown** | Maximum decline of the net value curve from its peak | < 15% Robust |
| **Avg Turnover** | Daily average of two-way turnover | < 50% Ideal |

## 🛠️ Advanced: Command Line Operations

In addition to the Web interface, this project fully supports automated command-line execution:

```bash
# 1. Run full pipeline (specifying parameters)
python run_full_pipeline.py \
    --start-oos 2022-01-01 \
    --train-window 756 \
    --top-n 30 \
    --cost-bps 0.0005 \
    --execution-strategy hysteresis

# 2. Run parameter grid search
python analysis/cost_sensitivity_grid.py \
    --data-file data/regime_data_latest.csv \
    --top-ns 20,30,40

# 3. Run robustness validation (Multiple start dates x Multiple windows)
python analysis/robustness_validator.py \
    --strategy hysteresis \
    --start-oos 2021-01-01,2022-01-01
```

## 🤝 Contribution & Feedback

Pull Requests and Issues are welcome!

- **Bug Feedback**: Please provide reproduction steps and screenshots of logs.
- **Feature Suggestions**: Feel free to propose new factor ideas or execution strategies.

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

⭐ **Like this project? Please give it a Star to support!**

📢 **Disclaimer**: This project is for academic research and quantitative strategy development reference only and does not constitute any investment advice. Markets carry risk; exercise caution in live trading.
