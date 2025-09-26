# 📊 数据设置指南

## 🎯 数据文件说明

为了避免GitHub的100MB文件大小限制，本项目采用**智能数据分层策略**：

### ✅ **已包含的文件 (直接可用)**
- **股票价格数据**: 300+只股票的历史价格 (`000002.csv` - `688981.csv`)
- **预测结果**: 模型预测输出 (`predictions_*.csv`)
- **执行指标**: 策略表现数据 (`pipeline_execution_*.json`)
- **可视化图表**: 性能报告图表 (`*.png`, `*.pdf`)
- **汇总报告**: 各类分析汇总 (`*summary*.csv`)

### 🚫 **需要生成的大文件 (>50MB)**
这些文件被排除以避免GitHub限制，但可以轻松重新生成：
- `*factor_data_zscore_*.csv` - 因子数据文件
- `*regime_data_*.csv` - 波动率制度数据
- `*market_volatility_data_*.csv` - 市场波动率数据

## 🚀 **快速设置步骤**

### 1️⃣ 克隆项目
```bash
git clone https://github.com/1EchA/Volatility-Regime-Momentum.git
cd Volatility-Regime-Momentum
```

### 2️⃣ 安装环境
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 3️⃣ 生成大数据文件
```bash
# 方法1: 使用Streamlit界面 (推荐)
streamlit run app/streamlit_app.py
# 在侧边栏点击 "🚀 运行流水线"

# 方法2: 命令行生成
python3 run_full_pipeline.py --recompute-factors --recompute-regime
```

### 4️⃣ 验证完整性
启动应用后，在"总览"页面应该看到：
- ✅ 最新的策略表现指标
- ✅ 完整的数据文件列表
- ✅ 正常的图表显示

## 📈 **预期生成文件**

运行pipeline后，`data/` 目录将包含：
```
data/
├── 📋 因子数据 (大文件)
│   ├── simple_factor_data_zscore_YYYYMMDD_HHMMSS.csv
│   └── ... (多个版本)
├── 📊 制度数据 (大文件)
│   ├── volatility_regime_data_YYYYMMDD_HHMMSS.csv
│   └── market_volatility_data_YYYYMMDD_HHMMSS.csv
└── 📈 其他分析结果
    ├── predictions_*.csv
    ├── pipeline_execution_*.json
    └── performance_report_*.png
```

## 🔧 **常见问题**

**Q: 为什么有些数据文件没有上传？**
A: 因子数据和制度数据文件通常>100MB，超过GitHub限制。请运行pipeline重新生成。

**Q: 生成数据需要多长时间？**
A: 约3-10分钟，取决于您的机器性能。过程中会显示进度条。

**Q: 生成失败怎么办？**
A: 检查requirements.txt是否正确安装，或在GitHub Issues报告问题。

**Q: 可以使用已有的数据文件吗？**
A: 可以！如果您已有相应格式的数据文件，直接放入`data/`目录即可。

## 💡 **设计理念**

这种分层策略的优势：
- 🏃‍♂️ **快速克隆**: 避免下载几百MB的大文件
- 🔄 **灵活更新**: 用户可以生成最新的数据文件
- 📦 **完整体验**: 项目功能不受影响，体验完整
- 🛡️ **稳定性**: 避免GitHub的文件大小限制问题

---
💬 如有问题，欢迎在 [GitHub Issues](https://github.com/1EchA/Volatility-Regime-Momentum/issues) 反馈！