# Volatility-Regime-Momentum

> A股量化策略研究平台 - 基于波动率制度的条件动量效应研究

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目概述

本项目是一个专业的量化金融研究平台，聚焦于中国A股市场在不同波动率制度下的条件动量效应研究。平台通过识别市场波动率制度（正常/高波动/极高波动），发现并利用不同制度下动量因子的差异化表现，构建具有实战价值的量化投资策略。

### 核心特色

- **创新研究**: 首次在A股市场系统性验证波动率制度对动量效应的条件影响
- **完整工作流**: 从数据采集、因子构建、模型训练到策略回测的端到端解决方案
- **专业方法**: Fama-MacBeth回归、GARCH模型、Newey-West校正等学术级方法
- **交互界面**: Streamlit驱动的可视化平台，零代码操作体验
- **工业标准**: 支持500+股票并行处理，严格防前瞻偏差，完整的风险管理

## 🚀 快速开始

### 环境要求

- Python 3.12+ (最低支持3.10)
- 内存 ≥ 8GB (推荐16GB)
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

访问 http://localhost:8501 即可开始使用。

### 首次使用指南

1. **生成基础数据**
   - 点击侧边栏 "🚀 运行流水线" 按钮
   - 等待3-5分钟完成数据生成

2. **浏览核心功能**
   - **总览**: 查看策略整体表现和关键指标
   - **网格分析**: 探索最优参数组合
   - **个股查询**: 深度分析单只股票表现
   - **执行画像**: 评估换手率和交易成本

> 💡 **提示**: 大型数据文件(>50MB)需通过流水线生成，详见 [DATA_SETUP.md](DATA_SETUP.md)

## 🏗️ 项目架构

```
Volatility-Regime-Momentum/
├── app/                           # Streamlit Web应用
│   └── streamlit_app.py          # 交互式仪表板
├── analysis/                      # 分析模块集
│   ├── execution_strategies.py   # 执行策略实现
│   ├── performance_reporter.py   # 性能报告生成
│   └── model_enhancement_runner.py # 增强模型对比
├── data/                         # 数据存储目录
│   ├── *.csv                    # 股票价格数据
│   ├── predictions_*.csv        # 模型预测结果
│   └── performance_report_*     # 策略表现报告
├── docs/                         # 详细文档
├── tests/                        # 测试套件
├── run_full_pipeline.py          # 主流水线脚本
├── data_collector.py             # 数据采集模块
├── simple_factor_calculator.py   # 因子计算引擎
├── volatility_regime_analyzer.py # 制度识别模块
├── predictive_model.py           # 预测模型实现
└── requirements.txt              # 项目依赖
```

## 💡 核心功能

### 1. 数据处理与因子工程
- **数据覆盖**: 300+只A股主要标的，2020-2024年日频数据
- **因子体系**:
  - 动量因子: 1/5/10/20日收益率
  - 波动率因子: 历史波动率、GARCH波动率
  - 技术因子: 成交量、换手率、相对强弱
  - 基本面因子: PE、PB、市值因子
- **标准化方法**: Z-Score标准化、百分位排序、行业中性化

### 2. 波动率制度识别
- **GARCH建模**: 捕捉时变波动率特征
- **制度分类**:
  - 正常制度: 波动率 < 75分位
  - 高波动制度: 75分位 ≤ 波动率 < 90分位
  - 极高波动制度: 波动率 ≥ 90分位
- **动态更新**: 滚动窗口实时更新制度状态

### 3. 条件预测模型
- **Fama-MacBeth回归**: 两步横截面回归方法
- **制度条件建模**: 不同制度下独立选择有效因子
- **防前瞻偏差**: 严格的样本外测试，滚动窗口训练
- **统计校正**: Newey-West标准误，控制自相关和异方差

### 4. 策略构建与优化
- **多空策略**: TopN做多 + BottomN做空
- **执行策略**:
  - 基线策略: 每日完全再平衡
  - 滞后策略: 引入滞后带降低换手
  - EMA平滑: 信号指数平滑降噪
  - 低频策略: 按周/月调仓
- **成本优化**: 交易成本敏感性分析，换手率约束

### 5. 可视化分析平台

#### 总览仪表板
- 关键指标卡片: IC、IR、年化收益、最大回撤
- 策略净值曲线与回撤分析
- 参数敏感性热力图
- 制度贡献分解

#### 网格分析
- 多维参数空间搜索
- 换手率-收益权衡曲线
- 成本敏感性3D曲面
- 稳健性测试报告

#### 个股分析
- 价格走势与预测信号叠加
- 波动率制度背景着色
- 行业内排名时序图
- 进出场事件标注

#### 执行分析
- 换手率分布直方图
- 持仓重叠度热力图
- 交易成本影响瀑布图
- 容量约束评估

## 📊 关键指标说明

| 指标 | 定义 | 计算方法 | 参考值 |
|------|------|----------|--------|
| **IC** | 信息系数 | Spearman(预测值, 实际收益) | > 3% 优秀 |
| **IR** | 信息比率 | IC均值 / IC标准差 × √252 | > 0.5 优秀 |
| **年化收益** | 策略年化收益率 | (1+日均收益)^252 - 1 | > 20% 优秀 |
| **最大回撤** | 最大净值回撤 | max(峰值-谷值)/峰值 | < 20% 可接受 |
| **换手率** | 日均换手率 | (买入量+卖出量)/2/总持仓 | < 50% 理想 |
| **夏普比率** | 风险调整收益 | (年化收益-无风险)/年化波动 | > 1.0 优秀 |

## 🛠️ 高级用法

### 命令行操作

```bash
# 完整流水线运行
python run_full_pipeline.py \
    --start-oos 2022-01-01 \
    --train-window 252 \
    --top-n 30 \
    --bottom-n 30 \
    --execution-strategy hysteresis

# 参数网格搜索
python model_grid_search.py \
    --top_n 20,30,40 \
    --cost_bps 0.0003,0.0005,0.001

# 制度敏感性分析
python regime_model_sensitivity.py \
    --regime 高波动 \
    --factor momentum_5d,volatility_20d

# 增强模型对比
python analysis/model_enhancement_runner.py \
    --models baseline,elasticnet,gboost
```

### Docker容器部署

```bash
# 构建镜像
docker build -t vrm:latest .

# 运行容器
docker run -d \
    -p 8501:8501 \
    -v $(pwd)/data:/app/data \
    --name vrm-app \
    vrm:latest

# 查看日志
docker logs -f vrm-app
```

### Python API调用

```python
from simple_factor_calculator import SimpleFactorCalculator
from volatility_regime_analyzer import VolatilityRegimeAnalyzer
from predictive_model import PredictiveModel

# 初始化模块
calculator = SimpleFactorCalculator(universe_file='stock_universe.csv')
regime_analyzer = VolatilityRegimeAnalyzer()
model = PredictiveModel()

# 计算因子
factors_df = calculator.calculate_all_factors()

# 识别制度
regimes = regime_analyzer.identify_regimes(market_data)

# 生成预测
predictions = model.predict(factors_df, regimes)
```

## 📈 研究成果

### 实证发现
- **样本规模**: 300+只A股，240,000+日度观测值
- **研究周期**: 2020-2024年，覆盖多个市场周期
- **核心发现**:
  - 正常制度: 动量因子IC约2.5%，稳定但不显著
  - 高波动制度: 动量因子IC提升至4.2%，显著性增强
  - 极高波动制度: 动量因子IC达6.8%，高度显著

### 策略表现
- **信息系数(IC)**: 3.83%，日度预测准确性良好
- **信息比率(IR)**: 0.214，风险调整后表现稳健
- **年化收益**: 22.98%，显著跑赢市场基准
- **最大回撤**: -20.89%，风险控制在可接受范围
- **平均换手率**: 47.61%，交易成本可控

### 学术贡献
- 首次系统性验证A股市场波动率制度对动量效应的条件影响
- 提供了完整的实证框架和开源实现
- 可作为学术论文研究基础

## 📚 详细文档

- **[用户指南](docs/USER_GUIDE.md)** - 完整操作手册
- **[技术架构](docs/TECHNICAL.md)** - 系统设计与实现细节
- **[研究方法](docs/RESEARCH.md)** - 理论基础与研究设计
- **[文件格式](docs/FILE_FORMATS.md)** - 数据规范说明
- **[数据设置](DATA_SETUP.md)** - 大文件生成指南

## ❓ 常见问题

**Q: 为什么有些大文件不在仓库中？**
A: 因子数据和制度数据文件通常>100MB，超过GitHub限制。运行"🚀 运行流水线"即可本地生成。

**Q: 个股查询显示"无数据"？**
A: 确保已运行流水线生成预测文件，或检查该股票是否在股票池中。

**Q: 策略表现与预期不符？**
A: 检查数据时间范围、参数设置是否合理，可参考网格分析寻找最优参数。

**Q: 如何添加自定义因子？**
A: 在`simple_factor_calculator.py`中按现有格式添加因子计算逻辑。

**Q: 支持实时交易吗？**
A: 当前版本专注于研究回测，实盘交易需要额外的风控和执行模块。

## 🤝 贡献指南

欢迎各类贡献，包括但不限于：
- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档完善
- 🧪 测试用例
- 💡 研究想法

提交流程：
1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 📞 联系方式

- **项目维护者**: pingtianhechuan@gmail.com
- **问题反馈**: [GitHub Issues](https://github.com/1EchA/Volatility-Regime-Momentum/issues)
- **技术讨论**: 欢迎在Issues中发起讨论

## 🙏 致谢

感谢以下开源项目的支持：
- Streamlit - 交互式Web应用框架
- Pandas/NumPy - 数据处理基础库
- Plotly - 可视化图表库
- Scikit-learn/Statsmodels - 机器学习与统计建模

---

⭐ **如果这个项目对您有帮助，请给一个Star支持！**

📢 **声明**: 本项目仅供学术研究和教育用途，不构成投资建议。市场有风险，投资需谨慎。