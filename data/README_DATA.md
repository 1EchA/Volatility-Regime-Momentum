数据目录使用说明（轻量化）

仓库不包含真实或大体量数据文件，避免仓库过大。请在本目录放置或生成所需的数据文件。

常见文件与命名规则
- 预测文件：predictions_YYYYMMDD_HHMMSS.csv（示例见同目录的示例文件）
- 执行层产物：pipeline_execution_*_{timeseries.csv,metrics.json}
- 性能报告：performance_report_*_{metrics.json,timeseries.csv,...}
- 网格结果：turnover_strategy_grid_*.csv
- 稳健性：robustness_summary_*.csv
- 归档目录：archive/（用于保存历史产物）

必要字段示例（预测文件）
- date：交易日期，YYYY-MM-DD
- stock_code：六位证券代码，左侧补零
- y_pred：模型预测的次日收益评分（可为任意实数）
- y_true：真实次日收益（若无，可留空或为0，仅影响IC/回测）
- industry（可选）：行业名称
- ind_rank_pct（可选）：行业内分位（0–1，越大越靠前）

注意
- 大文件不纳入版本管理（见项目根目录 .gitignore）。
- 可通过侧边栏“一键运行预测流水线”自动生成上述文件。
- 若无数据，UI将提示你先运行流水线或导入文件。

