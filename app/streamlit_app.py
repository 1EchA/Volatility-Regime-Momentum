"""
Streamlit dashboard scaffold for the volatility-regime momentum project.

Features (trial version):
- Auto-detects the latest performance report output under data/ and visualises
  the cumulative long-short equity curve and drawdown.
- Displays summary metrics and regime contributions.
- Provides a sidebar to upload an alternative predictions CSV for quick
  inspection (head of the file).

Can be launched with: `streamlit run app/streamlit_app.py`
"""

from __future__ import annotations

import io
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

# 配置日志系统
# 确保logs目录存在
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(logs_dir / "app.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
try:
    from plotly import graph_objects as go
except Exception:
    go = None
try:
    from plotly.subplots import make_subplots
except Exception:
    make_subplots = None
try:
    from streamlit_plotly_events import plotly_events

    HAS_PLOTLY_EVENTS = True
except Exception:
    HAS_PLOTLY_EVENTS = False
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.execution_strategies import (
    baseline_daily,
    compute_ic_series_with_score,
    ema_hysteresis_combo,
    ema_smoothed,
    hysteresis_bands,
)
from analysis.performance_reporter import (
    compute_ic_series,
    compute_portfolio_timeseries,
    compute_summary_metrics,
)
from predictive_model import find_latest_regime_file

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def list_reports() -> list[Path]:
    metrics = sorted(
        _glob_exclude_archive("performance_report_*_metrics.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return metrics


def list_cost_grids() -> list[Path]:
    grids = sorted(
        _glob_exclude_archive("cost_sensitivity_grid_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return grids


def list_turnover_grids() -> list[Path]:
    grids = sorted(
        _glob_exclude_archive("turnover_strategy_grid_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return grids


def list_predictions(include_archive: bool = False) -> list[Path]:
    """列出预测文件。
    - 默认排除 data/archive 下的历史文件；
    - 当 include_archive=True 时，同时包含 data/archive 中的文件，
      便于在仅存归档的情况下仍可选择。
    """
    if include_archive and (DATA_DIR / "archive").exists():
        files = list(DATA_DIR.glob("predictions_*.csv"))
        files += list((DATA_DIR / "archive").glob("predictions_*.csv"))
        preds = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        preds = sorted(
            _glob_exclude_archive("predictions_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    return preds


def validate_predictions_file(path: str) -> tuple[bool, str]:
    """校验预测文件格式和必需列"""
    try:
        df = pd.read_csv(path, nrows=5)  # 只读前5行进行快速校验
        required_columns = ["date", "stock_code", "y_pred"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return False, f"缺少必需列: {missing_columns}"

        # 检查数据类型
        try:
            pd.to_datetime(df["date"])
            pd.to_numeric(df["y_pred"], errors="raise")
        except Exception as e:
            return False, f"数据格式错误: {str(e)}"

        return True, "文件格式正确"
    except Exception as e:
        return False, f"文件读取失败: {str(e)}"


@st.cache_data
def load_predictions_df(path: str) -> pd.DataFrame:
    # 先校验文件格式
    is_valid, error_msg = validate_predictions_file(path)
    if not is_valid:
        st.error(f"预测文件格式错误: {error_msg}")
        return pd.DataFrame()

    try:
        logger.info(f"Loading predictions file: {path}")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
        # 强制为数值，防止被当作字符串导致显示异常
        for col in ["y_pred", "y_true"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logger.info(f"Successfully loaded {len(df)} prediction records")
        return df
    except Exception as e:
        logger.error(f"Failed to load predictions file {path}: {str(e)}", exc_info=True)
        st.error(f"加载预测文件失败: {e}")
        return pd.DataFrame()


@st.cache_data
def load_csv_cached(path: str) -> pd.DataFrame:
    try:
        logger.info(f"Loading CSV file: {path}")
        df = pd.read_csv(path)
        logger.info(f"Successfully loaded CSV with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV file {path}: {str(e)}", exc_info=True)
        st.error(f"加载CSV文件失败 {path}: {e}")
        return pd.DataFrame()


def _glob_exclude_archive(pattern: str, exclude_dirs: list = None) -> list:
    """Glob files excluding specified directories (default: archive)"""
    if exclude_dirs is None:
        exclude_dirs = ["archive"]

    all_files = list(DATA_DIR.glob(pattern))
    # Filter out files in excluded directories
    filtered_files = []
    for file_path in all_files:
        # Check if any parent directory is in exclude_dirs
        if not any(part in exclude_dirs for part in file_path.parts):
            filtered_files.append(file_path)

    return filtered_files


def _latest(path_glob: str) -> Path | None:
    cands = sorted(_glob_exclude_archive(path_glob), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def render_price_signal_chart_new(
    one: pd.DataFrame,
    price_df: pd.DataFrame | None,
    regime_df: pd.DataFrame | None,
    show_regime: bool = True,
    show_score: bool = True,
    show_buy: bool = True,
    show_short: bool = True,
    show_close: bool = True,
) -> "go.Figure":
    """Robust Plotly figure (price + score + events + regime background)."""
    use_secondary = make_subplots is not None and go is not None
    fig = (
        make_subplots(specs=[[{"secondary_y": True}]]) if use_secondary else go.Figure()
    )

    # Enrich data for hover
    hover_cols = ["date"]
    if "y_pred" in one.columns:
        hover_cols.append("y_pred")
    if "ind_rank_pct" in one.columns:
        hover_cols.append("ind_rank_pct")
    if "regime" in one.columns:
        hover_cols.append("regime")

    # Price trace
    merged = None
    if price_df is not None and not price_df.empty:
        # Merge price with signals for unified hover data
        cols_to_merge = list(set(hover_cols) & set(one.columns))
        # Avoid duplicate date column
        if "date" in cols_to_merge:
            cols_to_merge.remove("date")

        if set(["open", "high", "low", "close"]).issubset(price_df.columns):
            merged = price_df[["date", "open", "high", "low", "close"]].merge(
                one[["date"] + cols_to_merge], on="date", how="left"
            )
            merged = merged.dropna(subset=["close"])

            # Pre-format hover strings for Candlestick (avoids hovertemplate compatibility issues)
            h_texts = []
            for _, r in merged.iterrows():
                score_str = f"{r['y_pred']:.4f}" if pd.notna(r.get("y_pred")) else "N/A"
                rank_str = (
                    f"{r['ind_rank_pct']:.1%}"
                    if pd.notna(r.get("ind_rank_pct"))
                    else "N/A"
                )
                txt = (
                    f"<b>{r['date'].strftime('%Y-%m-%d')}</b><br>"
                    f"开盘: {r['open']:.2f}<br>最高: {r['high']:.2f}<br>"
                    f"最低: {r['low']:.2f}<br>收盘: {r['close']:.2f}<br>"
                    f"预测分: {score_str}<br>"
                    f"行业分位: {rank_str}<br>"
                    f"制度: {r.get('regime', '-')}"
                )
                h_texts.append(txt)

            fig.add_trace(
                go.Candlestick(
                    x=merged["date"],
                    open=merged["open"],
                    high=merged["high"],
                    low=merged["low"],
                    close=merged["close"],
                    name="Price",
                    text=h_texts,
                    hoverinfo="text",
                ),
                secondary_y=False,
            )

        elif "close" in price_df.columns:
            merged = price_df[["date", "close"]].merge(
                one[["date"] + cols_to_merge], on="date", how="left"
            )
            merged = merged.dropna(subset=["close"])

            h_texts = []
            for _, r in merged.iterrows():
                score_str = f"{r['y_pred']:.4f}" if pd.notna(r.get("y_pred")) else "N/A"
                rank_str = (
                    f"{r['ind_rank_pct']:.1%}"
                    if pd.notna(r.get("ind_rank_pct"))
                    else "N/A"
                )
                txt = (
                    f"<b>{r['date'].strftime('%Y-%m-%d')}</b><br>"
                    f"收盘价: {r['close']:.2f}<br>"
                    f"预测分: {score_str}<br>"
                    f"行业分位: {rank_str}<br>"
                    f"制度: {r.get('regime', '-')}"
                )
                h_texts.append(txt)

            fig.add_trace(
                go.Scatter(
                    x=merged["date"],
                    y=merged["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="#1f77b4"),
                    text=h_texts,
                    hoverinfo="text",
                ),
                secondary_y=False,
            )

    # score line
    if show_score and "y_pred" in one.columns and not one["y_pred"].isna().all():
        if use_secondary:
            fig.add_trace(
                go.Scatter(
                    x=one["date"],
                    y=one["y_pred"],
                    mode="lines",
                    name="Score",
                    line=dict(color="#ff7f0e", width=1),
                    opacity=0.6,
                ),
                secondary_y=True,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=one["date"],
                    y=one["y_pred"],
                    mode="lines",
                    name="Score",
                    line=dict(color="#ff7f0e", width=1),
                    opacity=0.6,
                )
            )

    # Events
    evt = one[["date", "in_long", "in_short"]].copy()
    evt[["in_long", "in_short"]] = (
        evt[["in_long", "in_short"]].fillna(False).astype(bool)
    )
    prev = evt[["in_long", "in_short"]].shift(1).fillna(False)
    evt["long_open"] = (~prev["in_long"]) & (evt["in_long"])
    evt["short_open"] = (~prev["in_short"]) & (evt["in_short"])
    evt["long_close"] = (prev["in_long"]) & (~evt["in_long"])
    evt["short_close"] = (prev["in_short"]) & (~evt["in_short"])

    hover_event = "<b>%{x|%Y-%m-%d}</b><br>价格: %{y:.2f}<br>事件: %{customdata}"

    if merged is not None and not merged.empty:
        # Use merged to ensure we have prices for events
        # We need to re-merge events to merged to get prices
        marks_full = merged[["date", "close"]].merge(evt, on="date", how="inner")

        if show_buy:
            d = marks_full[marks_full["long_open"]]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=d["date"],
                        y=d["close"],
                        mode="markers",
                        name="买入(多开)",
                        marker=dict(symbol="triangle-up", color="#2ecc71", size=10),
                        customdata=["买入"] * len(d),
                        hovertemplate=hover_event,
                    ),
                    secondary_y=False,
                )
        if show_short:
            d = marks_full[marks_full["short_open"]]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=d["date"],
                        y=d["close"],
                        mode="markers",
                        name="做空(空开)",
                        marker=dict(symbol="triangle-down", color="#e74c3c", size=10),
                        customdata=["做空"] * len(d),
                        hovertemplate=hover_event,
                    ),
                    secondary_y=False,
                )
        if show_close:
            d = marks_full[marks_full["long_close"]]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=d["date"],
                        y=d["close"],
                        mode="markers",
                        name="多平仓",
                        marker=dict(
                            symbol="x", color="#2ecc71", size=8, line=dict(width=2)
                        ),
                        customdata=["清仓(多)"] * len(d),
                        hovertemplate=hover_event,
                    ),
                    secondary_y=False,
                )
            d = marks_full[marks_full["short_close"]]
            if not d.empty:
                fig.add_trace(
                    go.Scatter(
                        x=d["date"],
                        y=d["close"],
                        mode="markers",
                        name="空平仓",
                        marker=dict(
                            symbol="x", color="#e74c3c", size=8, line=dict(width=2)
                        ),
                        customdata=["清仓(空)"] * len(d),
                        hovertemplate=hover_event,
                    ),
                    secondary_y=False,
                )

    # Regime background
    if show_regime:
        try:
            reg_series = (
                one[["date", "regime"]].dropna()
                if "regime" in one.columns
                else pd.DataFrame()
            )
            if reg_series.empty and regime_df is not None:
                # use daily market regime sliced to price range
                if merged is not None and not merged.empty:
                    dmin, dmax = merged["date"].min(), merged["date"].max()
                else:
                    dmin, dmax = one["date"].min(), one["date"].max()
                reg_series = (
                    regime_df[
                        (regime_df["date"] >= dmin) & (regime_df["date"] <= dmax)
                    ][["date", "regime"]]
                    .dropna()
                    .drop_duplicates("date")
                )
            if not reg_series.empty:
                regimes = reg_series["regime"].astype(str).tolist()
                dates = reg_series["date"].tolist()
                starts = [dates[0]]
                labels = []
                for i in range(1, len(regimes)):
                    if regimes[i] != regimes[i - 1]:
                        starts.append(dates[i])
                        labels.append(regimes[i - 1])
                labels.append(regimes[-1])
                ends = dates[1:] + [dates[-1]]
                # 更柔和的背景色，避免遮挡主图
                cmap = {
                    "正常": "rgba(46,204,113,0.08)",
                    "高波动": "rgba(241,196,15,0.06)",
                    "极高波动": "rgba(231,76,60,0.07)",
                }
                for s, e, lab in zip(starts, ends, labels):
                    fig.add_vrect(
                        x0=pd.to_datetime(s),
                        x1=pd.to_datetime(e),
                        y0=0,
                        y1=1,
                        yref="paper",
                        fillcolor=cmap.get(lab, "rgba(127,127,127,0.12)"),
                        opacity=1.0,
                        layer="below",
                        line_width=0,
                    )
        except Exception:
            pass

    fig.update_layout(
        height=500, margin=dict(l=40, r=40, t=30, b=40), hovermode="x unified"
    )
    fig.update_xaxes(title_text="Date")
    if use_secondary:
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Score", secondary_y=True, showgrid=False)
    else:
        fig.update_yaxes(title_text="Price/Score")
    return fig


def time_range_selector(ts: pd.DataFrame, key_prefix: str = "ov") -> pd.DataFrame:
    """在总览中提供时间范围筛选，返回筛选后的时序数据。
    key_prefix 用于区分不同上下文下的控件状态键。
    """
    if ts is None or ts.empty or "date" not in ts.columns:
        return ts
    # 统一为 datetime
    if not np.issubdtype(ts["date"].dtype, np.datetime64):
        ts = ts.copy()
        ts["date"] = pd.to_datetime(ts["date"])
    opts = ["全部", "近90天", "近180天", "近365天", "自定义"]
    default_idx = 3 if st.session_state.get("cfg_simple_mode", True) else 0
    sel = st.selectbox("时间范围", opts, index=default_idx, key=f"{key_prefix}_range")
    if sel == "自定义":
        c1, c2 = st.columns(2)
        dmin = pd.to_datetime(ts["date"].min()).date()
        dmax = pd.to_datetime(ts["date"].max()).date()
        start = c1.date_input("开始日期", value=dmin, key=f"{key_prefix}_from")
        end = c2.date_input("结束日期", value=dmax, key=f"{key_prefix}_to")
        mask = (ts["date"].dt.date >= start) & (ts["date"].dt.date <= end)
        out = ts.loc[mask].copy()
        return out
    days = None
    if sel == "近90天":
        days = 90
    elif sel == "近180天":
        days = 180
    elif sel == "近365天":
        days = 365
    if days is None:
        return ts
    cutoff = ts["date"].max() - pd.Timedelta(days=days)
    return ts[ts["date"] >= cutoff].copy()


@st.cache_data
def load_stock_universe() -> pd.DataFrame:
    """加载股票池信息，返回包含代码、名称等信息的DataFrame"""
    stock_file = DATA_DIR.parent / "stock_universe.csv"
    try:
        if stock_file.exists():
            logger.info(f"Loading stock universe from: {stock_file}")
            df = pd.read_csv(stock_file)
            df["code"] = df["code"].astype(str).str.zfill(6)
            logger.info(f"Successfully loaded {len(df)} stocks in universe")
            return df
        else:
            logger.warning("Stock universe file not found, using empty dataset")
            st.warning("股票池文件不存在，将使用空数据集")
            return pd.DataFrame(columns=["code", "name"])
    except Exception as e:
        logger.error(f"Failed to load stock universe: {str(e)}", exc_info=True)
        st.error(f"加载股票池失败: {e}")
        return pd.DataFrame(columns=["code", "name"])


@st.cache_data(show_spinner=False)
def derive_stock_series_from_predictions(
    path: str,
    code: str,
    top_n: int,
    bottom_n: int,
    regime_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """内存友好的单票提取：按日期分块读取预测文件，计算该票是否进入TopN/BottomN，
    并在可用时合并制度与行业，计算行业分位。

    为避免内存压力，仅对目标股票计算行业分位：在逐日分组后，若当前股票有行业信息，
    则在当日同一行业内按 y_pred 计算分位。
    """
    code = str(code).zfill(6)
    cols = ["date", "stock_code", "y_pred", "y_true"]
    rows = []
    # 预处理 regime_df
    if regime_df is not None and not regime_df.empty:
        regime_df_local = regime_df[["date", "stock_code", "industry", "regime"]].copy()
        regime_df_local["date"] = pd.to_datetime(regime_df_local["date"])
        regime_df_local["stock_code"] = (
            regime_df_local["stock_code"].astype(str).str.zfill(6)
        )
    else:
        regime_df_local = None

    for chunk in pd.read_csv(path, usecols=lambda c: c in cols, chunksize=300000):
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["stock_code"] = chunk["stock_code"].astype(str).str.zfill(6)
        if regime_df_local is not None:
            chunk = chunk.merge(regime_df_local, on=["date", "stock_code"], how="left")
        for dt, g in chunk.groupby("date"):
            g_sorted = g.sort_values("y_pred", ascending=False)
            top_set = set(g_sorted.head(top_n)["stock_code"])
            bot_set = set(g_sorted.tail(bottom_n)["stock_code"])
            row = g[g["stock_code"] == code]
            if not row.empty:
                r = row.iloc[0]
                out_row = {
                    "date": dt,
                    "stock_code": code,
                    "y_pred": float(r["y_pred"]),
                    "y_true": float(r.get("y_true", float("nan"))),
                    "in_long": r["stock_code"] in top_set,
                    "in_short": r["stock_code"] in bot_set,
                }
                if "industry" in r and pd.notna(r["industry"]):
                    ind = r["industry"]
                    g_ind = g_sorted[g_sorted["industry"] == ind]
                    if not g_ind.empty:
                        # 使用降序排名计算百分位（越高越靠前）
                        g_ind = g_ind.reset_index(drop=True)
                        # rank_position 从 1 开始
                        try:
                            rank_position = (
                                int(
                                    g_ind.index[g_ind["stock_code"] == code].tolist()[0]
                                )
                                + 1
                            )
                            ind_count = len(g_ind)
                            out_row["industry"] = ind
                            out_row["regime"] = r.get("regime")
                            out_row["ind_rank_pct"] = 1.0 - (rank_position - 1) / max(
                                1, ind_count
                            )
                        except Exception:
                            pass
                rows.append(out_row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "stock_code",
                "y_pred",
                "y_true",
                "in_long",
                "in_short",
                "industry",
                "regime",
                "ind_rank_pct",
            ]
        )
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def list_codes_in_predictions(path: str) -> list[str]:
    """从预测文件里快速提取可用股票代码（去重、6位补零）。"""
    codes: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["stock_code"], chunksize=500000):
        s = chunk["stock_code"].astype(str).str.zfill(6)
        codes.update(s.unique().tolist())
    return sorted(codes)


def load_report(prefix: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    metrics_path = prefix
    base_name = metrics_path.name.replace("_metrics.json", "")
    ts_path = DATA_DIR / f"{base_name}_timeseries.csv"
    regimes_path = DATA_DIR / f"{base_name}_regime_contrib.csv"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)["metrics"]
    ts = (
        pd.read_csv(ts_path, parse_dates=["date"])
        if ts_path.exists()
        else pd.DataFrame()
    )
    regimes = pd.read_csv(regimes_path) if regimes_path.exists() else pd.DataFrame()
    return metrics, ts, regimes


st.set_page_config(page_title="Volatility Regime Momentum", layout="wide")

# 悬浮提示样式与小图标（全局复用）
st.markdown(
    """
    <style>
    .tip {display:inline-block; background:#eef; color:#555; border-radius:50%; width:16px; height:16px; text-align:center; font-size:12px; line-height:16px; margin-left:6px; cursor:help; position:relative;}
    .tip:hover::after { content: attr(data-tip); position:absolute; z-index:10; left: 20px; top: -2px; background:#333; color:#fff; padding:6px 8px; border-radius:4px; white-space:pre-wrap; max-width:360px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def info_icon(text: str):
    st.markdown(f'<span class="tip" data-tip="{text}">i</span>', unsafe_allow_html=True)


st.title("波动率制度动量策略 — 量化研究仪表盘")
tab_overview, tab_grid, tab_exec, tab_robust, tab_stock, tab_pack = st.tabs(
    ["📊 总览", "🔥 网格与曲面", "⚡ 执行测试", "🔒 稳健性", "📈 个股", "📦 打包"]
)

with tab_overview:
    reports = list_reports()

    # Helper to render metrics
    def render_metrics_row(m):
        cols = st.columns(4)
        cols[0].metric("多空年化收益", f"{m.get('ls_ann', 0):.2%}")
        cols[1].metric("信息比率 (IR)", f"{m.get('ls_ir', 0):.3f}")
        cols[2].metric("IC均值", f"{m.get('ic_mean', 0):.3f}")
        cols[3].metric("最大回撤", f"{m.get('max_drawdown', 0):.2%}")

    # Helper to render health monitor
    def render_health(m):
        ir = m.get("ls_ir", 0)
        dd = m.get("max_drawdown", 0)
        to = m.get("avg_turnover", 0)
        if ir > 0.5 and dd > -0.2 and to < 0.8:
            st.success("🟢 策略健康度：良好 (Robust)")
        elif ir < 0.2 or dd < -0.3:
            st.error("🔴 策略健康度：高风险 (High Risk)")
        else:
            st.warning("🟡 策略健康度：需关注 (Attention Needed)")

    # Helper to render unified chart
    def render_unified_chart(ts_data):
        if ts_data is None or ts_data.empty:
            st.info("暂无时序数据")
            return

        # Plotly Combined Chart
        if make_subplots is not None:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=("累计净值 (Equity Curve)", "动态回撤 (Drawdown)"),
            )

            # Equity
            fig.add_trace(
                go.Scatter(
                    x=ts_data["date"],
                    y=ts_data["cum_ls_net"],
                    name="多空净值 (Net)",
                    line=dict(color="#2ecc71", width=2),
                ),
                row=1,
                col=1,
            )
            if "cum_ls_gross" in ts_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ts_data["date"],
                        y=ts_data["cum_ls_gross"],
                        name="多空毛值 (Gross)",
                        line=dict(color="#95a5a6", width=1, dash="dot"),
                    ),
                    row=1,
                    col=1,
                )

            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=ts_data["date"],
                    y=ts_data["drawdown"],
                    name="回撤",
                    fill="tozeroy",
                    line=dict(color="#e74c3c", width=1),
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=550, margin=dict(l=40, r=40, t=40, b=40), hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(ts_data[["date", "cum_ls_net"]].set_index("date"))
            st.area_chart(ts_data[["date", "drawdown"]].set_index("date"))

    if not reports:
        # Fallback: 使用最新执行层产物作为概览
        exec_m = _latest("pipeline_execution_*_metrics.json")
        exec_ts = _latest("pipeline_execution_*_timeseries.csv")
        if exec_m and exec_ts:
            st.info("未找到性能报告，已使用最新执行层结果作为概览。")
            import json as _json

            meta = _json.loads(exec_m.read_text(encoding="utf-8"))
            metrics = meta.get("metrics", {})

            render_health(metrics)
            st.subheader("关键指标 (Key Metrics)")
            render_metrics_row(metrics)

            try:
                ts = load_csv_cached(str(exec_ts))
                ts["date"] = pd.to_datetime(ts["date"])
                ts_disp = time_range_selector(ts, key_prefix="ov_fb")
                st.subheader("策略表现 (Performance)")
                render_unified_chart(ts_disp)
            except Exception as e:
                st.error(f"图表加载失败: {e}")
            st.button("刷新总览", on_click=lambda: st.rerun())
        else:
            st.warning(
                "No performance reports found. Run `python run_full_pipeline.py` first."
            )
    else:
        options = {r.name: r for r in reports}
        selected_name = st.sidebar.selectbox(
            "Performance snapshot", list(options.keys()), index=0
        )
        metrics, ts, regimes = load_report(options[selected_name])
        heatmaps = sorted(
            DATA_DIR.glob(selected_name.replace("_metrics.json", "_heatmap*")),
            key=lambda p: p.name,
        )

        render_health(metrics)
        st.subheader("关键指标 (Key Metrics)")
        render_metrics_row(metrics)

        st.subheader("策略表现 (Performance)")
        if not ts.empty:
            ts2 = time_range_selector(ts, key_prefix="ov")
            render_unified_chart(ts2)
        else:
            st.info("Timeseries file missing for this report.")

        st.subheader("制度归因 (Regime Attribution)")
        if not regimes.empty:
            regimes_display = regimes.copy()
            # Format table
            regimes_display["ls_ann"] = regimes_display["ls_ann"].map(
                lambda v: f"{v:.2%}"
            )
            regimes_display["ls_ir"] = regimes_display["ls_ir"].map(
                lambda v: f"{v:.3f}"
            )
            regimes_display["ic_mean"] = regimes_display["ic_mean"].map(
                lambda v: f"{v:.3f}"
            )

            c1, c2 = st.columns([0.6, 0.4])
            with c1:
                st.table(
                    regimes_display[
                        ["regime", "ic_mean", "ls_ann", "ls_ir", "weight_days"]
                    ]
                )
            with c2:
                if go is not None:
                    fig_pie = go.Figure(
                        data=[
                            go.Pie(
                                labels=regimes["regime"],
                                values=regimes["weight_days"],
                                hole=0.4,
                            )
                        ]
                    )
                    fig_pie.update_layout(
                        title_text="制度时间占比",
                        height=300,
                        margin=dict(t=30, b=0, l=20, r=20),
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Regime contribution file missing for this report.")

        if heatmaps:
            st.subheader("参数热力图 (Heatmaps)")
            cols = st.columns(min(2, len(heatmaps)))
            for idx, img_path in enumerate(heatmaps):
                with cols[idx % len(cols)]:
                    st.image(
                        str(img_path), caption=img_path.name, use_container_width=True
                    )
        else:
            st.info(
                "Heatmap images not found. Run the performance reporter to generate them."
            )

        st.markdown("---")
        st.subheader("⚡ 快捷操作")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("加载默认策略快照"):
                try:
                    from json import load

                    p = (
                        DATA_DIR
                        / "config_snapshots"
                        / "default_hysteresis_top40_5bp.json"
                    )
                    if p.exists():
                        with open(p, "r", encoding="utf-8") as f:
                            cfg = load(f)
                        for k, v in {
                            "cfg_standardisation": cfg.get("standardisation"),
                            "cfg_start_oos": cfg.get("start_oos"),
                            "cfg_train_window": cfg.get("train_window"),
                            "cfg_alpha": cfg.get("alpha"),
                            "cfg_top_n": cfg.get("top_n"),
                            "cfg_bottom_n": cfg.get("bottom_n"),
                            "cfg_cost_bps_ui": cfg.get("cost_bps_ui"),
                            "cfg_neutral_shrink": cfg.get("neutral_shrink"),
                            "cfg_neutral_industries": cfg.get("neutral_industries"),
                            "cfg_exec_strategy": cfg.get("exec_strategy"),
                            "cfg_delta": cfg.get("delta"),
                            "cfg_ema_span": cfg.get("ema_span"),
                            "cfg_k": cfg.get("k"),
                            "cfg_swap_cap": cfg.get("swap_cap"),
                        }.items():
                            if v is not None:
                                st.session_state[k] = v
                        st.success("已加载默认快照，请在侧边栏查看参数。")
                    else:
                        st.warning("未找到默认快照文件。")
                except Exception as e:
                    st.error(f"加载失败: {e}")
        with c2:
            if st.button("用侧边栏参数运行流水线"):
                cfg = _current_config()
                cmd = [
                    "python3",
                    "run_full_pipeline.py",
                    "--standardisation",
                    cfg["standardisation"],
                    "--start-oos",
                    cfg["start_oos"],
                    "--train-window",
                    str(int(cfg["train_window"])),
                    "--alpha",
                    str(float(cfg["alpha"])),
                    "--top-n",
                    str(int(cfg["top_n"])),
                    "--bottom-n",
                    str(int(cfg["bottom_n"])),
                    "--cost-bps",
                    str(float(cfg["cost_bps_ui"]) / 10000.0),
                ]
                if cfg["recompute_factors"]:
                    cmd.append("--recompute-factors")
                if cfg["recompute_regime"]:
                    cmd.append("--recompute-regime")
                if float(cfg["neutral_shrink"]) > 0:
                    cmd.extend(["--neutral-shrink", str(cfg["neutral_shrink"])])
                    if cfg["neutral_industries"]:
                        cmd.extend(["--neutral-industries", cfg["neutral_industries"]])
                if cfg["run_cost_grid"]:
                    cmd.append("--run-cost-grid")
                if cfg["exec_strategy"] and cfg["exec_strategy"] != "none":
                    cmd.extend(
                        [
                            "--execution-strategy",
                            cfg["exec_strategy"],
                            "--delta",
                            str(int(cfg["delta"])),
                            "--ema-span",
                            str(int(cfg["ema_span"])),
                            "--k",
                            str(int(cfg["k"])),
                            "--swap-cap",
                            str(float(cfg["swap_cap"])),
                        ]
                    )
                if cfg["run_turnover_grid"]:
                    cmd.append("--run-turnover-grid")
                try:
                    logger.info(f"Starting pipeline with command: {' '.join(cmd)}")
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    logger.info("Pipeline completed successfully")
                    st.success("流水线完成")
                    st.text_area("运行日志", proc.stdout[-6000:], height=220)
                except subprocess.CalledProcessError as e:
                    logger.error(
                        f"Pipeline failed with return code {e.returncode}: {e.stderr}",
                        exc_info=True,
                    )
                    st.error("流水线失败")
                    st.text_area(
                        "错误日志", (e.stdout or "") + "\n" + (e.stderr or ""), height=280
                    )
        with c3:
            if st.button("生成报告包(快速)"):
                # 自动抓取最新产物
                latest_pred = sorted(
                    DATA_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime
                )
                latest_exec_ts = sorted(
                    DATA_DIR.glob("pipeline_execution_*_timeseries.csv"),
                    key=lambda p: p.stat().st_mtime,
                )
                latest_exec_m = sorted(
                    DATA_DIR.glob("pipeline_execution_*_metrics.json"),
                    key=lambda p: p.stat().st_mtime,
                )
                latest_grid = sorted(
                    DATA_DIR.glob("turnover_strategy_grid_*.csv"),
                    key=lambda p: p.stat().st_mtime,
                )
                latest_rb = sorted(
                    DATA_DIR.glob("robustness_summary_*.csv"),
                    key=lambda p: p.stat().st_mtime,
                )
                cmd = ["python3", "analysis/report_packager.py"]
                if latest_pred:
                    cmd.extend(["--predictions", str(latest_pred[-1])])
                if latest_exec_ts:
                    cmd.extend(["--execution-timeseries", str(latest_exec_ts[-1])])
                if latest_exec_m:
                    cmd.extend(["--execution-metrics", str(latest_exec_m[-1])])
                if latest_grid:
                    cmd.extend(["--turnover-grid", str(latest_grid[-1])])
                if latest_rb:
                    cmd.extend(["--robustness", str(latest_rb[-1])])
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, check=True
                    )
                    st.success("报告包已生成（见打包Tab下载）")
                    st.text(proc.stdout)
                except subprocess.CalledProcessError as e:
                    st.error("打包失败")
                    st.text_area(
                        "错误日志", (e.stdout or "") + "\n" + (e.stderr or ""), height=200
                    )

        st.markdown("---")
        st.subheader("🗂️ 最近产物快照")
        # 搜索最新产物并展示简要指标
        latest_pred = sorted(
            DATA_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime
        )
        latest_exec_ts = sorted(
            DATA_DIR.glob("pipeline_execution_*_timeseries.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        latest_exec_m = sorted(
            DATA_DIR.glob("pipeline_execution_*_metrics.json"),
            key=lambda p: p.stat().st_mtime,
        )
        latest_grid = sorted(
            DATA_DIR.glob("turnover_strategy_grid_*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        latest_rb = sorted(
            DATA_DIR.glob("robustness_summary_*.csv"), key=lambda p: p.stat().st_mtime
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Predictions**")
            if latest_pred:
                p = latest_pred[-1]
                st.write(p.name)
                st.caption(
                    f'更新时间: {pd.to_datetime(p.stat().st_mtime, unit="s").strftime("%F %T")}'
                )
                st.download_button("下载", data=p.read_bytes(), file_name=p.name)
            else:
                st.info("暂无")
        with c2:
            st.markdown("**Execution (metrics)**")
            if latest_exec_m:
                import json as _json

                pm = latest_exec_m[-1]
                meta = _json.loads(pm.read_text(encoding="utf-8"))
                m = meta.get("metrics", {})
                colsM = st.columns(2)
                colsM[0].metric("年化", f"{m.get('ls_ann', 0):.2%}")
                colsM[1].metric("IR", f"{m.get('ls_ir', 0):.3f}")
                colsM2 = st.columns(2)
                colsM2[0].metric("换手", f"{m.get('avg_turnover', 0):.2%}")
                colsM2[1].metric("回撤", f"{m.get('max_drawdown', 0):.2%}")
                st.caption(pm.name)
            else:
                st.info("暂无")
        with c3:
            st.markdown("**Turnover Grid**")
            if latest_grid:
                import pandas as _pd

                g = _pd.read_csv(latest_grid[-1])
                best_ir = g.sort_values("ls_ir", ascending=False).head(1)
                best_ann = g.sort_values("ls_ann", ascending=False).head(1)
                if not best_ir.empty:
                    st.write(
                        f"Top IR: IR={best_ir['ls_ir'].iloc[0]:.3f}, ann={best_ir['ls_ann'].iloc[0]:.2%}"
                    )
                if not best_ann.empty:
                    st.write(
                        f"Top Ann: ann={best_ann['ls_ann'].iloc[0]:.2%}, IR={best_ann['ls_ir'].iloc[0]:.3f}"
                    )
                st.caption(latest_grid[-1].name)
            else:
                st.info("暂无")
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**稳健性分析 (Robustness)**")
            if latest_rb:
                import pandas as _pd

                rb = _pd.read_csv(latest_rb[-1])
                st.write(f"组合数：{len(rb)}")
                st.write(f"IR均值：{rb['ls_ir'].mean():.3f}")
                st.caption(latest_rb[-1].name)
            else:
                st.info("暂无")

# 添加全局数据刷新功能
st.sidebar.markdown("**🔄 数据管理**")
if st.sidebar.button("🧹 清除缓存并刷新", help="清除所有缓存数据并重新加载页面", type="secondary"):
    st.cache_data.clear()
    st.sidebar.success("✅ 缓存已清除！页面即将刷新...")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**显示设置**")
simple_mode = st.sidebar.checkbox(
    "简洁模式", value=True, key="cfg_simple_mode", help="仅显示核心视图，隐藏高级可视化与参数"
)

st.sidebar.markdown("**⚙️ 一键运行预测流水线**")
with st.sidebar.form("run_pipeline_form"):
    if st.session_state.get("cfg_simple_mode", True):
        # 简洁模式：仅关键参数
        c1, c2 = st.columns(2)
        with c1:
            top_n_pipe = st.selectbox(
                "持仓股票数 TopN", [20, 30, 40], index=1, help="每日多头持仓数量"
            )
        with c2:
            cost_choice = st.selectbox("每侧成本(bps)", [3, 5], index=1, help="交易成本（基点）")
        delta_pipe = st.selectbox(
            "滞后带宽 Δ", [10, 15, 20], index=1, help="退出阈值带宽：Top(N+Δ)/Bottom(N+Δ)"
        )
        # 隐含/默认值
        standardisation = "zscore"
        start_oos = "2022-01-01"
        train_window = 756
        alpha = 1.0
        recompute_factors = False
        recompute_regime = False
        bottom_n_pipe = top_n_pipe
        cost_bps_ui = float(cost_choice)
        neutral_shrink = 0.0
        neutral_industries = ""
        run_cost_grid = False
        exec_strategy = "hysteresis"
        ema_span_pipe = 4
        k_pipe = 5
        swap_cap_pipe = 0.2
        run_turnover_grid = False
        # 高级设置（可选展开）
        with st.expander("高级设置（可选）"):
            start_oos = st.text_input("样本外起点", start_oos)
            train_window = st.number_input(
                "训练窗口(天)", min_value=252, max_value=1500, value=train_window, step=12
            )
            alpha = st.number_input(
                "岭回归α", min_value=0.01, max_value=10.0, value=alpha, step=0.1
            )
            recompute_factors = st.checkbox("重算因子", value=False)
            recompute_regime = st.checkbox("重算制度", value=False)
            run_turnover_grid = st.checkbox(
                "结束后自动跑换手率网格", value=False, help="使用本次预测直接跑滞后带精细扫描(40组)"
            )
            auto_refresh = st.checkbox("完成后刷新总览", value=True, help="自动刷新“总览”以载入最新产物")
        submit_run = st.form_submit_button("🚀 运行流水线")
    else:
        # 高级模式：保留全部参数
        col1, col2 = st.columns(2)
        with col1:
            standardisation = st.selectbox(
                "标准化方式",
                ["zscore", "rank"],
                index=0,
                key="cfg_standardisation",
                help="因子标准化：zscore或秩(rank)",
            )
            start_oos = st.text_input(
                "样本外起点",
                "2022-01-01",
                key="cfg_start_oos",
                help="滚动训练的样本外开始日期，YYYY-MM-DD",
            )
            train_window = st.number_input(
                "训练窗口(天)",
                min_value=252,
                max_value=1500,
                value=756,
                step=12,
                key="cfg_train_window",
                help="按天计的历史训练窗口长度",
            )
            alpha = st.number_input(
                "岭回归α",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key="cfg_alpha",
                help="Ridge正则强度",
            )
            recompute_factors = st.checkbox(
                "重算因子", value=False, key="cfg_recompute_factors", help="忽略缓存，重新计算因子数据"
            )
            recompute_regime = st.checkbox(
                "重算制度", value=False, key="cfg_recompute_regime", help="忽略缓存，重新计算波动率制度"
            )
        with col2:
            top_n_pipe = st.slider(
                "TopN(多头)", 5, 60, 30, 5, key="cfg_top_n", help="每日分数Top N构成长端"
            )
            bottom_n_pipe = st.slider(
                "BottomN(空头)", 5, 60, 30, 5, key="cfg_bottom_n", help="每日分数底部N构成空端"
            )
            cost_bps_ui = st.number_input(
                "成本(bps/边)",
                min_value=0.0,
                value=5.0,
                step=0.5,
                key="cfg_cost_bps_ui",
                help="双边成本，输入bps数值",
            )
            neutral_shrink = st.number_input(
                "行业中性收缩(0-1)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="cfg_neutral_shrink",
                help="对行业均值的收缩强度，0为关闭",
            )
            neutral_industries = st.text_input(
                "中性化行业(逗号分隔，可留空)",
                key="cfg_neutral_industries",
                help="仅对列出的行业执行中性化；为空表示全部行业",
            )
            run_cost_grid = st.checkbox(
                "同时生成成本敏感网格",
                value=False,
                key="cfg_run_cost_grid",
                help="在流水线结束时同时跑成本敏感网格",
            )
        st.markdown("— 执行策略（可选） —")
        exec_strategy = st.selectbox(
            "执行策略",
            ["none", "hysteresis", "ema_hysteresis", "lowfreq", "swapcap"],
            index=1,
            key="cfg_exec_strategy",
            help="选择低换手执行方案",
        )
        col3, col4 = st.columns(2)
        with col3:
            delta_pipe = st.number_input(
                "滞后带Delta",
                min_value=0,
                max_value=30,
                value=15,
                step=1,
                key="cfg_delta",
                help="退出阈值带宽：Top(N+Δ)/Bottom(N+Δ)",
            )
            ema_span_pipe = st.number_input(
                "EMA窗口",
                min_value=2,
                max_value=20,
                value=4,
                step=1,
                key="cfg_ema_span",
                help="对预测分数进行EMA平滑的窗口",
            )
        with col4:
            k_pipe = st.number_input(
                "低频k(日)",
                min_value=1,
                max_value=60,
                value=5,
                step=1,
                key="cfg_k",
                help="每k日再平衡一次",
            )
            swap_cap_pipe = st.slider(
                "换手上限(比例)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="cfg_swap_cap",
                help="每侧最多替换比例",
            )
        run_turnover_grid = st.checkbox(
            "结束后自动跑换手率网格",
            value=False,
            key="cfg_run_turnover_grid",
            help="使用本次预测直接跑滞后带精细扫描(40组)",
        )
        auto_refresh = st.checkbox("完成后刷新总览", value=False)
        submit_run = st.form_submit_button("🚀 运行流水线")

if submit_run:
    cmd = [
        "python3",
        "run_full_pipeline.py",
        "--standardisation",
        standardisation,
        "--start-oos",
        start_oos,
        "--train-window",
        str(int(train_window)),
        "--alpha",
        str(float(alpha)),
        "--top-n",
        str(int(top_n_pipe)),
        "--bottom-n",
        str(int(bottom_n_pipe)),
        "--cost-bps",
        str(float(cost_bps_ui) / 10000.0),
    ]
    if recompute_factors:
        cmd.append("--recompute-factors")
    if recompute_regime:
        cmd.append("--recompute-regime")
    if neutral_shrink > 0:
        cmd.extend(["--neutral-shrink", str(neutral_shrink)])
        if neutral_industries.strip():
            cmd.extend(["--neutral-industries", neutral_industries])
    if run_cost_grid:
        cmd.append("--run-cost-grid")
    if st.session_state.get("cfg_run_turnover_grid"):
        cmd.append("--run-turnover-grid")

    # 执行策略参数
    if exec_strategy != "none":
        cmd.extend(
            [
                "--execution-strategy",
                exec_strategy,
                "--delta",
                str(int(delta_pipe)),
                "--ema-span",
                str(int(ema_span_pipe)),
                "--k",
                str(int(k_pipe)),
                "--swap-cap",
                str(float(swap_cap_pipe)),
            ]
        )

    with st.spinner("正在运行流水线..."):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("流水线已完成")
            st.text_area("运行日志", proc.stdout[-8000:], height=240)
            if "auto_refresh" in locals() and auto_refresh:
                st.rerun()
        except subprocess.CalledProcessError as e:
            st.error("流水线运行失败")
            st.text_area("错误日志", (e.stdout or "") + "\n" + (e.stderr or ""), height=320)

# 快照目录与保存/加载逻辑
SNAP_DIR = DATA_DIR / "config_snapshots"
SNAP_DIR.mkdir(exist_ok=True)


def _current_config() -> dict:
    return {
        "standardisation": st.session_state.get("cfg_standardisation", "zscore"),
        "start_oos": st.session_state.get("cfg_start_oos", "2022-01-01"),
        "train_window": st.session_state.get("cfg_train_window", 756),
        "alpha": st.session_state.get("cfg_alpha", 1.0),
        "top_n": st.session_state.get("cfg_top_n", 30),
        "bottom_n": st.session_state.get("cfg_bottom_n", 30),
        "cost_bps_ui": st.session_state.get("cfg_cost_bps_ui", 5.0),
        "neutral_shrink": st.session_state.get("cfg_neutral_shrink", 0.0),
        "neutral_industries": st.session_state.get("cfg_neutral_industries", ""),
        "exec_strategy": st.session_state.get("cfg_exec_strategy", "hysteresis"),
        "delta": st.session_state.get("cfg_delta", 15),
        "ema_span": st.session_state.get("cfg_ema_span", 4),
        "k": st.session_state.get("cfg_k", 5),
        "swap_cap": st.session_state.get("cfg_swap_cap", 0.2),
        "run_cost_grid": st.session_state.get("cfg_run_cost_grid", False),
        "run_turnover_grid": st.session_state.get("cfg_run_turnover_grid", False),
        "recompute_factors": st.session_state.get("cfg_recompute_factors", False),
        "recompute_regime": st.session_state.get("cfg_recompute_regime", False),
    }


def _list_snapshots():
    return sorted(
        SNAP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )


uploaded = st.sidebar.file_uploader("Preview predictions CSV", type=["csv"])
if uploaded is not None:
    uploaded_df = pd.read_csv(uploaded)
    st.sidebar.write(uploaded_df.head())

# 快照保存按钮置于侧边栏流水线表单之后
st.sidebar.markdown("---")
with st.sidebar.form("snapshot_form"):
    st.markdown("**🗂️ 配置快照**")
    snap_name = st.text_input("快照名称(保存/覆盖)", "", key="cfg_snapshot_name")
    save_snap = st.form_submit_button("💾 保存当前配置为快照")
if save_snap:
    import json
    import time

    cfg = _current_config()
    name = st.session_state.get("cfg_snapshot_name") or time.strftime("%Y%m%d_%H%M%S")
    path = SNAP_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    st.sidebar.success(f"已保存: {path.name}")

# 默认策略快照（若不存在则创建一次）
try:
    default_path = SNAP_DIR / "default_hysteresis_top40_5bp.json"
    if not default_path.exists():
        import json

        default_cfg = {
            "standardisation": "zscore",
            "start_oos": "2022-01-01",
            "train_window": 756,
            "alpha": 1.0,
            "top_n": 40,
            "bottom_n": 40,
            "cost_bps_ui": 5.0,
            "neutral_shrink": 0.0,
            "neutral_industries": "",
            "exec_strategy": "hysteresis",
            "delta": 15,
            "ema_span": 4,
            "k": 5,
            "swap_cap": 0.2,
            "run_cost_grid": False,
            "run_turnover_grid": False,
            "recompute_factors": False,
            "recompute_regime": False,
        }
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, ensure_ascii=False, indent=2)
except Exception:
    pass

grids = list_cost_grids()
grid_map = {grid.name: grid for grid in grids}
choices = list(grid_map.keys())
if grids:
    st.sidebar.markdown("---")

# 成本敏感网格（健壮处理：无文件/陈旧选择）
selected_cost_key = st.session_state.get("ui_cost_grid_choice")
if selected_cost_key not in choices:
    selected_cost_key = choices[0] if choices else None

if st.session_state.get("cfg_simple_mode", True):
    with st.sidebar.expander("高级：成本敏感快照"):
        if choices:
            grid_choice = st.selectbox(
                "成本敏感快照",
                choices,
                index=choices.index(selected_cost_key),
                help="训练窗口/成本/TopN参数的敏感性结果",
            )
            st.session_state["ui_cost_grid_choice"] = grid_choice
        else:
            st.caption("暂无成本敏感网格结果。可在流水线勾选“同时生成成本敏感网格”。")
else:
    if choices:
        grid_choice = st.sidebar.selectbox(
            "成本敏感快照",
            choices,
            index=choices.index(selected_cost_key),
            help="训练窗口/成本/TopN参数的敏感性结果",
            key="ui_cost_grid_choice",
        )
        # 读取并展示
        try:
            grid_df = load_csv_cached(str(grid_map[grid_choice]))
        except Exception:
            grid_df = pd.DataFrame()
        if not grid_df.empty:
            top_k = st.sidebar.slider(
                "Top configurations to display",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
            )
            pivot_metric_options = [
                c for c in ["ls_ir", "ls_ann", "ic_mean"] if c in grid_df.columns
            ]
            pivot_metric = st.sidebar.selectbox(
                "Pivot metric", pivot_metric_options or ["ls_ir"], index=0
            )

            with tab_grid:
                st.subheader("成本敏感性分析 — 最佳配置 (按 LS IR)")
                st.dataframe(
                    grid_df.sort_values("ls_ir", ascending=False).head(top_k)[
                        [
                            "train_window",
                            "alpha",
                            "top_n",
                            "bottom_n",
                            "cost_bps",
                            "ls_ir",
                            "ls_ann",
                            "ic_mean",
                        ]
                    ]
                )

                st.subheader(f"成本敏感性透视 ({pivot_metric})")
                try:
                    pivot = grid_df.pivot_table(
                        index="top_n",
                        columns=["train_window", "cost_bps"],
                        values=pivot_metric,
                        aggfunc="max",
                    )
                    st.dataframe(pivot)
                except Exception:
                    st.info("该指标无法生成透视表。")
        else:
            with tab_grid:
                st.info("未能读取成本敏感网格文件，请先生成。")
    else:
        with tab_grid:
            st.info("暂无成本敏感网格结果。请在流水线中勾选“同时生成成本敏感网格”后重试。")

pred_files = list_predictions()
turnover_grids = list_turnover_grids()

# 新增执行策略选择面板
st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 执行策略测试**")

if turnover_grids:
    turnover_map = {grid.name: grid for grid in turnover_grids}
    turnover_choice = st.sidebar.selectbox(
        "换手率优化结果", list(turnover_map.keys()), index=0
    )
    turnover_df = load_csv_cached(str(turnover_map[turnover_choice]))

    # 简洁模式：只显示基础参数
    if st.session_state.get("cfg_simple_mode", True):
        # 只显示核心过滤器
        with st.sidebar.expander("⚙️ 高级：策略过滤器"):
            # 风险过滤器：最大回撤不超过阈值（默认=基线回撤+2个百分点，若无基线→按网格最深回撤+2pp）
            base_rows = turnover_df[
                turnover_df["strategy"].astype(str).str.contains("BASE", na=False)
            ]
            if not base_rows.empty:
                base_mdd_abs = float(base_rows["max_drawdown"].abs().mean())
                default_cap_pct = min(max(5.0, base_mdd_abs * 100 + 2.0), 60.0)
            else:
                # 取网格最深回撤作为参考（取绝对值）
                deepest = (
                    float(turnover_df["max_drawdown"].min())
                    if "max_drawdown" in turnover_df.columns
                    else -0.2
                )
                default_cap_pct = min(max(5.0, abs(deepest) * 100 + 2.0), 60.0)
            dd_cap_pct = st.slider(
                "最大回撤上限（%）",
                min_value=5.0,
                max_value=60.0,
                value=float(default_cap_pct),
                step=0.5,
                help="过滤超过该回撤的组合",
            )
            # 选择回撤阈值模式
            dd_mode = st.radio(
                "回撤阈值模式",
                ["相对基线(+pp)", "绝对值(%)"],
                index=0,
                horizontal=True,
                help="相对基线：默认基线回撤+2pp；绝对：直接指定%",
            )
            ir_min = st.slider(
                "最小IR",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="过滤IR低于该阈值的组合",
            )
            to_cap = st.slider(
                "换手率上限",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="过滤换手率超过该阈值的组合",
            )
    else:
        # 完整模式：显示所有参数
        # 风险过滤器：最大回撤不超过阈值（默认=基线回撤+2个百分点，若无基线→按网格最深回撤+2pp）
        base_rows = turnover_df[
            turnover_df["strategy"].astype(str).str.contains("BASE", na=False)
        ]
        if not base_rows.empty:
            base_mdd_abs = float(base_rows["max_drawdown"].abs().mean())  # 取绝对值的均值
            default_cap_pct = min(max(5.0, base_mdd_abs * 100 + 2.0), 60.0)
        else:
            deepest = (
                float(turnover_df["max_drawdown"].min())
                if "max_drawdown" in turnover_df.columns
                else -0.2
            )
            default_cap_pct = min(max(5.0, abs(deepest) * 100 + 2.0), 60.0)
        dd_cap_pct = st.sidebar.slider(
            "最大回撤上限（%）",
            min_value=5.0,
            max_value=60.0,
            value=float(default_cap_pct),
            step=0.5,
            help="过滤超过该回撤的组合",
        )
        # 选择回撤阈值模式
        dd_mode = st.sidebar.radio(
            "回撤阈值模式",
            ["相对基线(+pp)", "绝对值(%)"],
            index=0,
            horizontal=True,
            help="相对基线：默认基线回撤+2pp；绝对：直接指定%",
        )
        ir_min = st.sidebar.slider(
            "最小IR",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="过滤IR低于该阈值的组合",
        )
        to_cap = st.sidebar.slider(
            "换手率上限",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="过滤换手率超过该阈值的组合",
        )
    if dd_mode == "绝对值(%)":
        dd_cap_value = -dd_cap_pct / 100.0
    else:
        # 相对基线 +pp：以基线均值为底
        dd_cap_value = -default_cap_pct / 100.0
    # 应用回撤阈值过滤
    filtered = turnover_df[turnover_df["max_drawdown"] >= dd_cap_value].copy()

    # 其他过滤器已在上面设置
    filtered = filtered.query("ls_ir >= @ir_min and avg_turnover <= @to_cap").copy()
    # 过滤后为空时，放宽到未过滤集，避免表格为空
    if filtered.empty:
        filtered = turnover_df.copy()

    with tab_exec:
        # ---------------------------------------------------------
        # 🛠 沙箱演练 (Sandbox) - 移至 Exec Tab 顶部
        # ---------------------------------------------------------
        with st.expander("🛠 沙箱演练 (Sandbox)", expanded=True):
            st.markdown("**🔄 实时策略测试**")
            with st.form("custom_portfolio_in_tab"):
                if pred_files:
                    pred_map = {p.name: p for p in pred_files}
                    pred_choice = st.selectbox("预测文件", list(pred_map.keys()), index=0)
                else:
                    pred_map = {}
                    pred_choice = None
                    st.info("在 data/ 下未找到预测文件。")

                # 执行策略选择
                strat_opts = (
                    ["hysteresis_bands"]
                    if st.session_state.get("cfg_simple_mode", True)
                    else ["baseline_daily", "hysteresis_bands", "ema_hysteresis_combo"]
                )
                strategy_type = st.selectbox("执行策略", strat_opts, help="简洁模式仅保留滞后带方案")

                c1, c2, c3 = st.columns(3)
                with c1:
                    top_n_input = st.slider(
                        "持仓股票数 TopN", min_value=5, max_value=50, value=30, step=5
                    )
                with c2:
                    bottom_n_input = st.slider(
                        "空头股票数 BottomN", min_value=5, max_value=50, value=30, step=5
                    )
                with c3:
                    cost_input = st.number_input(
                        "每侧成本 (bps)", min_value=0.0, value=5.0, step=0.5
                    )

                c4, c5 = st.columns(2)
                with c4:
                    if strategy_type == "hysteresis_bands":
                        delta = st.slider("滞后带宽 Δ", 5, 25, 15)
                        ema_span = None
                    elif strategy_type == "ema_hysteresis_combo":
                        ema_span = st.slider("EMA窗口", 2, 15, 4)
                        delta = st.slider("滞后带宽 Δ", 5, 25, 15)
                    else:
                        delta = None
                        ema_span = None
                with c5:
                    st.markdown(
                        '<div style="height: 28px;"></div>', unsafe_allow_html=True
                    )
                    submit_custom = st.form_submit_button("🎯 计算并对比")
            if pred_files and submit_custom and pred_choice:
                pred_path = pred_map[pred_choice]
                custom_pred = load_predictions_df(str(pred_path))
                cost_decimal = cost_input / 10000.0

                # 根据策略类型计算结果
                if strategy_type == "baseline_daily":
                    ts_custom = baseline_daily(
                        custom_pred, top_n_input, bottom_n_input, cost_decimal
                    )
                    ic_series = compute_ic_series_with_score(custom_pred, "y_pred")
                elif strategy_type == "hysteresis_bands":
                    ts_custom = hysteresis_bands(
                        custom_pred, top_n_input, bottom_n_input, cost_decimal, delta
                    )
                    ic_series = compute_ic_series_with_score(custom_pred, "y_pred")
                elif strategy_type == "ema_hysteresis_combo":
                    ts_custom, scored_df = ema_hysteresis_combo(
                        custom_pred,
                        top_n_input,
                        bottom_n_input,
                        cost_decimal,
                        ema_span,
                        delta,
                    )
                    ic_series = compute_ic_series_with_score(scored_df, "y_score")
                else:
                    ts_custom = pd.DataFrame()
                    ic_series = pd.Series()

                if ts_custom.empty:
                    st.warning("应用参数后数据不足，无法计算指标。")
                else:
                    # 计算指标
                    custom_metrics = compute_summary_metrics(ts_custom, ic_series)

                    # 显示关键指标 (Chinese)
                    st.markdown(f"**测试结果: {strategy_type}**")
                    cols_custom = st.columns(5)
                    display_pairs = [
                        ("ls_ann", "年化收益"),
                        ("ls_ir", "信息比率"),
                        ("ic_mean", "IC均值"),
                        ("avg_turnover", "平均换手率"),
                        ("max_drawdown", "最大回撤"),
                    ]
                    for col, (key, label) in zip(cols_custom, display_pairs):
                        val = custom_metrics.get(key)
                        if key in ["max_drawdown", "ls_ann", "avg_turnover"]:
                            col.metric(label, f"{val:.2%}")
                        else:
                            col.metric(label, f"{val:.3f}")

                    # 显示图表 (Unified)
                    if make_subplots is not None:
                        fig_sb = make_subplots(
                            rows=2,
                            cols=1,
                            shared_xaxes=True,
                            row_heights=[0.7, 0.3],
                            vertical_spacing=0.03,
                        )
                        fig_sb.add_trace(
                            go.Scatter(
                                x=ts_custom["date"],
                                y=ts_custom["cum_ls_net"],
                                name="净值",
                                line=dict(color="#2ecc71"),
                            ),
                            row=1,
                            col=1,
                        )
                        fig_sb.add_trace(
                            go.Scatter(
                                x=ts_custom["date"],
                                y=ts_custom["drawdown"],
                                name="回撤",
                                fill="tozeroy",
                                line=dict(color="#e74c3c"),
                            ),
                            row=2,
                            col=1,
                        )
                        fig_sb.update_layout(
                            height=400,
                            margin=dict(l=20, r=20, t=20, b=20),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_sb, use_container_width=True)
                    else:
                        st.line_chart(
                            ts_custom[["date", "cum_ls_net"]].set_index("date")
                        )

        st.markdown("---")

        # 标题栏与刷新按钮
        exec_cols = st.columns([0.8, 0.2])
        with exec_cols[0]:
            st.subheader("🎯 换手率优化策略排行")
        with exec_cols[1]:
            if st.button("🔄 刷新执行数据", help="重新加载执行测试数据"):
                st.cache_data.clear()
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**按信息比率排序**")
            top_ir = filtered.sort_values("ls_ir", ascending=False).head(10)
            st.dataframe(
                top_ir[
                    [
                        "strategy",
                        "param",
                        "top_n",
                        "cost_bps",
                        "ls_ir",
                        "ls_ann",
                        "avg_turnover",
                        "max_drawdown",
                    ]
                ].round(4)
            )
        with col2:
            st.markdown("**按年化收益排序**")
            top_ann = filtered.sort_values("ls_ann", ascending=False).head(10)
            st.dataframe(
                top_ann[
                    [
                        "strategy",
                        "param",
                        "top_n",
                        "cost_bps",
                        "ls_ann",
                        "ls_ir",
                        "avg_turnover",
                        "max_drawdown",
                    ]
                ].round(4)
            )

        # 策略对比分析
        st.subheader("📊 策略效果对比")
        strategy_summary = (
            turnover_df.groupby("strategy")
            .agg(
                {
                    "ls_ir": ["mean", "max"],
                    "ls_ann": ["mean", "max"],
                    "avg_turnover": ["mean", "min"],
                    "max_drawdown": "mean",
                }
            )
            .round(4)
        )
        st.dataframe(strategy_summary)

        # 下载链接
        csv_data = filtered.to_csv(index=False)
        st.download_button(
            label="📥 下载完整结果CSV",
            data=csv_data,
            file_name=f"turnover_results_{turnover_choice.split('_')[-1].replace('.csv', '')}.csv",
            mime="text/csv",
        )
else:
    st.sidebar.info("运行 `python3 analysis/turnover_strategy_grid.py` 生成换手率优化结果")

# 提交执行策略网格任务
if not st.session_state.get("cfg_simple_mode", True):
    with tab_exec:
        with st.expander("🧮 高级：提交执行策略网格任务"):
            with st.form("turnover_grid_form"):
                grid_pred = st.selectbox(
                    "预测文件(用于网格)",
                    [p.name for p in pred_files] if pred_files else ["(无)"],
                )
                preset = st.selectbox(
                    "预设", ["自定义", "滞后带精细扫描(40组)", "组合策略(36组)"], index=1
                )
                strategies_in = st.text_input("策略集(逗号分隔)", "A,B,C,D,E,BASE")
                topns_in = st.text_input("TopNs", "20,30,40")
                costs_in = st.text_input("成本(十进制)", "0.0003,0.0005,0.0008")
                deltas_in = st.text_input("delta值", "0,10,20")
                emas_in = st.text_input("EMA窗口", "3,5,10")
                ks_in = st.text_input("低频k", "5,10,20")
                swapcaps_in = st.text_input("换手上限", "0.1,0.2,0.3")
                submit_grid = st.form_submit_button("📤 运行网格任务")
            if submit_grid:
                if not pred_files or grid_pred == "(无)":
                    st.error("未找到预测文件，无法运行网格任务。")
                else:
                    pred_path = str(DATA_DIR / grid_pred)
                    cmd = [
                        "python3",
                        "analysis/turnover_strategy_grid.py",
                        "--predictions",
                        pred_path,
                    ]
                    if preset == "滞后带精细扫描(40组)":
                        cmd.append("--fine-tune")
                    elif preset == "组合策略(36组)":
                        cmd.append("--combo-only")
                    else:
                        cmd.extend(
                            [
                                "--strategies",
                                strategies_in,
                                "--top-ns",
                                topns_in,
                                "--cost-bps",
                                costs_in,
                                "--delta-values",
                                deltas_in,
                                "--ema-spans",
                                emas_in,
                                "--k-values",
                                ks_in,
                                "--swap-caps",
                                swapcaps_in,
                            ]
                        )
                    with st.spinner("正在运行执行策略网格任务..."):
                        try:
                            proc = subprocess.run(
                                cmd, capture_output=True, text=True, check=True
                            )
                            st.success("网格任务完成")
                            st.text_area("任务日志", proc.stdout[-6000:], height=220)
                        except subprocess.CalledProcessError as e:
                            st.error("网格任务失败")
                            st.text_area(
                                "错误日志",
                                (e.stdout or "") + "\n" + (e.stderr or ""),
                                height=300,
                            )


# 稳健性验证面板（输出位于“稳健性”Tab）
with tab_robust:
    # 标题栏与刷新按钮
    robust_cols = st.columns([0.8, 0.2])
    with robust_cols[0]:
        st.subheader("🔒 稳健性验证（多起点×多窗口）")
    with robust_cols[1]:
        if st.button("🔄 刷新稳健性", help="重新加载稳健性验证数据"):
            st.cache_data.clear()
            st.rerun()

with tab_robust:
    with st.expander("🧪 高级：稳健性验证参数"):
        with st.form("robustness_form_in_tab"):
            rb_strategy = st.selectbox(
                "执行策略", ["hysteresis", "combo", "baseline"], index=0
            )
            rb_start_oos = st.text_input("起点(逗号分隔)", "2021-01-01,2022-01-01,2022-07-01")
            rb_train_windows = st.text_input("窗口(逗号分隔)", "756,900,1008")
            rb_top_n = st.number_input(
                "TopN", min_value=10, max_value=60, value=40, step=5
            )
            rb_cost_bps = st.number_input(
                "成本(bps/边)", min_value=0.0, value=5.0, step=0.5
            )
            rb_delta = st.number_input(
                "滞后带Delta", min_value=0, max_value=30, value=15, step=1
            )
            rb_ema = st.number_input(
                "EMA窗口", min_value=2, max_value=20, value=4, step=1
            )
            submit_rb = st.form_submit_button("🧪 运行稳健性验证")


def _list_robustness_csv():
    files = sorted(
        DATA_DIR.glob("robustness_summary_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


if submit_rb:
    cmd = [
        "python3",
        "analysis/robustness_validator.py",
        "--strategy",
        rb_strategy,
        "--start-oos",
        rb_start_oos,
        "--train-windows",
        rb_train_windows,
        "--top-n",
        str(int(rb_top_n)),
        "--cost-bps",
        str(float(rb_cost_bps) / 10000.0),
        "--delta",
        str(int(rb_delta)),
        "--ema-span",
        str(int(rb_ema)),
    ]
    with st.spinner("正在运行稳健性验证..."):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            with tab_robust:
                st.success("稳健性验证完成")
                st.text_area("运行日志", proc.stdout[-8000:], height=240)
        except subprocess.CalledProcessError as e:
            with tab_robust:
                st.error("稳健性验证失败")
                st.text_area(
                    "错误日志", (e.stdout or "") + "\n" + (e.stderr or ""), height=320
                )

rb_files = _list_robustness_csv()
with tab_robust:
    if rb_files:
        latest_rb = rb_files[0]
        st.markdown(f"最近结果文件：`{latest_rb.name}`")
        rb_df = load_csv_cached(str(latest_rb))
        st.dataframe(rb_df)
        # 透视表：按起点与窗口的 IR
        try:
            pivot_ir = rb_df.pivot_table(
                index="start_oos",
                columns="train_window",
                values="ls_ir",
                aggfunc="mean",
            )
            st.markdown("**稳健性矩阵（LS IR）**")
            st.dataframe(pivot_ir.round(3))
        except Exception:
            pass
    else:
        st.info("尚未生成稳健性结果。请在侧边栏发起一次验证。")

with tab_grid:
    # 成本—容量曲面与执行画像
    head_cols = st.columns([0.75, 0.17, 0.08])
    with head_cols[0]:
        st.subheader("📈 成本—容量曲面（基于换手率网格）")
    with head_cols[1]:
        if st.button("🔄 刷新网格数据", help="重新加载网格分析数据"):
            # 清除相关缓存
            for cache_key in list(st.session_state.keys()):
                if "grid" in cache_key.lower() or "turnover" in cache_key.lower():
                    del st.session_state[cache_key]
            st.cache_data.clear()
            st.rerun()
    with head_cols[2]:
        info_icon("展示 TopN × 成本(bps) 下，关键指标（IR/年化）的分布。简洁模式默认隐藏热力图，仅展示表格。")
    if turnover_grids:
        grid_map2 = {g.name: g for g in turnover_grids}
        grid_choice2 = st.selectbox(
            "选择换手率网格文件", list(grid_map2.keys()), index=0, help="来源于执行策略网格任务的结果文件"
        )
        grid_df2 = load_csv_cached(str(grid_map2[grid_choice2]))
        # 友好名称映射
        strat_map = {
            "B_hysteresis": "滞后带(Hysteresis)",
            "C_ema": "EMA平滑",
            "E_combo": "EMA+滞后带",
            "A_lowfreq": "低频再平衡",
            "D_swapcap": "换手上限",
            "BASE_daily": "基线日更",
        }
        inv_strat_map = {v: k for k, v in strat_map.items()}
        strat_options = [
            strat_map.get(s, s)
            for s in sorted(grid_df2["strategy"].dropna().unique().tolist())
        ]
        strat_label = st.selectbox("策略筛选", strat_options, help="选择想要查看的执行方案")
        strategy_filter = inv_strat_map.get(strat_label, strat_label)

        metric_map = {"ls_ir": "多空IR(信息比率)", "ls_ann": "多空年化"}
        metric_label = st.selectbox(
            "指标", list(metric_map.values()), index=0, help="选择热力图展示的目标指标"
        )
        metric_choice = {v: k for k, v in metric_map.items()}[metric_label]

        sub_df = grid_df2[grid_df2["strategy"] == strategy_filter].copy()
        if sub_df.empty:
            st.info("该策略暂无结果。")
        else:
            pivot = sub_df.pivot_table(
                index="top_n", columns="cost_bps", values=metric_choice, aggfunc="max"
            )
            # 可视化热力图（Plotly）
            z = pivot.values
            x_vals = list(pivot.columns)
            x = [str(c) for c in x_vals]
            y = [str(i) for i in pivot.index]
            show_heatmap = not st.session_state.get("cfg_simple_mode", True)
            show_heatmap = st.checkbox(
                "显示热力图", value=show_heatmap, help="关闭后仅展示表格视图，更利于阅读。"
            )
            selected = None
            if show_heatmap and go is not None:
                fig = go.Figure(
                    data=go.Heatmap(
                        z=z,
                        x=x,
                        y=y,
                        colorscale="YlGnBu",
                        colorbar=dict(title=metric_label),
                    )
                )
                # X轴格式化为 bps，如 3bp/5bp
                x_ticktext = [f"{int(round(float(v)*10000))}bp" for v in x_vals]
                fig.update_xaxes(
                    tickmode="array",
                    tickvals=x,
                    ticktext=x_ticktext,
                    title_text="成本(bps)",
                )
                fig.update_yaxes(title_text="TopN(股票数)")
                fig.update_layout(height=420, margin=dict(l=40, r=20, t=30, b=40))
                if HAS_PLOTLY_EVENTS:
                    selected_pts = plotly_events(
                        fig,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        key="heatmap_click",
                    )
                    selected = selected_pts[0] if selected_pts else None
                else:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("提示：要开启热力图点击联动，请安装依赖：pip install streamlit-plotly-events")
            else:
                st.dataframe(
                    pivot.rename(
                        columns=lambda c: f"{int(round(float(c)*10000))}bp"
                    ).style.background_gradient(cmap="YlGnBu")
                )

            # 参数联动：选择 TopN 与 Cost，基于网格中该组合的最优行一键重算并展示
            st.markdown("**参数联动与一键重算**")
            colA, colB, colC = st.columns(3)
            with colA:
                topn_options = sorted(sub_df["top_n"].unique().tolist())
                topn_sel = st.selectbox("TopN(股票数)", topn_options, help="每日多头持仓数量")
            with colB:
                cost_options = sorted(sub_df["cost_bps"].unique().tolist())
                cost_labels = [f"{int(round(c*10000))}bp" for c in cost_options]
                idx_default = 0
                cost_label = st.selectbox(
                    "成本(bps) i", cost_labels, index=idx_default, help="每侧交易成本（基点）"
                )
                cost_sel = cost_options[cost_labels.index(cost_label)]
            with colC:
                # 选用 IR 优先
                rows_choice = sub_df[
                    (sub_df["top_n"] == topn_sel) & (sub_df["cost_bps"] == cost_sel)
                ]
                chosen = rows_choice.sort_values("ls_ir", ascending=False).head(1)
                if chosen.empty:
                    st.warning("该组合无可用行")
                    chosen_row = None
                else:
                    st.write("已选网格行（IR优先）:")
                    st.dataframe(
                        chosen[
                            [
                                "strategy",
                                "param",
                                "top_n",
                                "cost_bps",
                                "ls_ir",
                                "ls_ann",
                                "avg_turnover",
                                "max_drawdown",
                            ]
                        ].round(4)
                    )
                    chosen_row = chosen.iloc[0]

            # 点击热力图联动覆盖选择
            if show_heatmap and go is not None and HAS_PLOTLY_EVENTS and selected:
                try:
                    clicked_topn = int(float(selected.get("y")))
                    clicked_cost = float(selected.get("x"))
                except Exception as e:
                    st.warning(f"读取点击位置失败：{e}")
                    clicked_topn = None
                    clicked_cost = None
                if clicked_topn in topn_options and clicked_cost in cost_options:
                    topn_sel = clicked_topn
                    cost_sel = clicked_cost
                    st.info(f"已根据热力图点击选择 TopN={topn_sel}, Cost={cost_sel}")
                    rows_choice = sub_df[
                        (sub_df["top_n"] == topn_sel) & (sub_df["cost_bps"] == cost_sel)
                    ]
                    chosen = rows_choice.sort_values("ls_ir", ascending=False).head(1)
                    chosen_row = chosen.iloc[0] if not chosen.empty else None

        # 选预测文件并执行重算
        if chosen_row is not None:
            pred_map2 = {p.name: p for p in pred_files} if pred_files else {}
            pred_name2 = st.selectbox(
                "用于重算的预测文件", list(pred_map2.keys()) if pred_map2 else ["(无)"]
            )
            run_apply = st.button("⚡ 用该配置一键重算并展示")
            if run_apply and pred_map2:
                df_pred = load_predictions_df(str(pred_map2[pred_name2]))
                top_n = int(chosen_row["top_n"])
                bottom_n = top_n
                cost_bps = float(chosen_row["cost_bps"])

                # 根据策略类型调用
                if strategy_filter.startswith("B_hysteresis"):
                    try:
                        # param 可能是数值或字符串
                        delta_val = int(
                            float(chosen_row.get("param", chosen_row.get("delta", 15)))
                        )
                    except Exception:
                        delta_val = int(chosen_row.get("delta", 15))
                    ts_custom = hysteresis_bands(
                        df_pred, top_n, bottom_n, cost_bps, delta=delta_val
                    )
                    ic_series = compute_ic_series_with_score(df_pred, "y_pred")
                    title = (
                        f"Hysteresis delta={delta_val}, TopN={top_n}, cost={cost_bps}"
                    )
                elif strategy_filter.startswith("E_combo"):
                    ema_span = int(chosen_row.get("ema_span", 3))
                    delta_val = int(chosen_row.get("delta", 12))
                    ts_custom, scored_df = ema_hysteresis_combo(
                        df_pred,
                        top_n,
                        bottom_n,
                        cost_bps,
                        ema_span=ema_span,
                        delta=delta_val,
                    )
                    ic_series = compute_ic_series_with_score(scored_df, "y_score")
                    title = f"EMA+Hysteresis ema={ema_span}, delta={delta_val}, TopN={top_n}, cost={cost_bps}"
                else:
                    ts_custom = baseline_daily(df_pred, top_n, bottom_n, cost_bps)
                    ic_series = compute_ic_series_with_score(df_pred, "y_pred")
                    title = f"Baseline TopN={top_n}, cost={cost_bps}"

                if ts_custom.empty:
                    st.warning("数据不足，无法计算指标。")
                else:
                    metrics_custom = compute_summary_metrics(ts_custom, ic_series)
                    st.markdown(f"**重算结果：{title}**")
                    colsX = st.columns(5)
                    for col, (k, label) in zip(
                        colsX,
                        [
                            ("ls_ann", "年化收益"),
                            ("ls_ir", "信息比率"),
                            ("ic_mean", "IC均值"),
                            ("avg_turnover", "平均换手率"),
                            ("max_drawdown", "最大回撤"),
                        ],
                    ):
                        val = metrics_custom.get(k)
                        if k in ["ls_ann", "avg_turnover", "max_drawdown"]:
                            col.metric(label, f"{val:.2%}")
                        else:
                            col.metric(label, f"{val:.3f}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.line_chart(
                            ts_custom[["date", "cum_ls_net"]].set_index("date")
                        )
                    with col2:
                        st.area_chart(ts_custom[["date", "drawdown"]].set_index("date"))
    else:
        st.info("尚无换手率网格结果。请先运行网格任务。")

    with tab_exec:
        st.subheader("👣 执行画像（估算法）")
        exec_ts_files = sorted(
            DATA_DIR.glob("pipeline_execution_*_timeseries.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if exec_ts_files:
            exec_map = {p.name: p for p in exec_ts_files}
            exec_choice = st.selectbox("选择执行后时序文件", list(exec_map.keys()), index=0)
            exec_df = pd.read_csv(exec_map[exec_choice], parse_dates=["date"])
            # 尝试读取同名前缀的metrics.json（若存在，优先使用其中的执行画像摘要）
            metrics_json = exec_map[exec_choice].with_name(
                exec_map[exec_choice].name.replace("_timeseries.csv", "_metrics.json")
            )
            profile = None
            if metrics_json.exists():
                try:
                    with open(metrics_json, "r", encoding="utf-8") as f:
                        prof = json.load(f)
                        profile = prof.get("execution_profile")
                except Exception:
                    profile = None
            colA, colB, colC = st.columns(3)
            with colA:
                topN_est = st.number_input(
                    "估算TopN", min_value=5, max_value=80, value=30, step=5
                )
            with colB:
                bottomN_est = st.number_input(
                    "估算BottomN", min_value=5, max_value=80, value=30, step=5
                )
            with colC:
                st.write("用于估算调仓笔数与半衰期")

            exec_df = exec_df.sort_values("date").reset_index(drop=True)
            # 真实指标优先：若有 added_* 列，按当日变化笔数估算调仓笔数
            if {"added_long", "added_short"}.issubset(exec_df.columns):
                exec_df["trades_per_day"] = (
                    exec_df["added_long"] + exec_df["added_short"]
                )
            else:
                n_mean = (topN_est + bottomN_est) / 2.0
                exec_df["trades_per_day"] = exec_df["turnover"] * n_mean

            # 显示优先使用 metrics.json 中的执行画像摘要
            if profile:
                avg_turn = float(
                    profile.get("avg_turnover", float(exec_df["turnover"].mean()))
                )
                avg_overlap_long = profile.get("avg_overlap_long")
                avg_overlap_short = profile.get("avg_overlap_short")
                avg_overlap = (
                    float(np.nanmean([avg_overlap_long, avg_overlap_short]))
                    if (avg_overlap_long is not None and avg_overlap_short is not None)
                    else float(1.0 - exec_df["turnover"].mean() / 2.0)
                )
                est_half_life = profile.get("half_life_long_days")
                half_life_short = profile.get("half_life_short_days")
                if est_half_life is None or np.isnan(est_half_life):
                    # 退化为重叠率估算
                    if 0 < avg_overlap < 1:
                        est_half_life = float(np.log(0.5) / np.log(avg_overlap))
                    else:
                        est_half_life = np.nan
            else:
                # 估算路径
                avg_turn = float(exec_df["turnover"].mean())
                avg_overlap = float(1.0 - exec_df["turnover"].mean() / 2.0)
                est_half_life = np.nan
                if 0 < avg_overlap < 1:
                    est_half_life = float(np.log(0.5) / np.log(avg_overlap))

            colsM = st.columns(4)
            colsM[0].metric("平均换手率", f"{avg_turn:.2%}")
            colsM[1].metric("平均重叠率(估)", f"{avg_overlap:.2%}")
            colsM[2].metric(
                "半衰期(天,估)",
                f"{est_half_life:.1f}" if np.isfinite(est_half_life) else "-",
            )
            colsM[3].metric("日均调仓笔数", f"{exec_df['trades_per_day'].mean():.1f}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**换手率轨迹**")
                st.line_chart(exec_df[["date", "turnover"]].set_index("date"))
            with col2:
                st.markdown("**重叠率与调仓笔数**")
                if (
                    "overlap_long" in exec_df.columns
                    and "overlap_short" in exec_df.columns
                ):
                    overlap_avg = exec_df[["overlap_long", "overlap_short"]].mean(
                        axis=1
                    )
                    plot_df = pd.DataFrame(
                        {
                            "date": exec_df["date"],
                            "avg_overlap": overlap_avg,
                            "trades_per_day": exec_df["trades_per_day"],
                        }
                    ).set_index("date")
                else:
                    est_overlap = 1.0 - exec_df["turnover"] / 2.0
                    plot_df = pd.DataFrame(
                        {
                            "date": exec_df["date"],
                            "avg_overlap": est_overlap,
                            "trades_per_day": exec_df["trades_per_day"],
                        }
                    ).set_index("date")
                st.line_chart(plot_df)
        else:
            st.info('尚无执行后时序文件。请在"一键运行预测流水线"启用执行策略。')

with tab_stock:
    # 个股查询
    col_head1, col_head2 = st.columns([0.9, 0.1])
    with col_head1:
        st.subheader("🔎 个股查询（信号与持仓轨迹）")
    with col_head2:
        info_icon("查询指定股票的预测分数、是否入选多/空持仓、行业内分位与制度背景。需选择预测文件与TopN/BottomN参数。")

    with st.form("single_stock_form"):
        # 加载股票池信息
        stock_universe = load_stock_universe()

        # 第一行：预测文件选择 + 归档开关 + 仅可用
        c_pred, c_arch, c_only = st.columns([0.65, 0.15, 0.20])
        include_arch_state = st.session_state.get("cfg_include_arch_preds", False)
        with c_pred:
            pred_files_ss = list_predictions(include_archive=include_arch_state)
            pred_map_ss = {p.name: p for p in pred_files_ss}
            pred_choice_ss = st.selectbox(
                "预测文件",
                list(pred_map_ss.keys()) if pred_map_ss else ["(无)"],
                help="用于计算每日排序与入选情况",
            )
        with c_arch:
            st.markdown('<div style="height: 38px;"></div>', unsafe_allow_html=True)
            include_arch = st.checkbox(
                "归档",
                value=include_arch_state,
                key="cfg_include_arch_preds",
                help="包含 data/archive 下的旧预测文件",
            )
        with c_only:
            st.markdown('<div style="height: 38px;"></div>', unsafe_allow_html=True)
            # 根据预测文件可选股票进行过滤（默认只显示有数据的）
            only_avail = st.checkbox(
                "仅可用", value=True, key="cfg_stock_only_avail", help="仅显示当前预测文件中存在数据的股票"
            )

        # 第二行：选择股票 + TopN + BottomN
        c_code, c_topn, c_botn = st.columns([0.50, 0.25, 0.25])
        with c_code:
            avail_codes = None
            if pred_map_ss and pred_choice_ss != "(无)":
                try:
                    avail_codes = set(
                        list_codes_in_predictions(str(pred_map_ss[pred_choice_ss]))
                    )
                except Exception:
                    avail_codes = None
            if not stock_universe.empty:
                # 构建“代码 (名称)”显示
                stock_map = {}
                for _, row in stock_universe.iterrows():
                    code = str(row["code"]).zfill(6)
                    if (
                        only_avail
                        and avail_codes is not None
                        and code not in avail_codes
                    ):
                        continue
                    name = (
                        str(row["name"])
                        if "name" in row and pd.notna(row["name"])
                        else code
                    )
                    stock_map[f"{code} ({name})"] = code
                options = list(stock_map.keys())
                if not options:  # 回退：无匹配时显示所有股票
                    stock_map = {
                        f"{str(row['code']).zfill(6)} ({row['name'] if 'name' in row else ''})": str(
                            row["code"]
                        ).zfill(
                            6
                        )
                        for _, row in stock_universe.iterrows()
                    }
                    options = list(stock_map.keys())
                # 默认选择（优先选择预测文件中第一个可用股票）
                default_idx = 0
                if avail_codes and only_avail:
                    # 如果有可用股票列表，选择其中第一个
                    first_avail = (
                        sorted(list(avail_codes))[0] if avail_codes else "000002"
                    )
                    for i, (k, v) in enumerate(stock_map.items()):
                        if v == first_avail:
                            default_idx = i
                            break
                else:
                    # 否则尝试选择000001，不存在则选择第一个
                    for i, (k, v) in enumerate(stock_map.items()):
                        if v == "000001":
                            default_idx = i
                            break
                stock_choice = st.selectbox(
                    "选择股票", options, index=default_idx, help="仅显示可查询的股票（可关闭过滤）"
                )
                code_input = stock_map.get(stock_choice, "000001")
            else:
                code_input = st.text_input("股票代码", "000001", help="6位数字代码，例如 000001")
        with c_topn:
            topn_ss = st.number_input(
                "TopN(多头)",
                min_value=5,
                max_value=80,
                value=30,
                step=5,
                help="每日分数最高的前N只进入多头",
            )
        with c_botn:
            botn_ss = st.number_input(
                "BottomN(空头)",
                min_value=5,
                max_value=80,
                value=30,
                step=5,
                help="每日分数最低的后N只进入空头",
            )
        submit_ss = st.form_submit_button("🔍 查询")

    # 显示股票池统计信息
    if not stock_universe.empty:
        st.markdown("**股票池概览**")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("股票总数", len(stock_universe))
        with col_stats2:
            if "行业" in stock_universe.columns and not stock_universe["行业"].isna().all():
                unique_industries = stock_universe["行业"].dropna().nunique()
                st.metric("行业数量", unique_industries)
            else:
                st.metric("行业数量", "--")
        with col_stats3:
            if (
                "市值排名" in stock_universe.columns
                and not stock_universe["市值排名"].isna().all()
            ):
                avg_rank = stock_universe["市值排名"].mean()
                st.metric("平均市值排名", f"{avg_rank:.0f}")
            else:
                st.metric("平均市值排名", "--")

        # 显示前10大股票
        with st.expander("📋 查看股票池前10大股票", expanded=False):
            display_cols = ["code", "name"]
            if "市值排名" in stock_universe.columns:
                display_cols.append("市值排名")
            if "总市值_亿元" in stock_universe.columns:
                display_cols.append("总市值_亿元")
            if "行业" in stock_universe.columns:
                display_cols.append("行业")

            top_stocks = stock_universe.head(10)[display_cols].copy()
            if "code" in top_stocks.columns:
                top_stocks["code"] = top_stocks["code"].astype(str).str.zfill(6)
            st.dataframe(top_stocks, use_container_width=True)
    else:
        st.warning("未找到股票池文件 stock_universe.csv，请确保文件存在。")

if submit_ss and pred_map_ss and pred_choice_ss != "(无)":
    try:
        pred_path_obj = pred_map_ss[pred_choice_ss]
        pred_path_str = str(pred_path_obj)
        code = str(code_input).zfill(6)
        # 先尝试读取最新制度文件（包含 industry 与 regime）
        regime_path = find_latest_regime_file()
        regime_df = None
        if regime_path:
            regime_df = pd.read_csv(
                regime_path, usecols=["date", "stock_code", "industry", "regime"]
            )
            regime_df["date"] = pd.to_datetime(regime_df["date"])
            regime_df["stock_code"] = regime_df["stock_code"].astype(str).str.zfill(6)

        # 文件较大时，采用分块法，仅提取该股逐日分数与入选标志（同时合并行业/制度，计算行业分位）
        if pred_path_obj.stat().st_size > 80 * 1024 * 1024:  # >80MB
            one = derive_stock_series_from_predictions(
                pred_path_str, code, int(topn_ss), int(botn_ss), regime_df=regime_df
            )
            dfp = None
        else:
            dfp = load_predictions_df(pred_path_str)
            # 计算当日排序与入选标志
            dfp["rank_desc"] = dfp.groupby("date")["y_pred"].rank(
                ascending=False, method="first"
            )
            counts = dfp.groupby("date")["stock_code"].transform("count")
            dfp["in_long"] = dfp["rank_desc"] <= topn_ss
            dfp["in_short"] = dfp["rank_desc"] > (counts - botn_ss)
            # 合并制度/行业
            if regime_df is not None and not dfp.empty:
                dfp = dfp.merge(regime_df, on=["date", "stock_code"], how="left")
            # 行业内分位（使用当日同行业分数分位）
            if "industry" in dfp.columns:
                dfp["ind_count"] = dfp.groupby(["date", "industry"])[
                    "stock_code"
                ].transform("count")
                dfp["ind_rank"] = dfp.groupby(["date", "industry"])["y_pred"].rank(
                    ascending=False, method="min"
                )
                dfp["ind_rank_pct"] = 1.0 - (dfp["ind_rank"] - 1) / dfp[
                    "ind_count"
                ].clip(lower=1)
        # 取目标股票
        if dfp is not None:
            one = (
                dfp[dfp["stock_code"] == code]
                .sort_values("date")
                .reset_index(drop=True)
            )
        if one.empty:
            st.warning("所选预测文件中未找到该股票。")
        else:
            # 记住本次查询结果用于后续勾选切换时复用
            st.session_state["ss_one"] = one.copy()
            st.session_state["ss_code"] = code
            st.session_state["ss_pred_path"] = pred_path_str
            # 读取价格
            price_path = DATA_DIR / f"{code}.csv"
            price_df = None
            if price_path.exists():
                try:
                    price_df = pd.read_csv(price_path)
                    price_df["date"] = pd.to_datetime(price_df["date"])
                except Exception:
                    price_df = None

            # 解析所属行业（优先 one，其次映射，再次股票池）
            stock_industry = None
            try:
                if "industry" in one.columns and not one["industry"].dropna().empty:
                    stock_industry = str(one["industry"].dropna().iloc[-1])
                if not stock_industry or stock_industry in ["未分类", "None", "nan", ""]:
                    import pandas as _pd

                    mp_path = DATA_DIR / "industry_mapping.csv"
                    if mp_path.exists():
                        mp = _pd.read_csv(mp_path, dtype={"code": str})
                        ind_col = (
                            "industry"
                            if "industry" in mp.columns
                            else ("行业" if "行业" in mp.columns else None)
                        )
                        if ind_col:
                            m = {
                                str(r["code"]).zfill(6): str(r[ind_col])
                                for _, r in mp.iterrows()
                            }
                            stock_industry = m.get(code) or stock_industry
                if (not stock_industry) or stock_industry in ["未分类", "None", "nan", ""]:
                    import pandas as _pd

                    uni_path = DATA_DIR.parent / "stock_universe.csv"
                    if uni_path.exists():
                        uni = _pd.read_csv(uni_path, dtype={"code": str})
                        ind_col = (
                            "industry"
                            if "industry" in uni.columns
                            else ("行业" if "行业" in uni.columns else None)
                        )
                        if ind_col:
                            m2 = {
                                str(r["code"]).zfill(6): str(r[ind_col])
                                for _, r in uni.iterrows()
                            }
                            stock_industry = m2.get(code) or stock_industry
            except Exception:
                pass
            if not stock_industry:
                stock_industry = "未分类"

            # 概览指标
            lastN = one.tail(252)
            long_days = int(lastN["in_long"].sum())
            short_days = int(lastN["in_short"].sum())
            colsS = st.columns(6)
            colsS[0].metric("近一年多头入选天数", f"{long_days}")
            colsS[1].metric("近一年空头入选天数", f"{short_days}")
            colsS[2].metric("最新分数", f"{one['y_pred'].iloc[-1]:.6f}")
            if "ind_rank_pct" in one.columns and not one["ind_rank_pct"].isna().all():
                colsS[3].metric("行业内最新分位", f"{one['ind_rank_pct'].iloc[-1]*100:.1f}%")
            if "regime" in one.columns and not one["regime"].isna().all():
                colsS[4].metric("最新制度", str(one["regime"].iloc[-1]))
            colsS[5].metric("所属行业", stock_industry)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**价格与信号图表**")
                # 轻量的参数面板（横向排布）
                t1, t2, t3, t4, t5 = st.columns(5)
                with t1:
                    show_regime = st.checkbox("制度轴", True, key="opt_regime")
                with t2:
                    show_score = st.checkbox("分数线", True, key="opt_score")
                with t3:
                    show_buy = st.checkbox("买入", True, key="opt_buy")
                with t4:
                    show_short = st.checkbox("做空", True, key="opt_short")
                with t5:
                    show_close = st.checkbox("清仓", True, key="opt_close")
                fig = render_price_signal_chart_new(
                    one,
                    price_df,
                    regime_df,
                    show_regime,
                    show_score,
                    show_buy,
                    show_short,
                    show_close,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{code}")
            with c2:
                st.markdown("**入选轨迹（1=多，-1=空）**")
                flag = one[["date", "in_long", "in_short"]].copy()
                flag["pos"] = np.where(
                    flag["in_long"], 1, np.where(flag["in_short"], -1, 0)
                )
                st.area_chart(flag[["date", "pos"]].set_index("date"))

            # 近期明细
            show_cols = ["date", "y_pred", "in_long", "in_short"]
            if "ind_rank_pct" in one.columns:
                show_cols.append("ind_rank_pct")
            if "regime" in one.columns:
                show_cols.append("regime")
            st.markdown("**近期明细（最近30天）**")
            disp = one[show_cols].tail(30).copy()
            if "ind_rank_pct" in disp.columns:
                disp["ind_rank_pct"] = disp["ind_rank_pct"].map(
                    lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
                )
            st.dataframe(disp)

            # 行业内分位曲线与事件表
            st.markdown("**行业内分位曲线（0-100%）**")
            if "ind_rank_pct" in one.columns and not one["ind_rank_pct"].isna().all():
                tmp = one[["date", "ind_rank_pct"]].dropna()
                tmp["ind_rank_pct"] = tmp["ind_rank_pct"] * 100
                st.line_chart(tmp.set_index("date"))
            else:
                if regime_df is None:
                    st.info("缺少行业分位数据（未找到制度文件）。可先运行“重算制度”。")
                else:
                    st.info("缺少行业分位数据（该股票或日期无行业信息）。")

            st.markdown("**进出场事件表（最近90天）**")
            ev = one[["date", "in_long", "in_short"]].copy()
            ev["pos"] = np.where(ev["in_long"], 1, np.where(ev["in_short"], -1, 0))
            ev["prev"] = ev["pos"].shift(1).fillna(0)
            ev = ev[ev["pos"] != ev["prev"]]
            ev = ev.tail(90)
            if not ev.empty:
                ev["event"] = ev["pos"].map({1: "进入多头", -1: "进入空头", 0: "移出持仓"})
                st.dataframe(ev[["date", "event"]])
            else:
                st.info("近90日无进出场变化。")

            # 调试信息（帮助定位“分数恒为-0.01”或制度上色异常）
            with st.expander("🛠 调试信息（仅开发用）", expanded=False):
                try:
                    st.write("预测文件：", pred_choice_ss)
                    st.write(
                        "样本区间：",
                        str(one["date"].min().date()) if not one.empty else "-",
                        "→",
                        str(one["date"].max().date()) if not one.empty else "-",
                    )
                    if "y_pred" in one.columns and not one.empty:
                        st.write(
                            "y_pred 统计：min=",
                            float(one["y_pred"].min()),
                            " max=",
                            float(one["y_pred"].max()),
                            " mean=",
                            float(one["y_pred"].mean()),
                        )
                        st.write("最近5天：")
                        st.dataframe(
                            one[["date", "y_pred", "in_long", "in_short"]].tail(5)
                        )
                    if "regime" in one.columns:
                        st.write("regime 非空天数：", int(one["regime"].notna().sum()))
                except Exception as _e:
                    st.write("调试面板异常：", str(_e))

            # 行业画像与同业对比
            with st.expander("🏷 行业画像与同业对比（按最新交易日）", expanded=False):
                try:
                    # 1) 确定该股票的行业（优先使用时序中携带的行业；否则从映射/股票池兜底）
                    stock_industry = None
                    if "industry" in one.columns and not one["industry"].dropna().empty:
                        stock_industry = str(one["industry"].dropna().iloc[-1])
                    if not stock_industry or stock_industry in [
                        "未分类",
                        "None",
                        "nan",
                        "",
                    ]:
                        try:
                            import pandas as _pd

                            mp = _pd.read_csv(
                                DATA_DIR / "industry_mapping.csv", dtype={"code": str}
                            )
                            ind_col = (
                                "industry"
                                if "industry" in mp.columns
                                else ("行业" if "行业" in mp.columns else None)
                            )
                            if ind_col:
                                m = {
                                    str(r["code"]).zfill(6): str(r[ind_col])
                                    for _, r in mp.iterrows()
                                }
                                stock_industry = m.get(code) or stock_industry
                        except Exception:
                            pass
                    if (not stock_industry) or stock_industry in [
                        "未分类",
                        "None",
                        "nan",
                        "",
                    ]:
                        try:
                            import pandas as _pd

                            uni = _pd.read_csv(
                                DATA_DIR.parent / "stock_universe.csv",
                                dtype={"code": str},
                            )
                            ind_col = (
                                "industry"
                                if "industry" in uni.columns
                                else ("行业" if "行业" in uni.columns else None)
                            )
                            if ind_col:
                                m = {
                                    str(r["code"]).zfill(6): str(r[ind_col])
                                    for _, r in uni.iterrows()
                                }
                                stock_industry = m.get(code) or stock_industry
                        except Exception:
                            pass

                    if not stock_industry or stock_industry in [
                        "未分类",
                        "None",
                        "nan",
                        "",
                    ]:
                        st.info("当前股票缺少行业信息（映射兜底也为空），暂无法生成同业对比。")
                    else:
                        target_date = pd.to_datetime(one["date"].max())
                        st.write("行业：", stock_industry, " | 日期：", target_date.date())

                        # 获取目标日的全量预测（优先使用已加载的 dfp；否则分块读取）
                        peers_all = None
                        if "dfp" in locals() and dfp is not None:
                            peers_all = dfp[dfp["date"] == target_date].copy()
                        else:
                            # 分块读取目标日数据
                            cols_need = ["date", "stock_code", "y_pred"]
                            extra_cols = []
                            if regime_df is not None:
                                extra_cols = []
                            tmp_rows = []
                            for chunk in pd.read_csv(
                                pred_path_str,
                                usecols=lambda c: c in set(cols_need + ["industry"]),
                                chunksize=300000,
                            ):
                                chunk["date"] = pd.to_datetime(chunk["date"])
                                chunk["stock_code"] = (
                                    chunk["stock_code"].astype(str).str.zfill(6)
                                )
                                day = chunk[chunk["date"] == target_date].copy()
                                if day.empty:
                                    continue
                                # 若无行业列，尝试和 regime_df 在该日合并
                                if (
                                    "industry" not in day.columns
                                    or day["industry"].isna().all()
                                ):
                                    if regime_df is not None:
                                        reg_day = regime_df[
                                            regime_df["date"] == target_date
                                        ][["date", "stock_code", "industry"]]
                                        day = day.merge(
                                            reg_day,
                                            on=["date", "stock_code"],
                                            how="left",
                                        )
                                tmp_rows.append(day)
                            if tmp_rows:
                                peers_all = pd.concat(tmp_rows, ignore_index=True)

                        if peers_all is None or peers_all.empty:
                            st.info("未能加载目标日的全量预测，无法生成同业对比。可选择较小的预测文件或在侧边栏重新运行流水线。")
                        else:
                            # 补齐行业信息：优先使用 regime_df 最近<=目标日的行业，其次使用 industry_mapping.csv，再其次 stock_universe.csv
                            try:
                                need_fill = (
                                    "industry" not in peers_all.columns
                                ) or peers_all["industry"].isna().all()
                            except Exception:
                                need_fill = True
                            if need_fill:
                                # 1) 使用 regime_df 最近日期的行业
                                try:
                                    if regime_df is not None and not regime_df.empty:
                                        reg_upto = regime_df[
                                            regime_df["date"] <= target_date
                                        ][["stock_code", "industry", "date"]].copy()
                                        if not reg_upto.empty:
                                            reg_upto = (
                                                reg_upto.sort_values(
                                                    ["stock_code", "date"]
                                                )
                                                .groupby("stock_code", as_index=False)
                                                .tail(1)[["stock_code", "industry"]]
                                            )
                                            peers_all = peers_all.merge(
                                                reg_upto, on="stock_code", how="left"
                                            )
                                except Exception:
                                    pass
                                # 2) 使用 industry_mapping.csv
                                try:
                                    import pandas as _pd

                                    mp = _pd.read_csv(
                                        DATA_DIR / "industry_mapping.csv",
                                        dtype={"code": str},
                                    )
                                    ind_col = (
                                        "industry"
                                        if "industry" in mp.columns
                                        else ("行业" if "行业" in mp.columns else None)
                                    )
                                    if ind_col:
                                        m = {
                                            str(r["code"]).zfill(6): str(r[ind_col])
                                            for _, r in mp.iterrows()
                                        }
                                        if "industry" not in peers_all.columns:
                                            peers_all["industry"] = peers_all[
                                                "stock_code"
                                            ].map(m)
                                        else:
                                            peers_all["industry"] = peers_all[
                                                "industry"
                                            ].fillna(peers_all["stock_code"].map(m))
                                except Exception:
                                    pass
                                # 3) 使用 stock_universe.csv
                                try:
                                    import pandas as _pd

                                    uni = _pd.read_csv(
                                        DATA_DIR.parent / "stock_universe.csv",
                                        dtype={"code": str},
                                    )
                                    ind_col = (
                                        "industry"
                                        if "industry" in uni.columns
                                        else ("行业" if "行业" in uni.columns else None)
                                    )
                                    if ind_col:
                                        m2 = {
                                            str(r["code"]).zfill(6): str(r[ind_col])
                                            for _, r in uni.iterrows()
                                        }
                                        if "industry" not in peers_all.columns:
                                            peers_all["industry"] = peers_all[
                                                "stock_code"
                                            ].map(m2)
                                        else:
                                            peers_all["industry"] = peers_all[
                                                "industry"
                                            ].fillna(peers_all["stock_code"].map(m2))
                                except Exception:
                                    pass

                            # 仅该行业
                            if (
                                "industry" not in peers_all.columns
                                or peers_all["industry"].isna().all()
                            ):
                                st.warning("目标日预测缺少行业列，且回退合并与映射均失败。")
                            peers = peers_all.copy()
                            peers["industry"] = peers.get("industry")
                            peers = peers[peers["industry"] == stock_industry].copy()
                            if peers.empty:
                                st.info("目标日在该行业下无同业数据。")
                            else:
                                peers["rank"] = peers["y_pred"].rank(
                                    ascending=False, method="min"
                                )
                                peers = peers.sort_values("rank")
                                # 定位本股票排名
                                me = peers[peers["stock_code"] == code]
                                if not me.empty:
                                    my_rank = int(me["rank"].iloc[0])
                                    st.write(
                                        f"本股票在行业内排名：第 {my_rank} 名 / 共 {len(peers)} 个样本"
                                    )
                                # 展示Top/Bottom
                                ctop, cbottom = st.columns(2)
                                with ctop:
                                    st.markdown("**行业 Top10（按 y_pred）**")
                                    st.dataframe(
                                        peers[["stock_code", "y_pred", "rank"]]
                                        .head(10)
                                        .reset_index(drop=True)
                                    )
                                with cbottom:
                                    st.markdown("**行业 Bottom10（按 y_pred）**")
                                    st.dataframe(
                                        peers[["stock_code", "y_pred", "rank"]]
                                        .tail(10)
                                        .reset_index(drop=True)
                                    )
                                # 导出
                                csv_out = peers[
                                    ["stock_code", "industry", "y_pred", "rank"]
                                ].to_csv(index=False)
                                st.download_button(
                                    "⬇️ 下载行业同业对比CSV",
                                    data=csv_out,
                                    file_name=f"peers_{code}_{target_date.date()}.csv",
                                    mime="text/csv",
                                )
                except Exception as _e:
                    st.error(f"行业对比生成失败：{_e}")
    except Exception as e:
        st.error(f"个股查询失败：{e}")
elif st.session_state.get("ss_one") is not None:
    # 复用上次查询结果，勾选切换时不清空
    try:
        one = st.session_state["ss_one"]
        code = st.session_state.get("ss_code", "000001")
        # 加载价格
        price_df = None
        price_path = DATA_DIR / f"{code}.csv"
        if price_path.exists():
            price_df = pd.read_csv(price_path)
            price_df["date"] = pd.to_datetime(price_df["date"])
        # 可能的制度文件
        try:
            from predictive_model import find_latest_regime_file as _find_reg

            rp = _find_reg()
            regime_df = None
            if rp:
                regime_df = pd.read_csv(
                    rp, usecols=["date", "stock_code", "industry", "regime"]
                )
                regime_df["date"] = pd.to_datetime(regime_df["date"])
                regime_df["stock_code"] = (
                    regime_df["stock_code"].astype(str).str.zfill(6)
                )
        except Exception:
            regime_df = None
        # 概览指标（复用缓存时同样显示所属行业）
        stock_industry = None
        try:
            if "industry" in one.columns and not one["industry"].dropna().empty:
                stock_industry = str(one["industry"].dropna().iloc[-1])
            if not stock_industry or stock_industry in ["未分类", "None", "nan", ""]:
                import pandas as _pd

                mp_path = DATA_DIR / "industry_mapping.csv"
                if mp_path.exists():
                    mp = _pd.read_csv(mp_path, dtype={"code": str})
                    ind_col = (
                        "industry"
                        if "industry" in mp.columns
                        else ("行业" if "行业" in mp.columns else None)
                    )
                    if ind_col:
                        m = {
                            str(r["code"]).zfill(6): str(r[ind_col])
                            for _, r in mp.iterrows()
                        }
                        stock_industry = m.get(code) or stock_industry
            if (not stock_industry) or stock_industry in ["未分类", "None", "nan", ""]:
                import pandas as _pd

                uni_path = DATA_DIR.parent / "stock_universe.csv"
                if uni_path.exists():
                    uni = _pd.read_csv(uni_path, dtype={"code": str})
                    ind_col = (
                        "industry"
                        if "industry" in uni.columns
                        else ("行业" if "行业" in uni.columns else None)
                    )
                    if ind_col:
                        m2 = {
                            str(r["code"]).zfill(6): str(r[ind_col])
                            for _, r in uni.iterrows()
                        }
                        stock_industry = m2.get(code) or stock_industry
        except Exception:
            pass
        if not stock_industry:
            stock_industry = "未分类"

        lastN = one.tail(252)
        long_days = int(lastN["in_long"].sum())
        short_days = int(lastN["in_short"].sum())
        colsS = st.columns(6)
        colsS[0].metric("近一年多头入选天数", f"{long_days}")
        colsS[1].metric("近一年空头入选天数", f"{short_days}")
        colsS[2].metric("最新分数", f"{one['y_pred'].iloc[-1]:.6f}")
        if "ind_rank_pct" in one.columns and not one["ind_rank_pct"].isna().all():
            colsS[3].metric("行业内最新分位", f"{one['ind_rank_pct'].iloc[-1]*100:.1f}%")
        if "regime" in one.columns and not one["regime"].isna().all():
            colsS[4].metric("最新制度", str(one["regime"].iloc[-1]))
        colsS[5].metric("所属行业", stock_industry)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**价格与信号图表**")
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                show_regime = st.checkbox("制度轴", True, key="opt_regime")
            with t2:
                show_score = st.checkbox("分数线", True, key="opt_score")
            with t3:
                show_buy = st.checkbox("买入", True, key="opt_buy")
            with t4:
                show_short = st.checkbox("做空", True, key="opt_short")
            with t5:
                show_close = st.checkbox("清仓", True, key="opt_close")
            fig = render_price_signal_chart_new(
                one,
                price_df,
                regime_df,
                show_regime,
                show_score,
                show_buy,
                show_short,
                show_close,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"main_chart_{code}")
        with c2:
            st.markdown("**入选轨迹（1=多，-1=空）**")
            flag = one[["date", "in_long", "in_short"]].copy()
            flag["pos"] = np.where(
                flag["in_long"], 1, np.where(flag["in_short"], -1, 0)
            )
            st.area_chart(flag[["date", "pos"]].set_index("date"))
    except Exception as e:
        st.error(f"个股查询复用缓存失败：{e}")
elif submit_ss and (not pred_map_ss or pred_choice_ss == "(无)"):
    st.warning("未找到可用的预测文件。请先运行侧边栏“运行流水线”，或勾选“包含归档”后重试。")

    # 数据导入（价格CSV）
    st.markdown("---")
    st.subheader("📥 数据导入（价格CSV）")
    with st.form("data_import_form_in_tab"):
        code_in = st.text_input("股票代码(6位)", "")
        file_in = st.file_uploader("选择CSV文件", type=["csv"], key="file_upload_stock")
        st.caption("期望包含列：date, open, high, low, close（缺失则回退为仅close线）")
        submit_import2 = st.form_submit_button("上传并保存到 data/")
    if submit_import2 and code_in and file_in is not None:
        try:
            df_imp = pd.read_csv(file_in)
            # 基本字段校验
            if "date" not in df_imp.columns:
                st.error("导入失败：缺少必需列 date")
            elif not (
                {"close"} <= set(df_imp.columns)
                or {"open", "high", "low", "close"} <= set(df_imp.columns)
            ):
                st.error("导入失败：至少需要 close 列（或完整 OHLC 列）")
            else:
                df_imp["date"] = pd.to_datetime(df_imp["date"], errors="coerce")
                df_imp = df_imp.dropna(subset=["date"])
                outp = DATA_DIR / f"{str(code_in).zfill(6)}.csv"
                df_imp.to_csv(outp, index=False, encoding="utf-8-sig")
                st.success(f"已保存至: {outp.name}")
        except Exception as e:
            st.error(f"导入失败: {e}")
with tab_pack:
    # 报告打包与下载
    st.subheader("🧷 报告打包与下载")
with tab_pack:
    with st.expander("📦 高级：自定义打包内容"):
        with st.form("report_pack_form"):
            pred_glob = st.text_input("预测文件(Glob，可空)", "")
            exec_metrics_glob = st.text_input(
                "执行指标JSON(Glob)", "data/pipeline_execution_*_metrics.json"
            )
            exec_ts_glob = st.text_input(
                "执行时序CSV(Glob)", "data/pipeline_execution_*_timeseries.csv"
            )
            grid_glob = st.text_input(
                "换手率网格CSV(Glob)", "data/turnover_strategy_grid_*.csv"
            )
            rb_glob = st.text_input("稳健性汇总CSV(Glob，可空)", "")
            visuals_glob = st.text_input(
                "可选图像(Globs, 逗号分隔)",
                "data/turnover_*.png,data/exec_profile_*_profile.png",
            )
            submit_pack = st.form_submit_button("📦 生成报告包")

if submit_pack:
    cmd = ["python3", "analysis/report_packager.py"]
    if pred_glob.strip():
        cmd.extend(["--predictions", pred_glob.strip()])
    if exec_metrics_glob.strip():
        cmd.extend(["--execution-metrics", exec_metrics_glob.strip()])
    if exec_ts_glob.strip():
        cmd.extend(["--execution-timeseries", exec_ts_glob.strip()])
    if grid_glob.strip():
        cmd.extend(["--turnover-grid", grid_glob.strip()])
    if rb_glob.strip():
        cmd.extend(["--robustness", rb_glob.strip()])
    if visuals_glob.strip():
        cmd.extend(["--visuals", visuals_glob.strip()])
    with st.spinner("正在生成报告包..."):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            with tab_pack:
                st.success("报告包已生成")
                st.text(proc.stdout)
        except subprocess.CalledProcessError as e:
            with tab_pack:
                st.error("报告打包失败")
                st.text_area(
                    "错误日志", (e.stdout or "") + "\n" + (e.stderr or ""), height=280
                )

# 自动发现并提供下载
with tab_pack:
    pkg_files = sorted(
        DATA_DIR.glob("report_package_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pkg_files:
        latest_pkg = pkg_files[0]
        with open(latest_pkg, "rb") as f:
            st.download_button(
                "⬇️ 下载最新报告包",
                data=f.read(),
                file_name=latest_pkg.name,
                mime="application/zip",
            )
    else:
        st.info("尚无报告包。请先生成。")

with tab_pack:
    st.markdown("---")
    st.subheader("🗂️ 配置快照（保存/加载）")
    snaps = _list_snapshots()
    if snaps:
        snap_map = {p.name: p for p in snaps}
        sel = st.selectbox("选择快照以加载", list(snap_map.keys()), index=0)
        if st.button("📥 加载并应用快照"):
            import json

            with open(snap_map[sel], "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # 写回 session_state 并重载页面
            assign = {
                "cfg_standardisation": cfg.get("standardisation"),
                "cfg_start_oos": cfg.get("start_oos"),
                "cfg_train_window": cfg.get("train_window"),
                "cfg_alpha": cfg.get("alpha"),
                "cfg_top_n": cfg.get("top_n"),
                "cfg_bottom_n": cfg.get("bottom_n"),
                "cfg_cost_bps_ui": cfg.get("cost_bps_ui"),
                "cfg_neutral_shrink": cfg.get("neutral_shrink"),
                "cfg_neutral_industries": cfg.get("neutral_industries"),
                "cfg_exec_strategy": cfg.get("exec_strategy"),
                "cfg_delta": cfg.get("delta"),
                "cfg_ema_span": cfg.get("ema_span"),
                "cfg_k": cfg.get("k"),
                "cfg_swap_cap": cfg.get("swap_cap"),
                "cfg_run_cost_grid": cfg.get("run_cost_grid"),
                "cfg_run_turnover_grid": cfg.get("run_turnover_grid"),
                "cfg_recompute_factors": cfg.get("recompute_factors"),
                "cfg_recompute_regime": cfg.get("recompute_regime"),
            }
            for k, v in assign.items():
                if v is not None:
                    st.session_state[k] = v
            st.success("已加载快照，正在应用...")
            st.rerun()
    else:
        st.info("尚未保存任何快照。可在侧边栏“保存快照”创建。")
