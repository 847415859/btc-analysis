# -*- coding: utf-8 -*-
"""
多币种多周期技术分析工具
数据来源: Binance 公开 API
周期: 15分钟 / 30分钟 / 1小时 / 2小时 / 4小时 / 8小时 / 日线 / 周线
技术指标: MA/EMA, RSI, Bollinger Bands, MACD, KDJ, OBV, Fibonacci, 支撑/阻力
输出: 交互式多标签 HTML 报告 + TXT 纯文本报告
"""

import os
import sys
import io
import time

# 统一输出编码为 UTF-8，避免 Windows GBK 下 emoji 报错
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# ─────────────────────────────────────────────
# 周期配置
# ─────────────────────────────────────────────

TIMEFRAMES = [
    # (interval, limit, label_cn, chart_show, sr_lookback, sr_pivot_n, sr_cluster_pct, fib_lookback)
    ("15m", 500, "15分钟",  250, 300, 3, 0.003, 50),
    ("30m", 500, "30分钟",  280, 300, 3, 0.004, 80),
    ("1h",  500, "1小时",   300, 300, 4, 0.006, 100),
    ("2h",  500, "2小时",   300, 300, 4, 0.008, 120),
    ("4h",  500, "4小时",   300, 300, 4, 0.010, 150),
    ("8h",  500, "8小时",   300, 300, 4, 0.012, 150),
    ("1d",  500, "日线",    300, 300, 5, 0.015, 150),
    ("1w",  200, "周线",    150, 100, 3, 0.025, 52),
]

# SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
SYMBOLS = ["BTCUSDT"]


# ─────────────────────────────────────────────
# A. 数据获取
# ─────────────────────────────────────────────

def fetch_data(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """通过 Binance API 获取 K 线数据（含 429 限流重试）"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"[限流] {symbol} {interval} 触发 429，等待 {wait}s (第 {attempt+1}/3 次)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"[错误] 无法获取 {symbol} {interval} 数据: {e}")
            return pd.DataFrame()
    else:
        print(f"[错误] {symbol} {interval} 超过重试上限，返回空数据")
        return pd.DataFrame()
    if resp is None:
        return pd.DataFrame()

    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["date", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# B. 技术指标计算
# ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    df["ma20"]  = ta.trend.sma_indicator(df["close"], window=20)
    df["ma50"]  = ta.trend.sma_indicator(df["close"], window=50)
    df["ma200"] = ta.trend.sma_indicator(df["close"], window=200)
    df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema50"] = ta.trend.ema_indicator(df["close"], window=50)

    rsi = ta.momentum.RSIIndicator(df["close"], window=14)
    df["rsi"] = rsi.rsi()

    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    macd = ta.trend.MACD(df["close"], window_fast=12, window_slow=26, window_sign=9)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"]   = macd.macd_diff()

    stoch = ta.momentum.StochasticOscillator(
        df["high"], df["low"], df["close"], window=9, smooth_window=3
    )
    df["kdj_k"] = stoch.stoch()
    df["kdj_d"] = stoch.stoch_signal()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv"] = obv.on_balance_volume()

    # ── VWAP（按日重置）──────────────────────────────
    tp = (df["high"] + df["low"] + df["close"]) / 3
    dates = df["date"].dt.date
    df["vwap"] = (
        (tp * df["volume"]).groupby(dates).cumsum()
        / df["volume"].groupby(dates).cumsum()
    )

    # ── ATR(14) + 止损线（1.5×ATR）────────────────────
    atr_ind = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    )
    df["atr"]          = atr_ind.average_true_range()
    df["atr_sl_long"]  = df["close"] - 1.5 * df["atr"]   # 做多止损线（价格下方）
    df["atr_sl_short"] = df["close"] + 1.5 * df["atr"]   # 做空止损线（价格上方）

    # ── 波动率百分位（过去 100 根）→ 自适应参数 ──────────
    atr_rank = df["atr"].rolling(100).rank(pct=True)
    _atr_rank_last = atr_rank.iloc[-1]
    high_vol = (not pd.isna(_atr_rank_last)) and (_atr_rank_last >= 0.75)

    bb_dev  = 2.5 if high_vol else 2.0
    rsi_win = 21  if high_vol else 14

    # 用自适应参数重新计算（覆盖前面固定参数版本）
    rsi2 = ta.momentum.RSIIndicator(df["close"], window=rsi_win)
    df["rsi"] = rsi2.rsi()
    bb2 = ta.volatility.BollingerBands(df["close"], window=20, window_dev=bb_dev)
    df["bb_upper"] = bb2.bollinger_hband()
    df["bb_mid"]   = bb2.bollinger_mavg()
    df["bb_lower"] = bb2.bollinger_lband()

    # ── ADX + DI(14) ─────────────────────────────────────
    adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
    df["adx"]     = adx_ind.adx()
    df["adx_pos"] = adx_ind.adx_pos()   # DI+
    df["adx_neg"] = adx_ind.adx_neg()   # DI-

    # 将自适应参数存入 DataFrame.attrs（供 generate_report 读取）
    df.attrs["high_vol"]     = bool(high_vol)
    df.attrs["bb_dev"]       = bb_dev
    df.attrs["rsi_win"]      = rsi_win
    df.attrs["atr_pct_rank"] = float(_atr_rank_last) if not pd.isna(_atr_rank_last) else 0.5

    # ── CVD / OFI（主动成交量差值分析）──────────────────────────────
    # taker_buy_base = 主动买方成交量（吃单买入），由 K 线 API 直接提供
    # delta > 0 → 主动买盘主导；delta < 0 → 主动卖盘主导
    if "taker_buy_base" in df.columns:
        df["taker_buy"]  = df["taker_buy_base"].astype(float)
        df["taker_sell"] = df["volume"] - df["taker_buy"]
        df["delta"]      = df["taker_buy"] - df["taker_sell"]
        df["cvd"]        = df["delta"].cumsum()
        # OFI: 每根K线的买卖失衡比，范围 [-1, +1]
        df["ofi"] = (df["delta"] / df["volume"].replace(0, np.nan)).fillna(0).clip(-1, 1)

    return df


# ─────────────────────────────────────────────
# C. 支撑 / 阻力识别
# ─────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, lookback: int = 250, pivot_n: int = 5,
                             cluster_pct: float = 0.015, ob_levels: list = None):
    recent = df.tail(lookback).copy()
    current_price = df["close"].iloc[-1]

    # ── 自适应 pivot_n：根据 ATR 百分位动态调整分形窗口 ─────────────
    if "atr" in df.columns:
        atr_series = df["atr"].dropna()
        if len(atr_series) >= 200:
            atr_hist     = atr_series.iloc[-200:].values
            current_atr  = float(atr_series.iloc[-1])
            atr_pct_rank = (
                np.searchsorted(np.sort(atr_hist), current_atr)
                / len(atr_hist) * 100
            )
            if atr_pct_rank >= 75:      # 高波动：缩小窗口，及时响应转折点
                pivot_n = max(2, pivot_n - 1)
            elif atr_pct_rank <= 25:    # 低波动：扩大窗口，过滤震荡噪音
                pivot_n = min(8, pivot_n + 2)

    # 1. 识别高低点 (Pivots) - 权重 1.0
    raw_levels = []
    for i in range(pivot_n, len(recent) - pivot_n):
        w_h = recent["high"].iloc[i - pivot_n: i + pivot_n + 1]
        w_l = recent["low"].iloc[i - pivot_n: i + pivot_n + 1]
        if recent["high"].iloc[i] == w_h.max():
            raw_levels.append((recent["high"].iloc[i], 1.0, "pivot"))
        if recent["low"].iloc[i] == w_l.min():
            raw_levels.append((recent["low"].iloc[i], 1.0, "pivot"))

    # 2. 识别成交量密集区 (Volume Profile) - 权重 2.0
    price_range = recent["close"].max() - recent["close"].min()
    if price_range > 0:
        num_bins = 50
        bins = np.linspace(recent["close"].min(), recent["close"].max(), num_bins + 1)
        vol_profile = np.zeros(num_bins)
        for _, row in recent.iterrows():
            idx = min(int((row["close"] - recent["close"].min()) / price_range * num_bins), num_bins - 1)
            vol_profile[idx] += row["volume"]

        # 取成交量最大的前 15 个 bin 的中心价
        bin_centers = (bins[:-1] + bins[1:]) / 2
        top_vol_idx = np.argsort(vol_profile)[-15:]
        for idx in top_vol_idx:
            raw_levels.append((bin_centers[idx], 2.0, "volume"))

    # 3. 订单簿挂单墙（来自 server.py 的实时深度快照）
    if ob_levels:
        for ob_price, ob_weight, ob_side in ob_levels:
            # 只收录当前价格 ±20% 范围内的价格层
            if 0.8 * current_price <= ob_price <= 1.2 * current_price:
                raw_levels.append((ob_price, ob_weight, ob_side))

    # 4. 聚类合并（支持 3-tuple，合并来源标签）
    def cluster_levels(levels_with_weight, thr=0.015):
        if not levels_with_weight:
            return []
        levels_with_weight.sort(key=lambda x: x[0])

        def _finalize(cluster):
            prices  = [x[0] for x in cluster]
            weights = [x[1] for x in cluster]
            # 合并来源标签：去重、保序，防止累积重复
            srcs = list(dict.fromkeys(
                s for x in cluster
                for s in (x[2] if len(x) > 2 else "pivot").split("+")
            ))
            return (np.average(prices, weights=weights), sum(weights), "+".join(srcs))

        clusters, cur = [], [levels_with_weight[0]]
        for item in levels_with_weight[1:]:
            if abs(item[0] - cur[-1][0]) / cur[-1][0] < thr:
                cur.append(item)
            else:
                clusters.append(_finalize(cur))
                cur = [item]
        clusters.append(_finalize(cur))
        return clusters

    consolidated = cluster_levels(raw_levels, thr=cluster_pct)

    # 5. 区分支撑/阻力
    supports    = [(p, w, src) for p, w, src in consolidated if p < current_price]
    resistances = [(p, w, src) for p, w, src in consolidated if p > current_price]

    # 6. 排序并取前5 (按距离排序)
    supports    = sorted(supports,    key=lambda x: x[0], reverse=True)[:5]
    resistances = sorted(resistances, key=lambda x: x[0])[:5]

    return supports, resistances


# ─────────────────────────────────────────────
# D. Fibonacci 展开
# ─────────────────────────────────────────────

def calculate_fibonacci(df: pd.DataFrame, lookback: int = 90):
    recent = df.tail(lookback)
    high   = recent["high"].max()
    low    = recent["low"].min()
    diff   = high - low
    levels = {f"Fib {r:.1%}": high - diff * r
              for r in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}
    return levels, high, low


# ─────────────────────────────────────────────
# E. 交互式图表（单周期）
# ─────────────────────────────────────────────

def create_chart(df: pd.DataFrame, supports, resistances,
                 fib_levels, fib_high, fib_low,
                 symbol: str = "BTC/USDT", tf_label: str = "", chart_show: int = 300):
    plot_df = df.tail(chart_show).copy()

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.40, 0.12, 0.14, 0.12, 0.12, 0.10],
        subplot_titles=(
            f"{symbol} {tf_label} K 线（均线 · 布林带 · 支撑/阻力 · Fibonacci）",
            "成交量 · OBV", "MACD（12,26,9）", "RSI（14）", "KDJ（9,3,3）", ""
        )
    )

    # ── K 线 ──
    fig.add_trace(go.Candlestick(
        x=plot_df["date"], open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"], name="K线",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False
    ), row=1, col=1)

    for col_name, color, name, extra in [
        ("ma20",  "#f9a825", "MA20",  {}),
        ("ma50",  "#ab47bc", "MA50",  {}),
        ("ma200", "#ef5350", "MA200", {}),
        ("ema20", "#29b6f6", "EMA20", {"dash": "dot"}),
        ("ema50", "#66bb6a", "EMA50", {"dash": "dot"}),
    ]:
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=plot_df[col_name],
            name=name, line=dict(color=color, width=1, **extra), opacity=0.85
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["bb_upper"],
        name="BB上轨", line=dict(color="rgba(100,181,246,0.6)", width=1, dash="dash")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["bb_lower"],
        name="BB下轨", line=dict(color="rgba(100,181,246,0.6)", width=1, dash="dash"),
        fill="tonexty", fillcolor="rgba(100,181,246,0.05)"
    ), row=1, col=1)

    x0, x1 = plot_df["date"].iloc[0], plot_df["date"].iloc[-1]
    for price, _ in supports:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                      line=dict(color="rgba(38,166,154,0.7)", width=1.5, dash="dot"), row=1, col=1)
        fig.add_annotation(x=x1, y=price, text=f"  支撑 ${price:,.0f}",
                           showarrow=False, font=dict(color="#26a69a", size=10),
                           xanchor="left", row=1, col=1)
    for price, _ in resistances:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                      line=dict(color="rgba(239,83,80,0.7)", width=1.5, dash="dot"), row=1, col=1)
        fig.add_annotation(x=x1, y=price, text=f"  阻力 ${price:,.0f}",
                           showarrow=False, font=dict(color="#ef5350", size=10),
                           xanchor="left", row=1, col=1)

    fib_colors = ["#b0bec5", "#80cbc4", "#ffcc02", "#ff9800", "#ff5722", "#e91e63", "#9c27b0"]
    for (label, fp), color in zip(fib_levels.items(), fib_colors):
        fig.add_shape(type="line", x0=x0, x1=x1, y0=fp, y1=fp,
                      line=dict(color=color, width=0.8, dash="dash"), row=1, col=1)
        fig.add_annotation(x=plot_df["date"].iloc[int(len(plot_df) * 0.02)], y=fp,
                           text=f"{label} ${fp:,.0f}", showarrow=False,
                           font=dict(color=color, size=9), row=1, col=1)

    # ── 成交量 + OBV ──
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(plot_df["close"], plot_df["open"])]
    fig.add_trace(go.Bar(
        x=plot_df["date"], y=plot_df["volume"],
        name="成交量", marker_color=vol_colors, showlegend=False, opacity=0.7
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["obv"],
        name="OBV", line=dict(color="#ffb300", width=1.5)
    ), row=2, col=1)

    # ── MACD ──
    hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                   for v in plot_df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(
        x=plot_df["date"], y=plot_df["macd_hist"],
        name="MACD柱", marker_color=hist_colors, showlegend=False
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["macd"],
                             name="MACD", line=dict(color="#29b6f6", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["macd_signal"],
                             name="信号线", line=dict(color="#ff7043", width=1.5)), row=3, col=1)

    # ── RSI ──
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["rsi"],
                             name="RSI(14)", line=dict(color="#ce93d8", width=1.5)), row=4, col=1)
    for lv, clr in [(70, "rgba(239,83,80,0.4)"), (30, "rgba(38,166,154,0.4)")]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=lv, y1=lv,
                      line=dict(color=clr, width=1, dash="dash"), row=4, col=1)

    # ── KDJ ──
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["kdj_k"],
                             name="K", line=dict(color="#66bb6a", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["kdj_d"],
                             name="D", line=dict(color="#ef5350", width=1.2)), row=5, col=1)
    fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["kdj_j"],
                             name="J", line=dict(color="#ffb300", width=1.2, dash="dot")), row=5, col=1)
    for lv in [80, 20]:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=lv, y1=lv,
                      line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dash"), row=5, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"{symbol} {tf_label} 综合技术分析  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            font=dict(size=15, color="#ffffff")
        ),
        height=1300,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=130, t=80, b=40),
        paper_bgcolor="#131722",
        plot_bgcolor="#131722",
    )
    fig.update_xaxes(gridcolor="#1e2130", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2130", zeroline=False)
    return fig


# ─────────────────────────────────────────────
# F. 综合分析报告（单周期）
# ─────────────────────────────────────────────

def generate_report(df: pd.DataFrame, supports, resistances,
                    fib_levels, symbol: str = "BTC/USDT", tf_label: str = "", verbose: bool = True):
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    price  = latest["close"]
    score  = 0
    lines  = []
    rd     = {}

    def emit(text=""):
        if verbose:
            print(text)
        lines.append(text)

    sep = "─" * 60
    ts  = latest["date"].strftime("%Y-%m-%d %H:%M") if tf_label in ("15分钟","30分钟","1小时","2小时","4小时","8小时") \
          else latest["date"].strftime("%Y-%m-%d")
    emit(f"\n{'═'*60}")
    emit(f"  {symbol} [{tf_label}] 技术分析报告  |  {ts}")
    emit(f"  当前收盘价: ${price:,.2f}")
    emit(f"{'═'*60}\n")
    rd["date"]  = ts
    rd["price"] = price
    rd["tf"]    = tf_label

    # ── 市场机制识别 ─────────────────────────────────────
    adx_val  = latest["adx"]     if "adx"     in df.columns else np.nan
    di_plus  = latest["adx_pos"] if "adx_pos" in df.columns else np.nan
    di_minus = latest["adx_neg"] if "adx_neg" in df.columns else np.nan
    high_vol     = df.attrs.get("high_vol",     False)
    bb_dev       = df.attrs.get("bb_dev",       2.0)
    rsi_win      = df.attrs.get("rsi_win",      14)
    atr_pct_rank = df.attrs.get("atr_pct_rank", 0.5)

    if not pd.isna(adx_val):
        if adx_val >= 25:   regime = "trending"
        elif adx_val <= 20: regime = "ranging"
        else:               regime = "transitional"
    else:
        regime = "unknown"

    # ── 均线 ──
    emit(f"【均线系统】{sep}")
    ma_items = [
        ("MA20",  latest["ma20"],  "短期趋势"),
        ("MA50",  latest["ma50"],  "中期趋势"),
        ("MA200", latest["ma200"], "长期趋势"),
        ("EMA20", latest["ema20"], "短期动态"),
        ("EMA50", latest["ema50"], "中期动态"),
    ]
    rd["ma"] = []
    for name, val, desc in ma_items:
        if pd.isna(val):
            continue
        diff_pct = (price - val) / val * 100
        is_above = price > val
        # 震荡市禁用 MA 突破评分（防假突破）
        ma_active = (regime != "ranging")
        if ma_active:
            score += 1 if is_above else -1
        regime_note = "" if ma_active else " [震荡市-暂停]"
        emit(f"  {name:<6}({desc}): ${val:>10,.2f}  {'上方 ↑' if is_above else '下方 ↓'} {diff_pct:+.2f}%  [{'+'if is_above else'-'}]{regime_note}")
        rd["ma"].append({"name": name, "desc": desc, "val": val,
                         "diff_pct": diff_pct, "above": is_above, "signal": "+" if is_above else "-",
                         "active": ma_active})

    # ── RSI ──
    rsi_val = latest["rsi"]
    emit(f"\n【RSI({rsi_win})】{sep}")
    rsi_score = 0
    if regime == "trending":
        # 趋势市：超买/超卖逆势信号禁用，只保留强弱方向
        if rsi_val >= 70:   rsi_signal = "超买(趋势市-暂不做空)"; rsi_score = 0
        elif rsi_val <= 30: rsi_signal = "超卖(趋势市-暂不做多)"; rsi_score = 0
        elif rsi_val >= 55: rsi_signal = "偏强区域 (55-70)";       rsi_score = +1
        elif rsi_val <= 45: rsi_signal = "偏弱区域 (30-45)";       rsi_score = -1
        else:               rsi_signal = "中性区域 (45-55)";        rsi_score = 0
    else:
        # 震荡市/过渡区：完整逆势评分
        if rsi_val >= 70:   rsi_signal = "超买区域 (>=70) 可能回调"; rsi_score = -2
        elif rsi_val <= 30: rsi_signal = "超卖区域 (<=30) 可能反弹"; rsi_score = +2
        elif rsi_val >= 55: rsi_signal = "偏强区域 (55-70)";          rsi_score = +1
        elif rsi_val <= 45: rsi_signal = "偏弱区域 (30-45)";          rsi_score = -1
        else:               rsi_signal = "中性区域 (45-55)";           rsi_score = 0
    score += rsi_score
    emit(f"  RSI = {rsi_val:.2f}  ->  {rsi_signal}")
    rd["rsi"] = {"val": rsi_val, "signal": rsi_signal, "score": rsi_score}

    # ── 布林带 ──
    bb_upper, bb_lower, bb_mid = latest["bb_upper"], latest["bb_lower"], latest["bb_mid"]
    bb_width_pct = (bb_upper - bb_lower) / bb_mid * 100
    emit(f"\n【布林带(20,2sigma)】{sep}")
    bb_score = 0
    if price >= bb_upper:   bb_signal = "触及/突破上轨 -> 强势但注意回调"; bb_score = -1
    elif price <= bb_lower: bb_signal = "触及/跌破下轨 -> 弱势但注意反弹"; bb_score = +1
    elif price > bb_mid:    bb_signal = "中轨上方 -> 偏多";                 bb_score = +1
    else:                   bb_signal = "中轨下方 -> 偏空";                 bb_score = -1
    score += bb_score
    emit(f"  上轨: ${bb_upper:,.2f}  中轨: ${bb_mid:,.2f}  下轨: ${bb_lower:,.2f}")
    emit(f"  带宽: {bb_width_pct:.2f}%  |  当前位置: {bb_signal}")
    rd["bb"] = {"upper": bb_upper, "mid": bb_mid, "lower": bb_lower,
                "width_pct": bb_width_pct, "signal": bb_signal, "score": bb_score}

    # ── MACD ──
    macd_val, macd_sig, macd_hist = latest["macd"], latest["macd_signal"], latest["macd_hist"]
    emit(f"\n【MACD(12,26,9)】{sep}")
    macd_score = 0
    if macd_val > macd_sig and prev["macd"] <= prev["macd_signal"]:
        macd_signal = "金叉 (刚发生)"; macd_score = +2
    elif macd_val < macd_sig and prev["macd"] >= prev["macd_signal"]:
        macd_signal = "死叉 (刚发生)"; macd_score = -2
    elif macd_val > macd_sig:
        macd_signal = "MACD 在信号线上方 -> 多头"; macd_score = +1
    else:
        macd_signal = "MACD 在信号线下方 -> 空头"; macd_score = -1
    score += macd_score
    hist_trend = "柱状图扩大 (动能增强)" if abs(macd_hist) > abs(prev["macd_hist"]) else "柱状图收缩 (动能减弱)"
    emit(f"  MACD: {macd_val:.2f}  信号线: {macd_sig:.2f}  柱: {macd_hist:.2f}")
    emit(f"  信号: {macd_signal}  |  {hist_trend}")
    rd["macd"] = {"val": macd_val, "signal_line": macd_sig, "hist": macd_hist,
                  "signal": macd_signal, "hist_trend": hist_trend, "score": macd_score}

    # ── KDJ ──
    k_val, d_val, j_val = latest["kdj_k"], latest["kdj_d"], latest["kdj_j"]
    emit(f"\n【KDJ(9,3,3)】{sep}")
    kdj_score = 0
    if regime == "trending":
        # 趋势市：超买/超卖逆势信号禁用，金叉/死叉顺势保留
        if k_val > d_val and prev["kdj_k"] <= prev["kdj_d"]:
            kdj_signal = "K 上穿 D -> 金叉(趋势加速)"; kdj_score = +2
        elif k_val < d_val and prev["kdj_k"] >= prev["kdj_d"]:
            kdj_signal = "K 下穿 D -> 死叉(趋势加速)"; kdj_score = -2
        elif k_val >= 80:
            kdj_signal = "超买(趋势市-持仓不做空)"; kdj_score = 0
        elif k_val <= 20:
            kdj_signal = "超卖(趋势市-持仓不做多)"; kdj_score = 0
        elif k_val > d_val:
            kdj_signal = "K > D 顺势偏多"; kdj_score = +1
        else:
            kdj_signal = "K < D 顺势偏空"; kdj_score = -1
    else:
        # 震荡市/过渡区：完整高抛低吸逻辑
        if k_val > d_val and prev["kdj_k"] <= prev["kdj_d"]:
            kdj_signal = "K 上穿 D -> 金叉买入信号"; kdj_score = +2
        elif k_val < d_val and prev["kdj_k"] >= prev["kdj_d"]:
            kdj_signal = "K 下穿 D -> 死叉卖出信号"; kdj_score = -2
        elif k_val >= 80:
            kdj_signal = "超买区域 (K>=80)"; kdj_score = -1
        elif k_val <= 20:
            kdj_signal = "超卖区域 (K<=20)"; kdj_score = +1
        elif k_val > d_val:
            kdj_signal = "K > D 偏多"; kdj_score = +1
        else:
            kdj_signal = "K < D 偏空"; kdj_score = -1
    score += kdj_score
    emit(f"  K: {k_val:.2f}  D: {d_val:.2f}  J: {j_val:.2f}")
    emit(f"  信号: {kdj_signal}")
    rd["kdj"] = {"k": k_val, "d": d_val, "j": j_val, "signal": kdj_signal, "score": kdj_score}

    # ── OBV ──
    obv_now, obv_prev = latest["obv"], df["obv"].iloc[-20]
    emit(f"\n【OBV 能量潮】{sep}")
    obv_up    = obv_now > obv_prev
    obv_trend = "上升 -> 资金流入，价格上涨得到支撑" if obv_up else "下降 -> 资金流出，价格下跌压力"
    obv_score = 1 if obv_up else -1
    score    += obv_score
    emit(f"  当前OBV: {obv_now:,.0f}  20根前: {obv_prev:,.0f}")
    emit(f"  趋势: {obv_trend}")
    rd["obv"] = {"now": obv_now, "prev": obv_prev, "trend": obv_trend,
                 "up": obv_up, "score": obv_score}

    # ── CVD（主动成交量差值背离）────────────────────────────────────
    cvd_score  = 0
    cvd_signal = "暂无数据"
    last_ofi   = 0.0
    last_delta = 0.0
    last_cvd   = 0.0
    emit(f"\n【CVD 成交量差值 / Order Flow】{sep}")
    if "cvd" in df.columns and "ofi" in df.columns and len(df) >= 20:
        recent     = df.tail(20)
        p_start    = float(recent["close"].iloc[0])
        p_end      = float(recent["close"].iloc[-1])
        cvd_start  = float(recent["cvd"].iloc[0])
        cvd_end    = float(recent["cvd"].iloc[-1])
        avg_vol    = float(recent["volume"].mean()) or 1.0
        norm_price = (p_end - p_start) / p_start          # 价格变化率
        norm_cvd   = (cvd_end - cvd_start) / avg_vol      # CVD 变化 / 平均成交量
        last_ofi   = float(df["ofi"].iloc[-1])
        last_delta = float(df["delta"].iloc[-1])
        last_cvd   = float(df["cvd"].iloc[-1])

        if norm_price > 0.005 and norm_cvd < -0.15:
            cvd_score  = -2
            cvd_signal = f"顶背离 ⚠️ 价格↑ 但主动卖盘主导(CVD Δ{norm_cvd:+.2f})，警惕派发出货"
        elif norm_price < -0.005 and norm_cvd > 0.15:
            cvd_score  = +2
            cvd_signal = f"底背离 ✅ 价格↓ 但主动买盘吸收(CVD Δ{norm_cvd:+.2f})，关注见底信号"
        elif norm_price > 0.003 and norm_cvd > 0.15:
            cvd_score  = +1
            cvd_signal = f"多头共振 ✅ 价格↑ + 主动买盘推动(Δ{norm_cvd:+.2f})，突破可信"
        elif norm_price < -0.003 and norm_cvd < -0.15:
            cvd_score  = -1
            cvd_signal = f"空头共振 ⚠️ 价格↓ + 主动卖盘主导(Δ{norm_cvd:+.2f})，下跌确认"
        elif last_ofi > 0.2:
            cvd_signal = f"当前K线主动买盘偏多 OFI {last_ofi:+.2f}"
        elif last_ofi < -0.2:
            cvd_signal = f"当前K线主动卖盘偏多 OFI {last_ofi:+.2f}"
        else:
            cvd_signal = f"主动买卖均衡 OFI {last_ofi:+.2f}"

        score += cvd_score
        emit(f"  OFI(最后K): {last_ofi:+.3f}  Delta: {last_delta:+,.0f}")
        emit(f"  近20K CVD变化率: {norm_cvd:+.3f}  价格变化率: {norm_price:+.3f}")
    else:
        emit("  taker_buy 数据不可用，跳过 CVD 分析")
    emit(f"  信号: {cvd_signal}  得分: {cvd_score:+d}")
    rd["cvd"] = {
        "score":      cvd_score,
        "signal":     cvd_signal,
        "last_ofi":   round(last_ofi,   4),
        "last_delta": round(last_delta, 2),
        "last_cvd":   round(last_cvd,   2),
    }

    # ── 支撑/阻力 ──
    emit(f"\n【支撑 / 阻力位】{sep}")
    emit("  阻力位 (从近到远):")
    res_list = []
    for level in sorted(resistances, key=lambda x: x[0]):
        pr, sr_score = level[0], level[1]
        src = level[2] if len(level) > 2 else "pivot"
        dist = (pr - price) / price * 100
        confidence = min(5.0, sr_score)
        emit(f"    ${pr:>10,.2f}  距当前 +{dist:.2f}%  强度:{confidence:.1f} ({sr_score:.1f}) [{src}]")
        res_list.append({"price": pr, "dist_pct": dist, "score": sr_score,
                         "stars_val": confidence, "source": src})

    emit("  支撑位 (从近到远):")
    sup_list = []
    for level in sorted(supports, key=lambda x: x[0], reverse=True):
        ps, sr_score = level[0], level[1]
        src = level[2] if len(level) > 2 else "pivot"
        dist = (price - ps) / price * 100
        confidence = min(5.0, sr_score)
        emit(f"    ${ps:>10,.2f}  距当前 -{dist:.2f}%  强度:{confidence:.1f} ({sr_score:.1f}) [{src}]")
        sup_list.append({"price": ps, "dist_pct": dist, "score": sr_score,
                         "stars_val": confidence, "source": src})

    rd["resistances"] = res_list
    rd["supports"]    = sup_list

    # ── Fibonacci ──
    emit(f"\n【Fibonacci 回调位】{sep}")
    fib_out = []
    for label, fp in sorted(fib_levels.items(), key=lambda x: x[1], reverse=True):
        dist = (fp - price) / price * 100
        marker = " <- 当前价附近" if abs(dist) < 2 else ""
        emit(f"  {label:<12}: ${fp:>10,.2f}  ({dist:+.2f}% {'↑' if fp>price else '↓'}){marker}")
        fib_out.append({"label": label, "price": fp, "dist_pct": dist, "near": abs(dist) < 2})
    rd["fib"] = fib_out

    # ── 市场机制写入 rd ──────────────────────────────────
    rd["regime"] = {
        "adx":          round(float(adx_val), 2)  if not pd.isna(adx_val)  else None,
        "di_plus":      round(float(di_plus),  2)  if not pd.isna(di_plus)  else None,
        "di_minus":     round(float(di_minus), 2)  if not pd.isna(di_minus) else None,
        "type":         regime,
        "high_vol":     high_vol,
        "bb_dev":       bb_dev,
        "rsi_win":      rsi_win,
        "atr_pct_rank": round(atr_pct_rank, 2),
    }

    # ── 综合信号 ──
    max_score = 18
    emit(f"\n{'═'*60}")
    if score >= 5:   
        overall = f"偏多头  (得分: {score:+d}/{max_score})"
        oc = "#26a69a"
        action = "做多 / 买入"
    elif score <= -5: 
        overall = f"偏空头  (得分: {score:+d}/{max_score})"
        oc = "#ef5350"
        action = "做空 / 卖出"
    else:            
        overall = f"中性震荡  (得分: {score:+d}/{max_score})"
        oc = "#f9a825"
        action = "观望 / 高抛低吸"

    emit(f"  综合信号: {overall}")
    emit(f"  建议操作: {action}")
    emit(f"  多空强度: {'█'*max(0,score)}{'░'*max(0,-score)}")
    emit(f"{'═'*60}\n")
    rd.update({"score": score, "max_score": max_score, "overall": overall,
               "overall_color": oc, "action": action,
               "bull_bar": "█"*max(0,score), "bear_bar": "░"*max(0,-score)})

    # ── ATR 风险管理（不参与评分，仅展示）────────────────
    _atr_val = df["atr"].iloc[-1]
    if pd.isna(_atr_val):
        _atr_val = 0.0
    atr_now      = float(_atr_val)
    atr_sl_long  = float(df["atr_sl_long"].iloc[-1])
    atr_sl_short = float(df["atr_sl_short"].iloc[-1])
    atr_pct      = atr_now / price * 100 if price else 0
    rd["atr"] = {
        "val":               round(atr_now, 4),
        "pct":               round(atr_pct, 2),
        "sl_long":           round(atr_sl_long, 4),
        "sl_short":          round(atr_sl_short, 4),
        "sl_long_dist_pct":  round((price - atr_sl_long)  / price * 100, 2) if price else 0,
        "sl_short_dist_pct": round((atr_sl_short - price) / price * 100, 2) if price else 0,
    }

    # ── VWAP 快照（供分析面板展示）────────────────────────
    _vwap_val = df["vwap"].iloc[-1]
    if not pd.isna(_vwap_val):
        vwap_val = float(_vwap_val)
        vwap_diff_pct = (price - vwap_val) / vwap_val * 100
        rd["vwap"] = {
            "val":      round(vwap_val, 4),
            "diff_pct": round(vwap_diff_pct, 2),
            "above":    price >= vwap_val,
        }
    else:
        rd["vwap"] = None

    return score, "\n".join(lines), rd


# ─────────────────────────────────────────────
# G. 导出 TXT 报告
# ─────────────────────────────────────────────

def export_txt_report(all_texts: list, filepath: str, symbol: str = "BTCUSDT"):
    header = ("=" * 60 + "\n"
              f"  {symbol} 多周期技术分析报告\n"
              f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
              "=" * 60 + "\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        for tf_label, text in all_texts:
            f.write(f"\n{'#'*60}\n#  周期: {tf_label}\n{'#'*60}\n")
            f.write(text + "\n")
    print(f"[导出] 文本报告已保存: {filepath}")


# ─────────────────────────────────────────────
# H. 多周期汇总表（各周期信号一览）
# ─────────────────────────────────────────────

def build_summary_table(all_rd: list) -> str:
    """生成 HTML 多周期信号汇总表"""
    def sc(s):
        if s > 0:  return "#26a69a"
        if s < 0:  return "#ef5350"
        return "#f9a825"

    header = "<tr><th>周期</th><th>价格</th><th>均线</th><th>RSI</th><th>布林带</th><th>MACD</th><th>KDJ</th><th>OBV</th><th>综合</th><th>建议操作</th></tr>"
    rows = ""
    for rd in all_rd:
        ma_s  = sum(1 if m["above"] else -1 for m in rd["ma"])
        
        # Action Color
        if rd["score"] >= 5: ac = "#26a69a"
        elif rd["score"] <= -5: ac = "#ef5350"
        else: ac = "#f9a825"

        rows += (
            f'<tr>'
            f'<td style="font-weight:700;color:#90caf9">{rd["tf"]}</td>'
            f'<td>${rd["price"]:,.2f}</td>'
            f'<td style="color:{sc(ma_s)}">{ma_s:+d}</td>'
            f'<td style="color:{sc(rd["rsi"]["score"])}">{rd["rsi"]["val"]:.1f}</td>'
            f'<td style="color:{sc(rd["bb"]["score"])}">{rd["bb"]["signal"].split("->")[0].strip()}</td>'
            f'<td style="color:{sc(rd["macd"]["score"])}">{rd["macd"]["signal"].split("->")[0].strip()[:12]}</td>'
            f'<td style="color:{sc(rd["kdj"]["score"])}">{rd["kdj"]["signal"].split("->")[0].strip()[:12]}</td>'
            f'<td style="color:{sc(rd["obv"]["score"])}">'
            f'  {"↑ 流入" if rd["obv"]["up"] else "↓ 流出"}</td>'
            f'<td style="color:{rd["overall_color"]};font-weight:700">'
            f'  {rd["overall"].split("(")[0].strip()}</td>'
            f'<td style="color:{ac};font-weight:700">{rd.get("action", "")}</td>'
            f'</tr>'
        )
    return f'<table><thead>{header}</thead><tbody>{rows}</tbody></table>'


# ─────────────────────────────────────────────
# I. 导出合并多标签 HTML 报告 (多币种支持)
# ─────────────────────────────────────────────

def generate_symbol_html_content(symbol, tf_results, is_active=False):
    """生成单个币种的 HTML 内容块"""
    all_rd = [rd for _, _, _, rd in tf_results]
    summary_table = build_summary_table(all_rd)
    
    # ── 各周期 Tab 内容 ──
    tab_buttons = ""
    tab_panels  = ""
    
    # 唯一ID前缀，避免不同币种冲突
    prefix = symbol.replace("/", "").replace("-", "")

    # ── 计算全局建议 ──
    # 权重: 1w=2.0, 1d=1.5, 4h=1.2, others=1.0
    w_map = {"1w": 2.0, "1d": 1.5, "4h": 1.2}
    total_w_score = 0
    total_weight = 0
    for tf_id, _, _, rd in tf_results:
        w = w_map.get(tf_id, 1.0)
        total_w_score += rd["score"] * w
        total_weight += w
    
    global_avg_score = total_w_score / total_weight if total_weight > 0 else 0
    
    if global_avg_score >= 4:
        g_trend = "强力看涨 (Bullish)"
        g_action = "建议分批建仓 / 持有"
        g_color = "#26a69a"
    elif global_avg_score >= 1:
        g_trend = "偏多震荡 (Weak Bullish)"
        g_action = "逢低做多"
        g_color = "#66bb6a"
    elif global_avg_score <= -4:
        g_trend = "强力看跌 (Bearish)"
        g_action = "清仓 / 做空"
        g_color = "#ef5350"
    elif global_avg_score <= -1:
        g_trend = "偏空震荡 (Weak Bearish)"
        g_action = "逢高做空 / 减仓"
        g_color = "#ff7043"
    else:
        g_trend = "横盘整理 (Neutral)"
        g_action = "观望 / 区间操作"
        g_color = "#f9a825"

    global_summary_html = f"""
    <div class="card" style="margin-bottom:15px; border-left: 5px solid {g_color}">
        <h2 style="font-size:14px; color:#d1d4dc; margin-bottom:5px">全局趋势分析汇总 (加权评分: {global_avg_score:+.1f})</h2>
        <div style="display:flex; gap:20px; align-items:center">
            <div style="font-size:18px; font-weight:700; color:{g_color}">{g_trend}</div>
            <div style="font-size:14px; color:#90a4ae">操作建议: <span style="color:#fff; font-weight:600">{g_action}</span></div>
        </div>
    </div>
    """

    for i, (tf_id, tf_label, fig, rd) in enumerate(tf_results):
        # 默认选中日线，如果找不到则选第一个
        active_btn   = "active" if tf_id == '1d' else ""
        active_panel = "active" if tf_id == '1d' else ""
        if "active" not in tab_buttons and i == len(tf_results)-1 and not active_btn:
             active_btn = "active"
             active_panel = "active"

        # 仅第一个币种的第一个图表包含 Plotly JS (CDN)，减少重复
        include_plotlyjs = False # 我们将在 HTML 头部统一引入 Plotly JS，这里不再嵌入
        chart_html = fig.to_html(full_html=False, include_plotlyjs=False)

        def sc(s):
            return "#26a69a" if s > 0 else ("#ef5350" if s < 0 else "#f9a825")

        rsi_c  = sc(rd["rsi"]["score"])
        bb_c   = sc(rd["bb"]["score"])
        macd_c = sc(rd["macd"]["score"])
        kdj_c  = sc(rd["kdj"]["score"])
        obv_c  = sc(rd["obv"]["score"])
        bull_w = max(0, rd["score"]) / rd["max_score"] * 100
        bear_w = max(0, -rd["score"]) / rd["max_score"] * 100

        # 均线行
        ma_rows = ""
        for m in rd["ma"]:
            c = "#26a69a" if m["above"] else "#ef5350"
            a = "▲" if m["above"] else "▼"
            ma_rows += (f'<tr><td>{m["name"]}</td><td>{m["desc"]}</td>'
                        f'<td>${m["val"]:,.2f}</td>'
                        f'<td style="color:{c}">{a} {m["diff_pct"]:+.2f}%</td>'
                        f'<td style="color:{c}">[{m["signal"]}]</td></tr>')

        # 支撑/阻力行
        res_rows = "".join(
            f'<tr><td style="color:#ef5350">阻力</td>'
            f'<td>${r["price"]:,.2f}</td>'
            f'<td style="color:#ef5350">+{r["dist_pct"]:.2f}%</td>'
            f'<td style="letter-spacing:1px"><span class="stars" style="--rating:{r["stars_val"]}"></span> <span style="font-size:11px;color:#78909c">({r["score"]:.1f})</span></td></tr>'
            for r in rd["resistances"]
        )
        sup_rows = "".join(
            f'<tr><td style="color:#26a69a">支撑</td>'
            f'<td>${s["price"]:,.2f}</td>'
            f'<td style="color:#26a69a">-{s["dist_pct"]:.2f}%</td>'
            f'<td style="letter-spacing:1px"><span class="stars" style="--rating:{s["stars_val"]}"></span> <span style="font-size:11px;color:#78909c">({s["score"]:.1f})</span></td></tr>'
            for s in rd["supports"]
        )

        # Fibonacci 行
        fib_rows = ""
        for fi in rd["fib"]:
            c = "#f9a825" if fi["near"] else "#b0bec5"
            nm = " &larr; 当前价附近" if fi["near"] else ""
            fib_rows += (f'<tr><td style="color:{c}">{fi["label"]}</td>'
                         f'<td style="color:{c}">${fi["price"]:,.2f}</td>'
                         f'<td style="color:{c}">{fi["dist_pct"]:+.2f}%{nm}</td></tr>')

        ma_score_total = sum(1 if m["above"] else -1 for m in rd["ma"])

        # Tab Button
        tab_buttons += f'<button class="tab-btn {active_btn}" onclick="switchTab(event, \'{prefix}\', \'{tf_id}\')">{tf_label}</button>\n'

        # Panel Content
        tab_panels += f'''
<div id="panel-{prefix}-{tf_id}" class="tab-panel {active_panel}">
  <div class="grid">
    <div class="card">
      <h2>均线系统</h2>
      <table><tr><th>均线</th><th>含义</th><th>当前值</th><th>偏离</th><th>信号</th></tr>
      {ma_rows}</table>
    </div>
    <div class="card">
      <h2>振荡指标</h2>
      <div class="signal-row"><span class="signal-label">RSI(14)</span>
        <span class="signal-value" style="color:{rsi_c}">{rd["rsi"]["val"]:.2f}</span>
        <span style="color:{rsi_c}">&nbsp;{rd["rsi"]["signal"]}</span></div>
      <div class="signal-row"><span class="signal-label">KDJ K/D/J</span>
        <span class="signal-value" style="color:{kdj_c}">{rd["kdj"]["k"]:.2f}/{rd["kdj"]["d"]:.2f}/{rd["kdj"]["j"]:.2f}</span></div>
      <div class="signal-row" style="padding-left:110px">
        <span style="color:{kdj_c}">{rd["kdj"]["signal"]}</span></div>
    </div>
    <div class="card">
      <h2>布林带 (20, 2&sigma;)</h2>
      <div class="signal-row"><span class="signal-label">上轨</span>
        <span class="signal-value" style="color:#ef5350">${rd["bb"]["upper"]:,.2f}</span></div>
      <div class="signal-row"><span class="signal-label">中轨</span>
        <span class="signal-value" style="color:#f9a825">${rd["bb"]["mid"]:,.2f}</span></div>
      <div class="signal-row"><span class="signal-label">下轨</span>
        <span class="signal-value" style="color:#26a69a">${rd["bb"]["lower"]:,.2f}</span></div>
      <div class="signal-row"><span class="signal-label">带宽</span>
        <span class="signal-value">{rd["bb"]["width_pct"]:.2f}%</span></div>
      <div class="signal-row"><span class="signal-label">当前位置</span>
        <span style="color:{bb_c}">{rd["bb"]["signal"]}</span></div>
    </div>
    <div class="card">
      <h2>MACD (12, 26, 9)</h2>
      <div class="signal-row"><span class="signal-label">MACD 线</span>
        <span class="signal-value">{rd["macd"]["val"]:.2f}</span></div>
      <div class="signal-row"><span class="signal-label">信号线</span>
        <span class="signal-value">{rd["macd"]["signal_line"]:.2f}</span></div>
      <div class="signal-row"><span class="signal-label">柱状图</span>
        <span class="signal-value" style="color:{macd_c}">{rd["macd"]["hist"]:.2f}</span></div>
      <div class="signal-row"><span class="signal-label">信号</span>
        <span style="color:{macd_c}">{rd["macd"]["signal"]}</span></div>
      <div class="signal-row"><span class="signal-label">动能</span>
        <span style="color:#78909c">{rd["macd"]["hist_trend"]}</span></div>
    </div>
    <div class="card">
      <h2>OBV 能量潮</h2>
      <div class="signal-row"><span class="signal-label">当前 OBV</span>
        <span class="signal-value">{rd["obv"]["now"]:,.0f}</span></div>
      <div class="signal-row"><span class="signal-label">20根前 OBV</span>
        <span class="signal-value">{rd["obv"]["prev"]:,.0f}</span></div>
      <div class="signal-row"><span class="signal-label">趋势</span>
        <span style="color:{obv_c}">{rd["obv"]["trend"]}</span></div>
    </div>
    <div class="card">
      <h2>支撑 / 阻力位</h2>
      <table><tr><th>类型</th><th>价格</th><th>距当前</th><th>强度</th></tr>
      {res_rows}{sup_rows}</table>
    </div>
    <div class="card">
      <h2>Fibonacci 回调位</h2>
      <table><tr><th>位置</th><th>价格</th><th>距当前</th></tr>
      {fib_rows}</table>
    </div>
    <div class="card">
      <h2>综合信号</h2>
      <div class="overall-box" style="border-left-color:{rd["overall_color"]}">
        <div class="overall-text" style="color:{rd["overall_color"]}">{rd["overall"]}</div>
        <div class="bar-wrap">
          <span class="bar-label">多</span>
          <div style="height:8px;background:#26a69a;border-radius:2px;width:{bull_w:.1f}%;min-width:2px"></div>
          <div style="height:8px;background:#ef5350;border-radius:2px;width:{bear_w:.1f}%;min-width:2px"></div>
          <span class="bar-label">空</span>
        </div>
      </div>
      <table><tr><th>指标</th><th>评分</th></tr>
        <tr><td>均线系统 (5项)</td><td style="color:{sc(ma_score_total)}">{ma_score_total:+d}</td></tr>
        <tr><td>RSI</td><td style="color:{rsi_c}">{rd["rsi"]["score"]:+d}</td></tr>
        <tr><td>布林带</td><td style="color:{bb_c}">{rd["bb"]["score"]:+d}</td></tr>
        <tr><td>MACD</td><td style="color:{macd_c}">{rd["macd"]["score"]:+d}</td></tr>
        <tr><td>KDJ</td><td style="color:{kdj_c}">{rd["kdj"]["score"]:+d}</td></tr>
        <tr><td>OBV</td><td style="color:{obv_c}">{rd["obv"]["score"]:+d}</td></tr>
        <tr style="font-weight:700"><td>合计 / 满分</td>
          <td style="color:{rd["overall_color"]}">{rd["score"]:+d} / {rd["max_score"]}</td></tr>
      </table>
    </div>
  </div>
  <hr class="divider">
  <div class="chart-wrap">{chart_html}</div>
</div>
'''

    display_style = "block" if is_active else "none"
    
    return f"""
    <div id="container-{symbol}" class="symbol-container" style="display:{display_style}">
        <div class="page-header">
            <div class="price-tag">${all_rd[3]["price"]:,.2f} <span style="font-size:13px;color:#90a4ae">日线收盘</span></div>
            <div style="color:#78909c;font-size:12px;margin-left:auto">数据来源: Binance API | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <!-- 全局汇总 -->
        {global_summary_html}

        <!-- 多周期汇总 -->
        <div class="summary-wrap">
          <div class="summary-title">多周期信号一览 ({symbol})</div>
          {summary_table}
        </div>

        <!-- 标签栏 -->
        <div class="tab-bar">
        {tab_buttons}
        </div>

        <!-- 各周期面板 -->
        {tab_panels}
    </div>
    """


def build_multisymbol_html(all_results: dict, filepath: str):
    """
    all_results: { "BTCUSDT": [(tf_id, tf_label, fig, rd), ...], "ETHUSDT": ... }
    """
    
    # 生成各币种的 HTML 内容
    symbols_html = ""
    symbol_options = ""
    
    first_symbol = True
    for symbol in SYMBOLS:
        if symbol not in all_results:
            continue
            
        is_active = first_symbol
        symbols_html += generate_symbol_html_content(symbol, all_results[symbol], is_active)
        
        selected = "selected" if is_active else ""
        symbol_options += f'<option value="{symbol}" {selected}>{symbol}</option>'
        first_symbol = False

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>多币种多周期技术分析报告 {datetime.now().strftime('%Y-%m-%d')}</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0e1117; color: #d1d4dc; font-family: 'Segoe UI','PingFang SC',sans-serif; font-size: 13px; }}
    h1   {{ font-size: 20px; font-weight: 600; color: #fff; margin: 0; }}
    h2   {{ font-size: 12px; font-weight: 600; color: #90a4ae; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 8px; }}
    
    .top-nav {{ background: #131722; border-bottom: 1px solid #2a2e39; padding: 10px 28px; display: flex; align-items: center; gap: 20px; }}
    .symbol-select {{ background: #0e1117; border: 1px solid #2a2e39; color: #fff; padding: 6px 12px; border-radius: 4px; font-size: 16px; font-weight: 600; cursor: pointer; outline: none; }}
    .symbol-select:hover {{ border-color: #546e7a; }}

    .page-header {{ background: linear-gradient(90deg,#131722,#1a2035); border-bottom: 1px solid #2a2e39; padding: 16px 28px; display: flex; align-items: center; gap: 20px; }}
    .price-tag   {{ font-size: 26px; font-weight: 700; color: #fff; }}
    .date-tag    {{ color: #78909c; font-size: 12px; margin-top: 2px; }}

    /* 汇总表 */
    .summary-wrap {{ padding: 14px 28px; background: #0e1117; }}
    .summary-title {{ color: #90a4ae; font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 8px; }}
    .summary-wrap table {{ width: 100%; border-collapse: collapse; }}
    .summary-wrap th {{ background: #131722; color: #546e7a; font-weight: 500; padding: 6px 10px; text-align: left; font-size: 11px; border-bottom: 1px solid #2a2e39; }}
    .summary-wrap td {{ padding: 7px 10px; border-bottom: 1px solid #1a1f2e; font-size: 12px; }}
    .summary-wrap tr:last-child td {{ border-bottom: none; }}

    /* 标签页 */
    .tab-bar {{ display: flex; gap: 2px; padding: 10px 28px 0; background: #0e1117; border-bottom: 1px solid #2a2e39; }}
    .tab-btn {{ background: transparent; border: none; border-bottom: 3px solid transparent; padding: 8px 18px; color: #78909c; font-size: 13px; cursor: pointer; transition: all .15s; }}
    .tab-btn:hover {{ color: #d1d4dc; }}
    .tab-btn.active {{ color: #90caf9; border-bottom-color: #90caf9; font-weight: 600; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}

    /* 卡片 */
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 12px; padding: 16px 28px; }}
    .card {{ background: #131722; border: 1px solid #2a2e39; border-radius: 8px; padding: 12px 14px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ color: #546e7a; font-weight: 500; padding: 4px 5px; text-align: left; border-bottom: 1px solid #1e2130; font-size: 11px; }}
    td {{ padding: 5px 5px; border-bottom: 1px solid #1a1f2e; }}
    tr:last-child td {{ border-bottom: none; }}
    .signal-row   {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .signal-label {{ color: #78909c; min-width: 100px; }}
    .signal-value {{ font-weight: 600; }}
    .overall-box  {{ margin: 4px 0 10px; padding: 10px 12px; border-radius: 6px; background: rgba(255,255,255,0.04); border-left: 4px solid; }}
    .overall-text {{ font-size: 15px; font-weight: 700; }}
    .bar-wrap     {{ display: flex; gap: 4px; align-items: center; margin-top: 6px; }}
    .bar-label    {{ font-size: 11px; color: #546e7a; }}
    .chart-wrap   {{ padding: 0 28px 28px; }}
    .divider      {{ border: none; border-top: 1px solid #2a2e39; margin: 0 28px; }}
    
    .stars {{
        display: inline-block;
        font-family: 'Times New Roman', serif;
        line-height: 1;
        font-size: 14px;
        position: relative;
    }}
    .stars::before {{
        content: '★★★★★';
        background: linear-gradient(90deg, #ffd54f calc(var(--rating) / 5 * 100%), #454545 calc(var(--rating) / 5 * 100%));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        color: transparent; /* Fallback for non-webkit */
    }}
  </style>
</head>
<body>

<div class="top-nav">
  <h1>多币种技术分析报告</h1>
  <div style="margin-left:auto;display:flex;align-items:center;gap:10px">
    <label for="symbolSelect" style="color:#78909c">选择币种:</label>
    <select id="symbolSelect" class="symbol-select" onchange="switchSymbol(this.value)">
      {symbol_options}
    </select>
  </div>
</div>

{symbols_html}

<script>
function switchSymbol(symbol) {{
    // 隐藏所有容器
    document.querySelectorAll('.symbol-container').forEach(el => el.style.display = 'none');
    // 显示选中容器
    const target = document.getElementById('container-' + symbol);
    if (target) {{
        target.style.display = 'block';
        // 触发 Plotly resize
        setTimeout(() => {{
            target.querySelectorAll('.plotly-graph-div').forEach(p => {{
                if (window.Plotly) Plotly.Plots.resize(p);
            }});
        }}, 50);
    }}
}}

function switchTab(evt, symbolPrefix, tfId) {{
    // 找到当前币种容器下的所有 tab-btn 和 tab-panel
    // 由于 ID 是唯一的 (panel-prefix-tfId)，我们只需要操作对应 ID
    // 但为了样式 active 切换，我们需要找到同组的 tabs
    
    // 更好的方式：通过 event.target 找到父级 tab-bar，然后处理兄弟元素
    const btn = evt.target;
    const tabBar = btn.parentElement;
    const container = tabBar.parentElement;
    
    // 移除该容器内所有 tab-btn 的 active
    tabBar.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    // 移除该容器内所有 tab-panel 的 active
    container.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    
    // 激活当前
    btn.classList.add('active');
    const panelId = 'panel-' + symbolPrefix + '-' + tfId;
    const panel = document.getElementById(panelId);
    if (panel) {{
        panel.classList.add('active');
        // Resize chart
        setTimeout(() => {{
             const plots = panel.querySelectorAll('.plotly-graph-div');
             plots.forEach(p => {{ if (window.Plotly) Plotly.Plots.resize(p); }});
        }}, 50);
    }}
}}
</script>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[导出] 完整多币种报告已保存: {filepath}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  多币种多周期技术分析工具")
    print("  支持币种: " + ", ".join(SYMBOLS))
    print("=" * 60)
    
    all_results = {}  # { "BTCUSDT": [...], ... }

    # 依次分析所有币种
    for symbol in SYMBOLS:
        print(f"\n{'='*30}\n  开始分析: {symbol}\n{'='*30}")
        
        try:
            tf_results  = []   # [(tf_id, tf_label, fig, rd), ...]
            all_texts   = []   # [(tf_label, report_text), ...]

            for interval, limit, label, chart_show, sr_lb, sr_pn, sr_cluster_pct, fib_lb in TIMEFRAMES:
                print(f"[{symbol}] [{label}] 正在获取数据...")
                df = fetch_data(symbol=symbol, interval=interval, limit=limit)
                if df.empty:
                    print(f"[{symbol}] [{label}] 跳过（数据获取失败）")
                    continue

                print(f"[{symbol}] [{label}] 最新: {df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
                df = calculate_indicators(df)
                supports, resistances = find_support_resistance(df, lookback=sr_lb, pivot_n=sr_pn, cluster_pct=sr_cluster_pct)
                fib_levels, fib_high, fib_low = calculate_fibonacci(df, lookback=fib_lb)

                # print(f"[{symbol}] [{label}] 正在分析...")
                score, report_text, rd = generate_report(
                    df, supports, resistances, fib_levels, symbol=symbol, tf_label=label, verbose=False
                )
                all_texts.append((label, report_text))

                # print(f"[{symbol}] [{label}] 正在生成图表...")
                fig = create_chart(df, supports, resistances, fib_levels, fib_high, fib_low,
                                   symbol=symbol, tf_label=label, chart_show=chart_show)
                tf_results.append((interval, label, fig, rd))

                # 避免触发 Binance 速率限制
                time.sleep(0.1)

            if tf_results:
                all_results[symbol] = tf_results
                
                # 导出单个币种的 TXT
                os.makedirs("output", exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                txt_path  = f"output/{symbol}_report_{ts}.txt"
                export_txt_report(all_texts, txt_path, symbol=symbol)
            else:
                print(f"[错误] {symbol} 所有周期数据获取失败")
        
        except Exception as e:
            print(f"[异常] 分析 {symbol} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        print("[错误] 没有获取到任何数据")
        return

    # 导出合并 HTML
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = f"output/multisymbol_report_{ts}.html"

    print(f"\n正在生成合并 HTML 报告...")
    build_multisymbol_html(all_results, html_path)

    print(f"\n[完成] 报告已生成: {html_path}")
    print("[报告] 正在浏览器中打开...")

    import webbrowser, pathlib
    webbrowser.open(pathlib.Path(html_path).resolve().as_uri())


if __name__ == "__main__":
    main()
