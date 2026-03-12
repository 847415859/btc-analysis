# -*- coding: utf-8 -*-
"""
BTC/USDT 多周期技术分析工具
数据来源: Binance 公开 API
周期: 15分钟 / 1小时 / 4小时 / 日线 / 周线
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
    # (interval, limit, label_cn, chart_show, sr_lookback, sr_pivot_n, fib_lookback)
    ("15m", 500, "15分钟",  250, 100, 3,  30),
    ("1h",  500, "1小时",   300, 200, 4,  72),
    ("4h",  500, "4小时",   300, 200, 4,  90),
    ("1d",  500, "日线",    300, 250, 5,  90),
    ("1w",  200, "周线",    150, 100, 3,  52),
]


# ─────────────────────────────────────────────
# A. 数据获取
# ─────────────────────────────────────────────

def fetch_btc_data(interval: str = "1d", limit: int = 500) -> pd.DataFrame:
    """通过 Binance API 获取 BTC/USDT K 线数据"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[错误] 无法获取 {interval} 数据: {e}")
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
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
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
    return df


# ─────────────────────────────────────────────
# C. 支撑 / 阻力识别
# ─────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, lookback: int = 250, pivot_n: int = 5):
    recent = df.tail(lookback).copy()
    current_price = df["close"].iloc[-1]

    pivot_highs, pivot_lows = [], []
    for i in range(pivot_n, len(recent) - pivot_n):
        w_h = recent["high"].iloc[i - pivot_n: i + pivot_n + 1]
        w_l = recent["low"].iloc[i - pivot_n: i + pivot_n + 1]
        if recent["high"].iloc[i] == w_h.max():
            pivot_highs.append(recent["high"].iloc[i])
        if recent["low"].iloc[i] == w_l.min():
            pivot_lows.append(recent["low"].iloc[i])

    price_range = recent["close"].max() - recent["close"].min()
    if price_range == 0:
        return [], []
    num_bins = 50
    bins = np.linspace(recent["close"].min(), recent["close"].max(), num_bins + 1)
    vol_profile = np.zeros(num_bins)
    for _, row in recent.iterrows():
        idx = min(int((row["close"] - recent["close"].min()) / price_range * num_bins), num_bins - 1)
        vol_profile[idx] += row["volume"]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    top_vol_idx = np.argsort(vol_profile)[-10:]
    vol_levels  = bin_centers[top_vol_idx].tolist()

    all_highs = pivot_highs + [p for p in vol_levels if p > current_price]
    all_lows  = pivot_lows  + [p for p in vol_levels if p < current_price]

    def cluster_levels(levels, thr=0.015):
        if not levels:
            return []
        levels = sorted(levels)
        clusters, cur = [], [levels[0]]
        for p in levels[1:]:
            if abs(p - cur[-1]) / cur[-1] < thr:
                cur.append(p)
            else:
                clusters.append(np.mean(cur))
                cur = [p]
        clusters.append(np.mean(cur))
        return clusters

    res = sorted(cluster_levels(all_highs), reverse=True)[:3]
    sup = sorted(cluster_levels(all_lows))[-3:]
    return [(p, 1.0) for p in sup], [(p, 1.0) for p in res]


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
                 tf_label: str = "", chart_show: int = 300):
    plot_df = df.tail(chart_show).copy()

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.40, 0.12, 0.14, 0.12, 0.12, 0.10],
        subplot_titles=(
            f"BTC/USDT {tf_label} K 线（均线 · 布林带 · 支撑/阻力 · Fibonacci）",
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
            text=f"BTC/USDT {tf_label} 综合技术分析  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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
                    fib_levels, tf_label: str = "", verbose: bool = True):
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
    ts  = latest["date"].strftime("%Y-%m-%d %H:%M") if tf_label in ("15分钟","1小时","4小时") \
          else latest["date"].strftime("%Y-%m-%d")
    emit(f"\n{'═'*60}")
    emit(f"  BTC/USDT [{tf_label}] 技术分析报告  |  {ts}")
    emit(f"  当前收盘价: ${price:,.2f}")
    emit(f"{'═'*60}\n")
    rd["date"]  = ts
    rd["price"] = price
    rd["tf"]    = tf_label

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
        score += 1 if is_above else -1
        emit(f"  {name:<6}({desc}): ${val:>10,.2f}  {'上方 ↑' if is_above else '下方 ↓'} {diff_pct:+.2f}%  [{'+'if is_above else'-'}]")
        rd["ma"].append({"name": name, "desc": desc, "val": val,
                         "diff_pct": diff_pct, "above": is_above, "signal": "+" if is_above else "-"})

    # ── RSI ──
    rsi_val = latest["rsi"]
    emit(f"\n【RSI(14)】{sep}")
    rsi_score = 0
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

    # ── 支撑/阻力 ──
    emit(f"\n【支撑 / 阻力位】{sep}")
    emit("  阻力位 (从近到远):")
    res_list = []
    for pr, _ in sorted(resistances, key=lambda x: x[0]):
        dist = (pr - price) / price * 100
        emit(f"    ${pr:>10,.2f}  距当前 +{dist:.2f}%")
        res_list.append({"price": pr, "dist_pct": dist})
    emit("  支撑位 (从近到远):")
    sup_list = []
    for ps, _ in sorted(supports, key=lambda x: x[0], reverse=True):
        dist = (price - ps) / price * 100
        emit(f"    ${ps:>10,.2f}  距当前 -{dist:.2f}%")
        sup_list.append({"price": ps, "dist_pct": dist})
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

    # ── 综合信号 ──
    max_score = 16
    emit(f"\n{'═'*60}")
    if score >= 5:   overall = f"偏多头  (得分: {score:+d}/{max_score})"; oc = "#26a69a"
    elif score <= -5: overall = f"偏空头  (得分: {score:+d}/{max_score})"; oc = "#ef5350"
    else:            overall = f"中性震荡  (得分: {score:+d}/{max_score})"; oc = "#f9a825"
    emit(f"  综合信号: {overall}")
    emit(f"  多空强度: {'█'*max(0,score)}{'░'*max(0,-score)}")
    emit(f"{'═'*60}\n")
    rd.update({"score": score, "max_score": max_score, "overall": overall,
               "overall_color": oc,
               "bull_bar": "█"*max(0,score), "bear_bar": "░"*max(0,-score)})

    return score, "\n".join(lines), rd


# ─────────────────────────────────────────────
# G. 导出 TXT 报告
# ─────────────────────────────────────────────

def export_txt_report(all_texts: list, filepath: str):
    header = ("=" * 60 + "\n"
              "  BTC/USDT 多周期技术分析报告\n"
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

    header = "<tr><th>周期</th><th>价格</th><th>均线</th><th>RSI</th><th>布林带</th><th>MACD</th><th>KDJ</th><th>OBV</th><th>综合</th></tr>"
    rows = ""
    for rd in all_rd:
        ma_s  = sum(1 if m["above"] else -1 for m in rd["ma"])
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
            f'</tr>'
        )
    return f'<table><thead>{header}</thead><tbody>{rows}</tbody></table>'


# ─────────────────────────────────────────────
# I. 导出合并多标签 HTML 报告
# ─────────────────────────────────────────────

def build_combined_html(tf_results: list, filepath: str):
    """
    tf_results: [(tf_id, tf_label, fig, report_data), ...]
    生成带标签页（Tab）的单 HTML 文件：顶部汇总 + 各周期详细卡片 + 图表
    """
    all_rd = [rd for _, _, _, rd in tf_results]

    # ── 各周期 Tab 内容 ──
    tab_buttons = ""
    tab_panels  = ""

    for i, (tf_id, tf_label, fig, rd) in enumerate(tf_results):
        active_btn   = "active" if i == 3 else ""  # 默认选中日线(index 3)
        active_panel = "active" if i == 3 else ""

        chart_html = fig.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False))

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
            f'<td style="color:#ef5350">+{r["dist_pct"]:.2f}%</td></tr>'
            for r in rd["resistances"]
        )
        sup_rows = "".join(
            f'<tr><td style="color:#26a69a">支撑</td>'
            f'<td>${s["price"]:,.2f}</td>'
            f'<td style="color:#26a69a">-{s["dist_pct"]:.2f}%</td></tr>'
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

        tab_buttons += f'<button class="tab-btn {active_btn}" onclick="switchTab(event,\'{tf_id}\')">{tf_label}</button>\n'

        tab_panels += f'''
<div id="panel-{tf_id}" class="tab-panel {active_panel}">
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
      <table><tr><th>类型</th><th>价格</th><th>距当前</th></tr>
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

    summary_table = build_summary_table(all_rd)

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>BTC/USDT 多周期技术分析  {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0e1117; color: #d1d4dc; font-family: 'Segoe UI','PingFang SC',sans-serif; font-size: 13px; }}
    h1   {{ font-size: 20px; font-weight: 600; color: #fff; }}
    h2   {{ font-size: 12px; font-weight: 600; color: #90a4ae; letter-spacing: .05em; text-transform: uppercase; margin-bottom: 8px; }}
    .page-header {{ background: linear-gradient(90deg,#131722,#1a2035); border-bottom: 1px solid #2a2e39; padding: 16px 28px; display: flex; align-items: center; gap: 20px; }}
    .price-tag   {{ font-size: 26px; font-weight: 700; color: #fff; margin-left: auto; }}
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
  </style>
</head>
<body>

<div class="page-header">
  <div>
    <h1>BTC/USDT &nbsp; 多周期技术分析报告</h1>
    <div class="date-tag">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; 来源: Binance API</div>
  </div>
  <div class="price-tag">${all_rd[3]["price"]:,.2f} <span style="font-size:13px;color:#90a4ae">日线收盘</span></div>
</div>

<!-- 多周期汇总 -->
<div class="summary-wrap">
  <div class="summary-title">多周期信号一览</div>
  {summary_table}
</div>

<!-- 标签栏 -->
<div class="tab-bar">
{tab_buttons}
</div>

<!-- 各周期面板 -->
{tab_panels}

<script>
function switchTab(evt, tfId) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  evt.target.classList.add('active');
  document.getElementById('panel-' + tfId).classList.add('active');
  // 触发 Plotly resize 以修复图表宽度
  setTimeout(function() {{
    var plots = document.getElementById('panel-' + tfId).querySelectorAll('.plotly-graph-div');
    plots.forEach(function(p) {{ if (window.Plotly) Plotly.Plots.resize(p); }});
  }}, 50);
}}
</script>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[导出] 完整多周期报告已保存: {filepath}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BTC/USDT 多周期技术分析工具")
    print("  周期: 15分钟 / 1小时 / 4小时 / 日线 / 周线")
    print("=" * 60)

    tf_results  = []   # [(tf_id, tf_label, fig, rd), ...]
    all_texts   = []   # [(tf_label, report_text), ...]

    for interval, limit, label, chart_show, sr_lb, sr_pn, fib_lb in TIMEFRAMES:
        print(f"\n[{label}] 正在获取数据...")
        df = fetch_btc_data(interval=interval, limit=limit)
        if df.empty:
            print(f"[{label}] 跳过（数据获取失败）")
            continue

        print(f"[{label}] 获取 {len(df)} 根K线，最新: {df['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
        df = calculate_indicators(df)
        supports, resistances = find_support_resistance(df, lookback=sr_lb, pivot_n=sr_pn)
        fib_levels, fib_high, fib_low = calculate_fibonacci(df, lookback=fib_lb)

        print(f"[{label}] 正在分析...")
        score, report_text, rd = generate_report(
            df, supports, resistances, fib_levels, tf_label=label
        )
        all_texts.append((label, report_text))

        print(f"[{label}] 正在生成图表...")
        fig = create_chart(df, supports, resistances, fib_levels, fib_high, fib_low,
                           tf_label=label, chart_show=chart_show)
        tf_results.append((interval, label, fig, rd))

        # 避免触发 Binance 速率限制
        time.sleep(0.3)

    if not tf_results:
        print("[错误] 所有周期数据获取失败，请检查网络连接")
        return

    # 导出文件
    os.makedirs("output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    txt_path  = f"output/btc_report_{ts}.txt"
    html_path = f"output/btc_report_{ts}.html"

    export_txt_report(all_texts, txt_path)
    build_combined_html(tf_results, html_path)

    print(f"\n[完成] 输出文件:")
    print(f"  文本报告: {txt_path}")
    print(f"  完整报告: {html_path}")
    print("[报告] 正在浏览器中打开...")

    import webbrowser, pathlib
    webbrowser.open(pathlib.Path(html_path).resolve().as_uri())


if __name__ == "__main__":
    main()
