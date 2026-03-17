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

# 导入链上数据引擎
try:
    from onchain_engine import OnChainMonitor
    onchain_monitor = OnChainMonitor()
except ImportError:
    onchain_monitor = None
    print("[Warning] onchain_engine.py not found. On-chain analysis will be disabled.")

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
# A-2. 扩展历史数据获取（回测专用，分批拼接）
# ─────────────────────────────────────────────

def fetch_data_extended(symbol: str = "BTCUSDT", interval: str = "1h",
                        total: int = 1500) -> pd.DataFrame:
    """
    分批拉取历史 K 线，突破单次 500 根上限，最多拉取 total 根。
    通过 endTime 参数向前滚动，逆向拼接后按时间正序返回。
    适用于回测引擎，不用于实时展示。
    """
    url      = "https://api.binance.com/api/v3/klines"
    per_req  = 500          # Binance 单次上限
    pages    = max(1, -(-total // per_req))   # 向上取整
    frames   = []
    end_ms   = None         # None = 当前时间

    for page in range(pages):
        params = {"symbol": symbol, "interval": interval, "limit": per_req}
        if end_ms is not None:
            params["endTime"] = end_ms

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 60))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(2)
        else:
            break

        raw = resp.json()
        if not raw:
            break

        df_page = pd.DataFrame(raw, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df_page["date"] = pd.to_datetime(df_page["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df_page[col] = df_page[col].astype(float)
        df_page["taker_buy_base"] = df_page["taker_buy_base"].astype(float)
        df_page = df_page[["date", "open_time", "open", "high", "low",
                            "close", "volume", "taker_buy_base"]]
        frames.append(df_page)

        # 下一页的截止时间 = 本页最早 K 线开盘时间 - 1ms
        end_ms = int(df_page["open_time"].iloc[0]) - 1
        time.sleep(0.15)   # 避免过快触发限流

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(reversed(frames), ignore_index=True)
    df_all = df_all.drop_duplicates(subset="open_time").sort_values("date")
    df_all = df_all[["date", "open", "high", "low", "close",
                     "volume", "taker_buy_base"]].reset_index(drop=True)
    return df_all.tail(total).reset_index(drop=True)


# ─────────────────────────────────────────────
# B. 技术指标计算
# ─────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame, tf_interval: str = "") -> pd.DataFrame:
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

    # ── VWAP ────────────────────────────────────────
    # 日线/周线：每根K线即一个交易日，按日重置的VWAP退化为典型价，
    # 改用20期滚动VWAP（类似VWAP锚定于近期成交重心）
    tp = (df["high"] + df["low"] + df["close"]) / 3
    if tf_interval in ("1d", "1w"):
        df["vwap"] = (
            (tp * df["volume"]).rolling(20, min_periods=1).sum()
            / df["volume"].rolling(20, min_periods=1).sum()
        )
    else:
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
                             cluster_pct: float = 0.015, ob_levels: list = None,
                             liq_levels: list = None,
                             vp_data: dict = None, fvg_zones: list = None):
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

    # 4. Volume Profile 节点（POC 权重最高 = 机构真实价值共识）
    if vp_data:
        poc = vp_data.get("poc")
        vah = vp_data.get("vah")
        val = vp_data.get("val")
        if poc and 0.8 * current_price <= poc <= 1.2 * current_price:
            raw_levels.append((poc, 3.5, "vp_poc"))   # POC：最高权重
        if vah and 0.8 * current_price <= vah <= 1.2 * current_price:
            raw_levels.append((vah, 2.5, "vp_vah"))
        if val and 0.8 * current_price <= val <= 1.2 * current_price:
            raw_levels.append((val, 2.5, "vp_val"))
        for node in (vp_data.get("hvn") or [])[:3]:
            np_ = node.get("price", 0)
            if np_ and 0.85 * current_price <= np_ <= 1.15 * current_price:
                raw_levels.append((np_, 1.8, "vp_hvn"))

    # 5. Fair Value Gap（FVG 中点和边界作为高精度 S/R）
    if fvg_zones:
        for fvg in fvg_zones[:6]:
            mp = fvg.get("mid", 0)
            bp = fvg["bottom"]; tp = fvg["top"]
            fvg_src = "fvg_bull" if fvg["type"] == "bullish_fvg" else "fvg_bear"
            # 中点是机构挂单核心价格，权重最高
            if mp and 0.85 * current_price <= mp <= 1.15 * current_price:
                raw_levels.append((mp, 2.5, fvg_src))
            # 缺口边界作为次级 S/R
            for edge in [bp, tp]:
                if edge and 0.85 * current_price <= edge <= 1.15 * current_price:
                    raw_levels.append((edge, 1.5, fvg_src))

    # 6. 清算密集区（来自 WebSocket 累积或 OI 估算）
    # 多头清算区 → 价格下方潜在支撑（多头止损密集，一旦触及会产生大量卖压，但也是空方目标位）
    # 空头清算区 → 价格上方潜在阻力（空头止损密集，轧空后变为强支撑）
    # 权重 2.5：介于订单簿(2-3)和成交量节点(2)之间，反映其真实的价格磁吸效应
    if liq_levels:
        for lc in liq_levels:
            lc_price = lc.get("price", 0)
            dominant = lc.get("dominant", "")
            if not lc_price:
                continue
            # 只收录当前价格 ±15% 范围内（清算区距离太远参考意义不大）
            if not (0.85 * current_price <= lc_price <= 1.15 * current_price):
                continue
            # 估算来源权重稍低（1.5），WebSocket 真实数据权重更高（2.5）
            liq_src    = lc.get("source", "estimated")
            liq_weight = 2.5 if liq_src == "websocket" else 1.5
            # 来源标签：区分多头/空头清算区
            if "多头" in dominant:
                raw_levels.append((lc_price, liq_weight, "liq_long"))
            elif "空头" in dominant:
                raw_levels.append((lc_price, liq_weight, "liq_short"))

    # 5. 聚类合并（支持 3-tuple，合并来源标签）
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

    # 6. 为每个聚类生成 reason 文字（解释为何选为支撑/阻力）
    def build_reason(src: str, weight: float, price: float, cur_price: float) -> str:
        parts  = src.split("+")
        reason_parts = []

        if "pivot" in parts:
            reason_parts.append("价格分形高低点")
        if "volume" in parts:
            reason_parts.append("成交量密集节点（筹码集中）")
        if any(p.startswith("orderbook_bid") for p in parts):
            reason_parts.append("订单簿大额买单挂墙")
        if any(p.startswith("orderbook_ask") for p in parts):
            reason_parts.append("订单簿大额卖单挂墙")
        if "liq_long" in parts:
            reason_parts.append("多头清算密集区（强平踩踏风险）")
        if "liq_short" in parts:
            reason_parts.append("空头清算密集区（轧空加速点）")
        if "vp_poc" in parts:
            reason_parts.append("成交量POC（市场真实价值共识点）")
        if any(p in ("vp_vah", "vp_val") for p in parts):
            reason_parts.append("成交量价值区间边界(VAH/VAL)")
        if "vp_hvn" in parts:
            reason_parts.append("高成交量节点(HVN)")
        if "fvg_bull" in parts:
            reason_parts.append("看涨价格失衡区(FVG↑)——回调磁吸")
        if "fvg_bear" in parts:
            reason_parts.append("看跌价格失衡区(FVG↓)——反弹磁吸")

        # 多源叠加说明
        n_sources = len(parts)
        if n_sources >= 3:
            confluence = "三源共振"
        elif n_sources == 2:
            confluence = "双源叠加"
        else:
            confluence = None

        # 权重强度说明
        if weight >= 6.0:
            strength_desc = "极强"
        elif weight >= 4.0:
            strength_desc = "强"
        elif weight >= 2.5:
            strength_desc = "中等"
        else:
            strength_desc = "一般"

        reason = "、".join(reason_parts) if reason_parts else "价格分形"
        if confluence:
            reason = f"{confluence}：{reason}"
        reason += f"（强度 {strength_desc} / {weight:.1f}）"
        return reason

    # 7. 区分支撑/阻力，附带 reason
    supports    = [(p, w, src, build_reason(src, w, p, current_price))
                   for p, w, src in consolidated if p < current_price]
    resistances = [(p, w, src, build_reason(src, w, p, current_price))
                   for p, w, src in consolidated if p > current_price]

    # 8. 排序并取前5：支撑从大到小，阻力从大到小（均按距离当前价由近→远）
    supports    = sorted(supports,    key=lambda x: x[0], reverse=True)[:5]
    resistances = sorted(resistances, key=lambda x: x[0], reverse=True)[:5]

    return supports, resistances


# ─────────────────────────────────────────────
# C-2. Volume Profile（VPVR · POC / VAH / VAL）
# ─────────────────────────────────────────────

def calculate_volume_profile(df: pd.DataFrame, lookback: int = 300,
                              num_bins: int = 100) -> dict:
    """
    计算成交量分布（Volume Profile Visible Range）
    POC = 成交量最密集价格 → 市场真实价值共识点
    VAH / VAL = 含70%成交量的价值区间上/下沿
    HVN = 高成交量节点，次级 S/R
    """
    recent = df.tail(lookback)
    if len(recent) < 10:
        return {}
    price_min = float(recent["low"].min())
    price_max = float(recent["high"].max())
    if price_max <= price_min:
        return {}

    span     = (price_max - price_min) / num_bins
    bins     = np.linspace(price_min, price_max, num_bins + 1)
    bin_ctrs = (bins[:-1] + bins[1:]) / 2
    bin_vol  = np.zeros(num_bins)

    for _, row in recent.iterrows():
        lo, hi, vol = float(row["low"]), float(row["high"]), float(row["volume"])
        if hi <= lo or vol <= 0 or span == 0:
            continue
        idx_lo = max(0,          int((lo - price_min) / span))
        idx_hi = min(num_bins-1, int((hi - price_min) / span))
        n_span = max(1, idx_hi - idx_lo + 1)
        bin_vol[idx_lo : idx_hi + 1] += vol / n_span

    poc_idx = int(np.argmax(bin_vol))
    poc     = float(bin_ctrs[poc_idx])

    # Value Area：以 POC 为核心向两侧扩展至 70% 总量
    total_vol  = float(bin_vol.sum())
    target_vol = total_vol * 0.70
    va_lo = va_hi = poc_idx
    va_vol = float(bin_vol[poc_idx])

    while va_vol < target_vol and (va_lo > 0 or va_hi < num_bins - 1):
        nxt_lo = float(bin_vol[va_lo - 1]) if va_lo > 0          else -1.0
        nxt_hi = float(bin_vol[va_hi + 1]) if va_hi < num_bins-1 else -1.0
        if nxt_lo >= nxt_hi and va_lo > 0:
            va_lo -= 1;  va_vol += bin_vol[va_lo]
        elif va_hi < num_bins - 1:
            va_hi += 1;  va_vol += bin_vol[va_hi]
        else:
            break

    val = float(bins[va_lo])
    vah = float(bins[va_hi + 1])
    va_width_pct = (vah - val) / poc * 100 if poc else 0

    sorted_idx = np.argsort(bin_vol)[::-1]
    hvn = []
    for idx in sorted_idx:
        if idx == poc_idx:
            continue
        hvn.append({"price": round(float(bin_ctrs[idx]), 2),
                    "vol_pct": round(float(bin_vol[idx] / total_vol * 100), 1)})
        if len(hvn) >= 5:
            break

    if   va_width_pct > 8: dist_type = "分散型（无明显共识）"
    elif va_width_pct < 3: dist_type = "集中型（强价值共识）"
    else:                  dist_type = "均衡型"

    current_price = float(df["close"].iloc[-1])
    max_vol       = float(bin_vol.max()) or 1.0
    top_idxs      = np.argsort(bin_vol)[::-1][:20]
    chart_bins    = sorted([
        {"price":   round(float(bin_ctrs[i]), 2),
         "vol_pct": round(float(bin_vol[i] / total_vol * 100), 1),
         "bar_pct": round(float(bin_vol[i] / max_vol * 100),   1)}
        for i in top_idxs
    ], key=lambda x: x["price"], reverse=True)

    return {
        "poc":          round(poc, 2),
        "vah":          round(vah, 2),
        "val":          round(val, 2),
        "va_width_pct": round(va_width_pct, 2),
        "dist_type":    dist_type,
        "poc_above":    poc > current_price,
        "hvn":          hvn,
        "chart_bins":   chart_bins,
    }


# ─────────────────────────────────────────────
# C-3. Fair Value Gap（价格失衡区 / FVG）
# ─────────────────────────────────────────────

def detect_fair_value_gaps(df: pd.DataFrame,
                            min_gap_pct:     float = 0.002,
                            max_age_bars:    int   = 150,
                            price_range_pct: float = 0.15) -> list:
    """
    识别价格失衡区（Fair Value Gap）
    Bullish FVG: K[i].low  > K[i-2].high  → 向上缺口，回调时强支撑
    Bearish FVG: K[i].high < K[i-2].low   → 向下缺口，反弹时强阻力
    机构惯于在 FVG 中点挂单等待价格回填，回填概率 >60%。
    """
    n = len(df)
    if n < 3:
        return []

    current_price = float(df["close"].iloc[-1])
    lo_bound      = current_price * (1 - price_range_pct)
    hi_bound      = current_price * (1 + price_range_pct)
    scan_start    = max(2, n - max_age_bars - 2)

    fvgs = []
    for i in range(scan_start, n):
        h0 = float(df["high"].iloc[i - 2]);  l0 = float(df["low"].iloc[i - 2])
        h2 = float(df["high"].iloc[i]);       l2 = float(df["low"].iloc[i])
        d2 = df["date"].iloc[i];              age = n - i

        # Bullish FVG
        if l2 > h0:
            gap_pct = (l2 - h0) / h0
            if gap_pct >= min_gap_pct:
                top, bot = l2, h0;  mid = (top + bot) / 2
                fp = 0.0
                if i < n - 1:
                    post_lo = float(df["low"].iloc[i + 1:].min())
                    if   post_lo <= bot:  fp = 100.0
                    elif post_lo < top:   fp = (top - post_lo) / (top - bot) * 100
                if fp < 90 and lo_bound <= mid <= hi_bound:
                    fvgs.append({
                        "type": "bullish_fvg", "top": round(top, 2),
                        "bottom": round(bot, 2), "mid": round(mid, 2),
                        "size_pct": round(gap_pct * 100, 3), "age_bars": age,
                        "formed_at": str(d2)[:16], "filled_pct": round(fp, 1),
                        "direction": "support",
                        "desc": f"FVG↑ ${bot:,.0f}–${top:,.0f} ({gap_pct*100:.2f}%)，{age}根前，已填{fp:.0f}%",
                    })

        # Bearish FVG
        if h2 < l0:
            gap_pct = (l0 - h2) / l0
            if gap_pct >= min_gap_pct:
                top, bot = l0, h2;  mid = (top + bot) / 2
                fp = 0.0
                if i < n - 1:
                    post_hi = float(df["high"].iloc[i + 1:].max())
                    if   post_hi >= top:  fp = 100.0
                    elif post_hi > bot:   fp = (post_hi - bot) / (top - bot) * 100
                if fp < 90 and lo_bound <= mid <= hi_bound:
                    fvgs.append({
                        "type": "bearish_fvg", "top": round(top, 2),
                        "bottom": round(bot, 2), "mid": round(mid, 2),
                        "size_pct": round(gap_pct * 100, 3), "age_bars": age,
                        "formed_at": str(d2)[:16], "filled_pct": round(fp, 1),
                        "direction": "resistance",
                        "desc": f"FVG↓ ${bot:,.0f}–${top:,.0f} ({gap_pct*100:.2f}%)，{age}根前，已填{fp:.0f}%",
                    })

    fvgs.sort(key=lambda x: abs(x["mid"] - current_price))
    return fvgs[:8]


# ─────────────────────────────────────────────
# C-4. Market Structure（BOS + CHoCH）
# ─────────────────────────────────────────────

def detect_market_structure(df: pd.DataFrame,
                             pivot_n: int = 3,
                             lookback: int = 150) -> dict:
    """
    识别市场结构：BOS（结构突破）和 CHoCH（性质改变）
    多头结构 (HH+HL)：顺势做多  |  空头结构 (LH+LL)：顺势做空
    BOS = 沿趋势突破摆动高/低点（延续）
    CHoCH = 逆趋势突破摆动低/高点（转变预警）
    """
    recent = df.tail(lookback).reset_index(drop=True)
    n      = len(recent)

    _empty = {
        "structure": "unknown", "structure_cn": "数据不足",
        "structure_score": 0,   "trading_bias": "数据不足",
        "events": [], "key_high": None, "key_low": None,
        "swing_highs": [], "swing_lows": [],
    }
    if n < pivot_n * 4 + 2:
        return _empty

    highs, lows = [], []
    for i in range(pivot_n, n - pivot_n):
        ph = float(recent["high"].iloc[i])
        pl = float(recent["low"].iloc[i])
        if ph == float(recent["high"].iloc[i - pivot_n : i + pivot_n + 1].max()):
            highs.append({"price": ph, "date": recent["date"].iloc[i], "idx": i})
        if pl == float(recent["low"].iloc[i - pivot_n : i + pivot_n + 1].min()):
            lows.append({"price": pl,  "date": recent["date"].iloc[i], "idx": i})

    if len(highs) < 2 or len(lows) < 2:
        return {**_empty, "structure_cn": "摆动点不足"}

    last_sh, prev_sh = highs[-1], highs[-2]
    last_sl, prev_sl = lows[-1],  lows[-2]
    hh = last_sh["price"] > prev_sh["price"];  hl = last_sl["price"] > prev_sl["price"]
    lh = last_sh["price"] < prev_sh["price"];  ll = last_sl["price"] < prev_sl["price"]

    if len(highs) >= 3 and len(lows) >= 3:
        sh3, sl3 = highs[-3], lows[-3]
        hh_s = (1 if hh else 0) + (1 if prev_sh["price"] > sh3["price"] else 0)
        hl_s = (1 if hl else 0) + (1 if prev_sl["price"] > sl3["price"] else 0)
        lh_s = (1 if lh else 0) + (1 if prev_sh["price"] < sh3["price"] else 0)
        ll_s = (1 if ll else 0) + (1 if prev_sl["price"] < sl3["price"] else 0)
    else:
        hh_s = 1 if hh else 0;  hl_s = 1 if hl else 0
        lh_s = 1 if lh else 0;  ll_s = 1 if ll else 0

    if   hh_s >= 1 and hl_s >= 1:
        st, st_cn, st_sc, bias = "bullish",     "多头结构 (HH+HL) ↑", +2, "高低点持续抬升，顺势做多，回调低点是买入机会"
    elif lh_s >= 1 and ll_s >= 1:
        st, st_cn, st_sc, bias = "bearish",     "空头结构 (LH+LL) ↓", -2, "高低点持续下移，顺势做空，反弹高点是卖出机会"
    elif hh_s >= 1 and ll_s >= 1:
        st, st_cn, st_sc, bias = "distribution","派发结构 (HH+LL)",    -1, "区间扩张，高点抬升但低点下移，等待方向突破"
    elif lh_s >= 1 and hl_s >= 1:
        st, st_cn, st_sc, bias = "accumulation","吸筹结构 (LH+HL)",    +1, "区间收窄震荡，等待 CHoCH↑ 确认多头方向"
    else:
        st, st_cn, st_sc, bias = "transitional","过渡震荡（方向未明）", 0,  "结构混乱，避免重仓，等待清晰高低点序列"

    events = []
    ksh, ksl = last_sh["price"], last_sl["price"]
    tail30 = recent.iloc[max(0, n - 31):]

    for j in range(1, len(tail30)):
        pc = float(tail30["close"].iloc[j - 1])
        cc = float(tail30["close"].iloc[j])
        cd = tail30["date"].iloc[j]
        if pc <= ksh < cc:
            et = "BOS_UP" if st in ("bullish", "accumulation") else "CHoCH_UP"
            events.append({"type": et, "price": round(ksh, 2), "date": str(cd)[:16],
                           "direction": "bullish",
                           "cn": "结构突破↑（趋势延续）" if et == "BOS_UP" else "性质改变↑（空转多！）"})
        if pc >= ksl > cc:
            et = "BOS_DOWN" if st in ("bearish", "distribution") else "CHoCH_DOWN"
            events.append({"type": et, "price": round(ksl, 2), "date": str(cd)[:16],
                           "direction": "bearish",
                           "cn": "结构突破↓（趋势延续）" if et == "BOS_DOWN" else "性质改变↓（多转空！）"})

    rev = events[-3:]
    if any(e["type"] == "CHoCH_UP"   for e in rev):
        st_sc = max(st_sc, +2);  bias = "CHoCH↑ 确认：空头结构已被打破，多头接管市场"
    if any(e["type"] == "CHoCH_DOWN" for e in rev):
        st_sc = min(st_sc, -2);  bias = "CHoCH↓ 确认：多头结构已被打破，空头接管市场"

    cp = float(recent["close"].iloc[-1])
    return {
        "structure":       st,
        "structure_cn":    st_cn,
        "structure_score": st_sc,
        "trading_bias":    bias,
        "events":          events[-5:],
        "key_high": {"price": round(last_sh["price"], 2), "date": str(last_sh["date"])[:16],
                     "dist_pct": round((last_sh["price"] - cp) / cp * 100, 2)},
        "key_low":  {"price": round(last_sl["price"], 2), "date": str(last_sl["date"])[:16],
                     "dist_pct": round((cp - last_sl["price"]) / cp * 100, 2)},
        "swing_highs": [{"price": round(h["price"], 2), "date": str(h["date"])[:16]} for h in highs[-5:]],
        "swing_lows":  [{"price": round(l["price"], 2), "date": str(l["date"])[:16]} for l in lows[-5:]],
    }


# ─────────────────────────────────────────────
# C-5. 流动性猎杀区估算（磁吸效应模型）
# ─────────────────────────────────────────────

def estimate_stop_hunt_zones(supports, resistances, current_price: float,
                              atr_val: float, oi_usdt: float = None) -> list:
    """
    机构视角：支撑/阻力位是止损单聚集地（流动性池）。
    价格受「磁吸效应」吸引至此，清扫止损后反转。

    核心结论：
      ① 不要在支撑位买入——要在支撑下方止损潮结束后买入
      ② 不要在阻力位卖出——要在阻力上方轧空结束后卖出
      ③ 磁力强度 = S/R权重 × 距离衰减(指数) × OI规模因子

    返回按磁力强度降序排列的猎杀区列表，每项包含：
      type, direction_label, sr_level, hunt_zone_low/high, hunt_target,
      magnetic_strength, hunt_probability, recommended_entry, stop_loss,
      expected_gain_pct, near_hunt, distance_pct, warning, action
    """
    import math
    zones = []

    if not atr_val or atr_val <= 0 or not current_price:
        return []

    # ATR 百分比（无量纲，用于归一化距离衰减的比较基准）
    atr_pct = atr_val / current_price

    # OI 规模因子：全市场杠杆仓位越多，止损单密度越高，磁吸越强
    # BTC 典型 OI：100亿~500亿 USDT → 归一化到 [0.6, 1.8]
    oi_factor = 1.0
    if oi_usdt and oi_usdt > 0:
        oi_factor = min(1.8, max(0.6, oi_usdt / 25_000_000_000))

    def _build_zone(sr_price, sr_weight, is_long_hunt):
        """构造单个猎杀区对象"""
        if is_long_hunt:
            distance_pct   = (current_price - sr_price) / current_price
            # 猎杀区：支撑下方 0.3~1.5×ATR（止损单密集带）
            hunt_target       = round(sr_price - 0.8  * atr_val, 2)
            hunt_zone_high    = round(sr_price - 0.3  * atr_val, 2)
            hunt_zone_low     = round(sr_price - 1.5  * atr_val, 2)
            # 推荐入场：止损潮尾部（猎杀区低端略上方）反转概率最高
            recommended_entry = round(hunt_zone_low   + 0.25 * atr_val, 2)
            stop_loss         = round(hunt_zone_low   - 0.5  * atr_val, 2)
            expected_gain_pct = round(
                (sr_price - recommended_entry) / recommended_entry * 100 + 2.0, 1)
            hunt_type         = "long_stop_hunt"
            direction_label   = "多头止损猎杀 ↓"
        else:
            distance_pct   = (sr_price - current_price) / current_price
            # 猎杀区：阻力上方 0.3~1.5×ATR（空头止损密集带）
            hunt_target       = round(sr_price + 0.8  * atr_val, 2)
            hunt_zone_low     = round(sr_price + 0.3  * atr_val, 2)
            hunt_zone_high    = round(sr_price + 1.5  * atr_val, 2)
            recommended_entry = round(hunt_zone_high  - 0.25 * atr_val, 2)
            stop_loss         = round(hunt_zone_high  + 0.5  * atr_val, 2)
            expected_gain_pct = round(
                (recommended_entry - sr_price) / sr_price * 100 + 2.0, 1)
            hunt_type         = "short_stop_hunt"
            direction_label   = "空头轧仓猎杀 ↑"

        # 距离衰减：指数函数，ATR 的 2 倍距离时衰减到 1/e ≈ 0.37
        distance_decay    = math.exp(-distance_pct / max(atr_pct * 2, 0.001))
        magnetic_strength = round(min(10.0, sr_weight * distance_decay * oi_factor), 2)

        if   magnetic_strength >= 6.0: prob = "高"
        elif magnetic_strength >= 3.5: prob = "中"
        else:                          prob = "低"

        # 近距离猎杀警报：价格已在 S/R 的 1×ATR 以内（即将触发）
        near_hunt = distance_pct < atr_pct

        if is_long_hunt:
            if near_hunt:
                warning = (f"⚠️ 价格距支撑 ${sr_price:,.0f} 仅 {distance_pct*100:.1f}%，"
                           f"止损猎杀即将触发！勿在支撑位接多")
                action  = (f"等待价格插穿 ${sr_price:,.0f} 止损潮结束后，"
                           f"在 ${recommended_entry:,.0f} 附近挂限价买单（止损 ${stop_loss:,.0f}）")
            else:
                warning = (f"${sr_price:,.0f} 支撑下方聚集多头止损单，"
                           f"跌破后将加速至 ${hunt_target:,.0f} 附近完成止损猎杀后反转")
                action  = (f"勿在 ${sr_price:,.0f} 追多；"
                           f"在 ${recommended_entry:,.0f} 预设限价买单，止损 ${stop_loss:,.0f}，"
                           f"预期反弹 +{expected_gain_pct:.1f}%")
        else:
            if near_hunt:
                warning = (f"⚠️ 价格距阻力 ${sr_price:,.0f} 仅 {distance_pct*100:.1f}%，"
                           f"空头轧仓即将触发！勿在阻力位追空")
                action  = (f"等待价格刺穿 ${sr_price:,.0f} 轧仓完成后，"
                           f"在 ${recommended_entry:,.0f} 附近挂限价卖单（止损 ${stop_loss:,.0f}）")
            else:
                warning = (f"${sr_price:,.0f} 阻力上方聚集空头止损单，"
                           f"突破后将加速至 ${hunt_target:,.0f} 附近完成轧空后反转")
                action  = (f"勿在 ${sr_price:,.0f} 追空；"
                           f"在 ${recommended_entry:,.0f} 预设限价卖单，止损 ${stop_loss:,.0f}，"
                           f"预期下跌 -{expected_gain_pct:.1f}%")

        return {
            "type":               hunt_type,
            "direction_label":    direction_label,
            "sr_level":           round(sr_price,         2),
            "sr_weight":          round(sr_weight,         2),
            "hunt_zone_low":      hunt_zone_low,
            "hunt_zone_high":     hunt_zone_high,
            "hunt_target":        hunt_target,
            "magnetic_strength":  magnetic_strength,
            "hunt_probability":   prob,
            "recommended_entry":  recommended_entry,
            "stop_loss":          stop_loss,
            "expected_gain_pct":  expected_gain_pct,
            "near_hunt":          near_hunt,
            "distance_pct":       round(distance_pct * 100, 2),
            "warning":            warning,
            "action":             action,
        }

    # 支撑位 → 多头止损猎杀区
    for sup in supports[:5]:
        sup_price  = sup[0] if isinstance(sup, (list, tuple)) else sup.get("price", 0)
        sup_weight = sup[1] if isinstance(sup, (list, tuple)) else sup.get("score", 1.0)
        if not sup_price or sup_price >= current_price:
            continue
        zones.append(_build_zone(sup_price, sup_weight, is_long_hunt=True))

    # 阻力位 → 空头轧仓猎杀区
    for res in resistances[:5]:
        res_price  = res[0] if isinstance(res, (list, tuple)) else res.get("price", 0)
        res_weight = res[1] if isinstance(res, (list, tuple)) else res.get("score", 1.0)
        if not res_price or res_price <= current_price:
            continue
        zones.append(_build_zone(res_price, res_weight, is_long_hunt=False))

    # 按磁力强度降序
    zones.sort(key=lambda x: x["magnetic_strength"], reverse=True)
    return zones


# ─────────────────────────────────────────────
# D. Fibonacci 展开
# ─────────────────────────────────────────────

def calculate_fibonacci(df: pd.DataFrame, lookback: int = 90):
    recent = df.tail(lookback)
    high   = recent["high"].max()
    low    = recent["low"].min()
    diff   = high - low
    # 回调位（0% = 摆动高点, 100% = 摆动低点）
    levels = {f"Fib {r:.1%}": high - diff * r
              for r in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}
    # 延伸位（向上超越高点 = 获利目标；向下超越低点 = 做空目标）
    # 上方延伸：low + diff * r  （专业交易者常用获利位）
    for r in [1.272, 1.414, 1.618, 2.0, 2.618]:
        pct = r * 100
        levels[f"Ext {pct:.1f}%"] = low + diff * r
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
    for level in supports:
        price = level[0]
        fig.add_shape(type="line", x0=x0, x1=x1, y0=price, y1=price,
                      line=dict(color="rgba(38,166,154,0.7)", width=1.5, dash="dot"), row=1, col=1)
        fig.add_annotation(x=x1, y=price, text=f"  支撑 ${price:,.0f}",
                           showarrow=False, font=dict(color="#26a69a", size=10),
                           xanchor="left", row=1, col=1)
    for level in resistances:
        price = level[0]
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
                    fib_levels, symbol: str = "BTC/USDT", tf_label: str = "", verbose: bool = True,
                    liq_clusters: list = None, df_backtest: pd.DataFrame = None,
                    market_structure: dict = None, fvg_zones: list = None,
                    volume_profile: dict = None):
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
    ma_group_raw = 0  # P0-C: 趋势组原始分，后续统一限幅 ±2
    for name, val, desc in ma_items:
        if pd.isna(val):
            continue
        diff_pct = (price - val) / val * 100
        is_above = price > val
        # 震荡市禁用 MA 突破评分（防假突破）
        ma_active = (regime != "ranging")
        if ma_active:
            ma_group_raw += 1 if is_above else -1   # 累积而非直接加总分
        regime_note = "" if ma_active else " [震荡市-暂停]"
        emit(f"  {name:<6}({desc}): ${val:>10,.2f}  {'上方 ↑' if is_above else '下方 ↓'} {diff_pct:+.2f}%  [{'+'if is_above else'-'}]{regime_note}")
        rd["ma"].append({"name": name, "desc": desc, "val": val,
                         "diff_pct": diff_pct, "above": is_above, "signal": "+" if is_above else "-",
                         "active": ma_active})
    # 动态权重：趋势市MA更可靠，震荡市MA信号噪音大
    # trending: MA上限±3（强趋势信号更有意义）；ranging: 限压±1（防假突破）
    _ma_cap = 3 if regime == "trending" else (1 if regime == "ranging" else 2)
    ma_group_score = max(-_ma_cap, min(_ma_cap, ma_group_raw))
    score += ma_group_score
    emit(f"  [趋势组 {regime}] 原始分 {ma_group_raw:+d} → 限幅后 {ma_group_score:+d}/±{_ma_cap}")
    rd["ma_group_score"] = ma_group_score

    # ── 动量组（RSI + MACD + KDJ）——限幅 ±3 ──────────────────
    momentum_group_raw = 0   # P0-C: 动量组原始分

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
    momentum_group_raw += rsi_score   # 累积到动量组
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
    # 震荡市BB均值回归信号更可靠，权重×1.5；趋势市BB只提供方向辅助，权重×0.5
    if regime == "trending":
        bb_score = round(bb_score * 0.5)   # 趋势市半权：±0或±1→0（边界信号仍保留）
    elif regime == "ranging":
        bb_score = round(bb_score * 1.5)   # 震荡市加权：±1→±1(ceil), ±2不会出现
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
    momentum_group_raw += macd_score   # 累积到动量组
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
    momentum_group_raw += kdj_score   # 累积到动量组
    emit(f"  K: {k_val:.2f}  D: {d_val:.2f}  J: {j_val:.2f}")
    emit(f"  信号: {kdj_signal}")
    rd["kdj"] = {"k": k_val, "d": d_val, "j": j_val, "signal": kdj_signal, "score": kdj_score}

    # 震荡市动量指标更准（RSI/KDJ均值回归），上限±4；趋势市限±2（防逆势）
    _mom_cap = 2 if regime == "trending" else (4 if regime == "ranging" else 3)
    momentum_group_score = max(-_mom_cap, min(_mom_cap, momentum_group_raw))
    score += momentum_group_score
    emit(f"\n  [动量组 RSI+MACD+KDJ {regime}] 原始分 {momentum_group_raw:+d} → 限幅后 {momentum_group_score:+d}/±{_mom_cap}")
    rd["momentum_group_score"] = momentum_group_score

    # ── 成交量组（OBV + CVD）——限幅 ±2 ──────────────────────
    volume_group_raw = 0   # P0-C: 成交量组原始分

    # ── OBV ──
    obv_now, obv_prev = latest["obv"], df["obv"].iloc[-20]
    emit(f"\n【OBV 能量潮】{sep}")
    obv_up    = obv_now > obv_prev
    obv_trend = "上升 -> 资金流入，价格上涨得到支撑" if obv_up else "下降 -> 资金流出，价格下跌压力"
    obv_score = 1 if obv_up else -1
    volume_group_raw += obv_score   # 累积到成交量组
    emit(f"  当前OBV: {obv_now:,.0f}  20根前: {obv_prev:,.0f}")
    emit(f"  趋势: {obv_trend}")
    rd["obv"] = {"now": obv_now, "prev": obv_prev, "trend": obv_trend,
                 "up": obv_up, "score": obv_score}

    # ── RSI 背离 + OBV 背离检测 ─────────────────────────────────
    emit(f"\n【背离检测 RSI & OBV Divergence】{sep}")
    div_score  = 0
    div_signals = []
    _DIV_WINDOW = 40   # 回溯窗口根数
    if len(df) >= _DIV_WINDOW + 5 and "rsi" in df.columns:
        _seg   = df.tail(_DIV_WINDOW).reset_index(drop=True)
        _close = _seg["close"].values
        _rsi   = _seg["rsi"].values
        _obv   = _seg["obv"].values
        _n     = len(_seg)

        def _find_pivot_highs(arr, radius=4):
            idx = []
            for i in range(radius, _n - radius):
                if arr[i] == max(arr[i-radius:i+radius+1]):
                    idx.append(i)
            return idx

        def _find_pivot_lows(arr, radius=4):
            idx = []
            for i in range(radius, _n - radius):
                if arr[i] == min(arr[i-radius:i+radius+1]):
                    idx.append(i)
            return idx

        ph_price = _find_pivot_highs(_close)
        pl_price = _find_pivot_lows(_close)
        ph_rsi   = _find_pivot_highs(_rsi)
        pl_rsi   = _find_pivot_lows(_rsi)
        ph_obv   = _find_pivot_highs(_obv)
        pl_obv   = _find_pivot_lows(_obv)

        # RSI 顶背离：价格新高但 RSI 低高
        if len(ph_price) >= 2 and len(ph_rsi) >= 2:
            pp1, pp2 = ph_price[-2], ph_price[-1]  # 先高、后高
            rp1 = min(ph_rsi, key=lambda x: abs(x - pp1))
            rp2 = min(ph_rsi, key=lambda x: abs(x - pp2))
            if (_close[pp2] > _close[pp1] * 1.002 and
                    _rsi[rp2] < _rsi[rp1] - 2 and pp2 > pp1):
                div_score -= 2
                div_signals.append("RSI顶背离 ⚠️ 价格↑新高 但RSI↓低高 → 上涨动能衰竭，警惕反转")
        # RSI 底背离：价格新低但 RSI 高低
        if len(pl_price) >= 2 and len(pl_rsi) >= 2:
            pp1, pp2 = pl_price[-2], pl_price[-1]
            rp1 = min(pl_rsi, key=lambda x: abs(x - pp1))
            rp2 = min(pl_rsi, key=lambda x: abs(x - pp2))
            if (_close[pp2] < _close[pp1] * 0.998 and
                    _rsi[rp2] > _rsi[rp1] + 2 and pp2 > pp1):
                div_score += 2
                div_signals.append("RSI底背离 ✅ 价格↓新低 但RSI↑高低 → 下跌动能衰竭，关注见底")
        # OBV 顶背离：价格新高但 OBV 低高
        if len(ph_price) >= 2 and len(ph_obv) >= 2:
            pp1, pp2 = ph_price[-2], ph_price[-1]
            op1 = min(ph_obv, key=lambda x: abs(x - pp1))
            op2 = min(ph_obv, key=lambda x: abs(x - pp2))
            if (_close[pp2] > _close[pp1] * 1.002 and
                    _obv[op2] < _obv[op1] * 0.998 and pp2 > pp1):
                div_score -= 1
                div_signals.append("OBV顶背离 ⚠️ 价格↑新高 但OBV↓ → 量价背离，资金未跟进")
        # OBV 底背离：价格新低但 OBV 高低
        if len(pl_price) >= 2 and len(pl_obv) >= 2:
            pp1, pp2 = pl_price[-2], pl_price[-1]
            op1 = min(pl_obv, key=lambda x: abs(x - pp1))
            op2 = min(pl_obv, key=lambda x: abs(x - pp2))
            if (_close[pp2] < _close[pp1] * 0.998 and
                    _obv[op2] > _obv[op1] * 1.002 and pp2 > pp1):
                div_score += 1
                div_signals.append("OBV底背离 ✅ 价格↓新低 但OBV↑ → 量价背离，资金暗中吸筹")

    div_score = max(-3, min(3, div_score))   # 限幅 ±3
    volume_group_raw += div_score            # 并入成交量组
    if div_signals:
        for s in div_signals:
            emit(f"  {s}")
    else:
        emit("  未检测到明显背离")
    emit(f"  背离得分: {div_score:+d}")
    rd["divergence"] = {"score": div_score, "signals": div_signals}

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

        volume_group_raw += cvd_score   # 累积到成交量组
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

    # 成交量组限幅 ±2：OBV+CVD 最多贡献 ±2
    volume_group_score = max(-2, min(2, volume_group_raw))
    score += volume_group_score
    emit(f"\n  [成交量组 OBV+CVD] 原始分 {volume_group_raw:+d} → 限幅后 {volume_group_score:+d}/±2")
    rd["volume_group_score"] = volume_group_score

    # ── 支撑/阻力 ──
    emit(f"\n【支撑 / 阻力位】{sep}")
    emit("  阻力位 (从近到远):")
    res_list = []
    for level in sorted(resistances, key=lambda x: x[0], reverse=True):
        pr       = level[0]
        sr_score = level[1]
        src      = level[2] if len(level) > 2 else "pivot"
        reason   = level[3] if len(level) > 3 else src
        dist     = (pr - price) / price * 100
        confidence = min(5.0, sr_score)
        emit(f"    ${pr:>10,.2f}  距当前 +{dist:.2f}%  强度:{confidence:.1f} ({sr_score:.1f}) [{src}]")
        emit(f"      └─ {reason}")
        res_list.append({"price": pr, "dist_pct": dist, "score": sr_score,
                         "stars_val": confidence, "source": src, "reason": reason})

    emit("  支撑位 (从近到远):")
    sup_list = []
    for level in sorted(supports, key=lambda x: x[0], reverse=True):
        ps       = level[0]
        sr_score = level[1]
        src      = level[2] if len(level) > 2 else "pivot"
        reason   = level[3] if len(level) > 3 else src
        dist     = (price - ps) / price * 100
        confidence = min(5.0, sr_score)
        emit(f"    ${ps:>10,.2f}  距当前 -{dist:.2f}%  强度:{confidence:.1f} ({sr_score:.1f}) [{src}]")
        emit(f"      └─ {reason}")
        sup_list.append({"price": ps, "dist_pct": dist, "score": sr_score,
                         "stars_val": confidence, "source": src, "reason": reason})

    rd["resistances"] = res_list
    rd["supports"]    = sup_list

    # ── K线形态识别 ─────────────────────────────────────────────────────────────
    emit(f"\n【K线形态识别 · Candlestick Patterns】{sep}")
    candle_patterns = []
    candle_score    = 0

    if len(df) >= 4:
        c0 = df.iloc[-1]   # 当前K线
        c1 = df.iloc[-2]   # 前1根
        c2 = df.iloc[-3]   # 前2根

        def _body(c):       return abs(float(c["close"]) - float(c["open"]))
        def _rng(c):        return max(1e-9, float(c["high"]) - float(c["low"]))
        def _upper_wick(c): return float(c["high"]) - max(float(c["open"]), float(c["close"]))
        def _lower_wick(c): return min(float(c["open"]), float(c["close"])) - float(c["low"])
        def _is_bull(c):    return float(c["close"]) >= float(c["open"])

        b0, r0 = _body(c0), _rng(c0)
        uw0, lw0 = _upper_wick(c0), _lower_wick(c0)
        b1, r1 = _body(c1), _rng(c1)

        # 是否处于 S/R 关键位附近（1.5%范围内）
        _sup_prices = [s[0] for s in (supports or []) if s[0] < price]
        _res_prices = [r[0] for r in (resistances or []) if r[0] > price]
        _at_sup = any(abs(price - sp) / price < 0.015 for sp in _sup_prices)
        _at_res = any(abs(price - rp) / price < 0.015 for rp in _res_prices)

        # ── 1. Pin Bar ──────────────────────────────────────────────────────────
        if r0 > price * 0.001 and b0 > 0:
            if lw0 >= 2.5 * b0 and lw0 >= 0.60 * r0:
                # 看涨Pin Bar：长下影线拒绝低价（锤子线）
                adj = +3 if _at_sup else +2
                candle_score += adj
                candle_patterns.append({
                    "name": "看涨Pin Bar",
                    "icon": "🔨",
                    "type": "bullish",
                    "score": adj,
                    "desc": f"长下影线拒绝低价 (下影{lw0/r0*100:.0f}% 实体{b0/r0*100:.0f}%)" +
                            (" + 支撑位共振" if _at_sup else ""),
                })
            elif uw0 >= 2.5 * b0 and uw0 >= 0.60 * r0:
                # 看跌Pin Bar：长上影线拒绝高价（射击之星）
                adj = -3 if _at_res else -2
                candle_score += adj
                candle_patterns.append({
                    "name": "看跌Pin Bar",
                    "icon": "⭐",
                    "type": "bearish",
                    "score": adj,
                    "desc": f"长上影线拒绝高价 (上影{uw0/r0*100:.0f}% 实体{b0/r0*100:.0f}%)" +
                            (" + 阻力位共振" if _at_res else ""),
                })

        # ── 2. 吞噬形态（Engulfing）────────────────────────────────────────────
        if (not _is_bull(c1) and _is_bull(c0) and
                float(c0["open"]) <= float(c1["close"]) and
                float(c0["close"]) >= float(c1["open"]) and
                b0 >= b1 * 0.9):
            # 看涨吞噬：前阴后阳，阳线完全包裹阴线实体
            adj = +3 if _at_sup else +2
            candle_score += adj
            candle_patterns.append({
                "name": "看涨吞噬",
                "icon": "🟢",
                "type": "bullish",
                "score": adj,
                "desc": f"阳线吞噬前阴线 (实体比 {b0/max(b1, 1e-9):.1f}×)" +
                        (" + 支撑位共振" if _at_sup else ""),
            })
        elif (_is_bull(c1) and not _is_bull(c0) and
                float(c0["open"]) >= float(c1["close"]) and
                float(c0["close"]) <= float(c1["open"]) and
                b0 >= b1 * 0.9):
            # 看跌吞噬：前阳后阴，阴线完全包裹阳线实体
            adj = -3 if _at_res else -2
            candle_score += adj
            candle_patterns.append({
                "name": "看跌吞噬",
                "icon": "🔴",
                "type": "bearish",
                "score": adj,
                "desc": f"阴线吞噬前阳线 (实体比 {b0/max(b1, 1e-9):.1f}×)" +
                        (" + 阻力位共振" if _at_res else ""),
            })

        # ── 3. Inside Bar（孕线/内包线）或 Doji（十字星）──────────────────────
        elif (float(c0["high"]) < float(c1["high"]) and
                float(c0["low"]) > float(c1["low"]) and
                r1 > 0 and r0 / r1 < 0.75):
            dir_hint = "多头" if score > 0 else ("空头" if score < 0 else "中性")
            candle_patterns.append({
                "name": "Inside Bar",
                "icon": "📦",
                "type": "neutral",
                "score": 0,
                "desc": f"价格压缩蓄力 (内包比{r0/r1*100:.0f}%)，当前倾向{dir_hint}突破，跟随突破方向入场",
            })
        elif r0 > price * 0.002 and b0 / r0 < 0.10:
            if   lw0 > uw0 * 3: doji_type = "蜻蜓十字"
            elif uw0 > lw0 * 3: doji_type = "墓碑十字"
            else:               doji_type = "标准十字"
            at_any = _at_sup or _at_res
            candle_patterns.append({
                "name": "十字星",
                "icon": "✚",
                "type": "neutral",
                "score": 0,
                "desc": f"{doji_type}，多空均衡等待方向选择" +
                        (" ⚠️ 关键位置出现，警惕反转" if at_any else ""),
            })

        # ── 4. 大实体 Marubozu（光头光脚）──────────────────────────────────────
        if r0 > price * 0.003 and b0 / r0 >= 0.80:
            if _is_bull(c0):
                candle_score += 1
                candle_patterns.append({
                    "name": "大阳实体",
                    "icon": "⬆️",
                    "type": "bullish",
                    "score": +1,
                    "desc": f"强势买盘主导 (实体{b0/r0*100:.0f}%)，多头强势",
                })
            else:
                candle_score -= 1
                candle_patterns.append({
                    "name": "大阴实体",
                    "icon": "⬇️",
                    "type": "bearish",
                    "score": -1,
                    "desc": f"强势卖盘主导 (实体{b0/r0*100:.0f}%)，空头强势",
                })

        # ── 5. 三白兵 / 三黑鸦 ─────────────────────────────────────────────────
        b2, r2 = _body(c2), _rng(c2)
        if (all(_is_bull(c) for c in [c0, c1, c2]) and
                float(c0["close"]) > float(c1["close"]) > float(c2["close"]) and
                b0 > r0 * 0.40 and b1 > r1 * 0.40 and b2 > r2 * 0.40):
            candle_score += 1
            candle_patterns.append({
                "name": "三白兵",
                "icon": "☀️",
                "type": "bullish",
                "score": +1,
                "desc": "连续三根实体阳线且高点递升，多头趋势强劲",
            })
        elif (all(not _is_bull(c) for c in [c0, c1, c2]) and
                float(c0["close"]) < float(c1["close"]) < float(c2["close"]) and
                b0 > r0 * 0.40 and b1 > r1 * 0.40 and b2 > r2 * 0.40):
            candle_score -= 1
            candle_patterns.append({
                "name": "三黑鸦",
                "icon": "🌑",
                "type": "bearish",
                "score": -1,
                "desc": "连续三根实体阴线且低点递降，空头趋势强劲",
            })

    candle_score = max(-4, min(4, candle_score))
    score += candle_score
    if candle_patterns:
        for p in candle_patterns:
            score_tag = f"[{p['score']:+d}]" if p["score"] != 0 else "[中性]"
            emit(f"  {p['icon']} {p['name']} {score_tag}: {p['desc']}")
    else:
        emit("  无明显K线形态信号")
    emit(f"  形态综合得分: {candle_score:+d}")
    rd["candle_patterns"] = {"score": candle_score, "patterns": candle_patterns}

    # ── Fibonacci ──
    emit(f"\n【Fibonacci 回调位 & 延伸位】{sep}")
    fib_out = []
    for label, fp in sorted(fib_levels.items(), key=lambda x: x[1], reverse=True):
        dist   = (fp - price) / price * 100
        is_ext = label.startswith("Ext")
        marker = " <- 当前价附近" if abs(dist) < 2 else ""
        type_tag = "[延伸]" if is_ext else "[回调]"
        emit(f"  {type_tag} {label:<12}: ${fp:>10,.2f}  ({dist:+.2f}% {'↑' if fp>price else '↓'}){marker}")
        fib_out.append({"label": label, "price": fp, "dist_pct": dist, "near": abs(dist) < 2, "ext": is_ext})
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

    # ── 清算密集区分析（参与评分）────────────────────────────────────
    liq_score   = 0
    liq_signals = []
    liq_out     = []
    rd["liq"] = {"score": 0, "signals": [], "clusters": [], "source": "none"}

    if liq_clusters:
        atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else 0.0

        # 把清算区分为多头清算区（SELL方向强平）和空头清算区（BUY方向强平）
        long_liq  = sorted([c for c in liq_clusters if "多头" in c.get("dominant", "")],
                           key=lambda x: x["price"], reverse=True)  # 从高到低（近的在前）
        short_liq = sorted([c for c in liq_clusters if "空头" in c.get("dominant", "")],
                           key=lambda x: x["price"])                 # 从低到高（近的在前）

        emit(f"\n【清算密集区分析】{sep}")
        emit(f"  当前价格: ${price:,.2f}  ATR: ${atr_val:,.2f}")

        # ── 场景1：逼近多头清算区（价格下方，距离 < 1.5×ATR）→ 风险警告 -1 ──
        # 多头止损密集在支撑位下方，价格逼近时一旦跌破会引发踩踏
        for lc in long_liq:
            lc_price = lc["price"]
            if lc_price >= price:
                continue  # 跳过价格上方的多头清算区（已穿越）
            dist = price - lc_price
            dist_pct = dist / price * 100
            if atr_val > 0 and dist < 1.5 * atr_val:
                liq_score -= 1
                sig = f"逼近多头清算区 ${lc_price:,.0f} (距离 -{dist_pct:.2f}%, <1.5×ATR) → 踩踏风险，支撑脆弱"
                liq_signals.append({"type": "warn_long_liq", "text": sig, "score": -1,
                                    "price": lc_price, "dist_pct": -dist_pct})
                emit(f"  ⚠️  {sig}")
                break  # 只取最近一个

        # ── 场景2：价格已跌破多头清算区（在其下方）→ 支撑失效 -2 ──
        for lc in long_liq:
            lc_price = lc["price"]
            dist_below = lc_price - price   # 正值 = 价格在清算区下方
            if dist_below > 0 and (atr_val == 0 or dist_below < 2.0 * atr_val):
                liq_score -= 2
                sig = f"已跌破多头清算区 ${lc_price:,.0f} (下方 {dist_below/price*100:.2f}%) → 多头出清，支撑失效"
                liq_signals.append({"type": "long_liq_broken", "text": sig, "score": -2,
                                    "price": lc_price, "dist_pct": dist_below / price * 100})
                emit(f"  🔴  {sig}")
                break

        # ── 场景3：逼近空头清算区（价格上方，距离 < 1.5×ATR）→ 轧空动能 +1 ──
        # 空头止损密集在阻力位上方，价格逼近时一旦突破会引发轧空加速上涨
        for lc in short_liq:
            lc_price = lc["price"]
            if lc_price <= price:
                continue
            dist = lc_price - price
            dist_pct = dist / price * 100
            if atr_val > 0 and dist < 1.5 * atr_val:
                liq_score += 1
                sig = f"逼近空头清算区 ${lc_price:,.0f} (距离 +{dist_pct:.2f}%, <1.5×ATR) → 轧空动能，突破可加速"
                liq_signals.append({"type": "short_liq_squeeze", "text": sig, "score": +1,
                                    "price": lc_price, "dist_pct": dist_pct})
                emit(f"  🟢  {sig}")
                break

        # ── 场景4：价格已突破空头清算区（在其上方）→ 突破可信 +2 ──
        for lc in reversed(short_liq):
            lc_price = lc["price"]
            dist_above = price - lc_price  # 正值 = 价格在清算区上方
            if dist_above > 0 and (atr_val == 0 or dist_above < 2.0 * atr_val):
                liq_score += 2
                sig = f"已突破空头清算区 ${lc_price:,.0f} (上方 {dist_above/price*100:.2f}%) → 空头出清，突破可信"
                liq_signals.append({"type": "short_liq_broken", "text": sig, "score": +2,
                                    "price": lc_price, "dist_pct": dist_above / price * 100})
                emit(f"  ✅  {sig}")
                break

        # ── 场景5：止损位建议（结合最近清算区，优化 ATR 止损线）──────────
        # 多头止损：不应设在多头清算密集区内部（会被踩踏带走），应设在其下方
        best_long_sl = None
        for lc in long_liq:
            lc_price = lc["price"]
            if lc_price < price:
                candidate = lc_price - (atr_val * 0.3 if atr_val else lc_price * 0.003)
                best_long_sl = round(candidate, 2)
                sig = f"多头优化止损: ${best_long_sl:,.2f} (清算区 ${lc_price:,.0f} 下方 0.3×ATR)"
                liq_signals.append({"type": "sl_long_opt", "text": sig, "score": 0,
                                    "price": best_long_sl, "dist_pct": (price - best_long_sl) / price * 100})
                emit(f"  📌  {sig}")
                break

        best_short_sl = None
        for lc in short_liq:
            lc_price = lc["price"]
            if lc_price > price:
                candidate = lc_price + (atr_val * 0.3 if atr_val else lc_price * 0.003)
                best_short_sl = round(candidate, 2)
                sig = f"空头优化止损: ${best_short_sl:,.2f} (清算区 ${lc_price:,.0f} 上方 0.3×ATR)"
                liq_signals.append({"type": "sl_short_opt", "text": sig, "score": 0,
                                    "price": best_short_sl, "dist_pct": (best_short_sl - price) / price * 100})
                emit(f"  📌  {sig}")
                break

        if not liq_signals:
            emit("  当前价格远离所有清算密集区，无额外信号")

        # liq_score 限幅：单模块最多 ±2，避免压制其他指标
        liq_score = max(-2, min(2, liq_score))
        score += liq_score

        # ── 止损猎杀区：修正 liq 止损建议（使用更精确的猎杀区止损价）──
        # 覆盖前面基于 0.3×ATR 的粗略止损，用猎杀区低端更精确地定位
        if best_long_sl is not None:
            for lc in long_liq:
                lc_price = lc["price"]
                if lc_price < price:
                    hunt_low = lc_price - 1.5 * atr_val
                    if hunt_low < best_long_sl:
                        best_long_sl = round(hunt_low, 2)
                    break

        liq_out = [{
            "price":    lc["price"],
            "qty":      lc.get("qty", 0),
            "count":    lc.get("count", 0),
            "long_qty": lc.get("long_qty", 0),
            "short_qty":lc.get("short_qty", 0),
            "dominant": lc.get("dominant", ""),
            "source":   lc.get("source", ""),
        } for lc in liq_clusters]

        rd["liq"] = {
            "score":    liq_score,
            "signals":  liq_signals,
            "clusters": liq_out,
            "source":   liq_clusters[0].get("source", "unknown") if liq_clusters else "none",
            "sl_long_opt":  best_long_sl,
            "sl_short_opt": best_short_sl,
        }
        emit(f"  清算区综合评分: {liq_score:+d}")

    # ── 市场结构分析（Market Structure · BOS / CHoCH）────────────────────
    ms_score = 0
    emit(f"\n【市场结构分析 · BOS / CHoCH】{sep}")
    if market_structure and market_structure.get("structure") not in ("unknown", None):
        ms = market_structure
        st_sc  = ms.get("structure_score", 0)
        st_cn  = ms.get("structure_cn",    "未知")
        bias   = ms.get("trading_bias",    "")
        events = ms.get("events",          [])
        kh     = ms.get("key_high",        {})
        kl     = ms.get("key_low",         {})
        ms_score = st_sc
        score   += ms_score

        emit(f"  结构: {st_cn}  (得分 {st_sc:+d})")
        emit(f"  操作偏向: {bias}")
        if kh:
            emit(f"  关键摆动高点: ${kh.get('price',0):,.2f}  ({kh.get('dist_pct',0):+.2f}%)")
        if kl:
            emit(f"  关键摆动低点: ${kl.get('price',0):,.2f}  (-{abs(kl.get('dist_pct',0)):.2f}%)")
        if events:
            emit(f"  近期结构事件（最近3个）:")
            for ev in events[-3:]:
                icon = "✅" if ev.get("direction") == "bullish" else "🔴"
                emit(f"    {icon} {ev.get('cn','')}  @${ev.get('price',0):,.2f}  {ev.get('date','')[:13]}")
    else:
        emit("  市场结构数据不足或未提供")

    rd["market_structure"] = market_structure or {}

    # ── Volume Profile 快照 ────────────────────────────────────────────
    emit(f"\n【Volume Profile · POC / VAH / VAL】{sep}")
    if volume_profile and volume_profile.get("poc"):
        vp = volume_profile
        poc = vp.get("poc", 0);  vah = vp.get("vah", 0);  val = vp.get("val", 0)
        emit(f"  POC (成交量最大价位): ${poc:,.2f}  {'↑ 价格上方' if vp.get('poc_above') else '↓ 价格下方'}")
        emit(f"  VAH (价值区间上沿):   ${vah:,.2f}")
        emit(f"  VAL (价值区间下沿):   ${val:,.2f}")
        emit(f"  价值区间宽度: {vp.get('va_width_pct',0):.1f}%  类型: {vp.get('dist_type','')}")
        if vp.get("hvn"):
            hvn_str = "  |  ".join(f"${h['price']:,.0f}({h['vol_pct']:.1f}%)" for h in vp["hvn"][:3])
            emit(f"  HVN 高成交量节点: {hvn_str}")
    else:
        emit("  Volume Profile 数据不足")

    rd["volume_profile"] = volume_profile or {}

    # ── Fair Value Gap 分析（FVG · 价格失衡区）────────────────────────
    fvg_score = 0
    emit(f"\n【Fair Value Gap · 价格失衡区】{sep}")
    _atr_for_fvg = float(df["atr"].iloc[-1]) \
        if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else 0.0
    if fvg_zones:
        emit(f"  发现 {len(fvg_zones)} 个未填充/部分填充 FVG（按距当前价排序）:")
        for fg in fvg_zones[:5]:
            icon = "🟢" if fg["type"] == "bullish_fvg" else "🔴"
            dist_pct = (fg["mid"] - price) / price * 100
            emit(f"  {icon} {fg['desc']}")
            emit(f"     中点 ${fg['mid']:,.2f}  距当前 {dist_pct:+.2f}%  已填充 {fg['filled_pct']:.0f}%")
        # FVG 评分：最近的 FVG 中点 < 1.5×ATR → 对方向评分 +/-1
        nearest = fvg_zones[0]
        near_dist = abs(nearest["mid"] - price)
        fvg_threshold = _atr_for_fvg * 1.5 if _atr_for_fvg else price * 0.015
        if near_dist < fvg_threshold:
            if nearest["type"] == "bullish_fvg" and nearest["mid"] < price:
                fvg_score = +1
                emit(f"  ✅ 近距看涨FVG支撑 (${nearest['mid']:,.2f})，回调后磁吸反弹概率高 → +1")
            elif nearest["type"] == "bearish_fvg" and nearest["mid"] > price:
                fvg_score = -1
                emit(f"  ⚠️ 近距看跌FVG压制 (${nearest['mid']:,.2f})，反弹至此遭阻力概率高 → -1")
        score += fvg_score
    else:
        emit("  当前价格范围内未发现显著 FVG")

    rd["fvg_zones"]  = fvg_zones or []
    rd["fvg_score"]  = fvg_score
    rd["ms_score"]   = ms_score

    # ── 止损猎杀区分析（磁吸效应模型，不单独计分，但输出操作建议）──────────
    emit(f"\n【流动性猎杀地图 · 磁吸效应】{sep}")
    _atr_for_hunt = float(df["atr"].iloc[-1]) \
        if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else 0.0
    stop_hunt_zones = estimate_stop_hunt_zones(
        supports, resistances, current_price=price, atr_val=_atr_for_hunt
    )

    hunt_score_adj = 0   # 当价格临近高磁力区时，对原有评分做小幅修正
    near_high_zones = [z for z in stop_hunt_zones
                       if z["near_hunt"] and z["hunt_probability"] == "高"]

    if stop_hunt_zones:
        emit(f"  机构视角：以下价位是止损单聚集地，价格受磁吸效应会优先吸收这些流动性再反转")
        for z in stop_hunt_zones[:4]:
            prob_icon = "🔴" if z["hunt_probability"] == "高" else ("🟡" if z["hunt_probability"] == "中" else "⚪")
            emit(f"  {prob_icon} [{z['direction_label']}] 关键位 ${z['sr_level']:,.0f} "
                 f"→ 猎杀目标 ${z['hunt_target']:,.0f}  磁力 {z['magnetic_strength']:.1f}/10  概率{z['hunt_probability']}")
            emit(f"     └─ {z['action']}")
        if near_high_zones:
            emit(f"\n  ⚠️⚠️  高磁力猎杀警报（{len(near_high_zones)} 个高概率区临近价格）")
            for z in near_high_zones:
                emit(f"     → {z['warning']}")
            # 临近高磁力猎杀区：当前"支撑/阻力"信号置信度打折（-1 修正）
            hunt_score_adj = -1 if any(z["type"] == "long_stop_hunt" for z in near_high_zones) \
                             else  1 if any(z["type"] == "short_stop_hunt" for z in near_high_zones) \
                             else 0
            score += hunt_score_adj
            emit(f"     评分修正: {hunt_score_adj:+d}（猎杀风险已纳入综合评分）")
    else:
        emit("  无有效 S/R 数据，跳过猎杀区估算")
        near_high_zones = []

    rd["stop_hunt_zones"] = stop_hunt_zones
    rd["hunt_score_adj"]  = hunt_score_adj
    emit(f"  共 {len(stop_hunt_zones)} 个猎杀区，高磁力 {sum(1 for z in stop_hunt_zones if z['hunt_probability']=='高')} 个")

    # ── 中间汇总（清算区评分后，回测前）─────────────────
    # 基础分：趋势组±3 + BB±2 + 动量组±4 + 成交量组±2 + 清算区±2 + 市场结构±2 + FVG±1 + 猎杀±1 + K线形态±4 = 21
    max_score = 21   # 回测后 +2，链上 +3
    emit(f"\n  [中间得分] {score:+d}/{max_score}（含清算区+市场结构+FVG，待回测修正）")

    # ── 链上真假突破过滤器 (On-Chain Fakeout Filter) ──────────────
    if onchain_monitor:
        emit(f"\n【链上流动性与真假突破预警 (On-Chain Analysis)】{sep}")
        # 简单判断当前技术面是否有突破迹象 (可根据当前 price 和关键阻力/支撑判断)
        tech_signal = "neutral"
        if resistances and len(resistances) > 0 and price > resistances[0][0] * 0.995:
            tech_signal = "breakout_up"  # 价格非常接近或已突破第一阻力位
        elif supports and len(supports) > 0 and price < supports[0][0] * 1.005:
            tech_signal = "breakdown_down" # 价格非常接近或已跌破第一支撑位
            
        onchain_res = onchain_monitor.detect_fakeout(tech_signal)
        
        # 稳定币宏观数据展示
        sc_data = onchain_monitor.get_stablecoin_flows()
        if "error" not in sc_data:
            emit(f"  宏观资金面: {sc_data.get('macro_liquidity')} (USDT: ${sc_data.get('usdt_mcap',0)/1e9:.1f}B)")
            emit(f"  {sc_data.get('desc')}")
        
        # 突破验证结论
        emit(f"  突破验证: {onchain_res.get('desc')}")
        
        # 链上共振分计入总分 (最高 ±3 分)
        onchain_score = onchain_res.get("score", 0)
        score += onchain_score
        max_score += 3
        
        rd["onchain"] = {
            "tech_signal_detected": tech_signal,
            "status": onchain_res.get("status"),
            "desc": onchain_res.get("desc"),
            "score": onchain_score,
            "stablecoin": sc_data
        }
    else:
        rd["onchain"] = None

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

    # ── 即时回测（历史相似形态验证）────────────────────────
    emit(f"\n【历史形态回测验证】{sep}")
    try:
        # 优先使用扩展历史数据（更多样本），回退到主 df
        _df_bt = df_backtest if (df_backtest is not None and len(df_backtest) > len(df)) else df
        current_sig = _extract_signal_state(_df_bt, len(_df_bt) - 1)
        # hold_bars 根据时间框架自适应
        hb_map = {"15分钟": 10, "30分钟": 8, "1小时": 6, "2小时": 5, "4小时": 4, "8小时": 3}
        hb = hb_map.get(tf_label, 4)
        bt = run_backtest(_df_bt, current_sig, hold_bars=hb,
                          atr_sl_mult=1.5, atr_tp_mult=3.0,
                          min_samples=20, similarity_thresh=0.62)

        score += bt["score"]
        max_score += 2   # 回测模块最多 ±2 分，分母同步扩大

        if bt["win_rate"] is not None:
            desc = bt.get("description", {})
            emit(f"  {desc.get('signal_feature','')}")
            emit(f"  {desc.get('direction_reason','')}")
            emit(f"  样本: {bt['sample_count']} 次 (阈值{bt['similarity_thresh']:.0%})  最优持仓: {bt['hold_bars']} 根K线")
            emit(f"  胜率: {bt['win_rate']:.1%}  盈亏比: {bt['rr_ratio']:.2f}  期望值: {bt['expectancy']:+.3f}%/笔")
            emit(f"  均盈: +{bt['avg_win_pct']:.3f}%  均亏: {bt['avg_loss_pct']:.3f}%")
            emit(f"  {desc.get('exit_distribution','')}")
            if desc.get('consistency'):
                emit(f"  {desc['consistency']}")
            emit(f"  多周期期望: {desc.get('period_summary','')}")
            if desc.get('risk_notes'):
                for rn in desc['risk_notes']:
                    emit(f"  ⚠️ {rn}")
            emit(f"  {desc.get('backtest_result','')}")
            emit(f"  置信度: {bt['confidence_pct']}% ({bt['confidence_level']})  一致性: {bt['consistency']:.0%}")
            emit(f"  回测评分: {bt['score']:+d}")
        else:
            emit(f"  {bt.get('note', '样本不足')}")

        rd["backtest"] = bt
        # 把置信度也融入综合 rd
        rd["confidence_pct"]   = bt["confidence_pct"]
        rd["confidence_level"] = bt["confidence_level"]
    except Exception as e:
        emit(f"  回测异常: {e}")
        rd["backtest"] = None
        rd["confidence_pct"]   = None
        rd["confidence_level"] = "未知"

    # ── 综合信号最终更新（含回测评分后重算 overall）────────
    if score >= 5:
        overall = f"偏多头  (得分: {score:+d}/{max_score})"
        oc      = "#26a69a"
        action  = "做多 / 买入"
    elif score <= -5:
        overall = f"偏空头  (得分: {score:+d}/{max_score})"
        oc      = "#ef5350"
        action  = "做空 / 卖出"
    else:
        overall = f"中性震荡  (得分: {score:+d}/{max_score})"
        oc      = "#f9a825"
        action  = "观望 / 高抛低吸"

    rd.update({"score": score, "max_score": max_score, "overall": overall,
               "overall_color": oc, "action": action,
               "bull_bar": "█"*max(0,score), "bear_bar": "░"*max(0,-score)})

    rd["trade_plan"] = _generate_trade_plan(rd)
    return score, "\n".join(lines), rd


# ─────────────────────────────────────────────
# F-1b. 可执行入场方案生成器
# ─────────────────────────────────────────────

def _generate_trade_plan(rd: dict) -> dict:
    """
    基于 generate_report 输出的 rd 字典，生成可执行交易方案。
    包含：方向判定、入场价位（限价/市价/等待回调）、
         结构型止损（摆动低/高点优先）、TP1/TP2/TP3、R:R 比率、仓位建议。
    """
    score  = rd.get("score", 0) or 0
    price  = rd.get("price", 0) or 0
    max_sc = rd.get("max_score", 14) or 14
    if not price:
        return {"direction": "wait", "direction_cn": "观望 ↔️",
                "color": "#f9a825", "reason": "价格数据异常"}

    # ── 方向判定 ─────────────────────────────────────────────
    if   score >= 5:  direction = "long"
    elif score <= -5: direction = "short"
    else:             direction = "wait"

    if direction == "wait":
        sups = rd.get("supports")  or []
        ress = rd.get("resistances") or []
        res1 = ress[0].get("price") if ress else None
        sup1 = sups[0].get("price") if sups else None
        return {
            "direction": "wait", "direction_cn": "观望 ↔️", "color": "#f9a825",
            "reason": f"综合评分 {score:+d}/{max_sc}，多空信号不明确，建议等待方向确认",
            "condition_long":  f"突破阻力 {'$'+f'{res1:,.0f}' if res1 else '关键阻力位'} 且评分 ≥ +5 → 转多",
            "condition_short": f"跌破支撑 {'$'+f'{sup1:,.0f}' if sup1 else '关键支撑位'} 且评分 ≤ −5 → 转空",
        }

    is_long = (direction == "long")

    # ── 基础数据提取 ──────────────────────────────────────────
    atr_data = rd.get("atr") or {}
    atr_val  = float(atr_data.get("val") or 0)
    atr_pct  = float(atr_data.get("pct") or 0)   # e.g. 1.5 表示 1.5%

    liq_data     = rd.get("liq") or {}
    liq_sl_long  = liq_data.get("sl_long_opt")
    liq_sl_short = liq_data.get("sl_short_opt")

    ms_data  = rd.get("market_structure") or {}
    key_high = (ms_data.get("key_high") or {}).get("price")
    key_low  = (ms_data.get("key_low")  or {}).get("price")

    vp   = rd.get("volume_profile") or {}
    poc  = vp.get("poc")
    vah  = vp.get("vah")
    val_ = vp.get("val")
    # HVN 节点：成交量密集区，做多目标优先于普通S/R
    _hvn_list = [b["price"] for b in (vp.get("chart_bins") or [])
                 if b.get("is_hvn") and b.get("price")]

    hunt_zones = rd.get("stop_hunt_zones") or []

    # 支撑/阻力按距离排序
    sup_sorted = sorted(
        [s for s in (rd.get("supports") or []) if s.get("price", 0) < price],
        key=lambda x: x["price"], reverse=True   # 最近的在前
    )
    res_sorted = sorted(
        [r for r in (rd.get("resistances") or []) if r.get("price", 0) > price],
        key=lambda x: x["price"]                 # 最近的在前
    )

    fib_list  = rd.get("fib") or []
    fib_above = sorted([f for f in fib_list if f.get("price", 0) > price * 1.003],
                       key=lambda x: x["price"])
    fib_below = sorted([f for f in fib_list if f.get("price", 0) < price * 0.997],
                       key=lambda x: x["price"], reverse=True)

    # ATR 安全兜底
    if atr_val <= 0:
        atr_val = price * 0.015
    if atr_pct <= 0:
        atr_pct = atr_val / price * 100

    # ── 止损候选优先级辅助 ────────────────────────────────────
    _prio = {"structural": 0, "liq": 1, "support": 2, "resistance": 2, "atr": 3}

    def _best_sl(candidates):
        """从候选列表中按优先级选出最合理的止损价"""
        valid = [(lb, float(sl), tp) for lb, sl, tp in candidates
                 if sl and price * 0.85 < float(sl) < price * 1.15
                 and ((is_long and float(sl) < price) or (not is_long and float(sl) > price))]
        if not valid:
            return None, None, None
        return min(valid, key=lambda x: _prio.get(x[2], 9))

    # ── 入场区间检测（是否在 S/R 的 1.5×ATR 范围内）─────────
    in_range = lambda p: p and abs(price - p) / price < (atr_pct / 100) * 1.5

    # ══════════════════════════════════════════════════════
    if is_long:
        # ── 止损（结构型优先）────────────────────────────
        sl_cands = []
        if key_low and key_low < price:
            sl_cands.append(("结构型（摆动低点−0.5ATR）", key_low - 0.5 * atr_val, "structural"))
        if liq_sl_long:
            sl_cands.append(("清算区优化止损", liq_sl_long, "liq"))
        if sup_sorted:
            sl_cands.append(("最近支撑−0.5ATR", sup_sorted[0]["price"] - 0.5 * atr_val, "support"))
        sl_cands.append(("ATR×1.5止损", atr_data.get("sl_long") or price - 1.5 * atr_val, "atr"))
        sl_label, sl_price, sl_type = _best_sl(sl_cands)
        if sl_price is None:
            sl_price = round(price - 1.5 * atr_val, 2)
            sl_label, sl_type = "ATR×1.5止损", "atr"
        sl_price = round(sl_price, 2)
        sl_dist_pct = round((price - sl_price) / price * 100, 2)

        # ── 入场价位 ─────────────────────────────────────
        near_hunt = any(z.get("near_hunt") and z.get("type") == "long_stop_hunt"
                        for z in hunt_zones)
        if score >= 8 and not near_hunt:
            entry_method = "市价追入"
            entry_price  = price
            entry_note   = f"动能强劲（评分 {score:+d}），可直接市价入场"
        elif sup_sorted and in_range(sup_sorted[0]["price"]):
            entry_price  = round(sup_sorted[0]["price"] * 1.001, 2)
            entry_method = "限价挂单（支撑区）"
            entry_note   = f"价格已在支撑 ${sup_sorted[0]['price']:,.0f} 附近，可限价建仓"
        elif poc and in_range(poc) and poc < price:
            entry_price  = round(poc, 2)
            entry_method = "限价挂单（VP POC）"
            entry_note   = f"价格接近成交量POC ${poc:,.0f}，价值共识买入区"
        elif sup_sorted:
            entry_price  = round(sup_sorted[0]["price"] + atr_val * 0.1, 2)
            entry_method = "等待回调限价"
            entry_note   = f"建议等回调至支撑 ${sup_sorted[0]['price']:,.0f} 附近再入场"
        elif val_ and val_ < price:
            entry_price  = round(val_, 2)
            entry_method = "等待回调至VA下沿"
            entry_note   = f"等待回调至价值区间下沿 VAL ${val_:,.0f}"
        else:
            entry_price  = price
            entry_method = "市价（无明显支撑）"
            entry_note   = "无明显支撑参考位，谨慎控制仓位"
        entry_price    = round(entry_price, 2)
        entry_dist_pct = round((price - entry_price) / price * 100, 2)

        # ── 止盈目标 ─────────────────────────────────────
        tp_raw = []
        _dedup = lambda p, lst: not any(abs(p - t["price"]) / (p or 1) < 0.006 for t in lst)
        # 1. 优先用VP关键位：VAH > 价格 = 成交量价值区上沿（机构常用TP）
        if vah and vah > entry_price * 1.003 and _dedup(vah, tp_raw):
            tp_raw.append({"price": vah, "label": "VP VAH（价值区上沿）", "badge": "vp_vah"})
        # 2. VP HVN 节点（成交量节点 = 机构成本区，到达后常有阻力）
        for _h in sorted([h for h in _hvn_list if h > entry_price * 1.003]):
            if len(tp_raw) >= 3: break
            if _dedup(_h, tp_raw):
                tp_raw.append({"price": _h, "label": "VP HVN（成交量节点）", "badge": "vp_hvn"})
        # 3. 结构阻力位
        for r in res_sorted[:3]:
            if len(tp_raw) >= 3: break
            if _dedup(r["price"], tp_raw):
                tp_raw.append({"price": r["price"],
                               "label": f"阻力位（{r.get('source','pivot').split('+')[0]}）",
                               "badge": r.get("source", "")})
        # 4. Fibonacci 延伸位（ext=True优先于回调位）
        fib_ext_above = sorted([f for f in fib_above if f.get("ext")], key=lambda x: x["price"])
        fib_ret_above = sorted([f for f in fib_above if not f.get("ext")], key=lambda x: x["price"])
        for f in fib_ext_above + fib_ret_above:
            if len(tp_raw) >= 3: break
            if _dedup(f["price"], tp_raw):
                tp_raw.append({"price": f["price"], "label": f["label"], "badge": "fib"})
        # 5. 兜底：固定R倍数
        for mult in [2.0, 3.0, 4.5]:
            if len(tp_raw) >= 3: break
            tp_raw.append({"price": entry_price + mult * (entry_price - sl_price),
                           "label": f"×{mult:.0f}R 目标", "badge": ""})

        risk = entry_price - sl_price
        tp_levels = []
        for i, t in enumerate(sorted(tp_raw, key=lambda x: x["price"])[:3]):
            rr = round((t["price"] - entry_price) / risk, 2) if risk > 0 else 0
            tp_levels.append({"tp_label": f"TP{i+1}", "label": t["label"],
                               "price": round(t["price"], 2),
                               "dist_pct": round((t["price"] - entry_price) / entry_price * 100, 2),
                               "source": t.get("badge", ""), "rr": rr})

    # ══════════════════════════════════════════════════════
    else:  # SHORT
        # ── 止损（结构型优先）────────────────────────────
        sl_cands = []
        if key_high and key_high > price:
            sl_cands.append(("结构型（摆动高点+0.5ATR）", key_high + 0.5 * atr_val, "structural"))
        if liq_sl_short:
            sl_cands.append(("清算区优化止损", liq_sl_short, "liq"))
        if res_sorted:
            sl_cands.append(("最近阻力+0.5ATR", res_sorted[0]["price"] + 0.5 * atr_val, "resistance"))
        sl_cands.append(("ATR×1.5止损", atr_data.get("sl_short") or price + 1.5 * atr_val, "atr"))
        sl_label, sl_price, sl_type = _best_sl(sl_cands)
        if sl_price is None:
            sl_price = round(price + 1.5 * atr_val, 2)
            sl_label, sl_type = "ATR×1.5止损", "atr"
        sl_price = round(sl_price, 2)
        sl_dist_pct = round((sl_price - price) / price * 100, 2)

        # ── 入场价位 ─────────────────────────────────────
        near_hunt = any(z.get("near_hunt") and z.get("type") == "short_stop_hunt"
                        for z in hunt_zones)
        if score <= -8 and not near_hunt:
            entry_price  = price
            entry_method = "市价追入"
            entry_note   = f"动能强劲（评分 {score:+d}），可直接市价做空"
        elif res_sorted and in_range(res_sorted[0]["price"]):
            entry_price  = round(res_sorted[0]["price"] * 0.999, 2)
            entry_method = "限价挂单（阻力区）"
            entry_note   = f"价格已触及阻力 ${res_sorted[0]['price']:,.0f}，可限价做空"
        elif poc and in_range(poc) and poc > price:
            entry_price  = round(poc, 2)
            entry_method = "限价挂单（VP POC）"
            entry_note   = f"价格接近成交量POC ${poc:,.0f}，可限价做空"
        elif res_sorted:
            entry_price  = round(res_sorted[0]["price"] - atr_val * 0.1, 2)
            entry_method = "等待反弹限价"
            entry_note   = f"建议等反弹至阻力 ${res_sorted[0]['price']:,.0f} 附近再做空"
        elif vah and vah > price:
            entry_price  = round(vah, 2)
            entry_method = "等待反弹至VA上沿"
            entry_note   = f"等待反弹至价值区间上沿 VAH ${vah:,.0f}"
        else:
            entry_price  = price
            entry_method = "市价（无明显阻力）"
            entry_note   = "无明显阻力参考位，谨慎控制仓位"
        entry_price    = round(entry_price, 2)
        entry_dist_pct = round((entry_price - price) / price * 100, 2)

        # ── 止盈目标（做空：优先用VP关键位，再用支撑/Fib）─────
        tp_raw = []
        _dedup = lambda p, lst: not any(abs(p - t["price"]) / (p or 1) < 0.006 for t in lst)
        # 1. VP VAL（价值区下沿）= 空单首要目标
        if val_ and val_ < entry_price * 0.997 and _dedup(val_, tp_raw):
            tp_raw.append({"price": val_, "label": "VP VAL（价值区下沿）", "badge": "vp_val"})
        # 2. VP HVN 节点（下方成交量密集区）
        for _h in sorted([h for h in _hvn_list if h < entry_price * 0.997], reverse=True):
            if len(tp_raw) >= 3: break
            if _dedup(_h, tp_raw):
                tp_raw.append({"price": _h, "label": "VP HVN（成交量节点）", "badge": "vp_hvn"})
        # 3. 结构支撑位
        for s in sup_sorted[:3]:
            if len(tp_raw) >= 3: break
            if _dedup(s["price"], tp_raw):
                tp_raw.append({"price": s["price"],
                               "label": f"支撑位（{s.get('source','pivot').split('+')[0]}）",
                               "badge": s.get("source", "")})
        # 4. Fib 回调位
        for f in fib_below:
            if len(tp_raw) >= 3: break
            if _dedup(f["price"], tp_raw):
                tp_raw.append({"price": f["price"], "label": f["label"], "badge": "fib"})
        # 5. 兜底
        for mult in [2.0, 3.0, 4.5]:
            if len(tp_raw) >= 3: break
            tp_raw.append({"price": entry_price - mult * (sl_price - entry_price),
                           "label": f"×{mult:.0f}R 目标", "badge": ""})

        risk = sl_price - entry_price
        tp_levels = []
        for i, t in enumerate(sorted(tp_raw, key=lambda x: x["price"], reverse=True)[:3]):
            rr = round((entry_price - t["price"]) / risk, 2) if risk > 0 else 0
            tp_levels.append({"tp_label": f"TP{i+1}", "label": t["label"],
                               "price": round(t["price"], 2),
                               "dist_pct": round((entry_price - t["price"]) / entry_price * 100, 2),
                               "source": t.get("badge", ""), "rr": rr})

    # ── R:R 质量评估 ──────────────────────────────────────────
    rr1 = tp_levels[0]["rr"] if tp_levels else 0
    rr_quality = ("优秀 ✅" if rr1 >= 3.0 else
                  "良好 ✅" if rr1 >= 2.0 else
                  "勉强 ⚠️" if rr1 >= 1.5 else
                  "不建议 ❌")
    recommended = rr1 >= 1.5

    # R:R < 1.2 直接否决：风险收益比过低，不应生成可执行计划
    if rr1 < 1.2:
        return {
            "direction": "wait", "direction_cn": "观望 ↔️", "color": "#f9a825",
            "reason": (
                f"R:R 仅 {rr1:.1f}x（< 1.2x 最低门槛）。"
                f"TP1 {fmt(tp_levels[0]['price']) if tp_levels else '--'} 距入场仅 "
                f"{tp_levels[0]['dist_pct']:.2f}% ，而止损 {sl_dist_pct:.2f}%，"
                f"等待更优入场点或价格向TP方向移动后再评估。"
            ),
            "rr_tp1": rr1, "rr_quality": rr_quality, "recommended": False,
            "entry_price": entry_price, "sl_price": sl_price, "tp_levels": tp_levels,
        }

    # ── 仓位建议（1% 账户风险模型）───────────────────────────
    risk_pct = sl_dist_pct if sl_dist_pct > 0 else 1.5
    pos_pct  = min(100, round(1.0 / risk_pct * 100))
    pos_note = f"按账户 1% 风险：仓位约 {pos_pct}%（止损幅度 {risk_pct:.2f}%）"

    return {
        "direction":      direction,
        "direction_cn":   "做多 📈" if is_long else "做空 📉",
        "color":          "#26a69a" if is_long else "#ef5350",
        "score":          score,
        "max_score":      max_sc,
        "entry_method":   entry_method,
        "entry_price":    entry_price,
        "entry_dist_pct": entry_dist_pct,
        "entry_note":     entry_note,
        "sl_price":       sl_price,
        "sl_label":       sl_label,
        "sl_type":        sl_type,
        "sl_dist_pct":    sl_dist_pct,
        "tp_levels":      tp_levels,
        "rr_tp1":         rr1,
        "rr_quality":     rr_quality,
        "recommended":    recommended,
        "position_note":  pos_note,
        "near_hunt_warn": near_hunt,
    }


# ─────────────────────────────────────────────
# F-2. 即时回测引擎
# ─────────────────────────────────────────────

def _extract_signal_state(df: pd.DataFrame, i: int) -> dict:
    """
    从 DataFrame 第 i 行提取 15 维信号特征向量。
    维度越多，相似度筛选越精准，同时通过降低阈值保证样本量。
    """
    row = df.iloc[i]

    def _v(col):
        v = row.get(col, np.nan)
        return float(v) if not pd.isna(v) else None

    close = _v("close") or 1.0

    # ── 1. RSI 分区（5级）──────────────────────────────
    rsi = _v("rsi")
    if rsi is None:        rsi_zone = "unknown"
    elif rsi >= 70:        rsi_zone = "overbought"
    elif rsi <= 30:        rsi_zone = "oversold"
    elif rsi >= 55:        rsi_zone = "strong"
    elif rsi <= 45:        rsi_zone = "weak"
    else:                  rsi_zone = "neutral"

    # ── 2. 均线多空位置 ─────────────────────────────────
    ma20  = _v("ma20");  ma50 = _v("ma50");  ma200 = _v("ma200")
    above_ma20  = bool(close > ma20)  if ma20  else None
    above_ma50  = bool(close > ma50)  if ma50  else None
    above_ma200 = bool(close > ma200) if ma200 else None

    # ── 3. MA 短中期交叉状态（金叉/死叉/无）────────────────
    if ma20 and ma50:
        if   ma20 > ma50 * 1.001: ma_cross = "golden"    # MA20 > MA50 金叉区
        elif ma20 < ma50 * 0.999: ma_cross = "death"     # MA20 < MA50 死叉区
        else:                     ma_cross = "neutral"
    else:
        ma_cross = "unknown"

    # ── 4. MACD 方向 + 柱状图扩缩 ──────────────────────
    macd = _v("macd"); macd_sig = _v("macd_signal"); macd_hist = _v("macd_hist")
    macd_bull = bool(macd > macd_sig) if (macd is not None and macd_sig is not None) else None
    # 柱状图相对上一根
    if i > 0 and macd_hist is not None:
        prev_hist = df["macd_hist"].iloc[i-1]
        if not pd.isna(prev_hist):
            macd_expand = bool(abs(macd_hist) > abs(float(prev_hist)))
        else:
            macd_expand = None
    else:
        macd_expand = None

    # ── 5. 布林带位置 + 带宽状态 ────────────────────────
    bb_u = _v("bb_upper"); bb_l = _v("bb_lower"); bb_m = _v("bb_mid")
    if bb_u and bb_l and bb_m:
        if   close >= bb_u:  bb_zone = "above_upper"
        elif close <= bb_l:  bb_zone = "below_lower"
        elif close > bb_m:   bb_zone = "above_mid"
        else:                bb_zone = "below_mid"
        bb_width_pct = (bb_u - bb_l) / bb_m * 100
        # 带宽分位：用当前窗口 50 根历史估算
        window_bb = df["bb_upper"].iloc[max(0,i-49):i+1] - df["bb_lower"].iloc[max(0,i-49):i+1]
        bb_mid_w  = df["bb_mid"].iloc[max(0,i-49):i+1]
        width_ser = (window_bb / bb_mid_w.replace(0, np.nan) * 100).dropna()
        if len(width_ser) >= 10:
            bb_squeeze = bool(bb_width_pct <= float(width_ser.quantile(0.25)))  # 带宽压缩
        else:
            bb_squeeze = None
    else:
        bb_zone = "unknown"; bb_squeeze = None

    # ── 6. KDJ 状态 ─────────────────────────────────────
    kdj_k = _v("kdj_k"); kdj_d = _v("kdj_d")
    if kdj_k is not None and kdj_d is not None:
        if   kdj_k >= 80:    kdj_zone = "overbought"
        elif kdj_k <= 20:    kdj_zone = "oversold"
        elif kdj_k > kdj_d:  kdj_zone = "bull"
        else:                kdj_zone = "bear"
    else:
        kdj_zone = "unknown"

    # ── 7. OBV 趋势（20 根斜率方向）────────────────────
    if "obv" in df.columns and i >= 20:
        obv_now  = df["obv"].iloc[i]
        obv_prev = df["obv"].iloc[i-20]
        if not pd.isna(obv_now) and not pd.isna(obv_prev):
            obv_trend = "up" if float(obv_now) > float(obv_prev) else "down"
        else:
            obv_trend = "unknown"
    else:
        obv_trend = "unknown"

    # ── 8. ADX 市场机制 ─────────────────────────────────
    adx = _v("adx"); di_pos = _v("adx_pos"); di_neg = _v("adx_neg")
    if adx is None:        regime = "unknown"
    elif adx >= 25:        regime = "trending"
    elif adx <= 20:        regime = "ranging"
    else:                  regime = "transitional"
    # 趋势方向（DI+ vs DI-）
    if di_pos and di_neg:
        trend_dir = "up" if di_pos > di_neg else "down"
    else:
        trend_dir = "unknown"

    # ── 9. OFI（主动成交量方向）────────────────────────
    ofi = _v("ofi")
    if ofi is None:        ofi_dir = "unknown"
    elif ofi > 0.2:        ofi_dir = "buy"
    elif ofi < -0.2:       ofi_dir = "sell"
    else:                  ofi_dir = "neutral"

    # ── 10. ATR 百分位（波动率分层）─────────────────────
    if "atr" in df.columns:
        atr_win = df["atr"].iloc[max(0, i-99):i+1].dropna()
        atr_cur = _v("atr")
        if len(atr_win) >= 10 and atr_cur:
            atr_rank = float((atr_win < atr_cur).mean())
        else:
            atr_rank = 0.5
    else:
        atr_rank = 0.5
    vol_regime = "high" if atr_rank >= 0.75 else ("low" if atr_rank <= 0.25 else "normal")

    # ── 11. VWAP 位置 ────────────────────────────────────
    vwap = _v("vwap")
    above_vwap = bool(close > vwap) if vwap else None

    # ── 12. 价格动量（5 根收益率方向）───────────────────
    if i >= 5:
        price_5ago = float(df["close"].iloc[i-5])
        momentum   = "up" if close > price_5ago * 1.002 else ("down" if close < price_5ago * 0.998 else "flat")
    else:
        momentum = "unknown"

    return {
        # 核心信号（高权重）
        "rsi_zone":    rsi_zone,      # RSI 分区
        "regime":      regime,         # 市场机制（趋势/震荡）
        "trend_dir":   trend_dir,      # DI 趋势方向
        "vol_regime":  vol_regime,     # 波动率环境
        "macd_bull":   macd_bull,      # MACD 方向
        "bb_zone":     bb_zone,        # 布林带位置
        # 次要信号（中权重）
        "above_ma50":  above_ma50,     # 价格在 MA50 上下
        "above_ma200": above_ma200,    # 价格在 MA200 上下
        "ma_cross":    ma_cross,       # 金叉/死叉区
        "kdj_zone":    kdj_zone,       # KDJ 状态
        "obv_trend":   obv_trend,      # OBV 方向
        "momentum":    momentum,       # 5根动量
        # 辅助信号（低权重）
        "above_ma20":  above_ma20,     # 价格在 MA20 上下
        "macd_expand": macd_expand,    # MACD 柱扩张
        "bb_squeeze":  bb_squeeze,     # 布林带压缩
        "above_vwap":  above_vwap,     # VWAP 位置
        "ofi_dir":     ofi_dir,        # 主动成交量方向
    }


def _signal_similarity(s1: dict, s2: dict) -> float:
    """
    计算两个信号状态的加权相似度 [0, 1]。
    权重设计：核心市场结构指标权重高，辅助指标权重低，
    既保证形态相似度，又不过度苛刻导致样本不足。
    """
    keys_weight = {
        # 核心信号（权重 2.0）——决定市场性质，必须匹配
        "rsi_zone":    2.0,
        "regime":      2.0,
        "vol_regime":  2.0,
        # 重要信号（权重 1.5）——决定方向偏向
        "macd_bull":   1.5,
        "bb_zone":     1.5,
        "trend_dir":   1.5,
        "above_ma50":  1.5,
        "ma_cross":    1.5,
        # 次要信号（权重 1.0）
        "above_ma200": 1.0,
        "kdj_zone":    1.0,
        "obv_trend":   1.0,
        "momentum":    1.0,
        # 辅助信号（权重 0.5）——噪音较多，权重最低
        "above_ma20":  0.5,
        "macd_expand": 0.5,
        "bb_squeeze":  0.5,
        "above_vwap":  0.5,
        "ofi_dir":     0.5,
    }
    total_w = sum(keys_weight.values())
    match_w = 0.0
    for k, w in keys_weight.items():
        v1, v2 = s1.get(k), s2.get(k)
        if v1 is None or v2 is None:
            match_w += w * 0.5   # 缺失数据算半分
        elif v1 == v2:
            match_w += w
    return match_w / total_w


def run_backtest(df: pd.DataFrame, current_signal: dict,
                 hold_bars: int = 5,
                 atr_sl_mult: float = 1.5,
                 atr_tp_mult: float = 3.0,
                 min_samples: int = 20,
                 similarity_thresh: float = 0.62) -> dict:
    """
    即时回测引擎：在 df（建议 1500 根以上）中找出与 current_signal
    相似的历史形态，多维度统计胜率、盈亏比、期望值，并生成详细描述。

    参数:
        df                : 含全部指标的长历史 DataFrame
        current_signal    : _extract_signal_state 生成的当前信号字典
        hold_bars         : 主要持仓 K 线数
        atr_sl_mult       : 止损倍数（×ATR）
        atr_tp_mult       : 止盈倍数（×ATR）
        min_samples       : 最少有效样本数
        similarity_thresh : 相似度阈值，降低可增加样本量
    """
    # ── 1. 确定信号方向 ─────────────────────────────────
    rsi_z  = current_signal.get("rsi_zone", "neutral")
    bb_z   = current_signal.get("bb_zone",  "neutral")
    macd_b = current_signal.get("macd_bull", None)
    regime = current_signal.get("regime", "unknown")
    td     = current_signal.get("trend_dir", "unknown")
    mom    = current_signal.get("momentum", "unknown")

    bull_hints = sum([
        rsi_z in ("oversold", "strong"),
        bb_z  in ("above_mid", "above_upper"),
        macd_b is True,
        current_signal.get("above_ma50")  is True,
        current_signal.get("above_ma200") is True,
        current_signal.get("ma_cross")    == "golden",
        current_signal.get("kdj_zone")    in ("bull", "oversold"),
        current_signal.get("obv_trend")   == "up",
        td    == "up",
        mom   == "up",
        current_signal.get("above_vwap")  is True,
    ])
    bear_hints = sum([
        rsi_z in ("overbought", "weak"),
        bb_z  in ("below_mid", "below_lower"),
        macd_b is False,
        current_signal.get("above_ma50")  is False,
        current_signal.get("above_ma200") is False,
        current_signal.get("ma_cross")    == "death",
        current_signal.get("kdj_zone")    in ("bear", "overbought"),
        current_signal.get("obv_trend")   == "down",
        td    == "down",
        mom   == "down",
        current_signal.get("above_vwap")  is False,
    ])
    direction     = "long" if bull_hints >= bear_hints else "short"
    bull_strength = bull_hints / max(1, bull_hints + bear_hints)  # 多头倾向强度

    n         = len(df)
    warm_bars = 200   # 指标预热所需
    start_idx = warm_bars
    # 多持仓周期：[hold_bars, hold_bars*2, hold_bars//2]，取最优者作为主结果
    hold_variants = sorted(set([
        max(2, hold_bars // 2),
        hold_bars,
        hold_bars * 2,
    ]))

    # ── 2. 扫描历史相似形态 ──────────────────────────────
    matches = []   # 每条: {idx, sim, entry, atr}
    for i in range(start_idx, n - max(hold_variants) - 1):
        hist_sig = _extract_signal_state(df, i)
        sim      = _signal_similarity(current_signal, hist_sig)
        if sim < similarity_thresh:
            continue
        atr_i = float(df["atr"].iloc[i]) if "atr" in df.columns and not pd.isna(df["atr"].iloc[i]) \
                else float(df["close"].iloc[i]) * 0.015
        matches.append({
            "idx":   i,
            "sim":   sim,
            "entry": float(df["close"].iloc[i]),
            "atr":   atr_i,
            "date":  str(df["date"].iloc[i])[:16],
        })

    # ── 3. 多持仓周期回测 ───────────────────────────────
    def _calc_period(hb: int):
        """对给定持仓周期 hb 跑回测，返回统计字典。"""
        wins = []; losses = []
        sl_hits = 0; tp_hits = 0; timeout_exits = 0
        details = []
        for m in matches:
            i           = m["idx"]
            entry_price = m["entry"]
            atr_i       = m["atr"]
            sl_price = entry_price - atr_sl_mult * atr_i if direction == "long" \
                       else entry_price + atr_sl_mult * atr_i
            tp_price = entry_price + atr_tp_mult * atr_i if direction == "long" \
                       else entry_price - atr_tp_mult * atr_i

            hit_sl = False; hit_tp = False
            exit_pct = 0.0
            for j in range(i + 1, min(i + hb + 1, n)):
                hi = float(df["high"].iloc[j])
                lo = float(df["low"].iloc[j])
                if direction == "long":
                    if lo <= sl_price:
                        hit_sl = True; sl_hits += 1
                        exit_pct = (sl_price - entry_price) / entry_price * 100; break
                    if hi >= tp_price:
                        hit_tp = True; tp_hits += 1
                        exit_pct = (tp_price - entry_price) / entry_price * 100; break
                else:
                    if hi >= sl_price:
                        hit_sl = True; sl_hits += 1
                        exit_pct = (entry_price - sl_price) / entry_price * 100 * -1; break
                    if lo <= tp_price:
                        hit_tp = True; tp_hits += 1
                        exit_pct = (entry_price - tp_price) / entry_price * 100; break

            if not hit_sl and not hit_tp:
                timeout_exits += 1
                ep  = float(df["close"].iloc[min(i + hb, n - 1)])
                exit_pct = (ep - entry_price) / entry_price * 100
                if direction == "short":
                    exit_pct = -exit_pct

            if exit_pct > 0:
                wins.append(exit_pct)
            else:
                losses.append(exit_pct)
            details.append({**m, "exit_pct": round(exit_pct, 3), "win": exit_pct > 0})

        total = len(wins) + len(losses)
        if total == 0:
            return None
        win_rate   = len(wins) / total
        avg_win    = float(np.mean(wins))   if wins   else 0.0
        avg_loss   = float(np.mean(losses)) if losses else 0.0
        rr         = abs(avg_win / avg_loss) if avg_loss != 0 else 99.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        return {
            "hold_bars":     hb,
            "total":         total,
            "wins":          len(wins),
            "losses":        len(losses),
            "win_rate":      round(win_rate,  4),
            "avg_win_pct":   round(avg_win,   3),
            "avg_loss_pct":  round(avg_loss,  3),
            "rr_ratio":      round(min(rr, 99.0), 2),
            "expectancy":    round(expectancy, 3),
            "sl_hits":       sl_hits,
            "tp_hits":       tp_hits,
            "timeout_exits": timeout_exits,
            "details":       details[-8:],   # 最近 8 条样本
        }

    period_results = {}
    for hb in hold_variants:
        r = _calc_period(hb)
        if r:
            period_results[hb] = r

    # ── 4. 主结果取期望值最优的持仓周期 ─────────────────
    total_matches = len(matches)

    if total_matches < min_samples:
        return {
            "win_rate":         None,
            "loss_rate":        None,
            "avg_win_pct":      None,
            "avg_loss_pct":     None,
            "expectancy":       None,
            "rr_ratio":         None,
            "sample_count":     total_matches,
            "min_samples":      min_samples,
            "confidence_level": "极低",
            "confidence_pct":   max(5, int(total_matches / min_samples * 30)),
            "signal_direction": direction,
            "bull_strength":    round(bull_strength, 3),
            "period_results":   period_results,
            "detail":           [m for m in matches[-8:]],
            "score":            0,
            "note":             (
                f"历史相似形态仅 {total_matches} 次（阈值 {similarity_thresh:.0%}，"
                f"需 ≥{min_samples}），样本不足；"
                f"多头信号 {bull_hints} 项，空头信号 {bear_hints} 项"
            ),
        }

    # 选期望值最高的持仓周期作为主结果
    best_hb  = max(period_results, key=lambda h: period_results[h]["expectancy"])
    best     = period_results[best_hb]
    win_rate = best["win_rate"]
    avg_win  = best["avg_win_pct"]
    avg_loss = best["avg_loss_pct"]
    rr_ratio = best["rr_ratio"]
    expectancy = best["expectancy"]

    # ── 5. 置信度计算（胜率 × 样本量权重 × 一致性系数）───
    sample_weight  = min(1.0, total_matches / 80)     # 80 条满权
    # 多周期一致性：三个持仓周期的期望值方向一致则加分
    period_exps    = [v["expectancy"] for v in period_results.values()]
    consistency    = sum(1 for e in period_exps if (e > 0) == (expectancy > 0)) / max(1, len(period_exps))
    raw_conf       = win_rate * sample_weight * (0.7 + 0.3 * consistency)
    if   raw_conf >= 0.60: conf_level = "高";   conf_pct = int(65 + raw_conf * 30)
    elif raw_conf >= 0.45: conf_level = "中";   conf_pct = int(45 + raw_conf * 40)
    elif raw_conf >= 0.30: conf_level = "低";   conf_pct = int(25 + raw_conf * 50)
    else:                  conf_level = "极低"; conf_pct = int(raw_conf * 80)
    conf_pct = min(95, max(5, conf_pct))

    # ── 6. 评分 ─────────────────────────────────────────
    if   expectancy >= 0.5:  bt_score = +2
    elif expectancy >= 0.2:  bt_score = +1
    elif expectancy >= -0.2: bt_score = 0
    elif expectancy >= -0.5: bt_score = -1
    else:                    bt_score = -2

    # ── 7. 详细描述生成 ──────────────────────────────────
    dir_cn     = "做多" if direction == "long" else "做空"
    regime_cn  = {"trending": "趋势市", "ranging": "震荡市",
                  "transitional": "过渡期", "unknown": "未知"}.get(regime, regime)
    vol_cn     = {"high": "高波动", "low": "低波动", "normal": "正常波动"}.get(
                  current_signal.get("vol_regime", "normal"), "正常波动")
    rsi_cn     = {"overbought": "超买区(≥70)", "oversold": "超卖区(≤30)",
                  "strong": "偏强区(55-70)", "weak": "偏弱区(30-45)",
                  "neutral": "中性区(45-55)"}.get(rsi_z, rsi_z)
    bb_cn      = {"above_upper": "布林上轨上方", "below_lower": "布林下轨下方",
                  "above_mid": "布林中轨上方", "below_mid": "布林中轨下方"}.get(bb_z, bb_z)
    ma_cross_cn = {"golden": "MA20>MA50 金叉区", "death": "MA20<MA50 死叉区",
                   "neutral": "均线纠缠"}.get(current_signal.get("ma_cross",""), "")
    momentum_cn = {"up": "价格上行动量", "down": "价格下行动量", "flat": "动量平缓"}.get(mom, "")
    obv_cn      = {"up": "OBV上升（资金流入）", "down": "OBV下降（资金流出）"}.get(
                   current_signal.get("obv_trend",""), "")

    # 构建当前形态特征描述
    feature_desc = "、".join(filter(None, [
        rsi_cn, bb_cn, ma_cross_cn, momentum_cn, obv_cn,
        f"ADX {'≥25' if regime=='trending' else '≤20'} {regime_cn}",
        vol_cn,
    ]))

    # 构建回测结论描述
    if win_rate >= 0.65:
        wr_desc = f"胜率 {win_rate:.1%} 显著偏高"
    elif win_rate >= 0.55:
        wr_desc = f"胜率 {win_rate:.1%} 略偏高"
    elif win_rate <= 0.40:
        wr_desc = f"胜率 {win_rate:.1%} 偏低"
    else:
        wr_desc = f"胜率 {win_rate:.1%} 接近随机"

    if rr_ratio >= 2.0:
        rr_desc = f"盈亏比 {rr_ratio:.2f} 优秀"
    elif rr_ratio >= 1.5:
        rr_desc = f"盈亏比 {rr_ratio:.2f} 良好"
    elif rr_ratio >= 1.0:
        rr_desc = f"盈亏比 {rr_ratio:.2f} 一般"
    else:
        rr_desc = f"盈亏比 {rr_ratio:.2f} 不佳"

    if expectancy > 0:
        exp_desc = f"期望正收益 +{expectancy:.3f}%/笔，策略具备统计优势"
    else:
        exp_desc = f"期望负收益 {expectancy:.3f}%/笔，当前形态历史表现欠佳"

    # 多周期一致性描述
    if len(period_results) >= 2:
        consistent_cnt = sum(1 for v in period_results.values() if v["expectancy"] > 0)
        if consistent_cnt == len(period_results):
            consistency_desc = f"三个持仓周期（{'/'.join(str(h)+'根' for h in sorted(period_results))}）期望值均为正，信号一致性强"
        elif consistent_cnt == 0:
            consistency_desc = f"三个持仓周期期望值均为负，跨周期信号一致性差"
        else:
            consistency_desc = f"持仓周期一致性一般（{consistent_cnt}/{len(period_results)} 个周期期望为正）"
    else:
        consistency_desc = ""

    # 出场方式分布描述
    sl_rate = best["sl_hits"] / max(1, best["total"])
    tp_rate = best["tp_hits"] / max(1, best["total"])
    to_rate = best["timeout_exits"] / max(1, best["total"])
    exit_desc = (
        f"历史出场分布：止盈触达 {tp_rate:.1%}、止损触达 {sl_rate:.1%}、"
        f"超时平仓 {to_rate:.1%}"
    )

    # 风险提示
    risk_notes = []
    if win_rate < 0.45:
        risk_notes.append("胜率偏低，需严格止损控制单笔亏损")
    if rr_ratio < 1.0:
        risk_notes.append("盈亏比不足1，即使胜率50%也会亏损，不建议参考")
    if total_matches < 30:
        risk_notes.append(f"样本仅 {total_matches} 条，统计意义有限")
    if consistency < 0.67:
        risk_notes.append("多周期信号不一致，持仓时长对结果影响大")
    if current_signal.get("vol_regime") == "high":
        risk_notes.append("当前处于高波动环境，实际波动可能超出历史均值")

    description = {
        "signal_feature":   f"当前形态：{feature_desc}",
        "direction_reason": (
            f"综合 {bull_hints} 项多头信号 vs {bear_hints} 项空头信号，"
            f"判断 {dir_cn} 方向（多头倾向 {bull_strength:.0%}）"
        ),
        "backtest_result":  f"{wr_desc}，{rr_desc}，{exp_desc}",
        "exit_distribution":exit_desc,
        "consistency":      consistency_desc,
        "risk_notes":       risk_notes,
        "best_hold_bars":   best_hb,
        "period_summary":   {
            str(hb): f"胜率{v['win_rate']:.1%} 期望{v['expectancy']:+.3f}%"
            for hb, v in sorted(period_results.items())
        },
    }

    return {
        "win_rate":         win_rate,
        "loss_rate":        round(1 - win_rate, 4),
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "expectancy":       expectancy,
        "rr_ratio":         rr_ratio,
        "sample_count":     total_matches,
        "min_samples":      min_samples,
        "similarity_thresh":similarity_thresh,
        "confidence_level": conf_level,
        "confidence_pct":   conf_pct,
        "consistency":      round(consistency, 2),
        "signal_direction": direction,
        "bull_strength":    round(bull_strength, 3),
        "bull_hints":       bull_hints,
        "bear_hints":       bear_hints,
        "hold_bars":        best_hb,
        "atr_sl_mult":      atr_sl_mult,
        "atr_tp_mult":      atr_tp_mult,
        "period_results":   {str(k): v for k, v in period_results.items()},
        "detail":           best.get("details", []),
        "description":      description,
        "score":            bt_score,
        "note": (
            f"历史 {total_matches} 次相似形态（阈值 {similarity_thresh:.0%}） | "
            f"最优持仓 {best_hb} 根 | {wr_desc} | {rr_desc} | "
            f"期望 {expectancy:+.3f}%/笔 | 置信度 {conf_pct}%({conf_level})"
        ),
    }


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
