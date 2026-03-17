# -*- coding: utf-8 -*-
"""
多币种实时分析服务器
每5分钟自动拉取最新数据并分析，通过 REST API 提供给前端仪表板
访问: http://localhost:5000
"""

import os
import sys
import io
import math
import time
import json
import socket
import threading
import contextlib
import collections
from datetime import datetime, timedelta

# ── Windows 中文主机名 GBK 编码补丁 ──────────────────────
_orig_getfqdn = socket.getfqdn
def _safe_getfqdn(name=""):
    try:
        return _orig_getfqdn(name)
    except UnicodeDecodeError:
        return name or "localhost"
socket.getfqdn = _safe_getfqdn
# ────────────────────────────────────────────────────────

import numpy as np
import requests as req
from flask import Flask, jsonify, render_template, request
from flask.json import JSONEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc_analysis import (
    fetch_data, fetch_data_extended, calculate_indicators,
    find_support_resistance, calculate_fibonacci,
    generate_report, TIMEFRAMES,
    calculate_volume_profile, detect_fair_value_gaps, detect_market_structure,
)
from onchain_engine import OnChainMonitor as _OnChainMonitor

# 链上数据缓存（每30分钟刷新，避免频繁请求外部API）
_onchain_cache: dict = {}
_onchain_cache_ts: float = 0.0
_ONCHAIN_TTL = 1800  # 30 minutes


class NumpyEncoder(JSONEncoder):
    """将 NumPy/Pandas 标量转为 Python 原生类型"""
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


app = Flask(__name__)
app.json_encoder = NumpyEncoder
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
# SYMBOLS = ["BTCUSDT"]
REFRESH_INTERVAL = 300  # 5 分钟

# 缓存结构:
# _cache = {
#   "BTCUSDT": { "last_update":..., "next_update":..., "updating":..., "ready":..., "timeframes": {...} },
#   "ETHUSDT": { ... },
#   ...
# }
_cache = {}
for s in SYMBOLS:
    _cache[s] = {
        "last_update": None,
        "next_update": None,
        "updating":    False,
        "ready":       False,
        "update_id":   0,
        "ob_pressure": None,   # 实时订单簿压力指数快照
        "timeframes":  {}
    }

_lock = threading.Lock()
_update_counter = 0   # 每完成一轮全量刷新后递增


class _TokenBucket:
    """线程安全令牌桶——全局 API 请求速率控制"""
    def __init__(self, rate: float, capacity: int):
        self._tokens   = float(capacity)
        self._capacity = float(capacity)
        self._rate     = rate          # 令牌/秒
        self._ts       = time.monotonic()
        self._lock     = threading.Lock()

    def acquire(self, cost: int = 1):
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._ts) * self._rate
                )
                self._ts = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                wait = (cost - self._tokens) / self._rate
            time.sleep(wait)


# 最多 5 次/秒，允许最多 10 次瞬间突发（远低于 Binance 2400 权重/分钟上限）
_rate_limiter = _TokenBucket(rate=5.0, capacity=10)

# 各时间框架 K 线数据的最短刷新间隔（秒）
# 超短周期需实时性，长周期无需每 5 分钟重新拉取
_TF_TTL = {
    "15m":  300,   # 5  分钟（与 REFRESH_INTERVAL 一致，每次都刷新）
    "30m":  300,
    "1h":   600,   # 10 分钟：每隔一个 5 分钟周期才真正拉取
    "2h":   600,
    "4h":   900,   # 15 分钟
    "8h":  1800,   # 30 分钟
    "1d":  3600,   # 1  小时
    "1w":  7200,   # 2  小时
}


def _safe(v):
    """将 NaN / None 转为 None，其余四舍五入到4位小数"""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 4)
    except Exception:
        return None


# 各时间框架对订单簿实时数据的相关性权重（越短越相关，日线/周线不使用）
_OB_TF_WEIGHTS = {
    "15m": 1.0, "30m": 1.0, "1h": 1.0,
    "2h":  0.6, "4h":  0.6, "8h": 0.6,
    "1d":  0.0, "1w":  0.0,
}


def fetch_ob_walls(symbol: str, depth_limit: int = 1000):
    """
    从 Binance 期货深度接口获取挂单墙 + 原始深度数据。
    返回 (walls, raw_data)：
      walls    = [(price, weight, side), ...] 显著挂单墙列表
      raw_data = {"bids": [[price, qty], ...], "asks": [...]} 原始数据供压力指数计算
    出错时返回 ([], None)。
    """
    try:
        r = req.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": symbol, "limit": depth_limit},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[OB] fetch_ob_walls {symbol} 失败: {e}")
        return [], None

    walls = []
    for side_key, side_label in [("bids", "orderbook_bid"), ("asks", "orderbook_ask")]:
        raw = data.get(side_key, [])
        if not raw:
            continue
        try:
            sizes = sorted(float(e[1]) for e in raw)
            if not sizes:
                continue
            median_sz = sizes[len(sizes) // 2]   # 中位数（无需 scipy/numpy）
            if median_sz <= 0:
                continue
            for entry in raw:
                price = float(entry[0])
                size  = float(entry[1])
                ratio = size / median_sz
                if ratio >= 5:    # 超强挂单墙 → 基础权重 3.0
                    walls.append((price, 3.0, side_label))
                elif ratio >= 3:  # 中等挂单墙 → 基础权重 2.0
                    walls.append((price, 2.0, side_label))
        except Exception as e:
            print(f"[OB] 解析 {side_label} 失败: {e}")
    return walls, data


# 各币种压力指数历史快照（用于计算速度/趋势）
_ob_pressure_history = {s: collections.deque(maxlen=5) for s in SYMBOLS}


def calculate_ob_pressure(bids: list, asks: list, ref_dist_pct: float = 0.01) -> dict:
    """
    距离加权订单簿压力指数。
    在中间价 ±ref_dist_pct 范围内，对每档挂单按距离线性衰减加权求和。
    weight_i = qty_i × (1 - dist_i / ref_dist)

    返回:
      ratio        — bid_pressure / ask_pressure（>1 买盘强，<1 卖盘强）
      bid_pressure — 加权买压合计
      ask_pressure — 加权卖压合计
      score        — -2/-1/0/+1/+2 信号强度
      mid          — 中间价
      signal       — 文字描述
    """
    if not bids or not asks:
        return None
    try:
        mid      = (float(bids[0][0]) + float(asks[0][0])) / 2
        ref_dist = mid * ref_dist_pct

        bid_p = sum(
            float(b[1]) * (1.0 - abs(mid - float(b[0])) / ref_dist)
            for b in bids
            if abs(mid - float(b[0])) <= ref_dist and float(b[1]) > 0
        )
        ask_p = sum(
            float(a[1]) * (1.0 - abs(float(a[0]) - mid) / ref_dist)
            for a in asks
            if abs(float(a[0]) - mid) <= ref_dist and float(a[1]) > 0
        )

        ratio = round(bid_p / ask_p, 3) if ask_p > 0 else 9.99

        if   ratio > 2.0:  score, signal = +2, "极强买压，做多信号"
        elif ratio > 1.5:  score, signal = +1, "买压偏强"
        elif ratio < 0.5:  score, signal = -2, "极强卖压，做空信号"
        elif ratio < 0.67: score, signal = -1, "卖压偏强"
        else:              score, signal =  0, "买卖基本平衡"

        return {
            "ratio":        ratio,
            "bid_pressure": round(bid_p, 1),
            "ask_pressure": round(ask_p, 1),
            "score":        score,
            "mid":          round(mid, 2),
            "signal":       signal,
        }
    except Exception as e:
        print(f"[OB] calculate_ob_pressure 失败: {e}")
        return None


def run_single_tf(symbol, interval, limit, label, chart_show, sr_lb, sr_pn, sr_cluster_pct, fib_lb,
                  ob_walls=None, liq_clusters=None):
    """分析单个周期，返回 {rd, candles, supports, resistances, fib}"""
    # ── 主数据（500根，用于指标计算、图表、S/R）──────────
    df = fetch_data(symbol=symbol, interval=interval, limit=limit)
    if df.empty:
        return None
    df = calculate_indicators(df, tf_interval=interval)

    # ── 回测专用扩展历史数据（1500根，提升样本量）──────────
    # 日线/周线本身数据量足够且拉取慢，直接复用 df；短周期单独拉
    BT_TOTAL = {
        "15m": 1500, "30m": 1500, "1h": 1500,
        "2h":  1200, "4h":  1200, "8h": 1000,
        "1d":  800,  "1w":  400,
    }
    bt_total = BT_TOTAL.get(interval, 1500)
    if bt_total > limit:
        df_bt = fetch_data_extended(symbol=symbol, interval=interval, total=bt_total)
        if df_bt.empty:
            df_bt = df
        else:
            df_bt = calculate_indicators(df_bt, tf_interval=interval)
    else:
        df_bt = df

    # 构建当前时间框架的 OB 层位（乘以 TF 相关性权重）
    ob_tf_factor = _OB_TF_WEIGHTS.get(interval, 0.0)
    ob_levels = []
    if ob_walls and ob_tf_factor > 0.0:
        for ob_price, ob_weight, ob_side in ob_walls:
            ob_levels.append((ob_price, round(ob_weight * ob_tf_factor, 2), ob_side))

    # 清算区与时间框架相关性权重：短周期（15m/1h）直接参考，长周期（日线/周线）弱化
    # 日内清算区对短周期 S/R 有强指引，对周线参考意义小
    _LIQ_TF_WEIGHTS = {
        "15m": 1.0, "30m": 1.0, "1h": 1.0,
        "2h":  0.8, "4h":  0.8, "8h": 0.5,
        "1d":  0.3, "1w":  0.0,
    }
    liq_tf_factor = _LIQ_TF_WEIGHTS.get(interval, 0.0)
    liq_levels_tf = []
    if liq_clusters and liq_tf_factor > 0.0:
        for lc in liq_clusters:
            # 复制一份并按 TF 权重缩放 qty（用于 S/R 聚类权重）
            liq_levels_tf.append({**lc, "_tf_weight": liq_tf_factor})

    # ── 三大新分析模块 ────────────────────────────────────────
    vp_data  = calculate_volume_profile(df)
    fvg_data = detect_fair_value_gaps(df)
    ms_data  = detect_market_structure(df)

    supports, resistances = find_support_resistance(
        df, lookback=sr_lb, pivot_n=sr_pn,
        cluster_pct=sr_cluster_pct, ob_levels=ob_levels,
        liq_levels=liq_levels_tf if liq_levels_tf else None,
        vp_data=vp_data, fvg_zones=fvg_data,
    )
    fib_levels, fib_high, fib_low = calculate_fibonacci(df, lookback=fib_lb)

    # 静默调用 generate_report（不打印到控制台）
    # df_bt 传入用于回测（历史更长），df 用于指标/价格引用
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, rd = generate_report(
            df, supports, resistances, fib_levels,
            symbol=symbol, tf_label=label, verbose=False,
            liq_clusters=liq_clusters if liq_clusters else None,
            df_backtest=df_bt,
            market_structure=ms_data,
            fvg_zones=fvg_data,
            volume_profile=vp_data,
        )

    # 准备图表用的 K 线数据（带指标值）
    ind_cols = ["ma20", "ma50", "ma200", "ema20", "ema50",
                "rsi", "bb_upper", "bb_mid", "bb_lower",
                "macd", "macd_signal", "macd_hist",
                "kdj_k", "kdj_d", "kdj_j", "obv",
                "vwap", "atr", "atr_sl_long", "atr_sl_short",
                "delta", "cvd", "ofi"]          # 主动成交量
    plot_df = df.tail(chart_show).reset_index(drop=True)
    candles = []
    for _, row in plot_df.iterrows():
        c = {
            "t": row["date"].strftime("%Y-%m-%dT%H:%M"),
            "o": row["open"], "h": row["high"],
            "l": row["low"],  "c": row["close"], "v": row["volume"]
        }
        for col in ind_cols:
            c[col] = _safe(row.get(col))
        candles.append(c)

    price = rd["price"]
    return {
        "rd": rd,
        "candles": candles,
        "supports":    [{"price":    s[0],
                         "dist_pct": round((price - s[0]) / price * 100, 2),
                         "score":    round(s[1], 2),
                         "source":   s[2] if len(s) > 2 else "pivot",
                         "reason":   s[3] if len(s) > 3 else ""}
                        for s in supports],
        "resistances": [{"price":    r[0],
                         "dist_pct": round((r[0] - price) / price * 100, 2),
                         "score":    round(r[1], 2),
                         "source":   r[2] if len(r) > 2 else "pivot",
                         "reason":   r[3] if len(r) > 3 else ""}
                        for r in resistances],
        "fib": [{"label": k, "price": v,
                 "dist_pct": round((v - price) / price * 100, 2),
                 "near": abs((v - price) / price * 100) < 2}
                for k, v in fib_levels.items()],
        "fvg_zones":        fvg_data or [],
        "volume_profile":   vp_data or {},           # 含 chart_bins，供前端绘制 VP 条形图
        "market_structure": ms_data or {},
    }


# ─────────────────────────────────────────────
# 后台分析线程
# ─────────────────────────────────────────────

def _analyze_symbol(symbol: str) -> None:
    """单个币种的完整分析任务；设计为可在线程池中并行运行。"""
    with _lock:
        _cache[symbol]["updating"] = True

    print(f"[服务器] 正在分析 {symbol}...")
    cur_tf   = _cache[symbol].get("timeframes", {})  # 当前缓存的各周期数据
    new_tf   = {}
    now_ts   = time.time()
    skipped  = []
    fetched  = []

    # 判断是否有需要更新的短周期 TF（决定是否取 OB 数据，避免不必要的深度请求）
    needs_ob = any(
        now_ts - cur_tf.get(iv, {}).get("_fetched_at", 0) >= _TF_TTL.get(iv, 300)
        for iv, *_ in TIMEFRAMES
        if _OB_TF_WEIGHTS.get(iv, 0.0) > 0
    )

    ob_walls  = []
    ob_raw    = None
    if needs_ob:
        _rate_limiter.acquire(2)   # depth 接口权重较重，消耗 2 个令牌
        ob_walls, ob_raw = fetch_ob_walls(symbol)
        if ob_walls:
            print(f"[服务器] {symbol} OB 挂单墙: {len(ob_walls)} 个价格层")

    # ── 订单簿压力指数（基于同一次深度快照）─────────────────
    ob_pressure_now = None
    if ob_raw:
        ob_pressure_now = calculate_ob_pressure(
            ob_raw.get("bids", []), ob_raw.get("asks", [])
        )
        if ob_pressure_now:
            # 追加时间戳供前端显示
            ob_pressure_now["ts"] = datetime.now().strftime("%H:%M:%S")
            # 追加到历史队列（最多 5 个快照），用于后续速度计算
            _ob_pressure_history[symbol].append(ob_pressure_now["ratio"])
            # 速度：当前 ratio 与最近历史均值之差，正 = 买压增强，负 = 买压减弱
            hist = list(_ob_pressure_history[symbol])
            if len(hist) >= 2:
                ob_pressure_now["velocity"] = round(hist[-1] - sum(hist[:-1]) / len(hist[:-1]), 3)
            else:
                ob_pressure_now["velocity"] = 0.0
            print(f"[服务器] {symbol} OB压力指数: ratio={ob_pressure_now['ratio']} score={ob_pressure_now['score']} v={ob_pressure_now['velocity']}")

    # 拉取最近 24h 清算数据（先取 WS 真实数据，不足则用估算兜底）
    liq_clusters_now = get_liq_clusters(symbol, hours=24, top_n=10)
    if not liq_clusters_now:
        with _lock:
            tf_snap = _cache.get(symbol, {}).get("timeframes", {})
        liq_clusters_now = estimate_liq_clusters_from_sr(symbol, tf_snap)
    if liq_clusters_now:
        print(f"[服务器] {symbol} 清算密集区: {len(liq_clusters_now)} 个 (来源: {liq_clusters_now[0].get('source','?')})")

    for interval, limit, label, chart_show, sr_lb, sr_pn, sr_cluster_pct, fib_lb in TIMEFRAMES:
        cached = cur_tf.get(interval, {})
        age    = now_ts - cached.get("_fetched_at", 0)
        ttl    = _TF_TTL.get(interval, 300)

        if age < ttl:
            # 数据仍在有效期内，直接复用，不消耗 API 配额
            new_tf[interval] = cached
            skipped.append(interval)
            continue

        _rate_limiter.acquire(1)   # 每次 K 线请求消耗 1 个令牌
        try:
            result = run_single_tf(symbol, interval, limit, label,
                                   chart_show, sr_lb, sr_pn, sr_cluster_pct, fib_lb,
                                   ob_walls=ob_walls, liq_clusters=liq_clusters_now)
            if result:
                result["_fetched_at"] = now_ts
                new_tf[interval] = result
                fetched.append(interval)
            elif cached:
                new_tf[interval] = cached   # 失败时保留旧数据
        except Exception as e:
            print(f"[服务器] {symbol} {label} 分析失败: {e}")
            if cached:
                new_tf[interval] = cached

    if skipped:
        print(f"[服务器] {symbol} TTL 内跳过: {', '.join(skipped)}")
    if fetched:
        print(f"[服务器] {symbol} 已更新: {', '.join(fetched)}")

    # ── P0-B: MTF 多时间框架方向对齐过滤 ─────────────────────
    # 用日线+周线定大方向，低周期信号与大方向冲突时降低置信度
    _daily_rd  = (new_tf.get("1d") or {}).get("rd") or {}
    _weekly_rd = (new_tf.get("1w") or {}).get("rd") or {}
    _daily_score  = _daily_rd.get("score",  0) or 0
    _weekly_score = _weekly_rd.get("score", 0) or 0
    _htf_raw = _daily_score * 0.7 + _weekly_score * 0.3
    if   _htf_raw >=  4: _htf_dir = "bull"
    elif _htf_raw <= -4: _htf_dir = "bear"
    else:                _htf_dir = "neutral"

    _MTF_FILTER_SET = {"15m", "30m", "1h", "2h", "4h", "8h"}
    for _tf_interval, _tf_data in new_tf.items():
        if _tf_interval not in _MTF_FILTER_SET:
            continue
        _rd = _tf_data.get("rd")
        if not _rd:
            continue
        _ltf_score = _rd.get("score", 0) or 0
        _max_s     = _rd.get("max_score", 14) or 14
        _penalty   = 0
        _mtf_note  = ""

        if _htf_dir == "bull" and _ltf_score < -2:
            _penalty  = +2
            _mtf_note = f"⚠️ MTF逆向: 日线偏多(HTF={_htf_raw:+.1f})，空信号可靠性降低 {_penalty:+d}"
        elif _htf_dir == "bear" and _ltf_score > +2:
            _penalty  = -2
            _mtf_note = f"⚠️ MTF逆向: 日线偏空(HTF={_htf_raw:+.1f})，多信号可靠性降低 {_penalty:+d}"
        elif _htf_dir != "neutral":
            _sign = "↑" if _htf_dir == "bull" else "↓"
            _mtf_note = f"✅ MTF共振: 日线{_sign}(HTF={_htf_raw:+.1f})，信号方向一致"

        _rd["mtf_bias"] = {
            "htf_direction": _htf_dir,
            "htf_raw":       round(_htf_raw, 2),
            "daily_score":   _daily_score,
            "weekly_score":  _weekly_score,
            "penalty":       _penalty,
            "note":          _mtf_note,
        }

        if _penalty != 0:
            _adj = _ltf_score + _penalty
            _rd["score"] = _adj
            if   _adj >=  5: _rd["overall"] = f"偏多头  (得分: {_adj:+d}/{_max_s}) [MTF修正]"; _rd["overall_color"] = "#26a69a"; _rd["action"] = "做多 / 买入"
            elif _adj <= -5: _rd["overall"] = f"偏空头  (得分: {_adj:+d}/{_max_s}) [MTF修正]"; _rd["overall_color"] = "#ef5350"; _rd["action"] = "做空 / 卖出"
            else:            _rd["overall"] = f"中性震荡  (得分: {_adj:+d}/{_max_s}) [MTF修正]"; _rd["overall_color"] = "#f9a825"; _rd["action"] = "观望 / 高抛低吸"
            print(f"[MTF] {symbol} {_tf_interval}: 原分{_ltf_score:+d} → {_adj:+d} | {_mtf_note}")
            # 当HTF方向与LTF信号严重冲突时，降级trade_plan为观望
            _tp = _rd.get("trade_plan") or {}
            _tp_dir = _tp.get("direction")
            _is_conflict = ((_htf_dir == "bear" and _tp_dir == "long") or
                            (_htf_dir == "bull" and _tp_dir == "short"))
            if _is_conflict:
                _rd["trade_plan"] = {
                    **_tp,
                    "recommended": False,
                    "htf_conflict": True,
                    "htf_conflict_note": (
                        f"⚠️ HTF过滤: 大周期方向{'偏空' if _htf_dir=='bear' else '偏多'}(日线{_daily_score:+d}/周线{_weekly_score:+d})，"
                        f"当前{_tf_interval}{'做多' if _tp_dir=='long' else '做空'}信号属逆势操作，胜率大幅降低"
                    ),
                }

    now = datetime.now()
    with _lock:
        _cache[symbol]["timeframes"]  = new_tf
        _cache[symbol]["last_update"] = now.strftime("%Y-%m-%d %H:%M:%S")
        _cache[symbol]["updating"]    = False
        _cache[symbol]["ready"]       = True
        if ob_pressure_now is not None:
            _cache[symbol]["ob_pressure"] = ob_pressure_now

    print(f"[服务器] {symbol} 完成")


def analysis_worker():
    global _update_counter
    from concurrent.futures import ThreadPoolExecutor

    while True:
        print(f"\n[服务器] ===== 开始更新分析数据 {datetime.now().strftime('%H:%M:%S')} =====")

        # 并行分析所有币种（各币种独立无依赖，适合 I/O 密集型线程并行）
        with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
            list(executor.map(_analyze_symbol, SYMBOLS))

        # 全量完成后统一打版本戳 & next_update，确保跨币种时间戳一致
        _update_counter += 1
        now    = datetime.now()
        next_t = now + timedelta(seconds=REFRESH_INTERVAL)
        with _lock:
            for symbol in SYMBOLS:
                _cache[symbol]["update_id"]   = _update_counter
                _cache[symbol]["next_update"] = next_t.strftime("%Y-%m-%d %H:%M:%S")

        print(f"[服务器] 全部完成 (update_id={_update_counter})，下次更新: {next_t.strftime('%H:%M:%S')}\n")
        time.sleep(REFRESH_INTERVAL)


# ─────────────────────────────────────────────
# Flask 路由
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    symbol = request.args.get("symbol", "BTCUSDT")
    if symbol not in _cache:
        return jsonify({"error": "Invalid symbol"}), 400
        
    with _lock:
        data = _cache[symbol]
        now    = datetime.now()
        next_t = data.get("next_update")
        if next_t:
            try:
                ndt       = datetime.strptime(next_t, "%Y-%m-%d %H:%M:%S")
                remaining = max(0, int((ndt - now).total_seconds()))
            except Exception:
                remaining = REFRESH_INTERVAL
        else:
            remaining = REFRESH_INTERVAL
        return jsonify({
            "symbol":            symbol,
            "last_update":       data.get("last_update"),
            "next_update":       data.get("next_update"),
            "seconds_remaining": remaining,
            "updating":          data.get("updating", False),
            "ready":             data.get("ready", False),
            "update_id":         data.get("update_id", 0),
        })


@app.route("/api/analysis")
def analysis():
    """返回所有周期的分析摘要（不含大体积 K 线数据）"""
    symbol = request.args.get("symbol", "BTCUSDT")
    if symbol not in _cache:
        return jsonify({"error": "Invalid symbol"}), 400

    with _lock:
        data = _cache[symbol]
        result = {}
        for tf_key, tf_data in data["timeframes"].items():
            result[tf_key] = {
                "rd":               tf_data["rd"],
                "supports":         tf_data["supports"],
                "resistances":      tf_data["resistances"],
                "fib":              tf_data["fib"],
                "fvg_zones":        tf_data.get("fvg_zones",       []),
                "volume_profile":   {k: v for k, v in tf_data.get("volume_profile", {}).items() if k != "chart_bins"},
                "market_structure": tf_data.get("market_structure",{}),
            }
        return jsonify({
            "symbol":      symbol,
            "ready":       data["ready"],
            "last_update": data["last_update"],
            "update_id":   data.get("update_id", 0),
            "ob_pressure": data.get("ob_pressure"),
            "timeframes":  result,
        })


@app.route("/api/klines/<interval>")
def klines(interval):
    """返回指定周期的 K 线 + 指标数据"""
    symbol = request.args.get("symbol", "BTCUSDT")
    if symbol not in _cache:
        return jsonify({"error": "Invalid symbol"}), 400

    with _lock:
        data = _cache[symbol]
        tf_data = data["timeframes"].get(interval)
        if not tf_data:
            return jsonify({"error": "数据未就绪，请稍候..."}), 404
        return jsonify({
            "candles":          tf_data["candles"],
            "supports":         tf_data["supports"],
            "resistances":      tf_data["resistances"],
            "fib":              tf_data["fib"],
            "fvg_zones":        tf_data.get("fvg_zones",       []),
            "volume_profile":   tf_data.get("volume_profile",  {}),
            "market_structure": tf_data.get("market_structure",{}),
        })


@app.route("/api/footprint/<interval>")
def footprint(interval):
    """获取最近 K 线的踏印数据（逐笔主动买卖量，按价格档位分桶）"""
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    if symbol not in _cache:
        return jsonify({"error": "Invalid symbol"}), 400

    with _lock:
        tf_data = _cache[symbol]["timeframes"].get(interval)
    if not tf_data:
        return jsonify({"error": "数据未就绪，请稍候..."}), 404

    candles = tf_data.get("candles", [])
    if not candles:
        return jsonify({"error": "无K线数据"}), 404

    last = candles[-1]
    TF_MS = {
        "15m": 900_000,   "30m": 1_800_000,  "1h":  3_600_000,
        "2h":  7_200_000, "4h":  14_400_000, "8h":  28_800_000,
        "1d":  86_400_000,"1w":  604_800_000,
    }
    try:
        t0       = datetime.strptime(last["t"], "%Y-%m-%dT%H:%M")
        start_ms = int(t0.timestamp() * 1000)
        end_ms   = start_ms + TF_MS.get(interval, 3_600_000) - 1
    except Exception as e:
        return jsonify({"error": f"时间解析失败: {e}"}), 500

    _rate_limiter.acquire(2)
    try:
        r = req.get(
            "https://api.binance.com/api/v3/aggTrades",
            params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000},
            timeout=15,
        )
        trades = r.json()
        if not isinstance(trades, list):
            return jsonify({"error": "API 返回异常", "raw": trades}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 动态档位大小（根据价格量级）
    price = float(last["c"])
    if   price > 50_000: bucket = 200.0
    elif price > 10_000: bucket = 100.0
    elif price >  1_000: bucket = 10.0
    elif price >    100: bucket = 1.0
    elif price >     10: bucket = 0.1
    else:                bucket = 0.01

    from collections import defaultdict
    buckets = defaultdict(lambda: {"buy": 0.0, "sell": 0.0})
    for tr in trades:
        p = round(float(tr["p"]) / bucket) * bucket
        q = float(tr["q"])
        if tr["m"]:    # isBuyerMaker=True → 主动卖方（卖方吃单）
            buckets[p]["sell"] += q
        else:          # 主动买方（买方吃单）
            buckets[p]["buy"] += q

    result = sorted([
        {"price": k,
         "buy":   round(v["buy"],  4),
         "sell":  round(v["sell"], 4),
         "delta": round(v["buy"] - v["sell"], 4)}
        for k, v in buckets.items()
    ], key=lambda x: x["price"], reverse=True)

    poc_price = max(result, key=lambda x: x["buy"] + x["sell"])["price"] if result else price

    return jsonify({
        "symbol":        symbol,
        "interval":      interval,
        "candle_time":   last["t"],
        "current_price": price,
        "bucket_size":   bucket,
        "total_trades":  len(trades),
        "poc_price":     poc_price,
        "buckets":       result,
    })


@app.route("/api/price")
def live_price():
    """实时价格 + 24h 统计"""
    symbol = request.args.get("symbol", "BTCUSDT")
    try:
        r = req.get("https://api.binance.com/api/v3/ticker/24hr",
                    params={"symbol": symbol}, timeout=5)
        d = r.json()
        return jsonify({
            "symbol":      d["symbol"],
            "price":       float(d["lastPrice"]),
            "change_pct":  float(d["priceChangePercent"]),
            "high24h":     float(d["highPrice"]),
            "low24h":      float(d["lowPrice"]),
            "volume24h":   float(d["quoteVolume"]), # USDT volume
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _analyze_oi(oi_history: list) -> dict:
    """分析持仓量趋势：上升/下降幅度，用于判断突破真伪"""
    if not oi_history or len(oi_history) < 5:
        return {"trend": "unknown", "change_pct": 0.0, "signal": "数据不足"}
    ois  = [o["oi"] for o in oi_history]
    half = max(1, len(ois) // 2)
    recent_avg = sum(ois[-half:]) / half
    prev_avg   = sum(ois[:half])  / half
    change_pct = (recent_avg - prev_avg) / prev_avg * 100 if prev_avg else 0
    if   change_pct >=  5: trend, signal = "快速上升", "大量新仓入场"
    elif change_pct >=  2: trend, signal = "温和上升", "持仓稳步增加"
    elif change_pct <= -5: trend, signal = "快速下降", "大量旧仓平仓"
    elif change_pct <= -2: trend, signal = "温和下降", "持仓缓慢减少"
    else:                  trend, signal = "平稳",     "持仓量无明显变化"
    return {
        "trend":      trend,
        "signal":     signal,
        "change_pct": round(change_pct, 2),
        "recent_avg": round(recent_avg, 0),
        "oi_max":     round(max(ois), 0),
        "oi_min":     round(min(ois), 0),
    }


def _cluster_liquidations(orders: list, bucket_size: float = 500.0) -> list:
    """将强平订单按价格分桶，返回 Top-5 清算密集区"""
    buckets = collections.defaultdict(lambda: {"qty": 0.0, "count": 0, "long_qty": 0.0, "short_qty": 0.0})
    for o in orders:
        try:
            price = float(o.get("averagePrice") or o.get("ap") or o.get("price", 0))
            qty   = float(o.get("executedQty") or o.get("z") or o.get("q", 0))
            side  = o.get("side") or o.get("S", "")
            key   = round(price / bucket_size) * bucket_size
            buckets[key]["qty"]   += qty
            buckets[key]["count"] += 1
            if side == "SELL": buckets[key]["long_qty"]  += qty  # 多头被爆仓
            else:              buckets[key]["short_qty"] += qty  # 空头被爆仓
        except Exception:
            continue
    if not buckets:
        return []
    sorted_b = sorted(buckets.items(), key=lambda x: x[1]["qty"], reverse=True)[:5]
    return [
        {"price":     price,
         "qty":       round(b["qty"], 2),
         "count":     b["count"],
         "long_qty":  round(b["long_qty"], 2),
         "short_qty": round(b["short_qty"], 2),
         "dominant":  "多头清算" if b["long_qty"] >= b["short_qty"] else "空头清算"}
        for price, b in sorted_b
    ]


# ─────────────────────────────────────────────
# 清算 WebSocket 累积器
# ─────────────────────────────────────────────

# 存储结构: {symbol: deque of {price, qty, side, ts}}
# 保留最近 48 小时内的清算记录
_LIQ_WINDOW_HOURS  = 48
_LIQ_PERSIST_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liq_data.json")
_LIQ_FLUSH_INTERVAL = 300   # 每 5 分钟落盘一次（秒）

_liq_store: dict = {s: collections.deque(maxlen=50_000) for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]}
_liq_lock  = threading.Lock()
_liq_ws_running = False


# ── 持久化：加载 ────────────────────────────────────────────────────────────
def _liq_load():
    """启动时从 liq_data.json 恢复历史清算数据，自动丢弃 48h 外的过期记录。"""
    if not os.path.exists(_LIQ_PERSIST_FILE):
        print("[清算持久化] 未找到 liq_data.json，跳过恢复")
        return
    try:
        with open(_LIQ_PERSIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cutoff  = time.time() - _LIQ_WINDOW_HOURS * 3600
        total   = 0
        with _liq_lock:
            for sym, records in data.items():
                if sym not in _liq_store:
                    continue
                valid = [r for r in records if r.get("ts", 0) >= cutoff]
                _liq_store[sym].extend(valid)
                total += len(valid)
        print(f"[清算持久化] 已恢复 {total} 条记录（来自 {_LIQ_PERSIST_FILE}）")
    except Exception as e:
        print(f"[清算持久化] 恢复失败: {e}")


# ── 持久化：落盘 ────────────────────────────────────────────────────────────
def _liq_flush():
    """将内存中的清算数据序列化写入 liq_data.json（原子写：先写临时文件再重命名）。"""
    tmp = _LIQ_PERSIST_FILE + ".tmp"
    try:
        cutoff = time.time() - _LIQ_WINDOW_HOURS * 3600
        with _liq_lock:
            payload = {
                sym: [r for r in dq if r["ts"] >= cutoff]
                for sym, dq in _liq_store.items()
            }
        total = sum(len(v) for v in payload.values())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))   # compact，节省空间
        os.replace(tmp, _LIQ_PERSIST_FILE)                  # 原子替换，防止写到一半崩溃
        return total
    except Exception as e:
        print(f"[清算持久化] 落盘失败: {e}")
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
        return 0


# ── 持久化：定时落盘线程 ────────────────────────────────────────────────────
def _liq_flush_worker():
    """每 _LIQ_FLUSH_INTERVAL 秒将清算数据落盘一次。"""
    while True:
        time.sleep(_LIQ_FLUSH_INTERVAL)
        total = _liq_flush()
        if total:
            print(f"[清算持久化] 落盘完成，共 {total} 条")


def _liq_ws_worker():
    """后台线程：订阅 Binance 全市场强平推流，累积到本地 _liq_store。
    自动重连，异常后等待 5s 重试。
    """
    global _liq_ws_running
    _liq_ws_running = True

    # 懒加载 websocket-client，避免强依赖
    try:
        import websocket as _ws_lib
    except ImportError:
        print("[清算WS] 未安装 websocket-client，请运行: pip install websocket-client")
        print("[清算WS] 清算密集区将使用 OI 估算方案代替")
        _liq_ws_running = False
        return

    url = "wss://fstream.binance.com/ws/!forceOrder@arr"
    TRACKED = set(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])

    def on_message(ws, raw):
        try:
            msg = json.loads(raw)
            # 支持单条 {"e":"forceOrder",...} 或数组
            events = msg if isinstance(msg, list) else [msg]
            now_ts = time.time()
            cutoff  = now_ts - _LIQ_WINDOW_HOURS * 3600

            with _liq_lock:
                for ev in events:
                    o = ev.get("o") or ev  # 嵌套格式 {"e":"forceOrder","o":{...}}
                    sym  = o.get("s", "")
                    if sym not in TRACKED:
                        continue
                    try:
                        price = float(o.get("ap") or o.get("p", 0))   # 成交均价优先
                        qty   = float(o.get("z") or o.get("q", 0))    # 已成交量
                        side  = o.get("S", "")                         # BUY/SELL
                        ts    = int(o.get("T", now_ts * 1000)) / 1000
                        if price <= 0 or qty <= 0:
                            continue
                        # 修剪过期数据（每次写入时顺便淘汰头部过期项）
                        dq = _liq_store[sym]
                        while dq and dq[0]["ts"] < cutoff:
                            dq.popleft()
                        dq.append({"price": price, "qty": qty, "side": side, "ts": ts})
                    except Exception:
                        pass
        except Exception as e:
            print(f"[清算WS] 消息解析异常: {e}")

    def on_error(ws, err):
        print(f"[清算WS] 错误: {err}")

    def on_close(ws, code, msg):
        print(f"[清算WS] 连接断开 ({code}), 5s 后重连...")

    def on_open(ws):
        print("[清算WS] 已连接 Binance 全市场强平推流")

    while True:
        try:
            ws = _ws_lib.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            print(f"[清算WS] 连接异常: {e}")
        time.sleep(5)


def get_liq_clusters(symbol: str, hours: int = 24, top_n: int = 7) -> list:
    """
    从本地累积的清算数据中提取密集区。
    hours: 只看最近 N 小时的清算
    top_n: 返回 Top N 价格区间
    返回: [{"price", "qty", "count", "long_qty", "short_qty", "dominant", "source"}, ...]
    """
    cutoff = time.time() - hours * 3600
    with _liq_lock:
        records = [r for r in _liq_store.get(symbol, []) if r["ts"] >= cutoff]

    if not records:
        return []

    # 动态档位：按价格量级自动选 bucket_size
    prices = [r["price"] for r in records]
    avg_price = sum(prices) / len(prices)
    if   avg_price > 50_000: bucket = 500.0
    elif avg_price > 10_000: bucket = 200.0
    elif avg_price >  1_000: bucket = 20.0
    elif avg_price >    100: bucket = 2.0
    else:                    bucket = 0.5

    buckets = collections.defaultdict(lambda: {"qty": 0.0, "count": 0, "long_qty": 0.0, "short_qty": 0.0})
    for r in records:
        key = round(r["price"] / bucket) * bucket
        buckets[key]["qty"]   += r["qty"]
        buckets[key]["count"] += 1
        if r["side"] == "SELL":
            buckets[key]["long_qty"]  += r["qty"]   # SELL 方向 = 多头被清算
        else:
            buckets[key]["short_qty"] += r["qty"]   # BUY  方向 = 空头被清算

    sorted_b = sorted(buckets.items(), key=lambda x: x[1]["qty"], reverse=True)[:top_n]
    return [
        {"price":     p,
         "qty":       round(b["qty"], 4),
         "count":     b["count"],
         "long_qty":  round(b["long_qty"], 4),
         "short_qty": round(b["short_qty"], 4),
         "dominant":  "多头清算" if b["long_qty"] >= b["short_qty"] else "空头清算",
         "source":    "websocket"}
        for p, b in sorted_b
    ]


def estimate_liq_clusters_from_sr(symbol: str, tf_data: dict) -> list:
    """
    方案③：当 WebSocket 数据不足时，用 OI + S/R + ATR 估算潜在清算区。

    逻辑：
    - 多头清算区 = 关键支撑位下方 1~1.5× ATR（多头密集止损/强平区）
    - 空头清算区 = 关键阻力位上方 1~1.5× ATR（空头密集止损/强平区）
    """
    results = []
    try:
        # 优先用 1h 周期的数据（平衡精度和稳定性）
        tf = tf_data.get("1h") or tf_data.get("4h") or {}
        rd = tf.get("rd", {})
        atr_info = rd.get("atr", {})
        atr_val  = atr_info.get("val", 0)
        if not atr_val:
            return []

        current_price = rd.get("price", 0)
        if not current_price:
            return []

        supports    = tf.get("supports", [])
        resistances = tf.get("resistances", [])

        # 多头清算区：支撑位下方 1.0~1.5 ATR（多头止损密集）
        for sup in supports[:3]:
            sup_price = sup.get("price", 0)
            if not sup_price:
                continue
            liq_price = round(sup_price - 1.2 * atr_val, 2)
            dist_pct  = round((current_price - liq_price) / current_price * 100, 2)
            results.append({
                "price":     liq_price,
                "qty":       0,
                "count":     0,
                "long_qty":  0,
                "short_qty": 0,
                "dominant":  "多头清算(估算)",
                "source":    "estimated",
                "basis":     f"支撑 ${sup_price:,.0f} 下方 1.2×ATR",
                "dist_pct":  dist_pct,
            })

        # 空头清算区：阻力位上方 1.0~1.5 ATR（空头止损密集）
        for res in resistances[:3]:
            res_price = res.get("price", 0)
            if not res_price:
                continue
            liq_price = round(res_price + 1.2 * atr_val, 2)
            dist_pct  = round((liq_price - current_price) / current_price * 100, 2)
            results.append({
                "price":     liq_price,
                "qty":       0,
                "count":     0,
                "long_qty":  0,
                "short_qty": 0,
                "dominant":  "空头清算(估算)",
                "source":    "estimated",
                "basis":     f"阻力 ${res_price:,.0f} 上方 1.2×ATR",
                "dist_pct":  dist_pct,
            })

    except Exception as e:
        print(f"[清算估算] {symbol} 异常: {e}")

    return results


def get_liq_stats(symbol: str) -> dict:
    """返回 WebSocket 累积器的统计信息（供前端状态展示）"""
    with _liq_lock:
        dq = _liq_store.get(symbol, collections.deque())
        total = len(dq)
        if total == 0:
            return {"total": 0, "hours_covered": 0, "ws_running": _liq_ws_running}
        oldest = dq[0]["ts"]
        hours  = round((time.time() - oldest) / 3600, 1)
    return {"total": total, "hours_covered": hours, "ws_running": _liq_ws_running}


def _analyze_funding(rate_pct: float, history: list) -> dict:
    """
    从资深交易员视角解读资金费率。
    资金费率是逆向情绪指标：
      极度正值 → 多头过拥挤 → 警惕多杀多（逆向看空）
      极度负值 → 空头过拥挤 → 警惕空杀空（逆向看多）
    """
    rates = [h["rate"] for h in history] if history else [rate_pct]

    # 百分位（当前费率在历史分布中的位置）
    if len(rates) > 1:
        percentile = round(sum(1 for r in rates if r <= rate_pct) / len(rates) * 100, 1)
    else:
        percentile = 50.0

    avg_nd = round(sum(rates) / len(rates), 4)
    days   = round(len(rates) * 8 / 24, 0)   # 实际覆盖天数

    # 趋势：最近10期均值 vs 前10期均值
    if len(rates) >= 20:
        recent  = sum(rates[-10:]) / 10
        earlier = sum(rates[-20:-10]) / 10
        if   recent > earlier + 0.0005: trend = "上升"
        elif recent < earlier - 0.0005: trend = "下降"
        else:                           trend = "平稳"
    else:
        trend = "数据不足"

    # 信号 + 逆向逻辑解读
    if rate_pct >= 0.10:
        signal = "极度偏多"
        score  = -3   # 逆向：多头严重过热 → 情绪面偏空
        color  = "#ef5350"
        interp = "多头极度拥挤，历史上此水平后72h内常见5%-15%回调，警惕多杀多瀑布"
        action = "谨慎追多，建议等待费率回落至0.05%以下再介入；持仓者考虑减仓对冲"
    elif rate_pct >= 0.05:
        signal = "偏多"
        score  = -1
        color  = "#ff9800"
        interp = "多头情绪偏热，短期回调风险上升，但强势行情中费率可维持高位"
        action = "控制杠杆，止损上移保护利润；不建议此时重仓追多"
    elif rate_pct <= -0.10:
        signal = "极度偏空"
        score  = +3   # 逆向：空头严重过热 → 情绪面偏多
        color  = "#26a69a"
        interp = "空头极度拥挤，历史上此水平后常出现急速反弹，空头回补可推动10%+涨幅"
        action = "关注反弹做多机会，但需结合支撑位确认；避免盲目追空"
    elif rate_pct <= -0.05:
        signal = "偏空"
        score  = +1
        color  = "#42a5f5"
        interp = "空头情绪偏高，价格下行动能可能减弱，留意超卖反弹"
        action = "空单注意止盈，关注能否出现价格背离信号"
    elif -0.01 <= rate_pct <= 0.01:
        signal = "中性"
        score  = 0
        color  = "#9e9e9e"
        interp = "多空力量均衡，费率接近零值是最健康的市场状态，趋势可持续性较强"
        action = "跟随主趋势操作，费率不构成额外阻力"
    elif rate_pct > 0:
        signal = "温和偏多"
        score  = 0
        color  = "#fdd835"
        interp = "多头略占优，属正常牛市状态，暂无过热警示"
        action = "正常持仓，关注费率是否持续攀升"
    else:
        signal = "温和偏空"
        score  = 0
        color  = "#fdd835"
        interp = "空头略占优，属正常回调状态，暂无过冷警示"
        action = "轻仓观望，关注支撑位是否有效"

    return {
        "signal":     signal,
        "score":      score,
        "color":      color,
        "interp":     interp,
        "action":     action,
        "percentile": percentile,
        "avg_nd":     avg_nd,     # 实际取样周期均值（约166天）
        "days":       days,       # 实际覆盖天数（供前端标签）
        "trend":      trend,
        "extreme":    abs(rate_pct) >= 0.05,
    }


def _analyze_fr_advanced(historical: list, oi_history: list) -> dict:
    """
    专业级资金费率深度分析（四维）：
      1. FR 动量      ── 连续走向比单期数值更危险
      2. FR × OI复合 ── 双指标共振才是真实信号
      3. FR-价格时序  ── FR 领先价格 4-8h 的逆向机会
    """
    if not historical or len(historical) < 5:
        return {}

    rates = [h["rate"] for h in historical]

    # ── 1. FR 动量：连续同向计数 + 斜率 ─────────────────────────────────
    slope_3  = (rates[-1] - rates[-4]) / 3  if len(rates) >= 4  else 0.0
    slope_10 = (rates[-1] - rates[-11]) / 10 if len(rates) >= 11 else slope_3

    consec_up = consec_down = 0
    for i in range(len(rates) - 1, 0, -1):
        d = rates[i] - rates[i - 1]
        if   d > 0  and consec_down == 0: consec_up   += 1
        elif d < 0  and consec_up   == 0: consec_down += 1
        else: break
        if consec_up + consec_down >= 10: break

    is_accel = abs(slope_3) > abs(slope_10) * 1.5   # 近期斜率加速

    if   slope_3 >= 0.015 and consec_up >= 3:
        mom_label, mom_score = "🔴 快速上升", -2
        mom_desc = (f"FR已连续 {consec_up} 期走高（3期斜率 {slope_3:+.4f}%/期），"
                    f"多头仓位正在快速积累，{'加速上升！' if is_accel else ''}"
                    f"类2021年5月前夕特征，清算瀑布风险极高")
    elif slope_3 >= 0.008 or consec_up >= 3:
        mom_label, mom_score = "🟡 缓慢上升", -1
        mom_desc = (f"FR趋势向上（3期斜率 {slope_3:+.4f}%/期，连续{consec_up}期），"
                    f"多头情绪升温，注意是否持续加速")
    elif slope_3 <= -0.015 and consec_down >= 3:
        mom_label, mom_score = "🟢 快速下降", +2
        mom_desc = (f"FR已连续 {consec_down} 期走低（3期斜率 {slope_3:+.4f}%/期），"
                    f"空头持仓成本快速攀升，{'加速下行！' if is_accel else ''}"
                    f"轧空动能积聚，警惕空头止损连锁反应")
    elif slope_3 <= -0.008 or consec_down >= 3:
        mom_label, mom_score = "🟡 缓慢下降", +1
        mom_desc = (f"FR趋势向下（3期斜率 {slope_3:+.4f}%/期，连续{consec_down}期），"
                    f"空头情绪升温，关注超卖反弹时机")
    else:
        mom_label, mom_score = "⚪ 平稳", 0
        mom_desc = (f"FR变化平缓（3期斜率 {slope_3:+.4f}%/期），"
                    f"多空力量均衡，无动量信号")

    momentum = {
        "slope_3p":    round(slope_3,  5),
        "slope_10p":   round(slope_10, 5),
        "consec_up":   consec_up,
        "consec_down": consec_down,
        "is_accel":    is_accel,
        "label":       mom_label,
        "score":       mom_score,
        "desc":        mom_desc,
    }

    # ── 2. FR × OI 复合指标 ───────────────────────────────────────────
    fr_oi = None
    if oi_history and len(oi_history) >= 6:
        ois = [o["oi"] for o in oi_history]
        oi_recent = sum(ois[-3:]) / 3
        oi_prev   = sum(ois[-6:-3]) / 3 if len(ois) >= 6 else ois[0]
        oi_chg    = (oi_recent - oi_prev) / oi_prev * 100 if oi_prev else 0

        fr_up = slope_3 >  0.005
        fr_dn = slope_3 < -0.005
        oi_up = oi_chg  >  2.0
        oi_dn = oi_chg  < -2.0

        if fr_up and oi_up:
            foi_sig, foi_risk, foi_score = "⚠️ 多头过度拥挤", "极高", -2
            foi_desc = (f"FR↑({slope_3:+.4f}%/期) × OI↑(+{oi_chg:.1f}%) = "
                        f"多头仓位急速堆积，清算瀑布风险极高；"
                        f"需同时满足：FR下穿0.05% + OI开始下降，方可考虑做多")
        elif fr_dn and oi_dn:
            foi_sig, foi_risk, foi_score = "✅ 空头快速离场", "低(空方)", +2
            foi_desc = (f"FR↓({slope_3:+.4f}%/期) × OI↓({oi_chg:.1f}%) = "
                        f"空头止盈/止损回补离场，回补需求释放，"
                        f"短期价格支撑显著增强")
        elif fr_up and oi_dn:
            foi_sig, foi_risk, foi_score = "⚠️ 轧空行情(不可持续)", "中(回调)", 0
            foi_desc = (f"FR↑ × OI↓ = 价格上涨由空头止损回补推动（轧空），"
                        f"非真实新多头入场，突破可信度低，警惕拉高后迅速回落")
        elif fr_dn and oi_up:
            foi_sig, foi_risk, foi_score = "🔴 空头新仓主导", "高(下行)", -1
            foi_desc = (f"FR↓ × OI↑(+{oi_chg:.1f}%) = "
                        f"空头新开仓信心足，若关键支撑位破位将加速下跌")
        else:
            foi_sig, foi_risk, foi_score = "中性", "低", 0
            foi_desc = (f"FR({slope_3:+.4f}%/期) 与 OI({oi_chg:+.1f}%) "
                        f"无明显共振方向，无复合信号")

        fr_oi = {
            "fr_slope_3p": round(slope_3, 5),
            "oi_change":   round(oi_chg,  2),
            "signal":      foi_sig,
            "risk_level":  foi_risk,
            "score":       foi_score,
            "desc":        foi_desc,
        }

    # ── 3. FR-价格时序关系（FR 领先价格 4-8h）─────────────────────────
    timing = None
    if oi_history and len(oi_history) >= 3 and len(rates) >= 3:
        fr_8h_chg  = rates[-1] - rates[-3]      # 近8h（2个8h结算期）FR变化
        ois2       = [o["oi"] for o in oi_history]
        oi_8h_chg  = (ois2[-1] - ois2[-3]) / ois2[-3] * 100 if ois2[-3] else 0

        if fr_8h_chg > 0.02 and oi_8h_chg < -1:
            t_sig, t_score = "✅ 空头轧仓反弹预警", +2
            t_desc = (f"近8h FR突升 +{fr_8h_chg:.4f}% + OI下降 {oi_8h_chg:.1f}% = "
                      f"空头止损推动价格拉升（轧空），FR领先效应：预计价格4-8h内跟随强弹")
        elif fr_8h_chg < -0.02 and oi_8h_chg < -1:
            t_sig, t_score = "🔴 多头清算传导预警", -2
            t_desc = (f"近8h FR急降 {fr_8h_chg:.4f}% + OI下降 {oi_8h_chg:.1f}% = "
                      f"多头止损加速，FR先行下行，价格可能4-8h内跟随补跌")
        elif fr_8h_chg < -0.015 and oi_8h_chg > 3:
            t_sig, t_score = "🔴 价跌空头入场确认", -1
            t_desc = (f"近8h FR走低 {fr_8h_chg:.4f}% + OI上升 +{oi_8h_chg:.1f}% = "
                      f"新空仓快速进场，下跌动能有支撑，FR领先表明抛压尚未全部释放")
        elif fr_8h_chg > 0.015 and oi_8h_chg > 3:
            t_sig, t_score = "⚠️ 多头加速涌入预警", -1
            t_desc = (f"近8h FR走高 +{fr_8h_chg:.4f}% + OI上升 +{oi_8h_chg:.1f}% = "
                      f"新多仓快速堆积，短期或继续上涨，但拥挤度快速上升警惕急转")
        else:
            t_sig, t_score = "无明显时序信号", 0
            t_desc = (f"近8h FR变化 {fr_8h_chg:+.4f}%，OI变化 {oi_8h_chg:+.1f}%，"
                      f"时序信号强度不足")

        timing = {
            "fr_8h_change":  round(fr_8h_chg, 5),
            "oi_8h_change":  round(oi_8h_chg, 2),
            "signal":        t_sig,
            "score":         t_score,
            "desc":          t_desc,
        }

    return {
        "momentum": momentum,
        "fr_oi":    fr_oi,
        "timing":   timing,
    }


def _fetch_basis(symbol: str) -> dict:
    """
    获取季度合约期现价差（Basis）。
    Basis = (季度合约价格 - 永续合约价格) / 永续价格 × 100%
    年化 Basis = Basis × 365 / 剩余天数
    正溢价 → 机构看涨远期；负溢价（贴水）→ 机构看空远期。
    """
    base = symbol.upper().replace("USDT", "")
    try:
        info_r = req.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=8)
        syms   = info_r.json().get("symbols", [])
        quarterly = [
            s for s in syms
            if s.get("baseAsset") == base
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") in ("CURRENT_QUARTER", "NEXT_QUARTER")
            and s.get("status") == "TRADING"
        ]
        if not quarterly:
            return {}
        quarterly.sort(key=lambda s: s.get("deliveryDate", 0))
        nearest   = quarterly[0]
        q_symbol  = nearest["symbol"]
        expiry_ms = int(nearest.get("deliveryDate", 0))

        # 季度合约当前价
        qp_r     = req.get("https://fapi.binance.com/fapi/v1/ticker/price",
                            params={"symbol": q_symbol}, timeout=5)
        q_price  = float(qp_r.json()["price"])

        # 永续合约当前价（作为现货代理）
        pp_r     = req.get("https://fapi.binance.com/fapi/v1/ticker/price",
                            params={"symbol": symbol}, timeout=5)
        perp_p   = float(pp_r.json()["price"])

        basis_pct    = (q_price - perp_p) / perp_p * 100
        days_left    = max(1, (expiry_ms - int(time.time() * 1000)) / (1000 * 86400))
        annualized   = basis_pct * 365 / days_left

        if   basis_pct >  0.5: sig = "溢价偏高（机构看涨远期）"
        elif basis_pct >  0.1: sig = "正常溢价（偏多）"
        elif basis_pct < -0.5: sig = "贴水偏深（机构看空远期）"
        elif basis_pct < -0.1: sig = "轻微贴水（偏空）"
        else:                  sig = "期现平价（中性）"

        return {
            "quarterly_symbol": q_symbol,
            "quarterly_price":  round(q_price, 2),
            "perp_price":       round(perp_p, 2),
            "basis_pct":        round(basis_pct, 4),
            "annualized_pct":   round(annualized, 2),
            "days_to_expiry":   round(days_left, 1),
            "signal":           sig,
            "desc": (f"{q_symbol}（剩余 {days_left:.0f}天）vs {symbol} 永续："
                     f" 基差 {basis_pct:+.4f}%（年化 {annualized:+.2f}%）"),
        }
    except Exception as e:
        print(f"[Basis] {symbol} 获取失败: {e}")
        return {}


@app.route("/api/funding")
def funding_data():
    """资金费率 + 持仓量（合约市场实时数据，含历史 + 情绪分析）"""
    symbol = request.args.get("symbol", "BTCUSDT")
    result = {
        "symbol":               symbol,
        "funding_rate":         None,
        "next_funding_time":    None,
        "open_interest":        None,
        "open_interest_usdt":   None,
        "historical":           [],   # 最近500期历史费率（≈166天）
        "oi_history":           [],   # 持仓量历史
        "analysis":             None, # 情绪分析结论
        "oi_analysis":          None, # 持仓量趋势分析
        "liquidation_clusters": [],   # 清算密集区 Top-5
        "fr_advanced":          {},   # 专业级 FR 深度分析
        "basis":                {},   # 期现价差（季度合约 Basis）
    }

    # ── 历史资金费率（最近 500 期 ≈ 166 天，API 单次上限 1000）────────────────
    try:
        fr_r = req.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 500},
            timeout=8,
        )
        fr_list = fr_r.json()
        if isinstance(fr_list, list) and fr_list:
            historical = [
                {"time": int(x["fundingTime"]),
                 "rate": round(float(x["fundingRate"]) * 100, 4)}
                for x in fr_list
            ]
            result["historical"]        = historical
            latest                      = historical[-1]
            result["funding_rate"]      = latest["rate"]
            result["next_funding_time"] = fr_list[-1].get("fundingTime")
            result["analysis"]          = _analyze_funding(latest["rate"], historical)
    except Exception:
        pass

    # ── 合约持仓量（当前）──────────────────────────────
    try:
        oi_r = req.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=5,
        )
        oi_data = oi_r.json()
        if "openInterest" in oi_data:
            result["open_interest"] = float(oi_data["openInterest"])
    except Exception:
        pass

    # ── 持仓量历史（4h 周期，最近 30 期）────────────────
    try:
        oih_r = req.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "4h", "limit": 30},
            timeout=8,
        )
        oih_list = oih_r.json()
        if isinstance(oih_list, list):
            result["oi_history"] = [
                {"time": int(x["timestamp"]),
                 "oi":   round(float(x["sumOpenInterest"]), 2)}
                for x in oih_list
            ]
    except Exception:
        pass

    # ── 持仓价值（USDT）————使用 ticker 推算 ─────────────
    try:
        if result["open_interest"] is not None:
            tk_r = req.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": symbol},
                timeout=5,
            )
            mark_price = float(tk_r.json().get("price", 0))
            if mark_price > 0:
                result["open_interest_usdt"] = round(
                    result["open_interest"] * mark_price, 2
                )
    except Exception:
        pass

    # ── OI 趋势分析 ──────────────────────────────────────
    if result["oi_history"]:
        result["oi_analysis"] = _analyze_oi(result["oi_history"])

    # ── 专业级 FR 深度分析（动量 / FR×OI / 时序）────────────────────────
    result["fr_advanced"] = _analyze_fr_advanced(
        result["historical"], result["oi_history"]
    )

    # ── 期现价差（季度合约 Basis）────────────────────────────────────────
    result["basis"] = _fetch_basis(symbol)

    # ── 清算密集区（双源：WebSocket 实时累积 + OI估算兜底）────────────
    ws_clusters = get_liq_clusters(symbol, hours=24, top_n=7)
    liq_stats   = get_liq_stats(symbol)

    if ws_clusters:
        # WebSocket 已积累到足够数据（≥10条记录）
        result["liquidation_clusters"] = ws_clusters
        result["liq_source"]           = "websocket"
        result["liq_stats"]            = liq_stats
    else:
        # WebSocket 数据不足（刚启动/断线），用 OI+S/R 估算兜底
        with _lock:
            tf_data = _cache.get(symbol, {}).get("timeframes", {})
        estimated = estimate_liq_clusters_from_sr(symbol, tf_data)
        result["liquidation_clusters"] = estimated
        result["liq_source"]           = "estimated"
        result["liq_stats"]            = liq_stats

    return jsonify(result)


@app.route("/api/liquidations")
def liquidations():
    """清算密集区独立端点（支持自定义时间窗口）"""
    symbol = request.args.get("symbol", "BTCUSDT").upper()
    hours  = int(request.args.get("hours", 24))
    top_n  = int(request.args.get("top", 7))

    if symbol not in _cache:
        return jsonify({"error": "Invalid symbol"}), 400

    ws_clusters = get_liq_clusters(symbol, hours=hours, top_n=top_n)
    liq_stats   = get_liq_stats(symbol)

    if ws_clusters:
        source   = "websocket"
        clusters = ws_clusters
    else:
        with _lock:
            tf_data = _cache.get(symbol, {}).get("timeframes", {})
        clusters = estimate_liq_clusters_from_sr(symbol, tf_data)
        source   = "estimated"

    persist_info = {
        "file":       _LIQ_PERSIST_FILE,
        "flush_every": _LIQ_FLUSH_INTERVAL,
        "exists":     os.path.exists(_LIQ_PERSIST_FILE),
        "size_kb":    round(os.path.getsize(_LIQ_PERSIST_FILE) / 1024, 1)
                      if os.path.exists(_LIQ_PERSIST_FILE) else 0,
    }

    return jsonify({
        "symbol":   symbol,
        "source":   source,
        "hours":    hours,
        "stats":    liq_stats,
        "persist":  persist_info,
        "clusters": clusters,
    })


@app.route("/api/onchain")
def onchain():
    """真链上数据端点（CoinMetrics Community API，30分钟缓存）"""
    global _onchain_cache, _onchain_cache_ts
    now = time.time()
    if _onchain_cache and (now - _onchain_cache_ts) < _ONCHAIN_TTL:
        return jsonify(_onchain_cache)
    try:
        monitor = _OnChainMonitor()
        data = monitor.get_coinmetrics_onchain("btc")
        _onchain_cache = data
        _onchain_cache_ts = now
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "source": "coinmetrics"}), 500


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 启动前先恢复历史清算数据
    _liq_load()

    t = threading.Thread(target=analysis_worker, daemon=True)
    t.start()

    # 启动清算 WebSocket 累积线程
    liq_ws_thread = threading.Thread(target=_liq_ws_worker, daemon=True, name="LiquidationWS")
    liq_ws_thread.start()

    # 启动定时落盘线程（每 5 分钟持久化一次）
    liq_flush_thread = threading.Thread(target=_liq_flush_worker, daemon=True, name="LiqFlush")
    liq_flush_thread.start()

    print("=" * 50)
    print("  多币种实时分析服务器")
    print("  支持币种: " + ", ".join(SYMBOLS))
    print("  访问地址: http://localhost:5000")
    print("  数据刷新: 每 5 分钟自动更新")
    print("=" * 50)

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
