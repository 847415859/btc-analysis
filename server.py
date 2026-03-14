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
    fetch_data, calculate_indicators,
    find_support_resistance, calculate_fibonacci,
    generate_report, TIMEFRAMES
)


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


def fetch_ob_walls(symbol: str, depth_limit: int = 1000) -> list:
    """
    从 Binance 期货深度接口获取挂单墙。
    返回 [(price, weight, side), ...] 列表，出错时返回 []。
    weight 为档位相对中位数的倍率（由 run_single_tf 乘以 TF 权重后传入 find_support_resistance）。
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
        return []

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
    return walls


def run_single_tf(symbol, interval, limit, label, chart_show, sr_lb, sr_pn, sr_cluster_pct, fib_lb, ob_walls=None):
    """分析单个周期，返回 {rd, candles, supports, resistances, fib}"""
    df = fetch_data(symbol=symbol, interval=interval, limit=limit)
    if df.empty:
        return None

    df = calculate_indicators(df)

    # 构建当前时间框架的 OB 层位（乘以 TF 相关性权重）
    ob_tf_factor = _OB_TF_WEIGHTS.get(interval, 0.0)
    ob_levels = []
    if ob_walls and ob_tf_factor > 0.0:
        for ob_price, ob_weight, ob_side in ob_walls:
            ob_levels.append((ob_price, round(ob_weight * ob_tf_factor, 2), ob_side))

    supports, resistances = find_support_resistance(
        df, lookback=sr_lb, pivot_n=sr_pn,
        cluster_pct=sr_cluster_pct, ob_levels=ob_levels
    )
    fib_levels, fib_high, fib_low = calculate_fibonacci(df, lookback=fib_lb)

    # 静默调用 generate_report（不打印到控制台）
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, rd = generate_report(
            df, supports, resistances, fib_levels,
            symbol=symbol, tf_label=label, verbose=False
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
        "supports":    [{"price": s[0],
                         "dist_pct": round((price - s[0]) / price * 100, 2),
                         "source":   s[2] if len(s) > 2 else "pivot"}
                        for s in supports],
        "resistances": [{"price": r[0],
                         "dist_pct": round((r[0] - price) / price * 100, 2),
                         "source":   r[2] if len(r) > 2 else "pivot"}
                        for r in resistances],
        "fib": [{"label": k, "price": v,
                 "dist_pct": round((v - price) / price * 100, 2),
                 "near": abs((v - price) / price * 100) < 2}
                for k, v in fib_levels.items()],
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

    ob_walls = []
    if needs_ob:
        _rate_limiter.acquire(2)   # depth 接口权重较重，消耗 2 个令牌
        ob_walls = fetch_ob_walls(symbol)
        if ob_walls:
            print(f"[服务器] {symbol} OB 挂单墙: {len(ob_walls)} 个价格层")

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
                                   ob_walls=ob_walls)
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

    now = datetime.now()
    with _lock:
        _cache[symbol]["timeframes"]  = new_tf
        _cache[symbol]["last_update"] = now.strftime("%Y-%m-%d %H:%M:%S")
        _cache[symbol]["updating"]    = False
        _cache[symbol]["ready"]       = True

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
                "rd":          tf_data["rd"],
                "supports":    tf_data["supports"],
                "resistances": tf_data["resistances"],
                "fib":         tf_data["fib"],
            }
        return jsonify({
            "symbol":      symbol,
            "ready":       data["ready"],
            "last_update": data["last_update"],
            "update_id":   data.get("update_id", 0),
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
            "candles":     tf_data["candles"],
            "supports":    tf_data["supports"],
            "resistances": tf_data["resistances"],
            "fib":         tf_data["fib"],
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
