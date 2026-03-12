# -*- coding: utf-8 -*-
"""
BTC/USDT 实时分析服务器
每5分钟自动拉取最新数据并分析，通过 REST API 提供给前端仪表板
访问: http://localhost:5000
"""

import os
import sys
import io
import math
import time
import socket
import threading
import contextlib
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
from flask import Flask, jsonify, render_template
from flask.json import JSONEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from btc_analysis import (
    fetch_btc_data, calculate_indicators,
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

# ─────────────────────────────────────────────
# 全局缓存
# ─────────────────────────────────────────────

_cache = {
    "last_update": None,
    "next_update": None,
    "updating": False,
    "ready": False,
    "timeframes": {}
}
_lock = threading.Lock()

REFRESH_INTERVAL = 300  # 5 分钟


def _safe(v):
    """将 NaN / None 转为 None，其余四舍五入到4位小数"""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(float(v), 4)
    except Exception:
        return None


def run_single_tf(interval, limit, label, chart_show, sr_lb, sr_pn, fib_lb):
    """分析单个周期，返回 {rd, candles, supports, resistances, fib}"""
    df = fetch_btc_data(interval=interval, limit=limit)
    if df.empty:
        return None

    df = calculate_indicators(df)
    supports, resistances = find_support_resistance(df, lookback=sr_lb, pivot_n=sr_pn)
    fib_levels, fib_high, fib_low = calculate_fibonacci(df, lookback=fib_lb)

    # 静默调用 generate_report（不打印到控制台）
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, rd = generate_report(
            df, supports, resistances, fib_levels,
            tf_label=label, verbose=False
        )

    # 准备图表用的 K 线数据（带指标值）
    ind_cols = ["ma20", "ma50", "ma200", "ema20", "ema50",
                "rsi", "bb_upper", "bb_mid", "bb_lower",
                "macd", "macd_signal", "macd_hist",
                "kdj_k", "kdj_d", "kdj_j", "obv"]
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
        "supports":    [{"price": s[0], "dist_pct": round((price - s[0]) / price * 100, 2)}
                        for s in supports],
        "resistances": [{"price": r[0], "dist_pct": round((r[0] - price) / price * 100, 2)}
                        for r in resistances],
        "fib": [{"label": k, "price": v,
                 "dist_pct": round((v - price) / price * 100, 2),
                 "near": abs((v - price) / price * 100) < 2}
                for k, v in fib_levels.items()],
    }


# ─────────────────────────────────────────────
# 后台分析线程
# ─────────────────────────────────────────────

def analysis_worker():
    global _cache
    while True:
        with _lock:
            _cache["updating"] = True

        print(f"\n[服务器] ===== 开始更新分析数据 {datetime.now().strftime('%H:%M:%S')} =====")
        new_tf = {}

        for interval, limit, label, chart_show, sr_lb, sr_pn, fib_lb in TIMEFRAMES:
            try:
                print(f"[服务器] 分析 {label} ({interval})...")
                result = run_single_tf(interval, limit, label,
                                       chart_show, sr_lb, sr_pn, fib_lb)
                if result:
                    new_tf[interval] = result
                    score = result["rd"]["score"]
                    overall = result["rd"]["overall"].split("(")[0].strip()
                    print(f"[服务器] {label} 完成 → {overall} {score:+d}")
                time.sleep(0.3)
            except Exception as e:
                print(f"[服务器] {label} 分析失败: {e}")

        now    = datetime.now()
        next_t = now + timedelta(seconds=REFRESH_INTERVAL)

        with _lock:
            _cache["timeframes"]  = new_tf
            _cache["last_update"] = now.strftime("%Y-%m-%d %H:%M:%S")
            _cache["next_update"] = next_t.strftime("%Y-%m-%d %H:%M:%S")
            _cache["updating"]    = False
            _cache["ready"]       = True

        print(f"[服务器] 分析完成，下次更新: {next_t.strftime('%H:%M:%S')}\n")
        time.sleep(REFRESH_INTERVAL)


# ─────────────────────────────────────────────
# Flask 路由
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def status():
    with _lock:
        now    = datetime.now()
        next_t = _cache.get("next_update")
        if next_t:
            try:
                ndt       = datetime.strptime(next_t, "%Y-%m-%d %H:%M:%S")
                remaining = max(0, int((ndt - now).total_seconds()))
            except Exception:
                remaining = REFRESH_INTERVAL
        else:
            remaining = REFRESH_INTERVAL
        return jsonify({
            "last_update":      _cache.get("last_update"),
            "next_update":      _cache.get("next_update"),
            "seconds_remaining": remaining,
            "updating":         _cache.get("updating", False),
            "ready":            _cache.get("ready", False),
        })


@app.route("/api/analysis")
def analysis():
    """返回所有周期的分析摘要（不含大体积 K 线数据）"""
    with _lock:
        result = {}
        for tf_key, data in _cache["timeframes"].items():
            result[tf_key] = {
                "rd":          data["rd"],
                "supports":    data["supports"],
                "resistances": data["resistances"],
                "fib":         data["fib"],
            }
        return jsonify({
            "ready":       _cache["ready"],
            "last_update": _cache["last_update"],
            "timeframes":  result,
        })


@app.route("/api/klines/<interval>")
def klines(interval):
    """返回指定周期的 K 线 + 指标数据"""
    with _lock:
        data = _cache["timeframes"].get(interval)
        if not data:
            return jsonify({"error": "数据未就绪，请稍候..."}), 404
        return jsonify({
            "candles":     data["candles"],
            "supports":    data["supports"],
            "resistances": data["resistances"],
            "fib":         data["fib"],
        })


@app.route("/api/price")
def live_price():
    """实时价格 + 24h 统计（轻量，供前端每10秒刷新）"""
    try:
        r = req.get("https://api.binance.com/api/v3/ticker/24hr",
                    params={"symbol": "BTCUSDT"}, timeout=5)
        d = r.json()
        return jsonify({
            "price":       float(d["lastPrice"]),
            "change_pct":  float(d["priceChangePercent"]),
            "high24h":     float(d["highPrice"]),
            "low24h":      float(d["lowPrice"]),
            "volume24h":   float(d["quoteVolume"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=analysis_worker, daemon=True)
    t.start()

    print("=" * 50)
    print("  BTC/USDT 实时分析服务器")
    print("  访问地址: http://localhost:5000")
    print("  数据刷新: 每 5 分钟自动更新")
    print("=" * 50)

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
