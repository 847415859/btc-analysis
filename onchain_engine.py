# -*- coding: utf-8 -*-
"""
市场数据监控引擎 (Market Data Monitor)

数据来源说明（重要区分）
────────────────────────────────────────────────────────────
◆ 真链上数据 (真实 On-Chain)
    来源: CoinMetrics Community API（免费，无需 API Key）
    指标: SOPR、已实现价格、活跃地址数、NVT 比率、矿工哈希率、交易数量
    意义: 反映比特币网络本身的持仓行为、矿工信心和链上估值

◆ 宏观流动性代理 (Macro Liquidity Proxy)
    来源: DeFiLlama（免费）
    指标: USDT/USDC 稳定币市值变化
    意义: 场外增量资金流入/流出的宏观代理（非链上）

◆ 合约衍生数据 (Futures-Derived)
    来源: Binance FAPI（永续合约 API）
    指标: 大户多空比、持仓量（OI）
    意义: 反映合约市场情绪，与链上语义无关
────────────────────────────────────────────────────────────
"""

import os
import time
import requests
from datetime import datetime, timedelta


class OnChainMonitor:
    def __init__(self):
        self.defillama_base_url   = "https://stablecoins.llama.fi"
        self.coinmetrics_base_url = "https://community-api.coinmetrics.io/v4"
        self.cache        = {}
        self.cache_expiry = 3600   # 1 小时缓存

    # ──────────────────────────────────────────────────────────────
    # 1. 真链上数据：CoinMetrics Community API（SOPR / 已实现价格 / 活跃地址等）
    # ──────────────────────────────────────────────────────────────

    def get_coinmetrics_onchain(self, asset: str = "btc") -> dict:
        """
        真实链上指标 — CoinMetrics Community API（完全免费，无需 API Key）

        指标说明：
          AdrActCnt   活跃地址数   → 每日在链上有收/发记录的唯一地址数，衡量网络活跃度
          Sopr        SOPR        → 已花费输出的盈亏比
                                    > 1.0: 持有者平均以盈利卖出（潜在供给压力）
                                    < 1.0: 持有者平均以亏损卖出（恐慌/底部特征）
          CapRealUSD  已实现市值   → 所有 UTXO 按最后移动时的价格计算的市值总和
                                    除以流通量 ≈ 链上"成本基础"均价（已实现价格）
          HashRate30d 30日均哈希率 → 矿工算力投入（矿工信心指标），单位 EH/s
          TxCnt       交易数量     → 每日链上交易笔数（网络使用率代理）
          NVTAdj      NVT 比率     → 已调整的网络价值/交易量比率（链上 P/E 比）
                                    > 65: 相对高估；< 45: 相对低估
        """
        cache_key = f"coinmetrics_{asset}"
        if (cache_key in self.cache and
                time.time() - self.cache[cache_key]["timestamp"] < self.cache_expiry):
            return self.cache[cache_key]["data"]

        end_dt   = datetime.utcnow()
        start_dt = end_dt - timedelta(days=14)
        metrics  = "AdrActCnt,Sopr,CapRealUSD,HashRate30d,TxCnt,NVTAdj"

        try:
            r = requests.get(
                f"{self.coinmetrics_base_url}/timeseries/asset-metrics",
                params={
                    "assets":     asset,
                    "metrics":    metrics,
                    "start_time": start_dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "end_time":   end_dt.strftime("%Y-%m-%dT00:00:00Z"),
                    "frequency":  "1d",
                    "sort":       "time",
                },
                timeout=15,
            )
            r.raise_for_status()
            rows = r.json().get("data", [])
            if not rows:
                return {"error": "CoinMetrics 返回空数据"}

            latest  = rows[-1]
            prev_7d = rows[-7] if len(rows) >= 7 else rows[0]

            def _f(row, key):
                v = row.get(key)
                return float(v) if v not in (None, "", "null") else None

            sopr      = _f(latest,  "Sopr")
            sopr_prev = _f(prev_7d, "Sopr")
            addr_cnt  = _f(latest,  "AdrActCnt")
            addr_prev = _f(prev_7d, "AdrActCnt")
            hash_rate = _f(latest,  "HashRate30d")   # H/s
            tx_cnt    = _f(latest,  "TxCnt")
            nvt       = _f(latest,  "NVTAdj")
            cap_real  = _f(latest,  "CapRealUSD")

            # 已实现价格 = 已实现市值 / 流通量（BTC 约 1960 万枚）
            BTC_SUPPLY = 19_600_000
            realized_price = round(cap_real / BTC_SUPPLY, 2) if cap_real else None

            # ── SOPR 信号 ──────────────────────────────────────────────
            sopr_score = 0
            if sopr is not None:
                if sopr >= 1.05:
                    sopr_signal = "盈利兑现期（持有者出货）"
                    sopr_score  = -1
                elif sopr >= 1.01:
                    sopr_signal = "温和盈利兑现（正常牛市）"
                    sopr_score  =  0
                elif sopr >= 0.97:
                    sopr_signal = "轻微亏损出售（临近底部）"
                    sopr_score  = +1
                else:
                    sopr_signal = "恐慌亏损出售（历史底部特征）"
                    sopr_score  = +2
                sopr_7d_trend = (
                    "上升（持有者信心增加）" if sopr_prev and sopr > sopr_prev * 1.002
                    else "下降（持有者抛压加重）" if sopr_prev and sopr < sopr_prev * 0.998
                    else "平稳"
                )
            else:
                sopr_signal, sopr_7d_trend = "数据不可用", "--"

            # ── 活跃地址变化 ──────────────────────────────────────────
            addr_chg_pct = None
            if addr_cnt and addr_prev:
                addr_chg_pct = round((addr_cnt - addr_prev) / addr_prev * 100, 1)
            addr_signal = (
                "活跃度↑ 网络扩张" if addr_chg_pct and addr_chg_pct > 5
                else "活跃度↓ 网络收缩" if addr_chg_pct and addr_chg_pct < -5
                else "活跃度稳定"
            )

            # ── NVT 估值 ────────────────────────────────────────────
            if nvt is not None:
                if nvt > 65:
                    nvt_signal = "偏高（市值相对高估）"
                elif nvt < 45:
                    nvt_signal = "偏低（市值相对低估）"
                else:
                    nvt_signal = "中性（估值合理）"
            else:
                nvt_signal = "数据不可用"

            # ── 哈希率（转换为 EH/s）────────────────────────────────
            hash_ehs = round(hash_rate / 1e18, 2) if hash_rate else None

            result = {
                "date":              latest.get("time", "")[:10],
                "source":            "CoinMetrics Community API",

                # SOPR
                "sopr":              round(sopr, 4)      if sopr      else None,
                "sopr_7d_ago":       round(sopr_prev, 4) if sopr_prev else None,
                "sopr_signal":       sopr_signal,
                "sopr_score":        sopr_score,
                "sopr_7d_trend":     sopr_7d_trend,

                # 活跃地址
                "active_addresses":     int(addr_cnt) if addr_cnt else None,
                "addr_7d_change_pct":   addr_chg_pct,
                "addr_signal":          addr_signal,

                # 已实现价格
                "realized_price":    realized_price,
                "cap_real_usd":      round(cap_real / 1e9, 1) if cap_real else None,  # B USD

                # 矿工哈希率
                "hash_rate_ehs":     hash_ehs,

                # 交易量
                "tx_count":          int(tx_cnt) if tx_cnt else None,

                # NVT
                "nvt_ratio":         round(nvt, 2) if nvt else None,
                "nvt_signal":        nvt_signal,
            }
            self.cache[cache_key] = {"timestamp": time.time(), "data": result}
            return result

        except Exception as e:
            print(f"[OnChain-CoinMetrics] 数据获取失败: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # 2. 宏观流动性代理：DeFiLlama 稳定币市值（非链上，仅宏观流动性参考）
    # ──────────────────────────────────────────────────────────────

    def get_stablecoin_flows(self) -> dict:
        """
        稳定币市值宏观流动性监控（DeFiLlama，免费无 API Key）
        ⚠️ 注意：这是宏观流动性代理指标，不是真正的链上数据
        """
        cache_key = "stablecoin_flows"
        if (cache_key in self.cache and
                time.time() - self.cache[cache_key]["timestamp"] < self.cache_expiry):
            return self.cache[cache_key]["data"]

        try:
            resp = requests.get(
                f"{self.defillama_base_url}/stablecoins", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            pegged_assets = data.get("peggedAssets", [])

            usdt = next((x for x in pegged_assets if x["symbol"] == "USDT"), None)
            if not usdt:
                return {"error": "无法获取 USDT 数据"}

            usdt_mcap       = usdt.get("circulating", {}).get("peggedUSD", 0)
            usdt_change_1d  = usdt.get("change_1d",  0)
            usdt_change_1m  = usdt.get("change_1m",  0)

            macro_liquidity = (
                "扩表 (增量资金入场)" if usdt_change_1m > 0
                else "缩表 (存量博弈/资金流出)"
            )

            flow_data = {
                "usdt_mcap":          usdt_mcap,
                "usdt_change_1d_pct": usdt_change_1d,
                "usdt_change_1m_pct": usdt_change_1m,
                "macro_liquidity":    macro_liquidity,
                "signal": (
                    "bullish" if usdt_change_1d > 0.1
                    else "bearish" if usdt_change_1d < -0.1
                    else "neutral"
                ),
                "desc": (
                    f"USDT 过去24h供应量变化 {usdt_change_1d:+.2f}%，"
                    f"过去1个月变化 {usdt_change_1m:+.2f}%"
                ),
                "source": "DeFiLlama (宏观流动性代理，非真链上)",
            }
            self.cache[cache_key] = {"timestamp": time.time(), "data": flow_data}
            return flow_data

        except Exception as e:
            print(f"[OnChain-DeFiLlama] 获取稳定币数据失败: {e}")
            return {"error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # 3. 合约衍生数据：Binance FAPI（大户多空比 + OI）
    #    ⚠️ 这是合约数据，不是链上数据
    # ──────────────────────────────────────────────────────────────

    def get_exchange_netflow(self, symbol: str = "BTC") -> dict:
        """
        合约市场情绪代理（Binance FAPI）
        ⚠️ 注意：这是合约衍生数据（大户多空比 + OI），不是真正的链上交易所净流量
        真实交易所净流量需使用 Glassnode / CryptoQuant 付费 API
        """
        try:
            url_ratio = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
            params    = {"symbol": f"{symbol}USDT", "period": "1d", "limit": 2}
            resp_r    = requests.get(url_ratio, params=params, timeout=5)
            data_r    = resp_r.json()

            url_oi   = "https://fapi.binance.com/fapi/v1/openInterest"
            resp_oi  = requests.get(url_oi, params={"symbol": f"{symbol}USDT"}, timeout=5)
            data_oi  = resp_oi.json()

            if not data_r or "code" in data_r:
                return {
                    "status":  "unknown",
                    "signal":  "neutral",
                    "desc":    "无法获取大户数据",
                    "source":  "Binance FAPI (合约衍生数据)",
                }

            latest_ratio = float(data_r[-1]["longShortRatio"])
            prev_ratio   = float(data_r[-2]["longShortRatio"]) if len(data_r) > 1 else latest_ratio
            current_oi   = float(data_oi.get("openInterest", 0))

            if latest_ratio > 1.2 and latest_ratio > prev_ratio:
                trend  = "decreasing"
                signal = "bullish"
                desc   = (
                    f"大户多头激增（多空比 {latest_ratio:.2f}），"
                    f"合约持仓 {current_oi:,.0f} 张，大户偏向做多。"
                )
            elif latest_ratio < 0.8 and latest_ratio < prev_ratio:
                trend  = "increasing"
                signal = "bearish"
                desc   = (
                    f"大户空头激增（多空比 {latest_ratio:.2f}），"
                    f"资金偏向做空或对冲抛压。"
                )
            else:
                trend  = "stable"
                signal = "neutral"
                desc   = (
                    f"大户多空均衡（多空比 {latest_ratio:.2f}），"
                    f"未见明显单边意图。"
                )

            return {
                "btc_exchange_reserve_trend": trend,
                "whale_long_short_ratio":     latest_ratio,
                "open_interest_btc":          current_oi,
                "signal":                     signal,
                "desc":                       desc,
                "source":                     "Binance FAPI (合约衍生数据，非链上交易所净流量)",
            }

        except Exception as e:
            print(f"[OnChain-FAPI] 获取合约数据失败: {e}")
            return {
                "btc_exchange_reserve_trend": "unknown",
                "signal":  "neutral",
                "desc":    "合约数据获取失败",
                "source":  "Binance FAPI",
            }

    # ──────────────────────────────────────────────────────────────
    # 4. 真假突破过滤器（结合真链上 + 宏观流动性）
    # ──────────────────────────────────────────────────────────────

    def detect_fakeout(self, tech_signal: str) -> dict:
        """
        真假突破过滤器
        tech_signal: 'breakout_up' | 'breakdown_down' | 其他

        优先使用真链上 SOPR 数据，辅以稳定币宏观流动性。
        """
        stablecoin = self.get_stablecoin_flows()
        exchange   = self.get_exchange_netflow("BTC")
        onchain    = self.get_coinmetrics_onchain("btc")

        if "error" in stablecoin:
            return {
                "status": "unknown",
                "score":  0,
                "desc":   "宏观数据缺失，无法验证突破真实性",
            }

        sc_signal    = stablecoin.get("signal", "neutral")
        ex_signal    = exchange.get("signal",   "neutral")
        sopr         = onchain.get("sopr")
        sopr_score   = onchain.get("sopr_score", 0)

        # ── 向上突破（做多验证）────────────────────────────────
        if tech_signal == "breakout_up":
            # 真突破需要：① SOPR < 1（亏损出售消化完毕，筹码交换健康）
            #             ② 稳定币增发（增量资金）
            #             ③ 大户不做空
            if sc_signal == "bearish" or exchange.get("btc_exchange_reserve_trend") == "increasing":
                return {
                    "status": "fakeout_warning",
                    "type":   "bull_trap",
                    "score":  -3,
                    "desc": (
                        "⚠️ 诱多警告 (Bull Trap)：技术面突破，"
                        "但缺乏 USDT 增量资金支持，且合约大户看空。"
                        "极可能是做市商洗盘，建议等链上确认。"
                    ),
                }
            if sopr and sopr < 1.0 and sc_signal == "bullish":
                return {
                    "status": "confirmed_breakout",
                    "score":  +3,
                    "desc": (
                        f"✅ 真突破确认：SOPR {sopr:.4f} < 1（弱手已出清）+ "
                        f"USDT 供应量增长，资金面完全支持此次上涨。"
                    ),
                }
            elif sc_signal == "bullish" and ex_signal == "bullish":
                return {
                    "status": "confirmed_breakout",
                    "score":  +2,
                    "desc": (
                        "✅ 突破基本确认：稳定币增发 + 大户偏多，"
                        "缺少 SOPR 链上最终确认，建议轻仓参与。"
                    ),
                }
            else:
                return {
                    "status": "unconfirmed",
                    "score":  0,
                    "desc": (
                        "⚪ 链上资金面中性，此次突破以场内存量资金博弈为主，"
                        "建议缩减仓位参与。"
                    ),
                }

        # ── 向下破位（做空验证）────────────────────────────────
        elif tech_signal == "breakdown_down":
            if ex_signal == "bullish" and sc_signal == "bullish":
                return {
                    "status": "fakeout_warning",
                    "type":   "bear_trap",
                    "score":  +3,
                    "desc": (
                        "⚠️ 诱空警告 (Bear Trap)：技术面破位，"
                        "但大户在做多且稳定币在增发。"
                        "很可能是清扫多头止损，建议等待反转信号。"
                    ),
                }
            if sopr and sopr > 1.02:
                return {
                    "status": "confirmed_breakdown",
                    "score":  -2,
                    "desc": (
                        f"✅ 真跌破确认：SOPR {sopr:.4f} > 1（持有者在盈利出货）"
                        f"，链上抛压真实存在。"
                    ),
                }
            elif exchange.get("btc_exchange_reserve_trend") == "increasing":
                return {
                    "status": "confirmed_breakdown",
                    "score":  -3,
                    "desc":   "✅ 真跌破确认：合约市场空头主导，下跌趋势成立。",
                }
            else:
                return {
                    "status": "unconfirmed",
                    "score":  0,
                    "desc": (
                        "⚪ 链上抛压不明显，下跌可能由清算引发（流动性真空），"
                        "注意快速反抽。"
                    ),
                }

        return {
            "status": "neutral",
            "score":  0,
            "desc":   "技术面无明显突破，数据仅作趋势参考。",
        }


if __name__ == "__main__":
    monitor = OnChainMonitor()

    print("=== 真链上数据（CoinMetrics）===")
    oc = monitor.get_coinmetrics_onchain("btc")
    if "error" not in oc:
        print(f"日期: {oc['date']}")
        print(f"SOPR: {oc['sopr']}  信号: {oc['sopr_signal']}")
        print(f"活跃地址: {oc['active_addresses']:,}  7日变化: {oc['addr_7d_change_pct']}%")
        print(f"已实现价格: ${oc['realized_price']:,.0f}" if oc["realized_price"] else "")
        print(f"NVT: {oc['nvt_ratio']}  {oc['nvt_signal']}")
        print(f"哈希率 (30d): {oc['hash_rate_ehs']} EH/s")
    else:
        print(f"链上数据获取失败: {oc['error']}")

    print("\n=== 宏观流动性代理（DeFiLlama）===")
    sc = monitor.get_stablecoin_flows()
    print(f"{sc.get('macro_liquidity')} - {sc.get('desc')}")

    print("\n=== 突破验证 (向上突破) ===")
    print(monitor.detect_fakeout("breakout_up"))
