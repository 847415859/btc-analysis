# # -*- coding: utf-8 -*-
# """
# 自动化回测与信号验证引擎 (Backtesting Engine)
# 功能：
# 1. 拉取长周期历史数据（K线 + 资金费率）并缓存到本地 CSV。
# 2. 提供策略接口，允许用户自定义交易规则（如 CVD 背离、指标组合）。
# 3. 执行回测，模拟逐根 K 线交易。
# 4. 生成绩效报告：胜率、盈亏比、最大回撤、夏普比率、收益曲线。
# """
#
# import os
# import time
# import pandas as pd
# import numpy as np
# import requests
# from datetime import datetime, timedelta
# import ta
#
# # 引入 btc_analysis 中的指标计算逻辑（复用）
# try:
#     from btc_analysis import calculate_indicators
# except ImportError:
#     # 如果无法导入（例如独立运行），则复制相关逻辑或简化处理
#     pass
#
# class DataLoader:
#     """数据加载器：负责获取长周期历史数据并管理本地缓存"""
#
#     def __init__(self, data_dir="data_cache"):
#         self.data_dir = data_dir
#         if not os.path.exists(data_dir):
#             os.makedirs(data_dir)
#
#     def fetch_klines(self, symbol="BTCUSDT", interval="1h", start_str="2021-01-01", end_str=None):
#         """
#         拉取 K 线数据（支持断点续传/缓存）。
#         start_str: 'YYYY-MM-DD'
#         """
#         filename = f"{self.data_dir}/{symbol}_{interval}_{start_str}_{end_str or 'now'}.csv"
#
#         if os.path.exists(filename):
#             print(f"[数据] 从缓存加载: {filename}")
#             df = pd.read_csv(filename, parse_dates=["date"])
#             return df
#
#         print(f"[数据] 开始从 Binance 拉取 {symbol} {interval} 数据 ({start_str} ~ {end_str or 'now'})...")
#
#         # 解析时间
#         start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
#         end_ts   = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
#
#         url = "https://api.binance.com/api/v3/klines"
#         limit = 1000
#         all_data = []
#         current_start = start_ts
#
#         while current_start < end_ts:
#             params = {
#                 "symbol": symbol,
#                 "interval": interval,
#                 "startTime": current_start,
#                 "endTime": end_ts,
#                 "limit": limit
#             }
#             try:
#                 resp = requests.get(url, params=params, timeout=10)
#                 resp.raise_for_status()
#                 data = resp.json()
#
#                 if not data:
#                     break
#
#                 all_data.extend(data)
#
#                 # 更新下一次开始时间 = 最后一根K线收盘时间 + 1ms
#                 last_close_time = data[-1][6]
#                 current_start = last_close_time + 1
#
#                 # 进度提示
#                 progress = (current_start - start_ts) / (end_ts - start_ts) * 100
#                 print(f"\r[数据] 下载进度: {progress:.2f}% ({len(all_data)} 根)", end="")
#
#                 time.sleep(0.1) # 避免触发限流
#
#             except Exception as e:
#                 print(f"\n[错误] 拉取失败: {e}，重试中...")
#                 time.sleep(2)
#                 continue
#
#         print("\n[数据] 下载完成，正在处理...")
#
#         if not all_data:
#             return pd.DataFrame()
#
#         df = pd.DataFrame(all_data, columns=[
#             "open_time", "open", "high", "low", "close", "volume",
#             "close_time", "quote_volume", "trades",
#             "taker_buy_base", "taker_buy_quote", "ignore"
#         ])
#
#         # 类型转换
#         numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_base"]
#         for col in numeric_cols:
#             df[col] = df[col].astype(float)
#
#         df["date"] = pd.to_datetime(df["open_time"], unit="ms")
#
#         # 简单清洗
#         df = df[["date", "open", "high", "low", "close", "volume", "taker_buy_base"]]
#
#         # 保存缓存
#         df.to_csv(filename, index=False)
#         print(f"[数据] 已保存至 {filename}")
#
#         return df
#
#     def fetch_funding_rates(self, symbol="BTCUSDT", start_str="2021-01-01", end_str=None):
#         """
#         拉取资金费率历史（支持分页拉取长周期数据）。
#         """
#         filename = f"{self.data_dir}/{symbol}_funding_{start_str}_{end_str or 'now'}.csv"
#         if os.path.exists(filename):
#             print(f"[数据] 从缓存加载资金费率: {filename}")
#             df = pd.read_csv(filename, parse_dates=["fundingTime"])
#             return df
#
#         print(f"[数据] 开始拉取 {symbol} 资金费率...")
#         start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
#         end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)
#
#         url = "https://fapi.binance.com/fapi/v1/fundingRate"
#         all_data = []
#         current_start = start_ts
#
#         while current_start < end_ts:
#             params = {
#                 "symbol": symbol,
#                 "startTime": current_start,
#                 "endTime": end_ts,
#                 "limit": 1000
#             }
#             try:
#                 resp = requests.get(url, params=params, timeout=10)
#                 resp.raise_for_status()
#                 data = resp.json()
#
#                 if not data:
#                     break
#
#                 all_data.extend(data)
#                 last_time = data[-1]["fundingTime"]
#                 current_start = last_time + 1
#
#                 time.sleep(0.1)
#             except Exception as e:
#                 print(f"[错误] 拉取资金费率失败: {e}")
#                 time.sleep(2)
#                 continue
#
#         if not all_data:
#             return pd.DataFrame()
#
#         df = pd.DataFrame(all_data)
#         df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
#         df["fundingRate"] = df["fundingRate"].astype(float)
#
#         # 保存
#         df.to_csv(filename, index=False)
#         print(f"[数据] 资金费率已保存至 {filename}")
#         return df
#
#     def merge_data(self, df_kline, df_funding):
#         """合并 K 线与资金费率（将费率对齐到 K 线时间，向前填充）"""
#         if df_funding.empty:
#             df_kline["fundingRate"] = 0.0
#             return df_kline
#
#         # 确保时间列名一致以便合并，或者使用 asof merge
#         df_kline = df_kline.sort_values("date")
#         df_funding = df_funding.sort_values("fundingTime")
#
#         # 使用 merge_asof 进行近似匹配（资金费率通常每8小时一次，K线可能更密）
#         # direction='backward' 表示 K 线时间匹配最近的过去一个费率时间
#         merged = pd.merge_asof(
#             df_kline,
#             df_funding[["fundingTime", "fundingRate"]],
#             left_on="date",
#             right_on="fundingTime",
#             direction="backward"
#         )
#
#         # 填充缺失值
#         merged["fundingRate"] = merged["fundingRate"].fillna(0.0)
#         merged.drop(columns=["fundingTime"], inplace=True)
#         return merged
#
#
#
# class Strategy:
#     """策略基类"""
#     def __init__(self):
#         self.position = 0  # 0: 无仓位, 1: 多头, -1: 空头
#         self.entry_price = 0.0
#         self.stop_loss = 0.0
#         self.take_profit = 0.0
#
#     def init(self, df):
#         """初始化：预计算指标"""
#         pass
#
#     def next(self, i, row, df):
#         """
#         处理每一根 K 线
#         i: 当前索引
#         row: 当前行数据
#         df: 全量数据（注意不要用到未来数据，只能用 df.iloc[:i+1]）
#         返回:
#            'buy', 'sell', 'close_long', 'close_short', or None
#         """
#         return None
#
#
# class BacktestEngine:
#     """回测引擎"""
#     def __init__(self, initial_capital=10000.0, commission=0.0004):
#         self.initial_capital = initial_capital
#         self.capital = initial_capital
#         self.commission = commission  # 手续费率，默认万4
#         self.trades = []
#         self.equity_curve = []
#
#     def run(self, df, strategy):
#         """执行回测"""
#         print(f"[回测] 开始执行策略回测 (数据量: {len(df)} 根)...")
#
#         strategy.init(df)
#
#         # 记录当前持仓状态
#         position = 0       # 0, 1, -1
#         entry_price = 0.0
#         entry_time = None
#
#         for i in range(len(df)):
#             if i < 100: continue  # 跳过预热期
#
#             row = df.iloc[i]
#             current_price = row["close"]
#             date = row["date"]
#
#             # 1. 检查止损止盈
#             if position == 1: # 多头
#                 if row["low"] <= strategy.stop_loss:
#                     self._execute_trade("stop_loss_long", date, strategy.stop_loss, -1, "止损平多")
#                     position = 0
#                 elif row["high"] >= strategy.take_profit:
#                     self._execute_trade("take_profit_long", date, strategy.take_profit, -1, "止盈平多")
#                     position = 0
#             elif position == -1: # 空头
#                 if row["high"] >= strategy.stop_loss:
#                     self._execute_trade("stop_loss_short", date, strategy.stop_loss, 1, "止损平空")
#                     position = 0
#                 elif row["low"] <= strategy.take_profit:
#                     self._execute_trade("take_profit_short", date, strategy.take_profit, 1, "止盈平空")
#                     position = 0
#
#             # 如果已经平仓，重置
#             if position == 0:
#                 strategy.position = 0
#
#             # 2. 获取策略信号
#             signal = strategy.next(i, row, df)
#
#             # 3. 执行信号
#             if signal == 'buy' and position == 0:
#                 price = current_price
#                 self._execute_trade("open_long", date, price, 1, "开多")
#                 position = 1
#                 entry_price = price
#                 strategy.position = 1
#                 strategy.entry_price = price
#                 # 策略需要在 next 中设置 sl/tp，或者在这里设置默认
#
#             elif signal == 'sell' and position == 0:
#                 price = current_price
#                 self._execute_trade("open_short", date, price, -1, "开空")
#                 position = -1
#                 entry_price = price
#                 strategy.position = -1
#                 strategy.entry_price = price
#
#             elif signal == 'close_long' and position == 1:
#                 self._execute_trade("close_long", date, current_price, -1, "平多")
#                 position = 0
#                 strategy.position = 0
#
#             elif signal == 'close_short' and position == -1:
#                 self._execute_trade("close_short", date, current_price, 1, "平空")
#                 position = 0
#                 strategy.position = 0
#
#             # 记录每日权益
#             self.equity_curve.append({"date": date, "equity": self.capital})
#
#         print("[回测] 完成。")
#         return self._calculate_metrics()
#
#     def _execute_trade(self, action, date, price, direction, reason):
#         """
#         执行交易并更新资金
#         direction: 1 (买入/平空), -1 (卖出/平多)
#         """
#         # 计算手续费（按名义价值）
#         # 假设全仓模式，使用当前全部资金作为保证金开仓（简单复利模型）
#         # 实际交易中通常是固定金额或比例，这里简化为：
#         # 开仓时：名义价值 = 资金 * 杠杆(默认1倍)
#         # 平仓时：名义价值 = 仓位数量 * 当前价格
#
#         # 简化逻辑：只追踪资金变化，不处理复杂的保证金和强平
#
#         trade_record = {
#             "date": date,
#             "action": action,
#             "price": price,
#             "reason": reason,
#             "pnl_pct": 0.0,
#             "pnl_amount": 0.0,
#             "commission": 0.0
#         }
#
#         # 找到最近的持仓
#         open_position = None
#         for t in reversed(self.trades):
#             if "open" in t["action"] and "closed" not in t:
#                 open_position = t
#                 break
#
#         if "open" in action:
#             # 开仓
#             notional = self.capital  # 全仓
#             commission = notional * self.commission
#             self.capital -= commission
#             trade_record["commission"] = commission
#             trade_record["size"] = notional / price # 币的数量
#
#         elif "close" in action or "stop" in action or "take" in action:
#             # 平仓
#             if open_position:
#                 size = open_position["size"]
#                 entry_price = open_position["price"]
#
#                 # 计算名义价值用于扣手续费
#                 notional = size * price
#                 commission = notional * self.commission
#                 self.capital -= commission
#                 trade_record["commission"] = commission
#
#                 # 计算盈亏
#                 if "long" in open_position["action"]: # 平多
#                     pnl_amount = (price - entry_price) * size
#                     pnl_pct = (price - entry_price) / entry_price
#                 else: # 平空
#                     pnl_amount = (entry_price - price) * size
#                     pnl_pct = (entry_price - price) / entry_price
#
#                 self.capital += pnl_amount
#
#                 trade_record["pnl_pct"] = pnl_pct
#                 trade_record["pnl_amount"] = pnl_amount
#
#                 # 标记开仓记录已关闭
#                 open_position["closed"] = True
#
#         trade_record["capital_after"] = self.capital
#         self.trades.append(trade_record)
#
#     def _calculate_metrics(self):
#         """计算绩效指标"""
#         df_trades = pd.DataFrame(self.trades)
#         if df_trades.empty:
#             return {"error": "无交易记录"}
#
#         closed_trades = df_trades[df_trades["action"].str.contains("close|stop|take")].copy()
#         if closed_trades.empty:
#             return {"error": "无平仓记录"}
#
#         # 胜率
#         wins = closed_trades[closed_trades["pnl_amount"] > 0]
#         losses = closed_trades[closed_trades["pnl_amount"] <= 0]
#         win_rate = len(wins) / len(closed_trades)
#
#         # 盈亏比
#         avg_win = wins["pnl_amount"].mean() if not wins.empty else 0
#         avg_loss = abs(losses["pnl_amount"].mean()) if not losses.empty else 0
#         profit_factor = avg_win / avg_loss if avg_loss != 0 else float('inf')
#
#         # 最大回撤
#         equity = [t["capital_after"] for t in self.trades]
#         peak = equity[0]
#         max_drawdown = 0
#         for val in equity:
#             if val > peak:
#                 peak = val
#             dd = (peak - val) / peak
#             if dd > max_drawdown:
#                 max_drawdown = dd
#
#         # 夏普比率 (简化：基于每笔交易收益率)
#         returns = closed_trades["pnl_pct"]
#         sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() != 0 else 0
#
#         return {
#             "total_trades": len(closed_trades),
#             "win_rate": win_rate,
#             "profit_factor": profit_factor,
#             "max_drawdown": max_drawdown,
#             "sharpe_ratio": sharpe,
#             "final_capital": self.capital,
#             "return_pct": (self.capital - self.initial_capital) / self.initial_capital
#         }
#
#
# class AdvancedStrategy(Strategy):
#     """
#     高级策略示例：CVD背离 + 费率过滤 + 趋势确认
#
#     逻辑：
#     1. 趋势过滤：价格在 MA100 之上只做多，之下只做空。
#     2. 资金费率：做多需费率 < 0.01% (不拥挤)；做空需费率 > -0.01%。
#        (用户提到“费率负值”，这里作为可选条件)
#     3. CVD 背离：
#        - 顶背离：价格创新高(过去20根)，但 CVD 未创新高 -> 做空
#        - 底背离：价格创新低(过去20根)，但 CVD 未创新低 -> 做多
#     """
#     def init(self, df):
#         # 1. 基础指标
#         df["ma50"] = ta.trend.sma_indicator(df["close"], window=50)
#         df["ma100"] = ta.trend.sma_indicator(df["close"], window=100)
#         df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)
#
#         # 2. CVD 计算
#         # delta = 主动买 - 主动卖 (这里简化为 taker_buy - taker_sell)
#         # taker_sell = volume - taker_buy
#         df["delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
#         df["cvd"] = df["delta"].cumsum()
#
#         # 3. 预计算滚动高低点 (用于背离检测)
#         window = 20
#         df["price_high"] = df["high"].rolling(window).max()
#         df["price_low"] = df["low"].rolling(window).min()
#         df["cvd_high"] = df["cvd"].rolling(window).max()
#         df["cvd_low"] = df["cvd"].rolling(window).min()
#
#     def next(self, i, row, df):
#         if i < 100: return None
#
#         close = row["close"]
#         ma100 = row["ma100"]
#         atr = row["atr"]
#         funding = row.get("fundingRate", 0.0)
#
#         # 1. 趋势判断
#         trend_long = close > ma100
#         trend_short = close < ma100
#
#         # 2. 费率过滤 (做多喜欢负费率或低费率，做空喜欢正费率或高费率)
#         # 负费率意味着空头支付多头，说明空头拥挤，利好做多
#         funding_long_ok = funding < 0.0001 # 费率 < 0.01% (甚至负值)
#         funding_short_ok = funding > -0.0001
#
#         # 3. 背离检测 (简化版)
#         # 实际背离需要找 Pivot Points，这里用滚动窗口近似
#         # 如果当前价格接近周期高点，但 CVD 远离周期高点 -> 顶背离
#
#         # 获取前一根 K 线数据用于比较
#         prev = df.iloc[i-1]
#
#         # 顶背离条件：
#         # 价格创 20 周期新高 (或接近)
#         price_new_high = row["high"] >= row["price_high"] * 0.999
#         # CVD 显著低于 20 周期高点
#         cvd_weak = row["cvd"] < row["cvd_high"] * 0.98 # 这里的阈值需根据 CVD 量级调整，百分比可能不适用数值
#         # 更稳健的写法：CVD 没有创新高
#         cvd_no_new_high = row["cvd"] < row["cvd_high"]
#
#         # 底背离条件：
#         # 价格创 20 周期新低
#         price_new_low = row["low"] <= row["price_low"] * 1.001
#         # CVD 没有创新低 (相对强势)
#         cvd_strong = row["cvd"] > row["cvd_low"]
#
#         signal = None
#
#         # 开多：趋势向上 + 资金费率合适 + 底背离 (或 CVD 强势)
#         if trend_long and funding_long_ok:
#             if price_new_low and cvd_strong: # 底背离
#                 if self.position == 0:
#                     self.stop_loss = close - 2.0 * atr
#                     self.take_profit = close + 5.0 * atr
#                     signal = 'buy'
#
#         # 开空：趋势向下 + 资金费率合适 + 顶背离
#         elif trend_short and funding_short_ok:
#             if price_new_high and cvd_no_new_high: # 顶背离
#                 if self.position == 0:
#                     self.stop_loss = close + 2.0 * atr
#                     self.take_profit = close - 5.0 * atr
#                     signal = 'sell'
#
#         # 移动止损/出场逻辑 (可选)
#         # 例如：跌破 MA50 平多
#         if self.position == 1 and close < row["ma50"]:
#              signal = 'close_long'
#         elif self.position == -1 and close > row["ma50"]:
#              signal = 'close_short'
#
#         return signal
#
# if __name__ == "__main__":
#     # 1. 初始化
#     loader = DataLoader()
#     symbol = "BTCUSDT"
#
#     # 2. 拉取数据 (示例：2023年至今)
#     print("正在准备回测数据...")
#     df_kline = loader.fetch_klines(symbol, "1h", "2023-01-01")
#     df_funding = loader.fetch_funding_rates(symbol, "2023-01-01")
#
#     # 3. 合并数据
#     df = loader.merge_data(df_kline, df_funding)
#
#     if not df.empty:
#         # 4. 运行策略
#         engine = BacktestEngine(initial_capital=10000)
#         strategy = AdvancedStrategy() # 使用高级策略
#         metrics = engine.run(df, strategy)
#
#         # 5. 输出报告
#         print("\n" + "="*45)
#         print(f"   高级策略回测报告 ({symbol} 1h)")
#         print(f"   策略: 趋势(MA100) + CVD背离 + 资金费率")
#         print("="*45)
#         if "error" in metrics:
#             print(f"错误: {metrics['error']}")
#         else:
#             print(f"回测区间: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
#             print(f"交易次数: {metrics['total_trades']}")
#             print(f"胜率:     {metrics['win_rate']:.2%}")
#             print(f"盈亏比:   {metrics['profit_factor']:.2f}")
#             print(f"最大回撤: {metrics['max_drawdown']:.2%}")
#             print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
#             print(f"总收益率: {metrics['return_pct']:.2%}")
#             print(f"最终资金: ${metrics['final_capital']:.2f}")
#         print("="*45)
#
