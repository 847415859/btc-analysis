# CryptoTA · 多币种实时技术分析系统

基于 Binance 公开 API 的加密货币技术分析 Dashboard，支持 BTC / ETH / SOL / BNB 四币种、八个时间框架的实时分析与可视化。

---

## 功能概览

### 行情数据
- 实时价格、24H 高低价、成交额
- 资金费率（当期 + 下次结算时间）
- 多币种切换（BTCUSDT / ETHUSDT / SOLUSDT / BNBUSDT）

### 技术指标
| 指标 | 说明 |
|------|------|
| MA 20 / 50 / 200 | 简单移动均线 |
| EMA 20 / 50 | 指数移动均线 |
| RSI | 自适应窗口（高波动期 21，常规 14） |
| 布林带 | 自适应标准差（高波动期 2.5σ，常规 2σ） |
| MACD (12/26/9) | 趋势动能 |
| KDJ (9/3) | 随机震荡 |
| OBV | 能量潮 |
| VWAP | 成交量加权均价（日内重置） |
| ATR (14) | 真实波动幅度 + 1.5× 止损线 |
| ADX / DI+ / DI- | 趋势强度与方向判断 |
| **CVD** | 累计成交量差值（主动买入 − 主动卖出） |
| **OFI** | 每根 K 线买卖失衡比 [-1, +1] |

### 支撑 / 阻力识别
- **Pivot 分形**：自适应窗口（基于 ATR 百分位，低波动期加宽过滤噪音，高波动期缩小快速响应）
- **成交量节点**：高成交量价格区间分箱
- **订单簿挂单墙**：实时读取 Binance 深度数据，3× 中位数以上大单层作为额外 S/R 节点，赋予更高权重
- **来源标注**：每个 S/R 层标记来源（pivot / volume / orderbook_bid / orderbook_ask）并在前端显示徽章

### 市场机制识别（Market Regime）
- ADX ≥ 25 → 趋势市，启用均线突破评分
- ADX ≤ 20 → 震荡市，禁用均线信号，改用高抛低吸策略
- ATR 百分位排名自动切换 RSI / 布林带参数

### 多时间框架共振矩阵
8 个时间框架（15m / 30m / 1H / 2H / 4H / 8H / 日 / 周）× 8 项指标（均线 / RSI / MACD / KDJ / 布林带 / OBV / CVD / VWAP），一屏看出跨周期共振方向。

### 合约数据分析
- **资金费率历史**（近 67 天，200 期）：走势图 + 历史百分位
- **资金费率背离检测**：价格走强但 FR 趋势下降 → 顶背离警报；价格下跌但 FR 上升 → 底背离信号
- **持仓量（OI）趋势**：半周期对比，判断新仓入场 / 旧仓离场
- **OI 突破验证**：价格↑ + OI↑ = 真突破；价格↑ + OI↓ = 轧空行情
- **清算密集区**：近期强平订单按价格档位聚合，标注多头/空头清算主导方向

### CVD / 主动成交量分析
- **CVD 背离**：价格创新高但 CVD 下降 → 派发顶；价格新低但 CVD 上升 → 吸筹底
- **OFI 成交量柱染色**：成交量柱颜色由主动买卖失衡方向决定（非 K 线涨跌），强度反映失衡程度
- **踏印图（Footprint Chart）**：点击「踏印图」按钮，拉取最近 K 线的逐笔成交，按动态价格档位分桶展示主动买/卖量与 Delta，标注 POC（成交最密集价位）

### 资深交易员综合研判
基于 18 分制评分系统（MA 5项 + RSI + BB + MACD + KDJ + OBV + CVD + VWAP），综合资金费率、OI 信号输出：
- 综合得分、多空建议（做多 / 做空 / 观望）
- 置信度分级（高 / 中 / 低）
- 多头因素列表 + 看空因素列表
- 风险提示（超买超卖、趋势背离等）

---

## 技术架构

```
btc-analysis/
├── server.py           # Flask 后端 + 后台分析线程
├── btc_analysis.py     # 指标计算、S/R识别、generate_report
├── templates/
│   └── dashboard.html  # 单页 PWA（Plotly.js 图表）
├── static/
│   ├── manifest.json
│   ├── sw.js           # Service Worker（PWA离线支持）
│   └── icon.svg
└── requirements.txt
```

### 后端（server.py）
- **并行刷新**：`ThreadPoolExecutor(max_workers=4)` 并行分析 4 个币种，刷新耗时从串行约 2 分钟压缩至单币种分析时间
- **版本号机制**：每轮全量完成后递增 `update_id`，前端检测变化自动静默刷新 + 右下角 toast 提示
- **分层 TTL**：15m/30m 每 5 分钟刷新；1h/2h 每 10 分钟；4h 每 15 分钟；8h 每 30 分钟；日线每小时；周线每 2 小时；减少无效 API 请求约 55%
- **令牌桶限流**：`TokenBucket(rate=5/s, capacity=10)`，K 线请求消耗 1 令牌，深度接口消耗 2 令牌，防止触发 Binance 429
- **429 自动重试**：读取 `Retry-After` 响应头，最多重试 3 次
- **订单簿挂单墙**：`fetch_ob_walls()` 按需懒加载（仅当短周期 TF 过期时才请求），`_OB_TF_WEIGHTS` 控制不同周期的 OB 相关性权重

### 前端（dashboard.html）
- 纯 HTML + Vanilla JS，无框架依赖
- Plotly.js 渲染交互式 K 线图（桌面 6 面板 + 移动端 2+4 面板）
- PWA：可添加至主屏幕，支持离线加载
- 响应式布局：桌面侧边栏 + 移动端底部 Tab

---

## API 端点

| 端点 | 说明 |
|------|------|
| `GET /` | Dashboard 主页 |
| `GET /api/analysis?symbol=` | 所有周期分析摘要（含 rd / S/R / Fibonacci） |
| `GET /api/klines/<interval>?symbol=` | 指定周期 K 线 + 指标数据（含 CVD/OFI/Delta） |
| `GET /api/status?symbol=` | 刷新状态（update_id / seconds_remaining） |
| `GET /api/price?symbol=` | 实时价格 + 24H 统计 |
| `GET /api/funding?symbol=` | 资金费率 + OI 历史 + OI 分析 + 清算密集区 |
| `GET /api/footprint/<interval>?symbol=` | 最近 K 线踏印数据（aggTrades 按价格分桶） |

---

## 快速启动

### 环境要求
- Python 3.8+
- 无需 Binance API Key（使用公开端点）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 启动服务
```bash
python server.py
# 或 Windows 双击 run_server.bat
```

浏览器访问 `http://localhost:5000`

---

## 数据来源

| 数据 | 接口 |
|------|------|
| K 线（含 taker_buy） | `api.binance.com/api/v3/klines` |
| 实时价格 / 24H | `api.binance.com/api/v3/ticker/24hr` |
| 订单簿深度 | `fapi.binance.com/fapi/v1/depth` |
| 资金费率历史 | `fapi.binance.com/fapi/v1/fundingRate` |
| 持仓量历史 | `fapi.binance.com/futures/data/openInterestHist` |
| 逐笔成交（踏印图） | `api.binance.com/api/v3/aggTrades` |

> 所有接口均为 Binance 公开接口，无需 API Key。

---

## 注意事项

- 踏印图每次拉取最多 1000 笔成交（Binance 单次上限），高频时段为样本数据
- 清算密集区接口（`allForceOrders`）已被 Binance 停用，系统静默忽略，不影响其他功能
- 本系统仅供技术研究参考，不构成投资建议
