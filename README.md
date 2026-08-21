# Survival-Line (生死線)

## Quickstart

```
pip install -r requirements.txt
python -m sdx.serve             # http://127.0.0.1:8765
```

係browser開http://127.0.0.1:8765, 生死線,訊號同指標會自動計算同畫出 — 預設用 yfinance，唔使API key、唔使config。

淨係想睇一張靜態圖，唔開server？`python -m sdx.viz 0388.HK` 會渲染一個獨立HTML圖表。

## 生死線是什麼

生死線一條由「轉角位」一級級砌起嘅支持/阻力**階梯**。
每次升段或跌段一反轉，只要突破咗上一段嘅全幅，就即刻畫出新線 — 唔會滯後，
亦唔會事後補畫。

**核心規則：**

> 線上持股，賺足全升浪；線下棄股，防止被綁

四條線，按以下次序交替出現（復甦線 → 支持線 → 死亡線 → 阻力線 → ...循環）：

| 線 | 意思 |
|---|---|
| **復甦線** | 跌段反轉做升段（跌→升轉角位）畫出嘅支持 |
| **支持線** | 確認咗嘅拉回低位，要「創新高 → 拉回 → 突破」呢個序列完成先會畫 |
| **死亡線** | 升段反轉做跌段（升→跌轉角位）畫出嘅阻力 — 同一支bar觸發清貨 |
| **阻力線** | 跌市入面「創新低 → 反彈 → 跌破」序列完成之後畫出嘅阻力 |

升市入面，目前嘅支持位會記做 `current_stop` — 佢跟住條線一路上，只會升，
唔會跌。

**訊號：**

- **量增即攻** — 當日高位突破前一日高位、收市喺全日波幅上半部、成交量高過
  前一日，就會觸發 — 但要喺升市入面、股價喺支持線之上，並且喺最近一個轉角位
  低位計起3日之內先算數。錯過就要等下一次拉回。
- **好友反攻** — 錘頭形態（長下影、短上影、細實體）嘅一支bar，成交量高過
  過去50日平均。純睇形態同成交量，同大市趨勢無關。
- **清貨** — 系統入面唯一嘅全數沽貨訊號：當一支bar嘅**最低位**（唔係收市價）
  跌穿 `current_stop` 就即刻觸發。呢個同畫死亡線係同一件事 — 見到一個，就見
  到另一個。

## Features

- 生死線：復甦線、死亡線、支持線、阻力線
- 訊號：量增即攻、好友反攻、減持、清貨
- 陰陽燭形態
- 指標
  - RSI：超賣 / 超買、背馳
  - MACD：背馳、差離
  - DMI：ADX方向
- Watchlist
- 檢查清單

## 陰陽燭形態與指標

### 陰陽燭形態

一共27個形態，分四組：

**單日**：十字星、鎚頭、倒轉鎚頭、吊頸、射擊之星、陀螺（陽/陰）

**雙日**：身懷六甲（陰/陽）、穿頭破腳（陰/陽）、曙光初現、烏雲蓋頂、十字胎（陰/陽）、平頂、平底

**三日**：黃昏之星、早晨之星、棄嬰頂、棄嬰底、黃昏十字星、早晨十字星、三隻烏鴉、三個白武士

**五日**：上升三部曲、下跌三部曲

### 指標

淨係用嚟睇市，唔會影響生死線本身。

預設：RSI(9)/SMA(6)、MACD(12,26,9)、DMI(DI 6, ADX 14)，全部用 Wilder
smoothing（同通達信/同花順/TradingView 一致）。

- **RSI**
  - **超賣 / 超買** — 參考線，預設25 / 75
  - **背馳** — 底背馳（睇好）：RSI轉角位低位升，但股價轉角位低位反而跌。
    頂背馳（睇淡）：頂部嘅鏡像
- **MACD** — 柱狀圖 `(DIF−DEA)×2`（乘2對齊通達信寫法）
  - **背馳** — 底背馳/頂背馳，用股價結構同DIF線比較，並要求回溯期內DIF嘅
    極值企喺同一邊零軸
  - **差離** — 呢個project自己加嘅延伸：同一套轉角位邏輯改用喺柱狀圖上（唔
    係DIF），無零軸過濾：牛差離（睇好）/ 熊差離（睇淡），方向由轉角位本身
    嘅柱狀圖正負決定
- **DMI**
  - **ADX方向** — ADX日對日變化決定背景色（升=青綠色，跌=橙色），ADX>40有
    紅色頂帶，另外有ADX 20/40虛線參考線

## Layout

```
sdx/                        engine
  types.py                   shared dataclasses/enums (R1-R12 references)
  candles.py                單日/雙日轉向陰陽燭形態 — bar reversal patterns
  classify.py                R1 bar classification, R2 內困 deferral
  pivots.py                  R3 轉角位, R4 內困 run resolution
  legs.py                    R5 段 determination
  lines.py                   R6-R9 生死線 ladder construction
  indicators.py               RSI / MACD / DMI (Wilder's smoothing)
  engine.py                   orchestrates the causal forward pass
  viz.py                      renders a standalone HTML chart
  data.py                     OHLCV loading (US equities + 388.HK)
  watchlist.py                watchlists.json CRUD
  watchlist_layout.py         watchlist_layout.json — sidebar order/sections
  alerts_log.py                data/alerts_log.json — persisted alert history
  serve.py                    local FastAPI server (127.0.0.1, no auth)

out/                         generated static chart output (gitignored)
data/                        OHLCV cache + alerts_log.json (gitignored)
watchlists.json              tracked symbols, editable from the served chart
watchlist_layout.json        sidebar display order and section grouping
```

## Commands

```
pip install -r requirements.txt # install dependencies

python -m sdx.serve             # http://127.0.0.1:8765
python -m sdx.viz 0388.HK       # standalone HTML chart, no server
python -m sdx.watchlist ls      # manage watchlists.json from the CLI
python -m sdx.alerts_log ls     # view persisted watchlist alert history
```

## Data sources

The engine supports two OHLCV sources, but only one is reachable from the
UI in this build:

- **yfinance** (`sdx/data.py`) — the default, and the only source the chart
  header exposes here. No API key needed. Covers US equities and 388.HK,
  daily bars only.
- **Webull** (`sdx/providers/webull.py`, developer.webull.hk OpenAPI) — the
  code still supports it (selectable intervals `5m/15m/30m/1h/4h/D/M/Y`,
  true MQTT push streaming for the currently-forming bar, `D` bars always
  seeded from yfinance regardless of source), but the header's data-source
  toggle is hidden, so it's not accessible from this chart. The 生死線
  ladder only runs on `D`, `4h`, `M`, `Y` (`webull_provider.LADDER_INTERVALS`);
  other intervals are chart + indicators only, no ladder. Live streaming is
  not offered for `M`/`Y` (no trading-calendar anchor for a "forming" bar).

### Webull API setup

Only needed if you want to use `source=webull` — yfinance-only usage needs
none of this.

1. Register an app at **developer.webull.hk** (covers both US-listed
   equities and 388.HK under one app key).
2. Copy `.env.example` to `.env` in the repo root (already gitignored —
   never commit real keys) and fill in:

   ```
   WEBULL_APP_KEY=...
   WEBULL_APP_SECRET=...
   WEBULL_REGION=hk
   ```

3. **Paper-trading keys only** — a PaperTrading app key authenticates
   against separate sandbox hosts the SDK doesn't default to, so also set:

   ```
   WEBULL_API_ENDPOINT=api.sandbox.webull.hk
   WEBULL_MQTT_ENDPOINT=data-api.sandbox.webull.hk
   ```

   Leave both unset for a production/live app key.

`sdx.serve` loads `.env` automatically on startup. Without
`WEBULL_APP_KEY`/`WEBULL_APP_SECRET` set, selecting `source=webull` fails
with a `WebullNotConfigured` error — yfinance keeps working either way.
