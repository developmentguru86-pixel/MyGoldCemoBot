# goldbot — systematisches Gold-System, serverlos vom Handy

Trend-Ensemble (Time-Series-Momentum, 3 Lookbacks) mit Volatility-Targeting und fraktionalem Kelly
auf H4-Bars. Gleicher Code für Sizing, Kosten, Rebalance-Regel und Kill-Switches im Backtest und live.

**Kein Renditeversprechen.** Das System optimiert Sharpe nach Kosten, nicht €/Tag. Ob es einen Edge hat,
zeigt der Walk-Forward-Report auf echten Daten – nicht dieser Text.

## Weg B: ohne eigenen Rechner (GitHub Actions + Phemex Testnet) — alles vom Handy

Der Bot läuft alle 4 Stunden als Cron-Job in GitHub Actions, handelt Gold als XAU/USDT-Perpetual im
OKX-Demo-Trading und meldet jeden Trade per Telegram. Kein Server, kein PC, kostenlos.
(Bybit, Binance und Kraken-Futures-Demo blocken GitHub-Runner per Geo-Sperre – gemessen, siehe `reports/probe*.log`. OKX-Demo antwortet.)

**Einmalig einrichten (≈ 15 Minuten, alles im Handy-Browser):**

1. **Phemex Testnet** (testnet.phemex.com, Browser): registrieren mit E-Mail – kein KYC, getrenntes
   Testsystem → Account → *API Management* → Key erstellen, Berechtigung *Trade* → API Key ID und
   API Secret kopieren (Secret wird nur einmal gezeigt).
2. **Telegram**: @BotFather → `/newbot` → Token kopieren. Dann @userinfobot anschreiben → deine
   Chat-ID kopieren. Deinen neuen Bot einmal anschreiben (sonst darf er dir nicht schreiben).
3. **GitHub** (github.com): neues **privates** Repo `goldbot` → *Add file → Upload files* → alle
   Dateien aus dem ZIP hochladen (im Handy-Browser „Desktop-Website anfordern"; Ordnerstruktur
   inkl. `.github/workflows/` muss erhalten bleiben – am einfachsten das ZIP am Handy entpacken und
   den Ordnerinhalt hochladen).
   Alternative: Fine-grained Token (nur dieses Repo, „Contents: write", 1 Tag gültig) an Claude
   geben, Claude pusht das Repo von dort aus.
4. Repo → *Settings → Secrets and variables → Actions → New repository secret*, vier Stück:
   `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` (`EXCHANGE_API_PASSPHRASE` nur für OKX).
5. Repo → *Actions* → ggf. „Enable workflows" → Workflow **backtest** → *Run workflow*.
   Nach 2–5 Minuten kommt der Report mit Chart auf Telegram, und `config.yaml` ist auf die
   echten Exchange-Werte (Lot-Step, Gebühr, Spread, Funding) synchronisiert.
6. Workflow **goldbot** → *Run workflow* mit `action = status`. Kommt die Statusmeldung auf Telegram,
   läuft ab jetzt der Cron automatisch: alle 4 Stunden ein Lauf, Trades und Kill-Switches als Push,
   um 00:xx UTC ein Tagesbericht.

**Bedienung vom Handy** (GitHub → Actions → goldbot → Run workflow):
`status` Konto/Position/letzte Entscheidungen · `flatten` alles schließen + pausieren ·
`reset_halt` nach einem Kill-Switch weiterhandeln · `trade` manueller Lauf.
Alles Sichtbare steht auch in `logs/journal_live.csv` im Repo (wird nach jedem Lauf committet).

**Was das für dich heißt:** Der Bot ist aktiv, sobald der Status-Lauf klappt. Er entscheidet
6× am Tag. In den ersten Tagen passiert sichtbar wenig – korrekt so. Das Demo-Konto zeigt, was das
System tut, Plus wie Minus.

**Sicherungen:** `exchange.demo: true` in `config.yaml` – Echtgeld verweigert der Bot ohne explizites
Flag. GitHub deaktiviert Cron-Workflows nach 60 Tagen ohne Repo-Aktivität; da der Bot seinen State
committet, bleibt das Repo aktiv. Wenn zwei Läufe kollidieren würden, wartet der zweite.

**Grenzen dieses Wegs:** GitHub-Cron kann um Minuten verzögern (bei H4 irrelevant). Das XAU-Perp ist
jung; der Backtest nutzt PAXG/USDT-Spot als längere Preis-Historie desselben Basiswerts (Kosten
weiterhin vom Perp modelliert). Ob OKX private Demo-Endpunkte von US-Runnern annimmt, zeigt der erste
`status`-Lauf; Ausweichbörsen mit Gold-Perps und Demo laut Probe: Kraken Futures, MEXC, BingX, Phemex.

---

## Weg A: eigener Windows-Rechner / VPS mit MetaTrader 5 (Swissquote)

Config: `config_mt5.yaml` (nach `config.yaml` kopieren). Rest wie unten.

## Struktur

```
config.yaml               alle Parameter (Kontrakt, Strategie, Kosten, Risiko, Bridge)
.github/workflows/      bot.yml (Cron alle 4h + Aktionen), backtest.yml
goldbot/
  strategy.py             Signale + Exposure (kein Look-ahead, per Test abgesichert)
  sizing.py               Exposure -> Lots (Min-Lot, Step, Leverage-Cap)
  costs.py                Spread/Slippage/Swap
  risk.py                 Kill-Switches (Tagesverlust, Max-DD, Bridge-Ausfälle)
  backtest.py             sequentielle Engine, Walk-Forward, Rebalance-Regel
  metrics.py              CAGR, Sharpe, Sortino, MaxDD, PSR, Block-Bootstrap
  live.py                 Bar-Close-Loop, Reconciliation, Journal
  broker/                 Interface, MT5-HTTP-Client, ccxt-Exchange-Broker, Paper-Broker
  supervisor.py           Telegram-Fernbedienung für Weg A (Dauerprozess)
  notify.py               Telegram-Push für Weg B (serverlos)
bridge/mt5_bridge.py      Referenz-Bridge (Flask + MetaTrader5), Endpoint-Vertrag
scripts/                  run_once (serverlos), fetch_history(_ccxt), run_backtest, run_live, bootstrap, status
tests/                    pytest (8 Tests, u.a. No-Look-ahead, Kill-Switches, Lot-Rundung)
```

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

Windows-Rechner mit MT5-Terminal (Swissquote-Demo eingeloggt):

```bash
pip install -r bridge/requirements-bridge.txt
python bridge/mt5_bridge.py --host 127.0.0.1 --port 5000
```

Wenn du deine bestehende `mt5_bridge.py` behalten willst: die Endpoints aus dem Docstring in
`bridge/mt5_bridge.py` sind der Vertrag, den `goldbot/broker/mt5_http.py` erwartet.

## Workflow (in dieser Reihenfolge)

**1. Echte Historie ziehen**
```bash
python scripts/fetch_history.py --years 6
```
Gibt `symbol_info` aus → `contract.*` in `config.yaml` eintragen (Preflight bricht sonst ab) und die
Swap-Werte in annualisierte % vom Notional umrechnen (`costs.swap_*_annual`). Median-Spread der Daten
mit `costs.spread` abgleichen.

**2. Backtest + Walk-Forward + Bootstrap**
```bash
python scripts/run_backtest.py --walk-forward --bootstrap
```
Walk-Forward: 2 Jahre Training, 6 Monate Test, Grid über Lookbacks × Target-Vol, Auswahl nach
In-Sample-Sharpe, OOS-Equity wird verkettet. Report in `reports/` (JSON, CSVs, PNG).

**Go/No-Go für Paper** — alles auf der *Walk-Forward-OOS-Kurve*, nicht in-sample:
- Sharpe ≥ 0.8 nach Kosten, PSR(>0) ≥ 0.90
- Max Drawdown ≤ 20 %
- Kosten (`total_cost + total_swap`) < 40 % des Brutto-P&L
- Mindestens 100 Trades OOS
- Kein Fenster mit `halted_at`

**3. Paper (echte Daten, simulierte Fills)** — mindestens 4 Wochen
```bash
python scripts/run_live.py --mode paper
```

**Go/No-Go für Live:** Paper-Journal (`logs/journal_paper.csv`) zeigt Trades, deren Richtung und
Größe mit einem parallel gerechneten Backtest desselben Zeitraums übereinstimmen; keine
Bridge-Ausfälle; realer Spread ≤ `risk.max_spread` in > 95 % der Bar-Closes.

**4. Live**
```bash
python scripts/run_live.py --mode live --dry-run   # loggt nur, sendet nichts
python scripts/run_live.py --mode live             # sendet Orders
```
Als Windows-Dienst / mit `nssm` laufen lassen. Nach Halt: `--reset-halt` erst nach Ursachenanalyse.

## Kill-Switches (identisch in Backtest und Live)

| Regel | Default | Wirkung |
|---|---|---|
| Tagesverlust | 3 % vom Tagesstart | flatten, keine neuen Trades bis Tageswechsel |
| Max Drawdown vom Hoch | 20 % | flatten, dauerhafter Halt (manueller Reset) |
| Spread-Guard | 0.80 USD/oz | kein neuer Einstieg (Glattstellen bleibt erlaubt) |
| Bridge-Ausfälle | 3 in Folge | Halt, keine Orders |

## Bekannte Grenzen — ehrlich

1. **Lot-Granularität bei 5.000 €.** 0,01 Lot = 1 oz ≈ 3.500–4.000 USD Notional ≈ 60–80 % des Kontos.
   Sizing ist damit faktisch binär (0 / 0,01 / 0,02 Lot); Vol-Targeting und Kelly greifen erst ab
   ~20.000 € sauber. Das siehst du im Backtest an `time_in_market` und `avg_abs_lots`.
2. **Kontowährung.** Alle Rechnungen in Kontowährung des MT5-Kontos. Bei EUR-Konto rechnet MT5 die
   USD-P&L selbst um; Backtest ignoriert EUR/USD-Drift (bei dieser Größenordnung zweitrangig).
3. **Swaps** sind broker-spezifisch (Punkte oder %, Triple-Swap mittwochs). Echte Werte aus
   `symbol_info` nehmen, nicht die Defaults.
4. **Demo ≠ Live.** Demo-Fills sind besser als echte. Erst ab Live-Journal weißt du deine Slippage.
5. **Zeitstempel** sind MT5-Serverzeit, als UTC etikettiert. Tagesgrenzen der Risikoregeln folgen
   der Serverzeit. Konsistent, aber nicht Wien-Lokalzeit.
6. **Walk-Forward** stellt die Position an jedem Fenster-Start neu her (leicht überschätzte Kosten).
7. `data/SYNTHETIC_H4.csv` ist **synthetisch** (GARCH + Regime-Drift) und existiert nur, damit die
   Pipeline ohne Bridge durchläuft. Zahlen daraus bedeuten nichts.

## Was das System *nicht* macht

Kein fixes Tagesziel, kein Martingale, kein Nachlegen bei Verlust, kein Grid, kein Hebel über
`max_leverage`. Wenn der Walk-Forward auf echten Daten negativ ist, ist die richtige Handlung:
nicht live gehen.
