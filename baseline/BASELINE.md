# Referenz-Baseline v3 (eingefroren 13.09.2026)

Jede Änderung wird ab jetzt gegen exakt diese Zahlen verglichen. Gleiche Daten, gleiche Kosten,
gleiche Walk-Forward-Splits, gleiche Purged-CV-Folds.

## Konfiguration
`baseline/config_baseline_v3.yaml` — D1, long-only, 5 Sleeves (XAU 10 / BTC 25 / ETH 25 / SOL 20 / XRP 20),
lookbacks 20/60/120 Tage, einstimmiger Einstieg, kostenbewusstes Kelly, exposure_scale 24.

## Gemessen (Walk-Forward OOS, Langhistorie, ungehebelt, preisrelative Kosten)

| Sleeve | Historie | WF Sharpe | CAGR | MaxDD | CV-Median | pos. Folds |
|---|---|---|---|---|---|---|
| XAU | 26,0 J | 0,135 | +0,6 % | −17,6 % | 0,449 | 4/6 |
| BTC | 12,0 J | 1,161 | +8,4 % | −12,0 % | 1,064 | 5/6 |
| ETH | 8,9 J | 0,807 | +5,0 % | −9,8 % | 0,665 | 3/6 |
| SOL | 6,4 J | 0,862 | +3,3 % | −3,5 % | 0,393 | 4/6 |
| XRP | 8,9 J | 0,833 | +3,6 % | −4,8 % | 0,347 | 4/6 |

Portfolio (Union-Fenster, config-Gewichte): Sharpe 0,951 · CAGR 1,9 % · Vol 2,0 % · MaxDD −4,3 % · PSR 1,0 · P(Verlust 1J) 26 %
Gates: 8/14 bestanden.

Korrelationen der OOS-Strategierenditen: XAU zu allem ~0,0 · BTC-ETH 0,20 · BTC-SOL 0,10 · ETH-XRP 0,27 · SOL-XRP 0,03

## Negative Ergebnisse (bereits getestet, verworfen)
- Meta-Labeling auf D1: 69 Events in 6 Jahren, Median 0,67 -> 0,05. Zu wenig Lerndaten.
- Transfer-Meta (H4-Training -> D1-Anwendung): 160 Events, Median 0,22 -> -0,09, besser in 4/18 Folds.
- Confidence-Sizing (Tiers nach Signalstärke): vom Selektor in 6/8 Fenstern verworfen.
- ER-Regimefilter auf D1: vom Selektor in 7/8 Fenstern verworfen (auf H4 dagegen gewählt).
- Dynamische Allokation nach Trailing-OOS-Sharpe: 0,19 gegen 1,14 statisch.
- 15M-Zeitrahmen: -15 bis -19 % p.a., Sharpe -3 bis -5, P(Verlust) 100 %.
- Shorts: in jedem Asset und fast jedem Fenster negativ. Seit v3 deaktiviert.
