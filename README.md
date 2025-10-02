📊 Stock Screener Python

Ein Python-Tool zum Screening von Aktien aus großen Indizes wie S&P 500, Nasdaq-100, DAX40 und BMV IPC.
Das Skript analysiert fundamentale Kennzahlen (Free Cash Flow, EV/EBITDA, ROE, Piotroski F-Score etc.) und erstellt eine Value-, Quality- und Growth-Scorecard.

✨ Features

Screening von großen Indizes: S&P 500, Nasdaq-100, DAX40, BMV IPC

Berechnung zentraler Kennzahlen:

Free Cash Flow, Margen, ROE, D/E, EV/EBITDA

Dividendenrendite, Buybacks, Piotroski-F-Score

Umsatz-CAGR (3Y, 5Y), OCF-Wachstum

Scoring-Modell (Value / Quality / Growth)

Exportierbare Ergebnisse: CSV, Excel, JSON, SQLite, Parquet

Robust gegen fehlende Daten (Retry-Mechanismus, Error-Handling)

🚀 Installation
git clone https://github.com/<dein-user>/stock-screener-python.git
cd stock-screener-python

# Virtuelle Umgebung (empfohlen)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt

⚡ Verwendung

Screening des S&P 500 und Export nach CSV:

python screener.py --universes sp500 --out results.csv


Mehrere Indizes kombinieren:

python screener.py --universes sp500,nasdaq100,dax40 --out results.xlsx


Nur bestimmte Aktien:

python screener.py --universes sp500 --only AAPL,MSFT --out apple_msft.csv


Top 20 nach Score:

python screener.py --universes sp500 --top-n 20 --out top20.csv

⚙️ Parameterübersicht
Parameter	Beschreibung	Beispiel
--universes	Indizes auswählen	sp500,nasdaq100,dax40
--only	Nur bestimmte Ticker screenen	AAPL,MSFT
--out	Ausgabedatei (CSV/Excel/JSON/…)	results.xlsx
--top-n	Top-N Aktien nach Score	20
--fcf-margin-min	Mindest-FCF-Marge (Standard: 20 %)	0.25


📊 Beispielausgabe
Eine Beispieldatei Aktien-Scores ist im Repository enthalten:  
➡ [stock_screener_sample.xlsx](./stock_screener_sample.xlsx)




📄 Beispielausgabe                              

| PASS | Score Overall | Ticker | Name                         | Price   | FCF Yield | ROE TTM | Piotroski F | EV/EBITDA | Rev CAGR (3Y) |
| ---- | ------------- | ------ | ---------------------------- | ------- | --------- | ------- | ----------- | --------- | ------------- |
| ✅    | 61.7          | FICO   | Fair Isaac Corporation       | 1548.36 | 2.0 %     | 72.4 %  | 8           | 44.2x     | 9.3 %         |
| ✅    | 56.9          | MTD    | Mettler-Toledo International | 1209.54 | 3.3 %     | 15.7 %  | 7           | 22.8x     | 1.4 %         |
| ✅    | 50.6          | HOLX   | Hologic Inc.                 | 65.79   | 6.3 %     | 11.5 %  | 8           | 12.2x     | –10.6 %       |
| ✅    | 48.3          | PM     | Philip Morris International  | 164.00  | 3.5 %     | 8.6 %   | 7           | 17.1x     | 6.4 %         |


🛠️ Tech-Stack

Python 3.9+

pandas

numpy

yfinance

requests

tqdm

⚠️ Hinweise & Limitierungen

Datenquelle: Yahoo Finance via yfinance → gelegentlich fehlende/fehlerhafte Daten möglich

Fokus: Fundamentale Kennzahlen, nicht für Intraday-Trading gedacht

Ergebnisse = Arbeitsprobe/Analyse-Tool, keine Anlageberatung

📜 Lizenz

MIT-Lizenz – frei nutzbar, veränderbar und weitergebbar (mit Quellenhinweis).

