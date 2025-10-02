#!/usr/bin/env python3



from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------- Optional dependencies ---------------------------- #
try:
    import requests
    from requests.adapters import HTTPAdapter

    try:
        from urllib3.util.retry import Retry  # urllib3>=1.26
    except Exception:
        from urllib3.util import Retry  # older layout
except Exception as e:
    print("This script requires 'requests'. Install with: pip install requests", file=sys.stderr)
    raise

try:
    import yfinance as yf
except Exception as e:
    print("This script requires 'yfinance'. Install with: pip install yfinance", file=sys.stderr)
    raise

try:
    from tqdm import tqdm

    TQDM = True
except Exception:
    TQDM = False

try:
    import requests_cache  # optional

    HAS_CACHE = True
except Exception:
    HAS_CACHE = False

pd.options.mode.chained_assignment = None

# ---------------------------- Logging ---------------------------- #
LOGGER = logging.getLogger("screener")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.INFO
    if quiet:
        level = logging.WARNING
    if verbose:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------- Networking helpers ---------------------------- #

def make_retry_session(
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        https_proxy: Optional[str] = None,
        cache_name: Optional[str] = None,
        cache_expire: int = 3600,
        timeout: int = 20,
) -> requests.Session:
    """Create a hardened HTTP session; optionally wrap with requests-cache."""
    if HAS_CACHE and cache_name:
        session = requests_cache.CachedSession(cache_name, expire_after=cache_expire)
    else:
        session = requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    session.headers.update(headers)

    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        status=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if https_proxy:
        session.proxies.update({"https": https_proxy, "http": https_proxy})

    # Attach a default timeout to the session via a wrapper


    return session


def safe_call(getter, retries: int = 2, wait: float = 1.0):
    """Call a zero-arg callable with retries and exponential backoff."""
    last_exc = None
    for i in range(retries + 1):
        try:
            return getter()
        except Exception as e:
            last_exc = e
            status = getattr(getattr(e, "response", None), "status_code", None)
            sleep_s = wait * (2.0 ** i)
            if status == 429:
                sleep_s *= 3
            jitter = random.random() * 0.5 * wait
            time.sleep(sleep_s + jitter)
    if last_exc:
        raise last_exc
    return None


# ---------------------------- Index universe loaders ---------------------------- #
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NASDAQ100 = "https://en.wikipedia.org/wiki/Nasdaq-100"
WIKI_DAX40 = "https://en.wikipedia.org/wiki/DAX"
WIKI_BMV_IPC_VARIANTS = [
    "https://en.wikipedia.org/wiki/IPC_(Mexican_Stock_Exchange)",
    "https://en.wikipedia.org/wiki/S%26P/BMV_IPC",
]


def _read_html_tables(url: str, session: requests.Session) -> List[pd.DataFrame]:
    resp = session.get(url)
    resp.raise_for_status()
    # Wrap in StringIO to avoid pandas FutureWarning on HTML parsing
    return pd.read_html(StringIO(resp.text))


def _extract_tickers_from_tables(tables: List[pd.DataFrame], candidates: Iterable[str]) -> Optional[List[str]]:
    cand_lower = {c.lower() for c in candidates}
    for tbl in tables:
        cols = [str(c) for c in tbl.columns]
        low = {c.lower() for c in cols}
        inter = low.intersection(cand_lower)
        if inter:
            col = [c for c in cols if c.lower() in inter][0]
            syms = (
                tbl[col]
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.replace(".", "-", regex=False)
                .str.strip()
                .tolist()
            )
            return syms
    return None


def load_sp500(session: requests.Session) -> List[str]:
    """Return S&P 500 tickers (no suffix)."""
    tables = _read_html_tables(WIKI_SP500, session)
    syms = _extract_tickers_from_tables(tables, candidates=["Symbol", "Ticker"]) \
           or _extract_tickers_from_tables(tables, candidates=["Code"])  # rare alt
    if syms:
        return [s.replace(".", "-") for s in syms]
    raise RuntimeError("Could not locate constituents table for S&P 500")


def load_nasdaq100(session: requests.Session) -> List[str]:
    tables = _read_html_tables(WIKI_NASDAQ100, session)
    syms = _extract_tickers_from_tables(tables, candidates=["Ticker", "Symbol"])  # some pages say Ticker
    if syms:
        return [s.replace(".", "-") for s in syms]
    raise RuntimeError("Could not locate constituents table for Nasdaq-100")


def load_dax40(session: requests.Session) -> List[str]:
    tables = _read_html_tables(WIKI_DAX40, session)
    syms = _extract_tickers_from_tables(tables, candidates=["Ticker symbol", "Ticker", "Symbol"])
    if syms:
        syms = [s if s.endswith(".DE") else f"{s}.DE" for s in syms]
        return syms
    # Soft fallback minimal set (kept intentionally short)
    return [
        'SAP.DE', 'SIE.DE', 'ALV.DE', 'LIN.DE', 'MUV2.DE', 'BMW.DE', 'DTE.DE', 'BAS.DE', 'IFX.DE', 'VOW3.DE',
        'RWE.DE', 'HEN3.DE', 'ADS.DE', 'MBG.DE', 'BEI.DE', 'HEI.DE', 'FME.DE', 'PUM.DE', 'BAYN.DE', 'FRE.DE'
    ]


def load_bmv_ipc(session: requests.Session) -> List[str]:
    """Return S&P/BMV IPC tickers (Yahoo uses .MX suffix)."""
    for url in WIKI_BMV_IPC_VARIANTS:
        try:
            tables = _read_html_tables(url, session)
            syms = _extract_tickers_from_tables(tables, candidates=["Ticker", "Symbol"]) or []
            if syms:
                out: List[str] = []
                for s in syms:
                    s = s.replace("*", "").replace(" ", "").strip()
                    if not s:
                        continue
                    if not s.upper().endswith('.MX'):
                        s = f"{s}.MX"
                    out.append(s)
                # de-dup while preserving order
                seen, uniq = set(), []
                for x in out:
                    if x not in seen:
                        seen.add(x);
                        uniq.append(x)
                return uniq
        except Exception:
            continue
    # Fallback: a few liquid names
    return [
        "AMXL.MX", "WALMEX.MX", "BIMBOA.MX", "CEMEXCPO.MX", "GFNORTEO.MX",
        "GMEXICOB.MX", "KIMBERA.MX", "GRUMAB.MX", "ASURB.MX", "GAPB.MX",
    ]


def build_universe(universes: Sequence[str], session: requests.Session) -> List[str]:
    """Build a combined ticker universe from named indices; returns de-duped ordered list."""
    tickers: List[str] = []
    for u in universes:
        u_lower = u.lower().strip()
        try:
            if u_lower in {'sp500', 's&p500', 's&p 500'}:
                tickers += load_sp500(session)
            elif u_lower in {'nasdaq100', 'nasdaq-100', 'ndx'}:
                tickers += load_nasdaq100(session)
            elif u_lower in {'dax40', 'dax'}:
                tickers += load_dax40(session)
            elif u_lower in {'bmv_ipc', 'ipc', 'mexico', 'mx'}:
                tickers += load_bmv_ipc(session)
            else:
                LOGGER.warning("Unknown universe '%s'. Skipping.", u)
        except Exception as e:
            LOGGER.warning("Failed to load %s: %s", u, e)

    # De-duplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for s in tickers:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


# ---------------------------- Accounting helpers ---------------------------- #

def _find_row_key(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    idx_lower = {str(i).lower(): i for i in df.index}
    for c in candidates:
        key = idx_lower.get(c.lower())
        if key is not None:
            return key
    # try partial contains match
    for idx_l, orig in idx_lower.items():
        for c in candidates:
            if c.lower() in idx_l:
                return orig
    return None


def _series_row(df: pd.DataFrame, row_keys: List[str]) -> Optional[pd.Series]:
    key = _find_row_key(df, row_keys)
    if key is None:
        return None
    s = df.loc[key]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[0]
    # Ensure Series ordered most-recent first (yfinance already is, but normalize)
    return s


def _sum_last_n(df: pd.DataFrame, row_keys: List[str], n: int = 4) -> float:
    try:
        s = _series_row(df, row_keys)
        if s is None:
            return np.nan
        series = pd.to_numeric(s, errors='coerce').iloc[:n]
        return float(series.dropna().sum()) if not series.empty else np.nan
    except Exception:
        return np.nan


def _sum_range(df: pd.DataFrame, row_keys: List[str], start: int, n: int) -> float:
    try:
        s = _series_row(df, row_keys)
        if s is None:
            return np.nan
        series = pd.to_numeric(s, errors='coerce').iloc[start:start + n]
        return float(series.dropna().sum()) if not series.empty else np.nan
    except Exception:
        return np.nan


def _latest(df: pd.DataFrame, row_keys: List[str]) -> float:
    try:
        s = _series_row(df, row_keys)
        if s is None:
            return np.nan
        series = pd.to_numeric(s, errors='coerce').iloc[:1]
        return float(series.iloc[0]) if len(series) else np.nan
    except Exception:
        return np.nan


def _avg_last_n(df: pd.DataFrame, row_keys: List[str], n: int = 4) -> float:
    try:
        s = _series_row(df, row_keys)
        if s is None:
            return np.nan
        series = pd.to_numeric(s, errors='coerce').iloc[:n]
        return float(series.dropna().mean()) if not series.empty else np.nan
    except Exception:
        return np.nan


def _ratio_safe(a: float, b: float) -> float:
    if a is None or b is None or np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    try:
        return float(a) / float(b)
    except Exception:
        return np.nan


def _cagr(first: float, last: float, years: float) -> float:
    if any([x is None for x in (first, last)]) or any([np.isnan(x) for x in (first, last)]) or first <= 0 or years <= 0:
        return np.nan
    try:
        return (last / first) ** (1.0 / years) - 1.0
    except Exception:
        return np.nan


@dataclass
class Metrics:
    ticker: str
    long_name: Optional[str]
    price: Optional[float]
    market_cap: Optional[float]
    currency: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    beta: Optional[float]

    # Core metrics
    fcf_ttm: float
    revenue_ttm: float
    fcf_margin: float
    debt_latest: float
    equity_avg_qtr: float
    de_ratio: float

    net_income_ttm: float
    cfo_ttm: float
    roe_ttm: float

    op_income_ttm: float
    ebitda_ttm: float

    op_margin_cur: float
    op_margin_prev: float
    gross_margin_cur: float
    gross_margin_prev: float

    trailing_pe: float
    price_to_book: float
    ev_to_ebitda: float
    fcf_yield: float

    # Debt quality extras
    interest_expense_ttm: float
    ebit_ttm: float
    cash_latest: float
    net_debt_latest: float
    interest_coverage: float
    netdebt_ebitda: float
    buybacks_ttm: float
    dividends_ttm: float
    shareholder_yield: float

    # Growth
    rev_cagr_3y: float
    rev_cagr_5y: float
    ocf_growth_ttm: float

    # Additional valuation
    ev: float
    ev_to_sales: float
    ev_to_fcf: float
    dividend_yield_info: float

    # Composite quality
    piotroski_f: Optional[int]
    good_debt_proxy: bool

    mgmt_score: int
    undervalued_flag: bool


# ---------------------------- Core computation per ticker ---------------------------- #
class SoftRateLimiter:
    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / max(1e-9, rate_per_sec)
        self._next = 0.0
        try:
            import threading
            self._lock = threading.Lock()
        except Exception:
            self._lock = None

    def wait(self):
        now = time.monotonic()
        if self._lock:
            with self._lock:
                wait_for = max(0.0, self._next - now)
                if wait_for > 0:
                    time.sleep(wait_for)
                jitter = random.random() * self.min_interval
                self._next = max(now, self._next) + self.min_interval + jitter
        else:
            time.sleep(self.min_interval)


def _safe_info(t: yf.Ticker) -> Dict:
    """Get info dict robustly across yfinance versions."""
    info: Dict = {}
    try:
        # Newer yfinance
        info = t.get_info()
    except Exception:
        try:
            info = t.info or {}
        except Exception:
            info = {}
    return info or {}




def compute_metrics(ticker: str, session: requests.Session, retries: int = 2, retry_wait: float = 1.0) -> Metrics:
    t = yf.Ticker(ticker, session=session)

    # Try fast_info first (cheap); then info fallback
    long_name = None
    price = None
    market_cap = None
    shares_out = None
    currency = None
    beta = None
    sector = None
    industry = None
    country = None

    try:
        fi = t.fast_info
        price = fi.get('last_price') if isinstance(fi, dict) else getattr(fi, 'last_price', None)
        market_cap = fi.get('market_cap') if isinstance(fi, dict) else getattr(fi, 'market_cap', None)
        shares_out = fi.get('shares_outstanding') if isinstance(fi, dict) else getattr(fi, 'shares_outstanding', None)
        currency = fi.get('currency') if isinstance(fi, dict) else getattr(fi, 'currency', None)
        beta = fi.get('beta') if isinstance(fi, dict) else getattr(fi, 'beta', None)
    except Exception:
        fi = {}

    info: Dict = _safe_info(t)
    if not long_name:
        long_name = info.get('longName') or info.get('shortName')
    if market_cap is None:
        market_cap = info.get('marketCap')
    if price is None:
        price = info.get('currentPrice') or info.get('regularMarketPrice')
    if currency is None:
        currency = info.get('currency')
    sector = info.get('sector') or sector
    industry = info.get('industry') or industry
    country = info.get('country') or country
    if beta is None:
        beta = info.get('beta')

    # Price fallback via history
    if price is None:
        for i in range(retries + 1):
            try:
                hist = t.history(period='1d', auto_adjust=False)
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                    break
            except Exception:
                time.sleep(retry_wait * (1.5 ** i))

    if (market_cap is None or (
            isinstance(market_cap, float) and np.isnan(market_cap))) and price is not None and shares_out:
        try:
            market_cap = float(price) * float(shares_out)
        except Exception:
            pass

    # Pull statements with retries
    def _get(attr):
        return safe_call(lambda: getattr(t, attr), retries=retries, wait=retry_wait)

    try:
        q_fin = _get('quarterly_financials')
    except Exception:
        q_fin = pd.DataFrame()
    try:
        q_cf = _get('quarterly_cashflow')
    except Exception:
        q_cf = pd.DataFrame()
    try:
        q_bs = _get('quarterly_balance_sheet')
    except Exception:
        q_bs = pd.DataFrame()
    try:
        a_fin = _get('financials')  # annual
    except Exception:
        a_fin = pd.DataFrame()
    try:
        a_bs = _get('balance_sheet')  # annual BS for assets
    except Exception:
        a_bs = pd.DataFrame()

    # TTM core figures
    revenue_ttm = _sum_last_n(q_fin, ['Total Revenue', 'Total revenue', 'Revenue'], 4)
    op_income_ttm = _sum_last_n(q_fin, ['Operating Income', 'Operating income'], 4)
    gross_profit_ttm = _sum_last_n(q_fin, ['Gross Profit', 'Gross profit'], 4)
    net_income_ttm = _sum_last_n(q_fin, ['Net Income', 'Net income'], 4)

    cfo_ttm = _sum_last_n(q_cf, ['Total Cash From Operating Activities', 'Total Cash From Operating Activity',
                                 'Operating Cash Flow'], 4)
    capex_ttm = _sum_last_n(q_cf, ['Capital Expenditures', 'Capital Expenditure'], 4)
    fcf_ttm = np.nan
    if not (np.isnan(cfo_ttm) or np.isnan(capex_ttm)):
        # FCF = CFO + CapEx  (CapEx ist meist negativ → korrekter Abzug)
        fcf_ttm = float(cfo_ttm) + float(capex_ttm)

    # OCF growth (TTM vs prior TTM)
    cfo_prev_ttm = _sum_range(q_cf, ['Total Cash From Operating Activities', 'Total Cash From Operating Activity',
                                     'Operating Cash Flow'], 4, 4)
    ocf_growth_ttm = np.nan
    if not (np.isnan(cfo_prev_ttm) or np.isnan(cfo_ttm) or cfo_prev_ttm == 0):
        ocf_growth_ttm = (cfo_ttm - cfo_prev_ttm) / abs(cfo_prev_ttm)

    # Margins
    op_margin_ttm = _ratio_safe(op_income_ttm, revenue_ttm)
    gross_margin_ttm = _ratio_safe(gross_profit_ttm, revenue_ttm)
    fcf_margin = _ratio_safe(fcf_ttm, revenue_ttm)

    # Debt & equity
    total_debt_latest = _latest(q_bs, ['Total Debt', 'Total debt'])
    if np.isnan(total_debt_latest):
        short_lt_debt = _latest(q_bs, ['Short Long Term Debt', 'Short/Long Term Debt', 'Short Term Debt'])
        long_term_debt = _latest(q_bs, ['Long Term Debt', 'Long-term Debt'])
        if not np.isnan(short_lt_debt) or not np.isnan(long_term_debt):
            total_debt_latest = (0 if np.isnan(short_lt_debt) else short_lt_debt) + \
                                (0 if np.isnan(long_term_debt) else long_term_debt)

    equity_avg_q = _avg_last_n(q_bs, ['Total Stockholder Equity', 'Total Stockholders Equity', 'Stockholders Equity'],
                               4)
    de_ratio = _ratio_safe(total_debt_latest, equity_avg_q)

    # Debt quality pieces
    interest_expense_ttm = _sum_last_n(q_fin, ['Interest Expense', 'Interest expense', 'Interest Expense Non Operating',
                                               'Interest Expense Non-Operating'], 4)
    if not np.isnan(interest_expense_ttm):
        interest_expense_ttm = float(abs(interest_expense_ttm))  # use absolute for coverage

    # EBITDA
    ebitda_ttm = _sum_last_n(q_fin, ['Ebitda', 'EBITDA'], 4)
    depreciation_ttm = _sum_last_n(q_cf, ['Depreciation', 'Depreciation Amortization Depletion',
                                          'Depreciation & amortization'], 4)
    if (np.isnan(ebitda_ttm) or ebitda_ttm == 0) and (not np.isnan(op_income_ttm) and not np.isnan(depreciation_ttm)):
        ebitda_ttm = op_income_ttm + depreciation_ttm

    # Cash (for Net Debt)
    cash_latest = _latest(q_bs, ['Cash And Cash Equivalents', 'Cash and cash equivalents', 'Cash'])
    short_inv_latest = _latest(q_bs, ['Short Term Investments', 'Short-term Investments', 'Short Term Investment'])
    cash_like = (0 if np.isnan(cash_latest) else cash_latest) + (0 if np.isnan(short_inv_latest) else short_inv_latest)
    net_debt_latest = (0 if np.isnan(total_debt_latest) else total_debt_latest) - cash_like

    # Coverage/Leverage
    ebit_ttm = op_income_ttm  # approximation
    interest_coverage = _ratio_safe(ebit_ttm, interest_expense_ttm)
    netdebt_ebitda = _ratio_safe(net_debt_latest, ebitda_ttm)

    # Buybacks (positive number if cash spent on repurchases)
    buybacks_ttm_raw = _sum_last_n(q_cf, ['Repurchase Of Stock', 'Sale Purchase Of Stock', 'Sale/Purchase of Stock'], 4)
    buybacks_ttm = 0.0 if np.isnan(buybacks_ttm_raw) else -float(buybacks_ttm_raw)

    dividends_paid_ttm_raw = _sum_last_n(q_cf, ['Cash Dividends Paid', 'Common Stock Dividends Paid', 'Dividends Paid'],
                                         4)
    dividends_ttm = 0.0 if np.isnan(dividends_paid_ttm_raw) else -float(dividends_paid_ttm_raw)

    # Optionaler Fallback: wenn Dividenden im Cashflow fehlen, auf info['dividendYield'] zurückgreifen
    if dividends_ttm == 0.0 and market_cap and not np.isnan(market_cap):
        dy_info = info.get('dividendYield') if info else None
        if dy_info is not None and not np.isnan(dy_info) and dy_info > 0:
            # dividendYield ist eine Rate → TTM-Dividenden ≈ Rate * MarketCap
            dividends_ttm = float(dy_info) * float(market_cap)

    # Shareholder Yield = (Buybacks + Dividenden) / MarketCap
    shareholder_yield = np.nan
    if market_cap and not np.isnan(market_cap) and market_cap > 0:
        shareholder_yield = (buybacks_ttm + dividends_ttm) / float(market_cap)

    # ROE (approx.)
    # ROE (vorsichtig bei sehr kleinem/negativem EK)
    roe_ttm = np.nan
    if equity_avg_q is not None and not np.isnan(equity_avg_q) and equity_avg_q > 1e-6:
        roe_ttm = _ratio_safe(net_income_ttm, equity_avg_q)
    # sonst NaN lassen (bei Firmen mit stark negativem EK wegen Buybacks sind ROE-Werte sonst irreführend)

    # Trend checks (annual)
    op_margin_cur = np.nan
    op_margin_prev = np.nan
    gross_margin_cur = np.nan
    gross_margin_prev = np.nan

    rev_cagr_3y = np.nan
    rev_cagr_5y = np.nan

    try:
        if a_fin is not None and not a_fin.empty:
            cols = list(a_fin.columns)
            # Operating/Gross margins year-over-year (two most recent)
            if len(cols) >= 2:
                rev_key = _find_row_key(a_fin, ['Total Revenue', 'Revenue'])
                op_key = _find_row_key(a_fin, ['Operating Income'])
                gp_key = _find_row_key(a_fin, ['Gross Profit'])
                rev_cur = float(a_fin.loc[rev_key, cols[0]]) if rev_key else np.nan
                op_cur = float(a_fin.loc[op_key, cols[0]]) if op_key else np.nan
                gp_cur = float(a_fin.loc[gp_key, cols[0]]) if gp_key else np.nan
                rev_prev = float(a_fin.loc[rev_key, cols[1]]) if rev_key else np.nan
                op_prev = float(a_fin.loc[op_key, cols[1]]) if op_key else np.nan
                gp_prev = float(a_fin.loc[gp_key, cols[1]]) if gp_key else np.nan
                op_margin_cur = _ratio_safe(op_cur, rev_cur)
                op_margin_prev = _ratio_safe(op_prev, rev_prev)
                gross_margin_cur = _ratio_safe(gp_cur, rev_cur)
                gross_margin_prev = _ratio_safe(gp_prev, rev_prev)

            # CAGR based on endpoints (newest first)
            rev_row_key = _find_row_key(a_fin, ['Total Revenue', 'Revenue'])
            if rev_row_key is not None:
                series = pd.to_numeric(a_fin.loc[rev_row_key], errors='coerce').dropna()
                if len(series) >= 4:
                    rev_cagr_3y = _cagr(series.iloc[3], series.iloc[0], years=3)
                if len(series) >= 6:
                    rev_cagr_5y = _cagr(series.iloc[5], series.iloc[0], years=5)
    except Exception:
        pass

    trailing_pe = info.get('trailingPE') if info else np.nan
    price_to_book = info.get('priceToBook') if info else np.nan
    ev_to_ebitda = info.get('enterpriseToEbitda') if info else np.nan

    # Market-cap based yields are currency-less; EV computations
    fcf_yield = _ratio_safe(fcf_ttm, market_cap)
    ev = np.nan
    if market_cap is not None and not np.isnan(market_cap) and not np.isnan(net_debt_latest):
        ev = float(market_cap) + float(net_debt_latest)
    ev_to_sales = _ratio_safe(ev, revenue_ttm)
    ev_to_fcf = _ratio_safe(ev, fcf_ttm)

    # Dividend yield (info)
    dividend_yield_info = info.get('dividendYield') if info else np.nan

    # Management score proxy (0-4)
    mgmt_score = 0
    if not np.isnan(roe_ttm) and roe_ttm >= 0.10:
        mgmt_score += 1
    if not (np.isnan(cfo_ttm) or np.isnan(net_income_ttm)) and cfo_ttm > net_income_ttm:
        mgmt_score += 1
    if not (np.isnan(op_margin_cur) or np.isnan(op_margin_prev)) and op_margin_cur > op_margin_prev:
        mgmt_score += 1
    if not (np.isnan(gross_margin_cur) or np.isnan(gross_margin_prev)) and gross_margin_cur > gross_margin_prev:
        mgmt_score += 1

    good_debt_proxy = False
    if (not np.isnan(buybacks_ttm) and buybacks_ttm > 0) or (
            not np.isnan(interest_coverage) and interest_coverage >= 4):
        good_debt_proxy = True

    undervalued_flag = (
            (not np.isnan(fcf_yield) and fcf_yield >= 0.06) or
            (not np.isnan(trailing_pe) and trailing_pe < 15) or
            (not np.isnan(price_to_book) and price_to_book < 1.5) or
            (not np.isnan(ev_to_ebitda) and ev_to_ebitda < 10)
    )

    # ---------------- Piotroski F-Score (approx) ----------------
    fscore: Optional[int] = None
    try:
        score = 0
        # Profitability
        # ROA = NI / Total Assets (use annual assets if possible)
        total_assets = np.nan
        if a_bs is not None and not a_bs.empty:
            key = _find_row_key(a_bs, ['Total Assets', 'Total assets'])
            if key:
                total_assets = float(pd.to_numeric(a_bs.loc[key].iloc[0], errors='coerce'))
        if np.isnan(total_assets):
            total_assets = _latest(q_bs, ['Total Assets', 'Total assets'])
        roa = _ratio_safe(net_income_ttm, total_assets)
        if not np.isnan(roa) and roa > 0:
            score += 1
        if not np.isnan(cfo_ttm) and cfo_ttm > 0:
            score += 1
        # ΔROA > 0 (approx from annual NI, Assets year t vs t-1)
        try:
            if a_fin is not None and not a_fin.empty and a_bs is not None and not a_bs.empty:
                ni_row = _find_row_key(a_fin, ['Net Income'])
                ta_row = _find_row_key(a_bs, ['Total Assets', 'Total assets'])
                if ni_row and ta_row and len(a_fin.columns) >= 2 and len(a_bs.columns) >= 2:
                    ni_cur, ni_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_fin.loc[ni_row].iloc[:2]]
                    ta_cur, ta_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_bs.loc[ta_row].iloc[:2]]
                    roa_prev = _ratio_safe(ni_prev, ta_prev)
                    roa_cur = _ratio_safe(ni_cur, ta_cur)
                    if not (np.isnan(roa_prev) or np.isnan(roa_cur)) and roa_cur > roa_prev:
                        score += 1
        except Exception:
            pass
        # Accruals: CFO > NI
        if not (np.isnan(cfo_ttm) or np.isnan(net_income_ttm)) and cfo_ttm > net_income_ttm:
            score += 1

        # Leverage/Liquidity
        ltd_row = _find_row_key(a_bs,
                                ['Long Term Debt', 'Long-term Debt']) if a_bs is not None and not a_bs.empty else None
        ta_row = _find_row_key(a_bs, ['Total Assets', 'Total assets']) if a_bs is not None and not a_bs.empty else None
        if ltd_row and ta_row and len(a_bs.columns) >= 2:
            ltd_cur, ltd_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_bs.loc[ltd_row].iloc[:2]]
            ta_cur, ta_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_bs.loc[ta_row].iloc[:2]]
            lev_cur = _ratio_safe(ltd_cur, ta_cur)
            lev_prev = _ratio_safe(ltd_prev, ta_prev)
            if not (np.isnan(lev_cur) or np.isnan(lev_prev)) and lev_cur < lev_prev:
                score += 1
        ca_row = _find_row_key(a_bs, ['Total Current Assets',
                                      'Current Assets']) if a_bs is not None and not a_bs.empty else None
        cl_row = _find_row_key(a_bs, ['Total Current Liabilities',
                                      'Current Liabilities']) if a_bs is not None and not a_bs.empty else None
        if ca_row and cl_row and len(a_bs.columns) >= 2:
            ca_cur, ca_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_bs.loc[ca_row].iloc[:2]]
            cl_cur, cl_prev = [float(pd.to_numeric(x, errors='coerce')) for x in a_bs.loc[cl_row].iloc[:2]]
            cr_cur = _ratio_safe(ca_cur, cl_cur)
            cr_prev = _ratio_safe(ca_prev, cl_prev)
            if not (np.isnan(cr_cur) or np.isnan(cr_prev)) and cr_cur > cr_prev:
                score += 1
        # Shares not issued — optimistic default if we have shares info
        try:
            so_info = shares_out or (info.get('sharesOutstanding') if info else None)
            if so_info is not None:
                score += 1
        except Exception:
            pass

        # Operating efficiency
        if not (np.isnan(gross_margin_cur) or np.isnan(gross_margin_prev)) and gross_margin_cur > gross_margin_prev:
            score += 1
        if a_fin is not None and not a_fin.empty and a_bs is not None and not a_bs.empty and len(
                a_fin.columns) >= 2 and len(a_bs.columns) >= 2:
            rev_key = _find_row_key(a_fin, ['Total Revenue', 'Revenue'])
            ta_key = _find_row_key(a_bs, ['Total Assets', 'Total assets'])
            if rev_key and ta_key:
                sales_cur = float(pd.to_numeric(a_fin.loc[rev_key].iloc[0], errors='coerce'))
                assets_cur = float(pd.to_numeric(a_bs.loc[ta_key].iloc[0], errors='coerce'))
                sales_prev = float(pd.to_numeric(a_fin.loc[rev_key].iloc[1], errors='coerce'))
                assets_prev = float(pd.to_numeric(a_bs.loc[ta_key].iloc[1], errors='coerce'))
                at_cur = _ratio_safe(sales_cur, assets_cur)
                at_prev = _ratio_safe(sales_prev, assets_prev)
                if not (np.isnan(at_cur) or np.isnan(at_prev)) and at_cur > at_prev:
                    score += 1
        fscore = int(score)
    except Exception:
        fscore = None

    return Metrics(
        ticker=ticker,
        long_name=long_name,
        price=price,
        market_cap=market_cap,
        currency=currency,
        sector=sector,
        industry=industry,
        country=country,
        beta=beta,
        fcf_ttm=fcf_ttm,
        revenue_ttm=revenue_ttm,
        fcf_margin=fcf_margin,
        debt_latest=total_debt_latest,
        equity_avg_qtr=equity_avg_q,
        de_ratio=de_ratio,
        net_income_ttm=net_income_ttm,
        cfo_ttm=cfo_ttm,
        roe_ttm=roe_ttm,
        op_income_ttm=op_income_ttm,
        ebitda_ttm=ebitda_ttm,
        op_margin_cur=op_margin_cur,
        op_margin_prev=op_margin_prev,
        gross_margin_cur=gross_margin_cur,
        gross_margin_prev=gross_margin_prev,
        trailing_pe=trailing_pe,
        price_to_book=price_to_book,
        ev_to_ebitda=ev_to_ebitda,
        fcf_yield=fcf_yield,
        interest_expense_ttm=interest_expense_ttm,
        ebit_ttm=ebit_ttm,
        cash_latest=cash_like,
        net_debt_latest=net_debt_latest,
        interest_coverage=interest_coverage,
        netdebt_ebitda=netdebt_ebitda,
        buybacks_ttm=buybacks_ttm,
        dividends_ttm=dividends_ttm,
        shareholder_yield=shareholder_yield,
        rev_cagr_3y=rev_cagr_3y,
        rev_cagr_5y=rev_cagr_5y,
        ocf_growth_ttm=ocf_growth_ttm,
        ev=ev,
        ev_to_sales=ev_to_sales,
        ev_to_fcf=ev_to_fcf,
        dividend_yield_info=dividend_yield_info,
        piotroski_f=fscore,
        good_debt_proxy=good_debt_proxy,
        mgmt_score=mgmt_score,
        undervalued_flag=undervalued_flag,
    )


# ---------------------------- Scoring helpers ---------------------------- #

def _winsorize(series: pd.Series, low_q=0.02, high_q=0.98) -> pd.Series:
    s = series.copy()
    lo = s.quantile(low_q)
    hi = s.quantile(high_q)
    return s.clip(lower=lo, upper=hi)


def _pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.copy()
    s = _winsorize(s)
    return s.rank(pct=True, ascending=not higher_is_better) * 100.0


# ---------------------------- Screening + Orchestration ---------------------------- #

def screen_universe(
        tickers: Sequence[str],
        session: requests.Session,
        fcf_margin_min: float = 0.10,
        de_max: float = 1.0,
        min_mgmt_score: int = 3,
        min_fscore: Optional[int] = None,
        max_workers: int = 6,
        retries: int = 2,
        retry_wait: float = 1.0,
        rate_limit: int = 5,
        min_interest_coverage: float = 1.0,
        max_netdebt_ebitda: float = float('inf'),
        exclude_sectors: Optional[List[str]] = None,
        exclude_industries: Optional[List[str]] = None,
        only_currencies: Optional[List[str]] = None,
        sort_by: Optional[List[str]] = None,
        value_w: float = 0.45,
        quality_w: float = 0.35,
        growth_w: float = 0.20,
        no_progress: bool = False,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """
    Return (results_df, failures). rate_limit is approx max ticks/sec (soft).
    Computes Value/Quality/Growth composite scores and a PASS flag.
    """
    rows: List[dict] = []
    failures: List[Tuple[str, str]] = []

    limiter = SoftRateLimiter(rate_per_sec=max(rate_limit, 1))

    def task(tk: str):
        limiter.wait()
        try:
            m = compute_metrics(tk, session=session, retries=retries, retry_wait=retry_wait)
            return ("ok", tk, m)
        except Exception as e:
            return ("err", tk, e)

    executor = cf.ThreadPoolExecutor(max_workers=max_workers)
    futures = [executor.submit(task, tk) for tk in tickers]
    iterator = cf.as_completed(futures)
    if TQDM and (not no_progress):
        iterator = tqdm(iterator, total=len(futures), desc="Screening", leave=False)

    for fut in iterator:
        status, tk, res = fut.result()
        if status == "err":
            msg = f"{type(res).__name__}: {str(res)}"
            failures.append((tk, msg[:500]))
            continue
        m: Metrics = res
        rows.append({
            'Ticker': m.ticker,
            'Name': m.long_name,
            'Price': m.price,
            'Currency': m.currency,
            'MarketCap': m.market_cap,
            'Sector': m.sector,
            'Industry': m.industry,
            'Country': m.country,
            'Beta': m.beta,
            'FCF_TTM': m.fcf_ttm,
            'Revenue_TTM': m.revenue_ttm,
            'FCF_Margin': m.fcf_margin,
            'Debt_Latest': m.debt_latest,
            'Equity_AvgQ': m.equity_avg_qtr,
            'D_E': m.de_ratio,
            'NI_TTM': m.net_income_ttm,
            'CFO_TTM': m.cfo_ttm,
            'ROE_TTM': m.roe_ttm,
            'OpIncome_TTM': m.op_income_ttm,
            'EBITDA_TTM': m.ebitda_ttm,
            'OpMargin_CurY': m.op_margin_cur,
            'OpMargin_PrevY': m.op_margin_prev,
            'GrossMargin_CurY': m.gross_margin_cur,
            'GrossMargin_PrevY': m.gross_margin_prev,
            'Trailing_PE': m.trailing_pe,
            'Price_to_Book': m.price_to_book,
            'EV_to_EBITDA': m.ev_to_ebitda,
            'FCF_Yield': m.fcf_yield,
            'Interest_Expense_TTM': m.interest_expense_ttm,
            'EBIT_TTM': m.ebit_ttm,
            'CashLike_Latest': m.cash_latest,
            'NetDebt_Latest': m.net_debt_latest,
            'Interest_Coverage': m.interest_coverage,
            'NetDebt_EBITDA': m.netdebt_ebitda,
            'Buybacks_TTM': m.buybacks_ttm,
            'Dividends_TTM': m.dividends_ttm,
            'Shareholder_Yield': m.shareholder_yield,
            'Rev_CAGR_3Y': m.rev_cagr_3y,
            'Rev_CAGR_5Y': m.rev_cagr_5y,
            'OCF_Growth_TTM': m.ocf_growth_ttm,
            'EV': m.ev,
            'EV_to_Sales': m.ev_to_sales,
            'EV_to_FCF': m.ev_to_fcf,
            'DividendYield_Info': m.dividend_yield_info,
            'Piotroski_F': m.piotroski_f,
            'GoodDebt_Proxy': m.good_debt_proxy,
            'MgmtScore': m.mgmt_score,
            'Undervalued': m.undervalued_flag,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df, failures

    # Optional exclusions/filters by metadata
    if exclude_sectors:
        bad = {s.lower() for s in exclude_sectors}
        df = df[~df['Sector'].fillna('').str.lower().isin(bad)]
    if exclude_industries:
        badi = {s.lower() for s in exclude_industries}
        df = df[~df['Industry'].fillna('').str.lower().isin(badi)]
    if only_currencies:
        df = df[df['Currency'].fillna('').isin(only_currencies)]

    # Pass flags
    df['Pass_FCFMargin'] = df['FCF_Margin'].apply(lambda x: (not pd.isna(x)) and x >= fcf_margin_min)
    df['Pass_DE'] = df['D_E'].apply(lambda x: (not pd.isna(x)) and x <= de_max)

    if min_interest_coverage and np.isfinite(min_interest_coverage) and min_interest_coverage > 0:
        df['Pass_IntCov'] = df['Interest_Coverage'].apply(lambda x: (not pd.isna(x)) and x >= min_interest_coverage)
    else:
        df['Pass_IntCov'] = True

    if max_netdebt_ebitda and np.isfinite(max_netdebt_ebitda):
        df['Pass_NetDebtEBITDA'] = df['NetDebt_EBITDA'].apply(lambda x: (not pd.isna(x)) and x <= max_netdebt_ebitda)
    else:
        df['Pass_NetDebtEBITDA'] = True

    df['Pass_Mgmt'] = df['MgmtScore'].apply(lambda x: (x is not None) and (x >= min_mgmt_score))

    if min_fscore is not None:
        df['Pass_FScore'] = df['Piotroski_F'].apply(lambda x: (x is not None) and (x >= min_fscore))
    else:
        df['Pass_FScore'] = True

    # ---------------- Composite scores ----------------
    # VALUE: FCF_Yield high, Shareholder_Yield high, EV/EBITDA low, P/B low, P/E low
    value_components = {
        'FCF_Yield': (True, df.get('FCF_Yield')),
        'Shareholder_Yield': (True, df.get('Shareholder_Yield')),
        'EV_to_EBITDA': (False, df.get('EV_to_EBITDA')),
        'Price_to_Book': (False, df.get('Price_to_Book')),
        'Trailing_PE': (False, df.get('Trailing_PE')),
    }
    for name, (hib, s) in value_components.items():
        if s is not None:
            df[f'R_{name}'] = _pct_rank(pd.to_numeric(s, errors='coerce'), higher_is_better=hib)
    df['Value_Score'] = df[
        [c for c in df.columns if c.startswith('R_') and any(k in c for k in value_components.keys())]].mean(axis=1)

    # QUALITY: ROE high, Interest Coverage high, NetDebt/EBITDA low, GoodDebt True, MgmtScore high, Piotroski high
    quality_series = [
        _pct_rank(pd.to_numeric(df.get('ROE_TTM'), errors='coerce'), True),
        _pct_rank(pd.to_numeric(df.get('Interest_Coverage'), errors='coerce'), True),
        _pct_rank(pd.to_numeric(df.get('NetDebt_EBITDA'), errors='coerce'), False),
        df.get('GoodDebt_Proxy').fillna(False).astype(int) * 100.0,        _pct_rank(pd.to_numeric(df.get('MgmtScore'), errors='coerce'), True),
        _pct_rank(pd.to_numeric(df.get('Piotroski_F'), errors='coerce'), True),
    ]
    df['Quality_Score'] = pd.concat(quality_series, axis=1).mean(axis=1)

    # GROWTH: Rev CAGR 3Y high, OCF Growth high, Gross margin trend up
    gm_delta = (
        pd.to_numeric(df.get('GrossMargin_CurY'), errors='coerce')
        - pd.to_numeric(df.get('GrossMargin_PrevY'), errors='coerce')
    )
    growth_series = [
        _pct_rank(pd.to_numeric(df.get('Rev_CAGR_3Y'), errors='coerce'), True),
        _pct_rank(pd.to_numeric(df.get('OCF_Growth_TTM'), errors='coerce'), True),
        _pct_rank(gm_delta, True),
    ]
    df['Growth_Score'] = pd.concat(growth_series, axis=1).mean(axis=1)

    # OVERALL (weighted)
    total_w = max(1e-9, value_w + quality_w + growth_w)
    df['Score_Overall'] = (
        value_w * df['Value_Score'].fillna(0)
        + quality_w * df['Quality_Score'].fillna(0)
        + growth_w * df['Growth_Score'].fillna(0)
    ) / total_w

    # Final PASS gate
    df['PASS'] = (
        df['Pass_FCFMargin'] & df['Pass_DE'] & df['Pass_Mgmt'] & df['Undervalued']
        & df['Pass_IntCov'] & df['Pass_NetDebtEBITDA'] & df['Pass_FScore']
    )

    # Default ordering — customizable
    if sort_by is None:
        sort_by = [
            c for c in [
                'PASS','Score_Overall','Value_Score','Quality_Score','Growth_Score',
                'FCF_Yield','Shareholder_Yield','ROE_TTM','Rev_CAGR_3Y','Piotroski_F'
            ] if c in df.columns
        ]
    ascending = [False] * len(sort_by)
    df = df.sort_values(by=sort_by, ascending=ascending)

    # Friendly column order (grouped)
    cols = [
        'PASS','Score_Overall','Value_Score','Quality_Score','Growth_Score',
        'Ticker','Name','Price','Currency','MarketCap','Sector','Industry','Country','Beta',
        # Valuation
        'FCF_Margin','FCF_Yield','Shareholder_Yield','Trailing_PE','Price_to_Book',
        'EV_to_EBITDA','EV_to_Sales','EV_to_FCF',
        # Quality
        'D_E','Interest_Coverage','NetDebt_EBITDA','ROE_TTM','MgmtScore','Piotroski_F','GoodDebt_Proxy',
        # Flows & base
        'CFO_TTM','NI_TTM','FCF_TTM','Revenue_TTM','OpIncome_TTM','EBITDA_TTM',
        # Growth
        'Rev_CAGR_3Y','Rev_CAGR_5Y','OCF_Growth_TTM','OpMargin_CurY','OpMargin_PrevY',
        'GrossMargin_CurY','GrossMargin_PrevY',
        # Capital returns
        'Buybacks_TTM','Dividends_TTM','DividendYield_Info',
        # Debt snapshot
        'Debt_Latest','Equity_AvgQ','NetDebt_Latest','CashLike_Latest',
    ]
    df = df.reindex(columns=[c for c in cols if c in df.columns])

    return df, failures


# ---------------------------- I/O helpers ---------------------------- #

def _add_timestamp(path: str, ts: str) -> str:
    """Append _{ts} before the file extension."""
    if not path:
        return path
    head_tail = path.rsplit('/', 1)
    folder = head_tail[0] + '/' if len(head_tail) == 2 else ''
    fname = head_tail[-1]
    if '.' in fname:
        base, ext = fname.rsplit('.', 1)
        return f"{folder}{base}_{ts}.{ext}"
    else:
        return f"{folder}{fname}_{ts}"


def _write_outputs(
    df: pd.DataFrame,
    failures: List[Tuple[str, str]],
    out_csv: Optional[str],
    out_json: Optional[str],
    out_parquet: Optional[str],
    out_sqlite: Optional[str],
    out_xlsx: Optional[str],
    out_fail: Optional[str],
) -> None:
    if df is None or df.empty:
        LOGGER.warning('No rows returned. Writing empty CSV for provenance if requested...')
        if out_csv:
            try:
                pd.DataFrame().to_csv(out_csv, index=False)
                LOGGER.info("[OK] Empty Results CSV -> %s", out_csv)
            except Exception as e:
                LOGGER.error("Failed to write CSV: %s", e)
        if failures and out_fail:
            try:
                fdf = pd.DataFrame(failures, columns=['Ticker', 'Error'])
                fdf.to_csv(out_fail, index=False)
                LOGGER.info("[OK] Failures CSV -> %s (%d entries)", out_fail, len(failures))
            except Exception as e:
                LOGGER.warning("Could not write failures CSV: %s", e)
        return

    # CSV
    if out_csv:
        try:
            df.to_csv(out_csv, index=False)
            LOGGER.info("[OK] Results CSV -> %s (%d rows)", out_csv, len(df))
        except Exception as e:
            LOGGER.error("Failed to write CSV: %s", e)

    # JSON
    if out_json:
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2, default=str)
            LOGGER.info("[OK] JSON -> %s", out_json)
        except Exception as e:
            LOGGER.warning("Could not write JSON: %s", e)

    # Parquet
    if out_parquet:
        try:
            df.to_parquet(out_parquet, index=False)
            LOGGER.info("[OK] Parquet -> %s", out_parquet)
        except Exception as e:
            LOGGER.warning("Could not write Parquet (need pyarrow or fastparquet): %s", e)

    # SQLite
    if out_sqlite:
        try:
            con = sqlite3.connect(out_sqlite)
            df.to_sql('results', con, if_exists='replace', index=False)
            if failures:
                fdf = pd.DataFrame(failures, columns=['Ticker', 'Error'])
                fdf.to_sql('failures', con, if_exists='replace', index=False)
            con.close()
            LOGGER.info("[OK] SQLite -> %s (tables: results%s)", out_sqlite, ", failures" if failures else "")
        except Exception as e:
            LOGGER.warning("Could not write SQLite: %s", e)

    # XLSX
    if out_xlsx:
        try:
            with pd.ExcelWriter(out_xlsx) as writer:
                df.to_excel(writer, sheet_name='results', index=False)
                if failures:
                    fdf = pd.DataFrame(failures, columns=['Ticker', 'Error'])
                    fdf.to_excel(writer, sheet_name='failures', index=False)
            LOGGER.info("[OK] Excel -> %s", out_xlsx)
        except Exception as e:
            LOGGER.warning("Could not write Excel: %s", e)

    # Failures CSV
    if failures and out_fail:
        try:
            fdf = pd.DataFrame(failures, columns=['Ticker', 'Error'])
            fdf.to_csv(out_fail, index=False)
            LOGGER.info("[OK] Failures CSV -> %s (%d entries)", out_fail, len(failures))
        except Exception as e:
            LOGGER.warning("Could not write failures CSV: %s", e)


# ---------------------------- CLI ---------------------------- #

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(
        "Screen SP500/NDX/DAX/BMV IPC for quality FCF, manageable leverage (D/E<=de_max), "
        "mgmt quality, value, and growth; robust HTTP."
    ))
    p.add_argument('--universes', type=str, default='sp500,nasdaq100',
                   help='Comma-separated from {sp500,nasdaq100,dax40,bmv_ipc|ipc|mx}.')
    # in parse_args(...)
    p.add_argument('--yf-no-session', action='store_true',
                   help='Do not pass the custom requests session into yfinance (recommended).')
    p.add_argument('--roe-proxy', type=str, default='none',
                   choices=['none', 'roa', 'roic'],
                   help='Wie ROE ersetzt wird, wenn avg. Equity ≤ 0: none (NaN), roa (NI/avg Assets), roic (NOPAT/InvestedCapital).')

    p.add_argument('--fcf-margin-min', type=float, default=0.20,
                   help='Minimum FCF margin (decimal).')
    p.add_argument('--de-max', type=float, default=1.0,
                   help='Maximum Debt/Equity allowed.')
    p.add_argument('--min-mgmt-score', type=int, default=3,
                   help='Min management score (0-4).')
    p.add_argument('--min-fscore', type=int, default=None,
                   help='Optional: require minimum Piotroski F-Score (0-9).')

    p.add_argument('--min-interest-coverage', type=float, default=0.0,
                   help='Optional: require EBIT/|Interest Expense| >= this.')
    p.add_argument('--max-netdebt-ebitda', type=float, default=float('inf'),
                   help='Optional: require Net Debt / EBITDA <= this.')

    p.add_argument('--exclude-sectors', type=str, default='',
                   help='Comma-separated sector names to exclude (case-insensitive).')
    p.add_argument('--exclude-industries', type=str, default='',
                   help='Comma-separated industry names to exclude.')
    p.add_argument('--only-currencies', type=str, default='',
                   help='Comma-separated list of currencies to keep (e.g., USD,EUR,MXN).')

    p.add_argument('--max-workers', type=int, default=6, help='Max concurrent ticker fetches.')
    p.add_argument('--rate-limit', type=int, default=5, help='Approx max tickers per second (soft).')
    p.add_argument('--max-retries', type=int, default=3, help='HTTP retries for session.')
    p.add_argument('--retry-wait', type=float, default=1.2, help='Base wait/backoff seconds for attribute retries.')
    p.add_argument('--https-proxy', type=str, default=None, help='HTTPS proxy URL (optional).')

    p.add_argument('--cache', type=str, default=None, help='Optional requests-cache name (e.g., http_cache)')
    p.add_argument('--cache-expire', type=int, default=3600, help='Cache TTL seconds.')

    p.add_argument('--out', type=str, default='results.csv', help='Output CSV path')
    p.add_argument('--json', type=str, default=None, help='Optional JSON output path')
    p.add_argument('--parquet', type=str, default=None, help='Optional Parquet output path')
    p.add_argument('--sqlite', type=str, default=None, help='Optional SQLite .db output path')
    p.add_argument('--xlsx', type=str, default=None, help='Optional Excel .xlsx output path')
    p.add_argument('--failures', type=str, default='failures.csv',
                   help='Path to write fetch failures (CSV with columns Ticker,Error).')
    p.add_argument('--sort-by', type=str, default='',
                   help='Comma-separated columns to sort by (desc). Defaults to a sensible preset.')
    p.add_argument('--no-timestamp', action='store_true',
                   help='Do not append a timestamp to output filenames.')
    p.add_argument('--only', type=str, default='',
                   help='Optional: comma-separated tickers to restrict to after building the universe.')
    p.add_argument('--top-n', type=int, default=0,
                   help='Optional: keep only the top N rows after sorting.')

    p.add_argument('--select-cols', type=str, default='',
                   help='Optional: comma-separated list of columns to keep (others dropped).')
    p.add_argument('--drop-cols', type=str, default='',
                   help='Optional: comma-separated list of columns to drop.')

    p.add_argument('--value-w', type=float, default=0.45, help='Weight for Value_Score.')
    p.add_argument('--quality-w', type=float, default=0.35, help='Weight for Quality_Score.')
    p.add_argument('--growth-w', type=float, default=0.20, help='Weight for Growth_Score.')

    p.add_argument('--verbose', action='store_true', help='Verbose logging.')
    p.add_argument('--quiet', action='store_true', help='Quiet logging (warnings+).')
    p.add_argument('--no-progress', action='store_true', help='Disable tqdm progress bar.')


    return p.parse_args(argv)


# ---------------------------- Runner / I/O ---------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Hardened HTTP session
    session = make_retry_session(
        max_retries=args.max_retries,
        backoff_factor=args.retry_wait,
        https_proxy=args.https_proxy,
        cache_name=args.cache,
        cache_expire=args.cache_expire,
    )

    # --- quick environment smoke test: is yfinance basically working? ---
    try:
        t_test = yf.Ticker("AAPL", session=session)
        # _ = t_test.history(period="1d")  # <-- weg damit
        _ = t_test.fast_info  # leichtgewichtiger Check
        _ = t_test.quarterly_financials  # zweiter leichter Check
    except Exception as e:
        LOGGER.error("[FATAL] yfinance base access failed: %s", e)
        return 3

    # Universe
    universes = [u.strip() for u in (args.universes or '').split(',') if u.strip()]
    if not universes:
        LOGGER.error('No universes provided.')
        return 2

    try:
        tickers = build_universe(universes, session=session)
    except Exception as e:
        LOGGER.error("Failed to build universe: %s", e)
        return 2

    if not tickers:
        LOGGER.error('Universe is empty after loading.')
        return 2

    # Optional hard restriction after building the universe
    if args.only:
        only_set = {t.strip() for t in args.only.split(',') if t.strip()}
        tickers = [t for t in tickers if t in only_set]
        if not tickers:
            LOGGER.warning('After applying --only, no tickers remain.')

    # Parse filters
    exclude_sectors = [s.strip() for s in args.exclude_sectors.split(',') if s.strip()] if args.exclude_sectors else None
    exclude_industries = [s.strip() for s in args.exclude_industries.split(',') if s.strip()] if args.exclude_industries else None
    only_currencies = [s.strip() for s in args.only_currencies.split(',') if s.strip()] if args.only_currencies else None

    sort_by: Optional[List[str]] = None
    if args.sort_by:
        sort_by = [s.strip() for s in args.sort_by.split(',') if s.strip()]

    LOGGER.info("Screening %d tickers from: %s", len(tickers), ", ".join(universes))

    df, failures = screen_universe(
        tickers=tickers,
        session=session,
        fcf_margin_min=args.fcf_margin_min,
        de_max=args.de_max,
        min_mgmt_score=args.min_mgmt_score,
        min_fscore=args.min_fscore,
        max_workers=args.max_workers,
        retries=args.max_retries,
        retry_wait=args.retry_wait,
        rate_limit=args.rate_limit,
        min_interest_coverage=args.min_interest_coverage,
        max_netdebt_ebitda=args.max_netdebt_ebitda,
        exclude_sectors=exclude_sectors,
        exclude_industries=exclude_industries,
        only_currencies=only_currencies,
        sort_by=sort_by,
        value_w=args.value_w,
        quality_w=args.quality_w,
        growth_w=args.growth_w,
        no_progress=args.no_progress,
    )

    # Limit to top N if requested
    if args.top_n and args.top_n > 0 and df is not None and not df.empty:
        df = df.head(args.top_n)

    # Column selection
    if df is not None and not df.empty:
        if args.select_cols:
            keep = [c.strip() for c in args.select_cols.split(',') if c.strip()]
            keep = [c for c in keep if c in df.columns]
            if keep:
                df = df[keep]
        elif args.drop_cols:
            dropc = [c.strip() for c in args.drop_cols.split(',') if c.strip()]
            dropc = [c for c in dropc if c in df.columns]
            if dropc:
                df = df.drop(columns=dropc, errors='ignore')

    ts = time.strftime('%Y%m%d_%H%M%S')

    out_csv = args.out
    out_json = args.json
    out_parquet = args.parquet
    out_sqlite = args.sqlite
    out_xlsx = args.xlsx
    out_fail = args.failures
    if not args.no_timestamp:
        if out_csv:
            out_csv = _add_timestamp(out_csv, ts)
        if out_json:
            out_json = _add_timestamp(out_json, ts)
        if out_parquet:
            out_parquet = _add_timestamp(out_parquet, ts)
        if out_sqlite:
            out_sqlite = _add_timestamp(out_sqlite, ts)
        if out_xlsx:
            out_xlsx = _add_timestamp(out_xlsx, ts)
        if out_fail:
            out_fail = _add_timestamp(out_fail, ts)

    # Write outputs
    _write_outputs(df, failures, out_csv, out_json, out_parquet, out_sqlite, out_xlsx, out_fail)

    # Console summary
    passed = int(df['PASS'].sum()) if (df is not None and not df.empty and 'PASS' in df.columns) else 0
    LOGGER.info("[SUMMARY] Universe: %d | Results: %d | PASS: %d | Failures: %d",
                len(tickers), (0 if df is None else len(df)), passed, len(failures))

    # Quick preview
    try:
        if df is not None and not df.empty:
            preview_cols = [c for c in
                            ['PASS','Score_Overall','Ticker','Name','Price','Currency','FCF_Yield',
                             'Shareholder_Yield','ROE_TTM','Piotroski_F','MgmtScore'] if c in df.columns]
            top = df.head(min(20, len(df)))[preview_cols]
            with pd.option_context('display.max_rows', 30, 'display.width', 200):
                print('\nTop results (first rows):')
                print(top.to_string(index=False))
    except Exception:
        pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

