from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yahooquery import Ticker as YQ  # optional
except Exception:  # pragma: no cover
    YQ = None  # type: ignore

# ---------- Data model ----------

@dataclass
class Enrichment:
    # identity / classification
    name: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    category: Optional[str] = None
    asset_class: Optional[str] = None
    # fund stats
    expense_ratio_pct: Optional[float] = None
    aum_usd: Optional[float] = None
    inception_date: Optional[str] = None
    # nav
    nav_price: Optional[float] = None
    nav_premium_discount_pct: Optional[float] = None
    # price stats
    price_52w_high: Optional[float] = None
    price_52w_low: Optional[float] = None
    dd_from_52w_high_pct: Optional[float] = None
    # risk
    vol_30d_pct: Optional[float] = None
    vol_90d_pct: Optional[float] = None
    beta_3y: Optional[float] = None
    max_drawdown_1y_pct: Optional[float] = None
    sortino_1y: Optional[float] = None
    downside_dev_1y_pct: Optional[float] = None
    sharpe_1y: Optional[float] = None
    calmar_1y: Optional[float] = None
    drawdown_duration_1y_days: Optional[int] = None
    var_95_1d_pct: Optional[float] = None
    cvar_95_1d_pct: Optional[float] = None
    corr_1y: Optional[float] = None
    # dividends
    trailing_12m_div_ps: Optional[float] = None
    indicated_div_ps: Optional[float] = None
    trailing_12m_yield_pct: Optional[float] = None
    forward_yield_pct: Optional[float] = None
    forward_yield_method: Optional[str] = None  # indicated|t12m|t12m_on_shares|fallback_current
    div_growth_1y_pct: Optional[float] = None
    div_growth_3y_cagr_pct: Optional[float] = None
    div_growth_5y_cagr_pct: Optional[float] = None
    div_cv_12p: Optional[float] = None
    div_cut_3m_vs_12m_pct: Optional[float] = None
    derived_fields: Optional[List[str]] = None
    distribution_frequency: Optional[str] = None
    next_ex_date_est: Optional[str] = None
    # performance (TWR) total return using Adjusted Close
    twr_1m_pct: Optional[float] = None
    twr_3m_pct: Optional[float] = None
    twr_6m_pct: Optional[float] = None
    twr_12m_pct: Optional[float] = None
    # valuation (equities, best-effort from yq)
    pe_ttm: Optional[float] = None
    pb: Optional[float] = None
    payout_ratio: Optional[float] = None
    # ETF exposure (best-effort; often missing on covered-call ETFs)
    top_holdings: Optional[List[Dict[str, Any]]] = None
    sector_weights: Optional[Dict[str, float]] = None

    def asdict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ---------- Helpers ----------

@lru_cache(maxsize=4096)
def _yf_adj_series(symbol: str, period: str = "3y") -> pd.Series:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty or "Adj Close" not in df:
        return pd.Series(dtype=float)
    col = df["Adj Close"]
    if isinstance(col, pd.DataFrame):  # yfinance sometimes returns MultiIndex columns
        col = col.iloc[:, 0]
    s = pd.to_numeric(col, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).date
    return s

@lru_cache(maxsize=4096)
def _yf_close_series(symbol: str, period: str = "1y") -> pd.Series:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty or "Close" not in df:
        return pd.Series(dtype=float)
    col = df["Close"]
    if isinstance(col, pd.DataFrame):  # yfinance sometimes returns MultiIndex columns
        col = col.iloc[:, 0]
    s = pd.to_numeric(col, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).date
    return s

def _window_return_pct(series: pd.Series, days: int) -> Optional[float]:
    if series is None or len(series) < 5:
        return None
    tail = series.tail(days + 1)
    if len(tail) < 2:
        return None
    start = float(tail.iloc[0])
    end = float(tail.iloc[-1])
    if start <= 0:
        return None
    return round((end / start - 1.0) * 100.0, 3)

def _annualized_vol(series: pd.Series, window: int) -> Optional[float]:
    if series is None or len(series) < window + 1:
        return None
    r = np.log(series).diff().dropna()
    r_win = r.tail(window)
    if len(r_win) < window:
        return None
    sigma_d = float(r_win.std())
    return round(sigma_d * (252 ** 0.5) * 100.0, 3)

def _max_drawdown_pct(series: pd.Series, days: int = 252) -> Optional[float]:
    s = series.tail(days)
    if len(s) < 2:
        return None
    running_max = s.cummax()
    dd = (s / running_max - 1.0).min()
    return round(float(dd) * 100.0, 3)

def _sortino_1y(series: pd.Series) -> Optional[float]:
    s = series.tail(252)
    if len(s) < 30:
        return None
    r = np.log(s).diff().dropna()
    downside = r.copy()
    downside[downside > 0] = 0
    dd = float(np.sqrt((downside ** 2).mean()))
    if dd == 0:
        return None
    mu = float(r.mean())
    return round((mu / dd) * (252 ** 0.5), 3)


def _downside_dev_1y_pct(series: pd.Series) -> Optional[float]:
    s = series.tail(252)
    if len(s) < 30:
        return None
    r = np.log(s).diff().dropna()
    dn = r[r < 0]
    if len(dn) < 10:
        return None
    dd = float(dn.std()) * (252 ** 0.5) * 100.0
    return round(dd, 3)


def _sharpe_1y(series: pd.Series, rf_annual: float = 0.0) -> Optional[float]:
    # rf_annual is a decimal (0.05 = 5%). Default 0 to avoid external rate dependency.
    s = series.tail(252)
    if len(s) < 30:
        return None
    r = np.log(s).diff().dropna()
    if len(r) < 10:
        return None
    mu = float(r.mean()) * 252.0
    sigma = float(r.std()) * (252 ** 0.5)
    if sigma <= 0:
        return None
    return round((mu - rf_annual) / sigma, 3)


def _cagr_1y(series: pd.Series) -> Optional[float]:
    s = series.tail(252)
    if len(s) < 2:
        return None
    start = float(s.iloc[0])
    end = float(s.iloc[-1])
    if start <= 0:
        return None
    # annualize based on actual number of observations
    n = max(1, len(s) - 1)
    return (end / start) ** (252.0 / n) - 1.0


def _calmar_1y(series: pd.Series) -> Optional[float]:
    cagr = _cagr_1y(series)
    dd = _max_drawdown_pct(series, days=252)
    if cagr is None or dd is None:
        return None
    if dd == 0:
        return None
    return round((cagr * 100.0) / abs(float(dd)), 3)


def _drawdown_duration_1y_days(series: pd.Series) -> Optional[int]:
    s = series.tail(252)
    if len(s) < 2:
        return None
    running_max = s.cummax()
    underwater = (s < running_max)
    max_streak = 0
    streak = 0
    for v in underwater.values:
        if bool(v):
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return int(max_streak)


def _var_cvar_1d_pct(series: pd.Series, alpha: float = 0.05) -> tuple[Optional[float], Optional[float]]:
    s = series.tail(252)
    if len(s) < 30:
        return (None, None)
    r = s.pct_change().dropna()
    if len(r) < 10:
        return (None, None)
    q = float(np.quantile(r.values, alpha))
    tail = r[r <= q]
    cvar = float(tail.mean()) if len(tail) else None
    return (round(q * 100.0, 3), round(cvar * 100.0, 3) if cvar is not None else None)


def _dividend_payments(symbol: str, years: int = 6) -> pd.Series:
    """Raw dividend payments (not annualized), last `years` years."""
    try:
        t = yf.Ticker(symbol)
        div = t.dividends
        if div is None or len(div) == 0:
            return pd.Series(dtype=float)
        cutoff = pd.Timestamp(date.today() - timedelta(days=365 * years))
        div = div[div.index >= cutoff]
        return div.astype(float)
    except Exception:
        return pd.Series(dtype=float)


def _dividend_year_totals(div: pd.Series) -> pd.Series:
    if div is None or len(div) == 0:
        return pd.Series(dtype=float)
    # sum by calendar year
    df = div.to_frame("d")
    df["y"] = df.index.year
    return df.groupby("y")["d"].sum().sort_index()


def _div_growth_metrics(symbol: str) -> dict:
    div = _dividend_payments(symbol)
    yt = _dividend_year_totals(div)
    out: dict = {"div_growth_1y_pct": None, "div_growth_3y_cagr_pct": None, "div_growth_5y_cagr_pct": None, "div_cv_12p": None, "div_cut_3m_vs_12m_pct": None}
    if len(div) >= 4:
        last12 = div.tail(12)
        if len(last12) >= 4:
            m = float(last12.mean())
            if m != 0:
                out["div_cv_12p"] = round(float(last12.std()) / abs(m), 3)
        # cut detector using last 3 vs last 12 payments
        last3 = div.tail(3)
        if len(last3) == 3 and len(last12) >= 6:
            a3 = float(last3.mean())
            a12 = float(last12.mean())
            if a12 != 0:
                out["div_cut_3m_vs_12m_pct"] = round(((a3 / a12) - 1.0) * 100.0, 3)

    if len(yt) >= 2:
        y_last = float(yt.iloc[-1])
        y_prev = float(yt.iloc[-2])
        if y_prev != 0:
            out["div_growth_1y_pct"] = round(((y_last / y_prev) - 1.0) * 100.0, 3)

    def _cagr(n: int) -> Optional[float]:
        if len(yt) < n + 1:
            return None
        start = float(yt.iloc[-(n + 1)])
        end = float(yt.iloc[-1])
        if start <= 0:
            return None
        return ((end / start) ** (1.0 / n) - 1.0) * 100.0

    c3 = _cagr(3)
    if c3 is not None:
        out["div_growth_3y_cagr_pct"] = round(c3, 3)
    c5 = _cagr(5)
    if c5 is not None:
        out["div_growth_5y_cagr_pct"] = round(c5, 3)

    return out


def _beta_3y(symbol: str, benchmark: str = "^GSPC") -> Optional[float]:
    a = _yf_close_series(symbol, period="3y")
    b = _yf_close_series(benchmark, period="3y")
    if len(a) < 60 or len(b) < 60:
        return None
    df = pd.DataFrame({"a": np.log(a).diff(), "b": np.log(b).diff()}).dropna()
    if len(df) < 60:
        return None
    cov = float(np.cov(df["a"], df["b"])[0, 1])
    var = float(np.var(df["b"]))
    if var == 0:
        return None
    return round(cov / var, 3)


def _corr_1y(symbol: str, benchmark: str = "^GSPC") -> Optional[float]:
    a = _yf_close_series(symbol, period="1y")
    b = _yf_close_series(benchmark, period="1y")
    if len(a) < 60 or len(b) < 60:
        return None
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 60:
        return None
    ra = df["a"].pct_change().dropna()
    rb = df["b"].pct_change().dropna()
    if len(ra) < 30 or len(rb) < 30:
        return None
    r = pd.concat([ra, rb], axis=1).dropna()
    if len(r) < 30:
        return None
    return round(float(r.iloc[:, 0].corr(r.iloc[:, 1])), 3)

def _t12m_div_ps(symbol: str, as_of: date) -> Optional[float]:
    try:
        s = yf.Ticker(symbol).dividends
    except Exception:
        s = None
    if s is None or len(s) == 0:
        return None
    total = 0.0
    for idx, amt in s.items():
        d = getattr(idx, "date", lambda: idx)()
        try:
            dd = date.fromisoformat(str(d))
        except Exception:
            continue
        if as_of - timedelta(days=365) <= dd <= as_of:
            try:
                total += float(amt)
            except Exception:
                pass
    return round(total, 6)

def _infer_distribution_frequency(symbol: str) -> Optional[str]:
    try:
        s = yf.Ticker(symbol).dividends
    except Exception:
        s = None
    if s is None or len(s) < 3:
        return None
    # last ~6 ex-dates
    ex_dates: List[date] = []
    for idx, amt in list(s.items())[-6:]:
        d = getattr(idx, "date", lambda: idx)()
        try:
            ex_dates.append(date.fromisoformat(str(d)))
        except Exception:
            pass
    if len(ex_dates) < 3:
        return None
    ex_dates = sorted(ex_dates)
    gaps = [(ex_dates[i] - ex_dates[i-1]).days for i in range(1, len(ex_dates))]
    if not gaps:
        return None
    med = sorted(gaps)[len(gaps)//2]
    if med < 45:
        return "monthly"
    if med < 120:
        return "quarterly"
    return "irregular"

def _next_ex_estimate(last_ex_date: Optional[str], freq: Optional[str]) -> Optional[str]:
    if not last_ex_date or not freq:
        return None
    try:
        led = date.fromisoformat(last_ex_date)
    except Exception:
        return None
    delta = 30 if freq == "monthly" else 90 if freq == "quarterly" else None
    if not delta:
        return None
    return (led + timedelta(days=delta)).isoformat()

def _yq_blocks(symbol: str) -> Dict[str, Any]:
    if YQ is None:
        return {}
    try:
        t = YQ(symbol)
    except Exception:
        return {}
    def g(obj, key):
        try:
            return (getattr(t, obj) or {}).get(key, {})
        except Exception:
            return {}
    return {
        "price": g("price", symbol),
        "asset_profile": g("asset_profile", symbol),
        "fund_profile": g("fund_profile", symbol),
        "key_stats": g("key_stats", symbol),
        "summary_detail": g("summary_detail", symbol),
        "fund_holding_info": g("fund_holding_info", symbol),
    }

# ---------- Public API ----------

def enrich_holding(symbol: str, last_price: float, last_ex_date: Optional[str], as_of: date) -> Dict[str, Any]:
    e = Enrichment()

    # Profiles / fund stats / nav (best-effort via yahooquery)
    yq = _yq_blocks(symbol) if YQ else {}
    price = yq.get("price", {})
    ap = yq.get("asset_profile", {})
    fp = yq.get("fund_profile", {})
    ks = yq.get("key_stats", {})
    th = yq.get("fund_holding_info", {})

    e.name = price.get("longName") or price.get("shortName")
    e.exchange = price.get("exchange")
    e.currency = price.get("currency")
    e.sector = ap.get("sector")
    e.industry = ap.get("industry")
    e.category = fp.get("category") or fp.get("categoryName")
    e.asset_class = fp.get("category") or ap.get("sector")

    # Expense ratio / AUM / inception
    er = fp.get("annualReportExpenseRatio") or fp.get("feesExpensesInvestment")
    if isinstance(er, (float, int)):
        e.expense_ratio_pct = round(float(er) * 100.0, 3) if float(er) < 1 else round(float(er), 3)
    aum = fp.get("totalAssets") or price.get("marketCap")
    if isinstance(aum, (float, int)):
        e.aum_usd = float(aum)
    inc = fp.get("fundInceptionDate")
    if isinstance(inc, str) and len(inc) >= 10:
        e.inception_date = inc[:10]

    # NAV premium/discount
    nav = ks.get("navPrice")
    if isinstance(nav, (float, int)):
        e.nav_price = float(nav)
        if last_price and nav:
            try:
                e.nav_premium_discount_pct = round(((last_price - float(nav)) / float(nav)) * 100.0, 3)
            except Exception:
                pass

    # Valuation
    if isinstance(ks.get("trailingPE"), (float, int)): e.pe_ttm = float(ks["trailingPE"])
    if isinstance(ks.get("priceToBook"), (float, int)): e.pb = float(ks["priceToBook"])
    if isinstance(ks.get("payoutRatio"), (float, int)): e.payout_ratio = float(ks["payoutRatio"])

    # Dividends / yields (best-effort; prefer vendor fields but derive from yfinance when needed)
    sd = yq.get("summary_detail", {}) if isinstance(yq, dict) else {}
    e.trailing_12m_div_ps = _t12m_div_ps(symbol, as_of)

    # indicated_div_ps: prefer summaryDetail.dividendRate; fall back to price.dividendRate
    div_rate = None
    for src in (sd.get("dividendRate"), price.get("dividendRate")):
        if isinstance(src, (float, int)) and float(src) > 0:
            div_rate = float(src)
            break
    if div_rate is not None:
        e.indicated_div_ps = div_rate

    # trailing yield derived from trailing_12m_div_ps where possible
    if last_price and last_price > 0 and e.trailing_12m_div_ps is not None:
        e.trailing_12m_yield_pct = round((float(e.trailing_12m_div_ps) / float(last_price)) * 100.0, 3)

    # forward yield (best-effort): indicated -> trailing -> None
    if last_price and last_price > 0 and e.indicated_div_ps is not None:
        e.forward_yield_pct = round((float(e.indicated_div_ps) / float(last_price)) * 100.0, 3)
        e.forward_yield_method = "indicated"
    elif e.trailing_12m_yield_pct is not None:
        e.forward_yield_pct = float(e.trailing_12m_yield_pct)
        e.forward_yield_method = "t12m"
        e.derived_fields = (e.derived_fields or []) + ["forward_yield_pct"]

    # Dividend-derived growth/stability metrics (computed from yfinance dividends)
    try:
        gm = _div_growth_metrics(symbol)
        e.div_growth_1y_pct = gm.get("div_growth_1y_pct")
        e.div_growth_3y_cagr_pct = gm.get("div_growth_3y_cagr_pct")
        e.div_growth_5y_cagr_pct = gm.get("div_growth_5y_cagr_pct")
        e.div_cv_12p = gm.get("div_cv_12p")
        e.div_cut_3m_vs_12m_pct = gm.get("div_cut_3m_vs_12m_pct")
    except Exception:
        pass

    # Risk
    close_1y = _yf_close_series(symbol, period="1y")
    adj_3y = _yf_adj_series(symbol, period="3y")
    e.vol_30d_pct = _annualized_vol(close_1y, 30)
    e.vol_90d_pct = _annualized_vol(close_1y, 90)
    e.max_drawdown_1y_pct = _max_drawdown_pct(adj_3y if len(adj_3y)>0 else close_1y, 252)
    e.sortino_1y = _sortino_1y(adj_3y if len(adj_3y)>0 else close_1y)
    e.beta_3y = _beta_3y(symbol, "^GSPC")
    # Additional risk metrics (derived)
    base_series = adj_3y if len(adj_3y) > 0 else close_1y
    e.downside_dev_1y_pct = _downside_dev_1y_pct(base_series)
    e.sharpe_1y = _sharpe_1y(base_series, rf_annual=0.0)
    e.calmar_1y = _calmar_1y(base_series)
    e.drawdown_duration_1y_days = _drawdown_duration_1y_days(base_series)
    v, cv = _var_cvar_1d_pct(base_series, alpha=0.05)
    e.var_95_1d_pct = v
    e.cvar_95_1d_pct = cv
    e.corr_1y = _corr_1y(symbol, "^GSPC")

    # Performance (TWR using Adjusted Close total return)
    adj_1y = _yf_adj_series(symbol, period="1y")
    e.twr_1m_pct = _window_return_pct(adj_1y, 21)   # ~21 trading days
    e.twr_3m_pct = _window_return_pct(adj_1y, 63)
    e.twr_6m_pct = _window_return_pct(_yf_adj_series(symbol, period="2y"), 126)
    e.twr_12m_pct = _window_return_pct(_yf_adj_series(symbol, period="2y"), 252)

    # Distribution cadence & next ex-date estimate
    freq = _infer_distribution_frequency(symbol)
    e.distribution_frequency = freq
    e.next_ex_date_est = _next_ex_estimate(last_ex_date, freq)

    # ETF holdings / sector weights (best-effort)
    if isinstance(th, dict):
        holdings = None
        for k in ("holdings", "equityHoldings", "topHoldings"):
            if isinstance(th.get(k), list) and th.get(k):
                holdings = th.get(k)
                break
        if holdings:
            out = []
            for h in holdings[:10]:
                sym = h.get("symbol") or h.get("holdingSymbol")
                nm = h.get("holdingName") or h.get("name")
                w = h.get("holdingPercent")
                out.append({
                    "symbol": sym, "name": nm,
                    "weight_pct": round(float(w) * 100.0, 3) if isinstance(w, (float, int)) else None
                })
            e.top_holdings = out
        sw = th.get("sectorWeightings")
        if isinstance(sw, list) and sw:
            try:
                e.sector_weights = { list(d.keys())[0]: round(list(d.values())[0] * 100.0, 3) for d in sw if isinstance(d, dict) }
            except Exception:
                pass

    return e.asdict()
