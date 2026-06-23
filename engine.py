"""
AXIX Ω∞ — Topophysical Engine v2
All heavy computation lives here, separate from the UI.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import periodogram
import warnings
warnings.filterwarnings("ignore")


# ─── CORE METRICS ────────────────────────────────────────────────────────────

def hurst_exponent(ts: np.ndarray, min_lag: int = 2, max_lag: int = 50) -> float:
    lags = range(min_lag, min(max_lag, len(ts) // 2))
    tau = [np.std(np.subtract(ts[l:], ts[:-l])) for l in lags]
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 3:
        return 0.5
    lags_v, tau_v = zip(*valid)
    poly = np.polyfit(np.log(lags_v), np.log(tau_v), 1)
    return float(np.clip(poly[0], 0.01, 0.99))


def riemann_density(prices: np.ndarray, bandwidth: float = 0.5) -> np.ndarray:
    log_p = np.log(prices)
    sigma = bandwidth * np.std(log_p)
    if sigma == 0:
        return np.ones(len(prices))
    density = np.array([
        np.exp(-0.5 * ((log_p - log_p[i]) / sigma) ** 2).sum()
        for i in range(len(log_p))
    ])
    density /= density.max()
    return gaussian_filter1d(density, sigma=3)


def compute_axix_score(vol: float, hurst: float, momentum: float,
                       coherence: float, energy: float) -> dict:
    """
    AXIX Signal Score 0–100.
    Combines 5 topophysical dimensions into a single actionable number.
    """
    # Vol score: lower vol = higher score (stability)
    vol_score = float(np.clip(100 - vol * 200, 0, 100))
    # Hurst score: closer to 0.6-0.7 = ideal trending = high score
    hurst_score = float(np.clip((1 - abs(hurst - 0.65) / 0.35) * 100, 0, 100))
    # Momentum score: positive momentum = higher score
    mom_score = float(np.clip(50 + momentum * 500, 0, 100))
    # Coherence score: direct
    coh_score = float(np.clip(coherence, 0, 100))
    # Energy score: moderate energy = ideal (not too high, not too low)
    energy_score = float(np.clip(100 - abs(energy - 15) * 2, 0, 100))

    weights = [0.25, 0.25, 0.20, 0.15, 0.15]
    total = (vol_score * weights[0] + hurst_score * weights[1] +
             mom_score * weights[2] + coh_score * weights[3] +
             energy_score * weights[4])

    if total >= 75:
        signal = "EXPANSIÓN"
        color = "#00ff88"
    elif total >= 55:
        signal = "COHERENTE"
        color = "#58a6ff"
    elif total >= 35:
        signal = "EQUILIBRIO"
        color = "#f0883e"
    else:
        signal = "CONTRACCIÓN"
        color = "#f85149"

    return {
        "score": round(total, 1),
        "signal": signal,
        "color": color,
        "components": {
            "Estabilidad":  round(vol_score, 1),
            "Memoria":      round(hurst_score, 1),
            "Momentum":     round(mom_score, 1),
            "Coherencia":   round(coh_score, 1),
            "Energía":      round(energy_score, 1),
        }
    }


def compute_full_metrics(df: pd.DataFrame) -> dict:
    prices = df["Close"].values.astype(float)
    if len(prices) < 10:
        raise ValueError("Insufficient data")

    log_ret = np.diff(np.log(prices))
    vol     = float(np.std(log_ret) * np.sqrt(252))
    mu      = float(np.mean(log_ret))
    energy  = float(np.mean(np.abs(log_ret)) * np.sqrt(252) * 100)
    hurst   = hurst_exponent(np.log(prices))

    sq_ret      = log_ret ** 2
    coherence   = float((1 - min(abs(np.corrcoef(sq_ret[:-1], sq_ret[1:])[0,1]), 0.99)) * 100)
    momentum_21 = float(np.mean(log_ret[-21:])) if len(log_ret) >= 21 else mu
    period_ret  = float((prices[-1] / prices[0] - 1) * 100)
    last        = float(prices[-1])
    spread      = vol

    levels = {
        "strong_resistance": last * (1 + spread * 0.35),
        "resistance":        last * (1 + spread * 0.18),
        "equilibrium":       last,
        "support":           last * (1 - spread * 0.15),
        "strong_support":    last * (1 - spread * 0.30),
    }

    density = riemann_density(prices)

    # Monte Carlo — 2000 paths × 252 days
    np.random.seed(42)
    sims = np.exp(np.cumsum(np.random.normal(mu, np.std(log_ret), (2000, 252)), axis=1)) * last
    p10  = np.percentile(sims, 10, axis=0)
    p50  = np.percentile(sims, 50, axis=0)
    p90  = np.percentile(sims, 90, axis=0)

    # 90-day scenario probs
    final_90 = sims[:, 62]
    p_bull    = float((final_90 > last * 1.10).mean() * 100)
    p_lat     = float(((final_90 >= last * 0.95) & (final_90 <= last * 1.10)).mean() * 100)
    p_bear    = float((final_90 < last * 0.95).mean() * 100)
    tot       = p_bull + p_lat + p_bear
    p_bull, p_lat, p_bear = p_bull/tot*100, p_lat/tot*100, p_bear/tot*100

    axix = compute_axix_score(vol, hurst, momentum_21, coherence, energy)

    # Historical stats
    annual_rets = []
    if isinstance(df.index, pd.DatetimeIndex):
        for yr in df.index.year.unique():
            yr_prices = df.loc[df.index.year == yr, "Close"].values
            if len(yr_prices) >= 2:
                annual_rets.append(float((yr_prices[-1] / yr_prices[0] - 1) * 100))

    return {
        "prices":        prices,
        "log_returns":   log_ret,
        "vol_annual":    vol,
        "hurst":         hurst,
        "energy":        energy,
        "coherence":     coherence,
        "momentum_21":   momentum_21,
        "levels":        levels,
        "density":       density,
        "period_return": period_ret,
        "p_bull":        p_bull,
        "p_lateral":     p_lat,
        "p_bear":        p_bear,
        "proj_p10":      p10,
        "proj_p50":      p50,
        "proj_p90":      p90,
        "dominant_signal": axix["signal"],
        "last_price":    last,
        "first_price":   float(prices[0]),
        "max_price":     float(prices.max()),
        "min_price":     float(prices.min()),
        "mean_price":    float(prices.mean()),
        "std_price":     float(prices.std()),
        "annual_returns": annual_rets,
        "axix":          axix,
        "kurt":          float(stats.kurtosis(log_ret)),
        "skew":          float(stats.skew(log_ret)),
    }


# ─── YEAR-OVER-YEAR TABLE ────────────────────────────────────────────────────

def yoy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Returns year-by-year OHLC + return stats."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame()
    rows = []
    for yr in sorted(df.index.year.unique()):
        sub = df.loc[df.index.year == yr, "Close"].dropna()
        if len(sub) < 5:
            continue
        p = sub.values
        lr = np.diff(np.log(p))
        rows.append({
            "Año":        int(yr),
            "Apertura":   round(float(p[0]), 2),
            "Cierre":     round(float(p[-1]), 2),
            "Máximo":     round(float(p.max()), 2),
            "Mínimo":     round(float(p.min()), 2),
            "Retorno %":  round(float((p[-1]/p[0]-1)*100), 2),
            "Vol anual %":round(float(np.std(lr)*np.sqrt(252)*100), 2),
        })
    return pd.DataFrame(rows).set_index("Año")


# ─── BACKTESTING LEVELS ──────────────────────────────────────────────────────

def backtest_levels(df: pd.DataFrame, lookforward: int = 20) -> dict:
    """
    For each historical window, compute levels and check if price
    respected them in the next `lookforward` bars.
    Returns hit-rate stats.
    """
    prices = df["Close"].values.astype(float)
    n = len(prices)
    if n < 100:
        return {}

    windows = range(60, n - lookforward, max(1, (n - 60 - lookforward) // 40))
    res_hits, sup_hits, total = 0, 0, 0

    for i in windows:
        sub   = prices[:i]
        lr    = np.diff(np.log(sub))
        vol   = float(np.std(lr) * np.sqrt(252))
        last  = float(sub[-1])
        res   = last * (1 + vol * 0.18)
        sup   = last * (1 - vol * 0.15)
        fwd   = prices[i:i + lookforward]
        total += 1
        # Resistance: did price touch or bounce from res level?
        if fwd.max() >= res * 0.995:
            res_hits += 1
        # Support: did price hold above support?
        if fwd.min() >= sup * 0.985:
            sup_hits += 1

    if total == 0:
        return {}

    return {
        "windows_tested":    total,
        "resistance_hit_pct": round(res_hits / total * 100, 1),
        "support_held_pct":   round(sup_hits / total * 100, 1),
        "lookforward_bars":   lookforward,
    }


# ─── SPECTRAL / GUE GEOMETRY ─────────────────────────────────────────────────

def topophysical_geometry(log_returns: np.ndarray) -> dict:
    """
    Returns spectral and GUE-inspired geometric statistics.
    """
    n   = len(log_returns)
    # Power spectral density
    freqs, psd = periodogram(log_returns, fs=1.0)
    # Nearest-neighbor spacing (GUE analog)
    sorted_ret = np.sort(log_returns)
    spacings   = np.diff(sorted_ret)
    mean_sp    = spacings.mean()
    norm_sp    = spacings / mean_sp if mean_sp != 0 else spacings
    # GUE Wigner-Dyson fit parameter
    wd_fit, _  = stats.kstest(norm_sp, lambda x: 1 - np.exp(-np.pi * x**2 / 4))
    # Autocorrelation up to lag 20
    acf = [float(pd.Series(log_returns).autocorr(lag=k)) for k in range(1, 21)]
    return {
        "freqs":       freqs,
        "psd":         psd,
        "norm_spacings": norm_sp,
        "acf":         acf,
        "wd_ks_stat":  float(wd_fit),
        "gue_fit_quality": "Alta" if wd_fit < 0.05 else "Media" if wd_fit < 0.12 else "Baja",
    }
