"""
Palm Oil Price Prediction — Weather Data Extraction & Feature Engineering
==========================================================================
Data sources (all free, no paid keys required):
  - Open-Meteo Historical API  : ERA5-backed daily weather (precip, temp, solar)
  - NOAA CPC                   : ONI El Niño index (monthly)
  - NASA POWER                 : Solar radiation cross-check (optional)

Output: monthly DataFrame of engineered features ready for model training.

Install dependencies:
    pip install requests pandas numpy scipy
"""

import time
import requests
import numpy as np
import pandas as pd
from io import StringIO
from datetime import date, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# 1. CONFIGURATION — growing region coordinates
# ---------------------------------------------------------------------------

REGIONS = {
    "sabah":        {"lat": 5.5,  "lon": 117.5},   # East Malaysia (top producer)
    "riau_sumatra": {"lat": 0.5,  "lon": 102.0},   # Sumatra, Indonesia
    "west_kalim":   {"lat": 0.0,  "lon": 110.5},   # West Kalimantan
    "peninsular_my":{"lat": 4.0,  "lon": 102.0},   # Peninsular Malaysia
}

DEFAULT_START = "1990-01-01"
DEFAULT_END   = str(date.today() - timedelta(days=7))  # ERA5 has ~5-7 day lag


# ---------------------------------------------------------------------------
# 2. WEATHER EXTRACTION — Open-Meteo Historical API
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR      = "weather_cache"   # local CSV cache — skip re-fetching on reruns


def _cache_path(name: str, start: str, end: str) -> str:
    import os
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}_{start}_{end}.csv")


def fetch_weather(lat: float, lon: float,
                  start: str = DEFAULT_START,
                  end:   str = DEFAULT_END,
                  cache_name: Optional[str] = None,
                  max_retries: int = 6) -> pd.DataFrame:
    """
    Fetch daily ERA5 weather for a single lat/lon point.
    - Retries on 429/5xx with exponential backoff (2, 4, 8, 16, 32, 64 s).
    - Caches result to CSV so reruns are instant.
    Returns a DataFrame indexed by date with columns:
        precip_mm, tmax_c, tmean_c, tmin_c, solar_mj
    """
    # --- cache hit? ---
    if cache_name:
        cp = _cache_path(cache_name, start, end)
        import os
        if os.path.exists(cp):
            print(f"    (cache hit: {cp})")
            return pd.read_csv(cp, index_col=0, parse_dates=True)

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily": ",".join([
            "precipitation_sum",       # mm/day
            "temperature_2m_max",      # °C
            "temperature_2m_mean",     # °C
            "temperature_2m_min",      # °C
            "shortwave_radiation_sum", # MJ/m²/day
        ]),
        "timezone": "UTC",
        "models":   "era5",
    }

    # --- exponential backoff retry loop ---
    for attempt in range(max_retries):
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=90)

        if resp.status_code == 200:
            break
        elif resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** (attempt + 1)          # 2, 4, 8, 16, 32, 64 s
            print(f"    HTTP {resp.status_code} — waiting {wait}s before retry "
                  f"({attempt + 1}/{max_retries})...")
            time.sleep(wait)
        else:
            resp.raise_for_status()            # propagate unexpected errors
    else:
        resp.raise_for_status()                # exhausted retries

    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date":      pd.to_datetime(data["time"]),
        "precip_mm": data["precipitation_sum"],
        "tmax_c":    data["temperature_2m_max"],
        "tmean_c":   data["temperature_2m_mean"],
        "tmin_c":    data["temperature_2m_min"],
        "solar_mj":  data["shortwave_radiation_sum"],
    }).set_index("date")

    # --- write cache ---
    if cache_name:
        df.to_csv(_cache_path(cache_name, start, end))

    return df


def fetch_all_regions(start: str = DEFAULT_START,
                      end:   str = DEFAULT_END,
                      inter_request_delay: float = 5.0) -> pd.DataFrame:
    """
    Fetch weather for all REGIONS and return a single daily DataFrame
    with region-prefixed columns (e.g. sabah_precip_mm).

    inter_request_delay: seconds to wait between requests (default 5s).
    Open-Meteo's free tier allows ~10 req/min; 5s keeps us safely under.
    Results are cached to weather_cache/ so rerunning is instant.
    """
    frames = []
    for i, (name, coords) in enumerate(REGIONS.items()):
        print(f"  Fetching weather: {name} ({coords['lat']}, {coords['lon']})...")
        df = fetch_weather(
            coords["lat"], coords["lon"], start, end,
            cache_name=name,
        )
        df.columns = [f"{name}_{c}" for c in df.columns]
        frames.append(df)

        # Pause between requests — skip delay after the last one
        if i < len(REGIONS) - 1:
            print(f"    Waiting {inter_request_delay}s to respect rate limit...")
            time.sleep(inter_request_delay)

    return pd.concat(frames, axis=1)


# ---------------------------------------------------------------------------
# 3. ONI — NOAA CPC El Niño Index
# ---------------------------------------------------------------------------

# Primary source: NOAA CPC fixed-width table
# Fallback:       NOAA PSL plain two-column format (year + 12 monthly values)
ONI_SOURCES = [
    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "https://psl.noaa.gov/data/correlation/oni.data",
]

SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5,
    "MJJ": 6, "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10,
    "OND": 11, "NDJ": 12,
}

# Known 3-letter season codes — used to auto-detect the file format
_SEASON_CODES = set(SEASON_TO_MONTH.keys())


def _parse_cpc_format(text: str) -> pd.DataFrame:
    """
    Parse the NOAA CPC fixed-width ONI table.

    The file has gone through several layout revisions. We handle all of them:

    Format A (original, 5 cols):
        SEAS  YR   TOTAL   CLIM   ANOM
        DJF  1950  24.72  24.93  -0.21

    Format B (newer, 3 cols):
        SEAS  YR   ANOM
        DJF  1950  -0.21

    Format C (year-wide, 12 values per row — seen in some mirrors):
        YR  DJF  JFM  FMA  MAM  AMJ  MJJ  JJA  JAS  ASO  SON  OND  NDJ
        1950  -0.21  0.01  ...

    We auto-detect by inspecting the first non-blank, non-comment line.
    """
    lines = [l for l in text.splitlines() if l.strip()]

    # ---- find the header line and data start ----
    header_idx = None
    for i, line in enumerate(lines):
        upper = line.upper()
        if "SEAS" in upper or "YR" in upper:
            header_idx = i
            break

    data_lines = lines[header_idx + 1:] if header_idx is not None else lines

    # ---- detect format from first data line ----
    first = data_lines[0].split() if data_lines else []

    # Format C: first token is a 4-digit year, rest are 12 floats
    if first and first[0].isdigit() and len(first[0]) == 4 and len(first) >= 13:
        return _parse_wide_format(data_lines, header_line=lines[header_idx] if header_idx is not None else None)

    # Format A or B: first token is a season code (e.g. "DJF")
    records = []
    for line in data_lines:
        parts = line.split()
        if not parts:
            continue
        # Robustly skip any remaining header/comment lines
        if parts[0].upper() in ("SEAS", "YR") or not parts[0].upper() in _SEASON_CODES:
            continue
        if len(parts) < 3:
            continue
        try:
            season = parts[0].upper()
            year   = int(parts[1])
            # ANOM is the last column regardless of whether TOTAL/CLIM are present
            anom   = float(parts[-1])
            # Sanity check: ONI anomalies are always in [-4, +4]
            if not (-5.0 < anom < 5.0):
                continue
            records.append({"season": season, "year": year, "oni": anom})
        except (ValueError, IndexError):
            continue

    if not records:
        raise ValueError("CPC format parser produced 0 records — check raw text.")

    df = pd.DataFrame(records)
    df["month"] = df["season"].map(SEASON_TO_MONTH)
    df["date"]  = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.set_index("date")[["oni"]].sort_index()


def _parse_wide_format(lines: list, header_line: Optional[str] = None) -> pd.DataFrame:
    """
    Parse year-wide ONI format:
        YR  DJF  JFM  FMA  MAM  AMJ  MJJ  JJA  JAS  ASO  SON  OND  NDJ
        1950  -0.21  0.01  ...
    """
    # Reconstruct column order from header if available
    if header_line:
        cols = header_line.split()
        season_cols = [c.upper() for c in cols if c.upper() in _SEASON_CODES]
    else:
        season_cols = list(SEASON_TO_MONTH.keys())  # default order

    records = []
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        year = int(parts[0])
        values = parts[1:]
        for i, season in enumerate(season_cols):
            if i >= len(values):
                break
            try:
                anom = float(values[i])
                if anom <= -99:   # missing value sentinel
                    continue
                records.append({"season": season, "year": year, "oni": anom})
            except ValueError:
                continue

    df = pd.DataFrame(records)
    df["month"] = df["season"].map(SEASON_TO_MONTH)
    df["date"]  = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.set_index("date")[["oni"]].sort_index()


def _parse_psl_format(text: str) -> pd.DataFrame:
    """
    Parse NOAA PSL plain ONI file:
        First line: start_year  end_year
        Subsequent lines: year  jan feb mar apr may jun jul aug sep oct nov dec
        Missing values marked as -99.9 or -999
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    records = []
    for line in lines:
        parts = line.split()
        # Lines with 13 tokens are data rows (year + 12 months)
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        for month_idx, val_str in enumerate(parts[1:], start=1):
            try:
                val = float(val_str)
                if val <= -99:
                    continue
                records.append({"year": year, "month": month_idx, "oni": val})
            except ValueError:
                continue

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.set_index("date")[["oni"]].sort_index()


def fetch_oni(cache_path: str = "weather_cache/oni.csv") -> pd.DataFrame:
    """
    Download and parse the NOAA ONI index, trying multiple sources and
    format parsers. Caches the result to CSV.

    Returns a monthly Series indexed by date with column 'oni'.
    """
    import os
    if os.path.exists(cache_path):
        print(f"    (cache hit: {cache_path})")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    last_exc = None
    for url in ONI_SOURCES:
        print(f"    Trying ONI source: {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            text = resp.text

            # Print first 3 data lines for diagnosis
            preview = [l for l in text.splitlines() if l.strip()][:5]
            print(f"    File preview (first 5 lines):\n      " + "\n      ".join(preview))

            # Try CPC format first, then PSL
            try:
                df = _parse_cpc_format(text)
            except Exception:
                df = _parse_psl_format(text)

            if len(df) < 12:
                raise ValueError(f"Parsed only {len(df)} rows — likely a format mismatch.")

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path)
            print(f"    → {len(df)} monthly ONI values "
                  f"({df.index[0].year}–{df.index[-1].year})")
            return df

        except Exception as exc:
            print(f"    Failed ({type(exc).__name__}: {exc})")
            last_exc = exc

    raise RuntimeError(
        f"All ONI sources failed. Last error: {last_exc}\n"
        "Check your internet connection or try opening one of these URLs manually:\n"
        + "\n".join(ONI_SOURCES)
    )


# ---------------------------------------------------------------------------
# 4. DAILY → MONTHLY AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily weather to monthly for each region.
    Computes per-region:
        - total_precip_mm   : monthly rainfall sum
        - mean_tmax_c       : mean of daily max temps
        - mean_tmean_c      : mean of daily mean temps
        - heat_stress_days  : days with tmax > 33°C
        - mean_solar_mj     : mean daily solar radiation
        - dry_days          : days with < 1mm rain
    """
    monthly_parts = []
    for name in REGIONS:
        cols = {c: c.replace(f"{name}_", "") for c in daily.columns if c.startswith(f"{name}_")}
        sub = daily[[c for c in daily.columns if c.startswith(f"{name}_")]].copy()
        sub.columns = [c.replace(f"{name}_", "") for c in sub.columns]

        m = sub.resample("MS").agg(
            total_precip_mm=("precip_mm", "sum"),
            mean_tmax_c    =("tmax_c",    "mean"),
            mean_tmean_c   =("tmean_c",   "mean"),
            mean_solar_mj  =("solar_mj",  "mean"),
            heat_stress_days=("tmax_c",   lambda x: (x > 33).sum()),
            dry_days       =("precip_mm", lambda x: (x < 1.0).sum()),
        )
        m.columns = [f"{name}_{c}" for c in m.columns]
        monthly_parts.append(m)

    return pd.concat(monthly_parts, axis=1)


# ---------------------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def compute_dry_season_intensity(precip_series: pd.Series,
                                 dry_months: tuple = (6, 7, 8, 9)) -> pd.Series:
    """
    Dry Season Intensity (DSI): cumulative rainfall deficit vs
    long-run monthly mean during the primary dry season months (Jun–Sep).
    Higher DSI = more severe drought.
    Returns a monthly series, non-null only during dry season months.
    """
    monthly_avg = precip_series.groupby(precip_series.index.month).mean()
    deficit = precip_series.copy() * np.nan

    for idx in precip_series.index:
        if idx.month in dry_months:
            deficit[idx] = max(0, monthly_avg[idx.month] - precip_series[idx])

    return deficit.rename("dsi")


def compute_spei_proxy(precip: pd.Series, tmean: pd.Series,
                       window: int = 3) -> pd.Series:
    """
    Simplified SPEI proxy: z-score of (P - PET) over a rolling window.
    PET estimated via Thornthwaite method (function of mean temperature).
    Full SPEI requires the `spei` or `climate_indices` package; this is
    a lightweight approximation sufficient for ML feature use.
    """
    # Thornthwaite PET approximation (mm/month)
    def pet_thornthwaite(tmean_c: float) -> float:
        if tmean_c <= 0:
            return 0.0
        return 16.0 * (10.0 * max(tmean_c, 0) / 1200.0) ** 1.514  # simplified

    pet = tmean.apply(pet_thornthwaite)
    wb = precip - pet                        # water balance
    wb_roll = wb.rolling(window).sum()       # rolling sum
    spei = (wb_roll - wb_roll.mean()) / (wb_roll.std() + 1e-9)
    return spei.rename(f"spei_{window}m")


def lag_feature(series: pd.Series, lags: list) -> pd.DataFrame:
    """Create multiple lag columns from a series."""
    return pd.concat(
        {f"{series.name}_lag{k}m": series.shift(k) for k in lags},
        axis=1
    )


def rolling_feature(series: pd.Series, windows: list,
                    agg: str = "mean") -> pd.DataFrame:
    """Create rolling mean (or sum) columns from a series."""
    return pd.concat(
        {f"{series.name}_roll{w}m_{agg}": getattr(series.rolling(w), agg)()
         for w in windows},
        axis=1
    )


def engineer_features(monthly: pd.DataFrame,
                       oni: pd.DataFrame,
                       primary_region: str = "sabah") -> pd.DataFrame:
    """
    Build the full feature matrix from monthly weather and ONI data.

    Engineered features:
        Weather (per region):
            - Rolling 3/6/12-month total precipitation
            - Rolling 3-month mean temperature
            - Dry Season Intensity
            - SPEI proxy (3-month)
            - Heat stress days (raw + 3-month rolling)
            - Solar radiation anomaly vs 12-month trailing mean

        ENSO:
            - ONI raw (current month)
            - ONI lagged 6/9/12/15/18 months
            - ONI × DSI interaction (El Niño + drought = compounding stress)
            - ENSO phase flag (-1 La Niña, 0 Neutral, +1 El Niño)

        Cross-region:
            - Mean precipitation across all regions (area-wide signal)
            - Coefficient of variation of precip (spatial heterogeneity)
    """
    feat = pd.DataFrame(index=monthly.index)

    # --- Per-region features ---
    for name in REGIONS:
        p_col = f"{name}_total_precip_mm"
        t_col = f"{name}_mean_tmean_c"
        h_col = f"{name}_heat_stress_days"
        s_col = f"{name}_mean_solar_mj"

        if p_col not in monthly.columns:
            continue

        p = monthly[p_col]
        t = monthly[t_col]
        h = monthly[h_col]
        s = monthly[s_col]

        # Rolling precipitation
        for w in [3, 6, 12]:
            feat[f"{name}_precip_roll{w}m"] = p.rolling(w).sum()

        # Drought signals
        feat[f"{name}_dsi"] = compute_dry_season_intensity(p)
        feat[f"{name}_spei_3m"] = compute_spei_proxy(p, t, window=3)
        feat[f"{name}_spei_6m"] = compute_spei_proxy(p, t, window=6)

        # Temperature
        feat[f"{name}_tmean_roll3m"] = t.rolling(3).mean()
        feat[f"{name}_heat_days_roll3m"] = h.rolling(3).sum()

        # Solar radiation anomaly vs trailing 12-month average
        feat[f"{name}_solar_anom"] = s - s.rolling(12).mean()

    # --- ONI / ENSO features ---
    oni_monthly = oni.reindex(monthly.index, method="ffill")["oni"]

    feat["oni_current"] = oni_monthly
    for lag in [6, 9, 12, 15, 18]:
        feat[f"oni_lag{lag}m"] = oni_monthly.shift(lag)

    # ENSO phase flag: El Niño=+1, La Niña=-1, Neutral=0
    feat["enso_phase"] = pd.cut(
        oni_monthly, bins=[-10, -0.5, 0.5, 10],
        labels=[-1, 0, 1]
    ).astype(float)

    # ONI × DSI interaction (primary region)
    dsi_col = f"{primary_region}_dsi"
    if dsi_col in feat.columns:
        feat["oni_x_dsi"] = feat["oni_lag12m"].fillna(0) * feat[dsi_col].fillna(0)

    # --- Cross-region features ---
    precip_cols = [f"{n}_total_precip_mm" for n in REGIONS if f"{n}_total_precip_mm" in monthly.columns]
    if precip_cols:
        precip_matrix = monthly[precip_cols]
        feat["area_mean_precip"]  = precip_matrix.mean(axis=1)
        feat["area_cv_precip"]    = precip_matrix.std(axis=1) / (precip_matrix.mean(axis=1) + 1e-9)
        feat["area_precip_roll3m"] = feat["area_mean_precip"].rolling(3).sum()

    return feat


# ---------------------------------------------------------------------------
# 6. PIPELINE RUNNER
# ---------------------------------------------------------------------------

def run_pipeline(start: str = DEFAULT_START,
                 end:   str = DEFAULT_END,
                 primary_region: str = "sabah") -> pd.DataFrame:
    """
    Full end-to-end pipeline. Returns model-ready monthly feature DataFrame.
    """
    print("=" * 60)
    print("Palm oil weather feature pipeline")
    print(f"  Period : {start} → {end}")
    print(f"  Regions: {list(REGIONS.keys())}")
    print("=" * 60)

    # Step 1: Weather
    print("\n[1/4] Fetching daily weather from Open-Meteo (ERA5)...")
    daily = fetch_all_regions(start, end)
    print(f"  → {len(daily)} daily rows, {daily.shape[1]} columns")

    # Step 2: ONI
    print("\n[2/4] Fetching ONI index from NOAA CPC / PSL (with fallback)...")
    oni = fetch_oni()
    print(f"  → {len(oni)} monthly ONI observations ({oni.index[0].year}–{oni.index[-1].year})")

    # Step 3: Monthly aggregation
    print("\n[3/4] Aggregating daily → monthly...")
    monthly = aggregate_to_monthly(daily)
    print(f"  → {len(monthly)} monthly rows")

    # Step 4: Feature engineering
    print("\n[4/4] Engineering features...")
    features = engineer_features(monthly, oni, primary_region=primary_region)

    # Drop rows with all NaN (initial lag/rolling window warmup)
    features = features.dropna(how="all")

    print(f"\n  Done. Feature matrix: {features.shape[0]} rows × {features.shape[1]} columns")
    print(f"\n  Feature columns:\n  " + "\n  ".join(features.columns.tolist()))

    return features


# ---------------------------------------------------------------------------
# 7. QUICK DIAGNOSTIC — check data quality
# ---------------------------------------------------------------------------

def summarise(features: pd.DataFrame) -> pd.DataFrame:
    """Print a quick quality summary of the feature matrix."""
    summary = pd.DataFrame({
        "non_null_%": (features.notna().mean() * 100).round(1),
        "mean":        features.mean().round(3),
        "std":         features.std().round(3),
        "min":         features.min().round(3),
        "max":         features.max().round(3),
    })
    print("\nFeature quality summary:")
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# 8. ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run for the last 30 years — adjust dates as needed
    features = run_pipeline(
        start="1995-01-01",
        end=str(date.today() - timedelta(days=7)),
        primary_region="sabah",
    )

    # Save to CSV for downstream modelling
    out_path = "palm_oil_features.csv"
    features.to_csv(out_path)
    print(f"\nSaved to {out_path}")

    # Optional: quick quality check
    summarise(features)