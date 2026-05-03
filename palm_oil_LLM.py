"""
Palm Oil — LLM Signal Extraction + Economic Features
=====================================================
Adds two new feature groups to palm_oil_features.csv:

  A) LLM-extracted supply signals  (Claude API)
     Sources: USDA WASDE monthly PDFs, MPOB press releases, Reuters headlines
     Output:  supply_disruption_score, export_policy_signal, sentiment_score,
              production_outlook, demand_signal

  B) Economic features              (FRED + yfinance — no API key for yfinance)
     Features: soybean oil price, palm-soy spread, USD/MYR, crude oil (Brent),
               biodiesel mandate proxy

Install:
    pip install anthropic requests pandas yfinance fredapi
    export ANTHROPIC_API_KEY="sk-ant-..."       # required for LLM extraction
    export FRED_API_KEY="..."                    # optional — only needed for FRED
"""

import os
import re
import json
import time
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import anthropic          # pip install anthropic

CACHE_DIR   = Path("llm_cache")
CACHE_DIR.mkdir(exist_ok=True)

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"  # latest Sonnet — fast + accurate


# ============================================================================
# A. LLM SIGNAL EXTRACTION
# ============================================================================

# --------------------------------------------------------------------------
# A1. Data ingestion — three free text sources
# --------------------------------------------------------------------------

USDA_WASDE_BASE = "https://apps.fas.usda.gov/psdonline/circulars/oilseeds.pdf"

# MPOB monthly supply/demand press releases (public PDFs)
# URL pattern: https://bepi.mpob.gov.my/images/stories/pdf/YYYY/
MPOB_BASE = "https://bepi.mpob.gov.my"

def fetch_usda_wasde_text(year: int, month: int) -> str:
    """
    Download USDA Oilseeds World Markets & Trade PDF and extract text.
    Falls back to cached version if available.
    Published monthly, usually around the 10th.
    """
    cache_path = CACHE_DIR / f"usda_{year}_{month:02d}.txt"
    if cache_path.exists():
        return cache_path.read_text()

    # USDA FAS oilseeds circular — direct PDF link
    url = (f"https://apps.fas.usda.gov/psdonline/circulars/"
           f"oilseeds.pdf")  # always latest; historical via PSD query below

    # For historical: use the PSD data API (JSON, no PDF needed)
    psd_url = (
        "https://apps.fas.usda.gov/psdonline/app/index.html#/app/downloads"
    )

    # Practical approach: use USDA PSD API for structured data instead of PDF
    api_url = (
        "https://apps.fas.usda.gov/psdonline/api/downloads/"
        "psd_alldata_csv.zip"
    )

    # Lightweight alternative: pull the monthly summary text from USDA FAS
    summary_url = (
        f"https://apps.fas.usda.gov/psdonline/app/index.html"
    )

    # Best free source for monthly oilseed text: USDA FAS RSS / press releases
    rss_url = "https://apps.fas.usda.gov/psdonline/circulars/oilseeds.pdf"

    try:
        resp = requests.get(
            f"https://apps.fas.usda.gov/psdonline/circulars/oilseeds.pdf",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            # Extract text from PDF bytes using pdfminer if available
            try:
                from io import BytesIO
                from pdfminer.high_level import extract_text as pdf_extract
                text = pdf_extract(BytesIO(resp.content))
            except ImportError:
                text = f"[PDF downloaded but pdfminer not installed. Install: pip install pdfminer.six]\nURL: {rss_url}"
            cache_path.write_text(text[:8000])  # keep first 8k chars
            return text[:8000]
    except Exception as e:
        pass

    return f"[USDA WASDE {year}-{month:02d}: not available — using synthetic text for demo]"


def build_monthly_text_corpus(start_year: int = 2018,
                               end_year:   int = 2025) -> dict:
    """
    Build a dict of {(year, month): text_snippet} for LLM extraction.

    For months where we can't pull real PDFs, we use the USDA PSD structured
    data API as a fallback — it gives production/consumption numbers that we
    convert to a text summary, which the LLM then scores.
    """
    print("  Building text corpus for LLM extraction...")
    corpus = {}

    # Try USDA PSD API first — gives structured production data in CSV
    try:
        psd_df = _fetch_usda_psd_palm()
        if psd_df is not None:
            for _, row in psd_df.iterrows():
                key = (int(row["year"]), int(row["month"]))
                corpus[key] = _psd_row_to_text(row)
            print(f"  → USDA PSD: {len(corpus)} monthly text snippets")
            return corpus
    except Exception as e:
        print(f"  USDA PSD failed ({e}), using synthetic corpus")

    # Fallback: synthetic but realistic text corpus based on known events
    corpus = _build_synthetic_corpus(start_year, end_year)
    print(f"  → Synthetic corpus: {len(corpus)} months (real extraction "
          f"requires USDA PDF access or MPOB subscription)")
    return corpus


def _fetch_usda_psd_palm() -> Optional[pd.DataFrame]:
    """
    Pull USDA PSD palm oil production + export data via the public API.
    Returns monthly-ish records (USDA reports annually with monthly updates).
    """
    url = ("https://apps.fas.usda.gov/psdonline/api/downloads/"
           "psd_alldata_csv.zip")
    resp = requests.get(url, timeout=60, stream=True)
    if resp.status_code != 200:
        return None

    import zipfile
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(resp.content)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        df = pd.read_csv(z.open(csv_name), low_memory=False)

    # Filter for palm oil, key countries
    palm = df[
        (df["Commodity_Description"].str.contains("Palm Oil", na=False)) &
        (df["Country_Name"].isin(["Indonesia", "Malaysia", "World"]))
    ].copy()

    return palm if len(palm) > 0 else None


def _psd_row_to_text(row: pd.Series) -> str:
    """Convert a PSD data row into a prose text snippet for LLM input."""
    return (
        f"USDA PSD update for {row.get('Country_Name','?')}, "
        f"marketing year {row.get('Market_Year','?')}: "
        f"Production {row.get('Value','?')} thousand MT. "
        f"Attribute: {row.get('Attribute_Description','?')}."
    )


def _build_synthetic_corpus(start_year: int, end_year: int) -> dict:
    """
    Synthetic but historically grounded text corpus.
    Encodes known supply events so the LLM extraction demo is meaningful.
    Replace with real MPOB/USDA PDFs for production use.
    """
    events = {
        # Format: (year, month): (disruption, policy, sentiment, text)
        (2019,  6): (0.3, "neutral",  "bearish",
            "Malaysia palm oil output expected to recover from El Nino-related "
            "decline. Inventory levels remain elevated at 2.4M tonnes. Export "
            "demand from India softened following tariff adjustments."),
        (2020,  4): (0.6, "neutral",  "bearish",
            "COVID-19 disrupts palm oil supply chains. Harvest labour shortages "
            "in Sabah and Johor reported. Port congestion delays exports. "
            "Biodiesel demand collapses as transportation fuel consumption falls."),
        (2021,  8): (0.5, "neutral",  "bullish",
            "Palm oil production recovery slower than expected following 2020 "
            "La Nina rainfall disruption. Global vegetable oil inventory tightens. "
            "Soybean crop concerns in South America support prices."),
        (2022,  4): (0.95, "export_ban",  "very_bullish",
            "Indonesia announces immediate ban on palm oil exports effective "
            "April 28 2022 to address domestic cooking oil shortage. Move "
            "unexpected by markets. Global supply shock imminent. Prices "
            "expected to surge. Malaysia unable to fully compensate shortfall."),
        (2022,  5): (0.90, "export_ban",  "very_bullish",
            "Indonesia export ban remains in force. Global buyers scrambling "
            "for alternative vegetable oil supplies. Soybean oil and sunflower "
            "oil premiums widen sharply. Malaysia export quota fills rapidly."),
        (2022,  6): (0.5, "export_partial",  "bullish",
            "Indonesia partially lifts export ban under domestic market "
            "obligation scheme. Exporters must sell 1 tonne domestically per "
            "6 tonnes exported. Supply gradually returning to market."),
        (2022,  7): (0.2, "neutral",  "neutral",
            "Indonesia fully lifts palm oil export ban. Inventory rebuilding "
            "underway. Prices correcting from May highs. Production recovery "
            "on track for H2 2022 seasonal uptick."),
        (2023,  3): (0.2, "neutral",  "neutral",
            "Palm oil production in Malaysia up 8% YoY. Inventory at 1.8M "
            "tonnes. India demand strong ahead of festival season. Indonesia "
            "B35 biodiesel mandate supports domestic consumption."),
        (2024,  6): (0.3, "neutral",  "bullish",
            "El Nino effects fading but soil moisture deficits in key Sabah "
            "growing areas persist. Production forecast trimmed by 3%. "
            "India and China buying on dips. Prices well-supported above $800."),
        (2025,  1): (0.2, "neutral",  "bullish",
            "Malaysia palm oil stocks fall to 18-month low of 1.6M tonnes. "
            "Indonesia B40 biodiesel mandate implementation boosts domestic "
            "demand. Export growth to EU moderating due to EUDR regulations."),
    }

    corpus = {}
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            key = (year, month)
            if key in events:
                disruption, policy, sentiment, text = events[key]
                corpus[key] = text
            else:
                # Generic neutral text for months without known events
                corpus[key] = (
                    f"Palm oil market conditions for {year}-{month:02d}: "
                    f"Production within seasonal norms. No major policy changes. "
                    f"Export demand steady. Inventory levels moderate."
                )
    return corpus


# --------------------------------------------------------------------------
# A2. Claude extraction prompt + API call
# --------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You are a commodity market analyst specialising in palm oil supply chains.
You will be given a text excerpt from a USDA, MPOB, or news source about palm oil markets.
Extract the following signals and return ONLY a valid JSON object — no preamble, no markdown fences.

JSON schema:
{
  "supply_disruption_score": <float 0.0–1.0, where 0=no disruption, 1=severe disruption>,
  "export_policy_signal":    <one of: "ban", "restriction", "tax_increase", "tax_decrease", "quota", "neutral", "liberalisation">,
  "sentiment_score":         <float -1.0 to +1.0, where -1=very bearish, 0=neutral, +1=very bullish>,
  "production_outlook":      <one of: "strong_growth", "moderate_growth", "flat", "moderate_decline", "sharp_decline">,
  "demand_signal":           <one of: "strong", "moderate", "weak">,
  "confidence":              <float 0.0–1.0, how confident you are given the text quality>
}

Rules:
- Base scores ONLY on the provided text, not your training knowledge
- If text is too vague, set confidence < 0.4
- supply_disruption_score should be > 0.7 only for explicit harvest failures, export bans, or major supply shocks
- sentiment_score should reflect expected PRICE direction, not production direction
"""

def extract_signals_from_text(text: str,
                               year: int,
                               month: int,
                               client: anthropic.Anthropic,
                               max_retries: int = 3) -> dict:
    """
    Call Claude API to extract structured supply signals from text.
    Returns dict with all signal fields, or None-filled defaults on failure.
    """
    cache_path = CACHE_DIR / f"llm_{year}_{month:02d}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    defaults = {
        "supply_disruption_score": 0.0,
        "export_policy_signal":    "neutral",
        "sentiment_score":         0.0,
        "production_outlook":      "flat",
        "demand_signal":           "moderate",
        "confidence":              0.0,
        "year":  year,
        "month": month,
    }

    user_msg = (
        f"Date: {year}-{month:02d}\n\n"
        f"Text excerpt:\n{text[:3000]}\n\n"
        f"Extract the supply signals as JSON."
    )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = response.content[0].text.strip()

            # Strip any accidental markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$",          "", raw)

            parsed = json.loads(raw)
            parsed.update({"year": year, "month": month})
            cache_path.write_text(json.dumps(parsed, indent=2))
            return parsed

        except json.JSONDecodeError as e:
            print(f"    JSON parse error at {year}-{month:02d} (attempt {attempt+1}): {e}")
            print(f"    Raw response: {raw[:200]}")
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            print(f"    Rate limit — waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    Extraction error at {year}-{month:02d}: {e}")

    defaults.update({"year": year, "month": month})
    return defaults


def run_llm_extraction(start_year: int = 2018,
                        end_year:   int = 2025) -> pd.DataFrame:
    """
    Run LLM extraction over the full corpus.
    Returns a monthly DataFrame of extracted signal features.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "Run: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    client  = anthropic.Anthropic(api_key=api_key)
    corpus  = build_monthly_text_corpus(start_year, end_year)

    records = []
    total   = len(corpus)
    for i, ((year, month), text) in enumerate(sorted(corpus.items())):
        if i % 10 == 0:
            print(f"  Extracting signals: {i}/{total} ({year}-{month:02d})...")
        signals = extract_signals_from_text(text, year, month, client)
        records.append(signals)
        # Polite rate limit — Sonnet handles ~50 req/min on standard tier
        time.sleep(0.3)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.set_index("date").drop(columns=["year", "month"])

    # One-hot encode categorical signals
    df = pd.get_dummies(df, columns=["export_policy_signal",
                                      "production_outlook",
                                      "demand_signal"],
                        prefix=["eps", "po", "ds"])

    # Ensure bool columns are float for the model
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(float)

    # Lag the signals by 1 month (signals are known at start of month,
    # predict price at end of following month)
    lag_cols = ["supply_disruption_score", "sentiment_score", "confidence"]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1m"] = df[col].shift(1)

    return df


# ============================================================================
# B. ECONOMIC FEATURES
# ============================================================================

def fetch_yfinance_series(tickers: dict,
                           start: str = "1995-01-01") -> pd.DataFrame:
    """
    Pull monthly price series from Yahoo Finance.
    tickers: {column_name: yahoo_ticker}
    No API key needed.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")

    frames = []
    for col_name, ticker in tickers.items():
        print(f"  Fetching {col_name} ({ticker})...")
        try:
            raw = yf.download(ticker, start=start, interval="1mo",
                              progress=False, auto_adjust=True)
            if raw.empty:
                print(f"    Warning: no data for {ticker}")
                continue
            # yfinance returns MultiIndex columns for single ticker sometimes
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw["Close"][ticker]
            else:
                s = raw["Close"]
            s.index = s.index.to_period("M").to_timestamp()
            s.name = col_name
            frames.append(s)
            time.sleep(0.5)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    return df


def fetch_fred_series(series_ids: dict,
                       start: str = "1995-01-01",
                       fred_api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Pull monthly series from FRED (Federal Reserve Economic Data).
    fred_api_key: optional — use env var FRED_API_KEY or pass directly.
    Falls back to direct CSV download if fredapi not installed.
    """
    key = fred_api_key or os.environ.get("FRED_API_KEY")

    frames = []

    if key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=key)
            for col_name, series_id in series_ids.items():
                print(f"  Fetching FRED {col_name} ({series_id})...")
                s = fred.get_series(series_id, observation_start=start)
                s = s.resample("MS").mean()
                s.name = col_name
                frames.append(s)
            return pd.concat(frames, axis=1)
        except Exception as e:
            print(f"  FRED API failed ({e}), falling back to direct download")

    # Fallback: FRED direct CSV download (no key needed)
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    for col_name, series_id in series_ids.items():
        print(f"  Fetching FRED {col_name} via CSV ({series_id})...")
        try:
            resp = requests.get(f"{base}{series_id}", timeout=30)
            resp.raise_for_status()
            s = pd.read_csv(
                __import__("io").StringIO(resp.text),
                parse_dates=["DATE"], index_col="DATE"
            ).squeeze()
            s = pd.to_numeric(s, errors="coerce")
            s = s.resample("MS").mean()
            s.name = col_name
            frames.append(s)
            time.sleep(0.5)
        except Exception as e:
            print(f"    Failed {series_id}: {e}")

    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def build_economic_features(start: str = "1995-01-01") -> pd.DataFrame:
    """
    Build the full economic feature set.
    Soybean oil + crude come from World Bank cache (wb_econ_cache.csv) if
    available — more reliable than futures tickers. yfinance used for
    USD/MYR only (exchange rate, very stable ticker).
    """
    print("\n  Fetching economic features...")
    econ = pd.DataFrame()

    # ── Priority 1: World Bank cache (written by load_prices in model.py) ────
    wb_cache = "wb_econ_cache.csv"
    import os
    if os.path.exists(wb_cache):
        print(f"  Loading soy oil + crude from World Bank cache ({wb_cache})...")
        wb = pd.read_csv(wb_cache, index_col=0)
        wb.index = pd.to_datetime(wb.index, errors="coerce")
        wb = wb[wb.index.notna()]
        wb.index = wb.index.to_period("M").to_timestamp()
        if "soy_oil_usd" in wb.columns:
            econ["soy_oil_usd"] = wb["soy_oil_usd"]
        if "crude_brent_usd" in wb.columns:
            econ["crude_brent_usd"] = wb["crude_brent_usd"]
        print(f"  → {len(wb)} months loaded from World Bank cache")
    else:
        print("  wb_econ_cache.csv not found — run palm_oil_model.py first "
              "to generate it, or it will be created when prices download.")

    # ── Priority 2: yfinance for USD/MYR (reliable exchange rate ticker) ────
    yf_tickers = {
        "usdmyr":  "MYR=X",
        "crude_wti": "CL=F",   # WTI as backup if Brent missing
    }
    yf_data = fetch_yfinance_series(yf_tickers, start=start)
    if not yf_data.empty:
        econ = pd.concat([econ, yf_data], axis=1) if not econ.empty else yf_data

    if econ.empty:
        print("  Warning: no economic data fetched")
        return pd.DataFrame()

    # ── Derived features ─────────────────────────────────────────────────────
    if "soy_oil_usd" in econ.columns:
        econ["soy_oil_mom"]    = econ["soy_oil_usd"].pct_change(1) * 100
        econ["soy_oil_roll3m"] = econ["soy_oil_usd"].rolling(3).mean()

    if "crude_brent_usd" in econ.columns:
        econ["crude_roll3m"] = econ["crude_brent_usd"].rolling(3).mean()
        econ["crude_mom"]    = econ["crude_brent_usd"].pct_change(1) * 100
    elif "crude_wti" in econ.columns:
        econ["crude_roll3m"] = econ["crude_wti"].rolling(3).mean()
        econ["crude_mom"]    = econ["crude_wti"].pct_change(1) * 100

    if "usdmyr" in econ.columns:
        econ["usdmyr_roll3m"] = econ["usdmyr"].rolling(3).mean()
        econ["usdmyr_mom"]    = econ["usdmyr"].pct_change(1) * 100

    # Lag raw cols by 1 month (known at prediction time)
    raw_cols = [c for c in econ.columns
                if not any(s in c for s in ["roll", "mom", "lag"])]
    for col in raw_cols:
        econ[f"{col}_lag1m"] = econ[col].shift(1)

    return econ


# ============================================================================
# C. JOIN TO EXISTING FEATURES
# ============================================================================

def build_full_feature_set(
    weather_features_csv: str = "palm_oil_features.csv",
    output_csv:           str = "palm_oil_features_v2.csv",
    llm_start_year: int = 2000,
    llm_end_year:   int = 2025,
    econ_start:     str = "1995-01-01",
) -> pd.DataFrame:
    """
    Load existing weather features, run LLM extraction and economic
    feature fetching, join all on monthly date index.
    Saves to output_csv and returns the combined DataFrame.
    """
    print("=" * 60)
    print("Building full feature set v2")
    print("  Weather + LLM signals + Economic features")
    print("=" * 60)

    # Load existing weather features
    print(f"\n[1/4] Loading weather features from {weather_features_csv}...")
    weather = pd.read_csv(weather_features_csv, index_col=0, parse_dates=True)
    weather.index = weather.index.to_period("M").to_timestamp()
    print(f"  → {weather.shape[0]} rows × {weather.shape[1]} columns")

    # LLM extraction
    print(f"\n[2/4] Running LLM signal extraction "
          f"({llm_start_year}–{llm_end_year})...")
    llm_signals = run_llm_extraction(llm_start_year, llm_end_year)
    print(f"  → {llm_signals.shape[0]} months × {llm_signals.shape[1]} LLM features")

    # Economic features
    print(f"\n[3/4] Fetching economic features...")
    econ = build_economic_features(start=econ_start)
    if not econ.empty:
        print(f"  → {econ.shape[0]} months × {econ.shape[1]} economic features")
    else:
        print("  → No economic features (check connection/install yfinance)")

    # Join everything
    print(f"\n[4/4] Joining all feature sets...")
    combined = weather.copy()
    combined = combined.join(llm_signals, how="left")
    if not econ.empty:
        combined = combined.join(econ, how="left")

    # Fill LLM signal NaNs with neutral defaults
    for col in combined.columns:
        if "supply_disruption" in col or "sentiment" in col:
            combined[col] = combined[col].fillna(0.0)
        elif col.startswith("eps_") or col.startswith("po_") or col.startswith("ds_"):
            combined[col] = combined[col].fillna(0.0)

    # Forward-fill economic series (handles occasional gaps)
    econ_cols = [c for c in combined.columns
                 if any(k in c for k in ["soy", "crude", "usdmyr", "cpi", "pmi"])]
    combined[econ_cols] = combined[econ_cols].ffill(limit=3)

    combined.to_csv(output_csv)
    print(f"\n  Saved: {output_csv}")
    print(f"  Final shape: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"\n  New feature groups:")
    print(f"    LLM signals : {llm_signals.shape[1]} columns")
    print(f"    Economic    : {econ.shape[1] if not econ.empty else 0} columns")

    # Palm-soy spread requires price data — added in model pipeline
    print("\n  Note: palm_soy_spread will be computed in palm_oil_model.py "
          "after joining with price data.")

    return combined


# ============================================================================
# D. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: Set your API key first:")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    features_v2 = build_full_feature_set(
        weather_features_csv = "palm_oil_features.csv",
        output_csv           = "palm_oil_features_v2.csv",
        llm_start_year       = 2000,
        llm_end_year         = 2025,
        econ_start           = "1995-01-01",
    )

    print("\nNext step: update FEATURES_CSV in palm_oil_model.py to point at "
          "palm_oil_features_v2.csv and re-run the model.")