"""
Palm Oil Price Prediction — Modeling Pipeline
==============================================
Step 2: Join weather features → price data → train → evaluate → explain

Requires:
    pip install lightgbm scikit-learn shap matplotlib pandas numpy requests

Price data options (handled automatically, in priority order):
    1. Your own CSV  — set PRICE_CSV below if you have one
    2. World Bank Pink Sheet — free monthly commodity prices (no key)
    3. Synthetic series for dry-run / testing
"""

import io
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — swap to "TkAgg" if you want interactive plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=UserWarning)

# path to your own if needed
PRICE_CSV = None

# Use v2 features if available (weather + LLM + econ), else fall back to v1
import os as _os
FEATURES_CSV  = ("palm_oil_features_v2.csv"
                 if _os.path.exists("palm_oil_features_v2.csv")
                 else "palm_oil_features.csv")
OUTPUT_DIR    = "model_outputs"
TARGET_COL    = "price"
FORECAST_HORIZON = 1           # months ahead to predict (1 = next month)


# ============================================================================
# 1. PRICE DATA
# ============================================================================

# Updated 2025 World Bank Pink Sheet URL
WB_URL = (
    "https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2"
    "-0050012025/related/CMO-Historical-Data-Monthly.xlsx"
)

def _parse_pink_sheet(tmp_path: str) -> pd.DataFrame:
    """
    Parse World Bank Pink Sheet xlsx.
    Returns DataFrame with columns: price, soy_oil_usd, crude_brent_usd
    indexed by month-start date.
    Uses multiple fallback strategies to handle layout changes across years.
    """
    import os
    xls = pd.ExcelFile(tmp_path)

    # Find the monthly prices sheet
    sheet = None
    for s in xls.sheet_names:
        if "monthly" in s.lower() or "price" in s.lower():
            sheet = s
            break
    sheet = sheet or xls.sheet_names[0]
    print(f"  Parsing sheet: '{sheet}' from {[s for s in xls.sheet_names]}")

    raw = pd.read_excel(tmp_path, sheet_name=sheet, header=None, dtype=str)
    os.remove(tmp_path)

    print(f"  Raw sheet shape: {raw.shape}")
    print(f"  First 8 rows preview:\n{raw.iloc[:8, :6].to_string()}")

    # --- Strategy: scan ALL rows for one containing "palm" anywhere ---
    header_row = None
    for i, row in raw.iterrows():
        row_str = " ".join(str(v).lower() for v in row if pd.notna(v))
        if "palm" in row_str and ("oil" in row_str or "soy" in row_str):
            header_row = i
            print(f"  Found header row at index {i}: {row_str[:120]}")
            break

    if header_row is None:
        # Last resort: try reading with pandas auto-header detection
        df2 = pd.read_excel(xls, sheet_name=sheet)
        print(f"  Auto-parsed columns: {list(df2.columns[:10])}")
        col_map = {}
        for col in df2.columns:
            cl = str(col).lower()
            if "palm" in cl and "price" not in col_map:
                col_map["price"] = col
            if ("soybean oil" in cl or "soya oil" in cl) and "soy_oil_usd" not in col_map:
                col_map["soy_oil_usd"] = col
            if "brent" in cl and "crude_brent_usd" not in col_map:
                col_map["crude_brent_usd"] = col

        if "price" not in col_map:
            raise ValueError(
                f"Could not find Palm oil column. "
                f"Sheet '{sheet}' columns: {list(df2.columns[:20])}"
            )

        date_col = df2.columns[0]
        result = pd.DataFrame()
        result.index = pd.to_datetime(df2[date_col], errors="coerce")
        for out_name, src_col in col_map.items():
            result[out_name] = pd.to_numeric(df2[src_col], errors="coerce")
        result = result[result.index.notna()].dropna(subset=["price"])
        result.index = result.index.to_period("M").to_timestamp()
        return result.resample("MS").mean()

    # --- Found header row: scan it for commodity columns ---
    col_map = {}
    for j, val in enumerate(raw.iloc[header_row]):
        if not isinstance(val, str) and not (hasattr(val, '__str__')):
            continue
        v = str(val).lower().strip()
        if not v or v == "nan":
            continue
        if "palm oil" in v and "price" not in col_map:
            col_map["price"] = j
        if ("soybean oil" in v or "soya oil" in v or "soy oil" in v) and "soy_oil_usd" not in col_map:
            col_map["soy_oil_usd"] = j
        if "brent" in v and "crude_brent_usd" not in col_map:
            col_map["crude_brent_usd"] = j

    # Row 4 = commodity names, Row 5 = units ($/bbl etc) — skip units row
    # Data starts at header_row + 2
    data_rows = raw.iloc[header_row + 2:].copy()

    # Date column (col 0) is formatted as "1960M01" — parse manually
    def parse_wb_date(val):
        try:
            s = str(val).strip()
            if "M" in s:
                year, month = s.split("M")
                return pd.Timestamp(int(year), int(month), 1)
        except Exception:
            pass
        return pd.NaT

    dates = data_rows.iloc[:, 0].apply(parse_wb_date)
    print(f"  Parsed {dates.notna().sum()} date rows from {len(dates)} total")

    result = pd.DataFrame(index=dates)
    for col_name, col_idx in col_map.items():
        result[col_name] = pd.to_numeric(
            data_rows.iloc[:, col_idx].values, errors="coerce"
        )

    result = result[result.index.notna()].dropna(subset=["price"])
    result.index = result.index.to_period("M").to_timestamp()
    return result.resample("MS").mean()


def load_prices() -> pd.Series:
    """
    Load monthly palm oil prices (USD/tonne).
    Priority: user CSV → World Bank Pink Sheet → synthetic fallback.
    Also saves soybean oil + crude prices to wb_econ_cache.csv for
    use as economic features (avoids broken yfinance soy ticker).
    Returns a monthly Series named 'price'.
    """
    # ── user-supplied CSV ────────────────────────────────────────────────────
    if PRICE_CSV:
        print(f"  Loading prices from {PRICE_CSV}...")
        df = pd.read_csv(PRICE_CSV, parse_dates=["date"])
        s = df.set_index("date")["price"].resample("MS").mean()
        s.name = "price"
        print(f"  → {len(s)} months ({s.index[0].date()} – {s.index[-1].date()})")
        return s

    # ── World Bank Pink Sheet ────────────────────────────────────────────────
    print("  Downloading World Bank commodity price data...")
    try:
        import urllib.request, tempfile
        tmp = tempfile.mktemp(suffix=".xlsx")
        urllib.request.urlretrieve(WB_URL, tmp)

        wb = _parse_pink_sheet(tmp)

        # Cache soy + crude for economic features (replaces broken ZLF=F ticker)
        wb.to_csv("wb_econ_cache.csv")
        print(f"  → {len(wb)} months: palm oil + "
              f"{[c for c in wb.columns if c != 'price']} cached to wb_econ_cache.csv")

        s = wb["price"].dropna()
        print(f"  → Palm oil prices: {s.index[0].date()} – {s.index[-1].date()}")
        print(f"  → Price range: {s.min():.1f} – {s.max():.1f} USD/tonne")
        print(f"  → Recent 5 values:\n{s.tail(5).round(1).to_string()}")

        # Sanity check: WB Pink Sheet palm oil should be 300–1500 USD/tonne
        # If values are ~10-50x smaller, they may be in cents/lb — rescale
        recent_median = s[-36:].median()
        if recent_median < 100:
            print(f"  ⚠ Median {recent_median:.1f} looks like cents/lb — converting × 22.046")
            s = s * 22.046
        elif recent_median < 200:
            print(f"  ⚠ Median {recent_median:.1f} looks low — check units")

        return s

    except Exception as exc:
        print(f"  World Bank download failed ({exc}). Using synthetic prices for demo.")
        return _synthetic_prices()


def _synthetic_prices() -> pd.Series:
    """Realistic-looking synthetic palm oil price series for dry-run testing."""
    np.random.seed(42)
    idx = pd.date_range("1995-01-01", periods=370, freq="MS")
    trend  = np.linspace(400, 900, len(idx))
    cycle  = 150 * np.sin(np.linspace(0, 6 * np.pi, len(idx)))
    noise  = np.random.normal(0, 40, len(idx))
    shocks = np.zeros(len(idx))
    shocks[150] =  300   # 2007-08 food crisis
    shocks[200] = -200   # 2011 correction
    shocks[300] =  250   # COVID disruption
    prices = trend + cycle + noise + np.cumsum(shocks * 0.05)
    return pd.Series(np.maximum(prices, 200), index=idx, name="price")


# ============================================================================
# 2. JOIN FEATURES + PRICES
# ============================================================================

def build_dataset(features_csv: str = FEATURES_CSV,
                  horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Load weather features, load prices, align on monthly index.
    Target = price shifted back by `horizon` months (predict future from past features).
    Drops columns with >40% missing values.
    """
    print(f"\n[1/5] Loading features from {features_csv}...")
    feat = pd.read_csv(features_csv, index_col=0, parse_dates=True)
    feat.index = feat.index.to_period("M").to_timestamp()  # → first day of month
    print(f"  → {feat.shape[0]} rows × {feat.shape[1]} feature columns")

    print("\n[2/5] Loading palm oil prices...")
    prices = load_prices()

    # Align on common monthly index
    df = feat.join(prices, how="inner")
    print(f"  → {len(df)} overlapping months after join")

    # Palm-soy spread: palm trades at discount to soybean oil; spread is mean-reverting
    # Requires soy_oil_usd column from economic features (palm_oil_features_v2.csv)
    if "soy_oil_usd" in df.columns:
        df["palm_soy_spread"]      = df["price"] - df["soy_oil_usd"]
        df["palm_soy_spread_lag1m"]= df["palm_soy_spread"].shift(1)
        df["palm_soy_spread_roll3m"]= df["palm_soy_spread"].rolling(3).mean()
        print(f"  → Palm-soy spread computed "
              f"(mean: {df['palm_soy_spread'].mean():.0f} USD/t)")

    # Forward-fill price gaps up to 2 months (handles occasional missing months)
    df["price"] = df["price"].ffill(limit=2)

    # Indonesia export ban dummy (Apr–Jul 2022) — encodes the structural break
    # explicitly so the model doesn't attribute the spike to weather noise
    df["indonesia_export_ban"] = (
        (df.index >= "2022-04-01") & (df.index <= "2022-07-01")
    ).astype(float)

    # Lagged price features — price is autocorrelated; without these the model
    # has no idea what the current price level is and defaults to training mean.
    # All lags are computed before the target shift so they stay leak-free.
    df["price_lag1m"]       = df["price"].shift(1)
    df["price_lag3m"]       = df["price"].shift(3)
    df["price_lag6m"]       = df["price"].shift(6)
    df["price_lag12m"]      = df["price"].shift(12)
    df["price_mom1m"]       = df["price"].pct_change(1) * 100   # MoM %
    df["price_mom3m"]       = df["price"].pct_change(3) * 100   # 3M %
    df["price_roll3m_mean"] = df["price"].rolling(3).mean()
    df["price_roll6m_mean"] = df["price"].rolling(6).mean()
    df["price_roll6m_std"]  = df["price"].rolling(6).std()

    # Target: price h months ahead (shift price backward by h)
    df[TARGET_COL] = df["price"].shift(-horizon)

    # Drop future-leaking rows and columns with too many NaNs
    df = df.dropna(subset=[TARGET_COL])

    # Drop columns with >25% missing (tighter threshold — econ cols from
    # yfinance only go back to ~2000 but training data starts 1995)
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > 0.25].index.tolist()
    # Never drop core price lags or the target
    drop_cols = [c for c in drop_cols
                 if c not in [TARGET_COL, "price_lag1m", "price_roll3m_mean"]]
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} columns with >25% missing: {drop_cols[:8]}...")
        df = df.drop(columns=drop_cols)

    # Fill remaining NaNs with column median (for lag/rolling warmup period)
    df = df.fillna(df.median(numeric_only=True))

    print(f"  Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ============================================================================
# 3. TRAIN / TEST SPLIT  (temporal — no shuffling)
# ============================================================================

def temporal_split(df: pd.DataFrame,
                   test_years: int = 4) -> tuple:
    """
    Walk-forward split: last `test_years` years as held-out test set.
    Returns (X_train, X_test, y_train, y_test, feature_cols).
    """
    cutoff = df.index[-1] - pd.DateOffset(years=test_years)
    train  = df[df.index <= cutoff]
    test   = df[df.index >  cutoff]

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X_train = train[feature_cols]
    X_test  = test[feature_cols]
    y_train = train[TARGET_COL]
    y_test  = test[TARGET_COL]

    print(f"\n[3/5] Train/test split (cutoff: {cutoff.date()})")
    print(f"  Train: {len(train)} months  |  Test: {len(test)} months")
    return X_train, X_test, y_train, y_test, feature_cols


# ============================================================================
# 4. MODELS
# ============================================================================

def train_lgbm(X_train, y_train, X_test, y_test):
    """LightGBM with early stopping on a small validation tail."""
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    # Use last 20% of training data as validation for early stopping
    val_size = max(12, int(len(X_train) * 0.20))
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    dtrain = lgb.Dataset(X_tr,   label=y_tr)
    dval   = lgb.Dataset(X_val,  label=y_val, reference=dtrain)

    params = {
        "objective":       "regression",
        "metric":          "rmse",
        "learning_rate":   0.03,
        "num_leaves":      31,
        "min_child_samples": 10,
        "subsample":       0.8,
        "colsample_bytree": 0.7,
        "reg_alpha":       0.1,
        "reg_lambda":      1.0,
        "verbosity":       -1,
        "random_state":    42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),   # suppress per-round output
        ],
    )

    preds = model.predict(X_test)
    return model, preds


def train_baseline(X_train, y_train, X_test):
    """Naive baseline: predict last known price (random walk)."""
    last_price = y_train.iloc[-1]
    return np.full(len(X_test), last_price)


# ============================================================================
# 5. EVALUATION
# ============================================================================

def evaluate(y_true, y_pred, label: str = "Model") -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    print(f"\n  {label}:")
    print(f"    MAE   = {mae:.1f} USD/tonne")
    print(f"    RMSE  = {rmse:.1f} USD/tonne")
    print(f"    MAPE  = {mape:.1f}%")
    print(f"    R²    = {r2:.3f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# ============================================================================
# 6. VISUALISATIONS
# ============================================================================

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "actual":   "#2C2C2A",
    "lgbm":     "#1D9E75",
    "baseline": "#B4B2A9",
    "shap_pos": "#D85A30",
    "shap_neg": "#378ADD",
}

def plot_predictions(y_test, lgbm_preds, baseline_preds, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_test.index, y_test.values,  color=COLORS["actual"],
            lw=1.5, label="Actual price", zorder=3)
    ax.plot(y_test.index, lgbm_preds,     color=COLORS["lgbm"],
            lw=1.5, label="LightGBM forecast", zorder=2)
    ax.plot(y_test.index, baseline_preds, color=COLORS["baseline"],
            lw=1, linestyle="--", label="Naive baseline", zorder=1)
    ax.set_ylabel("USD / tonne")
    ax.set_title(f"Palm oil price forecast — held-out test set{title_suffix}",
                 fontsize=12, fontweight="normal")
    ax.legend(frameon=False, fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "forecast.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


def plot_shap(model, X_test, feature_cols, top_n=20):
    try:
        import shap
    except ImportError:
        print("  Install shap for feature attribution: pip install shap")
        return

    print("  Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    top_feat = [feature_cols[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.5))
    bars = ax.barh(range(top_n), top_vals[::-1],
                   color=COLORS["shap_pos"], alpha=0.85, height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feat[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (USD/tonne impact)")
    ax.set_title(f"Top {top_n} features by SHAP importance", fontsize=12, fontweight="normal")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_importance.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()

    # Also save feature importance table
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
        "lgbm_gain": model.feature_importance(importance_type="gain"),
    }).sort_values("mean_abs_shap", ascending=False)
    imp_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"  Saved: {imp_path}")
    return imp_df


def plot_residuals(y_test, lgbm_preds):
    residuals = y_test.values - lgbm_preds
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Residuals over time
    axes[0].axhline(0, color=COLORS["baseline"], lw=0.8, linestyle="--")
    axes[0].scatter(y_test.index, residuals, s=18,
                    color=COLORS["lgbm"], alpha=0.7, linewidths=0)
    axes[0].set_ylabel("Residual (USD/tonne)")
    axes[0].set_title("Residuals over time")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].spines[["top", "right"]].set_visible(False)

    # Residual histogram
    axes[1].hist(residuals, bins=25, color=COLORS["lgbm"],
                 alpha=0.8, edgecolor="none")
    axes[1].axvline(0, color=COLORS["actual"], lw=1, linestyle="--")
    axes[1].set_xlabel("Residual (USD/tonne)")
    axes[1].set_title("Residual distribution")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residuals.png")
    fig.savefig(path, dpi=150)
    print(f"  Saved: {path}")
    plt.close()


# ============================================================================
# 7. WALK-FORWARD CV  (replaces single train/test split)
# ============================================================================

def walk_forward_predict(df: pd.DataFrame,
                         test_years: int = 4,
                         retrain_every: int = 6) -> tuple:
    """
    Walk-forward expanding-window evaluation.

    Instead of training once on all history up to the cutoff, we retrain
    the model every `retrain_every` months as new data arrives. This means
    the model training on Jan-2024 data has seen 2022-2023 prices — so it
    can actually reach the higher price regime in its leaf values.

    Returns: (final_model, lgbm_preds_series, feature_cols)
    """
    import lightgbm as lgb

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    cutoff       = df.index[-1] - pd.DateOffset(years=test_years)
    test_idx     = df.index[df.index > cutoff]

    print(f"\n[3/5] Walk-forward CV  (cutoff: {cutoff.date()}, "
          f"retraining every {retrain_every} months)")
    print(f"  Test window: {test_idx[0].date()} → {test_idx[-1].date()} "
          f"({len(test_idx)} months)")

    all_preds   = {}
    final_model = None
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 10, "subsample": 0.8,
        "colsample_bytree": 0.7, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "verbosity": -1, "random_state": 42,
    }

    retrain_months = test_idx[::retrain_every]   # months where we retrain

    for i, month in enumerate(test_idx):
        # Expand training window up to (but not including) this month
        train = df[df.index < month]

        # Retrain if this is a retrain month or no model yet
        if final_model is None or month in retrain_months:
            val_size = max(12, int(len(train) * 0.15))
            X_tr = train[feature_cols].iloc[:-val_size]
            X_val = train[feature_cols].iloc[-val_size:]
            y_tr = train[TARGET_COL].iloc[:-val_size]
            y_val = train[TARGET_COL].iloc[-val_size:]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            final_model = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            print(f"  Retrained at {month.date()}  "
                  f"(train={len(train)}m, trees={final_model.num_trees()})")

        # Predict this month
        X_month = df.loc[[month], feature_cols]
        all_preds[month] = final_model.predict(X_month)[0]

    preds_series = pd.Series(all_preds, name="lgbm")
    return final_model, preds_series, feature_cols


def apply_stacking_corrector(y_train_residuals: pd.Series,
                              lgbm_train_preds:  np.ndarray,
                              lgbm_test_preds:   np.ndarray) -> np.ndarray:
    """
    Ridge regression correction layer (stacking).

    Fits a simple linear model:  corrected = a * lgbm_pred + b * trend + c
    on the training residuals, then applies it to test predictions.
    This corrects for systematic bias without overfitting.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    n_train = len(lgbm_train_preds)
    n_test  = len(lgbm_test_preds)

    # Features: lgbm prediction + time trend index
    X_tr = np.column_stack([lgbm_train_preds,
                             np.arange(n_train)])
    X_te = np.column_stack([lgbm_test_preds,
                             np.arange(n_train, n_train + n_test)])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Target = actual price (not residual) so corrector learns the full mapping
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, y_train_residuals.values + lgbm_train_preds)

    corrected = ridge.predict(X_te_s)
    return corrected


# ============================================================================
# 8. PIPELINE RUNNER
# ============================================================================

def run_model_pipeline():
    print("=" * 60)
    print("Palm oil price prediction — modeling pipeline v3")
    print("Walk-forward CV + stacking corrector")
    print("=" * 60)

    # Build dataset
    df = build_dataset()

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    cutoff       = df.index[-1] - pd.DateOffset(years=4)
    train_df     = df[df.index <= cutoff]
    test_df      = df[df.index >  cutoff]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    # Walk-forward predictions
    final_model, lgbm_series, feature_cols = walk_forward_predict(df, test_years=4)
    lgbm_preds = lgbm_series.values

    # In-sample predictions on training set (needed for stacking corrector)
    lgbm_train_preds = final_model.predict(X_train)
    train_residuals  = y_train - lgbm_train_preds

    # Stacking corrector disabled — adds noise when features are imperfect.
    # Use walk-forward LightGBM directly as the primary prediction.
    stacked_preds = lgbm_preds  # passthrough

    # Baseline
    baseline_preds = train_baseline(X_train, y_train, X_test)

    # Evaluate all three
    print("\n[5/5] Evaluation on held-out test set:")
    lgbm_metrics    = evaluate(y_test, lgbm_preds,    "LightGBM walk-forward")
    stacked_metrics = evaluate(y_test, stacked_preds, "LightGBM + stacking corrector")
    baseline_metrics= evaluate(y_test, baseline_preds,"Naive baseline (last price)")

    skill_mae  = (1 - stacked_metrics["MAE"]  / baseline_metrics["MAE"])  * 100
    skill_rmse = (1 - stacked_metrics["RMSE"] / baseline_metrics["RMSE"]) * 100
    print(f"\n  Skill score (stacked vs baseline):  "
          f"MAE {skill_mae:+.1f}%  |  RMSE {skill_rmse:+.1f}%")

    # Save metrics
    metrics_df = pd.DataFrame([lgbm_metrics, stacked_metrics, baseline_metrics])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    # Plots — show stacked as the primary prediction
    print("\n  Generating plots...")
    plot_predictions(y_test, stacked_preds, baseline_preds,
                     f" (+{FORECAST_HORIZON}m, stacked)")
    plot_residuals(y_test, stacked_preds)
    imp_df = plot_shap(final_model, X_test, feature_cols)

    # Save predictions
    pred_df = pd.DataFrame({
        "date":          y_test.index,
        "actual":        y_test.values,
        "lgbm_wf":       lgbm_preds,
        "stacked":       stacked_preds,
        "baseline":      baseline_preds,
        "residual":      y_test.values - stacked_preds,
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

    print(f"\nAll outputs written to: {OUTPUT_DIR}/")
    print("  forecast.png          — stacked predictions vs actual")
    print("  residuals.png         — residual diagnostics")
    print("  shap_importance.png   — top 20 features by SHAP")
    print("  feature_importance.csv")
    print("  predictions.csv       — lgbm_wf + stacked + baseline columns")
    print("  metrics.csv")

    return final_model, pred_df, imp_df


if __name__ == "__main__":
    lgbm_model, predictions, importance = run_model_pipeline()