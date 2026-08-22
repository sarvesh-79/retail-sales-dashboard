"""
pipeline.py — shared data cleaning and modeling functions.

Import this from BOTH the Jupyter notebook (for exploration) and app.py
(for the live dashboard), so cleaning/forecasting logic stays identical
in both places instead of drifting apart as you keep editing one or the
other.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def clean_data(df):
    """Auto-clean pipeline — mirrors the Phase 1 notebook cleanup steps."""
    report = {"rows_received": len(df)}

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Drop rows that are essentially blank (the ~30 junk rows found in Phase 1)
    key_cols = [c for c in ["order_id", "order_date", "customer_id"] if c in df.columns]
    if key_cols:
        df = df.dropna(subset=key_cols).copy()

    for col in ["age", "quantity", "customer_satisfaction", "days_to_ship"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "discount_pct" in df.columns:
        df["discount_pct"] = df["discount_pct"].fillna(0)  # missing = assumed no discount

    if "return_flag" in df.columns:
        if df["return_flag"].dtype == bool:
            df["return_flag"] = df["return_flag"].astype(int)
        elif df["return_flag"].dtype == object:
            df["return_flag"] = (
                df["return_flag"].astype(str).str.strip().str.title()
                .map({"Yes": 1, "No": 0, "True": 1, "False": 0})
            )

    before_dedup = len(df)
    df = df.drop_duplicates()

    report["rows_after_cleaning"] = len(df)
    report["duplicates_removed"] = before_dedup - len(df)
    return df, report


REQUIRED_COLUMNS = [
    "order_id", "order_date", "customer_id", "region", "city",
    "product_category", "payment_method", "sales_amount", "profit",
    "discount_pct", "quantity", "age", "days_to_ship",
]


def validate_columns(df):
    """Returns a list of required columns missing from df — empty list means all present."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def get_monthly_revenue(df):
    """Aggregate sales_amount into a monthly time series."""
    monthly = df.groupby(df["order_date"].dt.to_period("M"))["sales_amount"].sum()
    monthly.index = monthly.index.to_timestamp()
    return monthly


def forecast_revenue(monthly_revenue, horizon=6, seasonal_periods=12):
    """Fit the validated multiplicative-seasonal model and forecast forward.

    This is the exact model validated in Phase 2 (MAE ~273K vs the
    additive/trend-only alternatives that performed worse).
    """
    model = ExponentialSmoothing(
        monthly_revenue, trend="add", seasonal="mul", seasonal_periods=seasonal_periods
    )
    fit = model.fit()
    return fit.forecast(horizon)


def train_return_models(df):
    """Reproduces the Phase 2 return-prediction experiment.

    Kept here for documentation/reproducibility — NOT called by app.py,
    since both models scored near random chance (ROC-AUC ~0.48-0.56) and
    aren't reliable enough to use as a live prediction feature.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    features = ["product_category", "region", "unit_price", "discount_pct",
                "shipping_cost", "days_to_ship", "payment_method", "age", "gender", "quantity"]

    X = pd.get_dummies(df[features],
                        columns=["product_category", "region", "payment_method", "gender"],
                        drop_first=True)
    y = df["return_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    log_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])

    return {"random_forest_auc": rf_auc, "logistic_regression_auc": log_auc}