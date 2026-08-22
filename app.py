import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from pipeline import clean_data, get_monthly_revenue, forecast_revenue, validate_columns

st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_default_data():
    csv_path = os.path.join(BASE_DIR, "retail_sales_dataset.csv")  # looks next to app.py itself, not the terminal's folder
    df = pd.read_csv(csv_path)
    return clean_data(df)


# ---------------------------------------------------------
# Sidebar — choose data source
# ---------------------------------------------------------
st.sidebar.title("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload new sales data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df, report = clean_data(raw_df)

    missing_cols = validate_columns(df)
    if missing_cols:
        st.sidebar.error(
            f"Uploaded file is missing required columns: {', '.join(missing_cols)}. "
            "Showing the default dataset instead."
        )
        df, report = load_default_data()
    else:
        st.sidebar.success(f"{report['rows_received']} rows → {report['rows_after_cleaning']} after cleaning")
        st.sidebar.write(f"Duplicates removed: {report['duplicates_removed']}")
else:
    df, report = load_default_data()
    st.sidebar.info("Using default dataset")

# ---------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------
st.title("📊 Retail Sales Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"₹{df['sales_amount'].sum():,.0f}")   # swap ₹ if your data isn't INR
col2.metric("Total Orders", f"{df['order_id'].nunique():,}")
col3.metric("Avg Order Value", f"₹{df['sales_amount'].mean():,.0f}")
col4.metric("Total Profit", f"₹{df['profit'].sum():,.0f}")

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Sales & Activity", "Discount & Profit", "Revenue Forecast", "About the Return Model"]
)

with tab1:
    st.subheader("Sales by Region")
    st.bar_chart(df.groupby("region")["sales_amount"].sum().sort_values())

    st.subheader("Sales by Product Category")
    st.bar_chart(df.groupby("product_category")["sales_amount"].sum().sort_values())

    st.subheader("Payment Method Usage")
    st.bar_chart(df["payment_method"].value_counts())

    st.subheader("Most Active Cities (Top 10 by Orders)")
    st.bar_chart(df.groupby("city")["order_id"].count().sort_values(ascending=False).head(10))

with tab2:
    df["profit_margin_pct"] = (df["profit"] / df["sales_amount"]) * 100
    bins = [0, 5, 10, 15, 20, 30, 50, 100]   # adjust if discount_pct is stored as 0-1 instead of 0-100
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
    df["discount_range"] = pd.cut(df["discount_pct"], bins=bins, labels=labels, include_lowest=True)

    st.subheader("Avg Profit Margin by Discount Range")
    st.bar_chart(df.groupby("discount_range", observed=True)["profit_margin_pct"].mean())

    st.subheader("Profit by Product Category")
    st.bar_chart(df.groupby("product_category")["profit"].sum().sort_values())

with tab3:
    st.subheader("Monthly Revenue Forecast")
    monthly_revenue = get_monthly_revenue(df)
    horizon = st.slider("Months to forecast", 1, 12, 6)

    if len(monthly_revenue) >= 24:
        future_forecast = forecast_revenue(monthly_revenue, horizon=horizon)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(monthly_revenue.index, monthly_revenue.values, label="Historical")
        ax.plot(future_forecast.index, future_forecast.values, label="Forecast", linestyle="--", marker="o")
        ax.legend()
        ax.set_title("Revenue Forecast")
        st.pyplot(fig)
        st.write(future_forecast)
    else:
        st.warning("Need at least 24 months of data for a reliable seasonal forecast.")

with tab4:
    st.subheader("Return Prediction — Tested, No Reliable Signal Found")
    st.write(
        "Random Forest and Logistic Regression were both tested to predict order returns "
        "using order-level features (price, discount, shipping cost, shipping time, demographics). "
        "Both scored close to random chance (ROC-AUC ~0.48-0.56), indicating these features don't "
        "meaningfully predict return behavior in this dataset — likely because factors like product "
        "quality or fit aren't captured here. Reported as an honest finding rather than a forced feature. "
        "See pipeline.py's train_return_models() for the reproducible experiment."
    )
