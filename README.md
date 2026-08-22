# Retail Sales Dashboard

Interactive Streamlit dashboard for retail sales — auto-cleans newly uploaded
data, visualizes key business metrics, and forecasts future revenue.

## Live Demo
## Live Demo

🚀 **[Open the Retail Sales Analytics Dashboard](https://retail-salesdashboard.streamlit.app/)**

## Features
- **Auto-cleaning pipeline** for newly uploaded sales data — handles missing
  values, duplicate rows, and inconsistent formats automatically
- **Sales & activity breakdown** by region, city, product category, and
  payment method
- **Discount vs. profit margin analysis** — shows where discounting starts
  eating into margin
- **Revenue forecasting** — multiplicative seasonal Exponential Smoothing,
  ~9-11% average forecast error against actual monthly revenue
- **Documented return-prediction experiment** — Random Forest and Logistic
  Regression were both tested to predict order returns; neither found a
  reliable signal (ROC-AUC ~0.48-0.56). Reported here as an honest finding
  rather than forced into a feature that wouldn't hold up in practice.

## Tech Stack
Python · Pandas · Scikit-learn · Statsmodels · Streamlit · Matplotlib

## Project Structure
```
├── app.py                   # Streamlit dashboard
├── pipeline.py               # shared cleaning + forecasting functions
├── requirements.txt
├── retail_sales_dataset.csv
├── RetailSalesData.ipynb    # full analysis, including the return-prediction experiment
└── README.md
```

## Run Locally
```bash
git clone https://github.com/sarvesh-79/retail-sales-dashboard.git
cd retail-sales-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Key Finding
Return prediction was tested but found no reliable signal in the available
order-level features — a legitimate result suggesting return behavior
depends on factors (product quality, fit, customer history) not captured
in this dataset. Revenue forecasting was chosen as the primary predictive
feature instead, validated with a proper train/test split.
