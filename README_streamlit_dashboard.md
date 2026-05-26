# Mastercard Hidden Entrepreneur Streamlit Dashboard

Interactive fintech-style dashboard for the Mastercard ML case:
`consumer` vs `hidden entrepreneur`.

## What is included

- `app.py` - Streamlit entrypoint
- `mastercard_dashboard/` - modular code for data loading, modeling, pages, and UI
- `.streamlit/config.toml` - theme and Streamlit settings
- `requirements.txt` - Python dependencies

## Supported data sources

The dashboard auto-detects both `parquet` and `csv` from your Downloads folder:

- `business_cards_MDQ.parquet` or `businass_csv.csv`
- `consumer_cards_MDQ.parquet` or `cumsumer_csv.csv`
- `merchants_reference.parquet` or `merchants_ref.csv`

It also reads pre-generated artifacts from:

- `mastercard_hidden_entrepreneur_artifacts/`

## Run

```bash
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

If you want a short local wrapper from the project root, use:

```bash
.\streamlit.cmd run app.py
```

## Pages

1. `Overview`
   KPI cards, segment comparison, score distribution, business signals

2. `Card Profile`
   Card-level probability, transactions, merchants, MCCs, hourly activity, local explanation

3. `Transactions`
   Search + filters + interactive table

4. `Model Insights`
   Model comparison, ROC curves, confusion matrix, feature importance

5. `SHAP Explainability`
   SHAP summary image and local card explanation

## Notes

- The app prefers precomputed `card_level_features.parquet` if it exists.
- If card-level features are missing, it can rebuild them from raw transactions.
- For large transaction tables, `parquet` mode is much faster than `csv`.
