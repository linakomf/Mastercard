from __future__ import annotations

from pathlib import Path

import streamlit as st

from mastercard_dashboard.config import DashboardPaths, discover_paths
from mastercard_dashboard.data import load_dashboard_data
from mastercard_dashboard.modeling import train_model_bundle
from mastercard_dashboard.pages import (
    render_card_profile_page,
    render_shap_page,
    render_transactions_page,
)
from mastercard_dashboard.ui import (
    apply_fintech_theme,
    render_data_sources,
)


st.set_page_config(
    page_title="Mastercard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def build_paths(
    business_path: str,
    consumer_path: str,
    merchant_path: str,
    artifact_dir: str,
) -> DashboardPaths:
    artifact_root = Path(artifact_dir)
    return DashboardPaths(
        business_path=Path(business_path),
        consumer_path=Path(consumer_path),
        merchant_path=Path(merchant_path),
        artifact_dir=artifact_root,
        card_features_path=artifact_root / "card_level_features.parquet",
        feature_summary_path=artifact_root / "feature_summary_by_segment.csv",
        metrics_path=artifact_root / "model_comparison.csv",
        eda_summary_path=artifact_root / "eda_summary.json",
        confusion_matrix_path=artifact_root / "xgboost_confusion_matrix.png",
        feature_importance_path=artifact_root / "xgboost_feature_importance.png",
        shap_summary_path=artifact_root / "xgboost_shap_summary.png",
        local_explanations_path=artifact_root / "xgboost_local_explanations.csv",
        business_report_path=artifact_root / "business_report.md",
    )


def main() -> None:
    apply_fintech_theme()
    defaults = discover_paths()
    business_path = str(defaults.business_path)
    consumer_path = str(defaults.consumer_path)
    merchant_path = str(defaults.merchant_path)
    artifact_dir = str(defaults.artifact_dir)

    with st.sidebar:
        st.title("Mastercard")
        st.caption("Hidden Entrepreneur Detection")

        page_name = st.radio(
            "Navigation",
            options=[
                "Card",
                "Transactions",
                "SHAP",
            ],
            index=0,
        )

        if st.button("Reload dashboard data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    paths = build_paths(
        business_path=business_path,
        consumer_path=consumer_path,
        merchant_path=merchant_path,
        artifact_dir=artifact_dir,
    )

    missing_inputs = [
        str(path)
        for path in [paths.business_path, paths.consumer_path, paths.merchant_path]
        if not path.exists()
    ]
    if missing_inputs:
        st.error("Required input files were not found:")
        for missing_path in missing_inputs:
            st.markdown(f"- `{missing_path}`")
        render_data_sources(paths)
        st.stop()

    dashboard_data = load_dashboard_data(paths)
    model_bundle = train_model_bundle(dashboard_data["card_features"])

    if page_name == "Card":
        render_card_profile_page(dashboard_data, model_bundle, paths)
    elif page_name == "Transactions":
        render_transactions_page(dashboard_data, paths)
    else:
        render_shap_page(dashboard_data, model_bundle, paths)

    render_data_sources(paths)


if __name__ == "__main__":
    main()
