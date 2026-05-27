from __future__ import annotations

from pathlib import Path

import streamlit as st

from mastercard_dashboard.config import DashboardPaths, discover_paths
from mastercard_dashboard.data import load_dashboard_data
from mastercard_dashboard.modeling import train_model_bundle
from mastercard_dashboard.pages import (
    render_card_profile_page,
    render_model_insights_page,
    render_shap_page,
    render_transactions_page,
)
from mastercard_dashboard.ui import (
    apply_fintech_theme,
    render_data_sources,
    render_mastercard_logo_html,
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
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;padding:20px 8px 16px;">
                {render_mastercard_logo_html(28)}
                <div>
                    <div style="font-weight:700;font-size:16px;color:#e8e8f0;letter-spacing:0.3px;">Mastercard</div>
                    <div style="font-size:10px;color:#FF5F00;font-weight:600;letter-spacing:2px;text-transform:uppercase;">Data Quest 2026</div>
                </div>
            </div>
            <div style="height:1px;background:rgba(255,255,255,0.04);margin:0 8px 12px;"></div>
            <div style="font-size:10px;color:#555;font-weight:600;letter-spacing:2px;padding:4px 8px 8px;text-transform:uppercase;">Navigation</div>
            """,
            unsafe_allow_html=True,
        )

        page_name = st.radio(
            "Navigation",
            options=[
                "💳  Card Profile",
                "📊  Transactions",
                "🧠  SHAP",
                "⚙️  Model",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="height:1px;background:rgba(255,255,255,0.04);margin:12px 8px 12px;"></div>
            <div style="display:flex;align-items:center;gap:10px;padding:10px 8px;">
                <div style="width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#EB001B,#F79E1B);
                    display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:700;color:white;flex-shrink:0;">A</div>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#e8e8f0;">Analyst</div>
                    <div style="font-size:10px;color:#555;">PA Team · Almaty</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Reload data", use_container_width=True):
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

    if page_name == "💳  Card Profile":
        render_card_profile_page(dashboard_data, model_bundle, paths)
    elif page_name == "📊  Transactions":
        render_transactions_page(dashboard_data, paths)
    elif page_name == "🧠  SHAP":
        render_shap_page(dashboard_data, model_bundle, paths)
    else:
        render_model_insights_page(dashboard_data, model_bundle, paths)

    render_data_sources(paths)


if __name__ == "__main__":
    main()
