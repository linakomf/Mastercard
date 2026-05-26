from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from mastercard_dashboard.config import DashboardPaths


def apply_fintech_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(0, 163, 255, 0.08), transparent 28%),
                linear-gradient(180deg, #08111f 0%, #0e1728 100%);
            font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        }
        .stApp button,
        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp label,
        .stApp p,
        .stApp li,
        .stApp a,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            font-family: Inter, "Segoe UI", Roboto, Arial, sans-serif;
        }
        .material-icons,
        .material-icons-round,
        .material-icons-outlined,
        .material-symbols-rounded,
        .material-symbols-outlined,
        [class*="material-symbol"],
        [class*="material-icons"] {
            display: none !important;
            visibility: hidden !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a1324 0%, #111c31 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            min-width: 300px !important;
            max-width: 300px !important;
            position: relative !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            min-width: 300px !important;
            max-width: 300px !important;
        }
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 52px;
            background: linear-gradient(180deg, #0a1324 0%, #111c31 100%);
            z-index: 1000;
            pointer-events: none;
        }
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarHeader"] *,
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stBaseButton-headerNoPadding"],
        [data-testid="stBaseButton-headerNoPadding"] *,
        [data-testid="stIconMaterial"] {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            min-width: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stSidebarUserContent"],
        [data-testid="stSidebarContent"] {
            padding-top: 0.75rem !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 300px !important;
            max-width: 300px !important;
            transform: translateX(0) !important;
            margin-left: 0 !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            min-width: 300px !important;
            max-width: 300px !important;
            transform: translateX(0) !important;
            margin-left: 0 !important;
        }
        header,
        [role="banner"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stAppDeployButton"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
        }
        #MainMenu,
        footer {
            visibility: hidden;
        }
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] *,
        [data-testid="stSidebarNavCollapseButton"],
        [data-testid="stSidebarNavCollapseButton"] *,
        [data-testid="baseButton-headerNoPadding"],
        [data-testid="baseButton-headerNoPadding"] *,
        [aria-label*="sidebar"],
        [aria-label*="Sidebar"],
        [title*="sidebar"],
        [title*="Sidebar"],
        button[kind="header"],
        button[kind="header"] *,
        button:has([class*="material-symbol"]),
        button:has([class*="material-icons"]) {
            display: none !important;
            visibility: hidden !important;
            width: 0 !important;
            height: 0 !important;
            min-width: 0 !important;
            min-height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
            overflow: hidden !important;
        }
        div[data-testid="metric-container"] {
            background: rgba(16, 26, 44, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 12px 14px;
            border-radius: 16px;
        }
        div[data-testid="metric-container"] label {
            color: #9cb0d8 !important;
            font-size: 0.84rem !important;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            line-height: 1.05 !important;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }
        .dashboard-card {
            background: rgba(16, 26, 44, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 18px 20px;
        }
        .compact-kpi-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 8px;
        }
        .compact-kpi-card {
            background: rgba(16, 26, 44, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 12px 14px;
        }
        .compact-kpi-label {
            color: #9cb0d8;
            font-size: 0.8rem;
            margin-bottom: 8px;
        }
        .compact-kpi-value {
            color: #f5f9ff;
            font-size: 1.25rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 8px;
            word-break: break-word;
        }
        .compact-kpi-delta {
            color: #7ee7b5;
            font-size: 0.78rem;
            line-height: 1.3;
        }
        .dashboard-caption {
            color: #91a4c7;
            font-size: 0.9rem;
            margin-top: -4px;
        }
        .hero-panel {
            background: linear-gradient(135deg, rgba(20, 42, 76, 0.95), rgba(13, 24, 43, 0.98));
            border: 1px solid rgba(94, 142, 255, 0.22);
            border-radius: 20px;
            padding: 22px 24px;
            margin-bottom: 0.75rem;
        }
        .hero-kicker {
            color: #7ab8ff;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
        }
        .hero-title {
            color: #f5f9ff;
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            color: #b2c4e3;
            font-size: 0.98rem;
            line-height: 1.55;
            margin-bottom: 16px;
        }
        .signal-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .signal-chip {
            background: rgba(53, 194, 255, 0.12);
            border: 1px solid rgba(53, 194, 255, 0.22);
            color: #dff5ff;
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.86rem;
            line-height: 1.2;
        }
        .signal-chip.warning {
            background: rgba(245, 166, 35, 0.12);
            border-color: rgba(245, 166, 35, 0.28);
            color: #ffe8bd;
        }
        .signal-chip.success {
            background: rgba(46, 204, 113, 0.12);
            border-color: rgba(46, 204, 113, 0.24);
            color: #d8ffe7;
        }
        .section-card {
            background: rgba(14, 24, 42, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
        }
        .reason-card {
            background: rgba(14, 24, 42, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 132px;
        }
        .reason-card.success {
            border-color: rgba(46, 204, 113, 0.22);
        }
        .reason-card.warning {
            border-color: rgba(245, 166, 35, 0.22);
        }
        .reason-card.info {
            border-color: rgba(53, 194, 255, 0.22);
        }
        .reason-title {
            color: #f5f9ff;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .reason-metric {
            color: #7ee7b5;
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .reason-card.warning .reason-metric {
            color: #ffd48a;
        }
        .reason-card.info .reason-metric {
            color: #7ad7ff;
        }
        .reason-text {
            color: #b2c4e3;
            font-size: 0.93rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_data_sources(paths: DashboardPaths) -> None:
    loaded_data = pd.DataFrame(
        [
            {
                "Dataset": "Business",
                "File": paths.business_path.name,
                "Status": "Loaded" if paths.business_path.exists() else "Missing",
            },
            {
                "Dataset": "Consumer",
                "File": paths.consumer_path.name,
                "Status": "Loaded" if paths.consumer_path.exists() else "Missing",
            },
            {
                "Dataset": "Merchants",
                "File": paths.merchant_path.name,
                "Status": "Loaded" if paths.merchant_path.exists() else "Missing",
            },
        ]
    )
    with st.sidebar:
        show_loaded_data = st.toggle("Loaded data", value=False, key="show_loaded_data")
        if show_loaded_data:
            st.dataframe(loaded_data, use_container_width=True, hide_index=True)


def render_artifact_links(paths: DashboardPaths) -> None:
    with st.sidebar.expander("Generated Artifacts", expanded=False):
        for label, artifact_path in [
            ("Business report", paths.business_report_path),
            ("Card-level features", paths.card_features_path),
            ("Model comparison", paths.metrics_path),
            ("Confusion matrix", paths.confusion_matrix_path),
            ("Feature importance", paths.feature_importance_path),
            ("SHAP summary", paths.shap_summary_path),
        ]:
            status = "available" if Path(artifact_path).exists() else "missing"
            st.markdown(f"- {label}: `{artifact_path}` ({status})")


def render_kpi_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta=delta)


def render_compact_kpi_row(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items), gap="medium")
    for column, (label, value, delta) in zip(columns, items):
        column.markdown(
            (
                f'<div class="compact-kpi-card">'
                f'<div class="compact-kpi-label">{escape(label)}</div>'
                f'<div class="compact-kpi-value">{escape(value)}</div>'
                f'<div class="compact-kpi-delta">{escape(delta)}</div>'
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
