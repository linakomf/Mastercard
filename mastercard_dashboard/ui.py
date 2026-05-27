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
            display: none !important;
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
            padding-top: 1.25rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.sidebar-brand) {
            position: relative;
            z-index: 2;
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
            color: #ff9b9b;
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
            background: rgba(255, 107, 107, 0.12);
            border-color: rgba(255, 107, 107, 0.28);
            color: #ffd6d6;
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
            border-color: rgba(255, 107, 107, 0.28);
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
            color: #ff9b9b;
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
        .card-page-hero {
            background: linear-gradient(135deg, rgba(20, 42, 76, 0.95), rgba(13, 24, 43, 0.98));
            border: 1px solid rgba(94, 142, 255, 0.22);
            border-radius: 20px;
            padding: 22px 24px;
            margin-bottom: 1rem;
        }
        .card-page-title {
            color: #f5f9ff;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0 0 6px 0;
        }
        .card-page-subtitle {
            color: #b2c4e3;
            font-size: 0.96rem;
            line-height: 1.5;
            margin: 0;
        }
        .card-score-panel {
            background: linear-gradient(160deg, rgba(18, 36, 68, 0.96), rgba(12, 22, 40, 0.98));
            border: 1px solid rgba(94, 142, 255, 0.24);
            border-radius: 20px;
            padding: 20px 22px;
            margin-bottom: 0.5rem;
        }
        .card-score-panel.pattern {
            border-color: rgba(245, 166, 35, 0.35);
            box-shadow: 0 0 0 1px rgba(245, 166, 35, 0.08) inset;
        }
        .card-score-number {
            color: #f5f9ff;
            font-size: 2.35rem;
            font-weight: 750;
            line-height: 1;
            margin: 4px 0 10px 0;
        }
        .card-score-caption {
            color: #9cb0d8;
            font-size: 0.84rem;
            margin-bottom: 4px;
        }
        .card-id-line {
            color: #dfe9ff;
            font-size: 1.02rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-bottom: 2px;
        }
        .card-status-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .card-status-pill.pattern {
            background: rgba(245, 166, 35, 0.14);
            border: 1px solid rgba(245, 166, 35, 0.32);
            color: #ffe8bd;
        }
        .card-status-pill.regular {
            background: rgba(53, 194, 255, 0.12);
            border: 1px solid rgba(53, 194, 255, 0.24);
            color: #dff5ff;
        }
        .card-mini-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }
        .card-mini-metric {
            background: rgba(8, 14, 26, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 10px 12px;
        }
        .card-mini-metric-label {
            color: #91a4c7;
            font-size: 0.76rem;
            margin-bottom: 4px;
        }
        .card-mini-metric-value {
            color: #f5f9ff;
            font-size: 1rem;
            font-weight: 650;
        }
        .section-heading {
            color: #f5f9ff;
            font-size: 1.05rem;
            font-weight: 650;
            margin: 0 0 0.35rem 0;
        }
        .section-hint {
            color: #91a4c7;
            font-size: 0.86rem;
            margin: 0 0 0.75rem 0;
        }
        .card-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 0.75rem 0 1.1rem 0;
        }
        .card-kpi-card {
            background: rgba(16, 26, 44, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 14px 16px;
        }
        .card-kpi-label {
            color: #91a4c7;
            font-size: 0.8rem;
            margin-bottom: 8px;
        }
        .card-kpi-value {
            color: #f5f9ff;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .card-kpi-delta {
            color: #ff9b9b;
            font-size: 0.78rem;
        }
        .panel-card {
            background: rgba(14, 24, 42, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 16px 18px;
        }
        .merchant-list {
            display: flex;
            flex-direction: column;
            gap: 0;
        }
        .merchant-row {
            display: grid;
            grid-template-columns: 44px 1fr auto;
            gap: 14px;
            align-items: center;
            padding: 14px 4px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }
        .merchant-row:last-child {
            border-bottom: none;
        }
        .merchant-icon {
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.78rem;
            font-weight: 700;
            color: #f5f9ff;
        }
        .merchant-icon.tone-0 { background: #2f80ed; }
        .merchant-icon.tone-1 { background: #e74c3c; }
        .merchant-icon.tone-2 { background: #f5a623; }
        .merchant-icon.tone-3 { background: #eb5757; }
        .merchant-icon.tone-4 { background: #4f5d75; }
        .merchant-name {
            color: #f5f9ff;
            font-size: 0.95rem;
            font-weight: 650;
            margin-bottom: 4px;
        }
        .merchant-meta {
            color: #91a4c7;
            font-size: 0.8rem;
        }
        .merchant-amount {
            color: #f5f9ff;
            font-size: 0.98rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .geo-list {
            display: flex;
            flex-direction: column;
            gap: 14px;
            padding-top: 4px;
        }
        .geo-row {
            display: grid;
            grid-template-columns: 88px 1fr auto;
            gap: 12px;
            align-items: center;
        }
        .geo-country {
            color: #dfe9ff;
            font-size: 0.9rem;
            font-weight: 600;
        }
        .geo-bar-wrap {
            height: 10px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 999px;
            overflow: hidden;
        }
        .geo-bar {
            height: 100%;
            border-radius: 999px;
        }
        .geo-bar.tone-0 { background: #2f80ed; }
        .geo-bar.tone-1 { background: #35c2ff; }
        .geo-bar.tone-2 { background: #f5a623; }
        .geo-amount {
            color: #f5f9ff;
            font-size: 0.88rem;
            font-weight: 650;
            min-width: 72px;
            text-align: right;
        }
        .chart-card-title {
            color: #f5f9ff;
            font-size: 0.98rem;
            font-weight: 650;
            margin: 0 0 2px 0;
        }
        .chart-card-subtitle {
            color: #91a4c7;
            font-size: 0.8rem;
            margin: 0 0 10px 0;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            font-size: 0.95rem !important;
        }
        div[data-testid="stRadio"] {
            margin-bottom: 0.75rem;
        }
        .sidebar-brand {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
            padding: 8px 4px 22px 4px;
            margin-bottom: 0.25rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        .mastercard-mark {
            display: block;
            width: 52px;
            height: 32px;
            overflow: visible;
        }
        .sidebar-brand-title {
            color: #f5f9ff;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            line-height: 1;
            margin: 0;
        }
        [data-testid="stSidebar"] .stRadio > label {
            color: #91a4c7 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        (
            '<div class="sidebar-brand">'
            '<svg class="mastercard-mark" viewBox="0 0 52 32" xmlns="http://www.w3.org/2000/svg" aria-label="Mastercard">'
            '<circle cx="19" cy="16" r="13" fill="#EB001B"/>'
            '<circle cx="33" cy="16" r="13" fill="#F79E1B"/>'
            "</svg>"
            '<div class="sidebar-brand-title">Mastercard</div>'
            "</div>"
        ),
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


def render_card_page_hero(title: str, subtitle: str) -> None:
    st.markdown(
        (
            f'<div class="card-page-hero">'
            f'<div class="card-page-title">{escape(title)}</div>'
            f'<p class="card-page-subtitle">{escape(subtitle)}</p>'
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_card_score_panel(
    card_masked: str,
    score_pct: float,
    is_pattern: bool,
    predicted_label: str,
    confidence_pct: float,
    confidence_text: str,
) -> None:
    status_class = "pattern" if is_pattern else "regular"
    status_text = "Hidden entrepreneur pattern" if is_pattern else "Regular consumer profile"
    st.markdown(
        (
            f'<div class="card-score-panel {status_class}">'
            f'<span class="card-status-pill {status_class}">{escape(status_text)}</span>'
            f'<div class="card-id-line">{escape(card_masked)}</div>'
            f'<div class="card-score-caption">Hidden entrepreneur probability</div>'
            f'<div class="card-score-number">{score_pct:.1f}%</div>'
            f'<div class="card-mini-metrics">'
            f'<div class="card-mini-metric">'
            f'<div class="card-mini-metric-label">Model class</div>'
            f'<div class="card-mini-metric-value">{escape(predicted_label)}</div>'
            f"</div>"
            f'<div class="card-mini-metric">'
            f'<div class="card-mini-metric-label">Confidence ({escape(confidence_text)})</div>'
            f'<div class="card-mini-metric-value">{confidence_pct:.1f}%</div>'
            f"</div>"
            f"</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, hint: str = "") -> None:
    hint_html = f'<p class="section-hint">{escape(hint)}</p>' if hint else ""
    st.markdown(
        f'<h3 class="section-heading">{escape(title)}</h3>{hint_html}',
        unsafe_allow_html=True,
    )


def render_card_kpi_strip(items: list[tuple[str, str, str]]) -> None:
    cards_html = []
    for label, value, delta in items:
        cards_html.append(
            f'<div class="card-kpi-card">'
            f'<div class="card-kpi-label">{escape(label)}</div>'
            f'<div class="card-kpi-value">{escape(value)}</div>'
            f'<div class="card-kpi-delta">{escape(delta)}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="card-kpi-grid">{"".join(cards_html)}</div>',
        unsafe_allow_html=True,
    )


def render_merchant_purchase_list(rows: list[dict[str, str]]) -> None:
    if not rows:
        st.info("No transactions for this card.")
        return

    items_html = []
    for index, row in enumerate(rows):
        tone = index % 5
        items_html.append(
            f'<div class="merchant-row">'
            f'<div class="merchant-icon tone-{tone}">{escape(row["initials"])}</div>'
            f'<div>'
            f'<div class="merchant-name">{escape(row["name"])}</div>'
            f'<div class="merchant-meta">{escape(row["meta"])}</div>'
            f"</div>"
            f'<div class="merchant-amount">{escape(row["amount"])}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="panel-card"><div class="merchant-list">{"".join(items_html)}</div></div>',
        unsafe_allow_html=True,
    )


def render_geo_spend_list(rows: list[dict[str, str | float]]) -> None:
    if not rows:
        st.info("No country data available.")
        return

    max_amount = max(float(row["amount_value"]) for row in rows) or 1.0
    items_html = []
    for index, row in enumerate(rows):
        width_pct = max(float(row["amount_value"]) / max_amount * 100, 4.0)
        tone = index % 3
        items_html.append(
            f'<div class="geo-row">'
            f'<div class="geo-country">{escape(str(row["country"]))}</div>'
            f'<div class="geo-bar-wrap"><div class="geo-bar tone-{tone}" style="width:{width_pct:.1f}%"></div></div>'
            f'<div class="geo-amount">{escape(str(row["amount_label"]))}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="panel-card"><div class="geo-list">{"".join(items_html)}</div></div>',
        unsafe_allow_html=True,
    )


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
