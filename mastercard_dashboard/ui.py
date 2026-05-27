from __future__ import annotations

import math
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from mastercard_dashboard.config import DashboardPaths


def apply_fintech_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700&family=Space+Mono:wght@400;700&display=swap');
        @keyframes fadeUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(235,0,27,0.4); } 50% { box-shadow: 0 0 0 8px rgba(235,0,27,0); } }

        .stApp {
            background: #0a0a14;
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }
        .stApp button, .stApp input, .stApp textarea, .stApp select,
        .stApp label, .stApp p, .stApp li, .stApp a,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }

        .material-icons, .material-icons-round, .material-icons-outlined,
        .material-symbols-rounded, .material-symbols-outlined,
        [class*="material-symbol"], [class*="material-icons"] {
            display: none !important; visibility: hidden !important;
        }
        [data-testid="stSidebar"] {
            background: #0d0d1a;
            border-right: 1px solid rgba(255,255,255,0.04);
            min-width: 270px !important;
            max-width: 270px !important;
            position: relative !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            min-width: 270px !important;
            max-width: 270px !important;
        }
        [data-testid="stSidebar"]::before {
            content: "";
            position: absolute; top: 0; left: 0; right: 0; height: 52px;
            background: #0d0d1a;
            z-index: 1000; pointer-events: none;
        }
        [data-testid="stSidebarHeader"], [data-testid="stSidebarHeader"] *,
        [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stBaseButton-headerNoPadding"], [data-testid="stBaseButton-headerNoPadding"] *,
        [data-testid="stIconMaterial"] {
            display: none !important; visibility: hidden !important;
            width: 0 !important; height: 0 !important;
            min-width: 0 !important; min-height: 0 !important;
            padding: 0 !important; margin: 0 !important; overflow: hidden !important;
        }
        [data-testid="stSidebarUserContent"], [data-testid="stSidebarContent"] {
            padding-top: 0.75rem !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            min-width: 270px !important; max-width: 270px !important;
            transform: translateX(0) !important; margin-left: 0 !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            min-width: 270px !important; max-width: 270px !important;
            transform: translateX(0) !important; margin-left: 0 !important;
        }
        header, [role="banner"], [data-testid="stHeader"],
        [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] {
            display: none !important; visibility: hidden !important;
            height: 0 !important; min-height: 0 !important;
        }
        #MainMenu, footer { visibility: hidden; }
        [data-testid="collapsedControl"], [data-testid="collapsedControl"] *,
        [data-testid="stSidebarNavCollapseButton"], [data-testid="stSidebarNavCollapseButton"] *,
        [data-testid="baseButton-headerNoPadding"], [data-testid="baseButton-headerNoPadding"] *,
        [aria-label*="sidebar"], [aria-label*="Sidebar"],
        [title*="sidebar"], [title*="Sidebar"],
        button[kind="header"], button[kind="header"] *,
        button:has([class*="material-symbol"]), button:has([class*="material-icons"]) {
            display: none !important; visibility: hidden !important;
            width: 0 !important; height: 0 !important;
            min-width: 0 !important; min-height: 0 !important;
            padding: 0 !important; margin: 0 !important; overflow: hidden !important;
        }

        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d0d1a; }
        ::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }

        div[data-testid="metric-container"] {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 14px 16px;
            border-radius: 16px;
        }
        div[data-testid="metric-container"] label { color: #888 !important; font-size: 0.84rem !important; }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.85rem !important; line-height: 1.05 !important; color: #e8e8f0 !important;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

        .kpi-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 20px 22px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
            margin-bottom: 8px;
        }
        .kpi-card:hover { border-color: rgba(255,95,0,0.3); transform: translateY(-2px); }
        .kpi-card::before {
            content: '';
            position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, #EB001B, #FF5F00, #F79E1B);
            opacity: 0; transition: opacity 0.3s;
        }
        .kpi-card:hover::before { opacity: 1; }
        .kpi-card-label {
            color: #888; font-size: 0.72rem; text-transform: uppercase;
            letter-spacing: 0.12em; margin-bottom: 10px;
        }
        .kpi-card-value {
            color: #e8e8f0; font-size: 1.35rem; font-weight: 700;
            font-family: 'Space Mono', monospace; margin-bottom: 4px; word-break: break-word;
        }
        .kpi-card-delta { color: #666; font-size: 0.78rem; }

        .compact-kpi-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 16px 18px;
            transition: border-color 0.3s;
            position: relative;
            overflow: hidden;
        }
        .compact-kpi-card:hover { border-color: rgba(255,95,0,0.3); }
        .compact-kpi-label {
            color: #888; font-size: 0.75rem; text-transform: uppercase;
            letter-spacing: 0.1em; margin-bottom: 8px;
        }
        .compact-kpi-value {
            color: #e8e8f0; font-size: 1.2rem; font-weight: 700;
            font-family: 'Space Mono', monospace; line-height: 1.25;
            margin-bottom: 6px; word-break: break-word;
        }
        .compact-kpi-delta { color: #888; font-size: 0.78rem; line-height: 1.3; }

        .danger-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
        }
        .danger-badge.danger { background: rgba(235,0,27,0.12); color: #EB001B; }
        .danger-badge.safe { background: rgba(45,159,63,0.12); color: #2D9F3F; }

        .dashboard-card {
            background: rgba(18,18,42,0.9);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px; padding: 18px 20px;
        }
        .section-card {
            background: rgba(18,18,42,0.9);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px; padding: 16px 18px;
        }
        .reason-card {
            background: rgba(18,18,42,0.9);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px; padding: 16px 18px; min-height: 132px;
        }
        .reason-card.success { border-color: rgba(45,159,63,0.22); }
        .reason-card.warning { border-color: rgba(255,95,0,0.22); }
        .reason-card.info { border-color: rgba(247,158,27,0.22); }
        .reason-title { color: #e8e8f0; font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
        .reason-metric { color: #2D9F3F; font-size: 0.86rem; font-weight: 600; margin-bottom: 8px; }
        .reason-card.warning .reason-metric { color: #FF5F00; }
        .reason-card.info .reason-metric { color: #F79E1B; }
        .reason-text { color: #aaa; font-size: 0.93rem; line-height: 1.5; }

        .hero-panel {
            background: linear-gradient(135deg, rgba(18,18,42,0.98), rgba(12,12,28,0.99));
            border: 1px solid rgba(235,0,27,0.18);
            border-radius: 20px; padding: 22px 26px; margin-bottom: 0.75rem;
        }
        .hero-kicker { color: #FF5F00; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
        .hero-title { color: #e8e8f0; font-size: 1.7rem; font-weight: 700; margin-bottom: 6px; }
        .hero-subtitle { color: #aaa; font-size: 0.98rem; line-height: 1.55; margin-bottom: 16px; }

        .signal-chip-row { display: flex; flex-wrap: wrap; gap: 10px; }
        .signal-chip {
            background: rgba(255,95,0,0.1); border: 1px solid rgba(255,95,0,0.2);
            color: #ffd4b8; border-radius: 999px; padding: 6px 12px;
            font-size: 0.84rem; line-height: 1.2;
        }
        .signal-chip.warning { background: rgba(247,158,27,0.12); border-color: rgba(247,158,27,0.28); color: #ffe8bd; }
        .signal-chip.success { background: rgba(45,159,63,0.12); border-color: rgba(45,159,63,0.24); color: #d8ffe7; }

        .dashboard-caption { color: #888; font-size: 0.9rem; margin-top: -4px; }
        [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mastercard_logo_html(size: int = 28) -> str:
    w = int(size * 1.6)
    return (
        f'<svg width="{w}" height="{size}" viewBox="0 0 52 32">'
        '<circle cx="16" cy="16" r="14" fill="#EB001B"/>'
        '<circle cx="36" cy="16" r="14" fill="#F79E1B"/>'
        '<path d="M26 4.8a13.9 13.9 0 0 0-5 11.2A13.9 13.9 0 0 0 26 27.2a13.9 13.9 0 0 0 5-11.2A13.9 13.9 0 0 0 26 4.8z" fill="#FF5F00"/>'
        "</svg>"
    )


def render_gauge_html(value: int, size: int = 160) -> str:
    radius = 60
    circumference = math.pi * radius
    offset = circumference * (1 - value / 100)
    color = "#EB001B" if value > 75 else "#FF5F00" if value > 40 else "#2D9F3F"
    h = int(size * 0.65)
    return (
        f'<div style="text-align:center;">'
        f'<svg width="{size}" height="{h}" viewBox="0 0 160 100">'
        f'<path d="M20 90 A60 60 0 0 1 140 90" fill="none" stroke="#1a1a2e" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M20 90 A60 60 0 0 1 140 90" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"'
        f' stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>'
        f'<text x="80" y="78" text-anchor="middle" fill="white" font-size="28"'
        f' font-family="Space Mono, monospace" font-weight="700">{value}%</text>'
        f"</svg>"
        f"</div>"
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
