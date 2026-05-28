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

        * { box-sizing: border-box; }

        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d0d1a; }
        ::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideRight {
            from { opacity: 0; transform: translateX(-30px); }
            to   { opacity: 1; transform: translateX(0); }
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(235,0,27,0.4); }
            50%       { box-shadow: 0 0 0 8px rgba(235,0,27,0); }
        }

        .stApp {
            background: #0a0a14;
            font-family: 'DM Sans', 'Segoe UI', Roboto, Arial, sans-serif;
        }
        .stApp button, .stApp input, .stApp textarea, .stApp select,
        .stApp label, .stApp p, .stApp li, .stApp a,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            font-family: 'DM Sans', 'Segoe UI', Roboto, Arial, sans-serif;
        }

        .material-icons, .material-icons-round, .material-icons-outlined,
        .material-symbols-rounded, .material-symbols-outlined,
        [class*="material-symbol"], [class*="material-icons"] {
            display: none !important;
            visibility: hidden !important;
        }

        [data-testid="stSidebar"] {
            background: #0d0d1a;
            border-right: 1px solid rgba(255,255,255,0.04);
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
        #MainMenu, footer { visibility: hidden; }

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
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            padding: 12px 14px;
            border-radius: 16px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        div[data-testid="metric-container"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #EB001B, #FF5F00, #F79E1B);
            opacity: 0;
            transition: opacity 0.3s;
        }
        div[data-testid="metric-container"]:hover::before { opacity: 1; }
        div[data-testid="metric-container"]:hover { border-color: rgba(255,95,0,0.3); transform: translateY(-2px); }
        div[data-testid="metric-container"] label {
            color: #888 !important;
            font-size: 0.84rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            font-size: 1.85rem !important;
            line-height: 1.05 !important;
            color: #e8e8f0 !important;
            font-family: 'Space Mono', monospace !important;
        }
        div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
            font-size: 0.78rem !important;
            color: #FF5F00 !important;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        .dashboard-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
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
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 12px 14px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .compact-kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #EB001B, #FF5F00, #F79E1B);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .compact-kpi-card:hover { border-color: rgba(255,95,0,0.3); transform: translateY(-2px); }
        .compact-kpi-card:hover::before { opacity: 1; }
        .compact-kpi-label {
            color: #888;
            font-size: 0.8rem;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .compact-kpi-value {
            color: #e8e8f0;
            font-size: 1.25rem;
            font-weight: 650;
            line-height: 1.25;
            margin-bottom: 8px;
            word-break: break-word;
            font-family: 'Space Mono', monospace;
        }
        .compact-kpi-delta {
            color: #FF5F00;
            font-size: 0.78rem;
            line-height: 1.3;
        }

        .dashboard-caption {
            color: #888;
            font-size: 0.9rem;
            margin-top: -4px;
        }

        .hero-panel {
            background: linear-gradient(135deg, rgba(18,18,42,0.98), rgba(15,15,34,0.98));
            border: 1px solid rgba(235,0,27,0.2);
            border-radius: 20px;
            padding: 22px 24px;
            margin-bottom: 0.75rem;
        }
        .hero-kicker {
            color: #FF5F00;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .hero-title {
            color: #e8e8f0;
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            color: #aaa;
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
            background: rgba(255,95,0,0.1);
            border: 1px solid rgba(255,95,0,0.22);
            color: #e8e8f0;
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.86rem;
            line-height: 1.2;
        }
        .signal-chip.warning {
            background: rgba(247,158,27,0.12);
            border-color: rgba(247,158,27,0.28);
            color: #fde9b0;
        }
        .signal-chip.success {
            background: rgba(235,0,27,0.1);
            border-color: rgba(235,0,27,0.25);
            color: #ffccd0;
        }

        .section-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 16px 18px;
        }
        .section-card ul { padding-left: 18px; margin: 8px 0 0 0; }
        .section-card li { color: #aaa; font-size: 0.93rem; line-height: 1.8; }

        .reason-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 132px;
        }
        .reason-card.success { border-color: rgba(235,0,27,0.25); }
        .reason-card.warning { border-color: rgba(247,158,27,0.22); }
        .reason-card.info    { border-color: rgba(255,95,0,0.22); }
        .reason-title { color: #e8e8f0; font-size: 1rem; font-weight: 600; margin-bottom: 8px; }
        .reason-metric { color: #EB001B; font-size: 0.86rem; font-weight: 600; margin-bottom: 8px; font-family: 'Space Mono', monospace; }
        .reason-card.warning .reason-metric { color: #F79E1B; }
        .reason-card.info    .reason-metric { color: #FF5F00; }
        .reason-text { color: #aaa; font-size: 0.93rem; line-height: 1.5; }

        .card-page-hero {
            background: linear-gradient(135deg, rgba(18,18,42,0.98), rgba(15,15,34,0.98));
            border: 1px solid rgba(235,0,27,0.18);
            border-radius: 20px;
            padding: 22px 24px;
            margin-bottom: 1rem;
        }
        .card-page-title { color: #e8e8f0; font-size: 1.65rem; font-weight: 700; margin: 0 0 6px 0; }
        .card-page-subtitle { color: #aaa; font-size: 0.96rem; line-height: 1.5; margin: 0; }

        .card-score-panel {
            background: linear-gradient(160deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 20px;
            padding: 20px 22px;
            margin-bottom: 0.5rem;
        }
        .card-score-panel.pattern {
            border-color: rgba(235,0,27,0.35);
            box-shadow: 0 0 0 1px rgba(235,0,27,0.08) inset;
            animation: pulse 3s infinite;
        }
        .card-score-number {
            color: #e8e8f0;
            font-size: 2.35rem;
            font-weight: 750;
            line-height: 1;
            margin: 4px 0 10px 0;
            font-family: 'Space Mono', monospace;
        }
        .card-score-caption { color: #888; font-size: 0.84rem; margin-bottom: 4px; }
        .card-id-line {
            color: #e8e8f0;
            font-size: 1.02rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            margin-bottom: 2px;
            font-family: 'Space Mono', monospace;
        }
        .card-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .card-status-pill.pattern {
            background: rgba(235,0,27,0.12);
            border: 1px solid rgba(235,0,27,0.3);
            color: #ff8080;
        }
        .card-status-pill.regular {
            background: rgba(45,159,63,0.12);
            border: 1px solid rgba(45,159,63,0.3);
            color: #7fff99;
        }
        .card-mini-metrics {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            margin-top: 14px;
        }
        .card-mini-metric {
            background: rgba(10,10,20,0.55);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 10px 12px;
        }
        .card-mini-metric-label { color: #666; font-size: 0.76rem; margin-bottom: 4px; }
        .card-mini-metric-value {
            color: #e8e8f0;
            font-size: 1rem;
            font-weight: 650;
            font-family: 'Space Mono', monospace;
        }

        .section-heading { color: #e8e8f0; font-size: 1.05rem; font-weight: 650; margin: 0 0 0.35rem 0; }
        .section-hint    { color: #888; font-size: 0.86rem; margin: 0 0 0.75rem 0; }

        .card-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 0.75rem 0 1.1rem 0;
        }
        .card-kpi-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 14px 16px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .card-kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #EB001B, #FF5F00, #F79E1B);
            opacity: 0;
            transition: opacity 0.3s;
        }
        .card-kpi-card:hover { border-color: rgba(255,95,0,0.3); transform: translateY(-2px); }
        .card-kpi-card:hover::before { opacity: 1; }
        .card-kpi-label { color: #888; font-size: 0.8rem; margin-bottom: 8px; }
        .card-kpi-value {
            color: #e8e8f0;
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 6px;
            font-family: 'Space Mono', monospace;
        }
        .card-kpi-delta { color: #FF5F00; font-size: 0.78rem; }

        .panel-card {
            background: linear-gradient(145deg, #12122a, #0f0f22);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 16px 18px;
        }

        .merchant-list { display: flex; flex-direction: column; gap: 0; }
        .merchant-row {
            display: grid;
            grid-template-columns: 44px 1fr auto;
            gap: 14px;
            align-items: center;
            padding: 14px 4px;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .merchant-row:last-child { border-bottom: none; }
        .merchant-icon {
            width: 40px; height: 40px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.78rem; font-weight: 700; color: #fff;
        }
        .merchant-icon.tone-0 { background: linear-gradient(135deg, #EB001B, #FF5F00); }
        .merchant-icon.tone-1 { background: linear-gradient(135deg, #FF5F00, #F79E1B); }
        .merchant-icon.tone-2 { background: linear-gradient(135deg, rgba(247,158,27,0.7), #F79E1B); }
        .merchant-icon.tone-3 { background: linear-gradient(135deg, rgba(235,0,27,0.7), #EB001B); }
        .merchant-icon.tone-4 { background: rgba(255,255,255,0.1); }
        .merchant-name { color: #e8e8f0; font-size: 0.95rem; font-weight: 650; margin-bottom: 4px; }
        .merchant-meta { color: #666; font-size: 0.8rem; }
        .merchant-amount {
            color: #e8e8f0; font-size: 0.98rem; font-weight: 700;
            white-space: nowrap; font-family: 'Space Mono', monospace;
        }

        .geo-list { display: flex; flex-direction: column; gap: 14px; padding-top: 4px; }
        .geo-row { display: grid; grid-template-columns: 88px 1fr auto; gap: 12px; align-items: center; }
        .geo-country { color: #e8e8f0; font-size: 0.9rem; font-weight: 600; }
        .geo-bar-wrap { height: 10px; background: rgba(255,255,255,0.06); border-radius: 999px; overflow: hidden; }
        .geo-bar { height: 100%; border-radius: 999px; }
        .geo-bar.tone-0 { background: linear-gradient(90deg, #EB001B, #FF5F00); }
        .geo-bar.tone-1 { background: linear-gradient(90deg, #FF5F00, #F79E1B); }
        .geo-bar.tone-2 { background: #F79E1B; }
        .geo-amount {
            color: #e8e8f0; font-size: 0.88rem; font-weight: 650;
            min-width: 72px; text-align: right; font-family: 'Space Mono', monospace;
        }

        .chart-card-title { color: #e8e8f0; font-size: 0.98rem; font-weight: 650; margin: 0 0 2px 0; }
        .chart-card-subtitle { color: #888; font-size: 0.8rem; margin: 0 0 10px 0; }

        [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size: 0.95rem !important; }
        div[data-testid="stRadio"] { margin-bottom: 0.75rem; }

        .sidebar-brand {
            display: flex;
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
            padding: 8px 4px 22px 4px;
            margin-bottom: 0.25rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .mastercard-mark { display: block; width: 52px; height: 32px; overflow: visible; }
        .sidebar-brand-title {
            color: #e8e8f0;
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            line-height: 1;
            margin: 0;
        }
        .sidebar-brand-subtitle {
            color: #FF5F00;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            margin-top: 2px;
        }

        [data-testid="stSidebar"] .stRadio > label {
            color: #888 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* gauge */
        .gauge-container {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; padding: 8px 0;
        }
        .gauge-label {
            font-size: 10px; color: #888; margin-bottom: 6px;
            text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600;
        }
        .gauge-sublabel { margin-top: 8px; font-size: 13px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_gauge_chart(value: float, size: int = 140) -> str:
    radius = 60
    circumference = math.pi * radius
    offset = circumference - (value / 100) * circumference
    if value > 75:
        color = "#EB001B"
    elif value > 40:
        color = "#FF5F00"
    else:
        color = "#2D9F3F"
    w = int(size * 1.6)
    h = int(size * 0.65)
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 160 100">'
        f'<path d="M20 90 A60 60 0 0 1 140 90" fill="none" stroke="#1a1a2e" stroke-width="14" stroke-linecap="round"/>'
        f'<path d="M20 90 A60 60 0 0 1 140 90" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"'
        f' stroke-dasharray="{circumference:.2f}" stroke-dashoffset="{offset:.2f}"/>'
        f'<text x="80" y="78" text-anchor="middle" fill="white" font-size="26"'
        f' font-family="Space Mono, monospace" font-weight="700">{value:.0f}%</text>'
        f'</svg>'
    )


def render_sidebar_brand() -> None:
    st.markdown(
        (
            '<div class="sidebar-brand">'
            '<svg class="mastercard-mark" viewBox="0 0 52 32" xmlns="http://www.w3.org/2000/svg" aria-label="Mastercard">'
            '<circle cx="19" cy="16" r="13" fill="#EB001B"/>'
            '<circle cx="33" cy="16" r="13" fill="#F79E1B"/>'
            '<path d="M26 4.8a13.9 13.9 0 0 0-5 11.2A13.9 13.9 0 0 0 26 27.2a13.9 13.9 0 0 0 5-11.2A13.9 13.9 0 0 0 26 4.8z" fill="#FF5F00"/>'
            "</svg>"
            "<div>"
            '<div class="sidebar-brand-title">Mastercard</div>'
            '<div class="sidebar-brand-subtitle">Data Quest 2026</div>'
            "</div>"
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
    alert_text = "⚠ ALERT" if is_pattern else "✓ OK"
    status_color = "#EB001B" if is_pattern else "#2D9F3F"
    gauge_html = render_gauge_chart(score_pct)
    st.markdown(
        (
            f'<div class="card-score-panel {status_class}">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
            f'<span class="card-id-line">{escape(card_masked)}</span>'
            f'</div>'
            f'<span class="card-status-pill {status_class}">{alert_text}</span>'
            f'<div class="gauge-container">'
            f'<div class="gauge-label">Hidden entrepreneur probability</div>'
            f'{gauge_html}'
            f'<div class="gauge-sublabel" style="color:{status_color};">{escape(predicted_label)}</div>'
            f'</div>'
            f'<div class="card-mini-metrics">'
            f'<div class="card-mini-metric">'
            f'<div class="card-mini-metric-label">Model class</div>'
            f'<div class="card-mini-metric-value">{escape(predicted_label)}</div>'
            f'</div>'
            f'<div class="card-mini-metric">'
            f'<div class="card-mini-metric-label">Confidence ({escape(confidence_text)})</div>'
            f'<div class="card-mini-metric-value">{confidence_pct:.1f}%</div>'
            f'</div>'
            f'</div>'
            f'</div>'
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
