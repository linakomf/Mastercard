from __future__ import annotations

from html import escape
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mastercard_dashboard.config import DashboardPaths, FEATURE_LABELS, SEGMENT_LABELS
from mastercard_dashboard.data import query_transactions
from mastercard_dashboard.modeling import (
    PATTERN_TOP_PERCENTILE,
    build_business_explanation,
    get_local_shap_explanation,
    get_pattern_cards_ranked,
    get_pattern_portfolio_stats,
    load_submission_scores,
)
from mastercard_dashboard.ui import (
    render_card_kpi_strip,
    render_card_score_panel,
    render_compact_kpi_row,
    render_geo_spend_list,
    render_kpi_row,
    render_merchant_purchase_list,
    render_page_header,
    render_section_heading,
)


PLOTLY_TEMPLATE = "plotly_dark"
FINTECH_BLUE = "#2F80ED"
FINTECH_CYAN = "#35C2FF"
FINTECH_GREEN = "#FF6B6B"
FINTECH_RED = "#FF6B6B"
FINTECH_ORANGE = "#F5A623"


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def actual_segment_label(name: str) -> str:
    return SEGMENT_LABELS.get(name, name)


def get_date_bounds(eda_summary: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_candidates = [
        eda_summary.get("business", {}).get("date_start"),
        eda_summary.get("consumer", {}).get("date_start"),
    ]
    end_candidates = [
        eda_summary.get("business", {}).get("date_end"),
        eda_summary.get("consumer", {}).get("date_end"),
    ]

    start_values = [pd.to_datetime(value) for value in start_candidates if value]
    end_values = [pd.to_datetime(value) for value in end_candidates if value]

    if start_values and end_values:
        return min(start_values), max(end_values)

    today = pd.Timestamp.today().normalize()
    return today - timedelta(days=180), today


def build_segment_comparison_frame(feature_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [
        "avg_amount",
        "online_ratio",
        "recurring_ratio",
        "top_merchant_ratio",
        "avg_transaction_hour",
        "weekend_activity_ratio",
        "unique_merchants",
        "international_ratio",
    ]:
        rows.append(
            {
                "feature": feature,
                "feature_label": feature_label(feature),
                "consumer": float(feature_summary.loc["consumer", (feature, "mean")]),
                "hidden_entrepreneur": float(feature_summary.loc["business", (feature, "mean")]),
            }
        )
    return pd.DataFrame(rows)


def format_kzt(value: float) -> str:
    return f"{value:,.0f} KZT".replace(",", " ")


def format_kzt_compact(value: float) -> str:
    amount = float(value)
    if amount >= 1_000_000:
        compact = amount / 1_000_000
        text = f"{compact:.1f}M".replace(".0M", "M")
        return f"{text} KZT"
    if amount >= 1_000:
        return f"{amount / 1_000:.0f}K KZT"
    return format_kzt(amount)


def merchant_initials(name: str) -> str:
    cleaned = "".join(char for char in str(name) if char.isalnum())
    if len(cleaned) >= 2:
        return cleaned[:2].upper()
    return str(name)[:2].upper()


def format_int(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", " ")


@st.cache_data(show_spinner=False)
def load_global_shap_signals(local_explanations_path: str) -> pd.DataFrame:
    path = Path(local_explanations_path)
    if not path.exists():
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap", "mentions", "feature_label"])

    explanations = pd.read_csv(path)
    rows: list[pd.DataFrame] = []
    for idx in range(1, 6):
        feature_col = f"reason_{idx}_feature"
        shap_col = f"reason_{idx}_shap"
        if feature_col not in explanations.columns or shap_col not in explanations.columns:
            continue

        row_frame = explanations[[feature_col, shap_col]].dropna().rename(
            columns={feature_col: "feature", shap_col: "shap_value"}
        )
        row_frame["abs_shap"] = row_frame["shap_value"].abs()
        rows.append(row_frame)

    if not rows:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap", "mentions", "feature_label"])

    long_frame = pd.concat(rows, ignore_index=True)
    summary = (
        long_frame.groupby("feature", as_index=False)
        .agg(
            mean_abs_shap=("abs_shap", "mean"),
            mean_shap=("shap_value", "mean"),
            mentions=("feature", "size"),
        )
        .sort_values("mean_abs_shap", ascending=False)
    )
    summary["feature_label"] = summary["feature"].map(feature_label)
    return summary


def truncate_label(value: str, max_len: int = 18) -> str:
    return value if len(value) <= max_len else f"{value[: max_len - 1]}…"


def infer_mcc_category(mcc: str, merchant_name: str = "") -> str:
    mcc = str(mcc)
    merchant_lower = str(merchant_name).lower()

    keyword_rules = [
        (["google ads", "meta ads", "tiktok ads", "instagram promote", "linkedin ads", "yandex direct"], "Advertising and marketing"),
        (["hubspot", "atlassian", "shopify", "salesforce", "mailchimp"], "SaaS and business tools"),
        (["amazon web services", "microsoft azure", "hetzner", "digitalocean", "cloud"], "Cloud and infrastructure"),
        (["kcell", "tele2", "activ", "beeline"], "Telecom"),
        (["air astana", "airline", "airlines"], "Travel and airlines"),
        (["gas", "oil", "servicestations", "kazmunaygas", "qazaq oil", "sinooil", "compasgas"], "Fuel and auto"),
    ]
    for keywords, category in keyword_rules:
        if any(keyword in merchant_lower for keyword in keywords):
            return category

    exact_map = {
        "7311": "Advertising and marketing",
        "7372": "IT services and SaaS",
        "4814": "Telecom",
        "4900": "Utilities",
        "4511": "Airlines",
        "4111": "Transport",
        "4121": "Taxi and mobility",
        "5541": "Fuel and gas stations",
        "5542": "Fuel and gas stations",
        "5812": "Restaurants and cafes",
        "5814": "Fast food and delivery",
        "5912": "Pharmacy and health",
        "5999": "Miscellaneous retail",
        "5311": "General retail",
        "5732": "Electronics",
        "5691": "Clothing and accessories",
    }
    if mcc in exact_map:
        return exact_map[mcc]

    try:
        mcc_int = int(mcc)
    except ValueError:
        return "Other"

    if 3000 <= mcc_int < 3300 or 3500 <= mcc_int < 3800 or 4400 <= mcc_int < 4800:
        return "Travel and transport"
    if 4800 <= mcc_int < 4900:
        return "Telecom and digital services"
    if 5200 <= mcc_int < 6000:
        return "Retail and lifestyle"
    if 5500 <= mcc_int < 5600:
        return "Fuel and auto"
    if 5600 <= mcc_int < 5800:
        return "Apparel and goods"
    if 5800 <= mcc_int < 5900:
        return "Food and restaurants"
    if 7300 <= mcc_int < 7400:
        return "Business services"
    return "Other"


def add_mcc_labels(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()

    enriched = transactions.copy()
    enriched["mcc_category"] = enriched.apply(
        lambda row: infer_mcc_category(row["mcc"], row.get("merchant_name", "")),
        axis=1,
    )
    enriched["mcc_label"] = enriched["mcc_category"] + " (" + enriched["mcc"].astype(str) + ")"
    enriched["channel_label"] = enriched["channel"].replace({"online": "Online", "POS": "Offline"})
    return enriched


def apply_card_profile_filters(
    transactions: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    merchants: list[str],
    mcc_labels: list[str],
    countries: list[str],
    channels: list[str],
) -> pd.DataFrame:
    filtered = transactions.copy()
    if filtered.empty:
        return filtered

    if date_range is not None:
        start_date, end_date = date_range
        filtered = filtered.loc[
            (filtered["transaction_timestamp"] >= start_date)
            & (filtered["transaction_timestamp"] <= end_date)
        ]
    if merchants:
        filtered = filtered.loc[filtered["merchant_name"].isin(merchants)]
    if mcc_labels:
        filtered = filtered.loc[filtered["mcc_label"].isin(mcc_labels)]
    if countries:
        filtered = filtered.loc[filtered["country"].isin(countries)]
    if channels:
        filtered = filtered.loc[filtered["channel_label"].isin(channels)]
    return filtered


def build_business_summary(card_record: pd.Series, transactions: pd.DataFrame) -> tuple[str, list[str]]:
    badges: list[str] = []
    online_ratio = float(card_record["online_ratio"])
    recurring_ratio = float(card_record["recurring_ratio"])
    international_ratio = float(card_record["international_ratio"])

    category_text = ""
    if not transactions.empty:
        top_categories = (
            transactions["mcc_category"].value_counts().head(2).index.tolist()
        )
        if top_categories:
            category_text = ", ".join(top_categories)

    if online_ratio >= 0.7:
        badges.append("High online activity")
    if recurring_ratio >= 0.08:
        badges.append("Recurring payments")
    if international_ratio >= 0.2:
        badges.append("Cross-border activity")
    if not transactions.empty and transactions["mcc_category"].isin(
        ["SaaS and business tools", "Advertising and marketing", "Cloud and infrastructure", "IT services and SaaS"]
    ).any():
        badges.append("SaaS/Ads behavior")

    if int(card_record["predicted_class"]) == 1:
        summary = "Behavior is consistent with digital business activity."
        if category_text:
            summary += f" Main categories: {category_text}."
    else:
        summary = "Behavior is closer to a typical consumer pattern."
        if category_text:
            summary += f" Main categories: {category_text}."

    return summary, badges


def mask_card_number(card_number: str) -> str:
    card = str(card_number)
    if len(card) <= 8:
        return card
    return f"{card[:4]} •••• {card[-4:]}"


def render_badges(badges: list[str]) -> None:
    if not badges:
        return

    badges_html = "".join(
        [f'<span class="signal-chip success">{badge}</span>' for badge in badges]
    )
    st.markdown(
        f'<div class="signal-chip-row" style="margin: 8px 0 16px 0;">{badges_html}</div>',
        unsafe_allow_html=True,
    )


def compute_prediction_confidence(card_record: pd.Series) -> float:
    probability = float(card_record["entrepreneur_probability"])
    predicted_class = int(card_record["predicted_class"])
    return probability if predicted_class == 1 else 1 - probability


def confidence_label(confidence: float) -> str:
    if confidence >= 0.85:
        return "High"
    if confidence >= 0.65:
        return "Medium"
    return "Borderline"


def consumer_display_score(consumers: pd.DataFrame, card_number: str, card_record: pd.Series) -> float:
    consumer_row = consumers.loc[consumers["card_number"] == card_number]
    if not consumer_row.empty:
        return float(consumer_row.iloc[0]["display_score"])
    return float(card_record["entrepreneur_probability"])


def consumer_score_metrics(display_score: float) -> tuple[float, float, str]:
    score_pct = display_score * 100
    certainty = max(display_score, 1.0 - display_score)
    return score_pct, certainty * 100, confidence_label(certainty)


def resolve_selected_card(
    filtered_cards: pd.DataFrame,
    state_key: str,
    fallback_card: str,
) -> str:
    available = filtered_cards["card_number"].astype(str).tolist()
    selected = str(st.session_state.get(state_key, fallback_card))
    if selected not in available:
        selected = fallback_card
        st.session_state[state_key] = selected
    return selected


SEGMENT_LIST_LIMIT = 250


def render_card_purchases_by_merchant(transactions: pd.DataFrame, selected_card: str) -> None:
    st.markdown(
        f'<h3 class="section-heading">Where they shopped</h3>'
        f'<p class="section-hint">Card {escape(mask_card_number(selected_card))}</p>',
        unsafe_allow_html=True,
    )

    if transactions.empty:
        st.info("No transactions for this card.")
        return

    by_merchant = (
        transactions.groupby("merchant_name", dropna=False)
        .agg(
            tx_count=("transaction_amount_kzt", "size"),
            amount_kzt=("transaction_amount_kzt", "sum"),
            countries=("country", lambda values: ", ".join(sorted({str(v) for v in values if pd.notna(v)})[:4])),
        )
        .reset_index()
        .sort_values(["amount_kzt", "tx_count"], ascending=[False, False])
    )

    rows = []
    for record in by_merchant.itertuples(index=False):
        countries = str(record.countries) if pd.notna(record.countries) and record.countries else "—"
        rows.append(
            {
                "initials": merchant_initials(record.merchant_name),
                "name": str(record.merchant_name),
                "meta": f"{countries} · {int(record.tx_count)} tx",
                "amount": format_kzt_compact(float(record.amount_kzt)),
            }
        )
    render_merchant_purchase_list(rows)


def top_merchant_examples(transactions: pd.DataFrame, limit: int = 3) -> str:
    if transactions.empty:
        return "No data"

    merchants = (
        transactions.groupby("merchant_name")["transaction_amount_kzt"]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
        .index.tolist()
    )
    return ", ".join(merchants)


def build_investigation_findings(
    card_record: pd.Series,
    transactions: pd.DataFrame,
    local_explanation: pd.DataFrame,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    predicted_class = int(card_record["predicted_class"])
    merchant_examples = top_merchant_examples(transactions)

    if predicted_class == 1:
        if float(card_record["online_ratio"]) >= 0.7:
            findings.append(
                {
                    "title": "High online activity",
                    "metric": f"{card_record['online_ratio']:.0%} online",
                    "text": "Most transactions happen through digital channels, which is common for online-first business activity.",
                    "tone": "success",
                }
            )
        if float(card_record["recurring_ratio"]) >= 0.08 or float(card_record["recurring_capable_ratio"]) >= 0.25:
            findings.append(
                {
                    "title": "Recurring SaaS / ads payments",
                    "metric": f"{card_record['recurring_ratio']:.0%} recurring",
                    "text": f"The pattern is consistent with subscriptions and business tools. Leading merchants: {merchant_examples}.",
                    "tone": "success",
                }
            )
        if float(card_record["top_merchant_ratio"]) >= 0.3:
            findings.append(
                {
                    "title": "Concentrated merchant activity",
                    "metric": f"{card_record['top_merchant_ratio']:.0%} with the top merchant",
                    "text": "Spending is concentrated around a narrow merchant base, which looks more operational than retail-driven.",
                    "tone": "warning",
                }
            )
        if float(card_record["weekend_activity_ratio"]) <= 0.2 or float(card_record["avg_transaction_hour"]) <= 13:
            findings.append(
                {
                    "title": "Weekday operational pattern",
                    "metric": f"{card_record['weekend_activity_ratio']:.0%} on weekends",
                    "text": f"Low weekend activity and an average hour of {card_record['avg_transaction_hour']:.1f} suggest a working-day operating rhythm.",
                    "tone": "info",
                }
            )
    else:
        if float(card_record["online_ratio"]) < 0.7:
            findings.append(
                {
                    "title": "Insufficient digital intensity",
                    "metric": f"{card_record['online_ratio']:.0%} online",
                    "text": "The online share is not strong enough to support a business-like pattern.",
                    "tone": "info",
                }
            )
        if float(card_record["recurring_ratio"]) < 0.08:
            findings.append(
                {
                    "title": "Weak recurring pattern",
                    "metric": f"{card_record['recurring_ratio']:.0%} recurring",
                    "text": "Recurring payments tied to SaaS, ads, or tooling are limited.",
                    "tone": "info",
                }
            )
        if float(card_record["top_merchant_ratio"]) < 0.3:
            findings.append(
                {
                    "title": "Low merchant concentration",
                    "metric": f"{card_record['top_merchant_ratio']:.0%} with the top merchant",
                    "text": "The activity is not concentrated enough, so the card looks closer to a consumer profile.",
                    "tone": "warning",
                }
            )
        if float(card_record["weekend_activity_ratio"]) > 0.2:
            findings.append(
                {
                    "title": "No weekday operational pattern",
                    "metric": f"{card_record['weekend_activity_ratio']:.0%} on weekends",
                    "text": "Visible weekend activity is more consistent with regular consumer behavior.",
                    "tone": "warning",
                }
            )

    if len(findings) < 4:
        source = (
            local_explanation.loc[local_explanation["shap_value"] > 0]
            if predicted_class == 1
            else local_explanation.loc[local_explanation["shap_value"] < 0]
        )
        for _, row in source.head(6).iterrows():
            findings.append(
                {
                    "title": feature_label(row["feature"]),
                    "metric": f"SHAP {row['shap_value']:.2f}",
                    "text": f"A value of {row['feature_value']:.3f} for this feature has a meaningful effect on the final classification.",
                    "tone": "info",
                }
            )
            if len(findings) >= 4:
                break

    return findings[:4]


def render_reason_cards(findings: list[dict[str, str]]) -> None:
    if not findings:
        return

    first_row = st.columns(2, gap="large")
    for column, finding in zip(first_row, findings[:2]):
        column.markdown(
            f"""
            <div class="reason-card {finding['tone']}">
                <div class="reason-title">{finding['title']}</div>
                <div class="reason-metric">{finding['metric']}</div>
                <div class="reason-text">{finding['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(findings) > 2:
        second_row = st.columns(2, gap="large")
        for column, finding in zip(second_row, findings[2:4]):
            column.markdown(
                f"""
                <div class="reason-card {finding['tone']}">
                    <div class="reason-title">{finding['title']}</div>
                    <div class="reason-metric">{finding['metric']}</div>
                    <div class="reason-text">{finding['text']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_investigation_summary(
    card_record: pd.Series,
    explanation_points: list[str],
    transactions: pd.DataFrame,
) -> None:
    predicted_class = int(card_record["predicted_class"])
    probability = float(card_record["entrepreneur_probability"])
    confidence = compute_prediction_confidence(card_record)
    merchant_examples = top_merchant_examples(transactions)

    if predicted_class == 1:
        title = "The card aligns with a business-like profile"
        subtitle = (
            f"The score indicates a {probability:.1%} likelihood of hidden entrepreneur behavior "
            f"with {confidence_label(confidence).lower()} confidence ({confidence:.1%}). "
            f"Leading merchant footprint: {merchant_examples}."
        )
    else:
        title = "The card remains closer to a consumer profile"
        subtitle = (
            f"The hidden entrepreneur likelihood is {probability:.1%}; "
            f"the card stays in the consumer segment with {confidence_label(confidence).lower()} confidence ({confidence:.1%}). "
            f"Leading merchant footprint: {merchant_examples}."
        )

    summary_text = explanation_points[0] if explanation_points else "Only a limited number of business-oriented signals are visible for this card."
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-kicker">Investigation summary</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div class="hero-subtitle" style="margin-bottom:0;">{summary_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview_page(
    dashboard_data: dict[str, Any],
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
) -> None:
    render_page_header(
        "Overview",
        "Portfolio monitoring for hidden entrepreneur detection: quality, behavior signals, and segment differences.",
    )

    overview = dashboard_data["overview"]
    metrics_df = model_bundle["metrics_df"]
    xgb_row = metrics_df.loc[metrics_df["model"] == "XGBoost"].iloc[0]

    render_kpi_row(
        [
            ("Cards scored", f"{overview['total_cards']:,}", "Card-level portfolio"),
            ("Transactions", f"{overview['total_transactions']:,}", "Combined raw data"),
            ("Target = 1 share", f"{overview['target_rate']:.1%}", "Proxy share of business cards"),
            ("XGBoost ROC-AUC", f"{xgb_row['roc_auc']:.4f}", "Primary model"),
            ("XGBoost Precision", f"{xgb_row['precision']:.4f}", "Lower false positive risk"),
            ("XGBoost Recall", f"{xgb_row['recall']:.4f}", "High detection coverage"),
        ]
    )

    scored_cards = model_bundle["scored_cards"]
    feature_summary = dashboard_data["feature_summary"]
    comparison_frame = build_segment_comparison_frame(feature_summary)

    tab_portfolio, tab_behavior, tab_scores = st.tabs(
        ["Portfolio snapshot", "Behavior comparison", "Score distribution"]
    )

    with tab_portfolio:
        col_left, col_right = st.columns([1.05, 1], gap="large")

        with col_left:
            class_frame = pd.DataFrame(
                {
                    "segment": ["Consumer", "Hidden entrepreneur (proxy)"],
                    "cards": [overview["consumer_cards"], overview["business_cards"]],
                }
            )
            fig = px.pie(
                class_frame,
                names="segment",
                values="cards",
                hole=0.62,
                color="segment",
                color_discrete_map={
                    "Consumer": FINTECH_BLUE,
                    "Hidden entrepreneur (proxy)": FINTECH_ORANGE,
                },
                template=PLOTLY_TEMPLATE,
                title="Card portfolio structure",
            )
            fig.update_layout(
                legend_title_text="Segment",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            importance = model_bundle["feature_importance"].head(8).sort_values(ascending=True)
            importance_frame = importance.reset_index()
            importance_frame.columns = ["feature", "importance"]
            fig = px.bar(
                importance_frame,
                x="importance",
                y="feature",
                orientation="h",
                template=PLOTLY_TEMPLATE,
                title="Top hidden entrepreneur drivers",
                color="importance",
                color_continuous_scale=["#123B6D", FINTECH_CYAN],
            )
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Importance score",
                yaxis_title="Feature",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_behavior:
        fig = go.Figure()
        for column_name, label, color in [
            ("consumer", "Consumers", FINTECH_BLUE),
            ("hidden_entrepreneur", "Hidden entrepreneurs (proxy)", FINTECH_ORANGE),
        ]:
            fig.add_trace(
                go.Bar(
                    x=comparison_frame["feature_label"],
                    y=comparison_frame[column_name],
                    name=label,
                    marker_color=color,
                )
            )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Average behavior: business vs consumer",
            barmode="group",
            xaxis_title="Feature",
            yaxis_title="Average value",
            legend_title_text="Segment",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            comparison_frame.rename(
                columns={
                    "feature_label": "Feature",
                    "consumer": "Consumer average",
                    "hidden_entrepreneur": "Hidden entrepreneur average",
                }
            )[["Feature", "Consumer average", "Hidden entrepreneur average"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_scores:
        chart_frame = scored_cards.copy()
        chart_frame["actual_segment"] = chart_frame["target"].map(
            {0: "Consumer", 1: "Hidden entrepreneur (proxy)"}
        )
        fig = px.histogram(
            chart_frame,
            x="entrepreneur_probability",
            color="actual_segment",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            color_discrete_map={
                "Consumer": FINTECH_BLUE,
                "Hidden entrepreneur (proxy)": FINTECH_ORANGE,
            },
            template=PLOTLY_TEMPLATE,
            title="Hidden entrepreneur probability distribution",
        )
        fig.update_layout(
            xaxis_title="Predicted probability",
            yaxis_title="Card count",
            legend_title_text="Actual segment",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Folder with exported PNG/CSV artifacts: `{paths.artifact_dir}`"
        )


def render_card_profile_page(
    dashboard_data: dict[str, Any],
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
) -> None:
    portfolio_stats = get_pattern_portfolio_stats(model_bundle, paths)
    pattern_count = int(portfolio_stats["pattern_count"])
    regular_count = int(portfolio_stats["regular_count"])
    top_pct = int(PATTERN_TOP_PERCENTILE * 100)

    scored_cards = model_bundle["scored_cards"].copy()
    scored_cards["card_number"] = scored_cards["card_number"].astype(str)
    consumers = scored_cards.loc[scored_cards["target"] == 0].copy()
    pattern_ids = portfolio_stats["pattern_card_ids"]

    if paths.submission_path.exists():
        score_lookup = load_submission_scores(str(paths.submission_path))
        consumers = consumers.merge(score_lookup, on="card_number", how="left")
        consumers["display_score"] = consumers["score"].fillna(consumers["entrepreneur_probability"])
    else:
        consumers["display_score"] = consumers["entrepreneur_probability"]

    segment_options = {
        "pattern": f"Pattern · {pattern_count:,}",
        "regular": f"Regular · {regular_count:,}",
        "all": f"All consumer · {len(consumers):,}",
    }
    segment_frames = {
        "pattern": consumers.loc[consumers["card_number"].isin(pattern_ids)],
        "regular": consumers.loc[~consumers["card_number"].isin(pattern_ids)],
        "all": consumers,
    }

    segment_key = st.radio(
        "Segment",
        options=list(segment_options.keys()),
        format_func=lambda key: segment_options[key],
        horizontal=True,
        key="card_segment_filter",
        label_visibility="collapsed",
    )

    filtered_cards = (
        segment_frames[segment_key]
        .sort_values(["display_score", "card_number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if filtered_cards.empty:
        st.warning("No cards in the selected segment.")
        return

    card_selection_key = "card_profile_selected_card"
    segment_card_ids = filtered_cards["card_number"].astype(str).tolist()
    if st.session_state.get("card_profile_prev_segment") != segment_key:
        st.session_state[card_selection_key] = segment_card_ids[0]
        st.session_state["card_profile_prev_segment"] = segment_key

    try:
        pattern_cards_ranked = get_pattern_cards_ranked(model_bundle, paths)
    except Exception as exc:
        st.error(f"Failed to load scores: {exc}")
        pattern_cards_ranked = pd.DataFrame()

    if segment_key == "pattern" and not pattern_cards_ranked.empty:
        list_frame = pattern_cards_ranked.copy()
    else:
        list_frame = filtered_cards.head(SEGMENT_LIST_LIMIT).copy()
        list_frame["#"] = range(1, len(list_frame) + 1)
        list_frame["pattern_pct"] = (list_frame["display_score"] * 100).round(1)

    default_card = str(list_frame.iloc[0]["card_number"])

    table_frame = list_frame.copy()
    if "pattern_pct" not in table_frame.columns:
        table_frame["pattern_pct"] = (table_frame["display_score"] * 100).round(1)
    table_view = table_frame.rename(
        columns={"card_number": "Card number", "pattern_pct": "Score, %"}
    )
    rank_column = "#" if "#" in table_view.columns else "№"
    table_view = table_view[[rank_column, "Card number", "Score, %"]].copy()

    selected_card = resolve_selected_card(filtered_cards, card_selection_key, default_card)
    card_matches = scored_cards.loc[scored_cards["card_number"] == selected_card]
    if card_matches.empty:
        st.warning("Card not found.")
        return

    card_record = card_matches.iloc[0]
    is_pattern = selected_card in pattern_ids
    display_score = consumer_display_score(consumers, selected_card, card_record)
    score_pct, confidence_pct, confidence_text = consumer_score_metrics(display_score)

    transactions = query_transactions(
        paths=paths,
        merchant_ref=dashboard_data["merchant_ref"],
        card_number=selected_card,
        limit=5000,
    )
    transactions = add_mcc_labels(transactions)

    if not transactions.empty:
        min_tx_date = transactions["transaction_timestamp"].min().normalize()
        max_tx_date = transactions["transaction_timestamp"].max().normalize()
        default_date_range = (min_tx_date.date(), max_tx_date.date())
        with st.expander("Period filters", expanded=False):
            profile_date_range = st.date_input(
                "Period",
                value=default_date_range,
                min_value=min_tx_date.date(),
                max_value=max_tx_date.date(),
                key="card_profile_date_range",
            )
        if len(profile_date_range) == 2:
            profile_start = pd.Timestamp(profile_date_range[0])
            profile_end = pd.Timestamp(profile_date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            transactions = apply_card_profile_filters(
                transactions=transactions,
                date_range=(profile_start, profile_end),
                merchants=[],
                mcc_labels=[],
                countries=[],
                channels=[],
            )

    render_card_kpi_strip(
        [
            ("Pattern cards", f"{pattern_count:,}", f"top {top_pct}% consumer"),
            ("Probability", f"{score_pct:.1f}%", "hidden entrepreneur"),
            ("Confidence", f"{confidence_pct:.1f}%", confidence_text.lower()),
            ("Transactions", f"{len(transactions):,}", "in period"),
        ]
    )

    list_col, profile_col = st.columns([1.55, 1], gap="large")
    with list_col:
        render_section_heading("Card ranking", "Click a row to open the card profile.")
        table_selection = st.dataframe(
            table_view,
            use_container_width=True,
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key="card_list_table",
            column_config={
                "Score, %": st.column_config.ProgressColumn(
                    "Score, %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
        )
        if table_selection.selection.rows:
            selected_row = table_selection.selection.rows[0]
            st.session_state[card_selection_key] = str(table_frame.iloc[selected_row]["card_number"])

    with profile_col:
        render_card_score_panel(
            card_masked=mask_card_number(selected_card),
            score_pct=score_pct,
            is_pattern=is_pattern,
            predicted_label=str(card_record["predicted_label"]),
            confidence_pct=confidence_pct,
            confidence_text=confidence_text,
        )

    if transactions.empty:
        st.info("Not enough transactions for charts and merchant list.")
        return

    monthly_amount = (
        transactions.assign(tx_month=transactions["transaction_timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby("tx_month")["transaction_amount_kzt"]
        .sum()
        .reset_index()
        .sort_values("tx_month")
    )
    monthly_amount["month_label"] = monthly_amount["tx_month"].dt.strftime("%b")
    period_start = transactions["transaction_timestamp"].min()
    period_end = transactions["transaction_timestamp"].max()
    period_label = f"{period_start.strftime('%b %Y')} — {period_end.strftime('%b %Y')}"

    weekday_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    weekday_breakdown = (
        transactions.assign(weekday=transactions["transaction_timestamp"].dt.dayofweek)
        .groupby("weekday")
        .size()
        .reindex(range(7), fill_value=0)
        .reset_index(name="Transactions")
    )
    weekday_breakdown["weekday_label"] = weekday_breakdown["weekday"].map(weekday_map)

    top_countries = (
        transactions.groupby("country")["transaction_amount_kzt"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .reset_index()
    )
    geo_rows = [
        {
            "country": str(row.country),
            "amount_value": float(row.transaction_amount_kzt),
            "amount_label": format_kzt_compact(float(row.transaction_amount_kzt)),
        }
        for row in top_countries.itertuples(index=False)
    ]

    chart_top_left, chart_top_right = st.columns(2, gap="medium")
    with chart_top_left:
        st.markdown(
            f'<p class="chart-card-title">Monthly spend</p><p class="chart-card-subtitle">{escape(period_label)}</p>',
            unsafe_allow_html=True,
        )
        fig = px.line(
            monthly_amount,
            x="month_label",
            y="transaction_amount_kzt",
            template=PLOTLY_TEMPLATE,
            markers=True,
        )
        fig.update_traces(line_color=FINTECH_CYAN, marker=dict(size=7, color=FINTECH_CYAN))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
        )
        fig.update_yaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

    with chart_top_right:
        st.markdown(
            '<p class="chart-card-title">Activity by weekday</p><p class="chart-card-subtitle">Transaction count</p>',
            unsafe_allow_html=True,
        )
        fig = px.bar(
            weekday_breakdown,
            x="weekday_label",
            y="Transactions",
            template=PLOTLY_TEMPLATE,
            color_discrete_sequence=[FINTECH_ORANGE],
        )
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    _, chart_geo_col = st.columns([1, 1], gap="medium")
    with chart_geo_col:
        st.markdown(
            '<p class="chart-card-title">Spending geography</p><p class="chart-card-subtitle">Amount in KZT</p>',
            unsafe_allow_html=True,
        )
        render_geo_spend_list(geo_rows)

    render_card_purchases_by_merchant(transactions, str(selected_card))


def render_transactions_page(
    dashboard_data: dict[str, Any],
    paths: DashboardPaths,
) -> None:
    render_page_header(
        "Transactions",
        "Search and filter the transaction stream by date, MCC, merchant, channel, and recurring status.",
    )

    merchant_ref = dashboard_data["merchant_ref"]
    eda_summary = dashboard_data["eda_summary"]
    min_date, max_date = get_date_bounds(eda_summary)
    default_start = max(min_date, max_date - timedelta(days=30))

    with st.sidebar:
        st.markdown("### Transaction filters")
        search_text = st.text_input("Search by card / merchant / MCC", value="")
        date_range = st.date_input(
            "Period",
            value=(default_start.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        selected_mcc = st.multiselect(
            "MCC filter",
            options=sorted(merchant_ref["mcc"].astype(str).unique().tolist()),
        )
        selected_merchants = st.multiselect(
            "Merchant filter",
            options=sorted(merchant_ref["merchant_name"].astype(str).unique().tolist()),
        )
        selected_channels = st.multiselect(
            "Channel filter",
            options=["online", "POS"],
            default=["online", "POS"],
        )
        recurring_only = st.selectbox(
            "Recurring filter",
            options=["All", "Only recurring", "Only non-recurring"],
            index=0,
        )
        row_limit = st.slider("Row limit", min_value=100, max_value=5000, value=1000, step=100)

    start_date = pd.Timestamp(date_range[0]) if len(date_range) == 2 else pd.Timestamp(default_start)
    end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) if len(date_range) == 2 else pd.Timestamp(max_date)

    transactions = query_transactions(
        paths=paths,
        merchant_ref=merchant_ref,
        search_text=search_text,
        start_date=start_date,
        end_date=end_date,
        mccs=selected_mcc,
        merchant_names=selected_merchants,
        channels=selected_channels,
        recurring_only=recurring_only,
        limit=row_limit,
    )

    total_amount = float(transactions["transaction_amount_kzt"].sum()) if not transactions.empty else 0.0
    avg_amount = float(transactions["transaction_amount_kzt"].mean()) if not transactions.empty else 0.0

    render_kpi_row(
        [
            ("Transactions after filter", f"{len(transactions):,}", "Current query"),
            ("Amount after filter", f"{total_amount:,.0f} KZT", "Sum of displayed rows"),
            ("Average amount", f"{avg_amount:,.0f} KZT", "Current sample"),
            ("Countries", str(transactions["country"].nunique() if not transactions.empty else 0), "Unique countries"),
        ]
    )

    charts_tab, table_tab = st.tabs(["Breakdowns", "Interactive table"])

    with charts_tab:
        if transactions.empty:
            st.warning("No transactions matched the current filters.")
        else:
            breakdown_left, breakdown_right = st.columns(2, gap="large")

            with breakdown_left:
                top_merchants = (
                    transactions.groupby("merchant_name")["transaction_amount_kzt"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(12)
                    .sort_values()
                    .reset_index()
                )
                fig = px.bar(
                    top_merchants,
                    x="transaction_amount_kzt",
                    y="merchant_name",
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    title="Top merchants in current filter",
                    color_discrete_sequence=[FINTECH_CYAN],
                )
                fig.update_layout(
                    xaxis_title="Amount (KZT)",
                    yaxis_title="Merchant",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

            with breakdown_right:
                mcc_breakdown = (
                    transactions.groupby("mcc")
                    .size()
                    .sort_values(ascending=False)
                    .head(12)
                    .sort_values()
                    .reset_index(name="Transactions")
                )
                fig = px.bar(
                    mcc_breakdown,
                    x="Transactions",
                    y="mcc",
                    orientation="h",
                    template=PLOTLY_TEMPLATE,
                    title="MCC distribution",
                    color_discrete_sequence=[FINTECH_ORANGE],
                )
                fig.update_layout(
                    xaxis_title="Transactions",
                    yaxis_title="MCC",
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

    with table_tab:
        display_frame = transactions[
            [
                "transaction_timestamp",
                "card_number",
                "merchant_name",
                "transaction_amount_kzt",
                "mcc",
                "country",
                "channel",
                "is_recurring",
            ]
        ].rename(
            columns={
                "transaction_timestamp": "Date",
                "card_number": "Card",
                "merchant_name": "Merchant",
                "transaction_amount_kzt": "Amount (KZT)",
                "mcc": "MCC",
                "country": "Country",
                "channel": "Channel",
                "is_recurring": "Recurring",
            }
        )
        st.dataframe(display_frame, use_container_width=True, hide_index=True)


def render_model_insights_page(
    dashboard_data: dict[str, Any],
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
) -> None:
    render_page_header(
        "Model Validation & Business Interpretation",
        "Validation view focused on business trade-offs, explainability, and realistic interpretation.",
    )

    metrics_df = model_bundle["metrics_df"].copy()
    metrics_df = metrics_df.round(
        {"precision": 4, "recall": 4, "f1_score": 4, "roc_auc": 4}
    )
    xgb_row = metrics_df.loc[metrics_df["model"] == "XGBoost"].iloc[0]
    metrics_df["primary"] = metrics_df["model"].eq("XGBoost").map({True: "Primary", False: ""})

    render_kpi_row(
        [
            ("Primary model", "XGBoost", "Selected model"),
            ("ROC-AUC", f"{xgb_row['roc_auc']:.4f}", "Class separation quality"),
            ("Precision", f"{xgb_row['precision']:.4f}", "False positive control"),
            ("Recall", f"{xgb_row['recall']:.4f}", "Detection coverage"),
        ]
    )

    top_left, top_right = st.columns([1.15, 1], gap="large")

    with top_left:
        st.markdown("### Model comparison")
        st.dataframe(
            metrics_df[
                ["model", "precision", "recall", "f1_score", "roc_auc", "primary"]
            ].rename(
                columns={
                    "model": "Model",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1-score",
                    "roc_auc": "ROC-AUC",
                    "primary": "Selection",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "All three models show near-perfect metrics, so the main question is not which one wins on headline numbers, but which one is easier to explain, more stable, and more usable in practice."
        )

    with top_right:
        st.markdown("### Why XGBoost was selected")
        st.markdown(
            """
            <div class="section-card">
                <b>Primary model selection</b><br><br>
                <ul>
                    <li>it handles nonlinear behavior patterns well</li>
                    <li>it offers strong explainability through SHAP and feature importance</li>
                    <li>it performs well on aggregated transaction features</li>
                    <li>it provides a practical score for review and investigation workflows</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    middle_left, middle_right = st.columns([1, 1], gap="large")

    with middle_left:
        st.markdown("### XGBoost confusion matrix")
        confusion = model_bundle["xgb_confusion_matrix"]
        fig = go.Figure(
            data=go.Heatmap(
                z=confusion,
                x=["Predicted: consumer", "Predicted: hidden entrepreneur"],
                y=["Actual: consumer", "Actual: hidden entrepreneur"],
                colorscale="Blues",
                text=confusion,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Confusion matrix",
            xaxis_title="Predicted class",
            yaxis_title="Actual class",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with middle_right:
        st.markdown("### Feature importance")
        importance_frame = (
            model_bundle["feature_importance"].head(12).sort_values(ascending=True).reset_index()
        )
        importance_frame.columns = ["feature", "importance"]
        importance_frame["feature"] = importance_frame["feature"].map(feature_label)
        fig = px.bar(
            importance_frame,
            x="importance",
            y="feature",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            title="Feature importance",
            color="importance",
            color_continuous_scale=["#123B6D", FINTECH_CYAN],
        )
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Importance score",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    lower_left, lower_right = st.columns([1, 1], gap="large")

    with lower_left:
        st.markdown("### Business interpretation of errors")
        st.markdown(
            """
            <div class="section-card">
                <b>False Positive</b><br>
                A regular consumer card is flagged as a hidden entrepreneur.
                Business effect: unnecessary outreach, extra review, or misdirected sales/risk attention.<br><br>
                <b>False Negative</b><br>
                A hidden entrepreneur remains in the consumer segment.
                Business effect: missed migration to business products, underestimated portfolio value, and weaker targeting.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Threshold explanation")
        st.markdown(
            """
            <div class="section-card">
                The current threshold is <b>0.50</b>.<br><br>
                Lowering the threshold would capture more potential hidden entrepreneurs, but review volume and false positives would rise.<br><br>
                Raising the threshold would improve precision further, but increase the risk of missing cards with genuine business-like activity.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with lower_right:
        st.markdown("### Validation limitations")
        st.markdown(
            """
            <div class="section-card">
                <ul>
                    <li>the metrics are almost perfect, which may indicate a highly separable or partially synthetic dataset</li>
                    <li>the target is based on a proxy label, business card vs consumer card, not confirmed SME ground truth</li>
                    <li>out-of-time validation is still needed before production rollout</li>
                    <li>drift should be monitored on online ratio, recurring patterns, and merchant concentration</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Realistic interpretation")
        st.markdown(
            """
            <div class="section-card">
                This section is meant as an enterprise validation view:
                how explainable the solution is, what trade-offs it creates, and why it can support investigation workflows rather than just academic benchmarking.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Show ROC curves", expanded=False):
        roc_frame = pd.concat(model_bundle["roc_curves"].values(), ignore_index=True)
        fig = px.line(
            roc_frame,
            x="fpr",
            y="tpr",
            color="model",
            template=PLOTLY_TEMPLATE,
            title="ROC curve comparison",
            color_discrete_sequence=[FINTECH_CYAN, FINTECH_GREEN, FINTECH_ORANGE],
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="#8A94A6", dash="dash"),
        )
        fig.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            legend_title_text="Model",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)



def render_shap_page(
    dashboard_data: dict[str, Any],
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
) -> None:
    render_page_header(
        "SHAP",
        "Review global SHAP patterns and local feature contributions for the selected card.",
    )

    scored_cards = model_bundle["scored_cards"].sort_values(
        ["entrepreneur_probability", "card_number"], ascending=[False, True]
    )
    flagged_only = st.toggle("Show only cards flagged as hidden entrepreneurs", value=True)
    if flagged_only:
        scored_cards = scored_cards.loc[scored_cards["predicted_class"] == 1]

    if scored_cards.empty:
        st.warning("No cards are available for the current SHAP filter.")
        return

    selected_card = st.selectbox(
        "Select a card for local explanation",
        scored_cards["card_number"].astype(str).tolist(),
        index=0,
    )
    card_frame = model_bundle["scored_cards"].loc[
        model_bundle["scored_cards"]["card_number"] == selected_card
    ].copy()
    card_record = card_frame.iloc[0]
    local_explanation, _ = get_local_shap_explanation(model_bundle, card_frame)
    positive_features = local_explanation.loc[local_explanation["shap_value"] > 0].head(8)
    explanation_points = build_business_explanation(card_record, local_explanation)

    summary_tab, local_tab = st.tabs(["SHAP summary", "Local card explanation"])

    with summary_tab:
        global_shap = load_global_shap_signals(str(paths.local_explanations_path))
        if not global_shap.empty:
            summary_chart = global_shap.head(12).sort_values("mean_abs_shap", ascending=True)
            fig = px.bar(
                summary_chart,
                x="mean_abs_shap",
                y="feature_label",
                orientation="h",
                template=PLOTLY_TEMPLATE,
                title="Global SHAP signal ranking",
                color="mean_shap",
                color_continuous_scale=[FINTECH_RED, "#6E7B91", FINTECH_GREEN],
                custom_data=["mentions"],
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Mean |SHAP|=%{x:.4f}<br>"
                    "Average direction=%{marker.color:.4f}<br>"
                    "Mentions=%{customdata[0]}<extra></extra>"
                )
            )
            fig.update_layout(
                xaxis_title="Mean absolute SHAP value",
                yaxis_title="Feature",
                coloraxis_colorbar_title="Avg SHAP",
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("This chart summarizes the strongest recurring SHAP signals across exported card explanations.")

        if Path(paths.shap_summary_path).exists():
            st.image(
                str(paths.shap_summary_path),
                caption="SHAP summary plot for the holdout card sample",
                use_container_width=True,
            )
        else:
            st.info("The pre-generated SHAP image was not found.")

    with local_tab:
        render_compact_kpi_row(
            [
                ("Card", selected_card, "Selected profile"),
                ("Score", f"{card_record['entrepreneur_probability']:.1%}", "Final XGBoost score"),
                ("Predicted class", card_record["predicted_label"], "Local explanation target"),
                ("Actual segment", actual_segment_label(str(card_record["segment_name"])), "Proxy label"),
            ]
        )

        local_chart = local_explanation.head(12).iloc[::-1]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=local_chart["shap_value"],
                    y=local_chart["feature"].map(feature_label),
                    orientation="h",
                    marker_color=[
                        FINTECH_GREEN if value >= 0 else FINTECH_RED
                        for value in local_chart["shap_value"]
                    ],
                    customdata=local_chart["feature_value"],
                    hovertemplate="<b>%{y}</b><br>SHAP=%{x:.4f}<br>Value=%{customdata:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Local SHAP contribution",
            xaxis_title="Contribution to hidden entrepreneur score",
            yaxis_title="Feature",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns([1.05, 1], gap="large")

        with left:
            st.subheader("Top positive features")
            if positive_features.empty:
                st.info("No positive SHAP drivers were found for this card.")
            else:
                st.dataframe(
                    positive_features[
                        ["feature", "feature_value", "shap_value", "direction"]
                    ].rename(
                        columns={
                            "feature": "Feature",
                            "feature_value": "Value",
                            "shap_value": "SHAP",
                            "direction": "Direction",
                        }
                    ).assign(Feature=lambda df: df["Feature"].map(feature_label)),
                    use_container_width=True,
                    hide_index=True,
                )

        with right:
            st.subheader("What drives this score")
            for point in explanation_points:
                st.markdown(f"- {point}")
