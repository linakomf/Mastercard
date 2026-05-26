from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TRANSACTION_COLUMNS = [
    "transaction_date",
    "transaction_timestamp",
    "transaction_amount_kzt",
    "mcc",
    "merchant_id",
    "channel",
    "bank_name",
    "country",
    "card_number",
    "card_tier",
    "tokenized",
    "is_recurring",
]

MERCHANT_COLUMNS = [
    "merchant_id",
    "merchant_name",
    "mcc",
    "merchant_country",
    "recurring_capable",
]

MODEL_FEATURE_COLUMNS = [
    "txn_count",
    "total_amount",
    "avg_amount",
    "median_amount",
    "max_amount",
    "std_amount",
    "unique_merchants",
    "top_merchant_ratio",
    "unique_mcc",
    "avg_transaction_hour",
    "night_activity_ratio",
    "weekend_activity_ratio",
    "online_ratio",
    "tokenized_ratio",
    "recurring_ratio",
    "international_ratio",
    "unique_countries",
    "avg_txn_per_day",
    "recurring_capable_ratio",
]

FEATURE_LABELS = {
    "txn_count": "Transaction count",
    "total_amount": "Total amount",
    "avg_amount": "Average amount",
    "median_amount": "Median amount",
    "max_amount": "Maximum amount",
    "std_amount": "Amount volatility",
    "unique_merchants": "Unique merchants",
    "top_merchant_ratio": "Top merchant share",
    "unique_mcc": "Unique MCCs",
    "avg_transaction_hour": "Average transaction hour",
    "night_activity_ratio": "Night activity ratio",
    "weekend_activity_ratio": "Weekend activity ratio",
    "online_ratio": "Online ratio",
    "tokenized_ratio": "Tokenized ratio",
    "recurring_ratio": "Recurring payment ratio",
    "international_ratio": "International ratio",
    "unique_countries": "Unique countries",
    "avg_txn_per_day": "Average transactions per day",
    "recurring_capable_ratio": "Recurring-capable merchant ratio",
}

SEGMENT_LABELS = {
    "consumer": "Consumer",
    "business": "Business card (proxy)",
}

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "mastercard_hidden_entrepreneur_artifacts"
DOWNLOADS_DIR = Path.home() / "Downloads"


def pick_first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


@dataclass(frozen=True)
class DashboardPaths:
    business_path: Path
    consumer_path: Path
    merchant_path: Path
    artifact_dir: Path
    card_features_path: Path
    feature_summary_path: Path
    metrics_path: Path
    eda_summary_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path
    shap_summary_path: Path
    local_explanations_path: Path
    business_report_path: Path


def discover_paths() -> DashboardPaths:
    artifact_dir = ARTIFACT_DIR
    return DashboardPaths(
        business_path=pick_first_existing(
            DOWNLOADS_DIR / "business_cards_MDQ.parquet",
            DOWNLOADS_DIR / "businass_csv.csv",
        ),
        consumer_path=pick_first_existing(
            DOWNLOADS_DIR / "consumer_cards_MDQ.parquet",
            DOWNLOADS_DIR / "cumsumer_csv.csv",
        ),
        merchant_path=pick_first_existing(
            DOWNLOADS_DIR / "merchants_reference.parquet",
            DOWNLOADS_DIR / "merchants_ref.csv",
        ),
        artifact_dir=artifact_dir,
        card_features_path=artifact_dir / "card_level_features.parquet",
        feature_summary_path=artifact_dir / "feature_summary_by_segment.csv",
        metrics_path=artifact_dir / "model_comparison.csv",
        eda_summary_path=artifact_dir / "eda_summary.json",
        confusion_matrix_path=artifact_dir / "xgboost_confusion_matrix.png",
        feature_importance_path=artifact_dir / "xgboost_feature_importance.png",
        shap_summary_path=artifact_dir / "xgboost_shap_summary.png",
        local_explanations_path=artifact_dir / "xgboost_local_explanations.csv",
        business_report_path=artifact_dir / "business_report.md",
    )
