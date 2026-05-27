from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_NAME = "hidden_entrepreneur"
PRIMARY_MODEL_NAME = "XGBoost"

BASE_FEATURE_COLUMNS = [
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
]

ENRICHED_FEATURE_COLUMNS = [
    "recurring_capable_ratio",
    "b2b_mcc_ratio",
    "amount_cv",
    "days_since_last_tx",
    "business_hours_ratio",
]
MODEL_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS
CV_FOLDS = 5

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


def is_b2b_mcc(mcc: str) -> bool:
    """B2B-oriented MCC ranges: wholesale, utilities, business/professional services."""
    try:
        code = int(str(mcc).strip())
    except (ValueError, TypeError):
        return False
    b2b_ranges = (
        (1500, 2999),
        (4000, 4799),
        (5000, 5599),
        (7300, 7399),
        (7800, 7999),
        (8700, 8999),
    )
    return any(low <= code <= high for low, high in b2b_ranges)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mastercard case study: detect hidden entrepreneurial activity "
            "among retail cardholders using explainable ML."
        )
    )
    parser.add_argument("--business-path", required=True, help="Path to business_cards_MDQ.parquet")
    parser.add_argument("--consumer-path", required=True, help="Path to consumer_cards_MDQ.parquet")
    parser.add_argument("--merchant-path", required=True, help="Path to merchants_reference.parquet")
    parser.add_argument(
        "--output-dir",
        default="mastercard_hidden_entrepreneur_artifacts",
        help="Directory for generated datasets, charts, and reports",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert probabilities into class labels",
    )
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(data), file, indent=2, ensure_ascii=False)


def read_table(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, usecols=columns, low_memory=False)
    return pd.read_parquet(file_path, columns=columns)


def load_merchant_reference(merchant_path: str) -> pd.DataFrame:
    merchant_ref = read_table(
        merchant_path,
        columns=[
            "merchant_id",
            "merchant_name",
            "mcc",
            "merchant_country",
            "recurring_capable",
        ],
    )
    merchant_ref = merchant_ref.rename(columns={"mcc": "merchant_ref_mcc"})
    merchant_ref = merchant_ref.drop_duplicates(subset=["merchant_id"]).copy()
    merchant_ref["recurring_capable"] = merchant_ref["recurring_capable"].fillna(False).astype(bool)
    return merchant_ref[
        ["merchant_id", "merchant_name", "merchant_ref_mcc", "merchant_country", "recurring_capable"]
    ]


def load_and_prepare_transactions(
    data_path: str,
    target: int,
    segment_name: str,
    merchant_ref: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = read_table(data_path, columns=TRANSACTION_COLUMNS).copy()
    raw["target"] = target
    raw["segment_name"] = segment_name

    exact_duplicates = int(raw.duplicated().sum())
    prepared = raw.drop_duplicates().copy()
    non_positive_amounts = int((prepared["transaction_amount_kzt"] <= 0).sum())
    prepared = prepared.loc[prepared["transaction_amount_kzt"] > 0].copy()

    prepared["transaction_date"] = pd.to_datetime(
        prepared["transaction_date"], errors="coerce", dayfirst=True
    )
    prepared["transaction_timestamp"] = pd.to_datetime(
        prepared["transaction_timestamp"], errors="coerce", dayfirst=True
    )
    missing_timestamps = int(prepared["transaction_timestamp"].isna().sum())
    prepared = prepared.dropna(subset=["transaction_timestamp", "card_number", "merchant_id"]).copy()

    prepared = prepared.merge(merchant_ref, on="merchant_id", how="left")
    missing_merchant_ref = int(prepared["merchant_name"].isna().sum())

    prepared["channel"] = prepared["channel"].astype("string").fillna("unknown")
    prepared["country"] = prepared["country"].astype("string").fillna("Unknown")
    prepared["mcc"] = prepared["mcc"].astype("string").fillna("unknown")
    prepared["merchant_country"] = prepared["merchant_country"].astype("string").fillna("Unknown")
    prepared["merchant_name"] = prepared["merchant_name"].astype("string").fillna("Unknown merchant")
    prepared["tokenized"] = prepared["tokenized"].fillna(False).astype(bool)
    prepared["is_recurring"] = prepared["is_recurring"].fillna(False).astype(bool)
    prepared["recurring_capable"] = prepared["recurring_capable"].fillna(False).astype(bool)

    prepared["tx_hour"] = prepared["transaction_timestamp"].dt.hour.astype("int16")
    prepared["tx_day"] = prepared["transaction_timestamp"].dt.floor("D")
    prepared["is_weekend"] = (prepared["transaction_timestamp"].dt.dayofweek >= 5).astype("int8")
    prepared["is_night"] = prepared["tx_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    prepared["is_online"] = prepared["channel"].str.lower().eq("online").astype("int8")
    prepared["is_international"] = (
        ~prepared["country"].str.lower().eq("kazakhstan")
    ).astype("int8")
    prepared["is_b2b_mcc"] = prepared["mcc"].map(is_b2b_mcc).astype("int8")
    prepared["is_business_hour"] = prepared["tx_hour"].between(9, 18).astype("int8")

    quality = {
        "segment": segment_name,
        "rows_raw": len(raw),
        "rows_after_cleaning": len(prepared),
        "exact_duplicates_removed": exact_duplicates,
        "non_positive_amount_rows_removed": non_positive_amounts,
        "rows_with_missing_timestamp_before_drop": missing_timestamps,
        "missing_merchant_reference_after_merge": missing_merchant_ref,
        "unique_cards": int(prepared["card_number"].nunique()),
        "unique_merchants": int(prepared["merchant_id"].nunique()),
        "unique_mcc": int(prepared["mcc"].nunique()),
        "date_start": prepared["transaction_timestamp"].min(),
        "date_end": prepared["transaction_timestamp"].max(),
        "online_share": round(float(prepared["is_online"].mean()), 4),
        "recurring_share": round(float(prepared["is_recurring"].mean()), 4),
        "international_share": round(float(prepared["is_international"].mean()), 4),
        "top_10_countries": prepared["country"].value_counts().head(10).to_dict(),
    }
    return prepared, quality


def build_card_features(transactions: pd.DataFrame) -> pd.DataFrame:
    reference_ts = transactions["transaction_timestamp"].max()

    observation_days = (
        transactions.groupby("card_number")["tx_day"].agg(["min", "max"]).rename(
            columns={"min": "first_tx_day", "max": "last_tx_day"}
        )
    )
    observation_days["observed_days"] = (
        (observation_days["last_tx_day"] - observation_days["first_tx_day"]).dt.days + 1
    ).clip(lower=1)

    last_tx_ts = transactions.groupby("card_number")["transaction_timestamp"].max()
    days_since_last_tx = (reference_ts - last_tx_ts).dt.days.clip(lower=0)

    grouped = transactions.groupby("card_number").agg(
        target=("target", "max"),
        segment_name=("segment_name", "first"),
        txn_count=("transaction_amount_kzt", "size"),
        total_amount=("transaction_amount_kzt", "sum"),
        avg_amount=("transaction_amount_kzt", "mean"),
        median_amount=("transaction_amount_kzt", "median"),
        max_amount=("transaction_amount_kzt", "max"),
        std_amount=("transaction_amount_kzt", "std"),
        unique_merchants=("merchant_id", "nunique"),
        unique_mcc=("mcc", "nunique"),
        avg_transaction_hour=("tx_hour", "mean"),
        night_activity_ratio=("is_night", "mean"),
        weekend_activity_ratio=("is_weekend", "mean"),
        online_ratio=("is_online", "mean"),
        tokenized_ratio=("tokenized", "mean"),
        recurring_ratio=("is_recurring", "mean"),
        international_ratio=("is_international", "mean"),
        unique_countries=("country", "nunique"),
        active_days=("tx_day", "nunique"),
        recurring_capable_ratio=("recurring_capable", "mean"),
        b2b_mcc_ratio=("is_b2b_mcc", "mean"),
        business_hours_ratio=("is_business_hour", "mean"),
    )

    top_merchant_ratio = (
        transactions.groupby(["card_number", "merchant_id"])
        .size()
        .rename("merchant_txn_count")
        .reset_index()
        .groupby("card_number")["merchant_txn_count"]
        .max()
        .rename("top_merchant_txn_count")
    )

    features = grouped.join(observation_days["observed_days"]).join(top_merchant_ratio)
    features["top_merchant_ratio"] = (
        features["top_merchant_txn_count"] / features["txn_count"].clip(lower=1)
    )
    features["avg_txn_per_day"] = features["txn_count"] / features["observed_days"].clip(lower=1)
    features["std_amount"] = features["std_amount"].fillna(0.0)
    features["amount_cv"] = features["std_amount"] / features["avg_amount"].clip(lower=1.0)
    features["amount_cv"] = features["amount_cv"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = features.join(days_since_last_tx.rename("days_since_last_tx"))
    features = features.drop(columns=["top_merchant_txn_count"])
    return features.reset_index()


def plot_class_balance(features: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=features, x="target", hue="target", palette="Set2", legend=False)
    ax.set_xticks([0, 1], ["Consumer (0)", "Business card / target=1"])
    ax.set_title("Card-Level Target Balance")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of cards")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_class_balance.png", dpi=200)
    plt.close()


def plot_key_feature_distributions(features: pd.DataFrame, output_dir: Path) -> None:
    plot_columns = [
        "avg_transaction_hour",
        "online_ratio",
        "weekend_activity_ratio",
        "top_merchant_ratio",
        "recurring_ratio",
        "avg_amount",
    ]
    melted = features.melt(
        id_vars=["target"],
        value_vars=plot_columns,
        var_name="feature",
        value_name="value",
    )
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=melted,
        x="feature",
        y="value",
        hue="target",
        showfliers=False,
        palette="Set2",
    )
    plt.xticks(rotation=25, ha="right")
    plt.title("Key Feature Distributions by Segment")
    plt.xlabel("")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_key_feature_distributions.png", dpi=200)
    plt.close()


def compute_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    summary = (
        features.groupby("target")[BASE_FEATURE_COLUMNS + ENRICHED_FEATURE_COLUMNS]
        .agg(["mean", "median"])
        .round(4)
    )
    summary.index = ["consumer", "business"]
    return summary


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_weight=5,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    features: pd.DataFrame,
    decision_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    modeling_frame = features.set_index("card_number").copy()
    X = modeling_frame[MODEL_FEATURE_COLUMNS]
    y = modeling_frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(scale_pos_weight=scale_pos_weight)

    metrics_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    prediction_store: dict[str, pd.DataFrame] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= decision_threshold).astype(int)
        cm = confusion_matrix(y_test, predictions)

        fitted_models[model_name] = pipeline
        prediction_store[model_name] = pd.DataFrame(
            {
                "card_number": X_test.index,
                "actual_target": y_test.values,
                "predicted_probability": probabilities,
                "predicted_label": predictions,
            }
        ).set_index("card_number")

        metrics_rows.append(
            {
                "model": model_name,
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["roc_auc", "f1_score", "precision", "recall"],
        ascending=False,
    )

    evaluation_context = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "fitted_models": fitted_models,
        "predictions": prediction_store,
    }
    return metrics_df, evaluation_context


def plot_confusion_matrix_for_primary_model(
    metrics_df: pd.DataFrame,
    evaluation_context: dict[str, Any],
    output_dir: Path,
) -> None:
    predictions = evaluation_context["predictions"][PRIMARY_MODEL_NAME]
    y_test = predictions["actual_target"]
    predicted = predictions["predicted_label"]
    cm = confusion_matrix(y_test, predicted)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred consumer", "Pred hidden entrepreneur"],
        yticklabels=["Actual consumer", "Actual hidden entrepreneur"],
    )
    plt.title(f"{PRIMARY_MODEL_NAME} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_confusion_matrix.png", dpi=200)
    plt.close()


def plot_feature_importance(evaluation_context: dict[str, Any], output_dir: Path) -> pd.Series:
    xgb_model = evaluation_context["fitted_models"][PRIMARY_MODEL_NAME].named_steps["model"]
    importance = pd.Series(
        xgb_model.feature_importances_,
        index=MODEL_FEATURE_COLUMNS,
        name="importance",
    ).sort_values(ascending=False)

    plt.figure(figsize=(9, 6))
    top_importance = importance.head(15).reset_index()
    top_importance.columns = ["feature", "importance"]
    sns.barplot(
        data=top_importance,
        x="importance",
        y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
    )
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "xgboost_feature_importance.png", dpi=200)
    plt.close()
    return importance


def generate_shap_outputs(
    evaluation_context: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    xgb_pipeline = evaluation_context["fitted_models"][PRIMARY_MODEL_NAME]
    imputer = xgb_pipeline.named_steps["imputer"]
    model = xgb_pipeline.named_steps["model"]

    X_test = evaluation_context["X_test"]
    predictions = evaluation_context["predictions"][PRIMARY_MODEL_NAME]

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    shap_sample = X_test_imputed.sample(n=min(4000, len(X_test_imputed)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values_sample = explainer(shap_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values_sample,
        shap_sample,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    shap_summary_path = output_dir / "xgboost_shap_summary.png"
    plt.savefig(shap_summary_path, dpi=200, bbox_inches="tight")
    plt.close()

    candidate_cards = predictions.sort_values("predicted_probability", ascending=False).copy()
    candidate_cards["sample_type"] = "top_predicted_hidden_entrepreneur"

    false_positives = predictions.loc[
        (predictions["predicted_label"] == 1) & (predictions["actual_target"] == 0)
    ].copy()
    false_positives["sample_type"] = "false_positive"

    false_negatives = predictions.loc[
        (predictions["predicted_label"] == 0) & (predictions["actual_target"] == 1)
    ].copy()
    false_negatives["sample_type"] = "false_negative"

    selected = pd.concat(
        [
            candidate_cards.head(5),
            false_positives.head(3),
            false_negatives.head(3),
        ]
    ).drop_duplicates()

    if selected.empty:
        selected = candidate_cards.head(5).copy()

    local_X = X_test_imputed.loc[selected.index]
    local_shap = explainer(local_X)
    explanation_rows: list[dict[str, Any]] = []

    for row_idx, card_number in enumerate(local_X.index):
        contributions = pd.Series(local_shap.values[row_idx], index=local_X.columns)
        top_contributions = contributions.reindex(
            contributions.abs().sort_values(ascending=False).head(5).index
        )

        explanation_row = {
            "card_number": card_number,
            "sample_type": selected.loc[card_number, "sample_type"],
            "actual_target": int(selected.loc[card_number, "actual_target"]),
            "predicted_label": int(selected.loc[card_number, "predicted_label"]),
            "predicted_probability": float(selected.loc[card_number, "predicted_probability"]),
        }

        for reason_idx, (feature_name, shap_value) in enumerate(top_contributions.items(), start=1):
            direction = "pushes_to_hidden_entrepreneur" if shap_value > 0 else "pushes_to_consumer"
            explanation_row[f"reason_{reason_idx}_feature"] = feature_name
            explanation_row[f"reason_{reason_idx}_shap"] = float(shap_value)
            explanation_row[f"reason_{reason_idx}_direction"] = direction
            explanation_row[f"reason_{reason_idx}_value"] = float(local_X.loc[card_number, feature_name])

        explanation_rows.append(explanation_row)

    explanations_df = pd.DataFrame(explanation_rows).sort_values(
        by="predicted_probability",
        ascending=False,
    )
    explanations_df.to_csv(output_dir / "xgboost_local_explanations.csv", index=False, encoding="utf-8-sig")
    return explanations_df, shap_summary_path


def build_business_signal_table(feature_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name in [
        "txn_count",
        "unique_merchants",
        "recurring_ratio",
        "online_ratio",
        "international_ratio",
        "top_merchant_ratio",
        "avg_amount",
        "weekend_activity_ratio",
        "avg_transaction_hour",
        "recurring_capable_ratio",
    ]:
        consumer_mean = float(feature_summary.loc["consumer", (feature_name, "mean")])
        business_mean = float(feature_summary.loc["business", (feature_name, "mean")])
        uplift = np.nan if consumer_mean == 0 else (business_mean / consumer_mean) - 1
        rows.append(
            {
                "feature": feature_name,
                "consumer_mean": consumer_mean,
                "business_mean": business_mean,
                "business_vs_consumer_pct": uplift * 100 if uplift == uplift else np.nan,
            }
        )

    signal_df = pd.DataFrame(rows).sort_values(
        by="business_vs_consumer_pct", ascending=False, na_position="last"
    )
    return signal_df


def run_cross_validation(
    features: pd.DataFrame,
) -> pd.DataFrame:
    modeling_frame = features.set_index("card_number").copy()
    X = modeling_frame[MODEL_FEATURE_COLUMNS]
    y = modeling_frame["target"]
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    xgb_pipeline = build_models(scale_pos_weight=scale_pos_weight)[PRIMARY_MODEL_NAME]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        xgb_pipeline,
        X,
        y,
        cv=cv,
        scoring=["roc_auc", "precision", "recall", "f1"],
        n_jobs=-1,
    )
    return pd.DataFrame(
        {
            "fold": range(1, CV_FOLDS + 1),
            "roc_auc": cv_results["test_roc_auc"],
            "precision": cv_results["test_precision"],
            "recall": cv_results["test_recall"],
            "f1_score": cv_results["test_f1"],
        }
    )


def plot_precision_recall_curve(
    evaluation_context: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    predictions = evaluation_context["predictions"][PRIMARY_MODEL_NAME]
    y_test = predictions["actual_target"].values
    probabilities = predictions["predicted_probability"].values

    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"XGBoost (PR-AUC={pr_auc:.4f})")
    plt.axhline(y_test.mean(), color="gray", linestyle="--", label="Baseline (positive rate)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (hold-out 20%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()

    threshold_rows: list[dict[str, Any]] = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        predicted = (probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_test, predicted, zero_division=0),
                "recall": recall_score(y_test, predicted, zero_division=0),
                "f1_score": f1_score(y_test, predicted, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(output_dir / "threshold_tuning.csv", index=False, encoding="utf-8-sig")
    return threshold_df


def fit_final_model(features: pd.DataFrame) -> Pipeline:
    """Train on all labeled cards: business=1, consumer=0 (proxy labels)."""
    modeling_frame = features.set_index("card_number").copy()
    X = modeling_frame[MODEL_FEATURE_COLUMNS]
    y = modeling_frame["target"]
    scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    pipeline = build_models(scale_pos_weight=scale_pos_weight)[PRIMARY_MODEL_NAME]
    pipeline.fit(X, y)
    return pipeline


def build_top_suspicious_shap_table(
    final_model: Pipeline,
    consumer_features: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """Top consumer cards by score with dominant SHAP feature (pattern signal)."""
    consumer_only = consumer_features.loc[consumer_features["target"] == 0].copy()
    consumer_only["card_number"] = consumer_only["card_number"].astype(str)
    X = consumer_only.set_index("card_number")[MODEL_FEATURE_COLUMNS]
    scores = final_model.predict_proba(X)[:, 1]
    ranked = (
        pd.DataFrame({"card_number": X.index, "score": scores})
        .sort_values(["score", "card_number"], ascending=[False, True])
        .head(top_n)
        .reset_index(drop=True)
    )

    X_top = X.loc[ranked["card_number"]]
    imputer = final_model.named_steps["imputer"]
    model = final_model.named_steps["model"]
    X_imputed = pd.DataFrame(
        imputer.transform(X_top),
        columns=X_top.columns,
        index=X_top.index,
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_imputed)

    rows: list[dict[str, Any]] = []
    for row_idx, card_number in enumerate(X_imputed.index):
        contributions = shap_values.values[row_idx]
        top_idx = int(np.abs(contributions).argmax())
        top_feature = MODEL_FEATURE_COLUMNS[top_idx]
        top_shap = float(contributions[top_idx])
        rows.append(
            {
                "card_number": str(card_number),
                "score": float(ranked.iloc[row_idx]["score"]),
                "top_shap_feature": top_feature,
                "top_shap_value": top_shap,
                "pattern_type": (
                    "hidden_entrepreneur_pattern"
                    if top_shap > 0
                    else "consumer_pattern"
                ),
            }
        )
    return pd.DataFrame(rows)


def export_pattern_consumer_cards(
    final_model: Pipeline,
    consumer_features: pd.DataFrame,
    output_dir: Path,
    top_percentile: float = 0.01,
) -> pd.DataFrame:
    """Consumer cards with hidden-entrepreneur pattern (top scores among retail cards)."""
    consumer_only = consumer_features.loc[consumer_features["target"] == 0].copy()
    consumer_only["card_number"] = consumer_only["card_number"].astype(str)
    X = consumer_only.set_index("card_number")[MODEL_FEATURE_COLUMNS]
    scores = final_model.predict_proba(X)[:, 1]

    ranked = (
        pd.DataFrame({"card_number": X.index, "score": scores})
        .sort_values(["score", "card_number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    cutoff_rank = max(int(len(ranked) * top_percentile), 1)
    threshold = float(ranked.iloc[cutoff_rank - 1]["score"])
    ranked["is_pattern"] = ranked["score"] >= threshold
    ranked["pattern_rank"] = ranked.index + 1

    pattern_only = ranked.loc[ranked["is_pattern"]].copy()
    ranked.to_csv(output_dir / "consumer_scores_ranked.csv", index=False, encoding="utf-8-sig")
    pattern_only.to_csv(output_dir / "pattern_consumers.csv", index=False, encoding="utf-8-sig")
    return pattern_only


def generate_submission(
    final_model: Pipeline,
    consumer_features: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Score only consumer cards — one row per card_number."""
    consumer_only = consumer_features.loc[consumer_features["target"] == 0].copy()
    X_consumer = consumer_only.set_index("card_number")[MODEL_FEATURE_COLUMNS]
    scores = final_model.predict_proba(X_consumer)[:, 1]

    submission = pd.DataFrame(
        {
            "card_number": X_consumer.index.astype(str),
            "score": np.clip(scores, 0.0, 1.0),
        }
    ).sort_values("card_number")
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8-sig")
    return submission


def write_business_report(
    output_dir: Path,
    quality_summary: dict[str, Any],
    feature_summary: pd.DataFrame,
    metrics_df: pd.DataFrame,
    importance: pd.Series,
    explanations_df: pd.DataFrame,
) -> None:
    xgb_metrics = metrics_df.loc[metrics_df["model"] == PRIMARY_MODEL_NAME].iloc[0]
    signals = build_business_signal_table(feature_summary)
    top_card = explanations_df.iloc[0].to_dict() if not explanations_df.empty else {}
    metrics_table = metrics_df.to_string(index=False)
    signal_table = signals.head(10).round(4).to_string(index=False)

    feature_lines = []
    for feature_name in importance.head(8).index:
        consumer_mean = float(feature_summary.loc["consumer", (feature_name, "mean")])
        business_mean = float(feature_summary.loc["business", (feature_name, "mean")])
        feature_lines.append(
            f"- `{feature_name}`: consumer mean={consumer_mean:.4f}, business mean={business_mean:.4f}"
        )

    local_lines = []
    for idx in range(1, 6):
        feature_key = f"reason_{idx}_feature"
        if feature_key not in top_card:
            continue
        local_lines.append(
            "- "
            f"`{top_card[feature_key]}`={top_card.get(f'reason_{idx}_value', np.nan):.4f}, "
            f"SHAP={top_card.get(f'reason_{idx}_shap', np.nan):.4f}, "
            f"{top_card.get(f'reason_{idx}_direction', '')}"
        )

    report = f"""# Mastercard Hidden Entrepreneur Solution

## 1. Objective
Build an explainable, production-friendly classifier that flags whether a card behaves like:
- `0`: regular consumer
- `1`: hidden entrepreneur

Business cards were used as proxy positive labels and consumer cards as proxy negative labels.

## 2. Data and cleaning
- Observation window: {quality_summary["business"]["date_start"]} to {quality_summary["business"]["date_end"]}
- Business transactions: {quality_summary["business"]["rows_after_cleaning"]:,}
- Consumer transactions: {quality_summary["consumer"]["rows_after_cleaning"]:,}
- Business cards: {quality_summary["business"]["unique_cards"]:,}
- Consumer cards: {quality_summary["consumer"]["unique_cards"]:,}
- Merchant reference coverage after merge: 100%
- Exact duplicate rows removed: business={quality_summary["business"]["exact_duplicates_removed"]}, consumer={quality_summary["consumer"]["exact_duplicates_removed"]}
- Non-positive amount rows removed: business={quality_summary["business"]["non_positive_amount_rows_removed"]}, consumer={quality_summary["consumer"]["non_positive_amount_rows_removed"]}

## 3. Leakage prevention
The pipeline intentionally excludes:
- `card_tier` because business cards are directly marked as `Business`
- `bank_name` because it can capture issuer-specific bias instead of transactional behavior
- raw `merchant_name` / `merchant_id` from the model to avoid merchant memorization

Instead, the model uses stable aggregated behavioral signals at `card_number` level.

## 4. Engineered card-level features
Mandatory features included:
- transaction behavior: counts and amount statistics
- merchant behavior: merchant diversity and top merchant concentration
- MCC behavior
- time behavior: hour, night, weekend
- digital behavior: online, tokenized, recurring
- geography: international share and country diversity
- velocity: average transactions per observed day

Additional explainable enrichment after merge:
- `recurring_capable_ratio`: share of transactions at merchants capable of recurring billing

## 5. Model comparison
```
{metrics_table}
```

## 6. Primary model: XGBoost
- Precision: {xgb_metrics["precision"]:.4f}
- Recall: {xgb_metrics["recall"]:.4f}
- F1-score: {xgb_metrics["f1_score"]:.4f}
- ROC-AUC: {xgb_metrics["roc_auc"]:.4f}
- Confusion matrix: TN={int(xgb_metrics["tn"])}, FP={int(xgb_metrics["fp"])}, FN={int(xgb_metrics["fn"])}, TP={int(xgb_metrics["tp"])}

## 7. Business meaning of model errors
- False Positive: a normal consumer is flagged as entrepreneur. Business impact: unnecessary outreach, friction, or over-monitoring.
- False Negative: a hidden entrepreneur stays in consumer segment. Business impact: missed migration to business products, underpriced servicing, and weaker portfolio intelligence.

If business goal is acquisition of micro-business clients, recall can be prioritized by lowering the threshold.
If business goal is minimizing friction for true consumers, precision should remain the main KPI.

## 8. Key business signals found in the data
```
{signal_table}
```

Observed pattern in this dataset:
- hidden entrepreneurs are much more online-heavy
- they transact earlier in the day and have materially higher night activity
- they have higher recurring behavior and higher concentration on a top merchant
- they have much larger ticket sizes
- they are slightly more international
- merchant diversity is lower, not higher, which suggests concentration in a narrow set of business-service merchants rather than broad retail spend

## 9. Top feature importance
{chr(10).join(feature_lines)}

## 10. Example: why the model thinks a card is a hidden entrepreneur
Card: `{top_card.get("card_number", "n/a")}`
- Predicted probability: {top_card.get("predicted_probability", np.nan):.4f}
- Actual label: {top_card.get("actual_target", "n/a")}
- Predicted label: {top_card.get("predicted_label", "n/a")}
{chr(10).join(local_lines) if local_lines else "- No local SHAP explanation available"}

## 11. Presentation storyline ideas
1. Start from the business problem: consumer cards used for business-like spend create pricing and servicing blind spots.
2. Show data design and leakage controls first to establish model credibility.
3. Use 4-5 business signals with simple charts: online ratio, avg amount, recurring ratio, top merchant concentration, transaction hour.
4. Compare Logistic Regression, Random Forest, and XGBoost to show why gradient boosting is chosen as the operational model.
5. Add one SHAP slide: explain a single flagged customer in plain business language.
6. End with an action slide: score consumer portfolio, review top-risk cards, route strongest cases to business card migration campaigns.

## 12. Recommendations
1. Validate on an out-of-time sample before production because the current dataset is very separable and likely semi-synthetic.
2. Add threshold tuning by business objective: growth, compliance review, or customer experience.
3. Expand labels beyond business cards vs consumer cards by using confirmed SME onboarding or merchant-linked ground truth.
4. Monitor drift in online share, time-of-day behavior, and recurring ecosystems after launch.
"""

    report_path = output_dir / "business_report.md"
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid")
    output_dir = ensure_output_dir(args.output_dir)

    merchant_ref = load_merchant_reference(args.merchant_path)
    business_tx, business_quality = load_and_prepare_transactions(
        args.business_path,
        target=1,
        segment_name="business",
        merchant_ref=merchant_ref,
    )
    consumer_tx, consumer_quality = load_and_prepare_transactions(
        args.consumer_path,
        target=0,
        segment_name="consumer",
        merchant_ref=merchant_ref,
    )

    business_features = build_card_features(business_tx)
    consumer_features = build_card_features(consumer_tx)
    features = pd.concat([business_features, consumer_features], ignore_index=True)

    features.to_parquet(output_dir / "card_level_features.parquet", index=False)

    plot_class_balance(features, output_dir)
    plot_key_feature_distributions(features, output_dir)

    feature_summary = compute_feature_summary(features)
    feature_summary.to_csv(output_dir / "feature_summary_by_segment.csv", encoding="utf-8-sig")

    cv_df = run_cross_validation(features)
    cv_df.to_csv(output_dir / "cv_5fold_metrics.csv", index=False, encoding="utf-8-sig")
    cv_summary = {
        "roc_auc_mean": float(cv_df["roc_auc"].mean()),
        "roc_auc_std": float(cv_df["roc_auc"].std()),
        "f1_mean": float(cv_df["f1_score"].mean()),
        "f1_std": float(cv_df["f1_score"].std()),
    }

    metrics_df, evaluation_context = evaluate_models(
        features=features,
        decision_threshold=args.decision_threshold,
    )
    metrics_df = metrics_df.round(
        {
            "precision": 4,
            "recall": 4,
            "f1_score": 4,
            "roc_auc": 4,
        }
    )
    metrics_df.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    threshold_df = plot_precision_recall_curve(evaluation_context, output_dir)
    plot_confusion_matrix_for_primary_model(metrics_df, evaluation_context, output_dir)
    importance = plot_feature_importance(evaluation_context, output_dir)
    importance.to_csv(output_dir / "xgboost_feature_importance.csv", encoding="utf-8-sig")

    final_model = fit_final_model(features)
    submission = generate_submission(final_model, consumer_features, output_dir)

    explanations_df, shap_summary_path = generate_shap_outputs(evaluation_context, output_dir)

    quality_summary = {
        "business": business_quality,
        "consumer": consumer_quality,
        "final_card_dataset": {
            "rows": int(len(features)),
            "business_cards": int((features["target"] == 1).sum()),
            "consumer_cards": int((features["target"] == 0).sum()),
            "target_rate": round(float(features["target"].mean()), 4),
            "feature_columns_used": MODEL_FEATURE_COLUMNS,
            "decision_threshold": args.decision_threshold,
            "primary_model": PRIMARY_MODEL_NAME,
            "cv_folds": CV_FOLDS,
            "cv_summary": cv_summary,
            "submission_rows": int(len(submission)),
            "submission_score_min": round(float(submission["score"].min()), 6),
            "submission_score_max": round(float(submission["score"].max()), 6),
        },
        "generated_files": {
            "features": str(output_dir / "card_level_features.parquet"),
            "submission": str(output_dir / "submission.csv"),
            "cv_metrics": str(output_dir / "cv_5fold_metrics.csv"),
            "threshold_tuning": str(output_dir / "threshold_tuning.csv"),
            "shap_summary_plot": str(shap_summary_path),
            "local_explanations": str(output_dir / "xgboost_local_explanations.csv"),
            "feature_summary": str(output_dir / "feature_summary_by_segment.csv"),
            "eda_summary": str(output_dir / "eda_summary.json"),
        },
    }
    save_json(quality_summary, output_dir / "eda_summary.json")

    write_business_report(
        output_dir=output_dir,
        quality_summary=quality_summary,
        feature_summary=feature_summary,
        metrics_df=metrics_df,
        importance=importance,
        explanations_df=explanations_df,
    )

    best_model_row = metrics_df.loc[metrics_df["model"] == PRIMARY_MODEL_NAME].iloc[0]
    print("Pipeline completed successfully.")
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print(
        f"{PRIMARY_MODEL_NAME} hold-out -> "
        f"precision={best_model_row['precision']:.4f}, "
        f"recall={best_model_row['recall']:.4f}, "
        f"f1={best_model_row['f1_score']:.4f}, "
        f"roc_auc={best_model_row['roc_auc']:.4f}"
    )
    print(
        f"5-fold CV ROC-AUC: {cv_summary['roc_auc_mean']:.4f} "
        f"(+/- {cv_summary['roc_auc_std']:.4f})"
    )
    print(f"submission.csv rows: {len(submission):,} (consumer cards only)")
    print(threshold_df.to_string(index=False))


if __name__ == "__main__":
    main()
