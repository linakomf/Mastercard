from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mastercard_dashboard.config import FEATURE_LABELS, MODEL_FEATURE_COLUMNS, DashboardPaths

PATTERN_TOP_PERCENTILE = 0.01

RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def load_submission_scores(submission_path: str) -> pd.DataFrame:
    scores = pd.read_csv(submission_path)[["card_number", "score"]]
    scores["card_number"] = scores["card_number"].astype(str)
    return scores


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


def get_pattern_portfolio_stats(
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
    top_percentile: float = PATTERN_TOP_PERCENTILE,
) -> dict[str, int | float]:
    scored = model_bundle["scored_cards"].copy()
    scored["card_number"] = scored["card_number"].astype(str)
    consumers = scored.loc[scored["target"] == 0].copy()

    if paths.submission_path.exists():
        scores = load_submission_scores(str(paths.submission_path))
    else:
        scores = consumers[["card_number", "entrepreneur_probability"]].rename(
            columns={"entrepreneur_probability": "score"}
        )

    scores = scores.sort_values(["score", "card_number"], ascending=[False, True]).reset_index(drop=True)
    pattern_slots = max(int(len(scores) * top_percentile), 1)
    threshold = float(scores.iloc[pattern_slots - 1]["score"])
    pattern_ids = set(
        scores.loc[scores["score"] >= threshold, "card_number"].astype(str).tolist()
    )

    return {
        "total_cards": int(len(scored)),
        "total_consumer": int(len(consumers)),
        "total_business": int((scored["target"] == 1).sum()),
        "pattern_count": int(len(pattern_ids)),
        "regular_count": int(len(consumers) - len(pattern_ids)),
        "pattern_threshold": threshold,
        "pattern_card_ids": pattern_ids,
    }


def get_pattern_cards_ranked(
    model_bundle: dict[str, Any],
    paths: DashboardPaths,
) -> pd.DataFrame:
    stats = get_pattern_portfolio_stats(model_bundle, paths)
    pattern_ids = stats["pattern_card_ids"]

    if paths.submission_path.exists():
        scores = load_submission_scores(str(paths.submission_path))
    else:
        scores = (
            model_bundle["scored_cards"]
            .loc[model_bundle["scored_cards"]["target"] == 0, ["card_number", "entrepreneur_probability"]]
            .rename(columns={"entrepreneur_probability": "score"})
        )

    scores["card_number"] = scores["card_number"].astype(str)
    ranked = (
        scores.loc[scores["card_number"].isin(pattern_ids)]
        .sort_values(["score", "card_number"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranked.insert(0, "№", range(1, len(ranked) + 1))
    ranked["pattern_pct"] = (ranked["score"] * 100).round(2)
    return ranked


@st.cache_resource(show_spinner="Preparing dashboard insights...")
def train_model_bundle(card_features: pd.DataFrame) -> dict[str, Any]:
    missing = [column for column in MODEL_FEATURE_COLUMNS if column not in card_features.columns]
    if missing:
        raise ValueError(
            f"Missing model features: {missing}. Run scripts/mastercard_hidden_entrepreneur_pipeline.py first."
        )

    modeling_frame = card_features.copy().set_index("card_number")
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
    models = build_models(scale_pos_weight)

    metrics_rows: list[dict[str, Any]] = []
    test_predictions: dict[str, pd.DataFrame] = {}
    fitted_models: dict[str, Pipeline] = {}
    roc_curves: dict[str, pd.DataFrame] = {}

    for model_name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        cm = confusion_matrix(y_test, predictions)
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)

        fitted_models[model_name] = pipeline
        test_predictions[model_name] = pd.DataFrame(
            {
                "card_number": X_test.index,
                "actual_target": y_test.values,
                "predicted_probability": probabilities,
                "predicted_label": predictions,
            }
        )
        roc_curves[model_name] = pd.DataFrame(
            {
                "fpr": fpr,
                "tpr": tpr,
                "threshold": thresholds,
                "model": model_name,
            }
        )

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

    full_scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    final_xgb_pipeline = build_models(full_scale_pos_weight)["XGBoost"]
    final_xgb_pipeline.fit(X, y)

    full_probabilities = final_xgb_pipeline.predict_proba(X)[:, 1]
    scored_cards = card_features.copy()
    scored_cards["entrepreneur_probability"] = full_probabilities
    scored_cards["predicted_class"] = (scored_cards["entrepreneur_probability"] >= 0.5).astype(int)
    scored_cards["predicted_label"] = scored_cards["predicted_class"].map(
        {1: "Hidden entrepreneur", 0: "Consumer"}
    )

    final_model = final_xgb_pipeline.named_steps["model"]
    feature_importance = pd.Series(
        final_model.feature_importances_,
        index=MODEL_FEATURE_COLUMNS,
        name="importance",
    ).sort_values(ascending=False)

    xgb_holdout = test_predictions["XGBoost"].copy()
    xgb_confusion = confusion_matrix(
        xgb_holdout["actual_target"],
        xgb_holdout["predicted_label"],
    )

    return {
        "metrics_df": metrics_df,
        "test_predictions": test_predictions,
        "roc_curves": roc_curves,
        "fitted_models": fitted_models,
        "final_xgb_pipeline": final_xgb_pipeline,
        "feature_importance": feature_importance,
        "scored_cards": scored_cards,
        "xgb_confusion_matrix": xgb_confusion,
    }


def get_local_shap_explanation(
    model_bundle: dict[str, Any],
    card_row: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    xgb_pipeline = model_bundle["final_xgb_pipeline"]
    imputer = xgb_pipeline.named_steps["imputer"]
    model = xgb_pipeline.named_steps["model"]

    card_frame = card_row[MODEL_FEATURE_COLUMNS].copy()
    imputed = pd.DataFrame(
        imputer.transform(card_frame),
        columns=card_frame.columns,
        index=card_frame.index,
    )
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(imputed)
    values = shap_values.values[0]
    base_value = float(np.ravel(shap_values.base_values)[0])

    explanation = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "feature_value": imputed.iloc[0].values,
            "shap_value": values,
        }
    )
    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation["direction"] = np.where(
        explanation["shap_value"] >= 0,
        "Pushes the score toward hidden entrepreneur",
        "Pushes the score toward consumer",
    )
    explanation = explanation.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return explanation, base_value


def build_business_explanation(card_record: pd.Series, explanation: pd.DataFrame) -> list[str]:
    statements: list[str] = []

    if float(card_record.get("online_ratio", 0)) >= 0.7:
        statements.append(
            f"Online activity is very high at {card_record['online_ratio']:.1%}, which is consistent with digital-first business behavior."
        )
    if float(card_record.get("recurring_ratio", 0)) >= 0.08:
        statements.append(
            f"Recurring payments account for {card_record['recurring_ratio']:.1%} of transactions, which is consistent with subscriptions, ads, SaaS, or business tools."
        )
    if float(card_record.get("top_merchant_ratio", 0)) >= 0.3:
        statements.append(
            f"{card_record['top_merchant_ratio']:.1%} of transactions are concentrated with one top merchant, which looks more operational than household spending."
        )
    if float(card_record.get("avg_amount", 0)) >= 100_000:
        statements.append(
            f"The average ticket is {card_record['avg_amount']:,.0f} KZT, noticeably above a typical consumer-card level."
        )
    if float(card_record.get("weekend_activity_ratio", 0)) <= 0.2:
        statements.append(
            f"Weekend activity is low at {card_record['weekend_activity_ratio']:.1%}, which is closer to a weekday operating pattern."
        )

    top_reasons = explanation.head(3)
    for _, row in top_reasons.iterrows():
        if row["shap_value"] > 0:
            statements.append(
                f"{FEATURE_LABELS.get(row['feature'], row['feature'])} = {row['feature_value']:.3f} is a strong positive contributor to the hidden entrepreneur score."
            )

    unique_statements: list[str] = []
    seen: set[str] = set()
    for statement in statements:
        if statement not in seen:
            unique_statements.append(statement)
            seen.add(statement)
    return unique_statements[:6]
