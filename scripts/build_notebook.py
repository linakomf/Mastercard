"""Generate mastercard_hidden_entrepreneur.ipynb from pipeline modules."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


ROOT = Path(__file__).resolve().parent.parent

cells = [
    md(
        "# Mastercard Hidden Entrepreneur Detection\n\n"
        "Полный ML-pipeline: EDA → очистка → признаки → обучение "
        "(business=1, consumer=0) → 5-fold CV → метрики → SHAP → "
        "**submission.csv** (только consumer-карты)."
    ),
    code(
        """
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
if not (ROOT / "scripts").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from mastercard_hidden_entrepreneur_pipeline import (
    MODEL_FEATURE_COLUMNS,
    build_card_features,
    compute_feature_summary,
    evaluate_models,
    fit_final_model,
    generate_shap_outputs,
    generate_submission,
    load_and_prepare_transactions,
    load_merchant_reference,
    plot_class_balance,
    plot_confusion_matrix_for_primary_model,
    plot_feature_importance,
    plot_key_feature_distributions,
    plot_precision_recall_curve,
    run_cross_validation,
)

sns.set_theme(style="whitegrid")
%matplotlib inline

DOWNLOADS = Path.home() / "Downloads"

def pick_download(*names: str) -> Path:
    for name in names:
        path = DOWNLOADS / name
        if path.exists():
            return path
    return DOWNLOADS / names[0]

BUSINESS_PATH = pick_download("businass_csv.csv", "business_cards_MDQ.parquet")
CONSUMER_PATH = pick_download("cumsumer_csv.csv", "consumer_cards_MDQ.parquet")
MERCHANT_PATH = pick_download("merchants_ref.csv", "merchants_reference.parquet")
OUTPUT_DIR = ROOT / "mastercard_hidden_entrepreneur_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DECISION_THRESHOLD = 0.5

print("Business:", BUSINESS_PATH)
print("Consumer:", CONSUMER_PATH)
print("Merchant:", MERCHANT_PATH)
print("Feature count:", len(MODEL_FEATURE_COLUMNS))
"""
    ),
    md("## 1. Загрузка справочника мерчантов и транзакций"),
    code(
        """
merchant_ref = load_merchant_reference(str(MERCHANT_PATH))

business_tx, business_quality = load_and_prepare_transactions(
    str(BUSINESS_PATH), target=1, segment_name="business", merchant_ref=merchant_ref
)
consumer_tx, consumer_quality = load_and_prepare_transactions(
    str(CONSUMER_PATH), target=0, segment_name="consumer", merchant_ref=merchant_ref
)

print("Business cards:", business_tx["card_number"].nunique())
print("Consumer cards:", consumer_tx["card_number"].nunique())
print(json.dumps({"business": business_quality, "consumer": consumer_quality}, indent=2, default=str))
"""
    ),
    md("## 2. EDA"),
    code(
        """
business_features = build_card_features(business_tx)
consumer_features = build_card_features(consumer_tx)
features = pd.concat([business_features, consumer_features], ignore_index=True)

display(features["target"].value_counts().rename({0: "consumer (0)", 1: "business (1)"}))
display(features.groupby("segment_name")[MODEL_FEATURE_COLUMNS[:6]].mean().round(3))

plot_class_balance(features, OUTPUT_DIR)
plot_key_feature_distributions(features, OUTPUT_DIR)
feature_summary = compute_feature_summary(features)
feature_summary.to_csv(OUTPUT_DIR / "feature_summary_by_segment.csv", encoding="utf-8-sig")
feature_summary
"""
    ),
    md(
        "## 3. Feature engineering\n\n"
        "Добавлены `b2b_mcc_ratio`, `amount_cv`, `days_since_last_tx`, "
        "`business_hours_ratio` — чтобы снизить доминирование `avg_transaction_hour`."
    ),
    code('features[["card_number", "segment_name", "target"] + MODEL_FEATURE_COLUMNS].head(3)'),
    md(
        "## 4. Обучение и оценка\n\n"
        "- **Обучение:** business-карты = 1, consumer-карты = 0 (proxy labels)\n"
        "- **Сабмит:** модель обучается на размеченных картах, скоры — только consumer\n"
        "- **5-fold stratified CV** + hold-out 80/20"
    ),
    code(
        """
cv_df = run_cross_validation(features)
cv_df.to_csv(OUTPUT_DIR / "cv_5fold_metrics.csv", index=False, encoding="utf-8-sig")
print(cv_df)
print("ROC-AUC mean:", round(cv_df["roc_auc"].mean(), 4), "+/-", round(cv_df["roc_auc"].std(), 4))

metrics_df, evaluation_context = evaluate_models(features, DECISION_THRESHOLD)
metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
metrics_df
"""
    ),
    md("## 5. Подбор порога (Precision-Recall)"),
    code("threshold_df = plot_precision_recall_curve(evaluation_context, OUTPUT_DIR)\nthreshold_df"),
    md("## 6. Confusion matrix, feature importance, SHAP"),
    code(
        """
plot_confusion_matrix_for_primary_model(metrics_df, evaluation_context, OUTPUT_DIR)
importance = plot_feature_importance(evaluation_context, OUTPUT_DIR)
importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance.csv", encoding="utf-8-sig")
importance.head(12)
"""
    ),
    code(
        """
explanations_df, shap_path = generate_shap_outputs(evaluation_context, OUTPUT_DIR)
from IPython.display import Image, display

display(Image(filename=str(shap_path)))
explanations_df.head()
"""
    ),
    md("## 7. Финальная модель и submission.csv"),
    code(
        """
final_model = fit_final_model(features)
submission = generate_submission(final_model, consumer_features, OUTPUT_DIR)

assert submission.shape[1] == 2
assert list(submission.columns) == ["card_number", "score"]
assert submission["score"].between(0, 1).all()
assert len(submission) == consumer_features["card_number"].nunique()

submission.head(10)
"""
    ),
    code("submission.describe()"),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out_path = ROOT / "mastercard_hidden_entrepreneur.ipynb"
out_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out_path}")
