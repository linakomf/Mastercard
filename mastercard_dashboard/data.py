from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds
import streamlit as st

from mastercard_dashboard.config import (
    DashboardPaths,
    MERCHANT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TRANSACTION_COLUMNS,
)


def detect_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def read_small_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    file_format = detect_format(path)
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if file_format == "parquet":
        if "usecols" in clean_kwargs:
            clean_kwargs["columns"] = clean_kwargs.pop("usecols")
        return pd.read_parquet(path, **clean_kwargs)
    if file_format == "csv":
        clean_kwargs.pop("columns", None)
        return pd.read_csv(path, **clean_kwargs)
    raise ValueError(f"Unsupported file format: {path}")


@st.cache_data(show_spinner=False)
def load_merchant_reference(path: Path) -> pd.DataFrame:
    merchants = read_small_table(path, usecols=MERCHANT_COLUMNS if path.suffix.lower() == ".csv" else None)
    merchants = merchants[MERCHANT_COLUMNS].drop_duplicates(subset=["merchant_id"]).copy()
    merchants["merchant_id"] = merchants["merchant_id"].astype(str)
    merchants["mcc"] = merchants["mcc"].astype(str)
    merchants["merchant_name"] = merchants["merchant_name"].astype(str)
    merchants["merchant_country"] = merchants["merchant_country"].astype(str)
    merchants["recurring_capable"] = merchants["recurring_capable"].fillna(False).astype(bool)
    return merchants


@st.cache_data(show_spinner=False)
def load_eda_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_transaction_frame(
    frame: pd.DataFrame,
    target: int,
    segment_name: str,
    merchant_ref: pd.DataFrame,
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["target"] = target
    prepared["segment_name"] = segment_name
    prepared = prepared.drop_duplicates()
    prepared = prepared.loc[prepared["transaction_amount_kzt"] > 0].copy()

    prepared["transaction_date"] = pd.to_datetime(
        prepared["transaction_date"],
        errors="coerce",
        dayfirst=True,
    )
    prepared["transaction_timestamp"] = pd.to_datetime(
        prepared["transaction_timestamp"],
        errors="coerce",
        dayfirst=True,
    )
    prepared = prepared.dropna(subset=["transaction_timestamp", "card_number", "merchant_id"]).copy()

    prepared["card_number"] = prepared["card_number"].astype(str)
    prepared["merchant_id"] = prepared["merchant_id"].astype(str)
    prepared["mcc"] = prepared["mcc"].astype(str)
    prepared["channel"] = prepared["channel"].astype(str)
    prepared["bank_name"] = prepared["bank_name"].astype(str)
    prepared["country"] = prepared["country"].astype(str)
    prepared["card_tier"] = prepared["card_tier"].astype(str)
    prepared["tokenized"] = prepared["tokenized"].fillna(False).astype(bool)
    prepared["is_recurring"] = prepared["is_recurring"].fillna(False).astype(bool)

    prepared = prepared.merge(
        merchant_ref.rename(columns={"mcc": "merchant_ref_mcc"}),
        on="merchant_id",
        how="left",
    )
    prepared["recurring_capable"] = prepared["recurring_capable"].fillna(False).astype(bool)

    prepared["tx_hour"] = prepared["transaction_timestamp"].dt.hour.astype("int16")
    prepared["tx_day"] = prepared["transaction_timestamp"].dt.floor("D")
    prepared["is_weekend"] = (prepared["transaction_timestamp"].dt.dayofweek >= 5).astype("int8")
    prepared["is_night"] = prepared["tx_hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")
    prepared["is_online"] = prepared["channel"].str.lower().eq("online").astype("int8")
    prepared["is_international"] = (
        ~prepared["country"].str.lower().eq("kazakhstan")
    ).astype("int8")
    return prepared


def build_card_features(transactions: pd.DataFrame) -> pd.DataFrame:
    observed_days = (
        transactions.groupby("card_number")["tx_day"]
        .agg(["min", "max"])
        .rename(columns={"min": "first_tx_day", "max": "last_tx_day"})
    )
    observed_days["observed_days"] = (
        (observed_days["last_tx_day"] - observed_days["first_tx_day"]).dt.days + 1
    ).clip(lower=1)

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
    )

    top_merchant = (
        transactions.groupby(["card_number", "merchant_id"])
        .size()
        .rename("merchant_txn_count")
        .reset_index()
        .groupby("card_number")["merchant_txn_count"]
        .max()
        .rename("top_merchant_txn_count")
    )

    features = grouped.join(observed_days["observed_days"]).join(top_merchant)
    features["std_amount"] = features["std_amount"].fillna(0.0)
    features["top_merchant_ratio"] = (
        features["top_merchant_txn_count"] / features["txn_count"].clip(lower=1)
    )
    features["avg_txn_per_day"] = features["txn_count"] / features["observed_days"].clip(lower=1)
    return features.drop(columns=["top_merchant_txn_count"]).reset_index()


@st.cache_data(show_spinner="Building card-level features...")
def build_card_features_from_raw(paths: DashboardPaths) -> pd.DataFrame:
    merchant_ref = load_merchant_reference(paths.merchant_path)

    business = read_small_table(
        paths.business_path,
        columns=TRANSACTION_COLUMNS if paths.business_path.suffix.lower() == ".parquet" else None,
        usecols=TRANSACTION_COLUMNS if paths.business_path.suffix.lower() == ".csv" else None,
        low_memory=False if paths.business_path.suffix.lower() == ".csv" else None,
    )
    consumer = read_small_table(
        paths.consumer_path,
        columns=TRANSACTION_COLUMNS if paths.consumer_path.suffix.lower() == ".parquet" else None,
        usecols=TRANSACTION_COLUMNS if paths.consumer_path.suffix.lower() == ".csv" else None,
        low_memory=False if paths.consumer_path.suffix.lower() == ".csv" else None,
    )

    business_prepared = prepare_transaction_frame(business, 1, "business", merchant_ref)
    consumer_prepared = prepare_transaction_frame(consumer, 0, "consumer", merchant_ref)
    features = pd.concat(
        [build_card_features(business_prepared), build_card_features(consumer_prepared)],
        ignore_index=True,
    )
    return features


@st.cache_data(show_spinner="Loading dashboard data...")
def load_dashboard_data(paths: DashboardPaths) -> dict[str, Any]:
    merchant_ref = load_merchant_reference(paths.merchant_path)
    eda_summary = load_eda_summary(paths.eda_summary_path)

    if paths.card_features_path.exists():
        card_features = pd.read_parquet(paths.card_features_path)
    else:
        card_features = build_card_features_from_raw(paths)

    card_features = card_features.copy()
    card_features["card_number"] = card_features["card_number"].astype(str)
    if "segment_name" not in card_features.columns:
        card_features["segment_name"] = card_features["target"].map({1: "business", 0: "consumer"})

    if paths.feature_summary_path.exists():
        feature_summary = pd.read_csv(paths.feature_summary_path, header=[0, 1], index_col=0)
    else:
        feature_summary = (
            card_features.groupby("target")[MODEL_FEATURE_COLUMNS]
            .agg(["mean", "median"])
            .round(4)
        )
        feature_summary.index = ["consumer", "business"]

    overview = {
        "total_cards": int(len(card_features)),
        "business_cards": int((card_features["target"] == 1).sum()),
        "consumer_cards": int((card_features["target"] == 0).sum()),
        "target_rate": float(card_features["target"].mean()),
        "total_transactions": int(
            eda_summary.get("business", {}).get("rows_after_cleaning", 0)
            + eda_summary.get("consumer", {}).get("rows_after_cleaning", 0)
        ),
    }
    return {
        "card_features": card_features,
        "feature_summary": feature_summary,
        "merchant_ref": merchant_ref,
        "eda_summary": eda_summary,
        "overview": overview,
    }


@st.cache_resource(show_spinner=False)
def get_parquet_transaction_dataset(paths: DashboardPaths) -> ds.Dataset:
    return ds.dataset(
        [str(paths.business_path), str(paths.consumer_path)],
        format="parquet",
    )


def apply_search(df: pd.DataFrame, search_text: str) -> pd.DataFrame:
    if not search_text:
        return df

    lowered = search_text.lower()
    mask = (
        df["card_number"].astype(str).str.lower().str.contains(lowered, na=False)
        | df["merchant_name"].astype(str).str.lower().str.contains(lowered, na=False)
        | df["merchant_id"].astype(str).str.lower().str.contains(lowered, na=False)
        | df["mcc"].astype(str).str.lower().str.contains(lowered, na=False)
        | df["country"].astype(str).str.lower().str.contains(lowered, na=False)
    )
    return df.loc[mask].copy()


def finalize_transaction_result(
    raw_transactions: pd.DataFrame,
    merchant_ref: pd.DataFrame,
    search_text: str,
    limit: int,
) -> pd.DataFrame:
    if raw_transactions.empty:
        return raw_transactions

    result = raw_transactions.merge(
        merchant_ref[["merchant_id", "merchant_name", "merchant_country", "recurring_capable"]],
        on="merchant_id",
        how="left",
    )
    result["segment_name"] = result["card_tier"].eq("Business").map(
        {True: "business", False: "consumer"}
    )
    result = apply_search(result, search_text)
    result["transaction_timestamp"] = pd.to_datetime(result["transaction_timestamp"], errors="coerce")
    result = result.sort_values("transaction_timestamp", ascending=False).head(limit).copy()
    result["is_online"] = result["channel"].str.lower().eq("online")
    return result


def _query_transactions_parquet(
    paths: DashboardPaths,
    merchant_ref: pd.DataFrame,
    card_number: str,
    search_text: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    mccs: list[str],
    merchant_names: list[str],
    channels: list[str],
    recurring_only: str,
    limit: int,
) -> pd.DataFrame:
    dataset = get_parquet_transaction_dataset(paths)
    filter_expression = None

    def merge_expression(current: Any, new_expr: Any) -> Any:
        return new_expr if current is None else current & new_expr

    if card_number:
        filter_expression = merge_expression(filter_expression, ds.field("card_number") == card_number)
    if start_date is not None:
        filter_expression = merge_expression(
            filter_expression,
            ds.field("transaction_timestamp") >= start_date.to_pydatetime(),
        )
    if end_date is not None:
        filter_expression = merge_expression(
            filter_expression,
            ds.field("transaction_timestamp") <= end_date.to_pydatetime(),
        )
    if mccs:
        filter_expression = merge_expression(filter_expression, ds.field("mcc").isin([str(v) for v in mccs]))
    if merchant_names:
        merchant_ids = merchant_ref.loc[
            merchant_ref["merchant_name"].isin(merchant_names), "merchant_id"
        ].astype(str).tolist()
        if merchant_ids:
            filter_expression = merge_expression(filter_expression, ds.field("merchant_id").isin(merchant_ids))
    if channels:
        filter_expression = merge_expression(filter_expression, ds.field("channel").isin(channels))
    if recurring_only == "Only recurring":
        filter_expression = merge_expression(filter_expression, ds.field("is_recurring") == True)
    if recurring_only == "Only non-recurring":
        filter_expression = merge_expression(filter_expression, ds.field("is_recurring") == False)

    table = dataset.to_table(
        columns=TRANSACTION_COLUMNS,
        filter=filter_expression,
    )
    raw_transactions = table.to_pandas()
    return finalize_transaction_result(raw_transactions, merchant_ref, search_text, limit)


def _filter_csv_chunk(
    chunk: pd.DataFrame,
    merchant_ref: pd.DataFrame,
    card_number: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    mccs: list[str],
    merchant_names: list[str],
    channels: list[str],
    recurring_only: str,
) -> pd.DataFrame:
    filtered = chunk.copy()
    filtered["transaction_timestamp"] = pd.to_datetime(
        filtered["transaction_timestamp"],
        errors="coerce",
        dayfirst=True,
    )
    filtered = filtered.dropna(subset=["transaction_timestamp"]).copy()
    filtered["merchant_id"] = filtered["merchant_id"].astype(str)
    filtered["card_number"] = filtered["card_number"].astype(str)
    filtered["mcc"] = filtered["mcc"].astype(str)

    if card_number:
        filtered = filtered.loc[filtered["card_number"] == card_number]
    if start_date is not None:
        filtered = filtered.loc[filtered["transaction_timestamp"] >= start_date]
    if end_date is not None:
        filtered = filtered.loc[filtered["transaction_timestamp"] <= end_date]
    if mccs:
        filtered = filtered.loc[filtered["mcc"].isin([str(v) for v in mccs])]
    if merchant_names:
        merchant_ids = merchant_ref.loc[
            merchant_ref["merchant_name"].isin(merchant_names), "merchant_id"
        ].astype(str)
        filtered = filtered.loc[filtered["merchant_id"].isin(merchant_ids)]
    if channels:
        filtered = filtered.loc[filtered["channel"].isin(channels)]
    if recurring_only == "Only recurring":
        filtered = filtered.loc[filtered["is_recurring"].astype(bool)]
    if recurring_only == "Only non-recurring":
        filtered = filtered.loc[~filtered["is_recurring"].astype(bool)]
    return filtered


def _query_transactions_csv(
    paths: DashboardPaths,
    merchant_ref: pd.DataFrame,
    card_number: str,
    search_text: str,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    mccs: list[str],
    merchant_names: list[str],
    channels: list[str],
    recurring_only: str,
    limit: int,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    chunk_size = 200_000

    for source_path in [paths.business_path, paths.consumer_path]:
        for chunk in pd.read_csv(
            source_path,
            usecols=TRANSACTION_COLUMNS,
            chunksize=chunk_size,
            low_memory=False,
        ):
            filtered = _filter_csv_chunk(
                chunk=chunk,
                merchant_ref=merchant_ref,
                card_number=card_number,
                start_date=start_date,
                end_date=end_date,
                mccs=mccs,
                merchant_names=merchant_names,
                channels=channels,
                recurring_only=recurring_only,
            )
            if not filtered.empty:
                results.append(filtered)

    raw_transactions = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=TRANSACTION_COLUMNS)
    return finalize_transaction_result(raw_transactions, merchant_ref, search_text, limit)


def query_transactions(
    paths: DashboardPaths,
    merchant_ref: pd.DataFrame,
    card_number: str = "",
    search_text: str = "",
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    mccs: list[str] | None = None,
    merchant_names: list[str] | None = None,
    channels: list[str] | None = None,
    recurring_only: str = "All",
    limit: int = 1000,
) -> pd.DataFrame:
    mccs = mccs or []
    merchant_names = merchant_names or []
    channels = channels or []

    if detect_format(paths.business_path) == "parquet" and detect_format(paths.consumer_path) == "parquet":
        return _query_transactions_parquet(
            paths=paths,
            merchant_ref=merchant_ref,
            card_number=card_number,
            search_text=search_text,
            start_date=start_date,
            end_date=end_date,
            mccs=mccs,
            merchant_names=merchant_names,
            channels=channels,
            recurring_only=recurring_only,
            limit=limit,
        )

    return _query_transactions_csv(
        paths=paths,
        merchant_ref=merchant_ref,
        card_number=card_number,
        search_text=search_text,
        start_date=start_date,
        end_date=end_date,
        mccs=mccs,
        merchant_names=merchant_names,
        channels=channels,
        recurring_only=recurring_only,
        limit=limit,
    )
