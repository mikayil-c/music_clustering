from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


def fit_labels(
    X_scaled: np.ndarray, n_clusters: int
) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = model.fit_predict(X_scaled)
    cluster_ids = sorted(int(label) for label in np.unique(raw_labels))
    one_based = {old_id: new_id for new_id, old_id in enumerate(cluster_ids, start=1)}
    labels = np.array(
        [one_based[int(label)] for label in raw_labels], dtype=int
    )
    return labels, float(time.perf_counter() - start), float(model.inertia_)


fit_labels = st.cache_data(show_spinner=False)(fit_labels)


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_values: range,
    reference_labels: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in k_values:
        labels, elapsed, inertia = fit_labels(X_scaled, k)
        metrics = calculate_metrics(X_scaled, labels)
        external = calculate_external_metrics(labels, reference_labels)
        rows.append(
            {
                "K değeri": k,
                "WCSS (Inertia)": inertia,
                "Silhouette Score": metrics["silhouette"],
                "Davies–Bouldin Index": metrics["davies_bouldin"],
                "Calinski–Harabasz Score": metrics["calinski_harabasz"],
                "Süre (sn)": elapsed,
                "Adjusted Rand Index": external["adjusted_rand"],
                "Normalized Mutual Information": external["normalized_mutual_info"],
                "Homogeneity": external["homogeneity"],
                "Completeness": external["completeness"],
                "V-measure": external["v_measure"],
            }
        )
    return pd.DataFrame(rows)


evaluate_k_range = st.cache_data(show_spinner=False)(evaluate_k_range)


def calculate_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    distinct = np.unique(labels)
    result: dict[str, Any] = {
        "cluster_count": len(distinct),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if len(X_scaled) >= 3 and 2 <= len(distinct) < len(X_scaled):
        try:
            result["silhouette"] = float(silhouette_score(
                X_scaled, labels, sample_size=min(1000, len(X_scaled)), random_state=42
            ))
            result["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
        except (ValueError, FloatingPointError):
            pass
    return result


def calculate_external_metrics(
    labels: np.ndarray, reference_labels: np.ndarray | None
) -> dict[str, float]:
    metric_names = (
        "adjusted_rand",
        "normalized_mutual_info",
        "homogeneity",
        "completeness",
        "v_measure",
    )
    result = {name: np.nan for name in metric_names}
    if reference_labels is None:
        return result

    predicted = np.asarray(labels)
    reference = np.asarray(reference_labels)
    if len(predicted) != len(reference):
        raise ValueError("Küme etiketleri ile dış doğrulama etiketlerinin uzunlukları eşleşmiyor.")
    valid = pd.notna(reference)
    if valid.sum() < 2 or len(np.unique(reference[valid])) < 2:
        return result

    predicted = predicted[valid]
    reference = reference[valid]
    result["adjusted_rand"] = float(adjusted_rand_score(reference, predicted))
    result["normalized_mutual_info"] = float(
        normalized_mutual_info_score(reference, predicted)
    )
    result["homogeneity"] = float(homogeneity_score(reference, predicted))
    result["completeness"] = float(completeness_score(reference, predicted))
    result["v_measure"] = float(v_measure_score(reference, predicted))
    return result


def make_cluster_names(
    X_scaled: np.ndarray, feature_cols: list[str], labels: np.ndarray
) -> dict[int, str]:
    z_df = pd.DataFrame(X_scaled, columns=feature_cols)
    z_df["_cluster"] = np.asarray(labels)
    means = z_df.groupby("_cluster").mean(numeric_only=True)
    names: dict[int, str] = {}
    for label in sorted({int(x) for x in labels}):
        if label in means.index:
            cluster_means = means.loc[label].dropna()
            strongest = cluster_means.abs().sort_values(ascending=False).head(2)
            descriptors = [
                f"{feature} {'görece yüksek' if cluster_means[feature] > 0 else 'görece düşük'}"
                for feature in strongest.index
            ]
        else:
            descriptors = []
        suffix = " · ".join(descriptors[:2])
        names[label] = (
            f"Küme {label}"
            + (f" — {suffix}" if suffix else " — belirgin tek profil yok")
        )
    return names


def safe_metric(value: Any, digits: int = 3) -> str:
    try:
        if pd.isna(value) or not np.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"
