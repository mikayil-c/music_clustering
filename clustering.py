from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
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


def _validate_clustering_input(X_scaled: np.ndarray, n_clusters: int) -> np.ndarray:
    X = np.asarray(X_scaled, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X_scaled must be a non-empty two-dimensional array.")
    if not np.isfinite(X).all():
        raise ValueError("X_scaled must contain only finite values.")
    if not 1 <= n_clusters <= len(X):
        raise ValueError("n_clusters must be between 1 and the number of samples.")
    return X


def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.maximum(distances, 0.0)


def _initialize_medoids(X: np.ndarray, n_clusters: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    medoid_indices = [int(rng.integers(len(X)))]
    min_distances = _squared_distances(X, X[medoid_indices]).ravel()
    for _ in range(1, n_clusters):
        next_index = int(np.argmax(min_distances))
        medoid_indices.append(next_index)
        new_distances = _squared_distances(X, X[[next_index]]).ravel()
        min_distances = np.minimum(min_distances, new_distances)
    return np.asarray(medoid_indices, dtype=int)


def _fit_kmedoids(
    X_scaled: np.ndarray, n_clusters: int, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray, float]:
    X = _validate_clustering_input(X_scaled, n_clusters)
    medoid_indices = _initialize_medoids(X, n_clusters)

    for _ in range(max_iter):
        distances = _squared_distances(X, X[medoid_indices])
        labels = np.argmin(distances, axis=1)
        updated_indices = medoid_indices.copy()
        used_indices: set[int] = set()

        for cluster_id in range(n_clusters):
            members = np.flatnonzero(labels == cluster_id)
            if len(members) == 0:
                nearest_distance = np.min(distances, axis=1)
                candidates = np.argsort(nearest_distance)[::-1]
            else:
                cluster_center = X[members].mean(axis=0)
                candidate_order = np.argsort(
                    _squared_distances(X[members], cluster_center[None, :]).ravel()
                )
                candidates = members[candidate_order]
            selected = next(
                (int(index) for index in candidates if int(index) not in used_indices),
                int(candidates[0]),
            )
            updated_indices[cluster_id] = selected
            used_indices.add(selected)

        if np.array_equal(updated_indices, medoid_indices):
            break
        medoid_indices = updated_indices

    distances = _squared_distances(X, X[medoid_indices])
    labels = np.argmin(distances, axis=1)
    return labels, medoid_indices, float(np.min(distances, axis=1).sum())


def fit_labels(
    X_scaled: np.ndarray,
    n_clusters: int,
    algorithm: str = "kmeans",
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[np.ndarray, float, float]:
    start = time.perf_counter()
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        raw_labels = model.fit_predict(X_scaled)
        objective = float(model.inertia_)
    elif algorithm == "kmedoids":
        raw_labels, _, objective = _fit_kmedoids(X_scaled, n_clusters)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        raw_labels = model.fit_predict(X_scaled)
        objective = np.nan
    elif algorithm == "gmm":
        model = GaussianMixture(
            n_components=n_clusters, random_state=42, n_init=5
        )
        raw_labels = model.fit_predict(X_scaled)
        objective = float(-model.score(X_scaled) * len(X_scaled))
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
    cluster_ids = sorted(int(label) for label in np.unique(raw_labels) if label >= 0)
    one_based = {old_id: new_id for new_id, old_id in enumerate(cluster_ids, start=1)}
    labels = np.array(
        [0 if int(label) == -1 else one_based[int(label)] for label in raw_labels],
        dtype=int,
    )
    return labels, float(time.perf_counter() - start), objective


fit_labels = st.cache_data(show_spinner=False)(fit_labels)


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_values: range,
    reference_labels: np.ndarray | None = None,
    algorithm: str = "kmeans",
    eps: float = 0.5,
    min_samples: int = 5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in k_values:
        labels, elapsed, objective = fit_labels(
            X_scaled, k, algorithm, eps=eps, min_samples=min_samples
        )
        metrics = calculate_metrics(X_scaled, labels)
        external = calculate_external_metrics(labels, reference_labels)
        rows.append(
            {
                "K değeri": k,
                "Objective": objective,
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


def measure_algorithm(
    X_scaled: np.ndarray,
    feature_cols: list[str],
    n_clusters: int,
    reference_labels: np.ndarray | None,
    algorithm: str,
    n_clusters_max: int,
    eps: float = 0.5,
    min_samples: int = 5,
) -> tuple[
    np.ndarray,
    float,
    dict[str, Any],
    dict[str, float],
    dict[int, str],
    pd.DataFrame,
]:
    labels, elapsed, _ = fit_labels(
        X_scaled,
        n_clusters,
        algorithm,
        eps=eps,
        min_samples=min_samples,
    )
    metrics = calculate_metrics(X_scaled, labels)
    external_metrics = calculate_external_metrics(labels, reference_labels)
    cluster_names = make_cluster_names(X_scaled, feature_cols, labels)
    k_evaluation_df = (
        evaluate_k_range(
            X_scaled,
            range(1, n_clusters_max + 1),
            reference_labels,
            algorithm,
            eps=eps,
            min_samples=min_samples,
        )
        if algorithm != "dbscan"
        else pd.DataFrame()
    )
    return (
        labels,
        elapsed,
        metrics,
        external_metrics,
        cluster_names,
        k_evaluation_df,
    )


measure_algorithm = st.cache_data(show_spinner=False)(measure_algorithm)


def clear_measurement_caches() -> None:
    measure_algorithm.clear()
    fit_labels.clear()
    evaluate_k_range.clear()


def calculate_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels)
    valid = labels != 0
    scored_X = np.asarray(X_scaled)[valid]
    scored_labels = labels[valid]
    distinct = np.unique(scored_labels)
    result: dict[str, Any] = {
        "cluster_count": len(distinct),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if len(scored_X) >= 3 and 2 <= len(distinct) < len(scored_X):
        try:
            result["silhouette"] = float(silhouette_score(
                scored_X,
                scored_labels,
                sample_size=min(1000, len(scored_X)),
                random_state=42,
            ))
            result["davies_bouldin"] = float(
                davies_bouldin_score(scored_X, scored_labels)
            )
            result["calinski_harabasz"] = float(
                calinski_harabasz_score(scored_X, scored_labels)
            )
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
    valid = pd.notna(reference) & (predicted != 0)
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
        if label == 0:
            names[label] = "Gürültü — DBSCAN tarafından ayrıştırılan noktalar"
            continue
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
