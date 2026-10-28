from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_LABELS


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def validate_dataset(path: Path) -> int:
    frame = pd.read_csv(path, low_memory=False)
    numeric_features = [name for name in FEATURE_LABELS if name in frame.columns]
    errors: list[str] = []
    warnings: list[str] = []

    if frame.empty:
        errors.append("The dataset is empty.")
    if len(numeric_features) < 2:
        errors.append("At least two expected numerical musical features are required.")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} completely duplicated rows.")

    missing_values = int(frame[numeric_features].isna().sum().sum())
    if missing_values:
        errors.append(f"Found {missing_values} missing values across musical features.")

    for feature in numeric_features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        invalid = int(values.isna().sum())
        non_finite = int((~np.isfinite(values.dropna())).sum())
        if invalid:
            errors.append(f"{feature}: found {invalid} non-numeric values.")
        if non_finite:
            errors.append(f"{feature}: found {non_finite} infinite values.")
        if feature in {
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "speechiness",
        }:
            out_of_range = int(((values < 0) | (values > 1)).sum())
            if out_of_range:
                errors.append(f"{feature}: found {out_of_range} values outside the [0, 1] range.")
        elif feature == "tempo" and (values <= 0).any():
            errors.append("tempo: found zero or negative values.")

    print(f"File: {path}")
    print(f"Rows: {len(frame):,} | Columns: {len(frame.columns):,}")
    print(f"Checked musical features: {', '.join(numeric_features)}")
    print(f"Completely duplicated rows: {duplicate_rows}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Result: the dataset passed the basic clustering checks.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation checks for a music CSV dataset.")
    parser.add_argument("path", type=Path, help="Dataset to validate")
    raise SystemExit(validate_dataset(parser.parse_args().path))
