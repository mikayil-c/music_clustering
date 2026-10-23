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
        errors.append("Veri seti boş.")
    if len(numeric_features) < 2:
        errors.append("En az iki beklenen sayısal müzikal özellik bulunmalı.")

    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"{duplicate_rows} tamamen tekrarlayan satır bulundu.")

    missing_values = int(frame[numeric_features].isna().sum().sum())
    if missing_values:
        errors.append(f"Müzikal özelliklerde toplam {missing_values} eksik değer bulundu.")

    for feature in numeric_features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        invalid = int(values.isna().sum())
        non_finite = int((~np.isfinite(values.dropna())).sum())
        if invalid:
            errors.append(f"{feature}: {invalid} sayısal olmayan değer bulundu.")
        if non_finite:
            errors.append(f"{feature}: {non_finite} sonsuz değer bulundu.")
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
                errors.append(f"{feature}: [0, 1] aralığı dışında {out_of_range} değer bulundu.")
        elif feature == "tempo" and (values <= 0).any():
            errors.append("tempo: sıfır veya negatif değer bulundu.")

    print(f"Dosya: {path}")
    print(f"Satır: {len(frame):,} | Sütun: {len(frame.columns):,}")
    print(f"Kontrol edilen müzikal özellik: {', '.join(numeric_features)}")
    print(f"Tamamen tekrarlayan satır: {duplicate_rows}")
    if warnings:
        print("Uyarılar:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("Hatalar:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Sonuç: veri seti kümeleme için temel kontrollerden geçti.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Müzik CSV dosyası için değişmezlik kontrolü.")
    parser.add_argument("path", type=Path, help="Kontrol edilecek CSV dosyası")
    raise SystemExit(validate_dataset(parser.parse_args().path))
