from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def remove_invalid_tempo_rows(input_path: Path, output_path: Path) -> int:
    frame = pd.read_csv(input_path, low_memory=False)
    if "tempo" not in frame.columns:
        raise ValueError("CSV dosyasında 'tempo' sütunu bulunamadı.")

    tempo = pd.to_numeric(frame["tempo"], errors="coerce")
    invalid_tempo = tempo.isna() | (tempo <= 0)
    removed_count = int(invalid_tempo.sum())

    cleaned = frame.loc[~invalid_tempo].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Girdi dosyası: {input_path}")
    print(f"Çıktı dosyası: {output_path}")
    print(f"Başlangıç satır sayısı: {len(frame):,}")
    print(f"Silinen geçersiz tempo satırı: {removed_count:,}")
    print(f"Son satır sayısı: {len(cleaned):,}")
    return removed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tempo değeri sıfır, negatif veya sayısal olmayan satırları siler."
    )
    parser.add_argument("input", type=Path, help="Temizlenecek CSV dosyası")
    parser.add_argument("output", type=Path, help="Temizlenmiş CSV dosyasının yolu")
    args = parser.parse_args()
    remove_invalid_tempo_rows(args.input, args.output)
