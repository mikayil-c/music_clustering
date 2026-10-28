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
        raise ValueError("The CSV file does not contain a 'tempo' column.")

    tempo = pd.to_numeric(frame["tempo"], errors="coerce")
    invalid_tempo = tempo.isna() | (tempo <= 0)
    removed_count = int(invalid_tempo.sum())

    cleaned = frame.loc[~invalid_tempo].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Initial row count: {len(frame):,}")
    print(f"Removed invalid tempo rows: {removed_count:,}")
    print(f"Final row count: {len(cleaned):,}")
    return removed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove rows with zero, negative, or non-numeric tempo values."
    )
    parser.add_argument("input", type=Path, help="CSV file to clean")
    parser.add_argument("output", type=Path, help="Path for the cleaned CSV file")
    args = parser.parse_args()
    remove_invalid_tempo_rows(args.input, args.output)
