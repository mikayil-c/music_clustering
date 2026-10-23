from __future__ import annotations

import pandas as pd


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def readable_song_name(
    row: pd.Series, title_col: str | None, artist_col: str | None
) -> str:
    title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else "Adsız şarkı"
    artist = str(row[artist_col]).strip() if artist_col and pd.notna(row[artist_col]) else "Sanatçı bilinmiyor"
    return f"{title} — {artist}"
