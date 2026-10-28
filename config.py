from __future__ import annotations

FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "tempo": "Tempo (BPM)",
    "acousticness": "Acousticness",
    "instrumentalness": "Instrumentalness",
    "valence": "Valence",
    "loudness": "Loudness (dB)",
    "speechiness": "Speechiness",
    "liveness": "Liveness",
}
FEATURE_LABELS_TR = {
    "danceability": "Dans edilebilirlik",
    "energy": "Enerji",
    "tempo": "Tempo (BPM)",
    "acousticness": "Akustiklik",
    "instrumentalness": "Enstrümantal içerik",
    "valence": "Valence",
    "loudness": "Ses yüksekliği (dB)",
    "speechiness": "Konuşma benzeri içerik",
    "liveness": "Canlı kayıt olasılığı",
}
TITLE_CANDIDATES = ["track_name", "track", "song_name", "title", "name"]
ARTIST_CANDIDATES = ["artists", "artist_name", "track_artist", "artist", "artist_s_name"]
GENRE_CANDIDATES = ["playlist_genre", "genre", "genres", "playlist_subgenre", "subgenre"]
