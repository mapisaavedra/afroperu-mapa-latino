"""Paso 1: arma el set de entrenamiento (géneros latinos) y el set peruano."""
import ast
import pandas as pd
from config import *

def clean(df):
    df = df.copy()
    df["duration_min"] = df["duration_ms"] / 60000
    return df.dropna(subset=FEATURES)

def build_reference():
    df = pd.read_csv(TRACKS_114K)
    df = df[df.track_genre.isin(GENRES)]
    # Algunas canciones aparecen con más de un género: se descartan por ambiguas
    n_gen = df.groupby("track_id").track_genre.nunique()
    df = df[df.track_id.isin(n_gen[n_gen == 1].index)].drop_duplicates("track_id")
    df = df.rename(columns={"track_genre": "genre"})
    return clean(df)[["track_id", "track_name", "artists", "genre"] + FEATURES]

def build_peru():
    df = pd.read_csv(TRACKS_1M)
    df["artist_list"] = df.artists.apply(ast.literal_eval)
    def tag(lst):
        if any(a in AFROPERU for a in lst): return "afroperuano"
        if any(a in CONTROL for a in lst): return "pop peruano (control)"
        return None
    df["group"] = df.artist_list.apply(tag)
    df = df[df.group.notna()].copy()
    df["main_artist"] = df.artist_list.apply(lambda l: next(a for a in l if a in AFROPERU + CONTROL))
    df["is_remix"] = df.name.str.contains("remix", case=False)
    df = df.drop_duplicates(["name", "main_artist"]).rename(columns={"id": "track_id", "name": "track_name"})
    return clean(df)[["track_id", "track_name", "main_artist", "group", "is_remix", "year"] + FEATURES]

if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ref, peru = build_reference(), build_peru()
    ref.to_csv(PROCESSED / "reference_latin.csv", index=False)
    peru.to_csv(PROCESSED / "peru.csv", index=False)
    print("Referencia:", ref.genre.value_counts().to_dict())
    print("Perú:", peru.groupby(["group", "main_artist"]).size().to_dict())
