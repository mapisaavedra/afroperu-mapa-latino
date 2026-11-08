"""Configuración compartida del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# Archivos de Kaggle (ver data/README.md)
TRACKS_114K = RAW / "spotify-tracks-dataset.csv"   # maharshipandya
TRACKS_1M = RAW / "tracks_features.csv"            # rodolfofigueroa

# Géneros latinos de referencia (1000 canciones c/u en el dataset de 114k)
GENRES = ["salsa", "tango", "samba", "reggaeton",
          "forro", "mpb", "pagode", "sertanejo"]

# Artistas afroperuanos verificados por álbum (ver notebook de verificación en README)
# Lucha Reyes NO se incluye: en el dataset es la cantante mexicana homónima.
AFROPERU = ["Susana Baca", "Novalima", "Perú Negro", "Chabuca Granda", "Eva Ayllón"]
# Grupo de control: pop peruano (debería caer lejos de lo afroperuano)
CONTROL = ["Gian Marco"]

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "mode", "duration_min"]

SEED = 42

# Paleta categórica (validada para daltonismo, orden fijo)
PALETTE = {
    "salsa": "#2a78d6", "tango": "#eb6834", "samba": "#1baf7a", "reggaeton": "#eda100",
    "forro": "#e87ba4", "mpb": "#008300", "pagode": "#6250d6", "sertanejo": "#e34948",
}
INK = "#1f1f1e"
MUTED = "#8a8980"
