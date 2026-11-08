"""Paso 3: ¿dónde cae la música afroperuana en el mapa latino?"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import umap
from config import *
from style import setup

setup()
ref = pd.read_csv(PROCESSED / "reference_latin.csv")
peru = pd.read_csv(PROCESSED / "peru.csv")
afro = peru[peru.group == "afroperuano"]
bundle = joblib.load(RESULTS / "best_model.joblib"); model = bundle["model"]
scaler = StandardScaler().fit(ref[FEATURES])
Zr, Zp = scaler.transform(ref[FEATURES]), scaler.transform(peru[FEATURES])

# 1) Perfil de cada género vs. afroperuano (z-score de la mediana)
prof = ref.groupby("genre")[FEATURES].median().loc[GENRES]
prof.loc["AFROPERUANO"] = afro[FEATURES].median()
prof.loc["pop peruano (control)"] = peru[peru.group != "afroperuano"][FEATURES].median()
z = (prof - ref[FEATURES].mean()) / ref[FEATURES].std()
fig, ax = plt.subplots(figsize=(10, 5.2))
im = ax.imshow(z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=35, ha="right"); ax.set_yticks(range(len(z)), z.index)
ax.axhline(len(GENRES) - 0.5, color="white", lw=3); ax.grid(False)
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        ax.text(j, i, f"{z.values[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                color="white" if abs(z.values[i, j]) > 0.9 else INK)
ax.set_title("Cómo suena cada género (mediana, en desviaciones estándar)\nRojo = más que el promedio latino · Azul = menos", loc="left")
fig.colorbar(im, ax=ax, shrink=0.7, label="z-score")
fig.tight_layout(); fig.savefig(FIGURES / "01_genre_profiles.png"); plt.close(fig)

# 2) Mapa UMAP: se ajusta solo con la referencia y se proyecta lo peruano
# Se excluye "mode" (binaria): parte el mapa en dos islas mayor/menor que no aportan
UF = [i for i, f in enumerate(FEATURES) if f != "mode"]
reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=SEED).fit(Zr[:, UF])
Er, Ep = reducer.embedding_, reducer.transform(Zp[:, UF])
fig, ax = plt.subplots(figsize=(11.5, 8))
for g in GENRES:
    m = (ref.genre == g).values
    ax.scatter(Er[m, 0], Er[m, 1], s=5, alpha=0.35, color=PALETTE[g], label=g, linewidths=0)
def densest(P, bins=10):  # punto de mayor densidad (no la mediana, que cae entre géneros)
    H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=bins)
    i, j = np.unravel_index(H.argmax(), H.shape)
    return (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
for g in GENRES:  # etiqueta directa donde el género es más denso
    c = densest(Er[(ref.genre == g).values])
    ax.text(*c, g, fontsize=10, fontweight="bold", color=INK, ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=PALETTE[g], lw=1.5, alpha=0.9))
ma = (peru.group == "afroperuano").values
ax.scatter(Ep[ma, 0], Ep[ma, 1], s=60, marker="D", color=INK, edgecolor="white", lw=1.2, label="afroperuano", zorder=5)
ax.scatter(Ep[~ma, 0], Ep[~ma, 1], s=55, marker="X", color=MUTED, edgecolor="white", lw=1, label="pop peruano (control)", zorder=5)
ax.set_title("El mapa latino: 8 géneros y dónde caen las canciones afroperuanas\nUMAP sobre 10 características de audio de Spotify", loc="left")
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=2, fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "00_mapa_latino.png"); plt.close(fig)

# 3) ¿A qué género "suena" cada artista según el modelo?
proba = pd.DataFrame(model.predict_proba(peru[FEATURES]), columns=model.classes_)[GENRES]
proba["main_artist"] = peru.main_artist.values
by_artist = proba.groupby("main_artist")[GENRES].mean()
by_artist["n"] = peru.main_artist.value_counts()
by_artist = by_artist.loc[[a for a in AFROPERU + CONTROL if a in by_artist.index]]
by_artist.round(3).to_csv(RESULTS / "peru_genre_probabilities.csv")
lab = [f"{a}  (n={int(n)})" for a, n in zip(by_artist.index, by_artist.n)]
fig, ax = plt.subplots(figsize=(10, 4.6))
left = np.zeros(len(by_artist))
for g in GENRES:
    v = by_artist[g].values
    ax.barh(lab, v, left=left, color=PALETTE[g], label=g, height=0.62, edgecolor="white", lw=2)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > 0.09: ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    left += v
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}"); ax.grid(axis="y", visible=False)
ax.set_title("¿A qué género latino 'suena' cada artista según el modelo?\nProbabilidad promedio asignada a cada género", loc="left")
ax.legend(ncol=8, loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=8.5, handlelength=1)
fig.tight_layout(); fig.savefig(FIGURES / "04_peru_genre_probabilities.png"); plt.close(fig)

# 4) ¿Suena a algo propio? Distancia al vecino latino más cercano
nn = NearestNeighbors(n_neighbors=11).fit(Zr)
d_ref = nn.kneighbors(Zr)[0][:, 1:].mean(1)        # excluye a sí misma
d_peru = nn.kneighbors(Zp, n_neighbors=10)[0].mean(1)
pct = (d_peru[:, None] > d_ref[None, :]).mean(1)    # percentil de "rareza"
peru["rareza_percentil"] = pct
idx = nn.kneighbors(Zp, n_neighbors=10)[1]
peru["genero_vecinos"] = [ref.genre.values[i].tolist() and pd.Series(ref.genre.values[i]).mode()[0] for i in idx]
peru["genero_modelo"] = model.predict(peru[FEATURES])
peru.to_csv(RESULTS / "peru_tracks_detail.csv", index=False)
fig, ax = plt.subplots(figsize=(8.5, 4.5))
bins = np.linspace(0, np.quantile(np.r_[d_ref, d_peru], 0.995), 45)
ax.hist(d_ref, bins=bins, density=True, color=PALETTE["salsa"], alpha=0.55, label="canciones de los 8 géneros latinos")
ax.hist(d_peru[ma], bins=bins, density=True, color=INK, alpha=0.75, label="afroperuano")
ax.axvline(np.median(d_ref), color=PALETTE["salsa"], ls="--", lw=1.5); ax.axvline(np.median(d_peru[ma]), color=INK, ls="--", lw=1.5)
ax.set_xlabel("Distancia promedio a sus 10 vecinos más parecidos (más = más 'rara')"); ax.set_ylabel("Densidad")
ax.set_title("¿La música afroperuana tiene un sonido propio?\nQué tan lejos está de cualquier canción del mapa latino", loc="left")
ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / "05_rareza.png"); plt.close(fig)

summary = {
    "n_afroperuano": int(ma.sum()), "n_control": int((~ma).sum()),
    "dist_mediana_ref": float(np.median(d_ref)), "dist_mediana_afro": float(np.median(d_peru[ma])),
    "dist_mediana_control": float(np.median(d_peru[~ma])),
    "pct_afro_mas_raras_que_80pct_ref": float((pct[ma] > 0.8).mean()),
    "genero_modelo_afro": peru[ma].genero_modelo.value_counts(normalize=True).round(3).to_dict(),
    "genero_vecinos_afro": peru[ma].genero_vecinos.value_counts(normalize=True).round(3).to_dict(),
    "acuerdo_modelo_vecinos_afro": float((peru[ma].genero_modelo == peru[ma].genero_vecinos).mean()),
    "afro_por_remix": peru[ma].groupby("is_remix").genero_modelo.agg(lambda s: s.value_counts(normalize=True).round(2).to_dict()).to_dict(),
}
import json; (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(by_artist.round(2).to_string())
