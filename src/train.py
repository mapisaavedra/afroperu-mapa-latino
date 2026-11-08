"""Paso 2: entrena y compara clasificadores de género latino."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from config import *
from style import setup

setup()
df = pd.read_csv(PROCESSED / "reference_latin.csv")
X, y = df[FEATURES], df.genre
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

models = {
    "Regresión logística (baseline)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=-1)
    m.fit(Xtr, ytr); p = m.predict(Xte)
    rows.append({"modelo": name, "f1_macro_cv": s.mean(), "f1_macro_cv_std": s.std(),
                 "accuracy_test": accuracy_score(yte, p), "f1_macro_test": f1_score(yte, p, average="macro")})
res = pd.DataFrame(rows).sort_values("f1_macro_cv", ascending=False)
RESULTS.mkdir(exist_ok=True); res.round(3).to_csv(RESULTS / "model_comparison.csv", index=False)
print(res.round(3).to_string(index=False))

best_name = res.iloc[0].modelo; best = models[best_name]
print("\nMejor:", best_name); rep = classification_report(yte, best.predict(Xte)); print(rep)
(RESULTS / "classification_report.txt").write_text(f"{best_name}\n\n{rep}")
joblib.dump({"model": best, "name": best_name}, RESULTS / "best_model.joblib")

# Matriz de confusión normalizada
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ConfusionMatrixDisplay.from_estimator(best, Xte, yte, normalize="true", cmap="Blues",
                                      values_format=".0%", ax=ax, colorbar=False)
ax.set_title(f"¿Con qué se confunde cada género?\n{best_name}, datos de prueba", loc="left")
ax.set_xlabel("Género predicho"); ax.set_ylabel("Género real")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
fig.tight_layout(); fig.savefig(FIGURES / "02_confusion_matrix.png"); plt.close(fig)

# Importancia por permutación
imp = permutation_importance(best, Xte, yte, n_repeats=8, random_state=SEED, n_jobs=-1, scoring="f1_macro")
order = np.argsort(imp.importances_mean)
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(np.array(FEATURES)[order], imp.importances_mean[order], color=PALETTE["salsa"], height=0.6)
ax.set_title("¿Qué características distinguen los géneros?\nCaída del F1 al desordenar cada una", loc="left")
ax.set_xlabel("Pérdida de F1 macro")
fig.tight_layout(); fig.savefig(FIGURES / "03_feature_importance.png"); plt.close(fig)
