# ============================================================
# ML température flamme (CH9) ~ richesse (phi) + % vapeur
# Dataset : 1 ligne = 1 seconde
# Pré-requis colonnes Excel : Time, phi, steam_pct, T_CH9
# Validation : GroupKFold par palier pour éviter le data leakage
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# 1) Charger le fichier Excel
# -----------------------------
FILEPATH = "S044.xlsm"     # <-- à adapter
SHEET = 0                       # ou "NomDeFeuille"

df = pd.read_excel(FILEPATH, sheet_name=SHEET)

# Sécurité : vérifier colonnes
required = ["Time", "phi", "steam_pct", "T_CH9"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes: {missing} ; colonnes dispo: {list(df.columns)}")

# -----------------------------
# 2) Pré-traitement minimal
# -----------------------------
# Convertir Time en datetime si possible (sinon garder tel quel mais trier)
df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
df = df.sort_values("Time" if df["Time"].notna().any() else "Time").reset_index(drop=True)

# Nettoyage NA
df = df.dropna(subset=["phi", "steam_pct", "T_CH9"]).reset_index(drop=True)

# Forcer types numériques
df["phi"] = pd.to_numeric(df["phi"], errors="coerce")
df["steam_pct"] = pd.to_numeric(df["steam_pct"], errors="coerce")
df["T_CH9"] = pd.to_numeric(df["T_CH9"], errors="coerce")
df = df.dropna(subset=["phi", "steam_pct", "T_CH9"]).reset_index(drop=True)

# -----------------------------
# 3) Créer palier_id
#    Nouveau palier si phi ou steam_pct change (dans la séquence)
# -----------------------------
df["palier"] = pd.to_numeric(df["palier"], errors="coerce")
df = df.dropna(subset=["palier"]).reset_index(drop=True)
df["palier"] = df["palier"].astype(int)

groups = df["palier"]


# (Optionnel) vérifier la taille de chaque palier
# print(df.groupby("palier_id").size().describe())

# -----------------------------
# 4) Définir X, y, groups
# -----------------------------
X = df[["phi", "steam_pct"]]
y = df["T_CH9"]
groups = df["palier"]

# -----------------------------
# 5) Pipeline modèle
#    - StandardScaler : utile pour Ridge (échelle comparable) [web:43]
#    - Ridge : baseline robuste pour petit nombre de variables
# -----------------------------
model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("reg", Ridge(alpha=1.0, random_state=0))
])

# -----------------------------
# 6) Cross-validation par groupes (paliers)
#    GroupKFold garantit qu’un palier n’est jamais dans train et test [web:106]
# -----------------------------
n_groups = groups.nunique()
n_splits = min(5, n_groups)  # éviter plus de folds que de paliers

cv = GroupKFold(n_splits=n_splits)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_res = cross_validate(
    model, X, y,
    cv=cv,
    groups=groups,
    scoring=scoring,
    return_train_score=True
)

# Résultats (convertir scores négatifs en positifs pour MAE/RMSE)
mae_test = -cv_res["test_mae"]
rmse_test = -cv_res["test_rmse"]
r2_test = cv_res["test_r2"]

print(f"GroupKFold (n_splits={n_splits}, n_paliers={n_groups})")
print(f"MAE  test:  mean={mae_test.mean():.3f}  std={mae_test.std():.3f}")
print(f"RMSE test:  mean={rmse_test.mean():.3f} std={rmse_test.std():.3f}")
print(f"R2   test:  mean={r2_test.mean():.3f}   std={r2_test.std():.3f}")

# -----------------------------
# 7) Entraîner le modèle final sur toutes les données
# -----------------------------
model.fit(X, y)

# -----------------------------
# 8) Exemple d’utilisation (prédire une condition)
# -----------------------------
example = pd.DataFrame({"phi": [0.75], "steam_pct": [5.0]})
pred_T = model.predict(example)[0]
print(f"Prédiction T_CH9 pour phi=0.75, vapeur=5% : {pred_T:.2f}")





# -----------------------------
# 9) Sauvegarde optionnelle des prédictions sur le dataset
# -----------------------------
df["T_pred"] = model.predict(X)
df.to_csv("dataset_with_predictions_ridge.csv", index=False)

# ← AJOUTER ICI LES 2 LIGNES ERREURS
df["erreur"] = df["T_CH9"] - df["T_pred"]
df["erreur_pct"] = (df["erreur"] / df["T_CH9"]) * 100



# -----------------------------
# PRÉSENTATION : 4 GRAPHIQUES PRO
# -----------------------------

# 1. CALCULER LES ERREURS (OBLIGATOIRE)
df["erreur"] = df["T_CH9"] - df["T_pred"]
df["erreur_pct"] = (df["erreur"] / df["T_CH9"]) * 100

# 2. GRAPHIQUES DE PRÉSENTATION
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

fig = plt.figure(figsize=(16, 10))
fig.suptitle(f'Modèle Ridge: T_CH9 (R²={r2_test.mean():.3f}, MAE={mae_test.mean():.1f}°C)', 
             fontsize=18, fontweight='bold', y=0.98)

gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# G1 : Réelle vs Prédite
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(df["T_CH9"], df["T_pred"], alpha=0.4, s=15, color='#0284c7')
min_T, max_T = df["T_CH9"].min(), df["T_CH9"].max()
ax1.plot([min_T, max_T], [min_T, max_T], 'r--', lw=2, label='Parfaite')
ax1.set_xlabel('T mesurée (°C)', fontweight='bold')
ax1.set_ylabel('T prédite (°C)', fontweight='bold')
ax1.set_title(f'① Validation (R²={r2_test.mean():.3f})', fontweight='bold')
ax1.legend(); ax1.grid(True, alpha=0.3)

# G2 : Histogramme erreurs
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df["erreur_pct"], bins=50, color='#10b981', alpha=0.7, edgecolor='black')
ax2.axvline(df["erreur_pct"].mean(), color='red', linestyle='--', lw=2, 
           label=f'Moy: {df["erreur_pct"].mean():.1f}%')
ax2.set_xlabel('Erreur relative (%)', fontweight='bold')
ax2.set_ylabel('Fréquence', fontweight='bold')
ax2.set_title('② Distribution Erreurs', fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3, axis='y')

# G3 : Performance par richesse
ax3 = fig.add_subplot(gs[1, 0])
richesses = sorted(df["phi"].unique())
mae_list = [mean_absolute_error(df[df["phi"]==p]["T_CH9"], df[df["phi"]==p]["T_pred"]) for p in richesses]
rmse_list = [np.sqrt(mean_squared_error(df[df["phi"]==p]["T_CH9"], df[df["phi"]==p]["T_pred"])) for p in richesses]
x = np.arange(len(richesses))
width = 0.35
ax3.bar(x - width/2, mae_list, width, label='MAE', color='#0284c7', alpha=0.8)
ax3.bar(x + width/2, rmse_list, width, label='RMSE', color='#f97316', alpha=0.8)
ax3.set_ylabel('Erreur (°C)', fontweight='bold')
ax3.set_xlabel('Richesse φ', fontweight='bold')
ax3.set_title('③ Performance par Richesse', fontweight='bold')
ax3.set_xticks(x); ax3.set_xticklabels(richesses)
ax3.legend(); ax3.grid(True, alpha=0.3, axis='y')

# G4 : Effet vapeur
ax4 = fig.add_subplot(gs[1, 1])
colors = ['#ef4444', '#f97316', '#10b981', '#0284c7']
for i, phi in enumerate(richesses):
    subset = df[df["phi"] == phi]
    ax4.scatter(subset["steam_pct"], subset["T_CH9"], label=f'φ={phi}', 
               color=colors[i], alpha=0.6, s=30)
ax4.set_xlabel('Vapeur (%)', fontweight='bold')
ax4.set_ylabel('T_CH9 (°C)', fontweight='bold')
ax4.set_title('③ Effet Vapeur par φ', fontweight='bold')
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.savefig('presentation_ridge.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✅ PRÉSENTATION SAUVEGARDÉE : presentation_ridge.png")
