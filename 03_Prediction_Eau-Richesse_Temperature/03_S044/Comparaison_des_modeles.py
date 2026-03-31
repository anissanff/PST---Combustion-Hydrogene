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
# 🏆 COMPARAISON 4 MODÈLES (ROBUSTE)
# -----------------------------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore scaler warning

print("\n" + "="*60)
print("🏆 COMPARAISON 4 MODÈLES")
print("="*60)

# 1) XGBoost (GradientBoosting)
print("1️⃣ XGBoost...")
model_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", GradientBoostingRegressor(n_estimators=100, random_state=0))
])
cv_res_xgb = cross_validate(model_xgb, X, y, cv=cv, groups=groups, scoring=scoring)
r2_xgb = cv_res_xgb["test_r2"].mean()
mae_xgb = -cv_res_xgb["test_mae"].mean()

# 2) Deep MLP
print("2️⃣ Deep MLP...")
model_mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=0, early_stopping=True))
])
cv_res_mlp = cross_validate(model_mlp, X, y, cv=cv, groups=groups, scoring=scoring)
r2_mlp = cv_res_mlp["test_r2"].mean()
mae_mlp = -cv_res_mlp["test_mae"].mean()

# 3) Gaussian Process (kernel simple + robuste)
print("3️⃣ Gaussian Process...")
kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(1.0, length_scale_bounds=(1e-2, 1e2))
model_gp = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=3, random_state=0))
])
cv_res_gp = cross_validate(model_gp, X, y, cv=cv, groups=groups, scoring=scoring, n_jobs=1)
r2_gp = cv_res_gp["test_r2"].mean()
mae_gp = -cv_res_gp["test_mae"].mean()

# 4) TABLEAU FINAL
print("\n📊 RÉSULTATS")
print("-" * 40)
print(f"{'Modèle':<12} {'R²':>7} {'MAE':>7}")
print("-" * 40)
print(f"{'Ridge':<12} {r2_test.mean():.3f}  {mae_test.mean():.1f}")
print(f"{'XGBoost':<12} {r2_xgb:.3f}    {mae_xgb:.1f}")
print(f"{'DeepMLP':<12} {r2_mlp:.3f}    {mae_mlp:.1f}")
print(f"{'GaussProc':<12} {r2_gp:.3f}    {mae_gp:.1f}")

# Gagnant
meilleurs = [("Ridge", r2_test.mean()), ("XGBoost", r2_xgb), 
             ("DeepMLP", r2_mlp), ("GaussProc", r2_gp)]
gagnant = max(meilleurs, key=lambda x: x[1])
print(f"\n🏆 GAGNANT : {gagnant[0]} (R² = {gagnant[1]:.3f})")
print("="*60)

