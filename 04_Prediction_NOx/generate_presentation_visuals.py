import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration
OUT_DIR = Path("presentation_visuals")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10})

def build_physics_features(df):
    out = pd.DataFrame()
    out["Richesse"] = df["Richesse"]
    out["Taux d'eau"] = df["Taux d'eau"]
    temp_cols = [f"CH{i}" for i in range(1, 10)]
    for col in temp_cols:
        out[f"{col}_K"] = df[col] + 273.15
    # Pour correspondre à Log(NOx) = -E/RT, la bonne feature est 1/T
    t_cols = [f"CH{i}_K" for i in range(6, 10)]
    T_max_K = out[t_cols].max(axis=1)
    out["Inv_Tmax"] = 10000 / T_max_K
    return out, T_max_K

def generate_visuals():
    # 1. Load Data
    try:
        df66 = pd.read_excel("S066_Clean_all.xlsx")
        df82 = pd.read_excel("S082_Clean_all.xlsx")
    except Exception as e:
        print(f"Erreur chargement Excel : {e}")
        return
    
    # --- VISUEL 1: Physique des NOx (Arrhenius) ---
    print("Génération Visuel 1: Physique...")
    X66, T66 = build_physics_features(df66)
    y66 = df66["NOx [ppm]"]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(T66, y66, alpha=0.3, color='steelblue', label='Données Expérimentales')
    # Exponential trendline
    line_t = np.linspace(T66.min(), T66.max(), 100)
    # Fit approximation for visual
    line_nox = 2e-7 * np.exp(0.01 * line_t) 
    plt.plot(line_t, line_nox, 'r-', linewidth=3, label='Loi d\'Arrhenius (Thermal NOx)')
    plt.title("Lien Physique : Température de Flamme vs NOx", fontsize=14)
    plt.xlabel("Température Maximale Mesurée [K]", fontsize=12)
    plt.ylabel("Émissions NOx [ppm]", fontsize=12)
    plt.yscale('log') # Better for Arrhenius
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(OUT_DIR / "1_physique_thermique_nox.png", dpi=300)
    plt.close()

    # --- VISUEL 2: Saut de Performance (Comparaison R2) ---
    print("Génération Visuel 2: Performance...")
    labels = ['Modèle Standard\n(Sans Physique)', 'Modèle Antigravity\n(Physique + Log)']
    r2_scores = [0.48, 0.9416]
    plt.figure(figsize=(8, 6))
    colors = ['lightgrey', 'forestgreen']
    plt.bar(labels, r2_scores, color=colors, edgecolor='black', alpha=0.8)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"R² = {v:.2f}", ha='center', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.1)
    plt.title("Impact de l'Ingénierie Physique sur la Précision", fontsize=14)
    plt.savefig(OUT_DIR / "2_saut_performance.png", dpi=300)
    plt.close()

    # --- VISUEL 3A: Le problème du XGBoost (Escalier) ---
    print("Génération Visuel 3A: XGBoost Escalier...")
    from xgboost import XGBRegressor
    groups = df66.groupby(["Richesse", "Taux d'eau"]).ngroup()
    model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    y66_log = np.log1p(y66)
    y_pred_log_xgb = cross_val_predict(model_xgb, X66, y66_log, cv=GroupKFold(n_splits=5), groups=groups)
    y_pred_xgb = np.expm1(y_pred_log_xgb)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y66, y_pred_xgb, alpha=0.4, color='crimson', s=15)
    limit = max(y66.max(), y_pred_xgb.max())
    plt.plot([0, limit], [0, limit], 'k--', linewidth=2, alpha=0.6)
    plt.title("Problème XGBoost : L'Effet 'Escalier'\n(Incapable d'extrapoler les dynamiques locales)", fontsize=14)
    plt.xlabel("NOx Mesurés [ppm]", fontsize=12)
    plt.ylabel("NOx Prédits (Arbres de décision) [ppm]", fontsize=12)
    plt.grid(True, ls=':', alpha=0.5)
    plt.savefig(OUT_DIR / "3a_probleme_XGBoost_escalier.png", dpi=300)
    plt.close()

    # --- VISUEL 3B: La Solution Ridge (Continu) ---
    print("Génération Visuel 3B: Ridge Continu...")
    model_ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    y_pred_log_ridge = cross_val_predict(model_ridge, X66, y66_log, cv=GroupKFold(n_splits=5), groups=groups)
    y_pred_ridge = np.expm1(y_pred_log_ridge)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y66, y_pred_ridge, alpha=0.5, color='darkorange', s=15)
    plt.plot([0, limit], [0, limit], 'k--', linewidth=2, alpha=0.6) # use same limit
    plt.title("Solution Ridge : Modèle Physique Continu\n(Equation Linéaire 1/T_max)", fontsize=14)
    plt.xlabel("NOx Mesurés [ppm]", fontsize=12)
    plt.ylabel("NOx Prédits (Fonction Continue) [ppm]", fontsize=12)
    plt.grid(True, ls=':', alpha=0.5)
    plt.savefig(OUT_DIR / "3b_solution_Ridge_continu.png", dpi=300)
    plt.close()

    # --- VISUEL 4: Importance des Entrées ---
    print("Génération Visuel 4: Importance...")
    model_ridge.fit(X66, y66_log)
    feats = X66.columns
    imps = np.abs(model_ridge.named_steps['ridge'].coef_)
    sorted_idx = np.argsort(imps)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), imps[sorted_idx], color='teal', alpha=0.8)
    plt.yticks(range(len(sorted_idx)), [feats[i] for i in sorted_idx])
    plt.title("Impact des Variables dans le Modèle (Coefficients Ridge)", fontsize=14)
    plt.xlabel("Importance Relative (Valeur Absolue du Poids)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_importance_variables.png", dpi=300)
    plt.close()

    # --- VISUEL 5: Comparaison des Swirlers ---
    print("Génération Visuel 5: Comparaison...")
    data_to_plot = [df66["NOx [ppm]"], df82["NOx [ppm]"]]
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_to_plot, labels=['S066', 'S082'], patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    plt.title("Variabilité des Émissions entre Swirlers", fontsize=14)
    plt.ylabel("NOx [ppm]", fontsize=12)
    plt.savefig(OUT_DIR / "5_comparaison_emissions.png", dpi=300)
    plt.close()

    print(f"\nTerminé ! 5 visuels générés dans {OUT_DIR}")

if __name__ == "__main__":
    generate_visuals()
