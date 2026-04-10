import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

print("\n--- CRÉATION DU DIGITAL TWIN (JUMEAU NUMÉRIQUE 'ZÉRO FLAMME') ---")
print("Mode Opérateur : Oublions les sondes, donnons juste les boutons de commande à l'IA !\n")

# 1. Chargement des données (Identique)
df66 = pd.read_excel("S066_Clean_all.xlsx", sheet_name='Data_Exploitee' if 'Data_Exploitee' in pd.ExcelFile('S066_Clean_all.xlsx').sheet_names else 0)
df82 = pd.read_excel("S082_Clean_all.xlsx", sheet_name='Data_Exploitee' if 'Data_Exploitee' in pd.ExcelFile('S082_Clean_all.xlsx').sheet_names else 0)

# =========================================================
# L'ASTUCE ICI : UNIQUEMENT DES COMMANDES OPÉRATEUR !
# =========================================================
def extract_operator_commands(df, is_s082):
    X = pd.DataFrame()
    X["Richesse_Bouton"] = df["Richesse"]
    X["Eau_Bouton"] = df["Taux d'eau"]
    X["Machine_S082"] = is_s082
    # AUCUN THERMOCOUPLE ! L'IA NE VOIT PAS LA FLAMME !
    return X

X66 = extract_operator_commands(df66, 0)
X82 = extract_operator_commands(df82, 1)

X_commandes = pd.concat([X66, X82], ignore_index=True)
y_vrai = np.concatenate([df66["NOx [ppm]"].values, df82["NOx [ppm]"].values])

# GroupKFold strict pour éviter l'overfitting temporel
g66 = [f"66_{g}" for g in df66.groupby(["Richesse", "Taux d'eau"]).ngroup()]
g82 = [f"82_{g}" for g in df82.groupby(["Richesse", "Taux d'eau"]).ngroup()]
groupes = np.concatenate([g66, g82])

# 2. Entraînement du Modèle Polynomial (Surface de Réponse Mathématique)
# Puisque nous n'avons pas la loi d'Arrhenius (pas de Température), 
# on aide le modèle en lui permettant de croiser Richesse et Eau au carré et au cube !
print("Entraînement de l'IA (Polynomial Multi-Variable) en cours...")
modele_zero_flamme = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False), # Simule les courbes chimiques non-linéaires !
    StandardScaler(),
    Ridge(alpha=0.5)
)

y_log = np.log1p(y_vrai)
y_pred_log = cross_val_predict(modele_zero_flamme, X_commandes, y_log, cv=GroupKFold(n_splits=5), groups=groupes)
y_pred_final = np.expm1(y_pred_log)

# On entraîne le modèle définitif pour les tests "En direct"
modele_zero_flamme.fit(X_commandes, y_log)

# 3. Évaluation
score = r2_score(y_vrai, y_pred_final)
print(f"✅ R2 d'Anticipation (Zéro Thermocouple !) : {score:.4f} (soit {score*100:.1f}%)")

# 4. Démonstration : Simulation Opérateur (Jouons à l'ingénieur !)
print("\n--- SIMULATEUR DE BUREAU ---")
scenarios = pd.DataFrame([
    {"Richesse_Bouton": 0.65, "Eau_Bouton": 3.0, "Machine_S082": 0}, # Flamme classique S066
    {"Richesse_Bouton": 0.85, "Eau_Bouton": 8.0, "Machine_S082": 0}, # Forte injection eau S066
    {"Richesse_Bouton": 0.70, "Eau_Bouton": 4.5, "Machine_S082": 1}, # Flamme moyenne S082
])

print("Cas simulés qui ne demandent pas d'allumer le brûleur :")
for i, test in scenarios.iterrows():
    # Prédiction pour ce scénario spécifique
    pred_log = modele_zero_flamme.predict(pd.DataFrame([test]))
    ppm_predit = np.expm1(pred_log)[0]
    
    geo = "S082" if test["Machine_S082"] == 1 else "S066"
    print(f" ⚙️  [Opérateur sur {geo}] Richesse: {test['Richesse_Bouton']} | Eau: {test['Eau_Bouton']}%  --> Prédiction NOx: {ppm_predit:.1f} ppm")

# 5. Visualisation (La Surface de Prédiction pour S066)
plt.figure(figsize=(10, 8))
plt.scatter(y_vrai, y_pred_final, color='purple', alpha=0.5, s=20, label='Prédiction sans aucune sonde')
max_val = max(y_vrai.max(), y_pred_final.max())
plt.plot([0, max_val], [0, max_val], color='black', linestyle='dashed', linewidth=2)
plt.title(f"Jumeau Numérique 'Zéro Flamme' (R² = {score:.3f})\nL'IA prédit la chimie sans thermomètre, juste avec les commandes !", fontsize=13, fontweight='bold')
plt.xlabel("NOx Réellement mesurés au Labo [ppm]", fontsize=12)
plt.ylabel("NOx Prédits de Bureau (Juste Richesse/Eau) [ppm]", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('Jumeau_Numerique_Sans_Sondes.png', dpi=300, bbox_inches='tight')
print("\nGraphique généré : 'Jumeau_Numerique_Sans_Sondes.png'")
