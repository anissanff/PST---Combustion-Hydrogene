import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

warnings.filterwarnings("ignore")

print("\n--- GÉNÉRATION DE LA SURFACE 3D DU JUMEAU NUMÉRIQUE ---")

# 1. Entraînement rapide du modèle "Zéro Flamme"
df66 = pd.read_excel("S066_Clean_all.xlsx", sheet_name='Data_Exploitee' if 'Data_Exploitee' in pd.ExcelFile('S066_Clean_all.xlsx').sheet_names else 0)
df82 = pd.read_excel("S082_Clean_all.xlsx", sheet_name='Data_Exploitee' if 'Data_Exploitee' in pd.ExcelFile('S082_Clean_all.xlsx').sheet_names else 0)

def extract_ops(df, is_s082):
    X = pd.DataFrame()
    X["Richesse_Bouton"] = df["Richesse"]
    X["Eau_Bouton"] = df["Taux d'eau"]
    X["Machine_S082"] = is_s082
    return X

X_master = pd.concat([extract_ops(df66, 0), extract_ops(df82, 1)], ignore_index=True)
y_master = np.concatenate([df66["NOx [ppm]"].values, df82["NOx [ppm]"].values])
y_log = np.log1p(y_master)

# Le modèle polynomial
modele = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), StandardScaler(), Ridge(alpha=0.5))
modele.fit(X_master, y_log)

# Création d'une Grille virtuelle respectant les vraies limites de l'expérience
richesse_min, richesse_max = X_master["Richesse_Bouton"].min(), X_master["Richesse_Bouton"].max()
eau_min, eau_max = X_master["Eau_Bouton"].min(), X_master["Eau_Bouton"].max()

richesse_range = np.linspace(richesse_min, richesse_max, 50)
eau_range = np.linspace(eau_min, eau_max, 50)

# Créer une matrice 2D mathématique
R_grid, E_grid = np.meshgrid(richesse_range, eau_range)

# Applatir pour faire les prédictions
R_flat = R_grid.flatten()
E_flat = E_grid.flatten()
# On va faire le dessin pour le swirler S066 (Machine_S082 = 0)
M_flat = np.zeros_like(R_flat) 

X_virtuel = pd.DataFrame({"Richesse_Bouton": R_flat, "Eau_Bouton": E_flat, "Machine_S082": M_flat})

# 3. L'IA prédit les milliers de points inventés !
pred_log_virtuel = modele.predict(X_virtuel)
NOx_pred_virtuel = np.expm1(pred_log_virtuel).reshape(R_grid.shape) # On reforme la matrice 2D

# 4. Affichage en 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Dessin de la nappe colorée
surf = ax.plot_surface(R_grid, E_grid, NOx_pred_virtuel, cmap='inferno', edgecolor='none', alpha=0.8)

# Décoration du graphique
ax.set_title("Le 'Cerveau' du Jumeau Numérique (Swirler S066)\nTopographie des émissions de NOx anticipées", fontsize=14, fontweight='bold')
ax.set_xlabel("Richesse commandée", fontsize=11)
ax.set_ylabel("Taux d'Eau injecté [%]", fontsize=11)
ax.set_zlabel("NOx Prédit par l'IA [ppm]", fontsize=11)

# Échelle de couleur
fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1, label='Concentration de NOx [ppm]')

# Angle de vue sympa
ax.view_init(elev=25, azim=135) 

plt.savefig('Surface_3D_Jumeau.png', dpi=300, bbox_inches='tight')
print("✅ Image 3D générée : 'Surface_3D_Jumeau.png'")
