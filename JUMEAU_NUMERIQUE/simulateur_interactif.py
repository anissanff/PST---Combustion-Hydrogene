import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
import os

warnings.filterwarnings("ignore")

# Entraînement furtif au lancement
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

modele = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), StandardScaler(), Ridge(alpha=0.5))
modele.fit(X_master, np.log1p(y_master))

os.system('cls' if os.name == 'nt' else 'clear')

print("="*60)
print(" 🚀 Bienvenue dans le Jumeau Numérique 'Zéro Flamme' 🚀")
print("="*60)
print("Appuyez sur 'q' pour quitter.\n")

while True:
    try:
        r = input("👉 Entrez une Richesse cible (ex: 0.85) : ")
        if r.lower() == 'q': break
        richesse = float(r.replace(',', '.'))
        
        e = input("👉 Entrez un Taux d'eau injecté en % (ex: 5) : ")
        if e.lower() == 'q': break
        eau = float(e.replace(',', '.'))
        
        m = input("👉 Sur quel Swirler (Taper 0 pour S066, 1 pour S082) : ")
        if m.lower() == 'q': break
        machine = int(m)
        
        # Le modèle tourne !
        test = pd.DataFrame([{"Richesse_Bouton": richesse, "Eau_Bouton": eau, "Machine_S082": machine}])
        pred_log = modele.predict(test)
        ppm = np.expm1(pred_log)[0]
        
        nom_machine = "S082" if machine == 1 else "S066"
        print("\n" + "-"*40)
        print(f"✅ Analyse IA pour le brûleur {nom_machine}")
        print(f"🔥 Avec une Richesse de {richesse} et {eau}% d'eau...")
        print(f"🏭 Émissions estimées à l'échappement : {ppm:.2f} ppm de NOx")
        print("-"*40 + "\n")
        
    except ValueError:
        print("❌ Saisie invalide. Veuillez entrer un nombre valide (ex: 0.85).")
