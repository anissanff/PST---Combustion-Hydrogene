import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# =========================
# CONFIG
# =========================
OUT_DIR = Path("outputs_multi_swirlers")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# FEATURE ENGINEERING
# =========================
def build_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Richesse"] = df["Richesse"]
    out["Water_Rate"] = df["Taux d'eau"]
    
    # Temperatures in Kelvin
    temp_cols = [f"CH{i}" for i in range(1, 10)]
    for col in temp_cols:
        out[f"{col}_K"] = df[col] + 273.15
        
    # Arrhenius Linearized Term (Inverse Temperature)
    t_cols = [f"CH{i}_K" for i in range(6, 10)]
    T_max_K = out[t_cols].max(axis=1)
    out["Inv_Tmax"] = 10000 / T_max_K
    out["Inv_CH9"] = 10000 / out["CH9_K"]
    
    # Non-linear Interactions
    out["phi_Tmax"] = out["Richesse"] * T_max_K
    out["water_Tmax"] = out["Water_Rate"] * T_max_K
    
    return out

def train_and_eval(X, y, groups, name):
    print(f"\n--- Training Model for {name} ---")
    y_log = np.log1p(y)
    
    cv = GroupKFold(n_splits=5)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    
    y_pred_log = cross_val_predict(model, X, y_log, cv=cv, groups=groups, n_jobs=-1)
    y_pred = np.maximum(0, np.expm1(y_pred_log))
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(f"  R2:  {r2:.4f}")
    print(f"  MAE: {mae:.4f} ppm")
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5, s=15)
    mn, mx = 0, max(y.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.title(f"NOx Pred - {name}\nR2={r2:.3f} | MAE={mae:.2f}")
    plt.xlabel("True [ppm]")
    plt.ylabel("Pred [ppm]")
    plt.savefig(OUT_DIR / f"{name}_pred_vs_true.png")
    plt.close()
    
    # Fit final and save
    model.fit(X, y_log)
    joblib.dump({"model": model, "features": list(X.columns)}, OUT_DIR / f"{name}_model.pkl")
    return r2, mae

def main():
    # 1. Load Data
    s066_path = "S066_Clean_all.xlsx"
    s082_path = "S082_Clean_all.xlsx"
    
    df66 = pd.read_excel(s066_path, sheet_name='Data_Exploitee')
    df82 = pd.read_excel(s082_path, sheet_name='Data_Exploitee')
    
    # Prepare S066
    df66["Exp_ID"] = df66.groupby(["Richesse", "Taux d'eau"]).ngroup()
    X66 = build_physics_features(df66)
    y66 = df66["NOx [ppm]"].to_numpy()
    g66 = df66["Exp_ID"].to_numpy()
    
    # Prepare S082
    df82["Exp_ID"] = df82.groupby(["Richesse", "Taux d'eau"]).ngroup()
    X82 = build_physics_features(df82)
    y82 = df82["NOx [ppm]"].to_numpy()
    g82 = df82["Exp_ID"].to_numpy()
    
    # 2. Individual Training
    r2_66, mae_66 = train_and_eval(X66, y66, g66, "S066")
    r2_82, mae_82 = train_and_eval(X82, y82, g82, "S082")
    
    # 3. Master Model (S066 + S082)
    print("\n--- Training Master Model ---")
    
    # Add Swirler Type (Categorical)
    X66_master = X66.copy()
    X66_master["is_S082"] = 0
    
    X82_master = X82.copy()
    X82_master["is_S082"] = 1
    
    X_master = pd.concat([X66_master, X82_master], ignore_index=True)
    y_master = np.concatenate([y66, y82])
    
    # Groups for Master: (Swirler, Phi, Water)
    df66["Global_Exp_ID"] = "66_" + df66["Exp_ID"].astype(str)
    df82["Global_Exp_ID"] = "82_" + df82["Exp_ID"].astype(str)
    g_master = pd.concat([df66["Global_Exp_ID"], df82["Global_Exp_ID"]], ignore_index=True).factorize()[0]
    
    r2_m, mae_m = train_and_eval(X_master, y_master, g_master, "Master_v1")
    
    # Synthesis
    print("\nFINAL SYNTHESIS:")
    print(f"Swirler S066 : R2 = {r2_66:.4f}")
    print(f"Swirler S082 : R2 = {r2_82:.4f}")
    print(f"Master Model : R2 = {r2_m:.4f}")

if __name__ == "__main__":
    main()
