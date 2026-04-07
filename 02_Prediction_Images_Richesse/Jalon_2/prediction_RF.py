#!/usr/bin/env python3
import os
import struct
import csv
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier



# --- 1. CONFIGURATION ---
SPE_DIR = "/Volumes/VERBATIM HD/PST/s066/SPE"
HEADER_SIZE = 4100
POOL_SIZE = 10 # Ajusté à 10 pour des images 1000x1000 (Vecteur de 15 625)
SEED = 42
N_PRED_SHOW = 150 

LABELS_BY_TIME = {
    "10_37_34": 0.65, "12_51_00": 0.75, "13_47_07": 0.85, "14_59_31": 0.95, #0%
    "11_06_44": 0.65, "13_14_00": 0.75, "14_25_32": 0.85, "15_31_37": 0.95, #5%
    "11_12_14": 0.65, "13_19_28": 0.75, "14_31_04": 0.85, "15_37_10": 0.95, #10%
    "11_17_49": 0.65, "13_25_07": 0.75, "14_36_39": 0.85, "15_42_48": 0.95, #15%
    "11_23_27": 0.65, "13_30_38": 0.75, "14_42_19": 0.85, "15_48_22": 0.95, #17.5%
    "11_29_20": 0.65, "13_36_16": 0.75, "14_47_52": 0.85, "15_53_53": 0.95, #20%  
    "11_35_07": 0.65, "13_41_44": 0.75, "14_53_35": 0.85, "15_59_25": 0.95, #22.5%  
}

DTYPE_MAP = {0: np.float32, 1: np.int32, 2: np.int16, 3: np.uint16, 
             5: np.float64, 6: np.uint8, 8: np.uint32}


# --- 2. OUTILS DE LECTURE ET TRAITEMENT ---

def read_header(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    xdim = struct.unpack_from("<H", header, 42)[0]
    ydim = struct.unpack_from("<H", header, 656)[0]
    dtype_code = struct.unpack_from("<H", header, 108)[0]
    num_frames = struct.unpack_from("<I", header, 1446)[0]
    return xdim, ydim, num_frames, DTYPE_MAP[dtype_code]

def apply_max_pooling(frame, pool_size):
    h, w = frame.shape
    new_h, new_w = h // pool_size, w // pool_size
    view = frame[:new_h * pool_size, :new_w * pool_size].reshape(
        new_h, pool_size, new_w, pool_size
    )
    return view.max(axis=(1, 3))

def frame_to_features(frame, pool_size):
    pooled = apply_max_pooling(frame.astype(np.float32), pool_size)
    return pooled.flatten()


# --- 3. CHARGEMENT DES DONNÉES ---

def load_datasets(valid_files, pool_size, rng):
    train_features, train_labels, holdout_refs = [], [], []
    
    print(f"[*] Traitement de {len(valid_files)} fichiers .spe...")
    
    for path, label_value in valid_files:
        x, y, num_frames, dtype = read_header(str(path))
        mm = np.memmap(str(path), mode="r", dtype=dtype, offset=HEADER_SIZE, shape=(num_frames, y, x))

        # Split 90% train / 10% holdout
        indices = np.arange(num_frames)
        rng.shuffle(indices)
        split_idx = int(0.9 * len(indices))
        
        # Train
        for i in indices[:split_idx]:
            train_features.append(frame_to_features(mm[i], pool_size))
            train_labels.append(label_value)
        
        # Holdout (pour prédiction finale)
        for i in indices[split_idx:]:
            holdout_refs.append((path, int(i)))

    return np.array(train_features), np.array(train_labels), holdout_refs

# --- 4. EXECUTION PRINCIPALE ---
def main():
    print("=== DÉMARRAGE DU PIPELINE ===")
    rng = np.random.default_rng(SEED)

    # Étape A: Identification
    spe_files = sorted(Path(SPE_DIR).glob("*.spe"))
    valid_files = [(p, LABELS_BY_TIME[p.stem.split()[-1]]) 
                   for p in spe_files if p.stem.split()[-1] in LABELS_BY_TIME]

    if not valid_files:
        print("[!] Erreur: Aucun fichier correspondant aux labels trouvé.")
        return

    # Étape B: Chargement & Feature Engineering
    xdim, ydim, _, _ = read_header(str(valid_files[0][0]))
    X_train, y_raw, holdout_refs = load_datasets(valid_files, POOL_SIZE, rng)
    
    print(f"[+] Données d'entraînement : {X_train.shape[0]} images")
    print(f"[+] Taille du vecteur de caractéristiques : {X_train.shape[1]}")
    print(f"[+] Images en attente de prédiction (holdout) : {len(holdout_refs)}")

    # Étape C: Encodage des labels
    label_values = sorted(list(set(y_raw)))
    label_to_class = {v: i for i, v in enumerate(label_values)}
    class_to_label = {i: v for v, i in label_to_class.items()}
    y_train = np.array([label_to_class[val] for val in y_raw])

    # Étape D: Entraînement
    print(f"[*] Entraînement du RandomForest (max_depth=8)...")
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=SEED)
    model.fit(X_train, y_train)
    print("[+] Modèle entraîné avec succès.")

    # --- Étape E corrigée dans votre main() ---

    # 6. Évaluation sur le Holdout
    if holdout_refs:
        print(f"[*] Génération de {min(N_PRED_SHOW, len(holdout_refs))} prédictions...")
        eval_rows = []
        pick = rng.choice(len(holdout_refs), size=min(N_PRED_SHOW, len(holdout_refs)), replace=False)
        
        # On récupère l'ordre des classes pour savoir à quoi correspondent les probabilités
        # model.classes_ contient les indices (0, 1, 2...)
        
        for idx in pick:
            path, f_idx = holdout_refs[idx]
            
            x, y, num_frames, dtype = read_header(str(path))
            mm = np.memmap(
                str(path), 
                mode="r", 
                dtype=dtype, 
                offset=HEADER_SIZE, 
                shape=(num_frames, y, x)
            )
            
            feat = frame_to_features(mm[f_idx], POOL_SIZE).reshape(1, -1)
            
            # Prédiction de la classe
            pred_class = int(model.predict(feat)[0])
            pred_label = class_to_label[pred_class]
            
            # --- AJOUT PREDICT_PROBA ---
            # predict_proba renvoie une liste de probabilités (ex: [0.1, 0.8, 0.1])
            probs = model.predict_proba(feat)[0]
            # On récupère la probabilité de la classe choisie (la plus haute)
            confiance = np.max(probs) 
            # ---------------------------

            time_key = path.stem.split()[-1]
            eval_rows.append([time_key, f"{pred_label:.2f}", f"{confiance:.4f}"])

        # Sauvegarde CSV avec la colonne de confiance
        output_path = Path(__file__).parent / "predictions_avec_probabilites.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Ajout de l'en-tête "confiance"
            writer.writerow(["image_name", "pred_label", "confiance"])
            writer.writerows(eval_rows)
            
        print(f"[+] Fichier enregistré avec probabilités : {output_path}")

if __name__ == "__main__":
    main()


