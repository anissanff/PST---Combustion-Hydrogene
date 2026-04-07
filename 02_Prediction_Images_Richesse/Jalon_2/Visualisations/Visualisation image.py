import matplotlib.pyplot as plt
import random
import numpy as np
import struct
from pathlib import Path

# --- CONFIG (à adapter) ---
SPE_DIR = "/Volumes/VERBATIM HD/PST/s066/SPE"
HEADER_SIZE = 4100

def read_frame_random(spe_dir):
    # 1. Choisir un fichier au hasard
    spe_files = list(Path(spe_dir).glob("*.spe"))
    chosen_file = random.choice(spe_files)
    
    # 2. Lire le header pour connaître les dimensions
    with open(chosen_file, "rb") as f:
        header = f.read(HEADER_SIZE)
    x = struct.unpack_from("<H", header, 42)[0]
    y = struct.unpack_from("<H", header, 656)[0]
    num_frames = struct.unpack_from("<I", header, 1446)[0]
    
    # On suppose l'uint16 pour l'affichage (dtype le plus courant en SPE)
    dtype = np.uint16 
    
    # 3. Ouvrir en memmap et choisir une frame au hasard
    mm = np.memmap(str(chosen_file), mode="r", dtype=dtype, offset=HEADER_SIZE, shape=(num_frames, y, x))
    frame_idx = random.randint(0, num_frames - 1)
    frame = np.array(mm[frame_idx], copy=True)
    
    return frame, chosen_file.name, frame_idx

# --- AFFICHAGE ---
try:
    img, filename, idx = read_frame_random(SPE_DIR)

    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap='gray') # 'viridis' ou 'magma' sont aussi sympas pour voir les contrastes
    plt.colorbar(label='Intensité des pixels')
    plt.title(f"Fichier : {filename}\nFrame : {idx} | Dim : {img.shape}")
    plt.axis('off') # Cache les axes gradués
    plt.show()

except Exception as e:
    print(f"Erreur lors de la lecture : {e}")