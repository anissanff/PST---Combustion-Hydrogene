import matplotlib.pyplot as plt
import numpy as np
import random
import struct
from pathlib import Path

# --- CONFIGURATION ---
SPE_DIR = "/Volumes/VERBATIM HD/PST/s066/SPE"
HEADER_SIZE = 4100
POOL_SIZE = 10

def read_frame_random(spe_dir):
    spe_files = list(Path(spe_dir).glob("*.spe"))
    chosen_file = random.choice(spe_files)
    
    with open(chosen_file, "rb") as f:
        header = f.read(HEADER_SIZE)
    x = struct.unpack_from("<H", header, 42)[0]
    y = struct.unpack_from("<H", header, 656)[0]
    num_frames = struct.unpack_from("<I", header, 1446)[0]
    
    # On ouvre en memmap (uint16 par défaut pour SPE)
    mm = np.memmap(str(chosen_file), mode="r", dtype=np.uint16, offset=HEADER_SIZE, shape=(num_frames, y, x))
    frame_idx = random.randint(0, num_frames - 1)
    return np.array(mm[frame_idx], copy=True), chosen_file.name

def apply_max_pooling(frame, pool_size):
    h, w = frame.shape
    # On calcule la taille finale (1024 -> 102)
    new_h, new_w = h // pool_size, w // pool_size
    
    # On tronque l'image pour qu'elle soit divisible par pool_size
    trimmed_frame = frame[:new_h * pool_size, :new_w * pool_size]
    
    # Transformation magique avec reshape pour isoler les blocs de 10x10
    # puis on prend le max sur les axes des blocs
    view = trimmed_frame.reshape(new_h, pool_size, new_w, pool_size)
    return view.max(axis=(1, 3))

# --- EXECUTION ET AFFICHAGE ---
try:
    img_original, filename = read_frame_random(SPE_DIR)
    img_pooled = apply_max_pooling(img_original, POOL_SIZE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Image Originale
    ax1.imshow(img_original, cmap='gray')
    ax1.set_title(f"Originale\n({img_original.shape[1]}x{img_original.shape[0]})")
    ax1.axis('off')

    # Image après Max Pooling
    ax2.imshow(img_pooled, cmap='gray')
    ax2.set_title(f"Après Max Pooling {POOL_SIZE}x{POOL_SIZE}\n({img_pooled.shape[1]}x{img_pooled.shape[0]})")
    ax2.axis('off')

    plt.suptitle(f"Fichier : {filename}", fontsize=14)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erreur : {e}")