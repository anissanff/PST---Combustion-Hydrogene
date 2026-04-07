#!/usr/bin/env python3
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

SPE_DIR = "/Volumes/VERBATIM HD/z_docs/s066/SPE"
HEADER_SIZE = 4100
SEED = 42

TIME_KEY = "11_12_14"  # change if needed

DTYPE_MAP = {
    0: np.float32,
    1: np.int32,
    2: np.int16,
    3: np.uint16,
    5: np.float64,
    6: np.uint8,
    8: np.uint32,
}

SOBEL_V = np.array(
    [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]
)


def read_header(path):
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    xdim = struct.unpack_from("<H", header, 42)[0]
    ydim = struct.unpack_from("<H", header, 656)[0]
    dtype_code = struct.unpack_from("<H", header, 108)[0]
    num_frames = struct.unpack_from("<I", header, 1446)[0]
    return xdim, ydim, num_frames, DTYPE_MAP[dtype_code]


def find_spe(time_key):
    matches = list(Path(SPE_DIR).glob(f"*{time_key}*.spe"))
    if not matches:
        raise FileNotFoundError(f"Aucun fichier pour {time_key} dans {SPE_DIR}")
    return matches[0]


def main():
    rng = np.random.default_rng(SEED)
    spe_path = find_spe(TIME_KEY)
    x, y, n, dtype = read_header(str(spe_path))

    mm = np.memmap(
        str(spe_path),
        mode="r",
        dtype=dtype,
        offset=HEADER_SIZE,
        shape=(n, y, x),
    )

    idx = int(rng.integers(0, n))
    frame = mm[idx].astype(np.float32)
    edges = ndimage.convolve(frame, SOBEL_V, mode="reflect")
    edges = np.abs(edges)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(frame, cmap="gray")
    axes[0].set_title(f"Brute: {TIME_KEY} frame {idx}")
    axes[0].axis("off")

    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Sobel vertical (|edges|)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
