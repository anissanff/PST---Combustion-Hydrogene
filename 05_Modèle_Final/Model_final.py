import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import glob
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

import untangle
from io import StringIO
from spe_loader import read_at

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_XLSX = os.path.join(BASE_DIR, "s066.xlsx")
MEAS_XLSX = os.path.join(BASE_DIR, "S066_Clean_all.xlsx")
PARAMS_SHEET = "parameters"
MEAS_SHEET = "Data_Exploitee"

FEATURE_IMAGE_SIZE = (32, 32)  # (h, w)


def parse_time_str(s: str) -> datetime.time:
    s = str(s).strip()
    return datetime.strptime(s, "%H %M %S").time()


def downsample_img(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    h, w = img.shape
    ys = np.linspace(0, h - 1, new_h).astype(int)
    xs = np.linspace(0, w - 1, new_w).astype(int)
    return img[np.ix_(ys, xs)]


def _read_footer(file):
    footer_pos = read_at(file, 678, 8, np.uint64)[0]
    file.seek(footer_pos)
    xmlbytes = file.read()
    xmltext = xmlbytes.decode("utf-8", errors="ignore")

    parser = untangle.make_parser()
    sax_handler = untangle.Handler()
    parser.setContentHandler(sax_handler)
    parser.parse(StringIO(xmltext))
    return sax_handler.root


def _get_dtype(file):
    dtype_code = read_at(file, 108, 2, np.uint16)[0]
    if dtype_code == 0:
        return np.float32
    if dtype_code == 1:
        return np.int32
    if dtype_code == 2:
        return np.int16
    if dtype_code == 3:
        return np.uint16
    if dtype_code == 8:
        return np.uint32
    raise ValueError(f"Unrecognized data type code: {dtype_code}")


def _get_dims(footer):
    blocks = footer.SpeFormat.DataFormat.DataBlock.DataBlock
    if isinstance(blocks, list):
        xdim = int(blocks[0]["width"])
        ydim = int(blocks[0]["height"])
    else:
        xdim = int(blocks["width"])
        ydim = int(blocks["height"])
    return xdim, ydim


def read_first_frame_roi0(filepath: str) -> np.ndarray:
    with open(filepath, "rb") as f:
        header_version = read_at(f, 1992, 3, np.float32)[0]
        if header_version < 3.0:
            raise ValueError(f"Unsupported SPE version {header_version}")
        _ = read_at(f, 1446, 2, np.uint16)[0]  # nframes

        footer = _read_footer(f)
        dtype = _get_dtype(f)
        xdim, ydim = _get_dims(footer)

        f.seek(4100)
        data = np.fromfile(f, dtype, xdim * ydim).reshape(ydim, xdim)
        return data


def extract_features_from_spe(filepath: str) -> np.ndarray:
    img = read_first_frame_roi0(filepath).astype(np.float32)
    img = np.log1p(img)
    img_small = downsample_img(img, FEATURE_IMAGE_SIZE[0], FEATURE_IMAGE_SIZE[1])
    stats = np.array([
        img.mean(),
        img.std(),
        img.min(),
        img.max(),
    ], dtype=np.float32)
    feats = np.concatenate([img_small.flatten(), stats])
    return feats


def plot_sample_predictions(df, test_idx, y_true, y_pred, out_path="pred_examples.png"):
    n_show = min(5, len(test_idx))
    if n_show == 0:
        return

    plt.figure(figsize=(12, 6))
    for i in range(n_show):
        idx = test_idx[i]
        spe_path = df.loc[idx, "spe_path"]
        img = read_first_frame_roi0(spe_path)
        ax = plt.subplot(2, 3, i + 1)
        ax.imshow(img, cmap="hot")
        ax.set_title(f"NOx real: {y_true[i]:.2f}\nNOx predit: {y_pred[i]:.2f}")
        ax.axis("off")

    plt.suptitle("Exemples de predictions (NOx)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_low_nox_conditions(df, out_path="low_nox_conditions.png"):
    vars_to_plot = ["Richesse", "Taux d'eau", "O2 [vol%]", "CH9"]

    plt.figure(figsize=(12, 8))
    for i, col in enumerate(vars_to_plot, start=1):
        if col == "Richesse":
            # Only keep specific richness values
            target_vals = [0.65, 0.75, 0.85, 0.95]
            tol = 1e-3
            grouped_x = []
            grouped_y = []
            for v in target_vals:
                subset = df[np.isclose(df[col], v, atol=tol)]
                if subset.empty:
                    continue
                grouped_x.append(v)
                grouped_y.append(round(subset["NOx [ppm]"].mean(), 3))
            grouped_x = np.array(grouped_x)
            grouped_y = np.array(grouped_y)
        else:
            # Bin into quantiles and compute mean NOx in each bin
            bins = pd.qcut(df[col], q=5, duplicates="drop")
            grouped_x = df.groupby(bins)[col].mean().values
            grouped_y = df.groupby(bins)["NOx [ppm]"].mean().round(3).values

        ax = plt.subplot(2, 2, i)
        ax.plot(grouped_x, grouped_y, marker="o")
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.2f}")
        ax.set_title(f"NOx moyen vs {col}")
        ax.set_ylabel("NOx [ppm]")
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.3f}")
        ax.set_xlabel(col)

    plt.suptitle("Conditions ou le NOx est le plus bas (par quantiles)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Load measurement data
    meas = pd.read_excel(MEAS_XLSX, sheet_name=MEAS_SHEET)
    if "Date" not in meas.columns or "Time" not in meas.columns:
        raise ValueError("Expected 'Date' and 'Time' columns in measurement sheet")

    # Build datetime columns
    meas["Date"] = pd.to_datetime(meas["Date"]).dt.date
    meas["Time"] = pd.to_datetime(meas["Time"].astype(str), format="%H:%M:%S").dt.time

    # Most common date used for mapping
    common_date = meas["Date"].mode().iloc[0]

    # Load parameters (time ranges per spe file)
    params = pd.read_excel(PARAMS_XLSX, sheet_name=PARAMS_SHEET)

    # Map spe file names to actual files
    spe_files = glob.glob(os.path.join(BASE_DIR, "*.spe"))
    spe_map = {}
    for name in params["spe file name"].dropna():
        key = str(name).strip()
        matches = [f for f in spe_files if f.endswith(f"{key}.spe")]
        if matches:
            spe_map[key] = matches[0]

    rows = []

    for _, row in params.iterrows():
        spe_key = row.get("spe file name")
        if pd.isna(spe_key):
            continue
        spe_key = str(spe_key).strip()
        if spe_key not in spe_map:
            print(f"[WARN] No .spe file match for {spe_key}")
            continue

        start_time = parse_time_str(row["start"])
        end_time = parse_time_str(row["end"])

        # Filter measurement data within time interval
        subset = meas[(meas["Date"] == common_date) &
                      (meas["Time"] >= start_time) &
                      (meas["Time"] <= end_time)]

        if subset.empty:
            print(f"[WARN] No measurement rows for {spe_key} between {start_time} and {end_time}")
            continue

        # Aggregate target variables
        agg = subset[["CH9", "Richesse", "Taux d'eau", "O2 [vol%]", "NOx [ppm]"]].mean()

        # Extract image features
        spe_path = spe_map[spe_key]
        feats = extract_features_from_spe(spe_path)
        img_mean = float(feats[-4])

        rows.append({
            "spe_key": spe_key,
            "spe_path": spe_path,
            "date": common_date,
            "start_time": start_time,
            "end_time": end_time,
            **agg.to_dict(),
            "img_mean": img_mean,
            **{f"f{i}": v for i, v in enumerate(feats)}
        })

    if not rows:
        raise RuntimeError("No rows were created. Check time ranges and data alignment.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(BASE_DIR, "dataset.csv"), index=False)

    # Train/test split
    feature_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feature_cols].values
    y = df["NOx [ppm]"].values

    if len(df) < 5:
        raise RuntimeError("Not enough samples to train a model. Need at least 5.")

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=42
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    joblib.dump(model, os.path.join(BASE_DIR, "rf_nox.pkl"))

    influence_vars = ["CH9", "Richesse", "Taux d'eau", "O2 [vol%]"]
    influence_rows = []
    for col in influence_vars:
        corr = df[[col, "NOx [ppm]"]].corr(method="spearman").iloc[0, 1]
        influence_rows.append({"Metric": f"Influence Spearman - {col}", "Value": corr})

    metrics_df = pd.DataFrame(
        [
            {"Metric": "MAE", "Value": mae},
            {"Metric": "RMSE", "Value": rmse},
            {"Metric": "R2", "Value": r2},
            {"Metric": "Corr Spearman - ImageMean vs NOx", "Value": df[["img_mean", "NOx [ppm]"]].corr(method="spearman").iloc[0, 1]},
            *influence_rows,
        ]
    )
    metrics_df.to_csv(os.path.join(BASE_DIR, "metrics.csv"), index=False)

    print("Samples:", len(df))
    print("Features:", len(feature_cols))
    print(metrics_df)

    # Visualizations
    plot_sample_predictions(df, test_idx, y_test, y_pred, out_path=os.path.join(BASE_DIR, "pred_examples.png"))
    plot_low_nox_conditions(df, out_path=os.path.join(BASE_DIR, "low_nox_conditions.png"))


if __name__ == "__main__":
    main()
