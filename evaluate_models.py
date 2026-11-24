"""
Evaluate trained models and baselines on the held-out test set.

What it does
  1. Baselines: persistence (y_hat = last input frame) and, with --pysteps,
     PySTEPS optical-flow extrapolation (Lucas-Kanade + semi-Lagrangian),
     applied independently to each altitude level.
  2. Metrics in physical units (dBZ, after inverse normalization):
     MSE [dBZ^2], RMSE [dBZ], MAE [dBZ], bias [dBZ], RMSE only where an echo is
     observed (obs >= first threshold), and categorical scores CSI/POD/FAR for
     each reflectivity threshold. Computed for all windows and for the windows
     whose 11 frames are contiguous in time (10-min steps).
  3. Figures with a shared colour scale + colour bar (dBZ), lon/lat axes, and
     error maps (prediction - observation).

Models are fed exactly as during training (utils.prepare_data).

Usage (see run_netuno.sh):
    python evaluate_models.py \
        --model "DLinear=outputs/DLinear_final_model_test_run.h5" \
        --model "ConvLSTM3D=outputs/ConvLSTM_12_test_run.h5" \
        --pysteps
Model spec: "Label=path". The architecture is inferred from the label/file name
(contains "dlinear" -> DLinear, otherwise ConvLSTM); force it with
"Label=path:dlinear" or "Label=path:convlstm".
"""
import os
import json
import argparse
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import prepare_data


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def inverse(scaler, a):
    return scaler.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)


def contiguous_mask(times, seq_len, pred_len, step_minutes):
    """True for windows whose seq_len+pred_len frames are spaced exactly `step`."""
    n = len(times) - seq_len - pred_len + 1
    if not np.issubdtype(np.asarray(times).dtype, np.datetime64):
        return None
    steps = np.diff(times).astype("timedelta64[s]").astype(np.int64) == step_minutes * 60
    span = seq_len + pred_len - 1  # number of steps inside one window
    return np.array([steps[i:i + span].all() for i in range(n)])


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def parse_model_spec(spec):
    label, _, rest = spec.partition("=")
    path, arch = rest, None
    if rest.rsplit(":", 1)[-1].lower() in ("dlinear", "convlstm"):
        path, arch = rest.rsplit(":", 1)
        arch = arch.lower()
    if arch is None:
        arch = "dlinear" if "dlinear" in (label + os.path.basename(path)).lower() else "convlstm"
    return label.strip(), path.strip(), arch


def load_model(path, arch, seq_len, pred_len, features):
    import tensorflow as tf
    from models.dlinear import DLinear

    if arch == "dlinear" and path.endswith(".h5"):
        # Legacy .h5 files saved before the kernel_size fix (same loader as eval_dlinear_model.ipynb)
        return tf.keras.models.load_model(
            path, compile=False,
            custom_objects={
                "DLinear": lambda *a, **kw: DLinear(seq_len=seq_len, pred_len=pred_len, features=features, *a, **kw),
                "mse": tf.keras.metrics.mse,
            },
        )
    return tf.keras.models.load_model(path, compile=False,
                                      custom_objects={"DLinear": DLinear, "mse": tf.keras.metrics.mse})


def predict_model(path, arch, arr, seq_len, pred_len, batch_size):
    cfg = {"architecture": "DLinear" if arch == "dlinear" else "ConvLSTM"}
    X, _ = prepare_data(arr, seq_len, pred_len, cfg=cfg)
    features = int(np.prod(arr.shape[1:]))
    model = load_model(path, arch, seq_len, pred_len, features)
    pred = model.predict(X, batch_size=batch_size, verbose=0)
    # Flat memory order of the output equals the order of the original frame,
    # so a plain reshape recovers (N, lat, lon, alt) for both architectures.
    return pred.reshape((X.shape[0],) + arr.shape[1:])


def pysteps_forecast(arr, scaler, seq_len, n_frames=3):
    """arr: normalized (T, lat, lon, alt) -> (N, lat, lon, alt) in dBZ, N = T - seq_len.

    One 10-min-ahead extrapolation per window and per altitude level, using the
    last `n_frames` input frames of the window to estimate the motion field.
    """
    from pysteps import motion, extrapolation

    oflow = motion.get_method("LK")
    extrap = extrapolation.get_method("semilagrangian")
    n = arr.shape[0] - seq_len
    _, ny, nx, nalt = arr.shape
    out = np.empty((n, ny, nx, nalt), dtype=np.float64)
    failures = 0
    for i in range(n):
        if i % 200 == 0:
            print(f"   pysteps {i}/{n}", flush=True)
        for a in range(nalt):
            frames = inverse(scaler, arr[i + seq_len - n_frames:i + seq_len, :, :, a].astype(np.float64))
            fill = np.nanmin(frames) if np.isfinite(frames).any() else 0.0
            frames = np.where(np.isfinite(frames), frames, fill)
            try:
                V = oflow(frames, verbose=False)
                out[i, :, :, a] = extrap(frames[-1], V, 1, outval=fill)[0]
            except Exception:
                failures += 1
                out[i, :, :, a] = frames[-1]  # fall back to persistence
    print(f"   pysteps done ({failures} fields fell back to persistence)")
    return out, failures


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(obs, pred, thresholds):
    ok = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[ok], pred[ok]
    err = p - o
    m = {
        "n_voxels": int(ok.sum()),
        "MSE_dBZ2": float(np.mean(err ** 2)),
        "RMSE_dBZ": float(np.sqrt(np.mean(err ** 2))),
        "MAE_dBZ": float(np.mean(np.abs(err))),
        "bias_dBZ": float(np.mean(err)),
        "err_std_dBZ": float(np.std(err)),
    }
    echo = o >= thresholds[0]
    m[f"RMSE_echo>={thresholds[0]:g}_dBZ"] = float(np.sqrt(np.mean(err[echo] ** 2))) if echo.any() else np.nan
    for t in thresholds:
        po, pp = o >= t, p >= t
        hits = np.sum(po & pp)
        miss = np.sum(po & ~pp)
        fa = np.sum(~po & pp)
        m[f"CSI@{t:g}"] = float(hits / (hits + miss + fa)) if (hits + miss + fa) else np.nan
        m[f"POD@{t:g}"] = float(hits / (hits + miss)) if (hits + miss) else np.nan
        m[f"FAR@{t:g}"] = float(fa / (hits + fa)) if (hits + fa) else np.nan
    return m


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_sample(idx, alt_idx, obs, preds, lats, lons, alts, out_path, cmap, vmin=None, vmax=None, time_label=""):
    names = ["Observed"] + list(preds)
    fields = [obs[idx, :, :, alt_idx]] + [preds[k][idx, :, :, alt_idx] for k in preds]
    if vmin is None:
        vmin = np.nanmin([np.nanmin(f) for f in fields])
    if vmax is None:
        vmax = np.nanmax([np.nanmax(f) for f in fields])
    errs = [f - fields[0] for f in fields[1:]]
    emax = np.nanmax([np.nanmax(np.abs(e)) for e in errs]) if errs else 1.0

    extent = [float(lons.min()), float(lons.max()), float(lats.min()), float(lats.max())]
    origin = "lower" if lats[0] < lats[-1] else "upper"
    ncol = len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 6.8), squeeze=False, constrained_layout=True)

    for j, (name, f) in enumerate(zip(names, fields)):
        ax = axes[0, j]
        im = ax.imshow(f, origin=origin, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        rmse = "" if j == 0 else f"\nRMSE = {np.sqrt(np.nanmean(errs[j - 1] ** 2)):.2f} dBZ"
        ax.set_title(name + rmse, fontsize=10)
        ax.set_xlabel("Longitude (°)")
        if j == 0:
            ax.set_ylabel("Latitude (°)")
    fig.colorbar(im, ax=axes[0, :].tolist(), label="Reflectivity (dBZ)", shrink=0.9)

    axes[1, 0].axis("off")
    for j, e in enumerate(errs, start=1):
        ax = axes[1, j]
        im_e = ax.imshow(e, origin=origin, extent=extent, cmap="RdBu_r", vmin=-emax, vmax=emax, aspect="auto")
        ax.set_title(f"{names[j]} − Observed", fontsize=10)
        ax.set_xlabel("Longitude (°)")
        if j == 1:
            ax.set_ylabel("Latitude (°)")
    if errs:
        fig.colorbar(im_e, ax=axes[1, 1:].tolist(), label="Error (dBZ)", shrink=0.9)

    fig.suptitle(f"Test window {idx} {time_label} — altitude {alts[alt_idx]:g} m")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def to_latex(df):
    """Minimal booktabs table (avoids the jinja2 dependency of DataFrame.to_latex)."""
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)

    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace(">=", r"$\geq$")

    lines = [r"\begin{tabular}{" + "l" * 2 + "r" * (df.shape[1] - 2) + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(fmt(v)) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-data", default="output_data/test_data.nc")
    parser.add_argument("--target-var", default="dbzh")
    parser.add_argument("--scaler", default="scaler.save")
    parser.add_argument("--model", action="append", default=[], help='"Label=path[:dlinear|convlstm]" (repeatable)')
    parser.add_argument("--pysteps", action="store_true", help="Also run the PySTEPS optical-flow baseline")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--step-minutes", type=int, default=10)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[20.0, 35.0], help="dBZ")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, nargs="*", default=[200],
                        help="Test-window indices to plot (the window with the highest observed "
                             "reflectivity is always added)")
    parser.add_argument("--alt-idx", type=int, nargs="*", default=[0], help="Altitude indices to plot")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed colour-scale min (dBZ)")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed colour-scale max (dBZ)")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--save-preds", action="store_true", help="Save all predictions (dBZ) to .npz")
    args = parser.parse_args()

    if args.pred_len != 1:
        raise SystemExit("Only pred_len=1 is supported (as in the paper).")

    out_dir = args.out_dir or os.path.join("outputs", "eval_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print("Output dir:", out_dir)

    ds = xr.open_dataset(args.test_data)
    da = ds[args.target_var]
    arr = da.values  # normalized, (time, lat, lon, altitude)
    print("Test array:", da.dims, arr.shape)
    lats = ds["lat"].values if "lat" in ds.coords else np.arange(arr.shape[1])
    lons = ds["lon"].values if "lon" in ds.coords else np.arange(arr.shape[2])
    alts = ds["altitude"].values if "altitude" in ds.coords else np.arange(arr.shape[3])
    times = ds["time"].values if "time" in ds.coords else None
    scaler = joblib.load(args.scaler)

    s = args.seq_len
    n = arr.shape[0] - s  # number of windows with pred_len=1
    obs = inverse(scaler, arr[s:])  # targets, dBZ, (N, lat, lon, alt)
    target_times = times[s:] if times is not None else None

    contig = contiguous_mask(times, s, args.pred_len, args.step_minutes) if times is not None else None
    if contig is not None:
        print(f"Windows: {n} | contiguous (all steps = {args.step_minutes} min): {contig.sum()} "
              f"| crossing a time gap: {(~contig).sum()}")

    preds = {"Persistence": inverse(scaler, arr[s - 1:-1])}
    extra = {}

    if args.pysteps:
        try:
            preds["PySTEPS (LK)"], extra["pysteps_fallbacks"] = pysteps_forecast(arr, scaler, s)
        except ImportError as e:
            print(f"WARNING: PySTEPS not available ({e}); skipping. Install: pip install pysteps opencv-python-headless")

    for spec in args.model:
        label, path, arch = parse_model_spec(spec)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found; skipping {label}")
            continue
        print(f"Predicting with {label} ({arch}) from {path}")
        preds[label] = inverse(scaler, predict_model(path, arch, arr, s, args.pred_len, args.batch_size))

    # ---------------- metrics ----------------
    rows, rows_alt = [], []
    subsets = {"all": np.ones(n, bool)}
    if contig is not None and (~contig).any():
        subsets["contiguous"] = contig
    for name, p in preds.items():
        for sub, mask in subsets.items():
            rows.append({"model": name, "subset": sub, "n_windows": int(mask.sum()),
                         **compute_metrics(obs[mask], p[mask], args.thresholds)})
        for a in range(obs.shape[-1]):
            err = p[..., a] - obs[..., a]
            rows_alt.append({"model": name, "altitude_m": float(alts[a]),
                             "RMSE_dBZ": float(np.sqrt(np.nanmean(err ** 2))),
                             "MAE_dBZ": float(np.nanmean(np.abs(err)))})

    df = pd.DataFrame(rows)
    df_alt = pd.DataFrame(rows_alt)
    df.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    df_alt.to_csv(os.path.join(out_dir, "metrics_by_altitude.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "extra": extra, "metrics": rows}, f, indent=2, default=str)
    cols = ["model", "subset", "MSE_dBZ2", "RMSE_dBZ", "MAE_dBZ", "bias_dBZ"] + \
           [c for c in df.columns if c.startswith(("RMSE_echo", "CSI@"))]
    with open(os.path.join(out_dir, "metrics_table.tex"), "w") as f:
        f.write(to_latex(df[cols]))

    pd.set_option("display.width", 200)
    print("\n=== Test metrics (dBZ) ===")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== RMSE by altitude (dBZ) ===")
    print(df_alt.pivot(index="altitude_m", columns="model", values="RMSE_dBZ").to_string(float_format=lambda v: f"{v:.2f}"))

    # ---------------- figures ----------------
    samples = [i for i in args.samples if 0 <= i < n]
    strongest = int(np.nanargmax(np.nanmax(obs.reshape(n, -1), axis=1)))
    samples = list(dict.fromkeys(samples + [strongest]))
    for i in samples:
        tl = f"({str(target_times[i])[:16]})" if target_times is not None else ""
        for a in args.alt_idx:
            if not 0 <= a < obs.shape[-1]:
                continue
            path = os.path.join(out_dir, f"sample{i}_alt{a}.png")
            plot_sample(i, a, obs, preds, lats, lons, alts, path, args.cmap, args.vmin, args.vmax, tl)
            print("Figure:", path)

    if args.save_preds:
        np.savez_compressed(os.path.join(out_dir, "predictions_dBZ.npz"), obs=obs,
                            **{k.replace(" ", "_"): v for k, v in preds.items()})
    print("\nDone. Results in", out_dir)


if __name__ == "__main__":
    main()
