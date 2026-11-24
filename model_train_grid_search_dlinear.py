"""
Grid search + final training for DLinear.

Fixes w.r.t. the version used in the EAMC paper:
  * `kernel_size` from the grid is now actually passed to the model (before it
    was silently ignored and every run used kernel_size=25).
  * Default kernels are 3 and 5: with seq_len=10, kernels >= 10 average the
    whole input window, so the old values (15, 25) were equivalent.
  * Per-fold validation MSE (and mean/std across folds) is saved to JSON.

The cross-validation protocol is kept identical to the paper (shuffled 5-fold
KFold over sliding windows, seed 42) so the results stay comparable.

Usage:
    python model_train_grid_search_dlinear.py --exp-name kernel_fix
    python model_train_grid_search_dlinear.py --kernels 3 5 9 --epochs 50 100
"""
import os
import json
import argparse

import numpy as np
import xarray as xr
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold, ParameterGrid

from models.dlinear import build_dlinear
from utils import prepare_data


def make_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]


def build_model(seq_len, features, pred_len, kernel_size):
    model = build_dlinear(seq_len=seq_len, features=features, pred_len=pred_len, kernel_size=kernel_size)
    model.optimizer.learning_rate = 1e-3
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default="output_data/train_data.nc")
    parser.add_argument("--target-var", default="dbzh")
    parser.add_argument("--output-path", default="outputs/")
    parser.add_argument("--exp-name", default="kernel_fix")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--pred-len", type=int, default=1)
    parser.add_argument("--kernels", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--epochs", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-fold-models", action="store_true",
                        help="Also save each fold model (not needed for evaluation).")
    args = parser.parse_args()

    for k in args.kernels:
        if k >= args.seq_len:
            print(f"WARNING: kernel_size={k} >= seq_len={args.seq_len}: the trend becomes ~the window mean.")

    os.makedirs(args.output_path, exist_ok=True)
    cfg = {"architecture": "DLinear"}

    ds = xr.open_dataset(args.data_path)
    arr = ds[args.target_var].values
    X, y = prepare_data(arr, args.seq_len, args.pred_len, cfg=cfg)
    seq_len, features = X.shape[1], X.shape[2]
    print("X shape:", X.shape, "| y shape:", y.shape)

    grid = list(ParameterGrid({
        "kernel_size": args.kernels,
        "batch_size": args.batch_sizes,
        "epochs": args.epochs,
    }))
    print(f"Total combinations: {len(grid)}")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    results = []
    for gi, params in enumerate(grid):
        print(f"[{gi + 1}/{len(grid)}] Testing params: {params}")
        fold_mse = []
        for fi, (train_idx, val_idx) in enumerate(kf.split(X)):
            model = build_model(seq_len, features, args.pred_len, params["kernel_size"])
            model.fit(
                X[train_idx], y[train_idx],
                validation_data=(X[val_idx], y[val_idx]),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=make_callbacks(),
                verbose=2,
            )
            val_mse = float(model.evaluate(X[val_idx], y[val_idx], verbose=0))
            fold_mse.append(val_mse)
            print(f"   fold {fi + 1}: val MSE (normalized) = {val_mse:.6f}")
            if args.save_fold_models:
                model.save(os.path.join(args.output_path, f"DLinear_{args.exp_name}_g{gi}_f{fi}.keras"))

        results.append({
            "params": params,
            "fold_mse": fold_mse,
            "mse": float(np.mean(fold_mse)),
            "mse_std_across_folds": float(np.std(fold_mse, ddof=1)),
        })
        # write after each combination so partial results survive a crash
        with open(os.path.join(args.output_path, f"DLinear_grid_{args.exp_name}.json"), "w") as f:
            json.dump(results, f, indent=2)

    results = sorted(results, key=lambda r: r["mse"])
    best = results[0]
    print("Grid search completed. Best:", best)
    with open(os.path.join(args.output_path, f"DLinear_grid_{args.exp_name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Final model on all training windows with the best params
    bp = best["params"]
    model = build_model(seq_len, features, args.pred_len, bp["kernel_size"])
    hist = model.fit(X, y, epochs=bp["epochs"], batch_size=bp["batch_size"],
                     validation_split=0.15, callbacks=make_callbacks(), verbose=2)
    final_path = os.path.join(args.output_path, f"DLinear_final_model_{args.exp_name}.keras")
    model.save(final_path)
    np.save(os.path.join(args.output_path, f"DLinear_final_model_hist_{args.exp_name}.npy"), hist.history)
    print("Final model saved to", final_path)


if __name__ == "__main__":
    main()
