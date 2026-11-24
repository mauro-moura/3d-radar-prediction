#!/usr/bin/env bash
# Revision experiments for the EAMC paper (items 1-4):
#   1. Persistence + PySTEPS (optical flow) baselines
#   2. Test metrics in dBZ (MSE, RMSE, MAE, bias, CSI/POD/FAR)
#   3. Figures with shared colour scale, colour bars and lon/lat axes
#   4. DLinear grid search with the kernel_size bug fixed (kernels 3, 5)
#
# Normally launched inside Docker by run_fatnode.sh. It also runs directly in
# any environment with the dependencies of docker/Dockerfile:
#   bash run_experiments.sh
#
# Variables you can override:
#   PYTHON=python3            interpreter
#   EXP=kernel_fix            experiment name for the new DLinear
#   SKIP_TRAIN=1              only evaluate (reuse DLinear_final_model_$EXP.keras)
#   CONVLSTM=outputs/ConvLSTM_12_test_run.h5   ConvLSTM model used in the paper
#   EVAL_ARGS="..."           extra args for evaluate_models.py
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
EXP=${EXP:-kernel_fix}
CONVLSTM=${CONVLSTM:-outputs/ConvLSTM_12_test_run.h5}
DLINEAR_OLD=${DLINEAR_OLD:-outputs/DLinear_final_model_test_run.h5}
DLINEAR_NEW=outputs/DLinear_final_model_${EXP}.keras
OUT_DIR=${OUT_DIR:-outputs/eval_${EXP}_$(date +%Y%m%d_%H%M%S)}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-2}

for f in output_data/train_data.nc output_data/test_data.nc scaler.save "$CONVLSTM" "$DLINEAR_OLD"; do
    [[ -e "$f" ]] || echo "WARNING: $f not found"
done

echo "== $(date) | host $(hostname) | python: $($PYTHON --version 2>&1)"
$PYTHON -c "import tensorflow as tf; print('TF', tf.__version__, '| GPUs:', tf.config.list_physical_devices('GPU'))"

# ---------------------------------------------------------------- item 4
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    echo "== [1/2] DLinear grid search (kernel_size fixed) — $(date)"
    $PYTHON model_train_grid_search_dlinear.py --exp-name "$EXP" --kernels 3 5
else
    echo "== [1/2] skipped training (SKIP_TRAIN=1)"
fi

# ---------------------------------------------------------- items 1, 2, 3
PYSTEPS_FLAG=""
if $PYTHON -c "import pysteps, cv2" >/dev/null 2>&1; then
    PYSTEPS_FLAG="--pysteps"
else
    echo "WARNING: pysteps/opencv not installed -> PySTEPS baseline skipped (pip install pysteps opencv-python-headless)"
fi

echo "== [2/2] Evaluation on the test set — $(date)"
# shellcheck disable=SC2086
$PYTHON evaluate_models.py \
    --out-dir "$OUT_DIR" \
    --model "DLinear (paper)=${DLINEAR_OLD}:dlinear" \
    --model "DLinear (k fixed)=${DLINEAR_NEW}:dlinear" \
    --model "ConvLSTM3D=${CONVLSTM}:convlstm" \
    --samples 200 --alt-idx 0 2 4 \
    --vmin 0 --vmax 70 \
    --save-preds $PYSTEPS_FLAG ${EVAL_ARGS:-}

echo "== Finished $(date). Results: $OUT_DIR (metrics.csv, metrics_table.tex, *.png)"
