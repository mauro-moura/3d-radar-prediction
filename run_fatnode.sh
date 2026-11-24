#!/usr/bin/env bash
# Run the revision experiments (run_experiments.sh) in Docker on the fatnode.
#
#   bash run_fatnode.sh              # build image if needed, start in background
#   tail -f logs/<name>.log          # follow progress (path printed at start)
#
# Variables:
#   GPU=auto|0|1          use the GPU image + --gpus all (auto: if nvidia-smi works)
#   SKIP_TRAIN=1          only evaluate (no DLinear retraining)
#   EXP, CONVLSTM, DLINEAR_OLD, EVAL_ARGS   forwarded to run_experiments.sh
#   REBUILD=1             force rebuilding the image
#   CPUS=16 MEMORY=64g    optional docker resource limits
set -euo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-auto}
if [[ "$GPU" == auto ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=1; else GPU=0; fi
fi

BASE=tensorflow/tensorflow:2.20.0
IMAGE=${IMAGE:-radar-eval:tf2.20}
RUN_ARGS=()
if [[ "$GPU" == 1 ]]; then
    # Docker >= 19.03 has --gpus; older versions (e.g. 18.09) need the nvidia runtime (nvidia-docker2).
    DOCKER_VER=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 0)
    if printf '%s\n19.03\n' "$DOCKER_VER" | sort -V -C; then
        if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
            RUN_ARGS+=(--runtime=nvidia -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}")
        else
            echo "WARNING: Docker $DOCKER_VER has no --gpus and no 'nvidia' runtime (install nvidia-docker2). Using CPU."
            GPU=0
        fi
    else
        RUN_ARGS+=(--gpus all)
    fi
fi
if [[ "$GPU" == 1 ]]; then
    BASE=tensorflow/tensorflow:2.20.0-gpu
    IMAGE=${IMAGE}-gpu
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "NVIDIA driver: ${DRIVER:-unknown} (TensorFlow 2.20 GPU needs CUDA 12 => driver >= 525)"
    if [[ -n "$DRIVER" ]] && ! printf '525\n%s\n' "$DRIVER" | sort -V -C; then
        echo "WARNING: driver older than 525: TensorFlow will not see the GPU. Use GPU=0 to run on CPU."
    fi
fi
[[ -n "${CPUS:-}" ]] && RUN_ARGS+=(--cpus "$CPUS")
[[ -n "${MEMORY:-}" ]] && RUN_ARGS+=(--memory "$MEMORY")

if [[ "${REBUILD:-0}" == 1 ]] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE (base $BASE)..."
    docker build --build-arg BASE="$BASE" -t "$IMAGE" docker/
fi

for f in output_data/train_data.nc output_data/test_data.nc scaler.save; do
    [[ -e "$f" ]] || echo "WARNING: $f not found in $(pwd)"
done

NAME=${NAME:-radar-eval-$(date +%Y%m%d_%H%M%S)}
mkdir -p logs outputs
LOG=logs/${NAME}.log

docker run -d --rm --name "$NAME" \
    ${RUN_ARGS[@]+"${RUN_ARGS[@]}"} \
    --user "$(id -u):$(id -g)" \
    --shm-size=2g \
    -v "$(pwd)":/workspace -w /workspace \
    -e SKIP_TRAIN -e EXP -e CONVLSTM -e DLINEAR_OLD -e EVAL_ARGS \
    "$IMAGE" \
    bash -c "bash run_experiments.sh 2>&1 | tee '$LOG'; exit \${PIPESTATUS[0]}" >/dev/null

echo "Started container $NAME (GPU=$GPU, image $IMAGE)"
echo "  log:   tail -f $LOG"
echo "  stop:  docker stop $NAME"
