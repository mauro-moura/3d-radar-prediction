# Runtime for run_experiments.sh. Code and data are mounted at /workspace,
# so the image only holds dependencies (rebuild only if these change).
#   CPU: docker build -t radar-eval:tf2.20 docker/
#   GPU: docker build --build-arg BASE=tensorflow/tensorflow:2.20.0-gpu -t radar-eval:tf2.20-gpu docker/
ARG BASE=tensorflow/tensorflow:2.20.0
FROM ${BASE}

# Packages go into a venv that sees the base image's TensorFlow
# (--system-site-packages). pip then never uninstalls anything from the base
# layers, which fails with "Directory not empty" on old Docker storage drivers.
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir \
        keras==3.11.3 \
        xarray netcdf4 pandas scikit-learn joblib matplotlib \
        pysteps opencv-python-headless

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHON=python3

WORKDIR /workspace
