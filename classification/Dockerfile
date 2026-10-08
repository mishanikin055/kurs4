FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY classification/requirements.in classification/requirements.lock.txt /tmp/
RUN python -m pip install --no-cache-dir -r /tmp/requirements.in -c /tmp/requirements.lock.txt && python -m pip check
ENV PYTHONPATH=/workspace PYTHONUNBUFFERED=1 TORCH_HOME=/workspace/classification/models/cache/torch HF_HOME=/workspace/data/imagenet/hf-cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
ENTRYPOINT ["python", "-m", "classification.cli"]
