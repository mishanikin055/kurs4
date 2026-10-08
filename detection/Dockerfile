FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY detection/requirements.in detection/requirements.lock.txt /tmp/
RUN python -m pip install --no-cache-dir -r /tmp/requirements.in -c /tmp/requirements.lock.txt && python -m pip check
ENV PYTHONPATH=/workspace PYTHONUNBUFFERED=1 HF_HOME=/workspace/detection/models/cache/huggingface TORCH_HOME=/workspace/detection/models/cache/torch YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
ENTRYPOINT ["python", "-m", "detection.cli"]
