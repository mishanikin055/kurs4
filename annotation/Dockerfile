FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY annotation/requirements.in annotation/requirements.lock.txt /tmp/
RUN python -m pip install --no-cache-dir -r /tmp/requirements.in -c /tmp/requirements.lock.txt && python -m pip check && chmod 1777 /opt/conda/lib/python3.11/site-packages/pycocoevalcap/tokenizer
ENV NLTK_DATA=/workspace/data/annotation_evaluator/nltk_data PYTHONPATH=/workspace PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
ENTRYPOINT ["python", "-m", "annotation.cli"]
