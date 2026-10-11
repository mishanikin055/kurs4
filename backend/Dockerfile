FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
RUN useradd --uid 1000 --create-home app \
    && mkdir -p /var/lib/kurs4/images /var/lib/kurs4/locks \
    && chown -R app:app /var/lib/kurs4
COPY backend /app/backend
USER app
CMD ["uvicorn", "backend.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Two isolated Transformers versions share only the fixed Torch/CUDA runtime.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2 AS worker
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt /tmp/backend.txt
RUN python -m pip install --no-cache-dir -r /tmp/backend.txt
COPY backend/requirements-detection.in backend/requirements-annotation.in /tmp/
COPY detection/requirements.lock.txt /tmp/detection.lock
COPY annotation/requirements.lock.txt /tmp/annotation-original.lock
# Keep ML pins; the shared current DB runtime needs a newer typing-extensions.
RUN sed '/^typing_extensions==/d; /^typing-extensions==/d' /tmp/annotation-original.lock > /tmp/annotation.lock
RUN python -m venv --system-site-packages /opt/detection \
    && /opt/detection/bin/pip install --no-cache-dir -r /tmp/requirements-detection.in -c /tmp/detection.lock \
    && /opt/detection/bin/pip check
RUN python -m venv --system-site-packages /opt/annotation \
    && /opt/annotation/bin/pip install --no-cache-dir -r /tmp/requirements-annotation.in -c /tmp/annotation.lock \
    && /opt/annotation/bin/pip check
RUN useradd --uid 1000 --create-home app
COPY backend /app/backend
ENV PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
USER app
CMD ["python", "-m", "backend.tasks"]

FROM api AS test
USER root
COPY backend/requirements-dev.txt /app/backend/requirements-dev.txt
RUN pip install --no-cache-dir -r backend/requirements-dev.txt
COPY pyproject.toml /app/pyproject.toml
USER app
CMD ["python", "-m", "pytest", "backend/tests", "-q"]
