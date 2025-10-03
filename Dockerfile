# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src
WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt
COPY src /app/src
COPY artifacts/ /app/artifacts/
COPY data/processed/ /app/data/processed/

FROM base AS runtime
COPY --from=builder /usr/local /usr/local
WORKDIR /app
COPY src /app/src
COPY artifacts/ /app/artifacts/
COPY data/processed/ /app/data/processed/
COPY entrypoint.sh /app/entrypoint.sh
EXPOSE 8000 8501
ENTRYPOINT ["/app/entrypoint.sh"]
