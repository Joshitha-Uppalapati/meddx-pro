#!/usr/bin/env bash
set -e
export PYTHONPATH=/app/src
uvicorn meddx.serve.api:app --host 0.0.0.0 --port 8000 &
streamlit run /app/src/meddx/serve/ui_app.py --server.port 8501 --server.address 0.0.0.0
