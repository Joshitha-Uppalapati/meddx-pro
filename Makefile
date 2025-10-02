test:
	PYTHONPATH=src pytest -q

run-api:
	PYTHONPATH=src uvicorn meddx.serve.api:app --reload --port 8000

run-ui:
	streamlit run src/meddx/serve/ui_app.py

run-all:
	PYTHONPATH=src uvicorn meddx.serve.api:app --reload --port 8000 & \
	streamlit run src/meddx/serve/ui_app.py
