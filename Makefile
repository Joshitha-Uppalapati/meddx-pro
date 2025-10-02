test:
	pytest -q

run-api:
	PYTHONPATH=src uvicorn meddx.serve.api:app --reload --port 8000

run-ui:
	streamlit run src/meddx/serve/ui_app.py

train:
	PYTHONPATH=src python3 -m meddx.train

eval:
	PYTHONPATH=src python3 -m meddx.evaluate
