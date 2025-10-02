.PHONY: format lint test run-api run-ui

run-api:
\tPYTHONPATH=src uvicorn meddx.serve.api:app --reload --port 8000

run-ui:
\tstreamlit run src/meddx/serve/ui_app.py

test:
\tPYTHONPATH=src pytest -q
