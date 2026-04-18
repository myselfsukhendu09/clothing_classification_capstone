.PHONY: install train dashboard api lint clean

install:
	pip install -e .

train:
	python -m scripts.train_models_pipeline

dashboard:
	streamlit run app/app.py

api:
	uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000

lint:
	ruff check .
	ruff format .

clean:
	rm -rf `find . -type d -name "__pycache__"`
	rm -rf .ruff_cache
	rm -rf *.egg-info
