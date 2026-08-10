.PHONY: install test lint train

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

train:
	python scripts/train.py --csv data/raw/AMZN.csv --config configs/timekan.yaml --output-dir runs/amzn
