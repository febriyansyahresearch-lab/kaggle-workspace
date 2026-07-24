.PHONY: setup test train-iris train-titanic train-housing train-clustering lint clean

setup:
	pip install -r requirements.txt

test:
	python -m pytest projects/ -v

train-iris:
	python -m projects.iris.src.train

train-titanic:
	python -m projects.titanic.src.train

train-housing:
	python -m projects.housing.src.train

train-clustering:
	python -m projects.clustering.src.cluster

lint:
	ruff check . --ignore E501 || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
