# Kaggle Project — ML Workspace

[![CI](https://github.com/febriyansyahresearch-lab/kaggle-project/actions/workflows/test.yml/badge.svg)](https://github.com/febriyansyahresearch-lab/kaggle-project/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)](projects/)

**Febriyansyah** — MTI, IT Security Leader (15+ yrs, Banking)

Monorepo of ML/data science projects for Kaggle, Colab, VS Code, and GitHub Codespaces.

## Projects

| Project | Type | Model | Tests |
|---|---|---|---|
| `projects/iris/` | Multi-class classification | RandomForest + FastAPI | 8 ✅ |
| `projects/titanic/` | Binary classification | RandomForest (balanced) | 6 ✅ |
| `projects/housing/` | Regression | Linear, RF, GBR | 6 ✅ |
| `projects/clustering/` | Unsupervised (K-Means) | K-Means + PCA | 5 ✅ |

## Setup

```bash
pip install -r requirements.txt
```

## Test

```bash
python -m pytest projects/ -v
```

## Usage

```bash
# Iris: train + API
python -m projects.iris.src.train
uvicorn projects.iris.src.api:app

# Titanic: train
python -m projects.titanic.src.train

# Housing: train and compare models
python -m projects.housing.src.train

# Clustering: run K-Means
python -m projects.clustering.src.cluster
```
