# Kaggle Project — ML Workspace

**Febriyansyah** — MTI, IT Security Leader (15+ yrs, Banking)

Monorepo of ML/data science projects for Kaggle, Colab, VS Code, and GitHub Codespaces.

## Projects

| Project | Type | Model | Tests |
|---|---|---|---|
| `projects/iris/` | Multi-class classification | RandomForest + FastAPI | 8 ✅ |
| `projects/titanic/` | Binary classification | RandomForest (balanced) | 6 ✅ |
| `projects/housing/` | Regression | Linear, RF, GBR | 6 ✅ |
| `projects/clustering/` | Unsupervised (K-Means) | K-Means + PCA | 6 ✅ |

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
