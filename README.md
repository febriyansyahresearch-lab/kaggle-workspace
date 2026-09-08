# Kaggle Workspace — ML & Data Science

[![CI](https://github.com/febriyansyahresearch-lab/kaggle-workspace/actions/workflows/test.yml/badge.svg)](https://github.com/febriyansyahresearch-lab/kaggle-workspace/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-25%20passed-brightgreen)](projects/)

**Febriyansyah** — MTI, IT Security Leader (15+ yrs, Banking)

Monorepo of ML/data science projects and Kaggle notebooks for Kaggle, Colab, VS Code, and GitHub Codespaces.

## Projects

| Project | Type | Model | Tests |
|---|---|---|---|
| `projects/iris/` | Multi-class classification | RandomForest + FastAPI | 8 ✅ |
| `projects/titanic/` | Binary classification | RandomForest (balanced) | 6 ✅ |
| `projects/housing/` | Regression | Linear, RF, GBR | 6 ✅ |
| `projects/clustering/` | Unsupervised (K-Means) | K-Means + PCA | 5 ✅ |

## Kaggle Notebooks

13 notebooks, each in its own folder under `notebooks/` (contains `<slug>.ipynb` + `kernel-metadata.json`), following the naming convention documented in [`docs/kaggle-naming-convention.md`](docs/kaggle-naming-convention.md).

| Category | Slug | Title |
|---|---|---|
| `classify-` | `classify-malimg` | Malimg Classification |
| `classify-` | `classify-breast-cancer` | Breast Cancer Classification |
| `eda-` | `eda-iris` | Iris EDA |
| `eda-` | `eda-titanic` | Titanic EDA & Feature Engineering |
| `eda-` | `eda-housing` | California Housing EDA |
| `eda-` | `eda-customer-segmentation` | Customer Segmentation EDA |
| `vision-` | `vision-malevis-generator` | Mini MaleVis Generator |
| `vision-` | `vision-image-size-check` | Image Size Check |
| `nlp-` | `nlp-retail-chatbot-rag` | Retail Chatbot RAG |
| `nlp-` | `nlp-retail-cs-ai` | Retail CS AI |
| `exp-` | `exp-transformer-mamba` | Transformer Mamba Experiment |
| `exp-` | `exp-transformer-test` | Transformer Model Test |
| `auto-` | `auto-github-runner` | GitHub Runner Automation |

### Push notebook ke Kaggle

```bash
# Aktifkan venv + token OAuth (kaggle CLI 2.x butuh ini untuk perintah kernels)
export PATH="/path/ke/.venv/bin:$PATH"
export KAGGLE_API_TOKEN=$(kaggle auth print-access-token)

# Push satu notebook
cd notebooks/eda-iris
kaggle kernels push

# Push semua notebook
for d in notebooks/*/; do (cd "$d" && kaggle kernels push); done
```

> Catatan: Kaggle membatasi 5 sesi CPU bersamaan — push berurutan dengan jeda bila perlu.

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
