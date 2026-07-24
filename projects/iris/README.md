# Iris Classifier — Multi-Class Classification

**Problem Statement:** Classify iris flower species (setosa, versicolor, virginica) from sepal and petal measurements — a canonical multi-class benchmark for supervised learning pipelines with deployment readiness.

## Methodology

1. **Preprocessing**: StandardScaler normalization, LabelEncoder for target
2. **Classification**: RandomForest (100 estimators)
3. **Metrics**: Accuracy, precision/recall/F1 per class
4. **Deployment**: FastAPI REST API with `/predict` and `/health` endpoints

## Key Concepts

- Multi-class classification with imbalanced corpus
- Ensemble learning (bagging + random subspace)
- API-first design for model serving

## References

- Fisher, R. A. (1936). "The use of multiple measurements in taxonomic problems". *Annals of Eugenics*.
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python". *JMLR*.

## Usage

```bash
python -m projects.iris.src.train
uvicorn projects.iris.src.api:app
```
