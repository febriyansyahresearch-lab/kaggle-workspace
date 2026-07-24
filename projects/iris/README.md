# Iris Classifier — Multi-Class Classification

**Dataset:** Iris (3 species: setosa, versicolor, virginica)  
**Model:** RandomForest  
**Deployment:** FastAPI  

## Methodology

1. Preprocessing: StandardScaler, LabelEncoder
2. Classification: RandomForest (100 estimators)
3. Metrics: Accuracy, precision/recall per class
4. API: FastAPI with `/predict` and `/health` endpoints

## Usage

```bash
python -m projects.iris.src.train
uvicorn projects.iris.src.api:app
```
