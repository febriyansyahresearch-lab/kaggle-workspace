# Titanic — Binary Classification with Feature Engineering

**Dataset:** Titanic (Survival prediction)  
**Model:** RandomForest with balanced class weighting  

## Methodology

1. **Data Generation**: Synthetic passenger data (500 records)
2. **Feature Engineering**: Age imputation, family size, age bins, fare bins
3. **Classification**: RandomForest with `class_weight="balanced"`
4. **Evaluation**: Accuracy, precision/recall per class

## Usage

```bash
python -m projects.titanic.src.train
```
