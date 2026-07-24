# Titanic — Binary Classification with Feature Engineering

**Problem Statement:** Predict passenger survival from demographic and ticket data. Imbalanced classes and missing values require careful feature engineering and balanced model training.

## Methodology

1. **Data Generation**: Synthetic passenger data (500 records, 7 features)
2. **Feature Engineering**: Age imputation (median), family size, age bins (5 groups), fare bins (4 quantiles), sex encoding, `is_alone` flag
3. **Classification**: RandomForest with `class_weight="balanced"`
4. **Evaluation**: Accuracy, precision/recall per class

## Key Concepts

- Handling missing values in structured data
- Feature engineering from raw attributes
- Class imbalance mitigation via cost-sensitive learning

## References

- Kaggle "Titanic: Machine Learning from Disaster" competition
- Chawla, N. V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique". *JAIR*.

## Usage

```bash
python -m projects.titanic.src.train
```
