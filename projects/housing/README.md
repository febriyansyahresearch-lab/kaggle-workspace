# Housing — Regression (Price Prediction)

**Problem Statement:** Predict continuous housing prices from geographic and demographic features. Multiple regression algorithms are benchmarked to identify the best-performing model for this task.

## Methodology

1. **Data Generation**: Synthetic California-style housing data (500 records, 8 features: income, age, rooms, bedrooms, population, occupancy, lat, lon)
2. **Models**: Linear Regression, RandomForest, Gradient Boosting — compared on same split
3. **Evaluation**: RMSE, R² score

## Key Concepts

- Regression model comparison and selection
- Feature-target correlation analysis
- Ensemble methods for variance reduction

## References

- Pace, R. K. & Barry, R. (1997). "Sparse spatial autoregressions". *Statistics & Probability Letters*.
- Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine". *Annals of Statistics*.

## Usage

```bash
python -m projects.housing.src.train
```
