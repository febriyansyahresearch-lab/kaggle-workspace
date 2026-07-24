# Customer Segmentation — Unsupervised Learning

**Problem Statement:** Segment customers into distinct groups based on purchasing behavior without labeled data — a common marketing analytics task for personalized targeting.

## Methodology

1. **Data Generation**: 300 customer records with 4 features (age, income, spending score, purchase frequency) from 3 latent clusters
2. **Clustering**: K-Means with elbow method for optimal k determination
3. **Dimensionality Reduction**: PCA for 2D visualization of cluster separation
4. **Evaluation**: Inertia, explained variance ratio

## Key Concepts

- Unsupervised learning without ground truth labels
- K-Means clustering with Euclidean distance
- Elbow method for hyperparameter selection
- PCA for feature space visualization

## References

- Lloyd, S. P. (1982). "Least squares quantization in PCM". *IEEE Trans. Information Theory*.
- Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer.

## Usage

```bash
python -m projects.clustering.src.cluster
```
