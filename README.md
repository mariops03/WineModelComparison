# 🍷 Wine Model Comparison

Comparative analysis of 5 machine learning classification models on the [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine). The goal is to classify wines into 3 cultivars based on 13 chemical properties, evaluating each model's performance through multiple metrics and statistical testing.

## Models Compared

| Model | Type |
|-------|------|
| K-Nearest Neighbors (KNN) | Instance-based |
| Support Vector Machine (SVM) | Hyperplane-based |
| Decision Tree | Rule-based |
| Random Forest | Ensemble |
| Neural Network (MLP) | Multi-layer perceptron |

## Results

| Model | Test Accuracy |
|-------|:------------:|
| SVM | **100%** |
| Random Forest | **100%** |
| Neural Network (MLP) | **100%** |
| KNN | 94.44% |
| Decision Tree | 94.44% |

A paired t-test (5-fold CV) between KNN and SVM showed no statistically significant difference in cross-validation performance despite the accuracy gap on the test set.

## Evaluation Metrics

Each model is evaluated on:
- **Accuracy** on the held-out test set (80/20 split)
- **Precision, Recall, F1-score** per class via classification report
- **Confusion matrix** heatmaps (seaborn)
- **AUC ROC** score (one-vs-rest, multi-class)
- **PCA scatter plot** (2 components) for class separability visualization
- **Bar chart** comparing accuracy across all models

## How to Run

```bash
# Install dependencies (no requirements.txt — install manually)
pip install scikit-learn matplotlib seaborn scipy

# Run the analysis
python analisis.py
```

The script loads the Wine dataset directly from `sklearn.datasets`, so no external data files are needed. All plots are displayed interactively via `matplotlib`.

## Project Structure

```
├── analisis.py     # Full analysis pipeline: preprocessing, training, evaluation, visualization
└── README.md
```

## Technical Details

- **Preprocessing**: StandardScaler normalization on all features
- **Split**: 80% train / 20% test (`random_state=42`)
- **Cross-validation**: 5-fold CV used for the paired t-test comparison
- **Dataset**: 178 samples, 13 features, 3 classes (built-in sklearn)
