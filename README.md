# WineModelComparison

This repository presents an analysis and comparison of various classification techniques applied to the well-known Wine Dataset. The aim is to predict wine categories based on chemical characteristics, utilizing multiple machine learning models and evaluating their performance.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models Evaluated](#models-evaluated)
- [Results and Evaluation](#results-and-evaluation)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
- [References](#references)

## Overview
The analysis explores how different machine learning algorithms can effectively classify wines into three categories based on 13 chemical properties. This classification problem provides insights into model performance in a multi-class setting with a small dataset. The analysis and experiments were conducted using Python's `scikit-learn` library.

## Dataset
The dataset used is the Wine Dataset from the UCI Machine Learning Repository, available within `scikit-learn`. It consists of 178 samples divided into three classes, each representing a distinct cultivar. The 13 features include attributes such as alcohol content, magnesium levels, color intensity, and others, which are useful for wine classification tasks.

## Installation
To run this project, ensure you have Python installed. Install the required packages with:

```bash
pip install -r requirements.txt
```

**Dependencies**:
- scikit-learn
- matplotlib
- seaborn
- scipy

## Usage
To execute the analysis, run `analisis.py`. This script loads the data, scales it, applies multiple classification algorithms, and outputs comparative results, including accuracy, confusion matrices, and AUC ROC scores.

```bash
python analisis.py
```

## Models Evaluated
The following classification models were implemented and compared:
1. **K-Nearest Neighbors (KNN)** - A non-parametric model that classifies based on proximity to the closest neighbors.
2. **Support Vector Machine (SVM)** - Finds an optimal separating hyperplane, with kernel functions to handle non-linear boundaries.
3. **Decision Tree** - Builds a decision tree based on feature splits, easy to interpret but prone to overfitting.
4. **Random Forest** - An ensemble of decision trees providing robust performance and reduced variance.
5. **Neural Network (MLP)** - A multi-layer perceptron, effective for complex patterns in small datasets.

## Results and Evaluation
Each model was evaluated based on:
- **Accuracy Score**: Overall classification accuracy.
- **Classification Report**: Precision, recall, and F1-score per class.
- **Confusion Matrix**: Visual representation of true vs. predicted classifications.
- **AUC ROC Score**: For multi-class classification performance.
  
**Summary of Results**:
- **SVM, Random Forest, and Neural Network** achieved perfect classification accuracy (100%).
- **KNN and Decision Tree** reached an accuracy of 94.44%.
- A paired t-test between KNN and SVM showed no statistically significant difference, suggesting comparable performance.

## Visualization
The following visualizations are included:
1. **Confusion Matrices** - Visualized for each model to display classification performance.
2. **Model Comparison** - A bar plot comparing accuracy scores across all models.
3. **PCA Scatter Plot** - Two principal components visualize the wine classes and highlight the separability of data points.

## Conclusion
The analysis demonstrates that SVM, Random Forest, and Neural Network are highly effective for wine classification, achieving the highest performance. While KNN and Decision Tree also performed well, their accuracy was slightly lower and showed susceptibility to noise and parameter sensitivity.

## References
This project was inspired by the following sources:

- Scikit-Learn Documentation - [Pedregosa et al., 2011](https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf)
- UCI Machine Learning Repository - Wine Dataset
- Academic resources on machine learning techniques for classification:
  - [https://doi.org/10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J)
  - [IEEE on Random Forests](https://ieeexplore.ieee.org/document/4160265)
  - [Research on Neural Networks in Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4916348/)
- Additional tutorials and examples:
  - PCA in 3D - [Scikit-Learn PCA Visualization](https://scikit-learn.org/0.22/auto_examples/decomposition/plot_pca_3d.html)
  - [StandardScaler Usage Guide](https://interactivechaos.com/es/manual/tutorial-de-machine-learning/standardscaler)
