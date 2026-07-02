# 📊 Statistical Machine Learning — Classical ML

Part of the [📘 Statistical Machine Learning for Noob](https://github.com/mdnuruzzamanKALLOL) series.

A complete tour of classical (pre-deep-learning) machine learning algorithms — regression, classification, ensembles, unsupervised learning, and model evaluation.

## 📚 Categories

| # | Category | Description | Algorithms | Status |
|---|----------|-------------|------------|--------|
| 01 | [Regression](Regression/) | Predicting continuous values | 8 | 🚧 |
| 02 | [Classification](Classification/) | Predicting discrete labels | 8 | 🚧 |
| 03 | [Ensemble Techniques](Ensemble_Techniques/) | Combining multiple models | 2 | 🚧 |
| 04 | [Unsupervised](Unsupervised/) | Clustering & dimensionality reduction | 7 | 🚧 |
| 05 | [Model Evaluation & Tuning](Model_Evaluation_Tuning/) | Validation, tuning, metrics | 4 | 🚧 |

## ⚖️ Algorithm Comparison (quick reference)

| Algorithm | Type | Interpretability | Handles Non-linearity |
|-----------|------|-------------------|------------------------|
| Linear/Logistic Regression | Both | High | No |
| Decision Tree | Both | High | Yes |
| Random Forest | Both | Medium | Yes |
| SVM | Both | Low | Yes (kernel) |
| KNN | Both | Medium | Yes |
| Naive Bayes | Classification | High | No |
| Gradient Boosting (XGBoost etc.) | Both | Low | Yes |
| K-Means / DBSCAN / GMM | Unsupervised | Medium | Varies |

## 📁 Structure

Each algorithm folder contains its own `README.md` (concept + math explanation) and a Jupyter notebook (`.ipynb`) with hands-on implementation.

## 📦 Datasets

All datasets — the 162-entry catalog ([CATALOG.md](https://github.com/mdnuruzzamanKALLOL/Datasets/blob/main/CATALOG.md)) and the actual CSV files — live in the central **[Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets)** repo, mapped to each algorithm category above.

## 🔗 Related

- [← Foundation](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation)
