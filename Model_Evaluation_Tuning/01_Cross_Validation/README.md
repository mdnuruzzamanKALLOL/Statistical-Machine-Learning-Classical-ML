# 🔁 Cross-Validation

> Status: ✅ Complete — [Open the notebook →](01_cross_validation.ipynb)

Every algorithm notebook in this series has used `cross_val_score` without fully deriving *why*. This topic formalizes it: a single train/test split gives one noisy estimate of performance — cross-validation averages over many splits to get a far more reliable one.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [CV Strategy Reference](#-cv-strategy-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

A single train/test split answers "how did the model do on *this* split?" Cross-validation answers the more useful question: "how would the model do on *any* reasonable split, on average, and how much does that vary?"

```
K-Fold (K=5):
Fold 1: [TEST][train][train][train][train]
Fold 2: [train][TEST][train][train][train]
Fold 3: [train][train][TEST][train][train]
Fold 4: [train][train][train][TEST][train]
Fold 5: [train][train][train][train][TEST]
                    ↓
        5 scores → mean ± std
```

Every point gets tested exactly once and trained on $K-1$ times — a far more complete use of limited data than a single 80/20 split.

---

## 🎯 Why This Topic Matters

- Every `cross_val_score` call from Classification and Ensemble Techniques onward was built on the exact mechanism derived here for the first time.
- **Nested CV** (§11) is the one concept in this notebook that changes how the *next* topic (Hyperparameter Tuning) must be done correctly — tuning and final reporting cannot share the same CV loop.
- **TimeSeriesSplit** (§10) is a leakage trap distinct from anything covered in Foundation's preprocessing topic — shuffling isn't just unnecessary for time-ordered data, it's actively wrong.
- The notebook's own honest results (§7, §11) — where the textbook effect didn't show up dramatically on this specific dataset — are as instructive as a clean demonstration: real effect sizes depend on data, and small-sample statistics are noisy.

---

## 🧮 Mathematical Explanation

### 1. K-Fold cross-validation estimate

Partition data into $K$ folds $F_1, \dots, F_K$. For each fold $k$, train on all data except $F_k$, evaluate on $F_k$ to get score $s_k$. The CV estimate is:

$$\overline{CV} = \frac{1}{K}\sum_{k=1}^K s_k, \qquad \widehat{\sigma}_{CV} = \sqrt{\frac{1}{K}\sum_{k=1}^K (s_k - \overline{CV})^2}$$

$\overline{CV}$ is a lower-variance estimate of true generalization performance than any single $s_k$, since it's an average over $K$ roughly-independent measurements rather than one.

### 2. Why averaging reduces variance

For $K$ measurements with individual variance $\sigma^2$ each, if they were fully independent:

$$\text{Var}(\overline{CV}) = \frac{\sigma^2}{K}$$

In reality, folds share overlapping training data (fold 1's training set and fold 2's training set overlap in $K-2$ of $K$ parts), so they're not fully independent, and the true variance reduction is less than the idealized $\sigma^2/K$ — but the direction (more folds → more stable estimate, up to a point) still holds. This is the exact same averaging-reduces-variance principle from bagging (Ensemble Techniques topic 01), now applied to *evaluation* instead of *prediction*.

### 3. The bias-variance tradeoff in choosing $K$

- **Small $K$** (e.g. $K=2$): each training fold is small (half the data), so the model trained per fold is a worse approximation of the model trained on everything — **higher bias** in the estimate. Fewer folds also means fewer, noisier measurements to average — but each is cheap to compute.
- **Large $K$** (up to $K=n$, i.e. LOOCV): each training fold is nearly the full dataset, so bias is minimal — but computing $n$ models is expensive, and LOOCV's $n$ scores are *highly correlated* with each other (each training set differs by only one point), so its variance-reduction benefit is smaller than $K=n$ would naively suggest.

$K=5$ or $K=10$ is the standard practical compromise — low enough bias, computationally reasonable, and empirically stable across most datasets.

### 4. Stratification

For classification with class proportions $p_1, \dots, p_c$ in the full dataset, `StratifiedKFold` constructs each fold so that fold $k$'s class proportions match $p_1, \dots, p_c$ as closely as possible — removing fold-composition class-imbalance as a source of noise in the CV estimate, independent of genuine model variance.

### 5. Nested cross-validation's bias correction

Using CV both to select hyperparameters (via `GridSearchCV`'s internal CV) and to report a final score on the *same* folds creates optimistic bias: the selected hyperparameters were implicitly chosen because they performed well on exactly the data now being used to judge them — a form of the same "testing on training data" mistake, one level removed. Nested CV separates this into an **outer loop** (never touched during tuning, used only for final scoring) and an **inner loop** (used only for tuning), so the reported score is honest.

---

## 📋 CV Strategy Reference

| Strategy | Use when | Key property |
|---|---|---|
| `KFold` | Regression, balanced classification | Simple, equal-size folds |
| `StratifiedKFold` | Classification (especially imbalanced) | Preserves class proportions per fold |
| `LeaveOneOut` | Very small datasets | Maximum training data per fold, expensive |
| `RepeatedKFold` | When the CV estimate itself needs to be more stable | Averages over multiple K-Fold shuffles |
| `TimeSeriesSplit` | Time-ordered / sequential data | Never trains on future to predict past |
| Nested CV (`GridSearchCV` inside an outer loop) | Whenever tuning AND reporting a final score | Removes tuning-induced optimistic bias |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Using plain `KFold` on imbalanced classification data** — a fold can end up with very few (or zero) minority-class examples purely by chance; always prefer `StratifiedKFold`.
2. **Shuffling time-ordered data** — `TimeSeriesSplit` exists specifically because a normal shuffle-based CV would let a model "see the future" during training, silently inflating scores in a way that never happens in real deployment.
3. **Reporting `GridSearchCV.best_score_` as a final, honest performance number** — it's the *tuning* metric, not a held-out test score; it carries the same optimistic bias nested CV is designed to remove (notebook §11 measured this directly: +0.0158 with a wide hyperparameter grid).
4. **Assuming more folds is always better** — LOOCV's $n$ highly-correlated estimates don't reduce variance as much as $n$ independent ones would, while costing $n$ full model fits; $K=5$–$10$ is usually the better tradeoff.
5. **Comparing algorithms using different CV folds for each** — always fix and reuse the same `cv` object (or the same `random_state`) across every model being compared (notebook §12), or fold-composition luck becomes a hidden confound.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.model_selection.KFold`, `StratifiedKFold` | Core K-Fold splitting strategies |
| `sklearn.model_selection.LeaveOneOut`, `RepeatedKFold` | Specialized CV variants |
| `sklearn.model_selection.TimeSeriesSplit` | Chronological, no-shuffle splitting |
| `sklearn.model_selection.cross_val_score`, `cross_validate` | Run CV and collect scores |
| `sklearn.model_selection.learning_curve` | CV score vs. training set size |
| `sklearn.model_selection.GridSearchCV` | Hyperparameter search (used here inside nested CV) |

---

## 📝 Self-Test Exercises

1. Using the variance formula in §2, explain why $K=10$'s CV estimate is more stable than $K=2$'s, even before considering bias.
2. Explain, using §3, why LOOCV's $n$ scores are more correlated with each other than $K=5$'s 5 scores are.
3. Given an imbalanced dataset with a rare class, describe a concrete scenario where plain `KFold` could produce a fold with zero examples of that class — connect it to why `StratifiedKFold` prevents this by construction.
4. Explain why `TimeSeriesSplit` never uses a shuffled split, tying your answer to what "generalization" means for a forecasting model deployed in the real world.
5. Notebook §11 found a positive (theory-matching) leakage gap only after widening the hyperparameter grid — explain, using §5, why a *wider* search makes the naive (non-nested) approach's optimism more pronounced.

---

## 📓 Notebook

30 executed code cells: a demonstration of single-split variance across 20 random seeds, a from-scratch K-Fold implementation matched against `cross_val_score`, fold-membership visualization, a Stratified vs plain KFold class-balance comparison (with an honest note when the std-reduction effect didn't show up dramatically on this run), LOOCV cost vs. 5-Fold, RepeatedKFold stability, TimeSeriesSplit visualized, a nested cross-validation implementation that correctly measured optimistic bias (+0.0158) once given a wide enough hyperparameter grid, a fair multi-algorithm comparison on shared folds, learning curves, `cross_validate` with multiple metrics, and a $K$-choice sensitivity check:

➡️ **[01_cross_validation.ipynb](01_cross_validation.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Visualizing cross-validation behavior](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)
- [scikit-learn: Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

---
[← Back to Model Evaluation & Tuning](../README.md)
