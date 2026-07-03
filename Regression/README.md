# 📈 Regression

Predicting continuous values from input features — 8 algorithms, one shared real dataset ([Diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)) used for a running, honest comparison from the first topic to the last.

## 📑 Table of Contents

1. [Algorithms in This Category](#-algorithms-in-this-category)
2. [Recommended Learning Path](#-recommended-learning-path)
3. [Algorithm Comparison](#️-algorithm-comparison)
4. [Final Category Results](#-final-category-results)
5. [Cross-Topic Threads](#-cross-topic-threads)
6. [When to Use Which](#-when-to-use-which)
7. [Notebook Standard](#-notebook-standard)
8. [Datasets](#-datasets)

---

## 📚 Algorithms in This Category

| # | Algorithm | Use Case | Status |
|---|-----------|----------|--------|
| 01 | [Linear Regression](01_Linear_Regression/) | Simple linear relationships, the OLS/gradient descent foundation | ✅ |
| 02 | [Polynomial Regression](02_Polynomial_Regression/) | Non-linear curve fitting via feature expansion | ✅ |
| 03 | [Ridge, Lasso & ElasticNet](03_Ridge_Lasso_ElasticNet/) | Regularized regression, multicollinearity, feature selection | ✅ |
| 04 | [SVR](04_SVR/) | Kernel-based regression, epsilon-insensitive loss | ✅ |
| 05 | [KNN Regression](05_KNN_Regression/) | Local, non-parametric, instance-based estimation | ✅ |
| 06 | [Decision Tree Regression](06_Decision_Tree_Regression/) | Rule-based, non-linear splits, fully interpretable | ✅ |
| 07 | [Random Forest Regression](07_Random_Forest_Regression/) | Bagged ensemble of trees, reduced variance | ✅ |
| 08 | [Gradient Boosting Regression](08_Gradient_Boosting_Regression/) | Sequential boosted ensemble, highest typical accuracy | ✅ |

Every topic folder contains a deep-dive `README.md` (concept, full math derivation, pitfalls, self-test exercises) and a Jupyter notebook (30+ executed code cells of hands-on implementation, verified against sklearn, with honestly-reported results even when they don't match textbook expectations).

---

## 🧭 Recommended Learning Path

This category is built to be read in order — later topics deliberately reuse earlier topics' datasets, findings, and even specific numbers:

```
01 Linear Regression  ──┐  Normal equation, gradient descent, OLS assumptions (VIF flags s1-s5 in Diabetes)
02 Polynomial Regression│  Same OLS machinery, expanded features, extrapolation danger
03 Ridge/Lasso/ElasticNet├─ Closes the loop: fixes 01's multicollinearity, revisits 02's overfitting
04 SVR                  │  A structurally different objective (epsilon-insensitive loss) + the kernel trick
05 KNN Regression       │  No training phase at all -- instance-based, can't extrapolate
06 Decision Tree        ├─ Ends on a deliberately weak result: single-tree instability, weakest R^2 so far
07 Random Forest        │  Directly re-tests 06's instability finding -- result is honestly mixed
08 Gradient Boosting  ──┘  Sequential correction instead of parallel averaging; closes with all 8 compared
```

Topics 01→03 and 06→07→08 are the two threads most worth reading consecutively; 04 and 05 stand more independently as alternative modeling philosophies (kernels, instance-based).

---

## ⚖️ Algorithm Comparison

| Algorithm | Model Type | Interpretability | Handles Non-Linearity | Needs Feature Scaling | Training Cost | Extrapolation Behavior |
|---|---|---|---|---|---|---|
| Linear Regression | Parametric, linear | High | No | Only for gradient descent, not the normal equation | Very low | Diverges linearly, unbounded |
| Polynomial Regression | Parametric, linear-in-params | Medium | Yes (via expansion) | Yes, especially at high degree | Low | Diverges wildly at high degree (Runge's phenomenon) |
| Ridge / Lasso / ElasticNet | Parametric, regularized linear | High (Lasso: automatic feature selection) | No | Yes, required for the penalty to be fair | Low | Same as Linear Regression, dampened by shrinkage |
| SVR | Kernel-based | Low | Yes (via kernel trick) | Yes, required for kernel distance | Medium-high | Bounded near training data, doesn't diverge wildly |
| KNN Regression | Instance-based, non-parametric | Medium | Yes (locally) | Yes, required for distance calculation | None (no fit step) | Structurally bounded — flattens to nearest-neighbor average |
| Decision Tree | Rule-based, non-parametric | High (readable if/else rules) | Yes | No — splits are scale-invariant | Low | Bounded — flattens to nearest leaf's mean |
| Random Forest | Ensemble (bagging) | Medium (feature importance) | Yes | No | Medium-high (100+ trees) | Bounded, same as single tree |
| Gradient Boosting | Ensemble (boosting) | Medium (feature importance) | Yes | No | Medium (fewer, shallower trees, but sequential) | Bounded, same as single tree |

---

## 🏆 Final Category Results

Every method above, tuned via cross-validation, evaluated on the **identical** Diabetes train/test split (see [topic 08, Section 12](08_Gradient_Boosting_Regression/08_gradient_boosting_regression.ipynb) for the executed comparison):

| Rank | Model | Test R² |
|---|---|---|
| 🥇 1 | SVR (RBF, tuned) | **0.4925** |
| 🥈 2 | Polynomial Regression | 0.4898 |
| 🥉 3 | Ridge / Lasso / ElasticNet | 0.4881 |
| 4 | Random Forest (tuned) | 0.4867 |
| 5 | Linear Regression | 0.4849 |
| 6 | Gradient Boosting (tuned) | 0.4749 |
| 7 | KNN Regression (tuned) | 0.4708 |
| 8 | Decision Tree (tuned) | 0.3588 |

**The honest takeaway:** the spread between the best and worst method is only **0.134 R² points** — and every method except a single unensembled Decision Tree lands within a fairly narrow band. No single algorithm family dominates this dataset. Sequential and parallel tree ensembles (07, 08) didn't beat a well-tuned kernel method (04) or even plain regularized linear regression (03) here. This is a real, dataset-specific result on Diabetes (10 features, ~330 training rows) — not a general claim that any one method is "best" in the abstract. The category's actual lesson is that **fundamentals (scaling, validation, honest metric reporting, checking assumptions)** matter more than picking the fashionable algorithm.

---

## 🧵 Cross-Topic Threads

A few findings deliberately carry forward across multiple notebooks rather than resetting with each new topic:

- **The multicollinearity thread (01 → 03):** Linear Regression's VIF analysis flagged `s1`-`s5` (VIF > 5) as unstable. Ridge later measured a **21% reduction** in bootstrap coefficient instability on exactly those flagged features — a fix verified, not just asserted.
- **The overfitting thread (02 → 03):** Polynomial Regression's 65-feature expansion overfit (test R² 0.424 vs. train 0.605). Ridge and Lasso were both applied to the same expanded features in topic 03, with Lasso's feature selection (R²=0.530) actually beating Ridge's uniform shrinkage here — a data-dependent result, not a rule.
- **The single-tree instability thread (06 → 07):** Decision Tree Regression found a single tree's feature importance swinging meaningfully across bootstrap resamples. Random Forest re-ran that *exact* experiment and found a genuinely mixed result — predictive performance improved substantially, but importance stability did not clearly improve on only 5 resamples. Reported honestly rather than forced to fit the "ensembling fixes everything" narrative.
- **The bias-variance tradeoff, expressed through a different hyperparameter every time:** polynomial degree (02), `alpha` (03), `C`/`epsilon`/`gamma` (04), `k` (05, with an inverted direction from every other topic), `max_depth`/`ccp_alpha` (06), `n_estimators`/`max_features` (07), `n_estimators`/`learning_rate` (08) — one underlying concept, seven different practical levers.
- **The feature-scaling requirement, verified rather than assumed, every time it applies:** gradient descent (01), high-degree polynomial terms (02), regularization penalties (03), kernel distance (04), neighbor distance (05) — and its deliberate *absence* for every tree-based method (06-08), verified numerically in topic 06.

---

## 🎯 When to Use Which

| Situation | Reach for |
|---|---|
| Relationship is genuinely linear, need interpretable coefficients | Linear Regression |
| Curved relationship, but a low-degree polynomial captures it | Polynomial Regression |
| Many correlated features, or want automatic feature selection | Ridge (correlated features) or Lasso (sparse truth) |
| Outliers present, or a non-linear relationship without wanting to hand-pick a polynomial degree | SVR |
| No time/data to train a global model, want purely local predictions | KNN Regression |
| Need fully transparent, auditable decision rules | Decision Tree Regression |
| Want strong general-purpose performance with minimal tuning | Random Forest Regression |
| Want to squeeze out maximum accuracy and can afford careful tuning | Gradient Boosting Regression (or XGBoost/LightGBM/CatBoost, compared in topic 08) |

---

## 📓 Notebook Standard

Every notebook in this category follows the same pipeline: **generate → validate (`nbformat`) → execute (`nbclient`) → re-validate → scan for errors/warnings → spot-check every output against the printed narrative → fix any mismatch honestly → write the topic README → update this category table → commit**. Each notebook has at least 30 executed code cells, and every claimed result (a "best" model, a "surprising" finding, a stability improvement) is checked against actual execution output before being written into the README — several genuinely counter-intuitive results survived this process unchanged (e.g., `RidgeCV` being *slower* than `LassoCV` in topic 03, `absolute_error` loss *not* being the most outlier-robust on a small sample in topic 08) because that's what the code actually produced.

---

## 📦 Datasets

The synthetic datasets used for visualizing individual algorithm behavior are generated inline in each notebook. The real dataset used for every cross-topic comparison — [`sklearn.datasets.load_diabetes`](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) — and the full 162-entry catalog of every dataset used across this entire series live in the central **[Datasets](https://github.com/mdnuruzzamanKALLOL/Datasets)** repo.

---
[← Back to Classical ML](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Regression&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
