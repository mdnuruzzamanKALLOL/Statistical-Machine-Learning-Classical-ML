# 📍 KNN Regression — Instance-Based, Non-Parametric Prediction

> Status: ✅ Complete — [Open the notebook →](05_knn_regression.ipynb)

Every regression method so far learned a fixed set of parameters (coefficients, support vectors) from the training data, then discarded the raw data itself for prediction. KNN Regression does the opposite: it learns nothing in advance and keeps the entire training set, predicting a new point's value as the average of its $k$ nearest training neighbors. The distance mechanics and curse-of-dimensionality math were already derived in [Classification / KNN](../../Classification/02_KNN_Classifier/) — this notebook applies the same machinery to continuous targets and finds where it behaves differently from every prior regression method.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Quick Reference](#-quick-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
Query point x --> find k closest training points --> average their y-values --> prediction

No coefficients. No support vectors. No training phase beyond storing the data.
Every prediction re-does the work from scratch.
```

This makes KNN Regression the most structurally different method in this category: it can represent genuinely discontinuous relationships without any special handling, and it can never extrapolate beyond what it has actually observed.

---

## 🎯 Why This Topic Matters

- **KNN handles discontinuities every parametric method in this category would struggle with** — §1-2 use a deliberately non-smooth step-function dataset, unlike every prior topic's smooth synthetic curves, and KNN fits it without any special accommodation.
- **A very small k can be the genuinely correct answer, not a red flag** — §3's cross-validation selected k=1 on the step-curve dataset; the notebook explains *why* (any k>1 necessarily averages across a discontinuity when near one, introducing real bias) rather than treating an extreme hyperparameter choice as suspicious by default.
- **KNN structurally cannot produce Polynomial Regression's extrapolation failure** — §8 shows predictions flattening to a constant just past the training range's edge, in direct, measured contrast to the wild divergence found in the Polynomial Regression topic.
- **The curse of dimensionality was measured for regression specifically, not just re-asserted from Classification** — §7 shows test R² collapsing from 0.93 (2 dimensions) to 0.11 (100 dimensions) with only 2 truly informative features throughout — the exact same effect first found for KNN Classification, now confirmed to hold for continuous targets too.

---

## 🧮 Mathematical Explanation

### 1. The prediction rule

$$\hat y(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i$$

where $N_k(x)$ is the set of the $k$ training points closest to $x$ by some distance metric. Notebook §2 verifies a from-scratch implementation matches `KNeighborsRegressor` exactly.

### 2. Distance-weighted variant

$$\hat y(x) = \frac{\sum_{i \in N_k(x)} w_i \, y_i}{\sum_{i \in N_k(x)} w_i}, \qquad w_i = \frac{1}{d(x, x_i)}$$

Closer neighbors get more influence than farther ones within the same $k$-neighborhood. §5 shows this advantage growing with $k$ — at large $k$, uniform weighting would otherwise treat far-away neighbors as equally important as close ones.

### 3. Bias-variance in $k$ (recap, inverted direction)

Small $k$: prediction depends on very few points — highly locally reactive (low bias, high variance). Large $k$: prediction averages many points — smooth but can blur real local structure (high bias, low variance). This is the same tradeoff seen with polynomial degree, `gamma`, and `alpha` in prior topics, but note the *direction* is reversed: here, a **smaller** hyperparameter value means **more** complexity, the opposite of degree/gamma/alpha where larger meant more complexity.

### 4. The curse of dimensionality (recap)

As full derived in the KNN Classification topic: in high dimensions, the ratio between the nearest and farthest neighbor distances approaches 1, meaning "nearest" stops being meaningfully different from "far" — degrading the core assumption that close points share label/target information. §7 measures this directly for regression.

---

## 📋 Quick Reference

| Concept | Key takeaway |
|---|---|
| No training phase | All computation happens at prediction time (§9's timing comparison shows cost scales with both dataset size and $k$) |
| Direction of $k$ | Small $k$ = high variance; large $k$ = high bias (inverted from polynomial degree/gamma/alpha) |
| Distance metric | Euclidean (p=2), Manhattan (p=1), and general Minkowski ($p$) are options, not fixed choices |
| Weighting | `distance` weighting matters more at larger $k$ |
| Extrapolation | Structurally bounded — flattens to nearest-neighbor average, never diverges |
| Scaling | Required, for the same distance-domination reason as SVR's kernel |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Treating an extreme cross-validated hyperparameter as automatically wrong** — §3 found k=1 selected on the step-curve dataset; rather than distrusting this, the notebook explains why it's the correct answer for data with genuine discontinuities.
2. **Forgetting to scale features before KNN Regression** — §6 shows a large-scale feature dominating distance calculations exactly as in SVR's kernel and KNN Classification.
3. **Assuming uniform weighting is always sufficient** — §5 shows distance weighting's advantage growing substantially at larger $k$ (up to a 0.277 R² improvement at $k=60$ on the step-curve data).
4. **Using KNN Regression on high-dimensional data without addressing the curse of dimensionality** — §7 shows test R² collapsing by over 0.8 points across 2 to 100 dimensions with a fixed 2 informative features; consider dimensionality reduction (a future Unsupervised topic) or feature selection first.
5. **Expecting KNN to extrapolate like a parametric model** — §8 shows predictions simply flattening past the training range's edge; this is a genuine safety property in some contexts (never wildly wrong) and a genuine limitation in others (never captures a real trend that continues past the observed range).
6. **Ignoring prediction-time cost at deployment** — §9 shows KNN has no separate training cost but every prediction re-searches the stored data, unlike every parametric method in this category which pays its cost once at fit time.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.neighbors.KNeighborsRegressor` | KNN regression estimator |
| `sklearn.model_selection.validation_curve` | Cross-validated $k$ selection |
| `sklearn.model_selection.GridSearchCV` | Joint tuning of $k$, weights, and distance metric |
| `sklearn.datasets.make_regression` | Synthetic data with a controlled number of informative dimensions |

---

## 📝 Self-Test Exercises

1. Using §1's prediction rule, compute the KNN prediction (k=3) by hand for a query point given training points with $y$-values $\{2, 4, 9\}$ as its three nearest neighbors.
2. Explain, using §3's math section, why the *direction* of the bias-variance tradeoff in $k$ is reversed compared to polynomial degree — what does "more complex" mean for each hyperparameter, concretely?
3. Section 3 found k=1 was the CV-selected value on the step-curve dataset. Explain why a smooth (non-discontinuous) dataset would likely NOT select k=1 as optimal, connecting your answer to what averaging across neighbors does to a smooth vs. a discontinuous function.
4. Using §7's dimensionality results, explain why adding irrelevant (noise) dimensions specifically harms a distance-based method more than it would harm, say, Lasso (which can learn to ignore irrelevant features via its coefficients).
5. Section 8 showed KNN's predictions flattening outside the training range. Propose a real-world scenario where this "safety" property (never predicting wildly) would be preferable to Polynomial Regression's behavior, and a different scenario where it would be a genuine disadvantage.

---

## 📓 Notebook

31 executed code cells: a from-scratch KNN regression implementation verified against `KNeighborsRegressor`, a fit visualized on a deliberately non-smooth step-function dataset, a single-query-point nearest-neighbor visualization, a $k$-sweep showing the (reversed-direction) bias-variance tradeoff with both a visual panel and a numeric train/test MSE gap table, cross-validated $k$ selection with an honest explanation of why k=1 was selected on this dataset, a distance-metric comparison (Euclidean/Manhattan/Minkowski), uniform-vs-distance weighting compared across multiple $k$ values, a feature-scaling necessity demonstration, a regression-specific curse-of-dimensionality measurement (test R² from 0.93 to 0.11 across 2-100 dimensions), an extrapolation-behavior demonstration contrasted directly against Polynomial Regression's failure mode, a joint `GridSearchCV` over $k$/weights/metric on Diabetes with a wall-clock prediction-time comparison, residual diagnostics, and a running comparison against every prior regression method in this category:

➡️ **[05_knn_regression.ipynb](05_knn_regression.ipynb)**

---

## 📚 Further Reading

- [Classification / KNN](../../Classification/02_KNN_Classifier/) — full distance metric and curse-of-dimensionality derivation
- [scikit-learn: Nearest Neighbors Regression](https://scikit-learn.org/stable/modules/neighbors.html#regression)
- [scikit-learn: Nearest Neighbors Regression example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html)

---
[← Back to Regression](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Regression.05_KNN_Regression&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
