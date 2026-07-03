# 📍 K-Nearest Neighbors (KNN) Classifier

> Status: ✅ Complete — [Open the notebook →](02_knn_classifier.ipynb)

A completely different philosophy from Logistic Regression: KNN learns **no parameters at all**. To classify a new point, it looks at the $k$ closest training examples and takes a majority vote.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Choosing k — Bias-Variance Reference](#-choosing-k--bias-variance-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

KNN is **instance-based** ("lazy") learning: there is no training phase in the usual sense — `.fit()` just stores the data. All the computation happens at prediction time:

```
new point → compute distance to every training point → sort → take k closest → majority vote (or average, for regression)
```

This is the polar opposite of Logistic Regression, which does all its work upfront (gradient descent) and then predicts cheaply forever after.

---

## 🎯 Why This Topic Matters

- It's the simplest possible **non-linear** classifier — no decision-boundary math required, the boundary just emerges from wherever the data happens to be.
- It makes **feature scaling's importance** undeniable — get it wrong and the model is provably broken, not just suboptimal (see §9 in the notebook).
- The **curse of dimensionality** (notebook §10) is a fundamental limitation that resurfaces in PCA, clustering, and any other distance-based method later in this series.
- `k` is the cleanest possible illustration of the **bias-variance tradeoff** that the Model Evaluation & Tuning notebooks formalize later.

---

## 🧮 Mathematical Explanation

### 1. Distance metrics

**Euclidean distance** (straight-line distance, the default):

$$d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^n (a_i - b_i)^2}$$

**Manhattan distance** (grid/city-block distance — same L1 norm from the Math Refresher):

$$d(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^n |a_i - b_i|$$

**Minkowski distance** (generalizes both):

$$d(\mathbf{a}, \mathbf{b}) = \left(\sum_{i=1}^n |a_i - b_i|^p\right)^{1/p}$$

$p=2$ gives Euclidean, $p=1$ gives Manhattan — they're not separate formulas, just two points on the same family.

### 2. The prediction rule

For a query point $x_q$, let $N_k(x_q)$ be the set of $k$ training points with smallest $d(x_q, x_i)$. Classification predicts the majority class:

$$\hat{y} = \text{mode}\big(\{y_i : x_i \in N_k(x_q)\}\big)$$

Distance-weighted voting instead weights each neighbor's vote by $\frac{1}{d(x_q, x_i)}$, so closer neighbors count more.

### 3. Bias-variance as a function of $k$

- $k=1$: the prediction is exactly the single nearest training point's label — zero bias toward any "average" behavior, but extremely sensitive to noise in that one point (**high variance**).
- $k=n$ (all training points): every query predicts the single global majority class, ignoring local structure entirely — extremely stable, but ignores the data (**high bias**).

The right $k$ sits between these extremes, found empirically via cross-validation (notebook §7) — there is no formula that computes the "correct" $k$ from first principles.

### 4. The curse of dimensionality

In high dimensions, the ratio of farthest-to-nearest-neighbor distance provably converges toward 1 as dimensionality $d \to \infty$, for most distributions:

$$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} \to 0$$

In words: in very high dimensions, *every* point becomes roughly equidistant from every other point, and "nearest neighbor" stops being a meaningful concept. The notebook demonstrates this empirically (§10) — it's a real, measurable effect, not a theoretical curiosity.

---

## ⚖️ Choosing k — Bias-Variance Reference

| $k$ | Bias | Variance | Boundary shape | Risk |
|---|---|---|---|---|
| Small (1–5) | Low | High | Jagged, chases noise | Overfitting |
| Medium | Balanced | Balanced | Smooth but locally responsive | — (usually the sweet spot) |
| Large (→ n) | High | Low | Nearly a straight majority-class line | Underfitting |

Always pick $k$ via cross-validation (notebook §7), and prefer **odd** $k$ for binary classification to avoid tie votes.

---

## ⚠️ Common Pitfalls & Gotchas

1. **Skipping feature scaling** — the single most damaging KNN mistake; demonstrated directly in the notebook (52.5% vs 98% accuracy on the exact same data, scaled vs unscaled).
2. **Picking $k$ arbitrarily** (e.g. always using $k=5$ because it's a common default) — always cross-validate; the notebook's Wine dataset picked a different optimal $k$ than the toy example.
3. **Using uniform voting with imbalanced classes and a large $k$** — the notebook's §15 shows this can push minority-class recall to literally **0%**; distance-weighting or a smaller $k$ recovers it.
4. **Ignoring the curse of dimensionality** — KNN degrades on high-dimensional data (hundreds of raw features) unless paired with dimensionality reduction (PCA, feature selection) first.
5. **Forgetting KNN is expensive at prediction time** — with $n$ training points and $d$ features, a naive prediction costs $O(nd)$; this is why sklearn defaults to KD-tree/Ball-tree structures (`algorithm="auto"`) that speed this up for larger datasets.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.neighbors.KNeighborsClassifier` | Fit/predict with KNN |
| `sklearn.neighbors.KNeighborsRegressor` | KNN for continuous targets |
| `scipy.spatial.distance.euclidean/cityblock/minkowski` | Distance metric functions |
| `sklearn.model_selection.cross_val_score` | Cross-validated accuracy for choosing $k$ |
| `weights="uniform"` vs `weights="distance"` | Voting scheme |
| `metric="euclidean"/"manhattan"/"chebyshev"` | Distance metric selection |

---

## 📝 Self-Test Exercises

1. Explain, without looking back, why $k=1$ has high variance and $k=n$ has high bias — connect it to what each extreme actually predicts.
2. Given two features on scales 0–1 and 0–1,000,000, predict what happens to KNN's distance calculation before scaling — then verify against notebook §9.
3. Why does agreement between the from-scratch and `sklearn` implementations hit 100% for KNN (notebook §5), while Logistic Regression's from-scratch and `sklearn` weights only matched approximately?
4. Using the curse-of-dimensionality table in notebook §10, explain in your own words why PCA (covered later, Unsupervised) is often applied *before* KNN on high-dimensional data.
5. Propose a fix (other than dropping rows) for the 0%-minority-recall failure mode in notebook §15, other than the two shown (smaller $k$, distance weighting).

---

## 📓 Notebook

32 executed cells: distance metric comparisons, a from-scratch KNN verified to match scikit-learn exactly, decision boundaries across $k$ values, cross-validated $k$ selection, uniform vs. distance weighting, a dramatic feature-scaling before/after (52.5% → 98% accuracy), an empirical curse-of-dimensionality demonstration, KNN regression, a full Wine dataset application, and a class-imbalance failure mode (0% → 100% minority recall depending on $k$/weighting):

➡️ **[02_knn_classifier.ipynb](02_knn_classifier.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [The Curse of Dimensionality — a visual explainer](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.02_KNN_Classifier&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
