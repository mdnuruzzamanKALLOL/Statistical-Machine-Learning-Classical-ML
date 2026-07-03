# 🌲 Random Forest Classifier

> Status: ✅ Complete — [Open the notebook →](05_random_forest_classifier.ipynb)

The direct fix for the previous notebook's discovery: a single decision tree is **unstable**. Random Forest's answer: train many trees on randomized variations of the data, and average their votes.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Hyperparameter Reference](#-hyperparameter-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Random Forest combines two ideas, neither new on its own:

1. **Bagging** (Bootstrap AGGregating) — train many trees, each on its own random resample (with replacement) of the training data.
2. **Random feature selection** — at every split, each tree only considers a random subset of features, not all of them.

```
Training data
   ├── bootstrap sample 1 (random subset of features per split) → Tree 1
   ├── bootstrap sample 2 (random subset of features per split) → Tree 2
   ├── ...
   └── bootstrap sample B (random subset of features per split) → Tree B
                                    ↓
                    majority vote across all B trees
```

Neither randomization alone fully decorrelates the trees; together, they ensure individual trees genuinely differ from each other, which is what makes averaging them actually reduce variance instead of just averaging near-identical copies.

---

## 🎯 Why This Topic Matters

- It's the cleanest possible real-world example of **why ensembles work** — a principle that reappears with every boosting algorithm later in this category.
- **OOB score** (§8) is a genuinely useful, free validation technique unique to bagging-based methods.
- It directly answers the exact weakness demonstrated in the Decision Tree notebook — instability of both predictions *and* feature importance — with measurable before/after numbers.
- It remains one of the strongest, lowest-effort baselines on real tabular data, and a common default choice in practice before reaching for more complex boosting methods.

---

## 🧮 Mathematical Explanation

### 1. Bootstrap sampling

Given $n$ training examples, a bootstrap sample draws $n$ examples **with replacement**. The probability any single example is *excluded* from one bootstrap sample:

$$P(\text{excluded}) = \left(1 - \frac{1}{n}\right)^n \xrightarrow{n \to \infty} e^{-1} \approx 0.368$$

So each tree trains on roughly **63.2%** of the unique original examples; the remaining **36.8%** are that tree's "out-of-bag" (OOB) points.

### 2. Bagging's variance reduction

For $B$ independent, identically-distributed predictors each with variance $\sigma^2$, the variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B f_b(x)\right) = \frac{\sigma^2}{B} \quad \text{(if fully independent)}$$

In reality, trees aren't fully independent (they're trained on overlapping, correlated bootstrap samples of the same data), so the real reduction is less dramatic than $\sigma^2/B$ — but the direction is right, and it's *exactly* why random feature selection matters: it further decorrelates the trees, pushing the ensemble closer to the ideal $\sigma^2/B$ reduction.

### 3. Random feature selection

At every split, instead of searching all $d$ features, the algorithm searches only a random subset of size $m < d$ (default $m = \sqrt{d}$ for classification). This means even a single overwhelmingly strong feature can't dominate every tree's structure — some trees simply never get to consider it at a given split, forcing them to find alternative, still-useful splits, which is precisely what decorrelates the ensemble.

### 4. Out-of-bag (OOB) score

Since each tree only trains on ~63.2% of the data, its OOB ~36.8% can be used as a mini validation set for that specific tree. Averaging each point's predictions only from trees that *didn't* train on it gives a full-dataset performance estimate — without touching a held-out test set at all:

$$\text{OOB score} = \frac{1}{n}\sum_{i=1}^n \mathbb{1}\Big[y_i = \text{majority vote of trees where } i \text{ was OOB}\Big]$$

### 5. Feature importance, averaged

Random Forest feature importance is the Decision Tree notebook's impurity-decrease formula, averaged across all $B$ trees:

$$\text{importance}(j) = \frac{1}{B}\sum_{b=1}^B \text{importance}_b(j)$$

Averaging is exactly why this ranking is far more stable across resamples than a single tree's (notebook §10 measures this directly).

---

## 🔧 Hyperparameter Reference

| Parameter | Effect | Default |
|---|---|---|
| `n_estimators` | Number of trees; more is almost always better, with diminishing returns | 100 |
| `max_features` | Features considered per split (`"sqrt"`, `"log2"`, a fraction, or `None`=all) | `"sqrt"` |
| `max_depth`, `min_samples_leaf` | Per-tree pruning (same as Decision Tree) | Unlimited |
| `oob_score` | Compute the free OOB validation estimate | `False` |
| `class_weight` | Reweight classes for imbalanced data | `None` |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Using too few trees** — `n_estimators=10` or less defeats the purpose; the variance-reduction benefit compounds with more trees (notebook §7 shows diminishing but still-positive returns well past 100).
2. **Setting `max_features=None` (all features)** — this removes the decorrelation mechanism that makes Random Forest different from plain bagging; usually worse than the `"sqrt"` default on wide feature sets.
3. **Comparing single-tree vs. forest variance unfairly** — a proper comparison requires resampling *both* methods the same way (notebook §4 documents a real mistake caught while building this: initially only the ensemble was resampled, artificially favoring the single tree's "stability").
4. **Treating OOB score as a full replacement for a real test set** — it's a strong estimate, but a genuinely held-out test set (never touched by any tree, ever) remains the gold standard when data allows it.
5. **Expecting single-tree interpretability** — a 100-tree forest has no single readable rule-list; use feature importance and partial dependence, not `plot_tree`, to understand a forest's behavior.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.ensemble.RandomForestClassifier` | Fit/predict with a random forest |
| `n_estimators`, `max_features`, `oob_score`, `class_weight` | Key hyperparameters |
| `.oob_score_` | Free validation estimate after fitting |
| `.feature_importances_` | Averaged, more stable feature importance |
| `sklearn.model_selection.cross_val_score` | Cross-validated accuracy for hyperparameter sweeps |

---

## 📝 Self-Test Exercises

1. Using the formula in §1, verify that as $n \to \infty$, roughly 36.8% of points are excluded from a bootstrap sample — compute the limit for a few finite $n$ (e.g. 10, 100, 1000) and see how quickly it converges.
2. Explain, using §3, why setting `max_features` too high (close to total feature count) tends to hurt Random Forest's benefit over plain bagging.
3. Given the OOB formula in §4, explain why OOB score requires no separate validation split at all — what data is each prediction actually evaluated on?
4. Notebook §10 shows Random Forest's top feature stayed the same across all 5 resamples, versus 3 different root features for a single tree (previous notebook). Explain this difference using the averaging argument from §5.
5. Propose why `class_weight="balanced"` (notebook §15) helps minority-class recall specifically at the level of Gini/entropy impurity calculations, not just at the final voting step.

---

## 📓 Notebook

32 executed cells: bootstrap sampling mechanics, a from-scratch bagging ensemble, a corrected/fair variance-reduction comparison (single tree std=0.0246 vs bagged std=0.0073 — a 3x+ stability improvement), random feature selection explained concretely, a full `n_estimators` sweep, OOB score validated against real test accuracy, feature importance stability compared directly against the single-tree notebook's instability, smoother decision boundaries, an overfitting-gap comparison, a `max_features` sweep, a full Breast Cancer evaluation (95.6% test accuracy), and a class-imbalance `class_weight` demonstration:

➡️ **[05_random_forest_classifier.ipynb](05_random_forest_classifier.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [scikit-learn: RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Breiman (2001): Random Forests (original paper)](https://link.springer.com/article/10.1023/A:1010933404324)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.05_Random_Forest_Classifier&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
