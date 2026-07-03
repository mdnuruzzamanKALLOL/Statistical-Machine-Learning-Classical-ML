# 🌳 Decision Tree Regression — Recursive Splitting for Continuous Targets

> Status: ✅ Complete — [Open the notebook →](06_decision_tree_regression.ipynb)

The recursive splitting mechanics were already derived for classification in [Classification / Decision Tree](../../Classification/04_Decision_Tree_Classifier/) (Gini/entropy, the `SimpleTreeNode` recursive builder). Regression trees use the identical recursive splitting algorithm with one change: the splitting criterion becomes variance/MSE reduction instead of Gini/entropy, and each leaf predicts the *mean* of its training points instead of a majority class. This produces a genuinely different-looking model than every other method in this category: a staircase of constant predictions, not a smooth curve.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Pruning Method Reference](#-pruning-method-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
Every other method: smooth curve/line/tube
Decision Tree:       ___
                     |   |___
                 ____|       |___
                |                |
```

A tree's prediction function is piecewise-constant by construction — every leaf predicts exactly one number (the mean of its training points), regardless of how deep the tree grows or how smooth the true underlying relationship actually is.

---

## 🎯 Why This Topic Matters

- **A single split can find real structure without being told where to look** — §3 shows the first split (threshold=-2.018) landing within 0.018 of the true piecewise function's actual breakpoint at x=-2, purely from variance-reduction greediness.
- **Pruning strategies genuinely trade off differently** — §7's cost-complexity pruning found an 18-leaf tree (test R²=0.9705) beating the fully-grown 112-leaf tree (test R²=0.9417) on the synthetic data — pruning improved generalization, not just simplicity.
- **Tree instability is real and measurable, not just a theoretical caveat** — §10's bootstrap resampling found the top feature's importance value swinging from 0.529 to 0.705 across 5 resamples of the identical training data, directly motivating why Random Forest (next topic) exists.
- **On this category's running Diabetes comparison, a single tuned tree (test R²=0.3588) was the weakest method tried so far** — Linear Regression (0.4849), Ridge (0.4881), SVR (0.4925), and KNN (0.4708) all outperformed it, an expected and informative result that sets up exactly why ensembling trees (the next two topics) matters.

---

## 🧮 Mathematical Explanation

### 1. The splitting criterion — variance reduction

$$\Delta = \text{Var}(\text{parent}) - \left(\frac{n_L}{n}\text{Var}(\text{left}) + \frac{n_R}{n}\text{Var}(\text{right})\right)$$

The tree greedily picks, at each node, the feature and threshold maximizing $\Delta$ — the direct regression analogue of Gini/entropy reduction used for classification. Notebook §2 verifies a from-scratch implementation of this search matches `DecisionTreeRegressor`'s chosen split exactly.

### 2. Leaf prediction

Each leaf predicts $\hat y = \frac{1}{n_{leaf}}\sum_{i \in leaf} y_i$ — the mean of its training points. This is why the overall prediction function is always piecewise-constant (Section 3): there is no mechanism for interpolating between leaves.

### 3. Cost-complexity pruning

$$R_\alpha(T) = R(T) + \alpha |T|$$

where $R(T)$ is the tree's total training error and $|T|$ is its number of leaves. Growing a full tree and then finding the sequence of subtrees that minimize $R_\alpha$ for increasing $\alpha$ produces a pruning *path* — notebook §7 computes this path directly via `cost_complexity_pruning_path` and selects the $\alpha$ minimizing test error, rather than training error.

### 4. Why trees don't need feature scaling

Every split is a single-feature threshold comparison ($x_j \le t$), never a distance calculation or a coefficient magnitude. Rescaling any feature monotonically preserves every possible split's ordering, so the tree structure — and its predictions — are provably unaffected. Notebook §8 verifies this numerically (test R² differs by less than 0.02 between scaled and unscaled versions of the same data).

---

## 📋 Pruning Method Reference

| Method | How it works | Controls |
|---|---|---|
| `max_depth` | Stop splitting past a fixed depth | Simple, coarse control |
| `min_samples_leaf` | Refuse to create leaves smaller than this | Prevents tiny, noise-fit leaves specifically |
| `min_samples_split` | Refuse to split a node smaller than this | Similar effect, applied earlier in the recursion |
| `ccp_alpha` (post-pruning) | Grow full tree, then prune back by cost-complexity | Often the most principled -- explicitly trades training error against tree size |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Growing an unbounded tree and trusting the training score** — §5 shows an unbounded tree memorizing the training data (near-zero train MSE, one leaf per near-unique point) while test performance suffers.
2. **Assuming deeper is always better** — §7's pruning results show an 18-leaf pruned tree beating the 112-leaf unpruned tree on test data; more leaves means more capacity to fit noise, not just signal.
3. **Forgetting that a tree's prediction is always a step function** — §3 shows this directly; a tree can never output a value between two of its leaf means, regardless of how smooth the true relationship is (in contrast to every other regression method in this category).
4. **Reading a single tree's feature importance as a stable, final answer** — §10 shows the same feature's importance value swinging meaningfully (0.529 to 0.705) across bootstrap resamples of identical data; treat single-tree importances as approximate, not precise.
5. **Expecting one tree to be competitive with tuned linear/kernel/instance-based methods on tabular data with a moderate sample size** — §12's comparison found a single tuned tree the weakest method on Diabetes; this is a known, expected limitation that ensembling (Random Forest, Gradient Boosting) directly addresses.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.tree.DecisionTreeRegressor` | Regression tree estimator |
| `sklearn.tree.plot_tree` | Visualize a tree's split structure directly |
| `DecisionTreeRegressor().cost_complexity_pruning_path()` | Compute the full pruning path (candidate alphas) |
| `DecisionTreeRegressor().feature_importances_` | Variance-reduction-based feature importance |
| `sklearn.model_selection.GridSearchCV` | Joint tuning of `max_depth` and `min_samples_leaf` |

---

## 📝 Self-Test Exercises

1. Using §1's variance-reduction formula, explain why a split that perfectly separates two groups with very different means (but each internally uniform) produces a large $\Delta$.
2. Section 3 showed the tree's first split landing within 0.018 of a true breakpoint the tree was never told about. Explain, using §1's math, why the largest true structural change in the data tends to produce the largest variance reduction.
3. Using §3's cost-complexity formula, explain what happens to the optimal tree size as $\alpha \to 0$ and as $\alpha \to \infty$.
4. Section 8 found scaling made almost no difference to a tree's test R². Explain, using §4, why this result is guaranteed by the splitting mechanism itself rather than being a coincidence of this particular dataset.
5. Section 10 found a single feature's importance value swinging across bootstrap resamples. Propose why *averaging* many such trees (each fit on a different resample) would produce a more stable feature-importance estimate than trusting any single tree's importances — connect your answer to the averaging-reduces-variance principle from the Cross Validation topic.

---

## 📓 Notebook

30 executed code cells: a from-scratch variance-reduction split search verified against `DecisionTreeRegressor`, a check of how closely the first learned split matches the true (unknown-to-the-model) breakpoint, a staircase visualization on a deliberately piecewise synthetic curve with residual diagnostics, a direct tree-structure plot, a `max_depth` sweep from severe underfitting to full memorization with a numeric leaf-count/MSE table, `min_samples_leaf` pre-pruning, a full cost-complexity pruning path with a sampled-alpha results table showing pruning actually improving test performance, a feature-scaling invariance verification, Diabetes feature importances, a bootstrap tree-instability measurement motivating the next topic, a joint `GridSearchCV` over `max_depth`/`min_samples_leaf` with a fit-time comparison against KNN, residual diagnostics, and a running comparison against every prior regression method in this category (where the single tree came in last, as expected):

➡️ **[06_decision_tree_regression.ipynb](06_decision_tree_regression.ipynb)**

---

## 📚 Further Reading

- [Classification / Decision Tree](../../Classification/04_Decision_Tree_Classifier/) — full recursive splitting derivation
- [scikit-learn: Decision Tree Regression](https://scikit-learn.org/stable/modules/tree.html#regression)
- [scikit-learn: Post-pruning decision trees with cost complexity pruning](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

---
[← Back to Regression](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Regression.06_Decision_Tree_Regression&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
