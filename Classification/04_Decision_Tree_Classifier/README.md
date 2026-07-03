# 🌳 Decision Tree Classifier

> Status: ✅ Complete — [Open the notebook →](04_decision_tree_classifier.ipynb)

A fundamentally different approach from every algorithm so far: no probability model, no distance metric, no linear score — just a sequence of "if feature $X$ is above/below threshold $t$, go left/right" rules, learned directly from data.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Pruning Parameter Reference](#-pruning-parameter-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

A decision tree is built **greedily**, top-down: at every node, try every feature and every possible threshold, pick whichever split best separates the classes, then repeat on each resulting child node until a stopping rule is hit.

```
Is worst_perimeter <= 114.4?
├── Yes → Is worst_concave_points <= 0.14?
│         ├── Yes → predict: benign
│         └── No  → predict: malignant
└── No  → predict: malignant
```

No coefficients, no distances, no probability distributions — just readable if/else rules, which is both the model's biggest strength (interpretability) and, alone, its biggest weakness (instability — see §15 in the notebook).

---

## 🎯 Why This Topic Matters

- It's the **first non-linear, non-probabilistic** model in this category — a genuinely different geometry (axis-aligned rectangles) from every prior algorithm's boundary shape.
- **Feature importance** (referenced but not derived in Foundation's Feature Engineering notebook) is fully derived here — it falls directly out of how much each feature's splits reduce impurity.
- Its core weakness — **instability to small data changes** — is the *entire* motivation for the next notebook, Random Forest. Understanding why a single tree is unstable is a prerequisite for understanding why averaging many trees helps.
- `max_depth` and friends are a clean, intuitive first example of **regularization for non-linear models**, distinct from Ridge/Lasso's weight-shrinkage approach.

---

## 🧮 Mathematical Explanation

### 1. Gini impurity

For a node with class proportions $p_1, \dots, p_k$:

$$G = 1 - \sum_{i=1}^k p_i^2$$

$G = 0$ for a pure node (all one class); $G$ is maximized (0.5 for binary) when classes are perfectly balanced. Gini measures "if I randomly labeled a point in this node according to the class distribution, how often would I be wrong?"

### 2. Entropy & information gain

Entropy, from information theory:

$$H = -\sum_{i=1}^k p_i \log_2(p_i)$$

**Information gain** of a split into left/right children:

$$IG = H(\text{parent}) - \left(\frac{n_L}{n}H(\text{left}) + \frac{n_R}{n}H(\text{right})\right)$$

A tree greedily picks, at every node, the split that maximizes $IG$ (or equivalently minimizes weighted Gini) — a **locally optimal**, not globally optimal, search, since checking every possible full tree structure is computationally intractable.

### 3. The greedy split search

For every feature $j$ and every candidate threshold $t$ (typically the midpoints between sorted unique values), compute the information gain of splitting on $x_j \leq t$ vs $x_j > t$, and keep whichever $(j, t)$ pair scores highest. This $O(\text{features} \times \text{samples} \times \log(\text{samples}))$ search (with sorting) repeats recursively at every new node — exactly what the notebook's `find_best_split` implements.

### 4. Feature importance

For feature $j$, importance is the total impurity decrease it's responsible for, summed across every node where it's used as the split feature, weighted by the fraction of samples reaching that node:

$$\text{importance}(j) = \sum_{\text{nodes split on } j} \frac{n_{\text{node}}}{n_{\text{total}}} \times \Delta \text{impurity}_{\text{node}}$$

This is the exact formula referenced (without derivation) in the Foundation repo's Feature Engineering notebook — now fully derived.

### 5. Why trees overfit without constraints

An unconstrained tree keeps splitting until every leaf is pure (or has one sample) — with enough depth, a tree can always perfectly fit any training set, including its noise. Training accuracy near 100% with meaningfully lower test accuracy (notebook §10) isn't a bug, it's the guaranteed default behavior of an unregularized tree.

---

## ✂️ Pruning Parameter Reference

| Parameter | Effect | Tree equivalent of |
|---|---|---|
| `max_depth` | Caps how many splits deep any path can go | A hard complexity ceiling |
| `min_samples_split` | A node needs at least this many samples to be split further | "Don't bother splitting tiny groups" |
| `min_samples_leaf` | Every leaf must have at least this many samples | Prevents leaves that memorize single outliers |
| `criterion` | `"gini"` or `"entropy"`/`"log_loss"` | Which impurity measure drives the greedy search |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Leaving `max_depth=None` (unlimited) by default** and being surprised by poor test performance — always constrain at least one pruning parameter and cross-validate it (notebook §11).
2. **Trusting a single tree's structure as "the" explanation** — notebook §15 shows a bootstrap resample of the exact same data can pick a different root feature entirely; a tree's specific structure is not a stable ground truth about feature relationships.
3. **Reading feature importance as causally meaningful** — like any importance measure, it reflects *predictive* usefulness within this specific fitted tree, not a causal claim about the real world.
4. **Forgetting trees don't need feature scaling** — since splits only compare a feature to itself at various thresholds, unlike KNN/Logistic Regression, unscaled features cause no mathematical problem (though scaling never hurts either).
5. **Assuming Gini vs entropy choice matters a lot** — notebook §14 shows they're typically very close; don't spend excessive tuning budget here.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.tree.DecisionTreeClassifier` | Fit/predict with a decision tree |
| `sklearn.tree.plot_tree` | Visualize the learned tree structure |
| `.feature_importances_` | Impurity-decrease-based feature importance |
| `.get_depth()`, `.get_n_leaves()` | Inspect tree complexity |
| `max_depth`, `min_samples_split`, `min_samples_leaf` | Pruning hyperparameters |
| `criterion="gini"/"entropy"/"log_loss"` | Impurity measure selection |

---

## 📝 Self-Test Exercises

1. Compute Gini impurity by hand for a node with 7 samples of class A and 3 of class B, then verify against the notebook's `gini_impurity` function.
2. Explain why a decision tree's decision boundary is always made of axis-aligned rectangles, never a diagonal line — tie your answer to how a single split condition is defined.
3. Given two trees, one with `max_depth=2` and one with `max_depth=None`, predict which has higher training accuracy and which likely has higher test accuracy — then verify against notebook §10.
4. Using the feature importance formula in §4, explain why a feature used only once, near a leaf, typically has lower importance than one used at the root.
5. Notebook §15 shows root-feature instability across bootstrap resamples — propose (in words) why *averaging predictions* from many such differently-structured trees might still produce a more reliable overall model, even though each individual tree is unstable.

---

## 📓 Notebook

32 executed cells: Gini impurity and information gain computed by hand, a from-scratch greedy best-split search, a full from-scratch recursive tree verified against scikit-learn (100% agreement), tree structure visualization, the axis-aligned "staircase" decision boundary, a dramatic overfitting demonstration across `max_depth` values, a pruning parameter sweep, feature importance on the Breast Cancer dataset (93.9% test accuracy), a Gini-vs-entropy comparison, and a bootstrap-resampling instability demonstration (3 different root features chosen across 8 resamples of the same data):

➡️ **[04_decision_tree_classifier.ipynb](04_decision_tree_classifier.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk) (visual intuition)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.04_Decision_Tree_Classifier&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
