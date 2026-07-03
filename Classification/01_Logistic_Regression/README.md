# 🏷️ Logistic Regression

> Status: ✅ Complete — [Open the notebook →](01_logistic_regression.ipynb)

The first classification algorithm in this series, and the direct bridge from the [Math Refresher](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation/tree/main/05_Math_Refresher)'s gradient descent to real predictive modeling. Despite the name, Logistic Regression is a **classifier**, not a regressor.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Threshold & Metric Tradeoffs](#-threshold--metric-tradeoffs)
5. [Regularization Reference](#-regularization-reference)
6. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
7. [Function Reference](#-function-reference-used-in-the-notebook)
8. [Self-Test Exercises](#-self-test-exercises)
9. [Notebook](#-notebook)
10. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Logistic Regression predicts $P(y=1 \mid x)$ — the probability an input belongs to class 1 — by computing a linear score $z = w \cdot x + b$ (exactly like Linear Regression) and squashing it through the **sigmoid function** into the range $(0, 1)$. A threshold (default 0.5) then converts that probability into a hard 0/1 label.

```
x → [linear score z = w·x + b] → [sigmoid σ(z)] → probability → [threshold] → class label
```

It is called "regression" because it's built from the same linear-score machinery as Linear Regression — but the output is a class, not a continuous value, which is why it lives in the Classification category.

---

## 🎯 Why This Topic Matters

- It's the simplest model where **gradient descent from the Math Refresher becomes a real classifier** — no new optimization concept, just a new cost function.
- Its **decision boundary is linear** — the same limitation Linear Regression has, and the same motivation for kernel methods (SVM) and tree-based models covered later.
- **Regularization (`C`)** here is the exact same L1/L2 machinery as Ridge/Lasso in the Regression category, just applied to a classification cost function.
- Its coefficients are **directly interpretable** on scaled features — a rare property among classifiers, and a good reason to reach for it as a first baseline on any classification problem.

---

## 🧮 Mathematical Explanation

### 1. The sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Properties that matter: $\sigma(0) = 0.5$, $\sigma(z) \to 1$ as $z \to \infty$, $\sigma(z) \to 0$ as $z \to -\infty$, and it's smooth/differentiable everywhere — required for gradient descent to work. Its derivative has a famously clean form:

$$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

### 2. The model

$$\hat{y} = P(y=1 \mid x) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

### 3. Log loss (binary cross-entropy)

Squared error (Linear Regression's cost function) produces a non-convex, poorly-behaved cost surface when composed with the sigmoid. **Log loss** is used instead, for a single example:

$$\mathcal{L}(y, \hat{y}) = -\big[y \log(\hat{y}) + (1-y)\log(1-\hat{y})\big]$$

Only one term is active per example (since $y \in \{0, 1\}$ zeroes out the other): if $y=1$, loss is $-\log(\hat{y})$ (large when $\hat{y}$ is small — confidently wrong); if $y=0$, loss is $-\log(1-\hat{y})$ (large when $\hat{y}$ is large). Averaged over $n$ samples:

$$J(w, b) = -\frac{1}{n}\sum_{i=1}^n \Big[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\Big]$$

This function **is** convex in $w, b$ — gradient descent is guaranteed to find the global minimum (unlike with a squared-error cost on a sigmoid output).

### 4. Gradient derivation

Differentiating $J$ with respect to $w$ (using the chain rule through $\hat{y} = \sigma(z)$, $z = w\cdot x + b$, and the clean sigmoid derivative from §1) simplifies remarkably:

$$\frac{\partial J}{\partial w} = \frac{1}{n}X^T(\hat{y} - y), \qquad \frac{\partial J}{\partial b} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)$$

Despite the sigmoid and logarithm involved, the gradient reduces to the same clean form as Linear Regression's: **(prediction − actual)**, averaged and weighted by the inputs. This is exactly what the notebook's from-scratch `train_logistic_regression` implements, and it's the same $x_{t+1} = x_t - \eta \nabla f(x_t)$ update rule from the Math Refresher.

### 5. Decision boundary

The predicted class flips at $\hat{y} = 0.5$, i.e. $\sigma(z) = 0.5$, i.e. $z = 0$:

$$w \cdot x + b = 0$$

This is a **hyperplane** (a line in 2D, a plane in 3D, ...) — Logistic Regression can only ever separate classes with a straight cut, regardless of how much data it sees. Curved boundaries require engineering polynomial/interaction features first (topic 04's `PolynomialFeatures`), exactly as with Linear Regression.

### 6. Regularized cost function

L2-regularized (Ridge-style) log loss, as scikit-learn uses by default:

$$J(w, b) = -\frac{1}{n}\sum_{i=1}^n \Big[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\Big] + \frac{1}{2C}\|w\|_2^2$$

Note `C` is the **inverse** of regularization strength (unlike Ridge's `alpha`, which is direct) — small `C` means strong regularization (small weights), large `C` means weak regularization (weights fit the data more tightly). L1-regularized ("Lasso-style") logistic regression swaps in $\|w\|_1$ and can zero out coefficients entirely, for the same reason described in the Math Refresher's L1/L2 norm section.

### 7. Multiclass extension (softmax)

For $k > 2$ classes, the sigmoid generalizes to the **softmax** function:

$$P(y=j \mid x) = \frac{e^{z_j}}{\sum_{c=1}^k e^{z_c}}, \qquad z_j = w_j \cdot x + b_j$$

Each class gets its own weight vector $w_j$; softmax normalizes all $k$ scores into a valid probability distribution that sums to 1. When $k=2$, softmax reduces exactly to the sigmoid — no coincidence, sigmoid is softmax's 2-class special case.

---

## ⚖️ Threshold & Metric Tradeoffs

| Threshold | Effect | Best when |
|---|---|---|
| Lower (e.g. 0.3) | More positives predicted → recall ↑, precision ↓ | Missing a positive is costly (disease screening) |
| Default (0.5) | Balanced | No strong asymmetry in error cost |
| Higher (e.g. 0.7) | Fewer positives predicted → precision ↑, recall ↓ | A false positive is costly (spam → important email) |

---

## 🔧 Regularization Reference

| Penalty | Effect on weights | `sklearn` param |
|---|---|---|
| L2 (default) | Shrinks smoothly, rarely exactly zero | `penalty="l2"` |
| L1 | Can zero out weights (built-in feature selection) | `penalty="l1"`, needs `solver="liblinear"` or `"saga"` |
| None | No regularization (matches the notebook's from-scratch version) | `penalty=None` |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting to scale features before regularized Logistic Regression** — the L2/L1 penalty treats all weights symmetrically, so a feature on a larger raw scale gets unfairly penalized less (or more) relative to its actual importance. Always scale first (topic 03).
2. **Reading raw coefficients as importance on unscaled features** — only valid when features share a scale; otherwise a coefficient's size reflects the feature's units, not its importance.
3. **Assuming 0.5 is always the right threshold** — it's a default, not a law; tune it based on the real-world cost of false positives vs. false negatives (see table above).
4. **Confusing `C` with `alpha`** — `C` is inverse regularization strength; a common source of "I set strong regularization but weights didn't shrink" bugs (see §6).
5. **Expecting a curved decision boundary** — Logistic Regression is linear by construction; if the true boundary is curved, either engineer polynomial features or use a non-linear model (SVM with a kernel, trees, boosting — all covered later in this category).

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.linear_model.LogisticRegression` | Fit a (regularized) logistic regression classifier |
| `.predict()` vs `.predict_proba()` | Hard labels vs calibrated probabilities |
| `sklearn.metrics.confusion_matrix` | TP/FP/TN/FN breakdown |
| `sklearn.metrics.accuracy_score/precision_score/recall_score/f1_score` | Core classification metrics |
| `sklearn.metrics.roc_curve`, `roc_auc_score` | Threshold-independent separability summary |
| `sklearn.metrics.log_loss` | Probability-calibrated cost function value |
| `sklearn.model_selection.train_test_split(stratify=)` | Leakage-safe, class-balanced split |
| `sklearn.preprocessing.StandardScaler` | Required before regularized fitting |

---

## 📝 Self-Test Exercises

1. Derive, on paper, why $\sigma(z) = 0.5$ implies $z = 0$ — and connect that to why the decision boundary is linear.
2. Given a model with `C=0.01` vs `C=100`, predict which one will have smaller-magnitude coefficients, then verify against section 14 of the notebook.
3. For a medical screening test, argue whether the classification threshold should be raised or lowered from 0.5, and why, using the precision/recall tradeoff table above.
4. Explain why log loss, not squared error, is used as Logistic Regression's cost function — what goes wrong with squared error on a sigmoid output?
5. Show that softmax with $k=2$ classes reduces algebraically to the sigmoid function.

---

## 📓 Notebook

36 executed cells: sigmoid/log-loss visualization, a from-scratch gradient-descent implementation verified against scikit-learn, decision boundary plots, threshold tuning, confusion matrix, ROC/AUC, an L1/L2 regularization sweep, multiclass softmax, and a full real-world application on the Breast Cancer Wisconsin dataset (98%+ test accuracy, 0.995 AUC):

➡️ **[01_logistic_regression.ipynb](01_logistic_regression.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) (visual intuition)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.01_Logistic_Regression&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
