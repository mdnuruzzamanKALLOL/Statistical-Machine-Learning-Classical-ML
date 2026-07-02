# ⚔️ Support Vector Machine (SVM) Classification

> Status: ✅ Complete — [Open the notebook →](06_svm_classification.ipynb)

A different optimization goal from every prior algorithm: SVM finds the boundary that maximizes the **margin** to the closest points of each class, and via the kernel trick, can bend that boundary into genuinely non-linear shapes.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Kernel Reference](#-kernel-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Among all boundaries that separate two classes, SVM picks the one **farthest** from both — maximizing the "margin," the gap between the boundary and the nearest point of each class. Only those nearest points ("support vectors") actually determine where the boundary sits; every other point could move further away without changing anything.

```
class A  ← margin →  boundary  ← margin →  class B
         ↑                                ↑
    support vectors                 support vectors
    (only these points matter)
```

The **kernel trick** extends this to non-linear boundaries by implicitly measuring similarity in a higher-dimensional space, without ever explicitly computing the coordinates in that space.

---

## 🎯 Why This Topic Matters

- It's the first algorithm in this series whose training objective is explicitly **geometric** (maximize a distance), not probabilistic (Naive Bayes) or impurity-based (trees).
- The **kernel trick** is a genuinely reusable idea beyond SVM — the same "compute similarity without computing coordinates" principle appears in kernel PCA and Gaussian Processes.
- `gamma` follows the exact same bias-variance pattern already seen with KNN's $k$ and tree depth — reinforcing that this pattern is universal across very different algorithm families, not specific to any one of them.
- The head-to-head comparison in the notebook's final section (§14) makes concrete a lesson worth internalizing early: **no single algorithm wins every dataset**.

---

## 🧮 Mathematical Explanation

### 1. The margin

For a linear boundary $w \cdot x + b = 0$, the distance from any point $x_i$ to the boundary is $\frac{|w \cdot x_i + b|}{\|w\|}$. SVM's objective is to maximize the smallest such distance across all training points (the margin), subject to every point being correctly classified:

$$\max_{w, b} \ \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1 \ \ \forall i$$

Equivalently (a cleaner form for optimization), minimize $\|w\|^2$ subject to the same constraints — smaller $\|w\|$ means a wider margin.

### 2. Soft margin

Real data is rarely perfectly separable. Introducing slack variables $\xi_i \geq 0$ that allow (and penalize) margin violations:

$$\min_{w,b,\xi} \ \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w \cdot x_i + b) \geq 1 - \xi_i$$

$C$ trades off margin width against violation tolerance — large $C$ penalizes violations heavily (narrow margin, tightly fit), small $C$ tolerates more violations (wide margin, more regularized). Same direction and naming as Logistic Regression's `C`.

### 3. Support vectors

At the optimal solution, only points where $\xi_i > 0$ or the constraint is exactly tight ($y_i(w\cdot x_i + b) = 1$) have non-zero weight in determining $w$ — these are the **support vectors**. Every other point's exact position is irrelevant to the final boundary, which is why the notebook's §3 shows only 3% of points mattering for a well-separated dataset.

### 4. The kernel trick

The optimization problem above (in its dual form) only ever needs **dot products** between pairs of points, never their raw coordinates. A **kernel function** $K(x_i, x_j)$ computes what the dot product *would be* in some higher-dimensional (possibly infinite-dimensional) feature space $\phi$, without ever computing $\phi(x)$ directly:

$$K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)$$

**Polynomial kernel:**
$$K(x_i, x_j) = (x_i \cdot x_j + c)^d$$

**RBF (Gaussian) kernel:**
$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

RBF measures similarity as a smoothly decaying function of distance — points far apart have near-zero similarity, points close together have similarity near 1. This is *why* it can wrap a boundary around concentric circles (notebook §6): "inside the circle" and "outside" become distinguishable by their pattern of similarities to nearby points, even though no straight line separates them in the original 2D space.

### 5. The `gamma` parameter

For the RBF kernel, $\gamma$ controls how quickly similarity decays with distance. Large $\gamma$ → similarity drops off fast → each point's influence is very local → the boundary can wrap tightly around individual points (overfitting risk). Small $\gamma$ → influence reaches far → smoother, more linear-like boundary (underfitting risk if too small).

---

## 🔀 Kernel Reference

| Kernel | Formula | Best for | Key parameter |
|---|---|---|---|
| `linear` | $x_i \cdot x_j$ | Linearly separable data, large datasets | `C` only |
| `poly` | $(x_i \cdot x_j + c)^d$ | Curved but polynomial-shaped boundaries | `degree`, `C` |
| `rbf` (default) | $\exp(-\gamma\|x_i - x_j\|^2)$ | Most general non-linear case | `gamma`, `C` |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting feature scaling** — like KNN, SVM's margin is a distance-based quantity; an unscaled feature dominates it exactly as it dominates KNN's distance calculation (notebook §10 shows the same dramatic before/after as the KNN notebook).
2. **Using `SVC(kernel="linear")` on large datasets** instead of `LinearSVC` — the notebook's §15 shows a real, measurable speed difference; `SVC` computes a full kernel matrix even for the linear case, `LinearSVC` doesn't.
3. **Tuning `C` and `gamma` independently, one at a time** — they interact (notebook §9's grid heatmap makes this visible); a joint grid search is more reliable than sequential tuning.
4. **Expecting probability outputs by default** — `SVC.predict_proba()` requires `probability=True` at construction (adds computational cost via internal cross-validation); `decision_function()` is available without this cost but isn't a calibrated probability.
5. **Assuming RBF is always the right kernel** — the notebook's Breast Cancer results show Logistic Regression matching SVM exactly here; always compare against simpler baselines (a lesson reinforced, not contradicted, by SVM's power on the circles dataset).

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.svm.SVC` | Kernelized SVM classifier |
| `sklearn.svm.LinearSVC` | Faster linear-only SVM, different optimizer |
| `kernel="linear"/"poly"/"rbf"` | Kernel selection |
| `C`, `gamma`, `degree` | Core hyperparameters |
| `.support_vectors_` | The points that determine the boundary |
| `.decision_function()` | Signed distance to the boundary |
| `sklearn.model_selection.GridSearchCV` | Joint hyperparameter search |

---

## 📝 Self-Test Exercises

1. Explain, using the margin formula in §1, why a smaller $\|w\|$ corresponds to a wider margin.
2. Given `C=1000` vs `C=0.01` on noisy data, predict which produces more support vectors, then verify against notebook §4.
3. Explain in your own words why the RBF kernel can separate concentric circles when no kernel-free straight line can — connect it to §4's similarity argument.
4. Using the gamma visualizations in notebook §8, describe what a `gamma` value tending toward infinity would do to the decision boundary in the limit.
5. Given the tie between Logistic Regression and SVM on the Breast Cancer dataset (notebook §14), propose one dataset characteristic that would make SVM meaningfully outperform Logistic Regression instead.

---

## 📓 Notebook

32 executed cells: margin visualization vs an arbitrary separating line, support vector identification (only 3% of points mattering), a soft-margin `C` sweep, the concentric-circles problem solved by RBF (linear kernel: 59% vs RBF: 100% accuracy), polynomial vs RBF kernel comparison on moons data, a `gamma` bias-variance sweep, joint `C`/`gamma` grid search with a heatmap, a dramatic feature-scaling before/after (49.5% → 98.5%), multiclass one-vs-one, a full Breast Cancer application (98.25% test accuracy, tied with Logistic Regression), and a `LinearSVC` vs `SVC` speed comparison:

➡️ **[06_svm_classification.ipynb](06_svm_classification.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [scikit-learn: SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [StatQuest: Support Vector Machines](https://www.youtube.com/watch?v=efR1C6CvhmE) (visual intuition)

---
[← Back to Classification](../README.md)
