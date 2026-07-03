# 🎯 Linear & Quadratic Discriminant Analysis (LDA & QDA)

> Status: ✅ Complete — [Open the notebook →](07_lda_qda.ipynb)

A close cousin of Gaussian Naive Bayes: both model each class as a Gaussian and classify via Bayes' theorem. The difference is exactly the "naive" independence assumption — LDA/QDA model the **full covariance matrix**, closing the loop back to the Math Refresher's covariance and eigenvector sections.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [LDA vs QDA vs GaussianNB Reference](#-lda-vs-qda-vs-gaussiannb-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Both LDA and QDA model each class as a multivariate Gaussian and classify a new point via Bayes' theorem — exactly Naive Bayes' approach, minus the independence assumption:

| | Naive Bayes | LDA | QDA |
|---|---|---|---|
| Covariance modeled | Diagonal only (features assumed independent) | Full, but **shared** across classes | Full, **separate per class** |
| Decision boundary | Can be curved (independent Gaussians) | Always **linear** | Always **quadratic** (curved) |
| Parameters to estimate | Fewest | Medium | Most |

LDA's shared-covariance assumption is what forces its boundary to be provably linear; QDA drops that assumption for more flexibility, at the cost of needing more data.

---

## 🎯 Why This Topic Matters

- It **completes the generative-model story** started with Naive Bayes — same Bayes'-theorem-based approach, progressively fewer simplifying assumptions (Naive Bayes → LDA → QDA).
- LDA's **dimensionality-reduction use** (notebook §6-7) is a direct, labeled counterpart to PCA — reinforcing the Math Refresher's eigenvector math from a supervised angle.
- The **QDA-fails-to-fit** demonstration (notebook §11) is a real, concrete example of why "more flexible" isn't automatically "better" — a lesson that recurs with every model in this series that has a complexity knob.

---

## 🧮 Mathematical Explanation

### 1. The shared setup — Bayes' theorem with Gaussian likelihoods

Both models assume $x \mid y=k \sim \mathcal{N}(\mu_k, \Sigma_k)$ and classify via:

$$\hat{y} = \arg\max_k \ P(y=k) \cdot \mathcal{N}(x; \mu_k, \Sigma_k)$$

exactly the Math Refresher's Bayes' theorem, with a multivariate Gaussian PDF as the likelihood term.

### 2. LDA — shared covariance, linear boundary

LDA assumes $\Sigma_k = \Sigma$ for every class (one shared covariance matrix). Taking the log of the ratio of two classes' posteriors and simplifying (the $\Sigma$-dependent quadratic terms cancel out because they're identical across classes) yields a **linear** discriminant function:

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log P(y=k)$$

Classification predicts $\arg\max_k \delta_k(x)$ — linear in $x$, hence a linear (hyperplane) decision boundary between any two classes.

### 3. QDA — per-class covariance, quadratic boundary

QDA allows $\Sigma_k$ to differ per class. The quadratic terms no longer cancel, leaving a genuinely quadratic discriminant function:

$$\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log P(y=k)$$

The $(x-\mu_k)^T \Sigma_k^{-1} (x-\mu_k)$ term is quadratic in $x$ — this is literally the same **Mahalanobis distance** form that appears in outlier detection, now driving a decision boundary.

### 4. Parameter count — why QDA needs more data

For $d$ features and $k$ classes:

- **LDA** estimates one shared $\Sigma$: $\frac{d(d+1)}{2}$ parameters, pooling data from *all* classes together.
- **QDA** estimates $k$ separate $\Sigma_k$ matrices: $k \cdot \frac{d(d+1)}{2}$ parameters, using only *each class's own* data.

With $d=13$ (Wine) and small per-class sample counts, QDA's covariance estimate can become **singular** (non-invertible) — not just less accurate, but literally impossible to fit, exactly as demonstrated in the notebook's §11.

### 5. LDA as dimensionality reduction

Beyond classification, LDA finds up to $k-1$ directions that maximize the ratio of **between-class variance** to **within-class variance** — formally, the generalized eigenvectors of $S_W^{-1} S_B$ (within-class and between-class scatter matrices). This is structurally the same eigenvector-decomposition machinery as PCA (Math Refresher, Part I §3), but PCA maximizes total variance (label-blind) while LDA maximizes class separability (label-aware).

---

## ⚖️ LDA vs QDA vs GaussianNB Reference

| | Covariance | Boundary | Data efficiency | Best when |
|---|---|---|---|---|
| GaussianNB | Diagonal, per-class | Curved (from independent Gaussians) | Highest | Many features, limited data, weak correlations |
| LDA | Full, shared | Linear | High | Correlated features, similar spread across classes |
| QDA | Full, per-class | Quadratic | Lowest | Correlated features, genuinely different spread per class, ample data |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Defaulting to QDA because it's "more flexible"** — notebook §5 shows it doesn't reliably beat LDA even when it *could*, and needs meaningfully more data to do so reliably.
2. **Not checking for singular covariance matrices on small/high-dimensional data** — QDA can fail to fit entirely (notebook §11's `LinAlgError`), not just underperform; this is a hard constraint, not a soft one.
3. **Confusing LDA's classification use with its dimensionality-reduction use** — `LinearDiscriminantAnalysis(n_components=k)` for projection and the same class used for `.predict()` are the same object, easy to conflate in code.
4. **Forgetting to scale features before LDA/QDA** — like every model whose math involves a covariance matrix, wildly different feature scales distort the estimated covariance structure.
5. **Assuming LDA's linear boundary is equivalent to Logistic Regression's** — they're both linear, but LDA is generative (models $P(x\mid y)$) while Logistic Regression is discriminative (models $P(y\mid x)$ directly); they can produce different boundaries even though both are "a straight line."

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.discriminant_analysis.LinearDiscriminantAnalysis` | LDA classifier and/or dimensionality reducer |
| `sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis` | QDA classifier |
| `.transform()` / `n_components=` | LDA's supervised projection |
| `shrinkage=` | Regularizes the covariance estimate (needs `solver="lsqr"` or `"eigen"`) |
| `sklearn.decomposition.PCA` | Unsupervised comparison projection |
| `.coef_` | LDA's linear discriminant coefficients |

---

## 📝 Self-Test Exercises

1. Starting from the two classes' discriminant functions in §2, show algebraically why the $x^T\Sigma^{-1}x$ term cancels when $\Sigma$ is shared but not when it isn't (§3) — this is *the* structural reason LDA is linear and QDA is quadratic.
2. Using the parameter-count formulas in §4, compute how many parameters QDA needs for Wine's 13 features and 3 classes, versus LDA's shared-covariance count.
3. Explain why LDA "never runs out of data" the way QDA can (notebook §11) — tie your answer to how each estimates its covariance matrix.
4. Contrast PCA and LDA's projections on the same data (notebook §7) — describe a scenario where PCA's top component would *not* be useful for classification, even though it captures the most variance.
5. GaussianNB, LDA, and QDA scored 0.9663, 0.9717, 0.9551 respectively on Wine (notebook §10) — propose why the *middle* assumption (LDA) won here, rather than the least (GaussianNB) or most (QDA) flexible one.

---

## 📓 Notebook

32 executed cells: a visual gap between GaussianNB's axis-aligned view and true correlated data, LDA's linear boundary, QDA's quadratic boundary, a controlled LDA-vs-QDA comparison on matched/mismatched covariance assumptions, LDA as supervised dimensionality reduction vs PCA side-by-side, a full Wine dataset application, a three-way GaussianNB/LDA/QDA comparison, a training-size sensitivity sweep where **QDA genuinely fails to fit** at small sample sizes (caught live and turned into a teaching example rather than hidden), multiclass LDA, shrinkage regularization, and coefficient interpretation:

➡️ **[07_lda_qda.ipynb](07_lda_qda.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Linear and Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)
- [scikit-learn: LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [StatQuest: Linear Discriminant Analysis](https://www.youtube.com/watch?v=azXCzI57Yfc) (visual intuition)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.07_LDA_QDA&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
