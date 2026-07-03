# 📉 PCA — Dimensionality Reduction via Variance Maximization

> Status: ✅ Complete — [Open the notebook →](05_pca.ipynb)

Every topic in this category so far has clustered data in its *original* feature space. PCA does something different in kind: it finds a new set of axes — linear combinations of the original features, ranked by how much variance they capture — and lets high-dimensional data be represented, visualized, or modeled with far fewer dimensions. This notebook derives PCA from the covariance matrix's eigendecomposition, connects it directly to the Linear Regression topic's multicollinearity finding, and tests whether it actually helps the clustering and distance-based methods from earlier in this series.

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
Original features: x1, x2, x3, x4  (4 arbitrary, possibly correlated axes)
                          |
              find NEW axes (PC1, PC2, ...) that are:
              - linear combinations of the original features
              - ranked by how much variance they capture
              - all mutually orthogonal
                          |
Keep only the top few PCs -> most of the data's real spread, fewer dimensions
```

---

## 🎯 Why This Topic Matters

- **A from-scratch implementation was validated on two datasets and cross-checked three separate ways** — the covariance-eigendecomposition result (§2), the direct SVD result (§6), and sklearn's own output (§2, §6) all agreed, and orthonormality of the resulting components was verified numerically (§2), not just asserted.
- **PCA's variance-based redundancy detection is the same phenomenon the Linear Regression topic's VIF found, from a completely different angle** — §9 shows the exact `s1`-`s5` features flagged with VIF > 5 in that topic concentrating far more heavily onto a shared handful of components (mean |loading| 0.92) than the other features (0.73) — two different mathematical tools independently finding the same redundancy.
- **Dimensionality reduction's practical benefit was tested, not assumed, three separate times** — it preserved Iris's species-separating structure well (§5), did not clearly improve Diabetes K-Means clustering by silhouette (§10), and — most instructively — actually *hurt* KNN's curse-of-dimensionality problem in a synthetic test (§11's connection to the KNN Regression topic), because PCA selects components by variance, not by relevance to any target, and the synthetic noise features there weren't lower-variance than the informative ones.
- **PCA's core limitation (linearity) was demonstrated concretely, not just stated** — §12's Swiss roll test shows PCA's best possible 2D projection destroying a curved manifold's real structure, directly motivating the next topic (t-SNE & UMAP).

---

## 🧮 Mathematical Explanation

### 1. The covariance matrix eigendecomposition

$$\Sigma = \frac{1}{n-1}X_c^TX_c, \qquad \Sigma v_i = \lambda_i v_i$$

Centering $X$ to $X_c$, the covariance matrix's eigenvectors $v_i$ are the principal component directions, and eigenvalues $\lambda_i$ are the variance captured along each. Sorting by descending $\lambda_i$ ranks components by importance.

### 2. The SVD connection

$$X_c = U S V^T$$

The right singular vectors $V$ of the centered data matrix are exactly the principal components, and $S^2/(n-1)$ gives the eigenvalues — without ever explicitly forming the covariance matrix. §6 verifies this produces identical results to the eigendecomposition route; sklearn's `PCA` uses SVD internally for better numerical stability.

### 3. Explained variance and reconstruction

$$\text{explained variance ratio}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

Projecting to $k$ components and back (`inverse_transform`) reconstructs an approximation of the original data; the reconstruction error (§7) is a direct, measurable quantity for exactly how much information a given $k$ discards — not just an abstract percentage.

### 4. Why PCA needs feature scaling more than most methods

PCA finds directions of maximum *numerical* variance. A feature with an inflated scale has inflated variance purely from units, and PCA has no mechanism to distinguish "large because informative" from "large because measured in different units" — §8 demonstrates a single arbitrarily-rescaled feature dominating PC1 almost entirely.

### 5. Why PCA doesn't know about the target

PCA is **unsupervised** — it ranks directions purely by how much they vary, with no reference to any target variable $y$. When noise features have comparable variance to genuinely informative ones (§11's synthetic curse-of-dimensionality test), PCA's top components can be just as noise-dominated as the original full feature set, and dimensionality reduction can fail to help — or even hurt — a downstream distance-based model.

---

## 📋 Quick Reference

| Concept | Key takeaway |
|---|---|
| Principal components | Orthogonal directions of maximum variance, ranked by eigenvalue |
| Explained variance ratio | Fraction of total variance each component captures |
| Scree plot / cumulative variance | Visual tools for choosing how many components to keep |
| Reconstruction error | Direct measure of information lost at a given component count |
| Whitening | Additionally rescales each component to unit variance |
| Linearity | PCA can only find linear combinations — cannot "unroll" curved manifolds |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Forgetting to scale features before PCA** — §8 shows a single inflated-scale feature dominating PC1 almost entirely, discarding real structure in the other features.
2. **Assuming PCA always helps a downstream model** — §10 (clustering) and §11 (KNN regression) both tested this directly rather than assuming it; the KNN case actually got *worse* after PCA, a genuinely instructive counter-example.
3. **Confusing "high variance" with "relevant to the target"** — §11 shows exactly why: PCA has no notion of $y$ at all, so noise features with high variance are treated identically to genuinely informative ones.
4. **Interpreting individual principal components as meaningful real-world quantities** — a PC is a linear combination of every original feature; §9's loadings table shows this concretely (every Diabetes feature contributes to every PC, just with different weights).
5. **Applying PCA to data with genuinely non-linear (curved) structure and expecting it to be preserved** — §12's Swiss roll result is a direct demonstration of this failure, not just a caveat.
6. **Treating eigenvector sign as meaningful** — §2 shows manual and sklearn PCA producing flipped-sign (but equally valid) components; always compare magnitudes or correct for sign before checking a match.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.decomposition.PCA` | Principal Component Analysis |
| `PCA().explained_variance_ratio_` | Fraction of variance per component |
| `PCA().inverse_transform()` | Reconstruct approximate original data from reduced components |
| `PCA(whiten=True)` | Additionally rescale each component to unit variance |
| `numpy.linalg.eigh`, `numpy.linalg.svd` | The two equivalent mathematical routes to PCA |

---

## 📝 Self-Test Exercises

1. Using §1's eigendecomposition, explain why the principal components are guaranteed to be mutually orthogonal — what property of the covariance matrix makes this true?
2. Section 9 found Linear Regression's VIF-flagged features loading more heavily onto shared components than other features. Explain, in your own words, why "correlated features" and "features that don't need many independent dimensions to represent" are really the same underlying idea, viewed from statistics vs. linear algebra respectively.
3. Using §5's explanation, describe a modification to the §11 synthetic dataset (something about how the noise features are generated) that would make PCA much more likely to help KNN's curse-of-dimensionality problem.
4. Section 12 showed PCA destroying the Swiss roll's structure. Explain, using §4 of the math section ("PCA can only find linear combinations"), why no choice of 2 linear axes could ever "unroll" a spiral.
5. Section 10 found PCA improving Diabetes clustering's silhouette score at 2 components but not helping cumulative model quality overall. Propose an experiment using the ARI-against-true-labels approach from earlier clustering topics (rather than silhouette alone) to check whether the "improvement" reflects genuinely better clusters or just a silhouette-score artifact of fewer dimensions.

---

## 📓 Notebook

30 executed code cells: a from-scratch PCA implementation (covariance eigendecomposition) validated against sklearn on two different datasets, an explicit orthonormality check, a direct SVD-based derivation cross-validated against both, a 2D principal-component-direction visualization, scree plots and cumulative explained variance, Iris reduced from 4D to 2D with a direct K-Means comparison against the K-Means topic's original-space result, reconstruction error measured across component counts, a feature-scaling necessity demonstration, a whitening demonstration, a direct re-examination of the Linear Regression topic's VIF-flagged multicollinear features through PCA's loading structure, a tested (not assumed) check of whether PCA preprocessing improves Diabetes clustering, a direct callback to the KNN Regression topic's curse-of-dimensionality finding (with an honest, instructive result showing PCA did NOT help when noise and signal share similar variance), and a Swiss roll demonstration of PCA's fundamental linearity limitation:

➡️ **[05_pca.ipynb](05_pca.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [scikit-learn: PCA vs. LDA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html)
- [Classification / LDA & QDA](../../Classification/07_LDA_QDA/) — the supervised counterpart to PCA's unsupervised dimensionality reduction
- [Jolliffe, *Principal Component Analysis*](https://link.springer.com/book/10.1007/b98835)

---
[← Back to Unsupervised](../README.md)
