# 🌌 t-SNE & UMAP — Non-Linear Dimensionality Reduction for Visualization

> Status: ✅ Complete — [Open the notebook →](06_tsne_umap.ipynb)

The PCA topic ended by showing a Swiss roll's curved structure destroyed by PCA's best possible *linear* 2D projection. t-SNE and UMAP are built specifically to solve that problem: both preserve local neighborhood structure through fundamentally non-linear embeddings, at the cost of losing something PCA guarantees — a stable, reusable, distance-preserving global coordinate system. This notebook derives a simplified t-SNE from scratch, re-tests the Swiss roll directly, and is explicit about what these tools are (excellent visualizations) and are not (reliable measurements of cluster size, inter-cluster distance, or density).

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [t-SNE vs. UMAP vs. PCA Reference](#-t-sne-vs-umap-vs-pca-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
PCA:          find the best LINEAR 2D projection (guaranteed global structure, deterministic)
t-SNE/UMAP:   find a 2D layout where NEARBY points stay nearby (local structure only,
              non-deterministic, no meaning in absolute size/distance/position)
```

Neither is strictly better — they answer different questions. PCA answers "what's the best flat summary of this data's variance?" t-SNE/UMAP answer "what does this data's neighborhood structure look like, if I let the layout bend?"

---

## 🎯 Why This Topic Matters

- **Both non-linear methods directly fixed the PCA topic's failure case** — §2 re-tests the identical Swiss roll and finds trustworthiness rising from PCA's 0.940 to t-SNE's 0.999 and UMAP's 0.998 — a measured, not just visual, confirmation.
- **A from-scratch t-SNE was validated the honest way — through outcome, not exact numeric match** — §3's simplified implementation (fixed-sigma affinities, no early exaggeration) reaches a comparable silhouette score to sklearn's optimized version (0.98 vs. 0.91) on identical toy data, explicitly acknowledging why an exact match isn't expected rather than forcing one.
- **The single most important pitfall was demonstrated concretely, not just warned about** — §6 engineers two clusters with a true 19.7x spread ratio in the original 10D space, and finds the embedded visual spread ratio comes out to 0.9x — essentially unrelated to the truth. This is measured evidence for why cluster size/distance on a t-SNE plot cannot be trusted.
- **Non-determinism was measured, not just mentioned** — §5 runs t-SNE three times with different seeds and shows visibly different absolute layouts but nearly identical trustworthiness scores (0.9935 each time) — the *local* structure is stable even though the *global* layout isn't.
- **UMAP's unique practical advantage (transforming new data) was demonstrated directly** — §8 fits UMAP on 1500 points and genuinely transforms 297 never-seen points into the existing embedding, something t-SNE has no mechanism for at all.

---

## 🧮 Mathematical Explanation

### 1. High-dimensional affinities

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \ne i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

Each point's affinity to every other point is a Gaussian-kernel probability, calibrated (via $\sigma_i$) so that a target "perplexity" — roughly, effective neighbor count — is reached. §3's simplified version fixes $\sigma$ globally rather than solving for it per-point, trading exactness for a compact, tractable implementation.

### 2. Low-dimensional affinities and the crowding problem

$$q_{ij} = \frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k \ne l}(1+\|y_k-y_l\|^2)^{-1}}$$

Using a heavy-tailed Student-t distribution (rather than another Gaussian) in the low-dimensional space specifically prevents moderate-distance points from being forced unnaturally close together — the "crowding problem" that motivated t-SNE's specific choice of kernel, distinguishing it from the earlier, less successful SNE method.

### 3. The optimization

Gradient descent minimizes $KL(P \| Q) = \sum_{i \ne j} p_{ij} \log(p_{ij}/q_{ij})$, moving low-dimensional points until their pairwise affinities best match the high-dimensional ones. §3's KL-divergence curve falling monotonically is this optimization observed directly — the same gradient-descent principle as every optimization-based method in this series, applied to point positions instead of model parameters.

### 4. Perplexity

Perplexity roughly sets the effective number of neighbors considered when calibrating each $\sigma_i$. §4 shows visually and numerically (via trustworthiness) how too-small perplexity fragments real structure and too-large perplexity blurs genuinely separate groups together — the same underlying local/global tradeoff seen with `k` in KNN and `eps` in DBSCAN.

### 5. UMAP's different theoretical foundation

UMAP is grounded in manifold learning and fuzzy topology rather than SNE's probabilistic framework, but shares the same practical goal (preserve local neighborhoods in a low-dimensional embedding). Its `n_neighbors` and `min_dist` parameters play roles similar to perplexity, but UMAP additionally learns an explicit mapping function, which is what makes `.transform()` on new data possible (§8) — something no probability-distribution-matching method like t-SNE can offer without re-optimizing from scratch.

---

## 📋 t-SNE vs. UMAP vs. PCA Reference

| Property | PCA | t-SNE | UMAP |
|---|---|---|---|
| Structure preserved | Global (variance) | Local (neighbors) | Local (neighbors), often more global than t-SNE |
| Deterministic | Yes | No (§5) | Mostly (seeded) |
| Cluster size/distance meaningful | Yes | **No (§6)** | **No** |
| Can transform new data | Yes | **No (§8)** | Yes (§8) |
| Speed (relative, §9) | Fastest | Slowest | Faster than t-SNE, slower than PCA |
| Best use case | Feature reduction for modeling, global structure | Exploratory visualization | Exploratory visualization + new-data support |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Reading cluster size or inter-cluster distance off a t-SNE/UMAP plot as meaningful** — §6's direct measurement (true 19.7x spread ratio → embedded 0.9x ratio) is the single most important lesson in this notebook; never do this.
2. **Expecting the same embedding from run to run** — §5 shows visibly different absolute layouts across seeds; only compare embeddings' *local* neighbor structure (e.g. via trustworthiness), never their absolute coordinates.
3. **Picking perplexity/n_neighbors carelessly** — §4 shows real visual and numeric differences across a wide perplexity range; there's no universal default that works for every dataset size and structure.
4. **Trying to use t-SNE like PCA for feature engineering on new/streaming data** — §8 shows this is structurally impossible for t-SNE (no `.transform()`); use UMAP or PCA if new-data support is required.
5. **Assuming non-linear always beats linear** — these methods answer a different question than PCA, not a strictly better one; §9-10 show the real speed cost of the added flexibility, and PCA remains the right choice when global structure or new-data transforms matter more than local visual separation.
6. **Trusting a from-scratch validation only if it numerically matches an optimized library implementation exactly** — §3's honest validation compares *outcomes* (both cleanly separate 3 known groups) rather than forcing an unrealistic exact match against sklearn's more sophisticated optimization procedure.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.manifold.TSNE` | t-SNE embedding |
| `umap.UMAP` | UMAP embedding (with `.transform()` support for new data) |
| `sklearn.manifold.trustworthiness` | Quantifies local neighborhood preservation |
| `scipy.spatial.distance.pdist`, `squareform` | Pairwise distance computation for the from-scratch implementation |

---

## 📝 Self-Test Exercises

1. Using §1's affinity formula, explain why a point very close to $x_i$ gets a $p_{j|i}$ close to the maximum possible value, and a point very far away gets a value close to zero.
2. Section 2 found both t-SNE and UMAP substantially raising trustworthiness over PCA on the Swiss roll. Explain, using §2 and §3 of the math section, why a heavy-tailed low-dimensional kernel specifically helps represent a manifold that must be "cut open" to lay flat.
3. Using §6's measured result (true 19.7x ratio → embedded 0.9x ratio), explain in your own words why this happens — what is t-SNE's objective function actually optimizing for, and why does it have no term that would preserve relative cluster spread?
4. Section 5 found near-identical trustworthiness scores across 3 different random seeds despite visibly different plots. Explain why trustworthiness (a neighbor-rank-based metric) is invariant to rotation and reflection of the embedding, while a raw coordinate comparison would not be.
5. Section 8 showed UMAP transforming new points via a learned mapping. Propose why this specific capability makes UMAP (but not t-SNE) usable as a preprocessing step in a production ML pipeline where new data arrives after the embedding was originally fit.

---

## 📓 Notebook

30 executed code cells: a direct re-test of the PCA topic's Swiss roll failure (t-SNE and UMAP both substantially improving trustworthiness), a from-scratch simplified t-SNE implementation with an explicit high-dimensional affinity-matrix intuition check, a KL-divergence convergence curve, an outcome-based validation against sklearn's TSNE (matching silhouette quality, not exact coordinates), a perplexity sweep with both visual and trustworthiness-table evidence, a non-determinism demonstration across 3 random seeds, a deliberately engineered cluster-size-distortion demonstration (the notebook's central pitfall lesson), a UMAP `n_neighbors`/`min_dist` sweep with a numeric trustworthiness table, a genuine new-data transform demonstration unique to UMAP, a PCA/t-SNE/UMAP speed comparison, a full three-method comparison on the real Digits dataset, and a robustness check confirming the method ranking holds across multiple metric settings:

➡️ **[06_tsne_umap.ipynb](06_tsne_umap.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)
- [UMAP documentation](https://umap-learn.readthedocs.io/)
- [Distill.pub: How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — the definitive visual guide to t-SNE's pitfalls
- [van der Maaten & Hinton (2008): Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

---
[← Back to Unsupervised](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Unsupervised.06_TSNE_UMAP&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
