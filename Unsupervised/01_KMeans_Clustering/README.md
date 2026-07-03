# 🎯 K-Means Clustering — Centroid-Based Partitioning

> Status: ✅ Complete — [Open the notebook →](01_kmeans_clustering.ipynb)

The first Unsupervised topic in this series, and a genuine shift in kind: every prior notebook had a target `y` to predict and could be checked against a held-out test set. Clustering has no labels at all — the algorithm must discover structure purely from feature geometry, and evaluating "did it find something real" requires different tools entirely (inertia, silhouette score, and — when ground truth happens to exist for validation purposes only — direct comparison against it).

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Choosing k Reference](#-choosing-k-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
1. Place k centroids (randomly, or via k-means++)
2. Assign every point to its nearest centroid
3. Move each centroid to the mean of the points assigned to it
4. Repeat steps 2-3 until nothing changes
```

This loop (Lloyd's algorithm) always converges, but only to a *local* optimum — which specific local optimum depends entirely on where the centroids started.

---

## 🎯 Why This Topic Matters

- **Clustering requires new evaluation tools entirely** — with no `y_test` to score against, this notebook introduces inertia (§2), the elbow method (§5), and the silhouette score (§6) as the unsupervised-learning analogues of the metrics covered in the Model Evaluation & Tuning topic.
- **A from-scratch implementation was validated rigorously, not loosely** — §2's first same-starting-point test found the manual and sklearn implementations disagreeing (inertia 1967 vs. 459), which turned out to be a fair-comparison problem (single random init vs. sklearn's default 10-restart best-of), not a bug — fixed by re-testing both algorithms from *identical* starting centroids, where they matched exactly.
- **The core spherical-cluster assumption fails in a measurable, visual way** — §8 shows ARI collapsing from 1.000 (spherical blobs) to 0.261 (interleaving crescents) on data K-Means is structurally incapable of separating, directly motivating the next two topics.
- **On real data, K-Means partially — not perfectly — recovers true structure** — §10 found an Adjusted Rand Index of 0.6201 clustering unlabeled Iris measurements against the true species, with the disagreement concentrated exactly where two species' petal measurements genuinely overlap (visualized directly in §10-11).

---

## 🧮 Mathematical Explanation

### 1. The K-Means objective (inertia / WCSS)

$$J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

Minimizing $J$ exactly is NP-hard; Lloyd's algorithm (assign, then update, repeat) is a heuristic that provably decreases $J$ every iteration and therefore always converges — but only to *a* local minimum, not necessarily *the* global one.

### 2. Why initialization matters

Since Lloyd's algorithm only ever decreases $J$ from wherever it starts, a poor starting point can converge to a clustering with meaningfully higher final inertia than a better starting point would reach. §7 measures this directly across 20 different random seeds.

### 3. K-Means++ initialization

Rather than picking all $k$ initial centroids uniformly at random, K-Means++ picks the first centroid randomly, then each subsequent centroid with probability proportional to its squared distance from the nearest already-chosen centroid — deliberately spreading the starting points apart. sklearn's default (`init="k-means++"`, `n_init=10`) combines this smarter starting strategy with multiple restarts, keeping only the best-inertia result.

### 4. The silhouette coefficient

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is the mean distance from point $i$ to other points in its own cluster (cohesion), and $b(i)$ is the mean distance to points in the nearest *other* cluster (separation). Ranges from $-1$ (likely misassigned) to $+1$ (well-separated), giving a single comparable number per candidate $k$ — unlike the elbow method, which requires visual judgment.

### 5. Adjusted Rand Index (validation only, never used for fitting)

Measures agreement between two labelings while correctly accounting for chance agreement and ignoring arbitrary cluster-ID numbering (cluster "0" in one run and cluster "2" in another can represent the same real group). Used throughout this notebook strictly as an honesty check against ground truth that would never be available in a genuine unsupervised setting — never as part of the algorithm itself.

---

## 📋 Choosing k Reference

| Method | How it works | Strength | Weakness |
|---|---|---|---|
| Elbow method | Plot inertia vs. $k$, look for where the drop-off flattens | Intuitive, visual | Requires subjective judgment about where the "elbow" is |
| Silhouette score | Compute one cohesion/separation score per $k$ | Single comparable number, no visual judgment | More expensive to compute (pairwise distances) |
| Domain knowledge | Use a known/expected number of groups | Most reliable when available | Not always available in real unsupervised settings |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Comparing two K-Means runs with different initialization strategies and calling a mismatch a bug** — §2's first validation attempt did exactly this (single random init vs. 10-restart best-of); the fix was testing from identical starting centroids, which the section now does explicitly before drawing any conclusion.
2. **Forgetting to scale features before clustering** — §9 shows a single artificially inflated feature completely dominating the distance calculation, dropping ARI from a near-perfect match to a poor one.
3. **Trusting a single K-Means run without restarts** — §7 shows meaningfully different final inertia across 20 different random seeds with only one initialization each; always use `n_init` > 1 (sklearn's default is 10) or K-Means++.
4. **Applying K-Means to non-spherical or unevenly-sized clusters** — §8's interleaving-crescents test is a direct, measured failure case; K-Means' only notion of "cluster" is distance to a single centroid, which cannot represent a crescent shape.
5. **Treating the elbow method as objective** — §5-6 both used to pick $k=4$ successfully here, but the elbow method specifically requires a human to read the plot; the silhouette score is preferred when a single automatable number is needed.
6. **Over-trusting ARI-style validation as if it were normal practice** — ARI requires ground-truth labels, which defeats the purpose of using an unsupervised method in a genuinely unlabeled setting; it's used here only because Iris happens to have known species, purely to sanity-check the method.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.cluster.KMeans` | K-Means clustering estimator |
| `sklearn.metrics.silhouette_score`, `silhouette_samples` | Aggregate and per-point cluster quality |
| `sklearn.metrics.adjusted_rand_score` | Compare a clustering against ground truth (validation only) |
| `sklearn.datasets.make_blobs`, `make_moons` | Synthetic data with known true structure |

---

## 📝 Self-Test Exercises

1. Using §1's objective function, explain why moving a centroid to the mean of its assigned points is guaranteed to not increase (and usually decrease) the total inertia.
2. Section 2 found the manual and sklearn implementations disagreeing by a factor of 4x in inertia before the fix. Explain why testing both from identical starting centroids was the correct way to isolate whether this was an implementation bug or an initialization difference.
3. Using §4's silhouette formula, explain what it means for a point to have a *negative* silhouette value — what would that point's $a(i)$ and $b(i)$ look like relative to each other?
4. Section 8 found K-Means failing on interleaving crescents. Describe, in geometric terms, why no possible placement of 2 centroids could correctly separate two interleaving crescent shapes.
5. Section 10 found K-Means recovering Iris's 3 species with ARI=0.62, not a perfect 1.0. Using the crosstab and per-point silhouette results, explain which specific pair of species is hardest to separate and why that's a real property of the data, not a flaw in the clustering.

---

## 📓 Notebook

30 executed code cells: a from-scratch Lloyd's-algorithm implementation, first mis-compared against sklearn (differing initialization) and then rigorously validated by forcing both to start from identical centroids, a step-by-step visualization of the assign/update loop converging, an Adjusted Rand Index check against synthetic ground truth, the elbow method and silhouette score computed and compared as two different ways to choose $k$ (both correctly recovering the true $k=4$), a random-vs-K-Means++ initialization comparison across 20 seeds, a direct demonstration of K-Means failing on non-spherical interleaving-crescent data, a feature-scaling necessity demonstration, and a full real-data application on unlabeled Iris measurements with a crosstab, per-point silhouette diagnostic plot, and an honest accounting of exactly where the clustering disagrees with true species labels:

➡️ **[01_kmeans_clustering.ipynb](01_kmeans_clustering.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [scikit-learn: Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [Arthur & Vassilvitskii (2007): k-means++: The Advantages of Careful Seeding](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

---
[← Back to Unsupervised](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Unsupervised.01_KMeans_Clustering&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
