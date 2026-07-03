# 🔵 DBSCAN — Density-Based Clustering With Built-In Noise Detection

> Status: ✅ Complete — [Open the notebook →](03_dbscan.ipynb)

K-Means always assigns every point to some cluster; Hierarchical clustering does too. DBSCAN is the first method in this series that can say "this point doesn't belong to any cluster" — a genuine noise/outlier label, not just a poorly-fit assignment. It also requires no upfront `k` at all (unlike K-Means) and, like single-linkage hierarchical clustering, can trace non-spherical shapes — but through a completely different mechanism: local point density rather than nearest-pair distance.

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

```
Core point:   has >= min_samples neighbors within eps        (dense interior)
Border point: within eps of a core point, but not core itself (edge of a cluster)
Noise point:  neither                                         (doesn't belong anywhere)

A cluster = every core point reachable from every other core point
            through a chain of eps-neighborhoods, plus their border points.
```

---

## 🎯 Why This Topic Matters

- **DBSCAN can do something neither prior method can, under any hyperparameter setting** — §6 injects 20 real outlier points into a blobs dataset; DBSCAN correctly flags most of them as noise (label -1) rather than forcing them into whichever real cluster happens to be nearest, which is all K-Means or Hierarchical Clustering could ever do.
- **The crescents fix works through a genuinely different mechanism than single linkage** — §5 re-tests the K-Means topic's moons failure (ARI=0.261) and reaches ARI=1.000, matching single linkage's fix from the Hierarchical Clustering topic, but via density-connectivity rather than nearest-pair chaining.
- **A "principled" eps-selection heuristic had a real bug, caught and fixed** — §9's naive "steepest single jump" elbow heuristic picked index 298 of 300 (driven by tail outliers, not the genuine elbow), giving a poor eps and ARI=0.712 on blobs; excluding the extreme tail before searching fixed it to ARI=0.968 — a concrete lesson in not trusting `argmax` blindly on real, noisy data.
- **DBSCAN's own limitation was measured, not just described** — §10 shows no single `eps` value correctly separating a dense cluster from a sparse one simultaneously, motivating the next topic (Gaussian Mixture Models).
- **On Iris specifically, DBSCAN did not beat either prior method** — §12 found ARI=0.5681, below K-Means' 0.6201 and hierarchical's 0.6153, completing a three-way honest comparison across three fundamentally different clustering mechanisms (centroids, linkage, density) on the same real dataset.

---

## 🧮 Mathematical Explanation

### 1. Core, border, and noise point definitions

For a point $p$, its $\epsilon$-neighborhood is $N_\epsilon(p) = \{q : d(p, q) \le \epsilon\}$. $p$ is a **core point** if $|N_\epsilon(p)| \ge \texttt{min\_samples}$. A **border point** lies in some core point's neighborhood without itself being core. Everything else is **noise**.

### 2. Density-reachability and clusters

Point $q$ is directly density-reachable from core point $p$ if $q \in N_\epsilon(p)$. A cluster is the set of all points reachable from some starting core point through a *chain* of directly-density-reachable steps — the same chain-following principle as single-linkage hierarchical clustering (§2 in that topic's README), but the criterion at each step is "has enough nearby neighbors" rather than "is the single closest pair."

### 3. The k-distance heuristic for choosing `eps`

For a fixed `min_samples` = $k$, compute every point's distance to its $k$-th nearest neighbor, sort these distances, and plot them. Points *inside* a cluster have small $k$-distances (densely surrounded); points in sparse regions or true outliers have large $k$-distances. The "elbow" where this sorted curve bends sharply upward is a principled `eps` estimate — but naively taking the single steepest jump (`argmax` of the differences) can be misled by one or two extreme outlier points at the very tail of the sorted list, as §9 demonstrates concretely.

---

## 📋 Hyperparameter Reference

| Hyperparameter | Controls | Too small | Too large |
|---|---|---|---|
| `eps` | Neighborhood radius | Most points become noise or tiny fragments | Distinct clusters merge together |
| `min_samples` | Density required to be "core" | Very lenient — nearly everything becomes a cluster | Very strict — most points become noise/border |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Trusting a naive k-distance elbow heuristic blindly** — §9 shows `argmax` of the sorted-distance differences picking a point driven entirely by tail outliers; always sanity-check against the actual plotted curve, or exclude the extreme tail before searching.
2. **Assuming DBSCAN always fixes non-spherical shapes better than every alternative** — §5 shows it matching (not exceeding) single linkage's fix on the moons dataset; different mechanisms can converge on similarly good answers without one being universally superior.
3. **Applying a single global `eps` to data with meaningfully different cluster densities** — §10 shows no `eps` value working for both a dense and a sparse cluster simultaneously; this is DBSCAN's core structural limitation.
4. **Forgetting feature scaling** — §11 shows the same scale-domination failure seen in every distance-based method this series has covered.
5. **Treating "more noise points detected" as inherently good or bad** — §7-8's `eps`/`min_samples` sweeps show noise count is a direct, controllable consequence of hyperparameter choice, not a fixed property of the data; it must be tuned and validated like any other hyperparameter.
6. **Assuming DBSCAN's noise detection is perfectly accurate** — §6's confusion-matrix-style breakdown (true/false positive/negative noise flags) shows real classification error in the noise detection itself; it's a genuinely useful capability, not an infallible one.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.cluster.DBSCAN` | Density-based clustering estimator |
| `DBSCAN().core_sample_indices_` | Indices of points classified as core |
| `sklearn.neighbors.NearestNeighbors` | Compute k-distances for the elbow heuristic |
| `sklearn.metrics.adjusted_rand_score` | Compare against ground truth (validation only) |

---

## 📝 Self-Test Exercises

1. Using §1's core-point definition, explain why increasing `min_samples` while holding `eps` fixed can only ever turn core points into non-core points, never the reverse.
2. Section 5 found DBSCAN and single-linkage hierarchical clustering both achieving ARI=1.000 on the moons dataset via different mechanisms. Explain, in your own words, how "density-connectivity" and "nearest-pair chaining" can arrive at the same result on a chain-like shape.
3. Section 9 found the naive elbow heuristic picking index 298 of 300. Using the k-distance definition in §3, explain why the very last few points in the sorted list are especially likely to produce artificially large jumps.
4. Using §10's varying-density experiment, explain why no single `eps` can work for both clusters — connect your answer to the core-point definition in §1 and how neighbor counts differ between dense and sparse regions at a fixed `eps`.
5. Section 6 measured DBSCAN's noise detection with a full true/false positive/negative breakdown rather than just reporting overall accuracy. Explain why this breakdown is more informative here than a single accuracy number would be, connecting your answer to the Evaluation Metrics topic's discussion of imbalanced classification.

---

## 📓 Notebook

31 executed code cells: a hand-traceable core/border/noise-point classification on a 7-point toy dataset (verified against sklearn), a from-scratch DBSCAN implementation validated on two structurally different datasets and hyperparameter settings, a direct visualization of core/border/noise points on real data, a re-test of the K-Means topic's interleaving-crescents failure (DBSCAN matching single linkage's perfect fix), a genuine noise-detection demonstration with injected outliers and a full true/false positive/negative breakdown, `eps` and `min_samples` sweeps, a k-distance elbow heuristic that was first shown failing naively and then fixed by excluding the extreme tail, a deliberately engineered varying-density failure case, a feature-scaling necessity demonstration, a re-test of the K-Means and Hierarchical Clustering topics' Iris recovery scores (an honest below-both-priors result), and a final three-method, three-dataset recap table closing out the clustering methods covered so far:

➡️ **[03_dbscan.ipynb](03_dbscan.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [scikit-learn: Demo of DBSCAN clustering algorithm](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html)
- [Ester et al. (1996): A Density-Based Algorithm for Discovering Clusters](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)

---
[← Back to Unsupervised](../README.md)
