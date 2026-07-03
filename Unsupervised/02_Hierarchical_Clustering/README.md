# 🌳 Hierarchical Clustering — Dendrograms and Nested Structure

> Status: ✅ Complete — [Open the notebook →](02_hierarchical_clustering.ipynb)

K-Means (previous topic) requires choosing $k$ upfront and only ever produces spherical clusters. Hierarchical clustering does neither: it builds a full nested tree of every possible clustering from $n$ individual points down to 1 giant cluster, letting $k$ be chosen *after* seeing the structure, and — depending on the linkage method — can represent non-spherical shapes K-Means structurally cannot. This notebook builds the algorithm from scratch, compares every standard linkage method, and directly re-tests the two findings from the K-Means topic: the interleaving-crescents failure case and the Iris recovery score.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Linkage Method Reference](#-linkage-method-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

```
Start:  every point is its own cluster (n clusters)
Step:   merge the two CLOSEST clusters into one
Repeat: until only 1 cluster remains

Result: a full tree (dendrogram) recording every merge and its distance --
        cut it at any height afterward to get any number of clusters
```

Unlike K-Means, the entire hierarchy is built once; choosing "how many clusters" becomes a question answered *after* fitting, not before.

---

## 🎯 Why This Topic Matters

- **A from-scratch implementation was validated on two different linkage rules, not just one** — §2 verifies the merge loop exactly matches `scipy.cluster.hierarchy.linkage` for both single and complete linkage, confirming the *loop* is correct (not just one hard-coded distance rule).
- **Hierarchical clustering directly fixed K-Means' worst failure** — §7 re-tests the interleaving-crescents dataset that gave K-Means an ARI of only 0.261; single linkage reached **ARI=1.000**, a clean, complete fix, because its "closest pair" merge rule can trace a curved chain of nearby points in a way no centroid-based method can.
- **That same strength has a real, measured cost** — §8's chaining demonstration shows single linkage merging two genuinely separate, well-formed clusters into one (259 points) plus a single outlier, purely because a thin bridge of noise connected them; ward linkage resisted the identical bridge.
- **On Iris specifically, hierarchical clustering did NOT beat K-Means** — §11 found the best linkage method (ward, ARI=0.6153) slightly *below* K-Means' 0.6201 from the previous topic, an honest, direct comparison reported as it came out rather than assumed to favor the "more sophisticated" method.
- **The flexibility has a real computational cost** — §12 measures hierarchical clustering running measurably slower than K-Means, with the gap widening as sample size grows.

---

## 🧮 Mathematical Explanation

### 1. Linkage criteria — four different definitions of "distance between clusters"

$$\text{Single: } \min_{a \in A, b \in B} d(a,b) \qquad \text{Complete: } \max_{a \in A, b \in B} d(a,b) \qquad \text{Average: } \frac{1}{|A||B|}\sum_{a,b} d(a,b)$$

**Ward linkage** instead merges whichever pair of clusters causes the smallest increase in total within-cluster variance — the same objective K-Means minimizes, applied one merge at a time rather than iteratively.

### 2. Why single linkage traces curves and chains

Single linkage only needs *one* close pair of points between two clusters to consider them near each other — this is exactly the mechanism that let it trace the crescents in §7 (each crescent is a chain of nearby points) and exactly the mechanism that caused the chaining failure in §8 (a thin noise bridge is also "one close pair after another").

### 3. The cophenetic correlation

The cophenetic distance between two points is the dendrogram height at which they first join the same cluster. Correlating all pairwise cophenetic distances against the true pairwise distances (§6) measures how faithfully a specific linkage method's tree represents the original data geometry — useful for choosing a linkage method before even deciding how many clusters to cut for.

### 4. Computational cost

Naive agglomerative clustering must find the closest pair among all active clusters at every merge step — $O(n^2)$ per step, $O(n^3)$ total in the worst case (efficient implementations like scipy's reduce this to roughly $O(n^2 \log n)$ using specialized data structures). K-Means, by contrast, costs roughly $O(nk)$ per iteration. §12 measures this gap directly rather than stating it abstractly.

---

## 📋 Linkage Method Reference

| Linkage | Distance definition | Strength | Weakness |
|---|---|---|---|
| Single | Minimum pairwise distance | Traces curves/chains (§7's crescent fix) | Prone to chaining through noise (§8) |
| Complete | Maximum pairwise distance | Produces compact, tight clusters | Can split large clusters unnecessarily |
| Average | Mean pairwise distance | A middle ground between single and complete | Less interpretable geometric meaning |
| Ward | Minimizes variance increase | Usually the best default (matches §11's best Iris result) | Implicitly favors similarly-sized, roughly spherical clusters, closer to K-Means' bias |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Assuming hierarchical clustering always beats K-Means because it's "more sophisticated"** — §11 found ward linkage (ARI=0.6153) slightly below K-Means (ARI=0.6201) on Iris; more flexible does not mean strictly better on every dataset.
2. **Using single linkage on noisy real-world data without checking for chaining** — §8 shows a single thin bridge of points merging two well-separated clusters into one; single linkage's curve-tracing strength is also its noise-sensitivity weakness.
3. **Forgetting feature scaling** — §9 shows ARI collapsing from 1.000 to 0.0032 when one feature's scale dominates the distance calculation, the same issue seen in every distance-based method this series has covered.
4. **Choosing $k$ by eye from a large, cluttered dendrogram** — §10's silhouette-on-a-cut-tree approach gives a single comparable number per $k$, the same principled alternative used for K-Means.
5. **Ignoring the computational cost for large datasets** — §12 shows the runtime gap versus K-Means widening as $n$ grows; hierarchical clustering is a poor choice for very large datasets regardless of its other advantages.
6. **Treating the cophenetic correlation as a substitute for validating actual cluster quality** — it measures how well the *tree* represents pairwise distances, not whether a specific cut of that tree produces a good clustering; use it alongside, not instead of, the silhouette score.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.cluster.AgglomerativeClustering` | Hierarchical clustering estimator (fixed $k$) |
| `scipy.cluster.hierarchy.linkage` | Build the full linkage matrix (merge history) |
| `scipy.cluster.hierarchy.dendrogram` | Visualize the linkage matrix as a tree |
| `scipy.cluster.hierarchy.fcluster` | Cut a linkage matrix at a given height/cluster count |
| `scipy.cluster.hierarchy.cophenet` | Compute cophenetic distances and correlation |

---

## 📝 Self-Test Exercises

1. Using §1's four linkage formulas, compute by hand the single- and complete-linkage distance between two 2-point clusters of your choosing, and confirm single ≤ complete always holds.
2. Section 7 found single linkage achieving a perfect ARI=1.000 on interleaving crescents. Explain, using §2, why complete or ward linkage would NOT be expected to achieve the same result on the same data (you can check this directly in the notebook's linkage-comparison panel).
3. Section 8's chaining demonstration used a thin, evenly-spaced bridge of points. Explain why a bridge with even wider point-to-point spacing (still connecting the two clusters, but more sparsely) might fail to cause the same chaining effect — connect your answer to §1's single-linkage formula.
4. Using §3's cophenetic distance definition, explain why two points that merge very early in the dendrogram (a small merge height) must have a small cophenetic distance, regardless of how far apart they are in the *original* feature space via other paths.
5. Section 12 measured hierarchical clustering's runtime gap versus K-Means widening with sample size. Propose, using the complexity classes in §4, roughly how much slower you'd expect hierarchical clustering to be at $n=10{,}000$ compared to $n=1{,}000$, and compare your prediction's order of magnitude to what the notebook's actual measured ratio suggests.

---

## 📓 Notebook

31 executed code cells: a from-scratch agglomerative clustering implementation validated against `scipy.cluster.hierarchy.linkage` for both single AND complete linkage (confirming the merge loop generalizes correctly), a hand-traceable dendrogram on an 8-point toy dataset, a four-way linkage method comparison (dendrograms and resulting clusterings side by side), a single-fit multiple-cut demonstration showing $k$ chosen after fitting, the cophenetic correlation computed for every linkage method, a direct re-test of the K-Means topic's interleaving-crescents failure (single linkage achieving a perfect ARI=1.000 fix), a deliberately engineered chaining failure case (259-vs-1 cluster split) contrasted against ward linkage's resistance to the same noise bridge, a feature-scaling necessity demonstration, silhouette-based $k$-selection by cutting one fitted tree, a direct re-test of the K-Means topic's Iris recovery score (an honest below-K-Means result for the best linkage method), a computational-cost timing comparison against K-Means across five sample sizes, and a final recap table comparing K-Means against the best hierarchical linkage across all three test datasets:

➡️ **[02_hierarchical_clustering.ipynb](02_hierarchical_clustering.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [scikit-learn: Comparison of the different linkage methods](https://scikit-learn.org/stable/auto_examples/cluster/plot_linkage_comparison.html)
- [scipy: Hierarchical clustering and dendrograms](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---
[← Back to Unsupervised](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Unsupervised.02_Hierarchical_Clustering&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
