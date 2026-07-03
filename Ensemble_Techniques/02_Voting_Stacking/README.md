# 🗳️ Voting & Stacking

> Status: ✅ Complete — [Open the notebook →](02_voting_stacking.ipynb)

Bagging combines many copies of the **same** algorithm; boosting builds a sequence of the **same** weak learner. Voting and Stacking combine **genuinely different** algorithm families — betting that their different mistakes will cancel out.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Ensemble Technique Comparison — the Whole Category](#-ensemble-technique-comparison--the-whole-category)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

**Voting** combines predictions with a fixed rule (majority vote, or an averaged probability). **Stacking** trains a second model (the "meta-model") to learn *how* to combine the base models' predictions, rather than using a fixed rule at all:

```
Voting:    [Model A, Model B, Model C] → fixed rule (vote/average) → prediction
Stacking:  [Model A, Model B, Model C] → predictions → META-MODEL (learned) → prediction
```

Both require the base models to actually differ from each other in *how* they make mistakes — otherwise there's nothing for either combination strategy to exploit.

---

## 🎯 Why This Topic Matters

- It's the only ensemble strategy in this series designed explicitly for **heterogeneous** base models — every other ensemble technique (Random Forest, Bagging, Boosting) combines many instances of one algorithm family.
- Stacking's leakage fix (notebook §9) is a direct, concrete rerun of Foundation topic 03's "fit on train, transform on the right split" rule, now inside an ensemble's internal mechanics rather than a preprocessing step.
- The notebook's honest final result (§12) — the single best base model beating every ensemble — is a genuinely important, non-obvious lesson: **diversity without competence isn't enough**, closing out this category with a realistic, not idealized, picture of when ensembling actually helps.

---

## 🧮 Mathematical Explanation

### 1. Hard voting

For $B$ base classifiers producing hard labels $h_1(x), \dots, h_B(x)$:

$$H(x) = \text{mode}\big(h_1(x), \dots, h_B(x)\big)$$

Identical in form to bagging's aggregation rule (Ensemble Techniques topic 01) — the difference is entirely in *what* is being combined (different algorithms vs. many resampled copies of one).

### 2. Soft voting

For base classifiers that output calibrated probabilities $P_b(y=k \mid x)$, optionally with per-model weights $w_b$:

$$H(x) = \arg\max_k \ \sum_{b=1}^B w_b \, P_b(y=k \mid x)$$

Soft voting uses strictly more information per model (a full probability distribution, not just the arg-max label), which is why it typically edges out hard voting when the base models' probability estimates are reasonably trustworthy.

### 3. Why diversity is the necessary condition

For voting to reduce error below the average base model's error, base models' mistakes need to be **less than perfectly correlated**. In the extreme case where every base model makes identical errors, majority voting changes nothing — $H(x) = h_1(x) = h_2(x) = \dots$ trivially. The notebook's error-correlation heatmap (§5) and controlled near-identical-KNN failure case (§6) make this a measured, not just asserted, requirement.

### 4. Stacking's meta-model formulation

For base models producing predictions $z_1, \dots, z_B$ (a vector of length $B$ per training example), stacking trains a meta-model $g$:

$$H(x) = g\big(z_1(x), \dots, z_B(x)\big) = g\big(h_1(x), \dots, h_B(x)\big)$$

The meta-model $g$ is trained on a **new dataset** whose features are the base models' own predictions — an unusual, second-level supervised learning problem, using the original labels $y$ as the target.

### 5. Why stacking needs cross-validated base predictions

If $g$ trains on base predictions $z_b(x_i) = h_b(x_i)$ where $h_b$ was **itself trained on $x_i$**, an overfit $h_b$ (e.g. an unpruned tree that memorized $x_i$'s label) produces an artificially perfect $z_b(x_i)$ that doesn't reflect $h_b$'s real-world accuracy — the meta-model learns to over-trust it. The fix: generate each $z_b(x_i)$ using a version of $h_b$ trained on a fold that **excludes** $x_i$ (via `cross_val_predict`), exactly mirroring the out-of-bag principle from bagging.

---

## ⚖️ Ensemble Technique Comparison — the Whole Category

With this notebook, all four ensemble strategies covered across this series can be compared directly:

| Technique | Combines | Diversity source | Combination rule | Reduces |
|---|---|---|---|---|
| Bagging | Many copies of one algorithm | Bootstrap resampling | Fixed (vote/average) | Variance |
| Random Forest | Many trees | Bootstrap + random feature subsets | Fixed (vote/average) | Variance |
| Boosting | A sequence of one weak learner type | Sequential error-correction | Weighted vote (learned weights) | Bias |
| Voting | Different algorithms | Different inductive biases | Fixed (vote/average) | Both, if diverse |
| Stacking | Different algorithms | Different inductive biases | **Learned** (meta-model) | Both, if diverse |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Voting/stacking near-identical base models** — notebook §6 shows this directly: four KNN variants differing only in $k$ produce almost no ensemble benefit over any one of them, since their errors are highly correlated.
2. **Skipping cross-validated base predictions in a hand-rolled stack** — the notebook's §8→§9 progression shows the leakage bug concretely (base models see their own training labels twice), and why `StackingClassifier`'s built-in `cv=` parameter isn't optional convenience, it's a correctness requirement.
3. **Assuming an ensemble always beats every individual base model** — notebook §12's honest result contradicts this; including weak base models can drag a diverse ensemble below its single best member.
4. **Using a needlessly complex meta-model** — notebook §14 shows a Random Forest meta-model not improving over Logistic Regression, since the meta-model's input space (one feature per base model) is small and usually doesn't need high capacity.
5. **Forgetting soft voting requires `predict_proba()`** — not every classifier supports it out of the box the same way (e.g. `SVC` needs `probability=True` explicitly, adding fit cost); hard voting is the fallback when calibrated probabilities aren't available or trustworthy.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.ensemble.VotingClassifier(voting="hard"/"soft", weights=)` | Fixed-rule ensemble combination |
| `sklearn.ensemble.StackingClassifier(final_estimator=, cv=)` | Learned, leakage-safe ensemble combination |
| `sklearn.ensemble.VotingRegressor`, `StackingRegressor` | Regression counterparts |
| `sklearn.model_selection.cross_val_predict(method="predict_proba")` | Leakage-safe meta-feature generation |
| `.named_estimators_`, `.final_estimator_` | Inspecting a fitted ensemble's internals |

---

## 📝 Self-Test Exercises

1. Using the formula in §3, explain precisely why voting provides zero benefit when every base model makes identical predictions on every example.
2. Explain, using §5, why an unpruned Decision Tree is a particularly risky base model to include in a *hand-rolled* (non-cross-validated) stack.
3. Given the error-correlation heatmap from notebook §5, propose which pair of base models you'd expect to benefit most from being combined, and why.
4. Notebook §12 shows the single best base model beating every ensemble. Propose a concrete change to the base model set (not the data) that would likely let an ensemble win instead.
5. Contrast hard and soft voting's information usage (§1 vs §2) — construct a scenario (in words) where they'd disagree on a final prediction despite most base models "agreeing."

---

## 📓 Notebook

32 executed cells: four heterogeneous base models individually evaluated, hard and soft voting, an error-correlation heatmap, a controlled near-identical-base-model failure case, weighted voting, a from-scratch stacking implementation with its leakage bug shown live and then fixed via cross-validated meta-features, scikit-learn's `StackingClassifier`, meta-model coefficient inspection, an honest full comparison (best single model actually winning here, with the resulting lesson embedded directly in the takeaways rather than hidden), a meta-model complexity comparison, and `VotingRegressor`/`StackingRegressor` for completeness:

➡️ **[02_voting_stacking.ipynb](02_voting_stacking.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Voting Classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)
- [scikit-learn: Stacked generalization](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- [Wolpert (1992): Stacked Generalization (original paper)](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)

---
[← Back to Ensemble Techniques](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Ensemble_Techniques.02_Voting_Stacking&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
