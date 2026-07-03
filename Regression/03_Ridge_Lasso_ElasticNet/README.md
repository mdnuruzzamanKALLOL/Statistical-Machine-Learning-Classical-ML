# 🎯 Ridge, Lasso & ElasticNet — Regularized Regression in Practice

> Status: ✅ Complete — [Open the notebook →](03_ridge_lasso_elasticnet.ipynb)

The full mathematical derivation of Ridge/Lasso/ElasticNet — closed-form solutions, the L1-vs-L2 geometric picture, the bias-variance tradeoff regularization controls — already lives in the [Model Evaluation & Tuning / Bias-Variance & Regularization](../../Model_Evaluation_Tuning/04_Bias_Variance_Regularization/) topic. This notebook instead treats the three methods as **regression algorithms to choose between**, closing the loop on two problems found earlier in this category: Linear Regression's multicollinearity (VIF > 5 on `s1`-`s5`) and Polynomial Regression's overfitting from 65 expanded features.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Algorithm Selection Guidance](#-algorithm-selection-guidance)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

This notebook is built around three concrete questions, each answered with a dedicated experiment rather than just theory:

```
1. Does Ridge actually fix the multicollinearity Linear Regression found?
   -> bootstrap-measure coefficient stability on the exact VIF-flagged features

2. Does Lasso actually recover a sparse model when the truth IS sparse?
   -> synthetic data with a known 5-of-20 sparse coefficient vector

3. Does ElasticNet actually handle correlated features better than Lasso?
   -> two engineered 0.99-correlated features with a shared true effect
```

---

## 🎯 Why This Topic Matters

- **Ridge measurably fixed a previously-identified real problem** — §3 found Ridge reduces bootstrap coefficient standard deviation by 21% on exactly the `s1`-`s5` features Linear Regression's VIF analysis flagged as unstable.
- **Lasso's sparse recovery isn't just theoretical** — §4 measured 100% recall (found all 5 true features) with precision improving from 0.25 to 1.00 as alpha increased through the sweep, directly showing the precision-recall tradeoff in feature selection.
- **The "which regularizer wins" question has no universal answer** — §7 found Lasso's feature selection (test R²=0.530) actually beat Ridge's uniform shrinkage (0.490) on the 65-feature polynomial-expanded Diabetes set, reversing what the notebook expected going in.
- **Wall-clock performance doesn't always match theoretical expectations** — §8's timing comparison found `LassoCV` (31ms) dramatically faster than `RidgeCV` (535ms) on this dataset, the opposite of the "Ridge has a closed-form solution so it's fastest" intuition — a genuinely useful reminder that asymptotic properties don't always predict real performance at a given problem size.

---

## 🧮 Mathematical Explanation

For the full derivations (Ridge/Lasso/ElasticNet objective functions, the L1-diamond-vs-L2-circle geometric picture, why regularization requires scaling), see the [Bias-Variance & Regularization topic](../../Model_Evaluation_Tuning/04_Bias_Variance_Regularization/). This section covers only what's new here.

### 1. Coefficient variance under resampling

$$\text{Var}(\hat\beta_j) \text{ across bootstrap resamples}$$

Rather than deriving Ridge's variance-reduction property abstractly, §3 measures it directly: fit the same model on 200 bootstrap resamples of the training data, and compare how much each flagged feature's coefficient swings across resamples for OLS vs. Ridge.

### 2. Sparse recovery: precision and recall on feature selection

$$\text{Precision} = \frac{|\text{Selected} \cap \text{True}|}{|\text{Selected}|}, \qquad \text{Recall} = \frac{|\text{Selected} \cap \text{True}|}{|\text{True}|}$$

Applying the same precision/recall framework from the Evaluation Metrics topic to *feature selection itself*: does Lasso's set of nonzero coefficients match the true generating model's nonzero set? §4's alpha sweep shows recall staying at 1.0 (never missing a true feature) while precision rises from 0.25 to 1.0 as alpha increases and irrelevant features get zeroed out.

### 3. The grouping effect, measured directly

For two features with correlation $\rho \to 1$ and a shared true effect, Lasso's solution is theoretically unstable — it can assign the entire effect to either feature depending on small data perturbations. ElasticNet's added L2 term makes the two features' coefficients converge toward equality as $\rho \to 1$. §5 constructs a $\rho=0.99$ pair and measures the absolute difference between the two fitted coefficients directly.

---

## 📋 Algorithm Selection Guidance

Based on this notebook's own measured results, not just general theory:

| Situation | Evidence from this notebook | Recommendation |
|---|---|---|
| Multicollinear features (VIF > 5) | §3: Ridge cut bootstrap coefficient std by 21% on the flagged features | Ridge |
| Suspect only a few features truly matter | §4: Lasso recovered the true sparse model with 100% recall | Lasso |
| Two or more features are highly correlated and both matter | §5: ElasticNet split credit far more evenly than Lasso | ElasticNet |
| Many correlated features from a polynomial/interaction expansion | §7: result was data-dependent — Lasso beat Ridge here, but isn't guaranteed to | Try both, compare via CV |
| Need to tune quickly on a small/medium dataset | §8: LassoCV was the fastest of the three CV variants here | Don't assume Ridge's closed form is always fastest — measure |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Assuming Ridge always "wins" on multicollinear data without measuring it** — §3 measured the actual stability improvement (21%) rather than asserting it; the real number matters for deciding whether the fix is worth the added bias.
2. **Expecting Lasso to recover a sparse model at any alpha** — §4 shows precision was only 0.25 at small alpha (kept all 20 features, most irrelevant) and only reached 1.0 at a properly tuned alpha; cross-validation, not a fixed default, is what makes sparse recovery work.
3. **Using Lasso on strongly correlated features expecting stable coefficients** — §5 shows Lasso can favor one of two nearly-identical features somewhat arbitrarily; if which specific feature gets credit matters for interpretation, ElasticNet or Ridge are safer choices.
4. **Assuming regularized regression always beats plain OLS on high-dimensional expanded features** — §7's honest finding was that the winner depends on the data; always compare via held-out test performance, not intuition about which method "should" be better.
5. **Assuming Ridge's closed-form solution makes it fastest in practice** — §8 measured the opposite on this problem size; profile before assuming.
6. **Comparing GridSearchCV and ElasticNetCV's chosen hyperparameters directly** — §6 shows they can select different `(alpha, l1_ratio)` pairs due to different internal search grids while reaching statistically similar test performance; compare test scores, not raw hyperparameter values.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.linear_model.Ridge`, `Lasso`, `ElasticNet` | The three regularized regression estimators |
| `sklearn.linear_model.RidgeCV`, `LassoCV`, `ElasticNetCV` | Cross-validated hyperparameter selection per method |
| `sklearn.datasets.make_regression(..., coef=True)` | Synthetic data with a known true sparse coefficient vector |
| `sklearn.model_selection.GridSearchCV` | Manual cross-check against `ElasticNetCV`'s internal search |

---

## 📝 Self-Test Exercises

1. Using §3's bootstrap results (OLS std=13.13, Ridge std=10.37 averaged across the 5 flagged features), explain in your own words what "coefficient std across 200 bootstraps" is actually measuring, and why a lower value is desirable.
2. Section 4 found Lasso's recall stayed at 1.0 across the entire alpha sweep while precision rose from 0.25 to 1.0. Explain why recall staying high is easier to achieve than precision rising — what would cause recall to drop instead?
3. Using the grouping-effect theory in §3 of the math section, explain why Lasso's coefficient split for the two correlated features (1.432 vs. 1.187) was less even than ElasticNet's (1.294 vs. 1.281).
4. Notebook §7 found Lasso beating Ridge on the polynomial-expanded feature set, reversing the notebook's incoming assumption. Propose a change to the experiment (different alpha range, different polynomial degree, different train/test split) that might flip this result, and explain your reasoning using the bias-variance framework from the Model Evaluation & Tuning topic.
5. Section 8 found LassoCV dramatically faster than RidgeCV despite Ridge having a closed-form solution. Research (or reason from first principles about) what `RidgeCV`'s default cross-validation strategy does differently from `LassoCV`'s that could explain this — is there a way to configure `RidgeCV` to close this gap?

---

## 📓 Notebook

30 executed code cells: a baseline comparison of all four methods (unregularized, Ridge, Lasso, ElasticNet) at default hyperparameters, a from-scratch bootstrap coefficient-stability measurement on the Linear Regression topic's VIF-flagged features (with a bar chart confirming Ridge's stabilizing effect), a synthetic sparse-recovery experiment with a full alpha-sweep precision/recall table, a correlated-feature grouping-effect demonstration comparing Lasso and ElasticNet's coefficient splits directly, `ElasticNetCV` tuning both `alpha` and `l1_ratio` simultaneously (cross-checked against manual `GridSearchCV`), a direct continuation of the Polynomial Regression topic's 65-feature expanded Diabetes model comparing Lasso's feature selection against the prior notebook's Ridge result, a final CV-tuned comparison of all three regularized methods with sparsity and coefficient-zeroing tables, and a wall-clock timing comparison that produced a genuinely counter-intuitive result (LassoCV fastest, not RidgeCV):

➡️ **[03_ridge_lasso_elasticnet.ipynb](03_ridge_lasso_elasticnet.ipynb)**

---

## 📚 Further Reading

- [Model Evaluation & Tuning / Bias-Variance & Regularization](../../Model_Evaluation_Tuning/04_Bias_Variance_Regularization/) — full mathematical derivations
- [scikit-learn: Linear Models — Ridge, Lasso, Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn: Lasso model selection](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html)
- [Zou & Hastie (2005): Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf)

---
[← Back to Regression](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Regression.03_Ridge_Lasso_ElasticNet&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
