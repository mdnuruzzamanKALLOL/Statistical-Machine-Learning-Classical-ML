# 🌀 Polynomial Regression — Feature Expansion for Non-Linear Curves

> Status: ✅ Complete — [Open the notebook →](02_polynomial_regression.ipynb)

Linear regression (previous topic) can only fit a straight line. Polynomial regression doesn't change the underlying algorithm at all — it changes the *features*, expanding $x$ into $x, x^2, x^3, \dots$ and fitting an ordinary linear model to the expanded set. This notebook covers what that expansion actually generates, how to pick a degree, and two dangers that come with the extra flexibility: extrapolation and combinatorial feature blowup.

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
Original feature:        x
Expanded (degree 3):     [1, x, x^2, x^3]
                              |
                     ordinary LinearRegression
                     fit on the expanded columns
                              |
                    y = b0 + b1*x + b2*x^2 + b3*x^3
```

The model is still **linear in its parameters** ($b_0, b_1, b_2, b_3$) — only the *relationship to the original $x$* becomes curved. This is why `PolynomialFeatures` + `LinearRegression` is a preprocessing trick, not a new algorithm, and why everything learned about OLS in the previous topic (normal equation, gradient descent, assumption-checking) still applies directly.

---

## 🎯 Why This Topic Matters

- **A straight line structurally cannot fit curved data** — notebook §1 measures this directly: plain linear regression scores a *negative* test R² (-0.047, worse than predicting the mean) on a genuinely quadratic reaction-yield relationship.
- **The feature count explosion is a real, measurable constraint** — §5 shows even a modest degree-3 expansion on 10 input features produces hundreds of columns, most of them interaction terms; §9 then demonstrates the resulting overfitting directly (train R²=0.605 vs. test R²=0.424, a 0.18 gap).
- **High-degree polynomials are dangerous outside their training range** — §6 shows a degree-12 model predicting a physically impossible -408% yield just 20 units past the training data's edge, while a linear model degrades far more gracefully at the same point.
- **Polynomial expansion isn't automatically beneficial** — §8's honest real-data check found the Diabetes `bmi` feature's best polynomial degree was 1 (plain linear), with zero measurable gain from curvature — a useful contrast to §1-6's synthetic dataset where curvature clearly helped.

---

## 🧮 Mathematical Explanation

### 1. The feature expansion

For one input feature $x$ and degree $d$:

$$\phi(x) = [1, x, x^2, \dots, x^d]$$

Then an ordinary least-squares fit is performed on $\phi(x)$ exactly as in the previous topic. Notebook §2 verifies a from-scratch implementation of this expansion matches `sklearn.preprocessing.PolynomialFeatures` exactly.

### 2. Multivariate expansion and interaction terms

With $n$ input features and degree $d$, the expansion includes every monomial $x_1^{a_1} x_2^{a_2} \cdots x_n^{a_n}$ with $\sum a_i \le d$ — not just powers of individual features, but cross terms like $x_1 x_2$ that capture joint effects neither feature alone can represent.

### 3. The combinatorial feature count

$$\text{Number of features} = \binom{n+d}{d}$$

This grows combinatorially in both $n$ (input feature count) and $d$ (degree) — notebook §5 tabulates this directly and shows Diabetes's 10 features become 65 columns at degree 2, and hundreds at degree 3.

### 4. Runge's phenomenon (extrapolation instability)

High-degree polynomials fit training data well but can oscillate with increasing amplitude near the edges of, and especially outside, the training range. This isn't a bug in the fitting procedure — it's an inherent property of high-degree polynomial interpolation, independent of noise. Notebook §6 demonstrates this concretely on real predictions rather than just an abstract curve shape.

### 5. Scale growth under exponentiation

If $x$ ranges over $[20, 100]$, then $x^{10}$ ranges over roughly $[10^{13}, 10^{19}]$ — an enormous scale mismatch between low- and high-degree columns in the same feature matrix. Notebook §7 measures this directly and shows it causing `Ridge` to hit numerically ill-conditioned linear algebra (`rcond` ~$10^{-41}$) when fit on unscaled polynomial features, in addition to distorting the regularization penalty itself (a problem introduced in the Linear Regression topic and covered in full in the next topic).

---

## 📋 Quick Reference

| Concept | Key takeaway |
|---|---|
| `PolynomialFeatures(degree=d)` | Expands $x \to [1, x, \dots, x^d]$; still fit with ordinary `LinearRegression` |
| `interaction_only=True` | Keeps cross terms ($x_1 x_2$) but drops pure powers ($x_1^2$) — useful for categorical-like features |
| Degree selection | Use `validation_curve`/cross-validation (§4), not just visual inspection |
| Feature count | Grows as $\binom{n+d}{d}$ — combinatorial, not linear, in both inputs and degree |
| Extrapolation | High-degree fits can be wildly wrong just outside the training range |
| Scaling | Required before regularizing polynomial features — power terms differ in scale by orders of magnitude |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Choosing degree by looking at the training-fit plot alone** — a high-degree curve can look like it "fits better" visually while actually having worse test error; §4's cross-validated selection is the honest check.
2. **Trusting predictions outside the training data's range** — §6 shows a degree-12 model predicting an impossible -408% yield just 20 units past the training range's edge; never extrapolate a high-degree polynomial fit.
3. **Applying polynomial expansion to many features without regularization** — §9 shows a 65-feature degree-2 expansion on Diabetes overfitting with a 0.18 train-test R² gap; pair polynomial features with Ridge/Lasso (next topic) once feature count grows.
4. **Forgetting to scale before regularizing polynomial features** — §7 shows a scale mismatch of many orders of magnitude between low- and high-degree columns, which can push the underlying linear algebra itself into numerical instability, not just distort the penalty.
5. **Assuming curvature exists without checking** — §8's honest result on Diabetes's `bmi` feature found zero real benefit from polynomial expansion; always compare against plain linear regression on held-out data before assuming a curved model is worth the added complexity.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.preprocessing.PolynomialFeatures` | Expand features into polynomial + interaction terms |
| `sklearn.pipeline.make_pipeline` | Chain `PolynomialFeatures` + model into one estimator |
| `sklearn.model_selection.validation_curve` | Cross-validated degree selection |
| `sklearn.linear_model.Ridge` | Regularized fit on expanded polynomial features |
| `math.comb` | Combinatorial feature-count formula verification |

---

## 📝 Self-Test Exercises

1. Using §1's formula for $\phi(x)$, write out the full expanded feature vector for $x=3$ at degree 4.
2. Explain, using §3's combinatorial formula, why going from 5 to 10 input features at a fixed degree causes a much larger jump in feature count than going from degree 2 to degree 3 at a fixed 5 input features (compute both to check your intuition).
3. Notebook §6 found a degree-12 model predicting -408% yield at temperature=120, well outside its 20-100 training range. Explain why a degree-1 (linear) model doesn't show the same instability, even though it was also a comparatively poor overall fit.
4. Using §7's scale-growth measurements, explain why fitting `Ridge(alpha=1.0)` directly on unscaled degree-10 polynomial features produced both a numerically unstable fit *and* a misleadingly small coefficient-magnitude sum compared to the scaled version.
5. Section 8 found polynomial expansion didn't help on Diabetes's `bmi` feature. Propose an experiment (a different feature, a feature combination, or a different degree range) that might reveal genuine curvature elsewhere in the dataset, and explain how you'd verify honestly whether it does.

---

## 📓 Notebook

30 executed code cells: a synthetic quadratic reaction-yield dataset motivating why a straight line structurally fails, a from-scratch polynomial feature expansion verified against `PolynomialFeatures` (including `interaction_only` behavior), underfitting/good-fit/overfitting shown visually across four degrees with a numeric train/test MSE table and residual diagnostics, cross-validated degree selection via `validation_curve` that correctly recovered the true quadratic degree, a combinatorial feature-count table verified against the $\binom{n+d}{d}$ formula, a Runge's-phenomenon extrapolation demonstration with concrete impossible predictions, a feature-scale-growth measurement showing why polynomial features need scaling (including a real ill-conditioned-matrix warning triggered on unscaled data), an honest real-data check on Diabetes's `bmi` feature finding no meaningful benefit from curvature, and a full multivariate polynomial-plus-Ridge comparison against plain linear regression with a measured overfitting gap:

➡️ **[02_polynomial_regression.ipynb](02_polynomial_regression.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Polynomial and Spline Features](https://scikit-learn.org/stable/modules/preprocessing.html#polynomial-features)
- [scikit-learn: Underfitting vs. Overfitting (polynomial degree example)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [Runge's phenomenon — Wikipedia](https://en.wikipedia.org/wiki/Runge%27s_phenomenon)

---
[← Back to Regression](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Regression.02_Polynomial_Regression&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
