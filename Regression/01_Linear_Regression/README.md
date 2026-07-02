# 📈 Linear Regression — OLS, Gradient Descent & Assumptions

> Status: ✅ Complete — [Open the notebook →](01_linear_regression.ipynb)

The first algorithm in this series' Regression category, and the foundation every later regression topic (Ridge/Lasso, SVR, tree-based regressors) will be compared against. This notebook derives Ordinary Least Squares two different ways — closed-form and gradient descent — verifies they agree, then checks the assumptions OLS actually depends on.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Fitting Method Reference](#-fitting-method-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Linear regression fits a straight line (or hyperplane, with multiple features) that minimizes the total squared vertical distance between the line and every data point:

```
y
|        *
|      *   \
|    *       \___ fitted line minimizes
|  *              sum of squared vertical
|*                 distances to every point
+------------------ x
```

There are two completely different ways to find that minimizing line: solve for it exactly in one step (the normal equation), or walk downhill on the cost surface iteratively (gradient descent). Both are derived and cross-checked in this notebook.

---

## 🎯 Why This Topic Matters

- **Every other regression algorithm in this category is a modification of this one** — Ridge/Lasso add a penalty term to the same cost function, SVR changes the loss function, tree-based regressors abandon the linear form entirely but still get compared against this baseline.
- **Gradient descent, introduced here from scratch, is the same optimization procedure used to train neural networks** — the batch/SGD/mini-batch distinction in §7 is directly the same distinction used when training deep learning models on large datasets.
- **"The model fits well" and "the coefficients can be trusted individually" are different claims** — §9-11's assumption-checking (residual diagnostics, VIF, confidence intervals) is what separates the two, and the notebook found a real example: the Diabetes dataset's `s1`-`s5` features have VIF > 5 (genuine multicollinearity), which directly explains why several of their individual coefficients aren't statistically significant despite a reasonable overall model fit.
- **A single outlier moved a fitted slope by 55%** — §10 measures this directly, motivating why raw OLS is sensitive to influential points in a way robust methods (covered in later topics) are designed to fix.

---

## 🧮 Mathematical Explanation

### 1. The cost function

$$J(\beta_0, \beta_1) = \frac{1}{2n}\sum_{i=1}^n (y_i - (\beta_0 + \beta_1 x_i))^2$$

The $\frac{1}{2}$ is a calculus convenience that cancels the 2 produced by differentiating the square — it doesn't change which $(\beta_0, \beta_1)$ minimizes $J$.

### 2. The normal equation (closed-form solution)

Setting $\nabla J = 0$ and solving analytically:

$$\hat\beta = (X^TX)^{-1}X^Ty$$

Exact in one step, as long as $X^TX$ is invertible (it isn't when features are perfectly collinear, or when there are more features than samples).

### 3. Gradient descent

$$\beta_j := \beta_j - \eta \frac{\partial J}{\partial \beta_j}, \qquad \frac{\partial J}{\partial \beta_0} = \frac{1}{n}\sum(\hat y_i - y_i), \qquad \frac{\partial J}{\partial \beta_1} = \frac{1}{n}\sum(\hat y_i - y_i)x_i$$

Because MSE is convex in $(\beta_0, \beta_1)$, gradient descent with a suitable learning rate $\eta$ is guaranteed to converge to the same global minimum the normal equation finds directly — notebook §4 verifies this numerically.

### 4. Why feature scale matters for gradient descent

With features on very different scales, the cost surface becomes an elongated, narrow valley rather than a roughly circular bowl. The gradient then points almost perpendicular to the direction of the minimum, forcing a much smaller learning rate to avoid overshooting — §6 demonstrates this directly: an unscaled two-feature dataset overflows to NaN at any learning rate above roughly $10^{-6}$, while the same data scaled with `StandardScaler` converges cleanly at a rate a million times larger.

### 5. Variance Inflation Factor (multicollinearity)

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ comes from regressing feature $j$ on every other feature. $VIF_j > 5$ flags a feature whose information substantially overlaps with the others — its own coefficient becomes unstable (large standard error) even when the model's overall fit is unaffected.

### 6. Coefficient standard errors and hypothesis testing

$$\text{Var}(\hat\beta) = \hat\sigma^2 (X^TX)^{-1}, \qquad \hat\sigma^2 = \frac{\sum(y_i - \hat y_i)^2}{n - p - 1}$$

Each coefficient's standard error is the square root of the corresponding diagonal entry. Dividing the coefficient by its standard error gives a t-statistic, testable against a t-distribution with $n-p-1$ degrees of freedom — the basis for asking "could this feature's true effect plausibly be zero?"

---

## 📋 Fitting Method Reference

| Method | Cost per step | Convergence | Best when |
|---|---|---|---|
| Normal Equation | One $O(p^3)$ matrix inversion | Exact, one step | Small/moderate feature count, dataset fits in memory |
| Batch Gradient Descent | Full dataset per step | Smooth, guaranteed for convex cost | Feature count too large for matrix inversion |
| Stochastic Gradient Descent (SGD) | One sample per step | Noisy, many small steps | Very large datasets, online/streaming learning |
| Mini-Batch Gradient Descent | Small batch per step | Between batch and SGD | The practical default for large-scale training |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Not scaling features before gradient descent** — §6 shows unscaled features forcing a learning rate so small that convergence barely progresses in 200 iterations, while scaled features converge cleanly at a rate a million times larger.
2. **Picking a learning rate that's too large** — §5 shows a learning rate of 1.5 causing the cost to explode within 20 iterations rather than converge; always check the cost history plot, not just the final number.
3. **Reading a large coefficient as a "strong effect" without checking its p-value** — §11 found `age`'s coefficient (47.75) statistically indistinguishable from zero (p=0.51), while a smaller-magnitude but more stable feature was highly significant; magnitude alone doesn't establish reliability.
4. **Ignoring multicollinearity** — §9's VIF analysis found `s1`-`s5` in the Diabetes dataset all exceed VIF=5, meaning their individual coefficients (though the model's overall R² is unaffected) shouldn't be interpreted as isolated per-feature effects.
5. **Assuming one influential outlier can't matter much** — §10 measured a single added point shifting a fitted slope by 55%; always visualize residuals and leverage before trusting a fitted line's coefficients on small or noisy datasets.
6. **Treating the normal equation as always superior** — §7's timing comparison shows it's fastest on small data, but its $O(p^3)$ matrix inversion becomes the bottleneck as feature count grows, which is exactly why gradient descent variants exist.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `numpy.linalg.inv` | Manual normal equation solve |
| `sklearn.linear_model.LinearRegression` | OLS via normal equation (verified against manual implementation) |
| `sklearn.linear_model.SGDRegressor` | sklearn's stochastic gradient descent regressor |
| `sklearn.preprocessing.StandardScaler` | Feature scaling before gradient descent |
| `scipy.stats.probplot`, `shapiro` | Residual normality diagnostics |
| `scipy.stats.t.cdf` | Coefficient significance testing |

---

## 📝 Self-Test Exercises

1. Using §1's cost function, compute $J$ by hand for a 3-point dataset of your choosing at two different $(\beta_0, \beta_1)$ pairs, and verify which one has lower cost matches your intuition about which line fits better.
2. Explain, using §3's gradient formulas, why gradient descent's update direction is always *opposite* the gradient — what would happen if the sign were flipped?
3. Using §4, explain in your own words why an elongated cost surface (from unscaled features) forces a smaller learning rate than a circular one, even though both surfaces have the same global minimum.
4. Notebook §9 found `s1`-`s5` all have VIF > 5. Explain what this means for interpreting `s1`'s individual coefficient, even though the model's overall test R² (0.4849) is unaffected by the multicollinearity.
5. Section 11 found `age` was not statistically significant (p=0.51) despite having a nonzero coefficient. Propose what additional data or experiment would help determine whether `age` truly has no effect, or whether the current sample size/multicollinearity is simply hiding a real effect.

---

## 📓 Notebook

30 executed code cells: 3D and contour visualization of the MSE cost surface, the normal equation implemented from scratch and verified against sklearn, gradient descent implemented from scratch with its optimization path plotted directly on the cost surface, a learning-rate sensitivity sweep including a deliberately diverging case, a feature-scaling demonstration where unscaled gradient descent overflows to NaN above a learning rate of roughly 1e-6, batch/SGD/mini-batch gradient descent compared against each other and against `sklearn.SGDRegressor`, a wall-clock timing comparison of all four fitting methods, multivariate regression on the real Diabetes dataset with train/test R² and a predicted-vs-actual plot, residual diagnostics (residuals-vs-fitted, Q-Q plot, Shapiro-Wilk test), a from-scratch Variance Inflation Factor implementation that found real multicollinearity among five features, a direct demonstration of a single outlier shifting a fitted slope by 55%, and manual coefficient standard errors / t-tests / p-values matched against the closed-form OLS inference formula:

➡️ **[01_linear_regression.ipynb](01_linear_regression.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Ordinary Least Squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [scikit-learn: Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)
- [An Introduction to Statistical Learning — Ch. 3 (Linear Regression)](https://www.statlearning.com/)
- [Variance Inflation Factor — statsmodels documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)

---
[← Back to Regression](../README.md)
