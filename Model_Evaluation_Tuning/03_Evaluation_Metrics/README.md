# 📏 Evaluation Metrics — Classification & Regression

> Status: ✅ Complete — [Open the notebook →](03_evaluation_metrics.ipynb)

Every notebook in this series has reported "accuracy" or "R²" as if that number were self-evidently the right way to judge a model. This topic derives where those numbers come from, why several of them can disagree with each other on the exact same predictions, and when accuracy is actively the wrong metric to trust.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Metric Reference](#-metric-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Every classification metric in this notebook — accuracy, precision, recall, F1, MCC, Cohen's kappa — is a different arithmetic combination of the exact same four counts from a confusion matrix:

```
                 Predicted 0    Predicted 1
Actual 0            TN              FP
Actual 1            FN              TP
```

Every regression metric — MSE, RMSE, MAE, R² — is a different way of summarizing the same set of residuals $y_i - \hat y_i$. Knowing the raw counts/residuals means every metric below is derivable, not a separate thing to memorize.

---

## 🎯 Why This Topic Matters

- **Accuracy actively lies on imbalanced data** — notebook §5 shows a classifier that always predicts the majority class scoring 94.67% accuracy while catching 0% of the minority class.
- **ROC-AUC can look good while Average Precision reveals the truth** — §7 measures both on the same imbalanced dataset: ROC-AUC 0.928 (looks great) vs. Average Precision 0.610 (the honest picture of minority-class difficulty).
- **Log loss sees what accuracy can't** — §9 shows two predictions that are both "wrong" at the 0.5 threshold can have wildly different log loss (4.61 vs. 0.92), because log loss penalizes *confidently* wrong predictions far more.
- **R² can rise on training data from pure noise, and adjusted R² is one of the few numbers built to catch it** — §11 adds 10 literally-random features and watches what happens to both.

---

## 🧮 Mathematical Explanation

### 1. The confusion matrix and everything derived from it

$$\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}, \quad \text{Precision}=\frac{TP}{TP+FP}, \quad \text{Recall}=\frac{TP}{TP+FN}, \quad F_1=\frac{2 \cdot P \cdot R}{P+R}$$

Precision answers "of everything I flagged positive, how much was actually positive?" Recall answers "of everything actually positive, how much did I catch?" F1 is their harmonic mean — it penalizes a large gap between the two more than a simple average would.

### 2. Matthews Correlation Coefficient & Cohen's Kappa

$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

MCC uses all four confusion-matrix cells symmetrically and equals 0 for a classifier no better than random guessing regardless of class balance — unlike accuracy, which drifts upward toward the majority-class proportion under imbalance (notebook §5 measures this directly: a dummy classifier scores MCC=0.0 exactly, honestly reflecting its uselessness, while its accuracy was 94.67%).

### 3. ROC curve and AUC

The ROC curve plots $\text{TPR}=\frac{TP}{TP+FN}$ against $\text{FPR}=\frac{FP}{FP+TN}$ as the decision threshold sweeps from 1 to 0. AUC — the area under this curve — equals the probability that a randomly chosen positive example is ranked above a randomly chosen negative one by the model's scores. It's threshold-independent and rank-based, not tied to any single classification cutoff.

### 4. Precision-Recall curve and Average Precision

Under severe class imbalance, FPR's denominator ($FP+TN$) is dominated by the huge negative class, so ROC-AUC can look strong even when precision on the rare positive class is poor. The PR curve plots precision against recall directly, and Average Precision (the area under it) is far more sensitive to minority-class performance — exactly what notebook §7 demonstrates on a synthetic 95%/5% dataset.

### 5. Log loss (cross-entropy)

$$\text{LogLoss} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log(\hat p_i) + (1-y_i)\log(1-\hat p_i)\right]$$

Because of the logarithm, a confident wrong prediction ($\hat p=0.01$ when $y=1$) contributes $-\log(0.01)=4.61$ to the loss, while a confident right prediction ($\hat p=0.99$) contributes only $-\log(0.99)=0.01$ — a 460x difference for symmetric confidence levels, which is exactly why log loss (not accuracy) is the standard training objective for probabilistic classifiers.

### 6. MSE, RMSE, MAE

$$\text{MSE}=\frac{1}{n}\sum(y_i-\hat y_i)^2, \quad \text{RMSE}=\sqrt{\text{MSE}}, \quad \text{MAE}=\frac{1}{n}\sum|y_i-\hat y_i|$$

Squaring in MSE/RMSE means a single large-residual outlier contributes disproportionately; MAE weights every residual linearly. Notebook §10 measures this directly: removing the single largest-residual test point drops RMSE far more than it drops MAE.

### 7. R² and adjusted R²

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}, \qquad R^2_{adj} = 1-(1-R^2)\frac{n-1}{n-p-1}$$

$R^2$ is the fraction of target variance explained relative to a "always predict the mean" baseline. On the exact data a model was *fit on*, $R^2$ can only rise or stay flat as features are added — including pure-noise features — because ordinary least squares always finds a fit at least as good with more parameters to work with. Adjusted $R^2$ subtracts a penalty proportional to $p$ (feature count) to counteract this; the notebook applies both to a held-out test set as well, where the effect of overfitting on noise can show up as *falling* raw test $R^2$, not just a smaller adjusted-$R^2$ gap.

---

## 📋 Metric Reference

| Metric | Type | Use when | Blind spot |
|---|---|---|---|
| Accuracy | Classification | Balanced classes | Misleading under imbalance |
| Precision / Recall | Classification | Cost of FP vs. FN differs | Neither alone tells the whole story |
| F1 | Classification | Single number balancing precision & recall | Assumes FP and FN costs are equal |
| MCC / Cohen's Kappa | Classification | Imbalanced classes, want one honest number | Less intuitive than accuracy |
| ROC-AUC | Classification | Threshold-independent ranking quality | Can look inflated under severe imbalance |
| Average Precision (PR-AUC) | Classification | Imbalanced classes specifically | Lower baseline than 0.5, less familiar |
| Log Loss | Classification | Probabilistic predictions matter (not just labels) | Sensitive to extreme/miscalibrated probabilities |
| MSE / RMSE | Regression | Large errors should be penalized heavily | Sensitive to outliers |
| MAE | Regression | All errors should be weighted equally | Less sensitive to a few very bad predictions |
| R² / Adjusted R² | Regression | Variance-explained interpretation wanted | Adjusted R² applied off-training-set is illustrative, not a standard use |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Trusting accuracy on imbalanced data** — notebook §5's dummy classifier scored 94.67% accuracy while having 0% recall on the class that actually mattered.
2. **Using ROC-AUC as the only metric under class imbalance** — §7 shows ROC-AUC (0.928) and Average Precision (0.610) telling very different stories on the same predictions; check both.
3. **Reporting the final classification threshold as always 0.5** — §4 and §13 show precision/recall/F1 all shift substantially as the threshold moves, and the "right" threshold depends on the actual cost of false positives vs. false negatives for the use case.
4. **Averaging multi-class metrics without picking macro vs. weighted deliberately** — §8 shows these can diverge meaningfully when class sizes differ; `macro` treats every class equally regardless of frequency, `weighted` doesn't.
5. **Assuming a calibration gap always means miscalibration** — §14's reliability diagram showed a real gap (0.238) on only ~14 test points per bin; with that little data per bin, some of the gap is unavoidable sampling noise, not necessarily a flaw in the model's probabilities. A larger test set (or fewer, wider bins) would be needed to separate the two causes with confidence.
6. **Applying adjusted R² casually to a test set** — the formula's theoretical motivation is for penalizing added parameters on the *same* data a model was fit on; §11 uses it on a held-out split for illustration, which is useful pedagogically but not the metric's textbook use case.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.metrics.confusion_matrix`, `classification_report` | Confusion-matrix-based summaries |
| `sklearn.metrics.precision_score`, `recall_score`, `f1_score` | Threshold-based classification metrics |
| `sklearn.metrics.roc_curve`, `roc_auc_score` | ROC curve and AUC |
| `sklearn.metrics.precision_recall_curve`, `average_precision_score` | PR curve and PR-AUC |
| `sklearn.metrics.matthews_corrcoef`, `cohen_kappa_score`, `balanced_accuracy_score` | Imbalance-robust classification metrics |
| `sklearn.metrics.log_loss` | Probabilistic classification loss |
| `sklearn.metrics.mean_squared_error`, `mean_absolute_error`, `r2_score` | Regression metrics |
| `sklearn.calibration.calibration_curve` | Reliability diagram data |
| `sklearn.metrics.make_scorer` | Wrap a custom function as a scorer for `cross_validate`/`GridSearchCV` |

---

## 📝 Self-Test Exercises

1. Using the formulas in §1, compute precision, recall, and F1 for TP=40, FP=10, FN=5, TN=145 by hand, then explain in words what each number means for this specific confusion matrix.
2. Explain, using §2, why MCC gives exactly 0 for a classifier that always predicts the majority class, while accuracy on the same classifier can be arbitrarily close to 1.
3. Given the ROC-AUC (0.928) and Average Precision (0.610) measured in §7 on the same imbalanced dataset, explain in your own words why these two numbers can diverge so much on identical predictions.
4. Using §5, compute the log loss for a true label of 0 with predicted probabilities of 0.5, 0.1, and 0.99 — which prediction is punished most severely, and why?
5. Notebook §11 found that adding 10 pure-noise features raised training R² but lowered test R² and adjusted R² — explain why OLS's training-R² guarantee (never decreases) does *not* extend to a held-out test set.

---

## 📓 Notebook

31 executed code cells: manual confusion matrix construction validated against sklearn, precision/recall/F1 derivation, a precision-recall-vs-threshold sweep, a synthetic 95%/5% imbalanced dataset showing accuracy's blind spot alongside balanced accuracy/MCC/kappa, manual ROC curve and trapezoidal-rule AUC matched against sklearn, a side-by-side ROC vs. Precision-Recall curve comparison on imbalanced data, multi-class metrics with macro/weighted/micro averaging derived by hand, manual log loss matched against sklearn plus a confident-vs-tentative-wrong-prediction comparison, manual MSE/RMSE/MAE with a direct single-outlier-removal sensitivity test, manual R²/adjusted R² with a pure-noise-feature injection experiment (honestly reporting the actual train-vs-test direction observed), residual plots, a custom cost-weighted scorer used inside `cross_validate`, business-constrained threshold tuning, a calibration reliability diagram (with an honest small-sample-noise caveat), and a full multi-classifier, multi-metric comparison table:

➡️ **[03_evaluation_metrics.ipynb](03_evaluation_metrics.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Model evaluation — quantifying prediction quality](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [scikit-learn: Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
- [scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Davis & Goadrich (2006): The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf)

---
[← Back to Model Evaluation & Tuning](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Model_Evaluation_Tuning.03_Evaluation_Metrics&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
