# 🎲 Naive Bayes

> Status: ✅ Complete — [Open the notebook →](03_naive_bayes.ipynb)

The most direct application of the [Math Refresher](https://github.com/mdnuruzzamanKALLOL/Statistical-Machine-Learning-Foundation/tree/main/05_Math_Refresher)'s Bayes' theorem in this entire series — Naive Bayes computes $P(\text{class} \mid \text{features})$ using exactly the prior × likelihood / evidence formula, class by class, prediction by prediction.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Variant Selection Guide](#-variant-selection-guide)
5. [Generative vs Discriminative](#-generative-vs-discriminative)
6. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
7. [Function Reference](#-function-reference-used-in-the-notebook)
8. [Self-Test Exercises](#-self-test-exercises)
9. [Notebook](#-notebook)
10. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Naive Bayes flips the classification question around. Instead of directly modeling "given these features, what's the class?" (what Logistic Regression does), it models "given this class, what features would I expect to see?" — then uses **Bayes' theorem** to invert that into the answer actually wanted:

$$P(C \mid x) = \frac{P(x \mid C)\, P(C)}{P(x)}$$

The "naive" part: to make $P(x \mid C)$ computable at all for many features at once, it assumes every feature is independent of every other feature, *given the class*. This is almost never exactly true, but the resulting classifier is fast, simple, and often surprisingly competitive.

---

## 🎯 Why This Topic Matters

- It's the first **generative** model in this series — see [Generative vs Discriminative](#-generative-vs-discriminative) below, a distinction that reappears with LDA/QDA later in this category.
- It trains via **closed-form parameter estimates** (means, variances, counts) — no gradient descent, no iterative optimization, a genuinely different training cost profile from every model so far.
- It's still the standard first choice for **text classification** (spam filters, sentiment analysis) — the notebook builds a real spam classifier from raw text.
- **Laplace smoothing** (§8 in the notebook) is a small but essential idea that reappears anywhere probabilities are estimated from counts.

---

## 🧮 Mathematical Explanation

### 1. Bayes' theorem for classification

For a class $C$ and a feature vector $x = (x_1, \dots, x_d)$:

$$P(C \mid x_1, \dots, x_d) = \frac{P(x_1, \dots, x_d \mid C)\, P(C)}{P(x_1, \dots, x_d)}$$

Since the denominator $P(x)$ is the same for every class being compared, classification only needs to find the class that maximizes the numerator:

$$\hat{C} = \arg\max_{C} \ P(x_1, \dots, x_d \mid C)\, P(C)$$

### 2. The naive independence assumption

Estimating the full joint likelihood $P(x_1, \dots, x_d \mid C)$ directly requires exponentially many parameters as $d$ grows. The naive assumption factorizes it into a simple product:

$$P(x_1, \dots, x_d \mid C) = \prod_{j=1}^d P(x_j \mid C)$$

This turns an intractable estimation problem into $d$ separate, easy 1D estimation problems — the entire reason Naive Bayes is fast and simple, at the cost of ignoring real feature interactions.

### 3. Gaussian Naive Bayes

For continuous features, assume $x_j \mid C \sim \mathcal{N}(\mu_{j,C}, \sigma^2_{j,C})$ — the exact Normal distribution PDF from the Math Refresher:

$$P(x_j \mid C) = \frac{1}{\sqrt{2\pi\sigma^2_{j,C}}} \exp\left(-\frac{(x_j - \mu_{j,C})^2}{2\sigma^2_{j,C}}\right)$$

$\mu_{j,C}$ and $\sigma^2_{j,C}$ are just the sample mean and variance of feature $j$, computed only from training examples in class $C$ — a closed-form estimate, no optimization loop required.

### 4. Multinomial Naive Bayes

For count features (e.g. word frequencies), the likelihood of observing counts $x = (x_1, \dots, x_d)$ given class $C$ follows a multinomial distribution, and the per-word probability estimate is:

$$P(x_j \mid C) = \frac{\text{count}(x_j, C) + \alpha}{\sum_{k} \text{count}(x_k, C) + \alpha d}$$

($\alpha$ is the Laplace smoothing constant — see §6.) This is literally "what fraction of all words in class $C$'s documents was word $j$," smoothed.

### 5. Bernoulli Naive Bayes

For binary features (presence/absence, not counts):

$$P(x_j \mid C) = p_{j,C}^{x_j} (1 - p_{j,C})^{(1 - x_j)}$$

where $p_{j,C}$ is the fraction of class-$C$ training examples where feature $j$ is present ($x_j = 1$). Critically, this formula also penalizes **absence** — a word that's *usually present* in spam but is *missing* from a given email counts as evidence *against* spam, something Multinomial's pure count-accumulation doesn't naturally do.

### 6. Laplace (additive) smoothing

Without smoothing, a feature value never seen with class $C$ in training gets $P(x_j \mid C) = 0$ exactly, which zeroes out the *entire* product in §2 — one unseen word overrides all other evidence. Laplace smoothing adds a small constant everywhere:

$$P(x_j \mid C) = \frac{\text{count}(x_j, C) + \alpha}{N_C + \alpha \cdot d}$$

With $\alpha = 1$ (the default, "add-one smoothing"), every possible feature value gets a small non-zero probability floor, however rare.

---

## 🔀 Variant Selection Guide

| Variant | Feature type | Formula basis | Typical use |
|---|---|---|---|
| `GaussianNB` | Continuous | Normal PDF | Sensor data, physical measurements |
| `MultinomialNB` | Non-negative counts | Multinomial distribution | Word counts, event frequencies, text classification |
| `BernoulliNB` | Binary (0/1) | Bernoulli distribution | Word presence/absence, binary flags |

Using the wrong variant for a feature type (e.g. `GaussianNB` on raw word counts) produces a real, mathematically-mismatched model, not just a suboptimal one.

---

## ⚔️ Generative vs Discriminative

| | Naive Bayes (Generative) | Logistic Regression (Discriminative) |
|---|---|---|
| Models | $P(x \mid C)$, inverts via Bayes | $P(C \mid x)$, directly |
| Assumes feature independence? | Yes (the "naive" part) | No |
| Training | Closed-form estimates | Iterative (gradient descent) |
| Can generate synthetic data? | Yes (sample from the learned $P(x \mid C)$) | No |
| Tends to win when | Small data, strong prior structure, categorical/text features | Larger data, correlated features |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Mismatching the NB variant to feature type** — see the selection guide above; this is the most common Naive Bayes mistake.
2. **Forgetting Laplace smoothing on sparse count/binary data** — the zero-frequency problem (notebook §7) silently and completely breaks predictions containing any unseen category, without raising an error.
3. **Ignoring strongly correlated features** — the independence assumption's violation (notebook §15) tends to hurt probability *calibration* more than raw accuracy; if you need well-calibrated probabilities (not just correct class labels), correlated features are a real risk.
4. **Applying GaussianNB to skewed/non-Normal continuous features** without checking — a heavily skewed feature (topic 04's log-transform candidates) violates GaussianNB's core assumption; consider transforming first.
5. **Confusing Multinomial and Bernoulli on text** — Multinomial captures word *repetition* as signal, Bernoulli only captures presence; picking wrong for a task where repetition matters (e.g. "free free free" being extra spammy) loses real information.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.naive_bayes.GaussianNB` | Continuous-feature Naive Bayes |
| `sklearn.naive_bayes.MultinomialNB` | Count-feature Naive Bayes (text classification) |
| `sklearn.naive_bayes.BernoulliNB` | Binary-feature Naive Bayes |
| `sklearn.feature_extraction.text.CountVectorizer` | Text → word-count feature matrix |
| `.predict_proba()` | Class probabilities, not just hard labels |
| `alpha=` parameter | Laplace smoothing strength |

---

## 📝 Self-Test Exercises

1. Derive, from the formula in §2, why estimating a full joint likelihood without the independence assumption requires exponentially many parameters as feature count grows.
2. Given a binary feature that's present in 100% of class-A training examples and 0% of class-B, predict what happens to a new class-B example that happens to have this feature present — with and without Laplace smoothing.
3. Explain why GaussianNB's decision boundary (notebook §6) can be curved, while Logistic Regression's is always a straight line.
4. For a document classification task where "URGENT URGENT URGENT" should score more suspiciously than "urgent," argue for Multinomial over Bernoulli.
5. Naive Bayes is generative — sketch (in words) how you'd use a fitted `GaussianNB` model to *generate* a synthetic example of a given class, rather than just classify one.

---

## 📓 Notebook

34 executed cells: a Bayes' theorem recap tying directly to the Math Refresher, a from-scratch Gaussian Naive Bayes verified to match scikit-learn exactly (100% agreement), decision boundary visualization, the zero-frequency problem demonstrated concretely, Laplace smoothing's fix, a real spam-classification pipeline with `MultinomialNB` and `BernoulliNB`, a Wine dataset application (97.2% accuracy), an empirical test of the independence assumption's real-world cost, and a head-to-head against Logistic Regression:

➡️ **[03_naive_bayes.ipynb](03_naive_bayes.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scikit-learn: CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) (visual intuition)

---
[← Back to Classification](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Classification.03_Naive_Bayes&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
