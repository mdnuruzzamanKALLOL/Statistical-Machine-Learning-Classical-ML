# 🎛️ Hyperparameter Tuning — GridSearchCV, RandomizedSearchCV, Optuna

> Status: ✅ Complete — [Open the notebook →](02_hyperparameter_tuning.ipynb)

Every algorithm notebook in this series picked hyperparameters somewhat by hand. This topic formalizes *how* to find good values systematically, using the previous topic's cross-validation machinery as its evaluation engine.

## 📑 Table of Contents

1. [Concept & Intuition](#-concept--intuition)
2. [Why This Topic Matters](#-why-this-topic-matters)
3. [Mathematical Explanation](#-mathematical-explanation)
4. [Search Strategy Reference](#-search-strategy-reference)
5. [Common Pitfalls & Gotchas](#️-common-pitfalls--gotchas)
6. [Function Reference](#-function-reference-used-in-the-notebook)
7. [Self-Test Exercises](#-self-test-exercises)
8. [Notebook](#-notebook)
9. [Further Reading](#-further-reading)

---

## 📖 Concept & Intuition

Every hyperparameter search strategy in this notebook does the same two things: (1) propose a candidate configuration, (2) score it with cross-validation. They differ only in *how* candidates are proposed:

```
Grid Search:    try EVERY combination in a fixed grid
Random Search:  try N random combinations from specified distributions
Bayesian (Optuna): try N combinations, but each one informed by all previous results
Successive Halving: try many candidates cheaply, keep only the best fraction, repeat with more resources
```

None of these change what "cross-validation" means (previous topic) — they change how efficiently the search space gets explored.

---

## 🎯 Why This Topic Matters

- It's the systematic version of the "try a few values by hand" approach every prior notebook actually used — formalizing intuition that was already being applied informally.
- **Random search's efficiency argument** (§8 in the notebook) is a genuinely counter-intuitive, important result: more thorough-*sounding* search (grid) isn't always more effective search.
- **Nested CV** from the previous topic reappears here in its most practical form (§15): the exact reason `best_score_` is not a number you should ever report as final model performance.
- Optuna's Bayesian approach is standard practice in real ML work — this is many readers' first hands-on exposure to it.

---

## 🧮 Mathematical Explanation

### 1. Grid search's combinatorial cost

For $d$ hyperparameters, each with $n_i$ candidate values, grid search's total cost (in model fits, with $K$-fold CV):

$$\text{Total fits} = K \cdot \prod_{i=1}^{d} n_i$$

The product (not sum) is why grid search scales so poorly — adding one more hyperparameter with even 3 options multiplies total cost by 3, regardless of how many hyperparameters already exist.

### 2. Random search's efficiency argument (Bergstra & Bengio, 2012)

Suppose only $d_{\text{eff}} < d$ hyperparameters actually meaningfully affect the score (the rest are near-irrelevant). A grid with $n$ values per dimension spends $n^d$ total trials but only ever tests $n$ *distinct* values along any single important dimension. Random search with $N$ trials tests, in expectation, close to $N$ *distinct* values along **every** dimension simultaneously — including the important ones — because each trial independently samples every hyperparameter. For the same budget $N \approx n^d$, random search explores each individual dimension far more densely whenever $d_{\text{eff}}$ is small relative to $d$ — exactly what notebook §8 measures directly (score varies far more across `max_depth` than across `min_samples_leaf`).

### 3. Bayesian optimization (TPE, Optuna's default)

Bayesian optimization models $P(\text{score} \mid \text{hyperparameters})$ using results seen so far, then picks the next candidate to evaluate by maximizing an **acquisition function** that balances *exploitation* (regions known to score well) against *exploration* (regions with high uncertainty, that could turn out even better). Optuna's default TPE (Tree-structured Parzen Estimator) approach models $P(\text{hyperparameters} \mid \text{score is good})$ and $P(\text{hyperparameters} \mid \text{score is bad})$ separately as two densities, and prefers candidates that are likely under the "good" density and unlikely under the "bad" one — sampling more from where evidence suggests. This lets the search improve monotonically over trials (notebook §10's running-best curve), a property grid and random search structurally lack.

### 4. Successive halving

Given $B$ candidates and a resource budget, successive halving evaluates all $B$ candidates cheaply (small resource $r_0$), keeps the top fraction $1/\eta$ (e.g. $\eta=2$ keeps the top half), doubles the resource for the survivors, and repeats:

$$B_0 = B \to B_1 = B/\eta \to B_2 = B/\eta^2 \to \dots$$

Total cost is far lower than evaluating every candidate at full resource, because most candidates are eliminated while still cheap to evaluate — exactly what notebook §12 measures (24 candidates in round 1, down to 2 in the final round).

### 5. Nested CV's role here (recap)

As derived in the previous topic: `GridSearchCV.best_score_` is measured on the same folds used to select the winning configuration — an optimistic number by construction. The honest number is either a genuinely held-out test set (used throughout this notebook, §15) or a full nested CV loop.

---

## 🔍 Search Strategy Reference

| Strategy | Best when | Cost profile |
|---|---|---|
| Grid Search | Few hyperparameters, all roughly equally important | Combinatorial, exhaustive |
| Random Search | Many hyperparameters, only a few matter | Linear in budget, no guarantee of optimum |
| Bayesian (Optuna) | Expensive-to-evaluate models, want sample efficiency | Linear in budget, improves over trials |
| Successive Halving | Cheap partial evaluation is meaningful (e.g. fewer trees, less data) | Sub-linear, eliminates weak candidates early |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Reporting `best_score_` as final model performance** — it's a tuning-time, optimistic number (notebook §15 measured a real gap: 0.9626 tuning vs 0.9561 honest test).
2. **Using grid search with many hyperparameters "just to be thorough"** — the combinatorial cost (§1) makes this impractical past 3-4 hyperparameters with more than a couple of values each; switch to random search or Optuna.
3. **Assuming grid search always finds the true optimum** — only within the *specified* grid; a poorly chosen grid can miss the best region entirely, something no exhaustiveness guarantee protects against.
4. **Comparing search strategies on different CV folds** — notebook §7's grid-vs-random comparison and the earlier grid-vs-manual mismatch (§3) both stem from this; always fix the CV strategy when comparing.
5. **Ignoring wall-clock time** — a search with a marginally better CV score but 5x the runtime (notebook §16) may not be the right practical choice, especially during iterative experimentation.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `sklearn.model_selection.GridSearchCV` | Exhaustive grid search |
| `sklearn.model_selection.RandomizedSearchCV` | Random sampling from distributions |
| `sklearn.model_selection.HalvingGridSearchCV` | Successive halving resource allocation |
| `scipy.stats.randint`, `uniform` | Distributions for `RandomizedSearchCV` |
| `optuna.create_study`, `.optimize()` | Bayesian (TPE) hyperparameter search |
| `GridSearchCV(scoring=[...], refit=...)` | Multi-metric scoring |

---

## 📝 Self-Test Exercises

1. Using the formula in §1, compute how many total model fits a 4-hyperparameter grid with 4 values each and 5-fold CV requires — and how that changes if a 5th hyperparameter with 3 values is added.
2. Explain, using §2, why random search's advantage over grid search grows *larger* as more irrelevant hyperparameters are added to the search space.
3. Describe, in your own words, the exploration/exploitation tradeoff in Bayesian optimization (§3) — what would a purely "exploitation-only" search do wrong?
4. Using §4's halving schedule, compute how many candidates survive after 3 rounds starting from 32 candidates with $\eta=2$.
5. Notebook §15 shows a gap between tuning-time CV score and honest test accuracy. Propose a scenario where this gap would be *larger* than what was observed on Breast Cancer — connect it to nested CV's bias argument from the previous topic.

---

## 📓 Notebook

30 executed code cells: a manual nested-loop grid search motivating the automation, `GridSearchCV` reproducing it (with an honest investigation into a fold-strategy mismatch when the two didn't agree exactly), a grid-search-landscape heatmap, combinatorial cost scaling, `RandomizedSearchCV`, a same-budget grid-vs-random comparison, a direct measurement of which hyperparameters matter most, Optuna/TPE Bayesian optimization with a visualized optimization history, a three-way search strategy comparison, `HalvingGridSearchCV`, SVM tuning, multi-metric scoring, the nested-CV-honesty recap with a measured gap, wall-clock timing, and a full score-distribution analysis:

➡️ **[02_hyperparameter_tuning.ipynb](02_hyperparameter_tuning.ipynb)**

---

## 📚 Further Reading

- [scikit-learn: Tuning hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html)
- [Bergstra & Bengio (2012): Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [scikit-learn: Successive Halving](https://scikit-learn.org/stable/modules/grid_search.html#successive-halving-user-guide)

---
[← Back to Model Evaluation & Tuning](../README.md)

<!-- page-views-badge -->
![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Model_Evaluation_Tuning.02_Hyperparameter_Tuning&left_color=%23555555&right_color=%23E67E22&left_text=Page%20Views)
