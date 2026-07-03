# 🛒 Association Rules (Apriori) — Market Basket Analysis

> Status: ✅ Complete — [Open the notebook →](07_association_rules_apriori.ipynb)

Every prior Unsupervised topic worked with continuous numerical features and distance or density. Association rule mining works with something structurally different: sets of discrete items, and the question isn't "how far apart are these points" but "which items tend to appear together." This is the final topic in the Classical ML series — Apriori's core insight (the downward-closure property) is what makes searching an otherwise combinatorially explosive space of itemsets tractable, and this notebook builds it from scratch, verifies it against `mlxtend`, and is explicit about a real, common trap: confidence alone can be a misleading measure of association.

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

```
Transactions:  {bread, milk}, {bread, diaper, beer}, {milk, diaper, beer, cola}, ...
                          |
        find ITEMSETS that appear together often enough (support >= threshold)
                          |
        turn frequent itemsets into RULES: "if A is in the basket, B likely is too"
                          |
        keep only rules that are both frequent (support) AND reliable (confidence)
        AND genuinely informative, not just coincidentally common (lift)
```

---

## 🎯 Why This Topic Matters

- **The Apriori principle is what makes this search tractable at all** — §4 demonstrates the downward-closure property directly: once a pair of items is found infrequent, every 3+ item superset containing that pair is *guaranteed* infrequent too, and can be skipped without ever computing its support.
- **A from-scratch Apriori implementation was validated at every stage, not just the final answer** — §5 validates frequent itemset discovery against `mlxtend.apriori` (15/15 itemsets, all support values matching), and a separate rule-generation function was validated against `mlxtend.association_rules` (11/11 rules matching exactly) — two independent checks, not one.
- **A real, common trap was constructed and caught, not just described** — §8 builds a dataset where `saffron → milk` has confidence=1.000 (looks like a perfect rule!) but lift=1.036 (barely above 1.0, revealing milk was *already* in 95% of transactions regardless of saffron) — concrete proof that confidence alone can be deeply misleading.
- **An even more subtle version of the same lesson showed up unprompted in the larger dataset** — §10's two deliberately-designed associations (`item_0`/`item_1` at ~40% co-occurrence, `item_5`/`item_6` at ~15%) produced lifts of 1.610 and 2.062 respectively — the *weaker*-designed association scored *higher* lift, a genuinely instructive, honestly-reported surprise that reinforces §8's core point: lift measures relative surprise, not raw co-occurrence frequency.
- **This closes the entire Classical ML repository** — 5 categories, 29 algorithms, ending on a data structure (discrete transactions) and toolkit (support/confidence/lift) unlike anything covered before, while keeping the same discipline: every claim checked against actual execution output.

---

## 🧮 Mathematical Explanation

### 1. Support

$$\text{support}(X) = \frac{|\{t \in T : X \subseteq t\}|}{|T|}$$

The fraction of all transactions containing itemset $X$ — an empirical marginal probability $P(X)$.

### 2. The Apriori (downward-closure) principle

If itemset $X$ is infrequent (support below threshold), then every superset $Y \supseteq X$ must also be infrequent, since $Y$'s transactions are a subset of $X$'s transactions (adding more required items can only keep or shrink the matching transaction count). This single fact is what lets Apriori prune entire branches of the itemset search tree without ever computing their support — §4 verifies this concretely, and §5's from-scratch implementation only ever builds candidate $k$-itemsets from *known-frequent* $(k-1)$-itemsets, directly exploiting this property.

### 3. Confidence

$$\text{confidence}(A \Rightarrow B) = \frac{\text{support}(A \cup B)}{\text{support}(A)} = P(B \mid A)$$

The empirical conditional probability of $B$ given $A$. Notably **not symmetric** — §6 shows `confidence(diaper→beer)` and `confidence(beer→diaper)` are different numbers even though both are built from the same joint support.

### 4. Lift

$$\text{lift}(A \Rightarrow B) = \frac{\text{confidence}(A \Rightarrow B)}{\text{support}(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$

Compares the rule's confidence against $B$'s baseline frequency. Lift $> 1$: $A$ and $B$ co-occur more than chance predicts (positive association). Lift $= 1$: statistically independent — $A$ tells you nothing about $B$ beyond $B$'s own base rate. Lift $< 1$: negative association. §8's entire pedagogical point is that confidence alone cannot distinguish "genuine association" from "both items are just individually common."

### 5. Complexity

Apriori's worst-case cost is still combinatorial in the number of items, but downward-closure pruning makes it far better than brute-force enumeration of all $2^n - 1$ possible itemsets in practice — §9 measures the actual growth in frequent-itemset count as `min_support` is lowered, and §12 compares Apriori's candidate-generation-then-test approach directly against FP-Growth's tree-compression approach, confirming both always find identical itemsets despite searching differently.

---

## 📋 Metric Reference

| Metric | Formula | Answers |
|---|---|---|
| Support | $P(A \cap B)$ | How common is this itemset overall? |
| Confidence | $P(B \mid A)$ | Given $A$, how often does $B$ follow? |
| Lift | $\frac{P(A \cap B)}{P(A)P(B)}$ | Is this association *more* than coincidence? |

---

## ⚠️ Common Pitfalls & Gotchas

1. **Trusting confidence alone to judge a rule's usefulness** — §8's saffron/milk trap (confidence=1.0, lift=1.036) is the central lesson of this notebook; always check lift alongside confidence.
2. **Assuming a stronger-designed association always produces higher lift** — §10 found the opposite in this dataset; lift depends on baseline frequency too, not just co-occurrence strength, so intuition here can mislead without checking the actual numbers.
3. **Forgetting confidence is directional** — §6 shows `confidence(diaper→beer) ≠ confidence(beer→diaper)`; always specify which item is the antecedent.
4. **Setting `min_support` too low without expecting the combinatorial consequences** — §9 shows frequent itemset count growing from 2 (at `min_support=0.5`) to 699 (at `min_support=0.01`) on the same data; a very low threshold can produce an unmanageably large rule set.
5. **Assuming Apriori and FP-Growth might find different rules** — §12 confirms both always find identical frequent itemsets; they differ only in search strategy/speed, never in what counts as "frequent."
6. **Treating one-hot-encoded item columns as if they were the same kind of feature as the rest of this series' notebooks** — this data has no meaningful distance, no ordering, and no missing-value handling in the usual sense; the Apriori/support/confidence/lift toolkit is specific to this transactional data structure.

---

## 🔑 Function Reference (used in the notebook)

| Function | Purpose |
|---|---|
| `mlxtend.preprocessing.TransactionEncoder` | Converts transaction lists into a one-hot boolean DataFrame |
| `mlxtend.frequent_patterns.apriori` | Frequent itemset mining (candidate-generation approach) |
| `mlxtend.frequent_patterns.fpgrowth` | Frequent itemset mining (tree-compression approach) |
| `mlxtend.frequent_patterns.association_rules` | Generates rules with support/confidence/lift from frequent itemsets |

---

## 📝 Self-Test Exercises

1. Using §1's support formula, compute by hand the support of `{bread, milk}` given a small transaction list of your choosing.
2. Section 4 demonstrated downward closure with a specific infrequent pair. Explain, in your own words, why a 3-item superset's support can never *exceed* any of its 2-item subsets' supports.
3. Using §3 and §4's formulas, compute confidence and lift by hand for a rule where $A$ and $B$ are statistically independent (i.e., $P(A \cap B) = P(A) \cdot P(B)$) — confirm the lift comes out to exactly 1.
4. Section 8 found a high-confidence, low-lift rule (`saffron → milk`). Construct your own example (in words, not code) of a different scenario where confidence would be high but lift would reveal the rule as uninformative.
5. Section 10 found the deliberately weaker association (`item_5`/`item_6`) scoring *higher* lift than the deliberately stronger one (`item_0`/`item_1`). Using §4's formula, propose what property of `item_0` and `item_1`'s *individual* baseline frequencies could explain this counter-intuitive result.

---

## 📓 Notebook

30 executed code cells: transaction one-hot encoding, a from-scratch support calculator verified against `mlxtend.apriori`, a concrete demonstration of the downward-closure pruning principle, a from-scratch level-wise Apriori implementation validated to find identical frequent itemsets (15/15) with identical support values as `mlxtend`, a from-scratch rule generator validated against `mlxtend.association_rules` (11/11 rules matching exactly), manual confidence and lift calculations cross-checked against the library, a deliberately constructed high-confidence-low-lift trap dataset, a `min_support` sweep showing combinatorial growth in frequent itemset count, a full rule-mining pass on a 1000-transaction synthetic dataset with two deliberately embedded associations (one recovered as expected, the other producing a genuinely surprising and honestly-reported higher lift despite being the weaker design), a support/confidence/lift visualization, a top-10-rules-by-lift bar chart, a confidence-vs-lift ranking comparison, and a final Apriori-vs-FP-Growth validation and speed comparison:

➡️ **[07_association_rules_apriori.ipynb](07_association_rules_apriori.ipynb)**

---

## 📚 Further Reading

- [mlxtend: Frequent Itemsets via Apriori](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [mlxtend: Association Rules](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- [Agrawal & Srikant (1994): Fast Algorithms for Mining Association Rules](https://www.vldb.org/conf/1994/P487.PDF)
- [Han et al.: Mining Frequent Patterns without Candidate Generation (FP-Growth)](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf)

---
[← Back to Unsupervised](../README.md)

<!-- page-views-badge -->
<div align="center" style="margin-top: 16px;">

![Page Views](https://visitor-badge.laobi.icu/badge?page_id=mdnuruzzamanKALLOL.ClassicalML.Unsupervised.07_Association_Rules_Apriori&left_color=%23FF6F00&right_color=%230e75b6&left_text=Page%20Views)

</div>
