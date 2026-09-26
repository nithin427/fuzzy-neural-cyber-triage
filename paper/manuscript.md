# Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage

**Short communication draft targeting *Knowledge-Based Systems***  
**Code & artifacts:** https://github.com/nithin427/fuzzy-neural-cyber-triage (private)

> Author placeholders: replace before submission. Empirical numbers below are from the seeded NSL-KDD run in `results/summary.json` (seed=42).

---

## Abstract

Security operations often score alerts but leave triage actions to ad hoc thresholds. This short communication presents a fuzzy–neural decision intelligence framework that couples a neural propensity model with fuzzy linguistic risk bands and a cost-aware action policy for cyber alert triage. A compact multilayer perceptron estimates attack propensity from NSL-KDD features. Triangular fuzzy membership functions map propensity into low, medium, and high risk. A transparent decision layer converts dominant bands into Monitor, Soft-block, or Escalate actions under an explicit relative cost model. On a reproducible CPU-scale experiment, gradient boosting achieves the strongest ROC-AUC (0.947), while the fuzzy–neural policy reduces mean expected triage cost to 3.069 versus 3.264 for a hard 0.5 threshold and 4.309 for a top-20% escalate rule, with higher attack coverage (0.686). The contribution is knowledge-facing decision support, not a new state-of-the-art detector.

**Keywords:** fuzzy systems; neural networks; decision intelligence; intrusion detection; intelligent decision support; knowledge-based systems

---

## 1. Introduction

Knowledge-based and AI techniques are increasingly used to support prediction and operational decisions. In cyber defence, the operator question is not only “Is this an attack?” but “What should we do next under limited response capacity?”

Black-box classifiers can rank alerts well yet remain hard to operationalize. Pure rule bases are transparent but brittle. Fuzzy–neural hybrids combine learned propensity with linguistic knowledge representation, aligning with *Knowledge-Based Systems* aims: intelligent decision support, computational intelligence, and knowledge presentation.

**Contributions.** (1) A three-layer pipeline: neural propensity, fuzzy risk banding, cost-aware triage. (2) Evaluation that reports decision quality (expected cost, coverage), not only AUC. (3) A public-data, sub-two-hour reproducible protocol (NSL-KDD).

---

## 2. Related work

Intrusion detection with logistic regression, boosting, and neural nets is extensively studied on NSL-KDD and related corpora. Cost-sensitive learning addresses asymmetric false negatives and false positives. Fuzzy and ANFIS-style models provide graded linguistic labels. The gap we target is the closed loop from learned score to fuzzy knowledge bands to explicit triage actions under cost—the decision-intelligence layer many accuracy-centric IDS papers omit.

---

## 3. Method

### 3.1 Problem

For alert features \(x\), output propensity \(p=f_\theta(x)\in[0,1]\), fuzzy grades \(\mu_L,\mu_M,\mu_H\), and action \(a\in\{\mathrm{Monitor},\mathrm{SoftBlock},\mathrm{Escalate}\}\).

### 3.2 Neural propensity

A feed-forward MLP (128–64–1) is trained with BCE and class imbalance weighting. Numeric features are standardized; categorical fields (`protocol_type`, `service`, `flag`) are one-hot encoded.

### 3.3 Fuzzy risk banding

Triangular memberships on \(p\) define Low / Medium / High. Dominant membership selects the linguistic band.

### 3.4 Cost-aware decision layer

Relative costs: monitor free; soft-block and escalate incur intervention cost; missing an attack incurs a high false-negative penalty. Dominant Low→Monitor, Medium→SoftBlock, High→Escalate. Baselines: hard threshold 0.5; escalate top 20% by score.

---

## 4. Experimental setup

**Data.** NSL-KDD (`KDDTrain+`, `KDDTest+`), binary normal vs attack. Train subsampled to 40,000 rows for CPU reproducibility; test capped at 20,000 (seed 42). Attack rate on test ≈ 0.568.

**Baselines.** Logistic regression; gradient boosting; neural ablation (scores only); proposed fuzzy–neural decision system (same neural scores + fuzzy–cost policy).

**Metrics.** Accuracy, F1, ROC-AUC, PR-AUC; mean expected decision cost; attack coverage; unnecessary escalations.

**Compute.** Single CPU Python run (well under two hours; ~2 minutes wall time after dependency install in our environment).

---

## 5. Results

### Table I. Predictive performance (test)

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----|---------|--------|
| Logistic regression | 0.752 | 0.740 | 0.818 | 0.879 |
| Gradient boosting | 0.786 | 0.774 | **0.947** | **0.951** |
| Neural net (ablation) | 0.773 | 0.766 | 0.908 | 0.923 |
| Fuzzy–neural (proposed) | 0.773 | 0.766 | 0.908 | 0.923 |

Proposed prediction metrics match the neural ablation by design: novelty is the decision layer.

### Table II. Decision quality (neural propensity)

| Policy | Mean expected cost | Attack coverage | Unnecessary escalations |
|--------|--------------------|-----------------|-------------------------|
| Prob. threshold 0.5 | 3.264 | 0.655 | 623 |
| Top-20% by score | 4.309 | 0.348 | 47 |
| Fuzzy–cost (proposed) | **3.069** | **0.686** | 603 |

**Fig. 1.** Architecture: features → MLP → fuzzy bands → triage actions (see `figures/fig1_architecture.png`).  
**Fig. 2.** ROC curves and mean expected cost vs threshold (`figures/fig2_roc_and_cost.png`).

**Interpretation.** Boosting leads on AUC. The fuzzy–neural policy improves triage cost and coverage versus simple score policies, and emits linguistic Monitor / Soft / Escalate decisions suitable for SOC knowledge workflows.

---

## 6. Discussion

**KBS fit.** Learning + fuzzy knowledge representation + decision support for prediction/warning systems.

**Limitations.** Public NSL-KDD; relative costs are scenario defaults; train subsampled; boosting still stronger as a pure detector—pair fuzzy policy with stronger propensity models in extensions. Fairness and concept drift are out of scope.

---

## 7. Conclusion

We presented a fuzzy–neural decision intelligence framework for cost-aware cyber alert triage. Reproducible NSL-KDD results show competitive neural propensity and improved decision cost/coverage under fuzzy linguistic actions. Future work: ANFIS co-training, operator-calibrated costs, and live SOC decision logs.

---

## References (core set; expand to 20–25 Q1 sources before submission)

1. L. A. Zadeh, “Fuzzy sets,” *Information and Control*, 1965.  
2. J.-S. R. Jang, “ANFIS,” *IEEE Trans. Syst., Man, Cybern.*, 1993.  
3. C. Elkan, “The foundations of cost-sensitive learning,” *IJCAI*, 2001.  
4. M. Tavallaee et al., “A detailed analysis of the KDD CUP 99 data set,” *IEEE CISDA*, 2009. (NSL-KDD)  
5. T. Chen and C. Guestrin, “XGBoost,” *KDD*, 2016.  
6. I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.

---

## Reproducibility

```bash
git clone https://github.com/nithin427/fuzzy-neural-cyber-triage.git
cd fuzzy-neural-cyber-triage && python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib pandas scikit-learn torch numpy tabulate
# place NSL-KDD KDDTrain+.txt / KDDTest+.txt under data/
python src/experiment.py
```
