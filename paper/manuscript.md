# Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage

**Short communication draft for *Knowledge-Based Systems* (Elsevier, Scopus Q1)**  
**Reproducible artifacts:** https://github.com/nithin427/fuzzy-neural-cyber-triage (private)

**Authors:** Sai Nithin Erigineni, Midwest University, serigineni@midwest.edu  
**Corresponding author:** Sai Nithin Erigineni (serigineni@midwest.edu)

> Metrics below match `results/summary.json` (seed = 42). Do not invent additional numbers.

---

## Abstract

Security operations centres often receive strong alert scores from machine learning models, but the step from score to operational action is still handled with brittle thresholds. This short communication proposes a fuzzy–neural decision intelligence framework that treats cyber alert triage as a knowledge-based decision problem rather than a classification endpoint. A compact multilayer perceptron estimates attack propensity from NSL-KDD features. Triangular fuzzy membership functions convert propensity into linguistic Low / Medium / High risk bands. A cost-aware decision layer maps those bands to Monitor, Soft-block, or Escalate actions under an explicit relative cost model. In a reproducible CPU-scale experiment, gradient boosting attains the highest ROC-AUC (0.947). The fuzzy–neural policy, using the neural propensity scores, reduces mean expected triage cost to 3.069 versus 3.264 for a hard 0.5 threshold and 4.309 for a top-20% escalate rule, while raising attack coverage to 0.686. The contribution is decision-facing knowledge representation and cost-aware actioning, not a new state-of-the-art detector. That framing matches *Knowledge-Based Systems* aims for intelligent decision support, computational intelligence, and knowledge presentation in prediction and warning systems.

**Keywords:** fuzzy systems; neural networks; decision intelligence; intrusion detection; intelligent decision support systems; knowledge-based systems; cost-sensitive triage

---

## 1. Introduction

Artificial intelligence and knowledge-based methods already support prediction in security operations. For intrusion alerts, analysts still need a second answer beyond accuracy: which events to monitor, which to soft-contain, and which to escalate when capacity and false-alarm cost are limited.

Two incomplete patterns dominate the literature. Accuracy-centric detectors optimise ROC-AUC or F1 and stop at a score or label. Pure expert rule bases are transparent but brittle when traffic shifts. Fuzzy–neural methods sit between these extremes: they learn from data while exposing linguistic grades operators can inspect. That mix matches the scope of *Knowledge-Based Systems*, which seeks systems that support prediction and decision-making with both theory and practical study.

**Research gap.** Prior fuzzy and neural IDS work mainly lifts detection rates. Fewer studies close the loop from learned propensity to linguistic knowledge bands to cost-aware triage actions, and then evaluate that loop with decision metrics such as expected cost and coverage rather than classifier metrics alone.

**Contributions.**
1. A three-layer fuzzy–neural decision pipeline for cyber alert triage (propensity → fuzzy bands → cost-aware actions).
2. An evaluation protocol that reports decision quality alongside standard predictive metrics on public NSL-KDD data.
3. A fully reproducible, short-cycle experiment (CPU, public data, open code) suitable for a short communication.

---

## 2. Related work and positioning

**Machine learning for intrusion detection.** Classical and modern learners on NSL-KDD and related corpora are extensively surveyed and benchmarked [4,7,8]. Gradient boosting and deep models often lead on AUC; operational triage policy is usually external to the model.

**Fuzzy and neuro-fuzzy IDS.** Adaptive neuro-fuzzy inference systems (ANFIS) and fuzzy–neural hybrids improve detection and, in some designs, interpretability [2,9–12]. These works demonstrate the value of fuzzy knowledge representation for intrusion patterns, yet typically report detection rate / false alarm rate rather than cost-aware action policies.

**Cost-sensitive and decision-support security analytics.** Cost-sensitive learning formalises asymmetric false negatives and false positives [3]. Security analytics and SOC tooling increasingly emphasise prioritisation and decision support [13,18]. Explainability methods such as SHAP can complement black-box scores for analyst trust [14], but explanation alone does not select Monitor / Soft-block / Escalate under budget.

**Positioning.** We do not claim a new highest AUC on NSL-KDD. We claim a compact knowledge-based decision layer that turns propensity into linguistic risk and cost-aware actions, scored with decision metrics that accuracy-only IDS papers usually omit.

---

## 3. Methodology

### 3.1 Problem formulation

Let each alert \(i\) have features \(x_i \in \mathbb{R}^d\) and binary label \(y_i \in \{0,1\}\) (normal vs attack). The system outputs:
- propensity \(p_i = f_\theta(x_i) \in [0,1]\),
- fuzzy memberships \(\mu_L(p_i),\ \mu_M(p_i),\ \mu_H(p_i)\),
- action \(a_i \in \{\mathrm{Monitor},\ \mathrm{SoftBlock},\ \mathrm{Escalate}\}\).

### 3.2 Neural propensity model

A multilayer perceptron with hidden widths 128 and 64, ReLU activations, dropout, and a sigmoid output is trained with class-weighted binary cross-entropy. Numeric features are standardised; `protocol_type`, `service`, and `flag` are one-hot encoded. Early model selection uses validation ROC-AUC.

### 3.3 Fuzzy risk banding (knowledge layer)

Triangular membership functions on \(p\) define linguistic Low, Medium, and High risk. The dominant membership supplies an inspectable risk label without discarding the continuous score. Optional secondary antecedents (for example connection duration) can refine rules such as: IF propensity is High AND duration is Short THEN escalate priority is Critical.

### 3.4 Cost-aware decision layer

Relative costs encode asymmetric operational penalties: missed attacks are expensive; unnecessary escalations waste analyst time; soft-blocks sit between monitor and escalate. Dominant Low → Monitor; Medium → SoftBlock; High → Escalate. Comparison policies: hard threshold 0.5 on \(p\); escalate the top 20% of scores.

---

## 4. Experimental setup

**Dataset.** NSL-KDD train/test splits [4], binary normal vs attack. For short-cycle reproducibility we subsample training to 40{,}000 rows and cap the test set at 20{,}000 (seed 42). Observed test attack rate ≈ 0.568.

**Baselines.** (i) Logistic regression; (ii) gradient boosting; (iii) neural net without fuzzy–decision layer (ablation); (iv) proposed fuzzy–neural decision system (neural scores + fuzzy–cost policy).

**Metrics.** Predictive: Accuracy, F1, ROC-AUC, PR-AUC. Decision: mean expected cost, attack coverage (fraction of attacks receiving SoftBlock or Escalate), unnecessary escalations.

**Environment.** Python, scikit-learn, PyTorch; CPU-only; wall time well under two hours (about two minutes after dependency install in our run).

---

## 5. Results

### Table 1. Predictive performance on NSL-KDD test sample

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|-------|----------|-----|---------|--------|
| Logistic regression | 0.752 | 0.740 | 0.818 | 0.879 |
| Gradient boosting | 0.786 | 0.774 | **0.947** | **0.951** |
| Neural net (ablation) | 0.773 | 0.766 | 0.908 | 0.923 |
| Fuzzy–neural (proposed) | 0.773 | 0.766 | 0.908 | 0.923 |

Prediction metrics of the proposed system match the neural ablation by construction. The novelty is the decision layer.

### Table 2. Decision quality using neural propensity scores

| Policy | Mean expected cost | Attack coverage | Unnecessary escalations |
|--------|--------------------|-----------------|-------------------------|
| Probability threshold 0.5 | 3.264 | 0.655 | 623 |
| Top-20% by score | 4.309 | 0.348 | 47 |
| Fuzzy–cost (proposed) | **3.069** | **0.686** | 603 |

**Figure 1.** System architecture: alert features → MLP propensity → fuzzy L/M/H bands → cost-aware triage actions (`figures/fig1_architecture.png`).

**Figure 2.** ROC comparison of propensity models and mean expected decision cost versus hard threshold, with fuzzy–neural and top-20% policies marked (`figures/fig2_roc_and_cost.png`).

**Claims restrained to evidence.** Boosting is the strongest pure detector in this run. The fuzzy–neural policy improves mean triage cost and attack coverage relative to the two simple score-to-action baselines and yields linguistic actions suitable for SOC knowledge workflows.

---

## 6. Discussion

**Alignment with journal aims.** The work joins data-driven learning, fuzzy knowledge presentation, and decision-support outputs for prediction and warning tasks in cybersecurity operations.

**What is and is not claimed.** We claim a decision-intelligence contribution. We do not claim state-of-the-art NSL-KDD detection. Pairing the fuzzy–cost layer with a stronger propensity model, for example boosting scores, is a natural extension that keeps the knowledge-based action narrative.

**Limitations.** Public benchmark traffic; scenario-default costs; training subsample for speed; no live SOC A/B test; fairness and concept drift left for future work.

**Ethics.** Triage automation should remain analyst-supervised; cost weights must be set by the operating organisation, not frozen from this paper.

---

## 7. Conclusion

This short communication presented a fuzzy–neural decision intelligence framework for cost-aware cyber alert triage. Neural propensity, fuzzy linguistic risk bands, and a cost-aware action policy jointly produce Monitor / Soft-block / Escalate decisions that improve expected triage cost and attack coverage over simple threshold and top-\(k\) policies on a reproducible NSL-KDD experiment. Future work includes ANFIS-style adaptive membership learning, operator-calibrated costs, stronger propensity backbones under the same decision layer, and phishing URL evaluation for cross-threat validation.

---

## References

[1] L. A. Zadeh, “Fuzzy sets,” *Information and Control*, vol. 8, no. 3, pp. 338–353, 1965. doi: 10.1016/S0019-9958(65)90241-X

[2] J.-S. R. Jang, “ANFIS: Adaptive-network-based fuzzy inference system,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 23, no. 3, pp. 665–685, 1993. doi: 10.1109/21.256541

[3] C. Elkan, “The foundations of cost-sensitive learning,” in *Proc. 17th Int. Joint Conf. Artificial Intelligence (IJCAI)*, 2001, pp. 973–978.

[4] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 data set,” in *Proc. IEEE Symp. Computational Intelligence for Security and Defense Applications (CISDA)*, 2009, pp. 1–6. doi: 10.1109/CISDA.2009.5356528

[5] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA, USA: MIT Press, 2016.

[6] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794. doi: 10.1145/2939672.2939785

[7] A. L. Buczak and E. Guven, “A survey of data mining and machine learning methods for cyber security intrusion detection,” *IEEE Communications Surveys & Tutorials*, vol. 18, no. 2, pp. 1153–1176, 2016. doi: 10.1109/COMST.2015.2494502

[8] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, “A survey of network-based intrusion detection data sets,” *Computers & Security*, vol. 86, pp. 147–167, 2019. doi: 10.1016/j.cose.2019.06.005

[9] D. Srilatha and G. K. Shyam, “Cloud-based intrusion detection using kernel fuzzy clustering and optimal type-2 fuzzy neural network,” *Cluster Computing*, vol. 24, pp. 2657–2672, 2021. doi: 10.1007/s10586-021-03281-9

[10] J. Liu, W. Yinchai, T. C. Siong, X. Li, L. Zhao, and F. Wei, “On the combination of adaptive neuro-fuzzy inference system and deep residual network for improving detection rates on intrusion detection,” *PLOS ONE*, vol. 17, no. 12, Art. no. e0278819, 2022. doi: 10.1371/journal.pone.0278819

[11] J. Liu, W. Yinchai, T. C. Siong, X. Li, L. Zhao, and F. Wei, “A hybrid interpretable deep structure based on adaptive neuro-fuzzy inference system, decision tree, and K-means for intrusion detection,” *Scientific Reports*, vol. 12, Art. no. 20770, 2022. doi: 10.1038/s41598-022-23765-x

[12] Y. Yang, K. Zheng, C. Wu, and Y. Yang, “BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset,” *IEEE Access*, vol. 8, pp. 29575–29585, 2020. doi: 10.1109/ACCESS.2020.2972627

[13] N. Sun, J. Zhang, P. Rimba, S. Gao, L. Y. Zhang, and Y. Xiang, “Data-driven cybersecurity incident prediction: A survey,” *IEEE Communications Surveys & Tutorials*, vol. 21, no. 2, pp. 1744–1772, 2019. doi: 10.1109/COMST.2018.2876973

[14] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[15] C. Modi, D. Patel, B. Borisaniya, H. Patel, A. Patel, and M. Rajarajan, “A survey of intrusion detection techniques in cloud,” *Journal of Network and Computer Applications*, vol. 36, no. 1, pp. 42–57, 2013. doi: 10.1016/j.jnca.2012.05.003

[16] H.-J. Liao, C.-H. R. Lin, Y.-C. Lin, and K.-Y. Tung, “Intrusion detection system: A comprehensive review,” *Journal of Network and Computer Applications*, vol. 36, no. 1, pp. 16–24, 2013. doi: 10.1016/j.jnca.2012.09.004

[17] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, “Survey of intrusion detection systems: techniques, datasets and challenges,” *Cybersecurity*, vol. 2, Art. no. 20, 2019. doi: 10.1186/s42400-019-0038-7

[18] M. A. Al-Garadi, A. Mohamed, A. K. Al-Ali, X. Du, I. Ali, and M. Guizani, “A survey of machine and deep learning methods for Internet of Things (IoT) security,” *IEEE Communications Surveys & Tutorials*, vol. 22, no. 3, pp. 1646–1685, 2020. doi: 10.1109/COMST.2020.2988293

[19] A. Pinto, L.-C. Herrera, Y. Donoso, and J. A. Gutierrez, “Survey on intrusion detection systems based on machine learning techniques for the protection of critical infrastructure,” *Sensors*, vol. 23, no. 5, Art. no. 2415, 2023. doi: 10.3390/s23052415

[20] H. Liu and B. Lang, “Machine learning and deep learning methods for intrusion detection systems: A survey,” *Applied Sciences*, vol. 9, no. 20, Art. no. 4396, 2019. doi: 10.3390/app9204396

[21] Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, “Kitsune: An ensemble of autoencoders for online network intrusion detection,” in *Proc. Network and Distributed System Security Symp. (NDSS)*, 2018. doi: 10.14722/ndss.2018.23204

[22] S. Moro, P. Cortez, and P. Rita, “A data-driven approach to predict the success of bank telemarketing,” *Decision Support Systems*, vol. 62, pp. 22–31, 2014. doi: 10.1016/j.dss.2014.03.001

[23] S. N. Erigineni, “Fuzzy-neural cyber triage experiment (code and figures),” GitHub repository, 2026. [Online]. Available: https://github.com/nithin427/fuzzy-neural-cyber-triage


## Reproducibility

```bash
git clone https://github.com/nithin427/fuzzy-neural-cyber-triage.git
cd fuzzy-neural-cyber-triage
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib pandas scikit-learn torch numpy tabulate
# Download NSL-KDD KDDTrain+.txt and KDDTest+.txt into data/
python src/experiment.py
```
