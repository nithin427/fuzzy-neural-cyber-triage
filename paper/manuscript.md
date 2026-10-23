---
title: "Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage"
---

**Article type:** Short Communication — *Knowledge-Based Systems* (Elsevier)\
**Author:** Sai Nithin Erigineni, DBA, Midwest University, 851 Parr Rd, Wentzville, MO 63385, United States (corresponding author: serigineni@midwest.edu)\
**Code and results:** <https://github.com/nithin427/fuzzy-neural-cyber-triage> (public)

# Abstract

Security operations centres (SOCs) receive risk scores from machine-learning detectors, yet the step from score to action is usually a single hand-set threshold. We treat alert triage as a knowledge-based decision problem: a multilayer perceptron (MLP) estimates attack propensity, fuzzy membership functions map it to Low, Medium and High risk bands, and three production rules assign Monitor, Soft-block or Escalate under an explicit cost matrix. On NSL-KDD (20,000 test alerts), gradient boosting is the stronger detector (ROC-AUC 0.947 versus 0.908), but on the same MLP scores the rule base lowers the mean triage cost to 3.068 (95% CI [3.016, 3.118]) from 3.264 for a 0.5 threshold and 4.309 for top-20% escalation, and raises attack coverage from 0.655 to 0.686 (Wilcoxon $p<10^{-100}$). It is cheaper than both crisp rules in all ten repeated runs ($p=0.002$) and than the threshold at every tested missed-attack cost from 2 to 30. Ablations bound the claim: the gain comes from the Soft-block band, and a Bayes cost rule is cheaper still (2.164). The contribution is an auditable decision layer with a decision-centric evaluation, not a new detector.

**Keywords:** Fuzzy logic; Neural networks; Decision intelligence; Intrusion detection; Alert triage; Cost-sensitive decisions; Knowledge-based systems

# Introduction

Security operations centres (SOCs) centralise the handling of security alerts [1], yet analysts receive more alerts than they can investigate, and alert fatigue is a central operational risk [2,3]. Intrusion-detection research, from classical data mining [4,5] to deep networks on NSL-KDD [6,7], usually stops at a score judged by accuracy or the area under the receiver operating characteristic curve (ROC-AUC). A SOC manager must instead decide, under finite analyst capacity and asymmetric error costs, which alerts to monitor, contain or escalate: a data-driven decision task [8,9].

Fuzzy and neuro-fuzzy systems suit this step because they express learned evidence as linguistic rules that operators can audit [10–13]. Table 1 shows the gap. Fuzzy intrusion detectors use rules to raise accuracy but output a binary alarm [14–16]; cost-sensitive learning [17,18] enters intrusion detection through training losses [19,20], not through the choice of action; explainable AI (XAI) explains scores after the fact [21–23]; and reinforcement-learning prioritisation ranks alerts without assigning actions [24]. Knowledge-based decision support [25] shows the value of explicit knowledge, but none of these studies combines an action-cost model, readable rules and graded responses.

| Study | Approach | Data | Cost | Rules | Graded | Tests |
|:---|:---|:---|:--:|:--:|:--:|:--:|
| *Fuzzy and neuro-fuzzy intrusion detection* | | | | | | |
| Toosi [14] | ANFIS + GA-tuned fuzzy rules | KDD Cup 99 | (✓) | ✓ | × | × |
| Hamamoto [15] | GA flow signature + fuzzy logic | Campus IP flows | × | ✓ | × | n.r. |
| Alalhareth [16] | LSTM, fuzzy-tuned training | WUSTL-EHMS-2020 | (✓) | × | × | × |
| *Deep and cost-sensitive detectors* | | | | | | |
| Vinayakumar [6] | Deep neural network | NSL-KDD and 5 others | × | × | × | × |
| Su [7] | BLSTM + attention (BAT-MC) | NSL-KDD | × | × | × | × |
| Gupta [19] | Cost-sensitive DNN ensemble | NSL-KDD and 2 others | (✓) | × | × | n.r. |
| Telikani [20] | Cost-sensitive AE + SVM | UNSW-NB15, BoT-IoT | (✓) | × | × | n.r. |
| *Explanation, prioritisation and decision support* | | | | | | |
| Wang [22] | SHAP explanations | NSL-KDD | × | (✓) | × | × |
| Alabbadi [23] | CNN/DNN/TabNet + XAI | TON_IoT | × | (✓) | × | n.r. |
| Wang [24] | RL alert prioritisation | 5 multi-step sets | × | × | (✓) | n.r. |
| Bag [25] | Temporal knowledge-graph DSS | 190 firms + 4 sets | × | (✓) | × | ✓ |
| **This work** | **MLP + fuzzy rules $R_1$–$R_3$** | **NSL-KDD, 10 seeds** | ✓ | ✓ | ✓ | ✓ |

: **Table 1.** Related work by design property. Only this work couples an explicit action-cost model, readable rules and graded SOC actions with significance tests; the cited studies end in an alarm, a class label or a ranking. Properties, not accuracies, are compared, because data sets and tasks differ.

*Note.* Cost: actions chosen under an explicit cost model; Rules: operator-readable decision logic; Graded: more than two actions; Tests: significance tests or CIs. ✓ present; (✓) partial (cost in training or binary decision only; post hoc explanation; ranking without actions); × absent; n.r. not reported in the accessible text. AE: autoencoder; (B)LSTM: (bidirectional) long short-term memory; DNN: deep neural network; DSS: decision support system; GA: genetic algorithm; RL: reinforcement learning; SHAP: Shapley additive explanations; SVM: support vector machine.

This Short Communication contributes (i) a knowledge-based layer that maps neural propensity to fuzzy risk bands and then to SOC actions through a stated production-rule base (Section 2); (ii) a decision-centric evaluation with paired tests, repeated runs and sensitivity sweeps (Sections 3 and 4); and (iii) ablations that locate the source and limits of the gain. The layer is detector-agnostic (Section 4).

# Proposed fuzzy–neural decision framework

## Neural propensity model

Each alert $i$ has features $x_i$ (38 numeric NSL-KDD fields, standardised; protocol, service and flag, one-hot encoded) and label $y_i\in\{0,1\}$ (normal, attack). An MLP $g_\theta$ with ReLU layers of width 128 and 64, dropout 0.2 and one logit output gives the attack propensity $$\begin{split}
p_i&=\sigma\big(g_\theta(x_i)\big),\\
\mathcal{L}(\theta)&=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_{+}\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big],\qquad w_{+}=\frac{N_{-}}{N_{+}},
\end{split}
\tag{1}$$ where $N_{-}$ and $N_{+}$ count normal and attack training records. Training uses Adam (learning rate $10^{-3}$, batch size 512, 20 epochs) and keeps the epoch with the best validation ROC-AUC.

## Fuzzy risk bands and production-rule base

Propensity is mapped to three linguistic risk bands [10] by shoulder and triangular membership functions, with $[z]_0^1=\min(1,\max(0,z))$: $$\begin{align}
\mu_L(p)&=\big[(0.45-p)/0.45\big]_0^1, \tag{2}\\
\mu_M(p)&=\begin{cases}\big[(p-0.15)/0.35\big]_0^1, & p<0.5,\\ \big[(0.85-p)/0.35\big]_0^1, & p\ge 0.5,\end{cases} \tag{3}\\
\mu_H(p)&=\big[(p-0.55)/0.45\big]_0^1. \tag{4}
\end{align}$$ The knowledge layer is the production-rule base $\mathcal{R}=\{R_1,R_2,R_3\}$:

| | |
|:-|:-----------------------------------------------------------|
| $R_1$: | IF risk IS Low THEN action $=$ Monitor (log the alert; no intervention); |
| $R_2$: | IF risk IS Medium THEN action $=$ Soft-block (automatic containment, e.g., rate limit); |
| $R_3$: | IF risk IS High THEN action $=$ Escalate (hand over to a human analyst). |

Rule $R_k$ fires with strength $\mu_k(p_i)$; conflicts are resolved by winner-take-all, with ties going to the less severe action: $$a_i = A\big(k_i^{*}\big),\qquad k_i^{*}=\arg\max_{k\in(L,M,H)}\mu_k(p_i),
\tag{5}$$ where $A$ maps $(L,M,H)$ to (Monitor, Soft-block, Escalate). Firing strengths are logged with every decision as its explanation (e.g., “$p=0.60$: $\mu_M=0.71$, $\mu_H=0.11$, Soft-block”). Solving $\mu_L=\mu_M$ and $\mu_M=\mu_H$ shows that Eq. (5) switches action at $p=9/32=0.281$ and $p=23/32=0.719$ (Fig. 1). The breakpoints were set by expert judgement before the test data were examined.

![**Fig. 1.** Fuzzy score-to-action mapping (seed-42 test set). The memberships of Eqs. (2)–(4) and rules $R_1$–$R_3$ switch action at $p=9/32$ and $23/32$ (dotted). The Soft-block band holds only 776 alerts (3.9%) yet yields the entire cost gain over the 0.5 threshold; the Bayes cost rule (dashed) soft-blocks every alert above $p=2/11$.](../figures/fig1_decision_bands.png)

## Cost model and decision metrics

Actions are scored with the relative cost matrix $$\begin{aligned}
C(\text{Monitor},y)&=C_{FN}\,y, & C(\text{Soft-block},y)&=1+(1-y),\\
C(\text{Escalate},y)&=3+3(1-y), & C_{FN}&=10,
\end{aligned}
\tag{6}$$ so a missed attack costs 10 units, containing an attack 1 unit and escalating a benign alert 6 units. Policies are compared by the mean triage cost $\bar C=\frac1N\sum_i C(a_i,y_i)$, the attack coverage $\sum_i \mathbf{1}[a_i\neq\text{Monitor}]\,y_i/\sum_i y_i$ and the number of unnecessary escalations (UE), i.e., escalated benign alerts. As a reference, we also report the plug-in Bayes cost rule $$a_i^{B}=\arg\min_{a}\;\big[(1-p_i)\,C(a,0)+p_i\,C(a,1)\big],
\tag{7}$$ which treats $p_i$ as a calibrated probability, an assumption that deep-model scores do not guarantee [26]. Because $C(\text{Soft-block},y)<C(\text{Escalate},y)$ for both classes, Eq. (7) never escalates and soft-blocks whenever $p_i>2/11$.

# Experimental design

*Data.* We use the NSL-KDD benchmark [27]. For CPU-scale reproducibility, 40,000 training records are sampled and split (stratified) into 34,000 training and 6,000 validation records; the scaler and encoder are fitted on training data only. The test set is capped at 20,000 records (attack rate 0.568) and contains attack types absent from training, so train/test shift is part of the task.

*Baselines.* The detectors are class-balanced logistic regression, gradient boosting (GB) [28] with scikit-learn defaults and the MLP alone; the proposed policy shares the MLP propensity, so any benefit must appear in decisions. Decision baselines act on the same scores: a Monitor/Escalate threshold at 0.5, the same threshold tuned on validation data to minimise cost (threshold moving [18]) and escalation of the top 20% of scores, a fixed analyst-capacity rule. The Bayes cost rule is a reference only; learned policies [24] need interaction data and are excluded.

*Statistics.* On the seed-42 test set, we report percentile 95% confidence intervals (CIs) from 2,000 paired bootstrap resamples of alerts [29] and two-sided Wilcoxon signed-rank tests on per-alert cost differences [30]. The full pipeline is repeated for seeds 42–51 (paired Wilcoxon test across seeds), and two sweeps vary the missed-attack cost $C_{FN}$ from 2 to 30 and the Medium-band half-width from 0.25 to 0.45.

# Results and discussion

| *Panel A. Detector quality (threshold 0.5 for accuracy and F1)* | | | | | | | |
|:---|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| Detector | | Accuracy | F1 | ROC-AUC | PR-AUC | | ROC-AUC, 10 seeds |
| Logistic regression | | 0.752 | 0.740 | 0.818 | 0.879 | | $0.812\pm0.008$ |
| Gradient boosting | | **0.786** | **0.774** | **0.947** | **0.951** | | $\mathbf{0.936\pm0.010}$ |
| MLP (input to proposed policy) | | 0.773 | 0.766 | 0.908 | 0.923 | | $0.900\pm0.007$ |
| *Panel B1. Decision policies on identical MLP scores* | | | | | | | |
| Policy | Scores | $\bar C$ | Coverage | UE | $\Delta\bar C$ | $p$ | $\bar C$, 10 seeds |
| Threshold 0.5 | MLP | 3.264 [3.210, 3.316] | 0.655 [0.646, 0.664] | 623 | $+0.196$ [0.180, 0.213] | $7.3{\times}10^{-124}$ | $3.246\pm0.076$ |
| Top-20% escalation | MLP | 4.309 [4.246, 4.372] | 0.348 [0.339, 0.357] | **47** | $+1.242$ [1.196, 1.285] | $<10^{-300}$ | $4.318\pm0.009$ |
| **Fuzzy rule policy (proposed)** | MLP | **3.068** [3.016, 3.118] | **0.686** [0.677, 0.694] | 603 | – | – | $\mathbf{3.088\pm0.081}$ |
| *Panel B2. Ablations and references* | | | | | | | |
| Crisp bands at $9/32$, $23/32$ | MLP | 3.068 | 0.686 | 603 | 0 | – | – |
| Validation-tuned threshold $t{=}0.53$ | MLP | 3.278 | 0.651 | 617 | $+0.210$ | $2.3{\times}10^{-123}$ | – |
| Fuzzy rule policy | GB | 3.004 | 0.676 | 222 | $-0.064$ | 0.017 | $2.967\pm0.100$ |
| Threshold 0.5 | GB | 3.189 | 0.644 | 241 | $+0.122$ | $6.4{\times}10^{-40}$ | – |
| Bayes cost rule (reference) | MLP | 2.164 | 0.701 | 0 | $-0.903$ | $<10^{-300}$ | $2.207\pm0.100$ |

: **Table 2.** Main results (seed 42; $N=20{,}000$ test alerts). Gradient boosting is the best detector, but on identical MLP scores the fuzzy rule policy has the lowest cost and highest coverage of the deployable policies. Brackets: paired bootstrap 95% CIs; $\Delta\bar C$: difference from the proposed policy; $p$: two-sided Wilcoxon signed-rank test on per-alert costs; 10 seeds: mean $\pm$ SD over seeds 42–51. Bold: best in Panels A and B1.

*Note.* PR-AUC: area under the precision–recall curve; $\bar C$: mean triage cost, Eq. (6); UE: unnecessary escalations. Actions (Monitor/Soft-block/Escalate): threshold 0.5, 11,943/0/8,057; top-20%, 16,000/0/4,000; proposed and crisp bands, 11,554/776/7,670; Bayes rule, 11,369/8,631/0. Panel B2 CIs: `results/table2_decision.csv`.

*Prediction versus decision.* Gradient boosting is the best detector on every metric (Table 2, Panel A) and beats the MLP in ROC-AUC in all ten seeds; we therefore make no detection claim. On identical MLP scores, the rule base lowers the mean triage cost by 0.196 per alert relative to the 0.5 threshold and by 1.242 relative to top-20% escalation, and raises coverage by 0.031 (95% CI [0.028, 0.034]) over the threshold (Panel B1). Rule base and threshold differ only on the 776 Medium-band alerts, which the rule base contains; this saves about 3,900 cost units per 20,000 alerts and sends 387 fewer alerts to analysts. Top-20% escalation minimises unnecessary escalations (47) but covers only 0.348 of attacks, and validation tuning ($t=0.53$; 3.278) does not close the gap because unseen attack types shift the test scores.

![**Fig. 2.** Mean triage cost versus missed-attack cost $C_{FN}$ (seed 42). The fuzzy rule policy is cheaper than the 0.5 threshold at every tested $C_{FN}$ from 2 to 30, with a gap that widens as missed attacks grow costlier; top-20% escalation wins only for $C_{FN}\le3$ (shaded). Error bars: paired bootstrap 95% CIs at the default $C_{FN}=10$ (dotted); dashed: Bayes reference.](../figures/fig2_cost_sensitivity.png)

*Robustness.* The rule base is cheaper than both crisp rules in all ten seeds ($3.088\pm0.081$ versus $3.246\pm0.076$ and $4.318\pm0.009$; Wilcoxon $p=0.002$ for both) and has higher coverage than the 0.5 threshold in every seed. Its advantage over the threshold grows from 0.055 at $C_{FN}=2$ to 0.549 at $C_{FN}=30$ (Fig. 2); top-20% escalation is cheaper only for $C_{FN}\le3$ (1.347 versus 1.640 at $C_{FN}=2$). Widening the Medium half-width from 0.25 to 0.45 enlarges the Soft-block band from 646 to 900 alerts and lowers the cost monotonically from 3.101 to 3.034; every width beats both crisp rules.

*What drives the gain.* Crisp bands at $p=9/32$ and $23/32$ reproduce every action (Panel B2), so the benefit comes from the containment band, not from fuzziness itself; the memberships add a graded explanation per decision. The rule base is not cost-optimal: the Bayes cost rule is 0.903 cheaper per alert because soft-blocking dominates escalation under Eq. (6), so it contains 8,631 alerts (43% of traffic). An oracle test-set threshold of at most 0.10 would also win (minimum 2.929 at $t=0.01$), but validation tuning cannot find it under shift. On GB scores, the same rules cost 3.004 ($p=0.017$ versus MLP scores) and cut unnecessary escalations from 603 to 222, so the layer transfers.

*Implications for SOC decision-makers.* First, detectors should be judged by the decisions they enable: on identical scores, the action rule changed the cost by up to 1.242 per alert. Second, the cost matrix sets the action mix, so organisations must price false containment before automating Soft-block. Third, band boundaries are a reviewable workload lever that needs no retraining. Fourth, named rules with logged firing strengths give each action an audit trail that post hoc attributions [22,23] lack, aiding compliance and human–AI teaming [3].

# Limitations

Fixed bands are sensitive to the score distribution, not only to the ranking: seed 48, the weakest replicate (cost 3.282, coverage 0.626), has a typical ROC-AUC (0.900) but the lowest accuracy at 0.5 (0.757). Adaptive, ANFIS-style tuning [11] or cost-derived bands could close the Bayes gap. NSL-KDD (2009) is a dated laboratory benchmark that misses modern enterprise, cloud and Internet of Things traffic [31,32], and the data were subsampled under one protocol. The cost matrix is illustrative, not elicited from a SOC; coverage counts containment as covering an attack; and there was no deployment, drift analysis or user study.

# Conclusion

As a knowledge-based decision layer, the fuzzy–neural rule base outperformed crisp threshold and capacity rules on NSL-KDD in cost and coverage across seeds, cost ratios and band widths. The gain comes from the containment band, and a Bayes rule derived from the cost matrix is cheaper still. Future work will derive bands from elicited SOC costs and validate them live.

# Data availability

The code, result tables and figure-generation scripts are publicly available at <https://github.com/nithin427/fuzzy-neural-cyber-triage>. NSL-KDD is available from the Canadian Institute for Cybersecurity [27].

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Grok Bot, an AI-assisted writing and coding tool, to help with language editing, code refactoring, reference checking and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. All results, tables and figures were generated by the author's executable code from the public dataset; no figure was created with generative AI.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# CRediT authorship contribution statement

**Sai Nithin Erigineni:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

# References

[1] M. Vielberth, F. Böhm, I. Fichtinger, G. Pernul, Security operations center: A systematic study and open challenges, IEEE Access 8 (2020) 227756–227779. https://doi.org/10.1109/ACCESS.2020.3045514.

[2] S. Tariq, M. Baruwal Chhetri, S. Nepal, C. Paris, Alert fatigue in security operations centres: Research challenges and opportunities, ACM Comput. Surv. 57 (9) (2025) 1–38. https://doi.org/10.1145/3723158.

[3] M. Baruwal Chhetri, S. Tariq, R. Singh, F. Jalalvand, C. Paris, S. Nepal, Towards human–AI teaming to mitigate alert fatigue in security operations centres, ACM Trans. Internet Technol. 24 (3) (2024) 1–22. https://doi.org/10.1145/3670009.

[4] H.-J. Liao, C.-H. Richard Lin, Y.-C. Lin, K.-Y. Tung, Intrusion detection system: A comprehensive review, J. Netw. Comput. Appl. 36 (1) (2013) 16–24. https://doi.org/10.1016/j.jnca.2012.09.004.

[5] A. L. Buczak, E. Guven, A survey of data mining and machine learning methods for cyber security intrusion detection, IEEE Commun. Surv. Tutor. 18 (2) (2016) 1153–1176. https://doi.org/10.1109/COMST.2015.2494502.

[6] R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, S. Venkatraman, Deep learning approach for intelligent intrusion detection system, IEEE Access 7 (2019) 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334.

[7] T. Su, H. Sun, J. Zhu, S. Wang, Y. Li, BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset, IEEE Access 8 (2020) 29575–29585. https://doi.org/10.1109/ACCESS.2020.2972627.

[8] G. Vial, Understanding digital transformation: A review and a research agenda, J. Strateg. Inf. Syst. 28 (2) (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003.

[9] Y. Duan, J. S. Edwards, Y. K. Dwivedi, Artificial intelligence for decision making in the era of Big Data – evolution, challenges and research agenda, Int. J. Inf. Manag. 48 (2019) 63–71. https://doi.org/10.1016/j.ijinfomgt.2019.01.021.

[10] L. A. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.

[11] J.-S. R. Jang, ANFIS: Adaptive-network-based fuzzy inference system, IEEE Trans. Syst. Man Cybern. 23 (3) (1993) 665–685. https://doi.org/10.1109/21.256541.

[12] P. V. de Campos Souza, Fuzzy neural networks and neuro-fuzzy networks: A review the main techniques and applications used in the literature, Appl. Soft Comput. 92 (2020) 106275. https://doi.org/10.1016/j.asoc.2020.106275.

[13] A. Fernández, F. Herrera, O. Cordón, M. J. del Jesus, F. Marcelloni, Evolutionary fuzzy systems for explainable artificial intelligence: Why, when, what for, and where to?, IEEE Comput. Intell. Mag. 14 (1) (2019) 69–81. https://doi.org/10.1109/MCI.2018.2881645.

[14] A. N. Toosi, M. Kahani, A new approach to intrusion detection based on an evolutionary soft computing model using neuro-fuzzy classifiers, Comput. Commun. 30 (10) (2007) 2201–2212. https://doi.org/10.1016/j.comcom.2007.05.002.

[15] A. H. Hamamoto, L. F. Carvalho, L. D. H. Sampaio, T. Abrão, M. L. Proença, Network anomaly detection system using genetic algorithm and fuzzy logic, Expert Syst. Appl. 92 (2018) 390–402. https://doi.org/10.1016/j.eswa.2017.09.013.

[16] M. Alalhareth, S.-C. Hong, An adaptive intrusion detection system in the Internet of Medical Things using fuzzy-based learning, Sensors 23 (22) (2023) 9247. https://doi.org/10.3390/s23229247.

[17] C. Elkan, The foundations of cost-sensitive learning, in: Proc. 17th Int. Jt. Conf. Artif. Intell. (IJCAI), 2001, pp. 973–978.

[18] Z.-H. Zhou, X.-Y. Liu, Training cost-sensitive neural networks with methods addressing the class imbalance problem, IEEE Trans. Knowl. Data Eng. 18 (1) (2006) 63–77. https://doi.org/10.1109/TKDE.2006.17.

[19] N. Gupta, V. Jindal, P. Bedi, CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based intrusion detection systems, Comput. Secur. 112 (2022) 102499. https://doi.org/10.1016/j.cose.2021.102499.

[20] A. Telikani, N. E. Rudbardeh, S. Soleymanpour, A. Shahbahrami, J. Shen, G. Gaydadjiev, R. Hassanpour, A cost-sensitive machine learning model with multitask learning for intrusion detection in IoT, IEEE Trans. Ind. Inform. 20 (3) (2024) 3880–3890. https://doi.org/10.1109/TII.2023.3314208.

[21] A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion 58 (2020) 82–115. https://doi.org/10.1016/j.inffus.2019.12.012.

[22] M. Wang, K. Zheng, Y. Yang, X. Wang, An explainable machine learning framework for intrusion detection systems, IEEE Access 8 (2020) 73127–73141. https://doi.org/10.1109/ACCESS.2020.2988359.

[23] A. Alabbadi, F. Bajaber, An intrusion detection system over the IoT data streams using eXplainable Artificial Intelligence (XAI), Sensors 25 (3) (2025) 847. https://doi.org/10.3390/s25030847.

[24] X. Wang, X. Yang, X. Liang, X. Zhang, W. Zhang, X. Gong, Combating alert fatigue with AlertPro: Context-aware alert prioritization using reinforcement learning for multi-step attack detection, Comput. Secur. 137 (2024) 103583. https://doi.org/10.1016/j.cose.2023.103583.

[25] S. Bag, S. Sarkar, I. Bose, Enhancing cybersecurity risk assessment using temporal knowledge graph-based explainable decision support system, Decis. Support Syst. 198 (2025) 114526. https://doi.org/10.1016/j.dss.2025.114526.

[26] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, S. Nahavandi, A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Inf. Fusion 76 (2021) 243–297. https://doi.org/10.1016/j.inffus.2021.05.008.

[27] M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: Proc. 2009 IEEE Symp. Comput. Intell. Secur. Def. Appl. (CISDA), 2009, pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528.

[28] J. H. Friedman, Greedy function approximation: A gradient boosting machine, Ann. Stat. 29 (5) (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451.

[29] B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, New York, 1994. https://doi.org/10.1201/9780429246593.

[30] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80–83. https://doi.org/10.2307/3001968.

[31] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, A. Hotho, A survey of network-based intrusion detection data sets, Comput. Secur. 86 (2019) 147–167. https://doi.org/10.1016/j.cose.2019.06.005.

[32] M. A. Al-Garadi, A. Mohamed, A. K. Al-Ali, X. Du, I. Ali, M. Guizani, A survey of machine and deep learning methods for Internet of Things (IoT) security, IEEE Commun. Surv. Tutor. 22 (3) (2020) 1646–1685. https://doi.org/10.1109/COMST.2020.2988293.
