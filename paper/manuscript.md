---
title: "Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage"
---

**Article type:** Short Communication — *Knowledge-Based Systems* (Elsevier)\
**Author:** Sai Nithin Erigineni, DBA, Midwest University, 851 Parr Rd, Wentzville, MO 63385, United States (corresponding author: serigineni@midwest.edu)\
**Code and results:** <https://github.com/nithin427/fuzzy-neural-cyber-triage> (public)

# Abstract

Security operations centres (SOCs) receive risk scores from machine-learning detectors, yet the step from score to action is usually a single hand-set threshold. This Short Communication treats cyber alert triage as a knowledge-based decision problem. A multilayer perceptron estimates attack propensity; triangular fuzzy membership functions map it to Low, Medium and High risk bands; and a three-rule production base assigns Monitor, Soft-block or Escalate, scored under an explicit relative cost matrix. On the NSL-KDD benchmark (seed 42; 20,000 test alerts), gradient boosting remains the strongest detector (ROC-AUC 0.947 versus 0.908 for the neural model). The fuzzy rule layer lowers the mean triage cost to 3.068 [95% bootstrap CI 3.016–3.118], against 3.264 for a 0.5 threshold and 4.309 for a top-20% escalation rule, and raises attack coverage from 0.655 to 0.686 (paired Wilcoxon $p<10^{-100}$). It is cheaper than both crisp rules in all ten repeated runs ($p=0.002$) and at every missed-attack cost from 2 to 30, except that top-20% escalation is cheaper when missed attacks cost 2–3 units. Ablations bound the claim. The gain comes from the Soft-block band, which is decision-equivalent to crisp cut-points at $p=9/32$ and $23/32$, and a Bayes rule that uses the cost matrix directly is cheaper still (2.164). The contribution is therefore an interpretable, auditable decision layer and a decision-centric evaluation protocol, not a new detector.

**Keywords:** Fuzzy logic; Neural networks; Decision intelligence; Intrusion detection; Alert triage; Cost-sensitive decisions; Knowledge-based systems

# Introduction

Security operations centres (SOCs) centralise the monitoring and handling of security alerts [1], yet analysts face far more alerts than they can investigate, and alert fatigue is a central operational risk [2,3]. Recent work prioritises alerts by reinforcement learning or human–AI teaming [3,4], but most intrusion-detection research, from surveys [5–7] to deep detectors evaluated on NSL-KDD [8,9], stops at a score judged by accuracy or ROC-AUC. A SOC manager must instead decide, under finite analyst capacity and asymmetric error costs, which alerts to monitor, contain or escalate, a step in the AI-supported, data-driven transformation of decision-making [10,11].

Fuzzy and neuro-fuzzy systems suit this step because they express learned evidence in linguistic terms that operators can audit [12–15]. In intrusion detection, however, fuzzy and neuro-fuzzy hybrids mainly target detection accuracy [16–18], cost-sensitive learning targets the classifier or its binary threshold rather than a graded action [19–22], and explainable AI explains scores without choosing actions [23–26]. Knowledge-based cyber-risk decision support, such as explainable knowledge-graph risk assessment [27], shows the value of explicit knowledge representations, but a score-to-action rule base for alert triage, evaluated with decision metrics, is still missing.

This Short Communication contributes (i) a knowledge-based layer that maps neural propensity to fuzzy risk bands and then to SOC actions through a stated production-rule base (Section 2); (ii) a decision-centric evaluation with paired tests, re-runs and sensitivity sweeps (Sections 3–4); and (iii) ablations that locate the gain. Unlike gradient boosting [28], the strongest baseline, whose output still needs a threshold, the layer is detector-agnostic.

# Proposed fuzzy–neural decision framework

![**Fig. 1.** Proposed framework. (a) Pipeline from alert features to neural propensity, fuzzy risk bands and rules $R_1$–$R_3$. (b) Membership functions, Eqs. (2)–(4); dotted lines mark the switch points $p=9/32$ and $23/32$ between Monitor, Soft-block and Escalate.](../figures/fig1_architecture.png)

## Neural propensity model

Each alert $i$ has features $x_i$ (38 numeric NSL-KDD fields standardised; protocol, service and flag one-hot encoded) and label $y_i\in\{0,1\}$ (normal, attack). A multilayer perceptron $g_\theta$ with ReLU layers of width 128 and 64, dropout 0.2 and one logit output, as in deep intrusion detectors [8], gives the attack propensity $$\begin{split}
p_i&=\sigma\big(g_\theta(x_i)\big),\\
\mathcal{L}(\theta)&=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_{+}\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big],\qquad w_{+}=\frac{N_{-}}{N_{+}},
\end{split}
\tag{1}$$ trained with Adam (learning rate $10^{-3}$, batch 512, 20 epochs), keeping the epoch with the best validation ROC-AUC.

## Fuzzy risk bands and production-rule base

Propensity is mapped to three linguistic risk bands [12] with shoulder and triangular membership functions, using $[z]_0^1=\min(1,\max(0,z))$: $$\begin{align}
\mu_L(p)&=\big[(0.45-p)/0.45\big]_0^1, \tag{2}\\
\mu_M(p)&=\begin{cases}\big[(p-0.15)/0.35\big]_0^1, & p<0.5,\\ \big[(0.85-p)/0.35\big]_0^1, & p\ge 0.5,\end{cases} \tag{3}\\
\mu_H(p)&=\big[(p-0.55)/0.45\big]_0^1. \tag{4}
\end{align}$$ The knowledge layer is the production-rule base $\mathcal{R}=\{R_1,R_2,R_3\}$:

| | |
|:-|:-----------------------------------------------------------|
| $R_1$: | IF risk IS Low THEN action $=$ Monitor (log the alert; no intervention); |
| $R_2$: | IF risk IS Medium THEN action $=$ Soft-block (automatic containment, e.g., rate-limiting); |
| $R_3$: | IF risk IS High THEN action $=$ Escalate (hand over to a human analyst). |

Each rule fires with strength $\mu_k(p_i)$, and conflicts are resolved by winner-take-all, with ties going to the less severe action: $$a_i = A\big(k_i^{*}\big),\qquad k_i^{*}=\arg\max_{k\in(L,M,H)}\mu_k(p_i),
\tag{5}$$ where $A$ maps $(L,M,H)$ to (Monitor, Soft-block, Escalate). Firing strengths are logged with every decision as its explanation (e.g., “$p=0.60$: $\mu_M=0.71$, $\mu_H=0.11$, Soft-block”). Solving $\mu_L=\mu_M$ and $\mu_M=\mu_H$ shows that Eq. (5) switches action at $p=9/32=0.281$ and $p=23/32=0.719$ (Fig. 1b).

## Cost model and decision metrics

Actions are scored with the relative cost matrix $$\begin{aligned}
C(\text{Monitor},y)&=C_{FN}\,y, & C(\text{Soft-block},y)&=1+(1-y),\\
C(\text{Escalate},y)&=3+3(1-y), & C_{FN}&=10.
\end{aligned}
\tag{6}$$ Policies are compared by the mean triage cost $\bar C=\frac1N\sum_i C(a_i,y_i)$, the attack coverage $\sum_i \mathbf{1}[a_i\neq\text{Monitor}]\,y_i/\sum_i y_i$, and unnecessary escalations (escalated benign alerts). As a reference we also report the plug-in Bayes rule $$a_i^{B}=\arg\min_{a}\;\big[(1-p_i)\,C(a,0)+p_i\,C(a,1)\big],
\tag{7}$$ which treats $p_i$ as calibrated, an assumption that deep-model scores do not guarantee [29]. Because $C(\text{Soft-block},y)<C(\text{Escalate},y)$ for both classes, Eq. (7) never escalates and soft-blocks whenever $p_i>2/11$.

# Experimental design

*Data.* We use the NSL-KDD benchmark [30]. For CPU-scale reproducibility, 40,000 training records are sampled and split 85/15 (stratified) into 34,000 for training and 6,000 for validation, with the scaler and one-hot encoder fitted on the training split only; the test set is capped at 20,000 records (attack rate 0.568). The test file contains attack types absent from training, so train/test shift is part of the task.

*Baselines.* The detectors are class-balanced logistic regression, gradient boosting [28] (scikit-learn defaults) and the MLP alone (neural ablation); the proposed system shares the MLP propensity, so its contribution must show in decisions. Decision baselines on the same scores are a Monitor/Escalate threshold at 0.5, the same threshold tuned on validation data to minimise cost (threshold moving [20]), and escalation of the top 20% of scores (a fixed analyst-capacity rule). The Bayes rule, Eq. (7), is a reference, not a competitor, as it needs the cost matrix and calibrated probabilities at run time. Learned policies (e.g., reinforcement learning [4]) need interaction data and are excluded.

*Statistics.* On the seed-42 test set we report percentile 95% confidence intervals from 2,000 paired bootstrap resamples of alerts [31] and two-sided Wilcoxon signed-rank tests on per-alert cost differences [32]. The full pipeline is repeated for seeds 42–51 (paired Wilcoxon test across seeds), and $C_{FN}\in[2,30]$ and the Medium-band half-width (0.25–0.45) are swept.

# Results and discussion

| Model | Acc. | F1 | ROC-AUC | PR-AUC | AUC (10 seeds) |
|:---|:--:|:--:|:--:|:--:|:--:|
| Logistic regression | 0.752 | 0.740 | 0.818 | 0.879 | $0.812\pm0.008$ |
| Gradient boosting | **0.786** | **0.774** | **0.947** | **0.951** | $\mathbf{0.936\pm0.010}$ |
| Neural net (ablation) | 0.773 | 0.766 | 0.908 | 0.923 | $0.900\pm0.007$ |
| Fuzzy–neural (proposed) | 0.773 | 0.766 | 0.908 | 0.923 | $0.900\pm0.007$ |

: **Table 1.** Predictive performance on the NSL-KDD test sample (seed 42; last column: mean $\pm$ SD of ROC-AUC over seeds 42–51). The proposed system uses the neural propensity, so its predictive metrics are identical by construction.

| Policy | Scores | $\bar C$ | Coverage | UE | $\Delta\bar C$ | $p$ | 10 seeds |
|:---|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| Threshold 0.5 | NN | 3.264 [3.210, 3.316] | 0.655 [0.646, 0.664] | 623 | $+0.196$ [0.180, 0.213] | $7.3{\times}10^{-124}$ | $3.246\pm0.076$ |
| Top-20% escalate | NN | 4.309 [4.246, 4.372] | 0.348 [0.339, 0.357] | **47** | $+1.242$ [1.196, 1.285] | $<10^{-300}$ | $4.318\pm0.009$ |
| **Fuzzy (proposed)** | NN | **3.068** [3.016, 3.118] | **0.686** [0.677, 0.694] | 603 | – | – | $\mathbf{3.088\pm0.081}$ |
| *Ablation and reference policies* | | | | | | | |
| Crisp bands$^{c}$ | NN | 3.068 | 0.686 | 603 | 0 | – | – |
| Tuned thr. $t{=}0.53$ | NN | 3.278 [3.224, 3.330] | 0.651 [0.641, 0.659] | 617 | $+0.210$ [0.194, 0.228] | $2.3{\times}10^{-123}$ | – |
| Fuzzy | GB | 3.004 [2.952, 3.055] | 0.676 [0.667, 0.684] | 222 | $-0.064$ [$-0.091$, $-0.037$] | 0.017 | $2.967\pm0.100$ |
| Threshold 0.5 | GB | 3.189 [3.137, 3.241] | 0.644 [0.635, 0.654] | 241 | $+0.122$ [0.093, 0.149] | $6.4{\times}10^{-40}$ | – |
| Bayes rule$^{b}$ | NN | 2.164 [2.113, 2.216] | 0.701 [0.692, 0.709] | 0 | $-0.903$ [$-0.921$, $-0.885$] | $<10^{-300}$ | $2.207\pm0.100$ |

: **Table 2.** Decision quality on the seed-42 test sample ($N=20{,}000$). Brackets: paired bootstrap 95% CIs; $\Delta\bar C$: mean cost relative to the proposed policy (positive = alternative more expensive); $p$: two-sided Wilcoxon signed-rank test on per-alert costs; last column: mean $\pm$ SD of $\bar C$ over seeds 42–51. UE = unnecessary escalations; NN/GB = neural/gradient-boosting propensity. $^{b}$Reference only: soft-blocks every alert with $p>2/11$ (8,631 of 20,000) and never escalates. $^{c}$Crisp cut-points at $p=9/32$ and $23/32$; actions identical to the proposed policy.

![**Fig. 2.** Predictive and decision evidence. (a) ROC curves (seed 42). (b) Mean triage cost of a Monitor/Escalate threshold policy versus threshold; horizontal lines: proposed, top-20% and Bayes-rule costs. (c) Sensitivity to the missed-attack cost $C_{FN}$ (default 10, dotted). (d) Paired per-seed costs for seeds 42–51; black markers: mean $\pm$ SD.](../figures/fig2_roc_and_cost.png)

*Prediction versus decision.* Gradient boosting is the best detector (Table 1, Fig. 2a), with a higher ROC-AUC than the MLP in all ten seeds; we make no detection claim. On identical neural scores, however, the rule base lowers the mean triage cost by 0.196 units per alert versus the 0.5 threshold and by 1.242 versus top-20% escalation (Table 2), and raises attack coverage by 0.031 (CI 0.028–0.034). The policies differ only on the 776 Medium-band alerts; a cost-tuned threshold ($t=0.53$) does no better. It beats both crisp rules in all ten re-runs (Fig. 2d; Wilcoxon $p=0.002$ for both), with higher coverage than the 0.5 threshold in every seed. At $C_{FN}=10$ the saving is about 3,900 cost units per 20,000 alerts, with fewer analyst hand-offs.

*Sensitivity.* The advantage over the 0.5 threshold holds for every $C_{FN}$ from 2 to 30 (Fig. 2c), but top-20% escalation is cheaper for $C_{FN}\le3$ ($C_{FN}=2$: 1.347 versus 1.640; $C_{FN}=3$: 1.718 versus 1.818). Widening the Medium half-width from 0.25 to 0.45 moves the cost monotonically from 3.101 to 3.034 and coverage from 0.680 to 0.691, always beating both thresholds.

*What drives the gain (ablation).* Crisp bands at $9/32$ and $23/32$ reproduce every action, so the benefit comes from the Soft-block band, not from fuzziness; the memberships add graded explanations and a principled way to set bands. The Bayes rule is much cheaper (Table 2) because soft-blocking dominates escalation under Eq. (6): it contains 43% of all traffic and needs no analyst. Managerially, with containment this cheap, the cost matrix, not the detector, sets the action mix, so organisations must calibrate $C$ (including disruption from false containment) before automating. A test-set threshold $\le0.10$ would also beat the rule base (Fig. 2b; minimum 2.929 at $t=0.01$); validation tuning cannot find it under train/test shift, but band positions matter.

*Detector-agnostic use and explainability.* On gradient-boosting scores the unchanged rules lower the cost further and cut unnecessary escalations from 603 to 222 (Table 2), so the layer complements a stronger detector. Named rules and firing strengths, not post-hoc attribution [24,25], explain each action, supporting audit trails and human–AI teaming [3].

# Limitations

*Failure modes.* Besides low $C_{FN}$, seed 48 is the weakest replicate (cost 3.282, coverage 0.626): its MLP has a typical ROC-AUC (0.900) but the lowest accuracy at 0.5 (0.757), so fixed bands are sensitive to the score distribution, not only to ranking. The rule base is also not cost-optimal under its own matrix (the Bayes gap); adaptive (e.g., ANFIS-style [13]) or cost-derived bands are the next step. The bands were set by expert judgement, not on test data.

*Data and scope.* NSL-KDD (2009) is a dated laboratory benchmark that does not reflect modern enterprise, cloud or IoT traffic [33,34]. The data were subsampled, under a single train/test protocol. The cost matrix is illustrative, not elicited from a SOC; coverage counts containment as covering an attack; and there was no live deployment, drift analysis or user study.

# Conclusion

Treating alert triage as a knowledge-based decision problem, a fuzzy–neural rule layer beat crisp threshold and capacity rules on NSL-KDD in cost and coverage; the gain comes from the containment band, and a cost-derived Bayes rule is cheaper still. SOC managers should judge detectors by the decisions they enable and make rules and cost assumptions explicit. Future work will elicit practitioner costs and use recent, live alert data.

# Data availability

The code, result tables and figure-generation scripts are publicly available at <https://github.com/nithin427/fuzzy-neural-cyber-triage>. The NSL-KDD dataset is publicly available from the Canadian Institute for Cybersecurity, University of New Brunswick [30].

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Grok Bot, an AI-assisted writing and coding tool, to help with language editing, code refactoring, reference checking and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. All results, tables and figures were generated by the author’s executable code from the public dataset; no figure was created with generative AI.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# CRediT authorship contribution statement

**Sai Nithin Erigineni:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

# References

[1] M. Vielberth, F. Böhm, I. Fichtinger, G. Pernul, Security operations center: A systematic study and open challenges, IEEE Access 8 (2020) 227756–227779. https://doi.org/10.1109/ACCESS.2020.3045514.

[2] S. Tariq, M. Baruwal Chhetri, S. Nepal, C. Paris, Alert fatigue in security operations centres: Research challenges and opportunities, ACM Comput. Surv. 57 (9) (2025) 1–38. https://doi.org/10.1145/3723158.

[3] M. Baruwal Chhetri, S. Tariq, R. Singh, F. Jalalvand, C. Paris, S. Nepal, Towards human–AI teaming to mitigate alert fatigue in security operations centres, ACM Trans. Internet Technol. 24 (3) (2024) 1–22. https://doi.org/10.1145/3670009.

[4] X. Wang, X. Yang, X. Liang, X. Zhang, W. Zhang, X. Gong, Combating alert fatigue with AlertPro: Context-aware alert prioritization using reinforcement learning for multi-step attack detection, Comput. Secur. 137 (2024) 103583. https://doi.org/10.1016/j.cose.2023.103583.

[5] H.-J. Liao, C.-H. Richard Lin, Y.-C. Lin, K.-Y. Tung, Intrusion detection system: A comprehensive review, J. Netw. Comput. Appl. 36 (1) (2013) 16–24. https://doi.org/10.1016/j.jnca.2012.09.004.

[6] A. L. Buczak, E. Guven, A survey of data mining and machine learning methods for cyber security intrusion detection, IEEE Commun. Surv. Tutor. 18 (2) (2016) 1153–1176. https://doi.org/10.1109/COMST.2015.2494502.

[7] N. Sun, J. Zhang, P. Rimba, S. Gao, L. Y. Zhang, Y. Xiang, Data-driven cybersecurity incident prediction: A survey, IEEE Commun. Surv. Tutor. 21 (2) (2019) 1744–1772. https://doi.org/10.1109/COMST.2018.2885561.

[8] R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, S. Venkatraman, Deep learning approach for intelligent intrusion detection system, IEEE Access 7 (2019) 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334.

[9] T. Su, H. Sun, J. Zhu, S. Wang, Y. Li, BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset, IEEE Access 8 (2020) 29575–29585. https://doi.org/10.1109/ACCESS.2020.2972627.

[10] G. Vial, Understanding digital transformation: A review and a research agenda, J. Strateg. Inf. Syst. 28 (2) (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003.

[11] Y. Duan, J. S. Edwards, Y. K. Dwivedi, Artificial intelligence for decision making in the era of Big Data – evolution, challenges and research agenda, Int. J. Inf. Manag. 48 (2019) 63–71. https://doi.org/10.1016/j.ijinfomgt.2019.01.021.

[12] L. A. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.

[13] J.-S. R. Jang, ANFIS: Adaptive-network-based fuzzy inference system, IEEE Trans. Syst. Man Cybern. 23 (3) (1993) 665–685. https://doi.org/10.1109/21.256541.

[14] P. V. de Campos Souza, Fuzzy neural networks and neuro-fuzzy networks: A review the main techniques and applications used in the literature, Appl. Soft Comput. 92 (2020) 106275. https://doi.org/10.1016/j.asoc.2020.106275.

[15] A. Fernández, F. Herrera, O. Cordón, M. J. del Jesus, F. Marcelloni, Evolutionary fuzzy systems for explainable artificial intelligence: Why, when, what for, and where to?, IEEE Comput. Intell. Mag. 14 (1) (2019) 69–81. https://doi.org/10.1109/MCI.2018.2881645.

[16] A. N. Toosi, M. Kahani, A new approach to intrusion detection based on an evolutionary soft computing model using neuro-fuzzy classifiers, Comput. Commun. 30 (10) (2007) 2201–2212. https://doi.org/10.1016/j.comcom.2007.05.002.

[17] A. H. Hamamoto, L. F. Carvalho, L. D. H. Sampaio, T. Abrão, M. L. Proença, Network anomaly detection system using genetic algorithm and fuzzy logic, Expert Syst. Appl. 92 (2018) 390–402. https://doi.org/10.1016/j.eswa.2017.09.013.

[18] M. Alalhareth, S.-C. Hong, An adaptive intrusion detection system in the Internet of Medical Things using fuzzy-based learning, Sensors 23 (22) (2023) 9247. https://doi.org/10.3390/s23229247.

[19] C. Elkan, The foundations of cost-sensitive learning, in: Proc. 17th Int. Jt. Conf. Artif. Intell. (IJCAI), 2001, pp. 973–978.

[20] Z.-H. Zhou, X.-Y. Liu, Training cost-sensitive neural networks with methods addressing the class imbalance problem, IEEE Trans. Knowl. Data Eng. 18 (1) (2006) 63–77. https://doi.org/10.1109/TKDE.2006.17.

[21] N. Gupta, V. Jindal, P. Bedi, CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based intrusion detection systems, Comput. Secur. 112 (2022) 102499. https://doi.org/10.1016/j.cose.2021.102499.

[22] A. Telikani, N. E. Rudbardeh, S. Soleymanpour, A. Shahbahrami, J. Shen, G. Gaydadjiev, R. Hassanpour, A cost-sensitive machine learning model with multitask learning for intrusion detection in IoT, IEEE Trans. Ind. Inform. 20 (3) (2024) 3880–3890. https://doi.org/10.1109/TII.2023.3314208.

[23] A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion 58 (2020) 82–115. https://doi.org/10.1016/j.inffus.2019.12.012.

[24] M. Wang, K. Zheng, Y. Yang, X. Wang, An explainable machine learning framework for intrusion detection systems, IEEE Access 8 (2020) 73127–73141. https://doi.org/10.1109/ACCESS.2020.2988359.

[25] A. Alabbadi, F. Bajaber, An intrusion detection system over the IoT data streams using eXplainable Artificial Intelligence (XAI), Sensors 25 (3) (2025) 847. https://doi.org/10.3390/s25030847.

[26] S. AL, S. Sagiroglu, Explainable artificial intelligence models in intrusion detection systems, Eng. Appl. Artif. Intell. 144 (2025) 110145. https://doi.org/10.1016/j.engappai.2025.110145.

[27] S. Bag, S. Sarkar, I. Bose, Enhancing cybersecurity risk assessment using temporal knowledge graph-based explainable decision support system, Decis. Support Syst. 198 (2025) 114526. https://doi.org/10.1016/j.dss.2025.114526.

[28] J. H. Friedman, Greedy function approximation: A gradient boosting machine, Ann. Stat. 29 (5) (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451.

[29] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, S. Nahavandi, A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Inf. Fusion 76 (2021) 243–297. https://doi.org/10.1016/j.inffus.2021.05.008.

[30] M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: Proc. 2009 IEEE Symp. Comput. Intell. Secur. Def. Appl. (CISDA), 2009, pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528.

[31] B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, New York, 1994. https://doi.org/10.1201/9780429246593.

[32] F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80–83. https://doi.org/10.2307/3001968.

[33] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, A. Hotho, A survey of network-based intrusion detection data sets, Comput. Secur. 86 (2019) 147–167. https://doi.org/10.1016/j.cose.2019.06.005.

[34] M. A. Al-Garadi, A. Mohamed, A. K. Al-Ali, X. Du, I. Ali, M. Guizani, A survey of machine and deep learning methods for Internet of Things (IoT) security, IEEE Commun. Surv. Tutor. 22 (3) (2020) 1646–1685. https://doi.org/10.1109/COMST.2020.2988293.
