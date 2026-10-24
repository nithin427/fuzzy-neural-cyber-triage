---
title: "Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage"
---

::: {custom-style="Author"}
Sai Nithin Erigineni\*
:::

::: {custom-style="Affiliation"}
DBA, Midwest University, 851 Parr Rd, Wentzville, MO 63385, United States\
\*Corresponding author. E-mail: serigineni@midwest.edu
:::

::: {custom-style="Affiliation"}
Article type: Short Communication. Submitted to *Knowledge-Based Systems*.
:::

# Abstract

::: {custom-style="Abstract Text"}
Security operations centres (SOCs) usually turn detector scores into actions with a single hand-set threshold. We treat alert triage as a knowledge-based decision problem: a multilayer perceptron (MLP) estimates attack propensity, fuzzy membership functions map it to Low, Medium and High risk bands, and three production rules assign Monitor, Soft-block or Escalate. On 20,000 NSL-KDD test alerts, gradient boosting is the stronger detector (ROC-AUC 0.947 against 0.908), yet on the same MLP scores the rule base lowers the mean triage cost to 3.068 (95% CI [3.016, 3.118]), against 3.264 for a 0.5 threshold and 4.309 for top-20% escalation (both Wilcoxon $p<10^{-100}$), and raises attack coverage from 0.655 to 0.686. It is cheaper than both crisp rules in all ten repeated runs ($p=0.002$) and than the threshold at every tested missed-attack cost. The gain comes from the Soft-block band, and a Bayes cost rule is cheaper still (2.164).
:::

::: {custom-style="Keywords"}
**Keywords:** Fuzzy logic; Neural networks; Decision intelligence; Intrusion detection; Alert triage; Cost-sensitive decisions; Knowledge-based systems
:::

# 1. Introduction

Security operations centres (SOCs) collect, analyse and respond to security alerts [1], but intrusion detectors produce far more alerts than analysts can investigate, most of them false positives [2]. Intrusion-detection research, from fuzzy and neuro-fuzzy detectors [3,4] to deep networks trained on NSL-KDD [5,6], usually ends with a risk score judged by accuracy or by the area under the receiver operating characteristic curve (ROC-AUC). A SOC manager instead has to decide which alerts to monitor, contain or escalate under limited analyst capacity and unequal error costs. Three-way decisions choose among acceptance, rejection and deferment under stated losses [7], and SOC studies learn when to defer alerts to analysts [8] or rank them by reinforcement learning [9], but these designs do not assign graded actions through readable rules.

| Study | Approach | Data | Cost | Rules | Graded | Tests |
|:---|:---|:---|:--:|:--:|:--:|:--:|
| Toosi [3] | Neuro-fuzzy + genetic tuning | KDD Cup 99 | (✓) | ✓ | × | × |
| Hamamoto [4] | Genetic signature + fuzzy logic | Campus IP flows | × | ✓ | × | n.r. |
| Vinayakumar [5] | Deep neural network | NSL-KDD + 5 others | × | × | × | × |
| Gupta [10] | Cost-sensitive deep ensemble | NSL-KDD + 2 others | (✓) | × | × | n.r. |
| Telikani [11] | Cost-sensitive autoencoder + SVM | UNSW-NB15, BoT-IoT | (✓) | × | × | n.r. |
| Wang [12] | SHAP explanations | NSL-KDD | × | (✓) | × | × |
| Wang [9] | Reinforcement-learning ranking | 5 multi-step sets | × | × | (✓) | n.r. |
| Bag [13] | Knowledge-graph decision support | 190 firms + 4 sets | × | (✓) | × | ✓ |
| **This work** | **MLP + fuzzy rules $R_1$–$R_3$** | **NSL-KDD, 10 seeds** | ✓ | ✓ | ✓ | ✓ |

: **Table 1.** Related work compared by design property, because data sets and tasks differ.

::: {custom-style="Table Note"}
Cost: actions chosen under an explicit cost model. Rules: readable decision logic. Graded: more than two actions. Tests: significance tests or confidence intervals. ✓ present, (✓) partial, × absent, n.r. not reported. SHAP: Shapley additive explanations; SVM: support vector machine.
:::

Fuzzy systems suit this step, because they express learned evidence as linguistic rules that operators can read and audit [14,15], but existing studies use them for other purposes (Table 1). Fuzzy intrusion detectors use rules to improve accuracy but still output a binary alarm [3,4,16]. Cost-sensitive learning [17] enters intrusion detection through weighted training losses [10,11], so costs shape the classifier rather than the action. Post hoc explainable artificial intelligence (XAI) explains a score after it has been produced [12,18,19], and knowledge-based decision support has shown the value of explicit domain knowledge in cyber-risk assessment [13]. To our knowledge, no study combines an action-cost model, readable rules and graded responses in one triage layer. This Short Communication proposes such a layer and evaluates it by the quality of its decisions.

# 2. Proposed fuzzy–neural decision framework

*Neural propensity model.* Alert $i$ has features $x_i$ and label $y_i\in\{0,1\}$, with 1 for an attack. The 38 numeric NSL-KDD fields are standardised, and the protocol, service and flag fields are one-hot encoded. An MLP $g_\theta$ with ReLU layers of width 128 and 64, dropout 0.2 and one logit output estimates the attack propensity: $$\begin{split}
p_i&=\sigma\big(g_\theta(x_i)\big),\\
\mathcal{L}(\theta)&=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_{+}\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big],\qquad w_{+}=\frac{N_{-}}{N_{+}},
\end{split}
\tag{1}$$ where $N_{-}$ and $N_{+}$ count normal and attack training records. Training uses Adam (learning rate $10^{-3}$, batch size 512) for 20 epochs and keeps the epoch with the best validation ROC-AUC.

*Fuzzy risk bands and production rules.* Two shoulder functions and one triangular function [14] map the propensity to three risk bands. With $[z]_0^1=\min(1,\max(0,z))$, the membership functions are $$\begin{align}
\mu_L(p)&=\big[(0.45-p)/0.45\big]_0^1, \tag{2}\\
\mu_M(p)&=\begin{cases}\big[(p-0.15)/0.35\big]_0^1, & p<0.5,\\ \big[(0.85-p)/0.35\big]_0^1, & p\ge 0.5,\end{cases} \tag{3}\\
\mu_H(p)&=\big[(p-0.55)/0.45\big]_0^1. \tag{4}
\end{align}$$ The rule base $\mathcal{R}=\{R_1,R_2,R_3\}$ reads: $R_1$, IF risk IS Low THEN Monitor (log the alert without intervention); $R_2$, IF risk IS Medium THEN Soft-block (apply automatic containment); and $R_3$, IF risk IS High THEN Escalate (hand the alert to a human analyst). Each rule $R_k$ fires with strength $\mu_k(p_i)$. The strongest rule sets the action, and ties go to the less severe action: $$a_i = A\big(k_i^{*}\big),\qquad k_i^{*}=\arg\max_{k\in\{L,M,H\}}\mu_k(p_i),
\tag{5}$$ where $A$ maps $L$, $M$ and $H$ to Monitor, Soft-block and Escalate. Logged firing strengths explain each decision: an alert with $p=0.60$ ($\mu_M=0.71$, $\mu_H=0.11$) is soft-blocked. Actions switch at $p=9/32=0.281$ and $p=23/32=0.719$, where $\mu_L=\mu_M$ and $\mu_M=\mu_H$ (Fig. 1). The breakpoints were set by expert judgement before the test data were examined.

![**Fig. 1.** Fuzzy mapping from attack propensity to action (seed-42 test set). Rules $R_1$–$R_3$ switch action at $p=9/32$ and $23/32$ (dotted); dashed: Bayes soft-block threshold $p=2/11$. The Soft-block band (776 alerts, 3.9%) yields the entire cost gain over the 0.5 threshold.](../figures/fig1_decision_bands.png)

*Cost model and decision metrics.* Actions are scored with the relative cost matrix $$\begin{aligned}
C(\text{Monitor},y)&=C_{FN}\,y, & C(\text{Soft-block},y)&=1+(1-y),\\
C(\text{Escalate},y)&=3+3(1-y), & C_{FN}&=10.
\end{aligned}
\tag{6}$$ Policies are compared by the mean triage cost $\bar C=(1/N)\sum_i C(a_i,y_i)$, the attack coverage $\sum_i \mathbf{1}[a_i\neq\text{Monitor}]\,y_i/\sum_i y_i$ and the number of unnecessary escalations (UE), that is, benign alerts sent to an analyst. As a reference, we report the plug-in Bayes cost rule $$a_i^{B}=\arg\min_{a}\;\big[(1-p_i)\,C(a,0)+p_i\,C(a,1)\big].
\tag{7}$$ which treats $p_i$ as calibrated, although deep-model scores do not guarantee calibration [20]. Because soft-blocking is cheaper than escalation for both classes, this rule never escalates and soft-blocks every alert with $p_i>2/11$.

# 3. Experimental design

*Data.* We used the NSL-KDD benchmark [21]. 40,000 training records were sampled and split, with stratification, into 34,000 training and 6,000 validation records, and the scaler and encoder were fitted on the training split only. The test set was capped at 20,000 records (attack rate 0.568) and contains attack types absent from training.

*Baselines.* The detectors were class-balanced logistic regression, gradient boosting (GB) [22] with scikit-learn defaults, and the MLP alone. The proposed policy uses the MLP propensity, so any benefit comes from better decisions rather than better scores. Three decision baselines acted on the same scores: a 0.5 threshold that escalates alerts above it and monitors the rest, the same threshold tuned on validation data to minimise cost (threshold moving [17]), and escalation of the 20% highest-scored alerts, which mimics a fixed analyst capacity.

*Statistics.* On the seed-42 test set, percentile 95% confidence intervals (CIs) come from 2,000 paired bootstrap resamples [23], and per-alert cost differences were tested with the two-sided Wilcoxon signed-rank test [24], which treats alerts as independent. The pipeline was repeated for seeds 42–51, with seed-level costs compared by a paired Wilcoxon test. Sensitivity sweeps varied $C_{FN}$ and the Medium half-width.

# 4. Results and discussion

| *Panel A. Detector quality (threshold 0.5 for accuracy and F1)* | | | | | | | |
|:---|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| Detector | | Accuracy | F1 | ROC-AUC | PR-AUC | | ROC-AUC, 10 seeds |
| Logistic regression | | 0.752 | 0.740 | 0.818 | 0.879 | | $0.812\pm0.008$ |
| Gradient boosting | | **0.786** | **0.774** | **0.947** | **0.951** | | $\mathbf{0.936\pm0.010}$ |
| MLP (input to proposed policy) | | 0.773 | 0.766 | 0.908 | 0.923 | | $0.900\pm0.007$ |
| *Panel B. Decision policies on identical MLP scores* | | | | | | | |
| Policy | Scores | $\bar C$ | Coverage | UE | $\Delta\bar C$ | $p$ | $\bar C$, 10 seeds |
| Threshold 0.5 | MLP | 3.264 [3.210, 3.316] | 0.655 [0.646, 0.664] | 623 | $+0.196$ [0.180, 0.213] | $7.3{\times}10^{-124}$ | $3.246\pm0.076$ |
| Top-20% escalation | MLP | 4.309 [4.246, 4.372] | 0.348 [0.339, 0.357] | **47** | $+1.242$ [1.196, 1.285] | $<10^{-300}$ | $4.318\pm0.009$ |
| **Fuzzy rule policy (proposed)** | MLP | **3.068** [3.016, 3.118] | **0.686** [0.677, 0.694] | 603 | – | – | $\mathbf{3.088\pm0.081}$ |

: **Table 2.** Main results on the seed-42 test set ($N=20{,}000$ alerts).

::: {custom-style="Table Note"}
Brackets: paired bootstrap 95% CIs. $\Delta\bar C$: cost difference from the proposed policy. $p$: two-sided Wilcoxon signed-rank test on per-alert costs. 10 seeds: mean $\pm$ SD over seeds 42–51. Bold: best in each panel. PR-AUC: area under the precision–recall curve. $\bar C$: mean triage cost, Eq. (6). UE: unnecessary escalations. Ablation and reference policies, with CIs: Supplementary Material (`table2_decision.csv`).
:::

*Prediction versus decision.* Gradient boosting is the best detector on every metric and beats the MLP in ROC-AUC in all ten seeds (Table 2, Panel A), so we make no claim about detection. On identical MLP scores, however, the rule base lowers the mean triage cost by 0.196 per alert relative to the 0.5 threshold and by 1.242 relative to top-20% escalation, and it raises coverage by 0.031 over the threshold (95% CI [0.028, 0.034]; Panel B). The two policies differ only on the 776 Medium-band alerts, which are contained instead of monitored or escalated, saving about 3,900 cost units per 20,000 alerts and sending 387 fewer alerts to analysts.

![**Fig. 2.** Mean triage cost against the missed-attack cost $C_{FN}$ (seed 42, log axis). Top-20% escalation is cheaper than the fuzzy rule policy only for $C_{FN}\le3$ (shaded). Error bars: paired bootstrap 95% CIs at the default $C_{FN}=10$ (dotted); dashed: Bayes reference.](../figures/fig2_cost_sensitivity.png)

*Robustness.* The rule base is cheaper than both crisp rules in every seed (Wilcoxon $p=0.002$ for both; means in Table 2), and its coverage exceeds that of the threshold in every seed. Its advantage over the threshold grows from 0.055 at $C_{FN}=2$ to 0.549 at $C_{FN}=30$ (Fig. 2), whereas top-20% escalation is cheaper only when $C_{FN}\le3$ (1.347 against 1.640 at $C_{FN}=2$). Widening the Medium half-width from 0.25 to 0.45 enlarges the Soft-block band from 646 to 900 alerts and lowers the cost from 3.101 to 3.034, and every width beats both crisp rules.

*Sources and limits of the gain.* Crisp bands with cut-points at $p=9/32$ and $23/32$ reproduce every action of the rule base (ablations in the Supplementary Material), so the gain comes from the intermediate containment band, while fuzziness adds a graded explanation of each decision. The rule base is not cost-optimal: the Bayes cost rule soft-blocks 8,631 alerts (43% of traffic) and is 0.903 cheaper per alert. Validation tuning moves the threshold only to $t=0.53$ (cost 3.278), and only an oracle threshold of 0.10 or lower, chosen on the test set, would beat the rule base (2.929 at $t=0.01$). On GB scores, the same rules cost 3.004, lower than on MLP scores ($p=0.017$), and cut unnecessary escalations from 603 to 222.

*Implications for SOC decision-makers.* Detectors should be judged by the decisions they enable: the action rule alone changed the cost by up to 1.242 per alert. Unlike a deferral band [8], Soft-block acts without an analyst, so false containment must be priced before it is automated. The band boundaries give managers a workload lever without retraining, useful when false positives dominate alert streams [2], and logged firing strengths give each action an audit trail that post hoc attributions lack [12,19].

*Limitations.* The study has four limitations. First, fixed bands respond to the distribution of the scores, not only to their ranking. Seed 48, the weakest replicate (cost 3.282, coverage 0.626), has a typical ROC-AUC of 0.900 but the lowest accuracy at the 0.5 threshold (0.757). Second, NSL-KDD is a laboratory benchmark released in 2009 [21] that predates modern enterprise and cloud traffic, and it was subsampled under a single protocol. Third, the cost matrix is illustrative rather than elicited from a real SOC, and coverage counts containment as covering an attack. Fourth, there was no deployment, drift analysis or user study.

# 5. Conclusion

A fuzzy–neural rule base can serve as a knowledge-based decision layer for alert triage: on NSL-KDD, it achieved lower cost and higher coverage than crisp threshold and capacity rules across seeds, cost ratios and band widths, through its intermediate containment band. Future work will elicit real SOC costs, test the layer in live operation and bound automatic-action errors by conformal classification with a reject option [25].

# Data availability

The code, results (with the ablations in `table2_decision.csv`) and figure scripts are in the Supplementary Material and at <https://github.com/nithin427/fuzzy-neural-cyber-triage>. NSL-KDD is available from the Canadian Institute for Cybersecurity at <https://www.unb.ca/cic/datasets/nsl.html> [21].

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Grok Bot, an AI-assisted writing and coding tool, to help with language editing, code refactoring, reference checking and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# CRediT authorship contribution statement

**Sai Nithin Erigineni:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

# References

::: {custom-style="Reference"}
[1]	M. Vielberth, F. Böhm, I. Fichtinger, G. Pernul, Security operations center: A systematic study and open challenges, IEEE Access 8 (2020) 227756–227779. https://doi.org/10.1109/ACCESS.2020.3045514.
:::

::: {custom-style="Reference"}
[2]	H. Albasheer, M. Md Siraj, A. Mubarakali, O. Elsier Tayfour, S. Salih, M. Hamdan, S. Khan, A. Zainal, S. Kamarudeen, Cyber-attack prediction based on network intrusion detection systems for alert correlation techniques: A survey, Sensors 22 (4) (2022) 1494. https://doi.org/10.3390/s22041494.
:::

::: {custom-style="Reference"}
[3]	A. N. Toosi, M. Kahani, A new approach to intrusion detection based on an evolutionary soft computing model using neuro-fuzzy classifiers, Comput. Commun. 30 (10) (2007) 2201–2212. https://doi.org/10.1016/j.comcom.2007.05.002.
:::

::: {custom-style="Reference"}
[4]	A. H. Hamamoto, L. F. Carvalho, L. D. H. Sampaio, T. Abrão, M. L. Proença, Network anomaly detection system using genetic algorithm and fuzzy logic, Expert Syst. Appl. 92 (2018) 390–402. https://doi.org/10.1016/j.eswa.2017.09.013.
:::

::: {custom-style="Reference"}
[5]	R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, S. Venkatraman, Deep learning approach for intelligent intrusion detection system, IEEE Access 7 (2019) 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334.
:::

::: {custom-style="Reference"}
[6]	T. Su, H. Sun, J. Zhu, S. Wang, Y. Li, BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset, IEEE Access 8 (2020) 29575–29585. https://doi.org/10.1109/ACCESS.2020.2972627.
:::

::: {custom-style="Reference"}
[7]	Y. Yao, Three-way decisions with probabilistic rough sets, Inf. Sci. 180 (3) (2010) 341–353. https://doi.org/10.1016/j.ins.2009.09.021.
:::

::: {custom-style="Reference"}
[8]	F. Jalalvand, M. Baruwal Chhetri, S. Nepal, C. Paris, Adaptive alert prioritisation in security operations centres via learning to defer with human feedback, Inf. Sci. 760 (2027) 124128. https://doi.org/10.1016/j.ins.2026.124128.
:::

::: {custom-style="Reference"}
[9]	X. Wang, X. Yang, X. Liang, X. Zhang, W. Zhang, X. Gong, Combating alert fatigue with AlertPro: Context-aware alert prioritization using reinforcement learning for multi-step attack detection, Comput. Secur. 137 (2024) 103583. https://doi.org/10.1016/j.cose.2023.103583.
:::

::: {custom-style="Reference"}
[10]	N. Gupta, V. Jindal, P. Bedi, CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based intrusion detection systems, Comput. Secur. 112 (2022) 102499. https://doi.org/10.1016/j.cose.2021.102499.
:::

::: {custom-style="Reference"}
[11]	A. Telikani, N. E. Rudbardeh, S. Soleymanpour, A. Shahbahrami, J. Shen, G. Gaydadjiev, R. Hassanpour, A cost-sensitive machine learning model with multitask learning for intrusion detection in IoT, IEEE Trans. Ind. Inform. 20 (3) (2024) 3880–3890. https://doi.org/10.1109/TII.2023.3314208.
:::

::: {custom-style="Reference"}
[12]	M. Wang, K. Zheng, Y. Yang, X. Wang, An explainable machine learning framework for intrusion detection systems, IEEE Access 8 (2020) 73127–73141. https://doi.org/10.1109/ACCESS.2020.2988359.
:::

::: {custom-style="Reference"}
[13]	S. Bag, S. Sarkar, I. Bose, Enhancing cybersecurity risk assessment using temporal knowledge graph-based explainable decision support system, Decis. Support Syst. 198 (2025) 114526. https://doi.org/10.1016/j.dss.2025.114526.
:::

::: {custom-style="Reference"}
[14]	L. A. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.
:::

::: {custom-style="Reference"}
[15]	A. Fernández, F. Herrera, O. Cordón, M. J. del Jesus, F. Marcelloni, Evolutionary fuzzy systems for explainable artificial intelligence: Why, when, what for, and where to?, IEEE Comput. Intell. Mag. 14 (1) (2019) 69–81. https://doi.org/10.1109/MCI.2018.2881645.
:::

::: {custom-style="Reference"}
[16]	M. Alalhareth, S.-C. Hong, An adaptive intrusion detection system in the Internet of Medical Things using fuzzy-based learning, Sensors 23 (22) (2023) 9247. https://doi.org/10.3390/s23229247.
:::

::: {custom-style="Reference"}
[17]	Z.-H. Zhou, X.-Y. Liu, Training cost-sensitive neural networks with methods addressing the class imbalance problem, IEEE Trans. Knowl. Data Eng. 18 (1) (2006) 63–77. https://doi.org/10.1109/TKDE.2006.17.
:::

::: {custom-style="Reference"}
[18]	A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion 58 (2020) 82–115. https://doi.org/10.1016/j.inffus.2019.12.012.
:::

::: {custom-style="Reference"}
[19]	A. Alabbadi, F. Bajaber, An intrusion detection system over the IoT data streams using eXplainable Artificial Intelligence (XAI), Sensors 25 (3) (2025) 847. https://doi.org/10.3390/s25030847.
:::

::: {custom-style="Reference"}
[20]	M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, S. Nahavandi, A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Inf. Fusion 76 (2021) 243–297. https://doi.org/10.1016/j.inffus.2021.05.008.
:::

::: {custom-style="Reference"}
[21]	M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: Proc. 2009 IEEE Symp. Comput. Intell. Secur. Def. Appl. (CISDA), 2009, pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528.
:::

::: {custom-style="Reference"}
[22]	J. H. Friedman, Greedy function approximation: A gradient boosting machine, Ann. Stat. 29 (5) (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451.
:::

::: {custom-style="Reference"}
[23]	B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, New York, 1994. https://doi.org/10.1201/9780429246593.
:::

::: {custom-style="Reference"}
[24]	F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80–83. https://doi.org/10.2307/3001968.
:::

::: {custom-style="Reference"}
[25]	J. Hallberg Szabadváry, T. Löfström, U. Johansson, C. Sönströd, E. Ahlberg, L. Carlsson, Classification with reject option: Distribution-free error guarantees via conformal prediction, Mach. Learn. Appl. 20 (2025) 100664. https://doi.org/10.1016/j.mlwa.2025.100664.
:::
