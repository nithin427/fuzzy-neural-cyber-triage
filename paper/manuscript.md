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
Security operations centres (SOCs) receive risk scores from machine-learning detectors, but the step from score to action usually rests on a single hand-set threshold. This paper treats alert triage as a knowledge-based decision problem. A multilayer perceptron (MLP) estimates attack propensity, fuzzy membership functions map it to Low, Medium and High risk bands, and three production rules assign Monitor, Soft-block or Escalate. On 20,000 NSL-KDD test alerts, gradient boosting is the stronger detector (ROC-AUC 0.947 against 0.908). On the same MLP scores, however, the rule base lowers the mean triage cost to 3.068 (95% CI [3.016, 3.118]), compared with 3.264 for a 0.5 threshold and 4.309 for top-20% escalation. It also raises attack coverage from 0.655 to 0.686 (Wilcoxon $p<10^{-100}$). The rule base is cheaper than both crisp rules in all ten repeated runs ($p=0.002$) and cheaper than the threshold at every tested missed-attack cost from 2 to 30. Ablations show that the gain comes from the Soft-block band and that a Bayes cost rule is cheaper still (2.164). The contribution is an auditable decision layer, not a new detector.
:::

::: {custom-style="Keywords"}
**Keywords:** Fuzzy logic; Neural networks; Decision intelligence; Intrusion detection; Alert triage; Cost-sensitive decisions; Knowledge-based systems
:::

# 1. Introduction

Security operations centres (SOCs) collect, analyse and respond to an organisation's security alerts [1]. Their analysts receive far more alerts than they can investigate, and the resulting alert fatigue is a central operational risk [2,3]. Intrusion-detection research has progressed from classical data mining [4,5] to deep networks trained on benchmarks such as NSL-KDD [6,7]. Most of this work, however, ends with a risk score and judges it by accuracy or by the area under the receiver operating characteristic curve (ROC-AUC). A SOC manager faces a different question: with limited analyst capacity and unequal error costs, which alerts should be monitored, contained or escalated? This is a data-driven decision task rather than a prediction task [8,9].

| Study | Approach | Data | Cost | Rules | Graded | Tests |
|:---|:---|:---|:--:|:--:|:--:|:--:|
| Toosi [10] | Neuro-fuzzy + genetic tuning | KDD Cup 99 | (✓) | ✓ | × | × |
| Hamamoto [11] | Genetic signature + fuzzy logic | Campus IP flows | × | ✓ | × | n.r. |
| Alalhareth [12] | LSTM, fuzzy-tuned training | WUSTL-EHMS-2020 | (✓) | × | × | × |
| Vinayakumar [6] | Deep neural network | NSL-KDD + 5 others | × | × | × | × |
| Su [7] | Bidirectional LSTM + attention | NSL-KDD | × | × | × | × |
| Gupta [13] | Cost-sensitive deep ensemble | NSL-KDD + 2 others | (✓) | × | × | n.r. |
| Telikani [14] | Cost-sensitive autoencoder + SVM | UNSW-NB15, BoT-IoT | (✓) | × | × | n.r. |
| Wang [15] | SHAP explanations | NSL-KDD | × | (✓) | × | × |
| Alabbadi [16] | Deep models + XAI | TON_IoT | × | (✓) | × | n.r. |
| Wang [17] | Reinforcement-learning ranking | 5 multi-step sets | × | × | (✓) | n.r. |
| Bag [18] | Knowledge-graph decision support | 190 firms + 4 sets | × | (✓) | × | ✓ |
| **This work** | **MLP + fuzzy rules $R_1$–$R_3$** | **NSL-KDD, 10 seeds** | ✓ | ✓ | ✓ | ✓ |

: **Table 1.** Related work compared by design property rather than accuracy, because data sets and tasks differ. Only this work combines an explicit action-cost model, readable rules and graded actions with significance testing.

::: {custom-style="Table Note"}
Cost: actions chosen under an explicit cost model. Rules: readable decision logic. Graded: more than two actions. Tests: significance tests or confidence intervals. ✓ present, (✓) partial, × absent, n.r. not reported. LSTM: long short-term memory; SHAP: Shapley additive explanations; SVM: support vector machine.
:::

Fuzzy and neuro-fuzzy systems suit this decision step, because they express learned evidence as linguistic rules that operators can read and audit [19–22]. As Table 1 shows, existing studies use these strengths for other purposes. Fuzzy intrusion detectors use rules to improve accuracy, but they still output a binary alarm [10–12]. Cost-sensitive learning [23,24] enters intrusion detection mainly through weighted training losses [13,14], so the cost model shapes the classifier but does not choose the action. Explainable artificial intelligence (XAI) methods explain a score after it has been produced [15,16,25]. Reinforcement learning can rank alerts by priority, yet the ranking does not assign an action [17]. Knowledge-based decision support has shown the value of explicit domain knowledge in cyber-risk assessment [18]. None of these studies, however, combines an action-cost model, readable rules and graded responses in one triage layer.

This Short Communication addresses that gap with three contributions. It proposes a knowledge-based layer that maps neural attack propensity to fuzzy risk bands and then to SOC actions through explicit rules (Section 2). It evaluates the layer by the quality of its decisions, using paired tests, repeated runs and sensitivity analyses (Sections 3 and 4). Finally, ablations show where the gain comes from and where it ends.

# 2. Proposed fuzzy–neural decision framework

*Neural propensity model.* Each alert $i$ has a feature vector $x_i$ and a label $y_i\in\{0,1\}$, where 1 denotes an attack. The 38 numeric NSL-KDD fields are standardised, and the protocol, service and flag fields are one-hot encoded. An MLP $g_\theta$ with two ReLU layers of width 128 and 64, a dropout rate of 0.2 and a single logit output estimates the attack propensity: $$\begin{split}
p_i&=\sigma\big(g_\theta(x_i)\big),\\
\mathcal{L}(\theta)&=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_{+}\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big],\qquad w_{+}=\frac{N_{-}}{N_{+}},
\end{split}
\tag{1}$$ where $N_{-}$ and $N_{+}$ are the numbers of normal and attack training records, so that $w_{+}$ offsets the class imbalance. The network is trained with Adam at a learning rate of $10^{-3}$ and a batch size of 512 for 20 epochs, and the epoch with the best validation ROC-AUC is kept.

*Fuzzy risk bands and production rules.* Two shoulder functions and one triangular function map the propensity to three linguistic risk bands [19]. With the clipping operator $[z]_0^1=\min(1,\max(0,z))$, the membership functions are $$\begin{align}
\mu_L(p)&=\big[(0.45-p)/0.45\big]_0^1, \tag{2}\\
\mu_M(p)&=\begin{cases}\big[(p-0.15)/0.35\big]_0^1, & p<0.5,\\ \big[(0.85-p)/0.35\big]_0^1, & p\ge 0.5,\end{cases} \tag{3}\\
\mu_H(p)&=\big[(p-0.55)/0.45\big]_0^1. \tag{4}
\end{align}$$ The knowledge layer is the production-rule base $\mathcal{R}=\{R_1,R_2,R_3\}$:

::: {custom-style="Rule"}
$R_1$: IF risk IS Low THEN Monitor: log the alert without intervention.
:::

::: {custom-style="Rule"}
$R_2$: IF risk IS Medium THEN Soft-block: apply automatic containment.
:::

::: {custom-style="Rule"}
$R_3$: IF risk IS High THEN Escalate: hand the alert to a human analyst.
:::

Each rule $R_k$ fires with strength $\mu_k(p_i)$. The strongest rule determines the action, and a tie goes to the less severe action: $$a_i = A\big(k_i^{*}\big),\qquad k_i^{*}=\arg\max_{k\in(L,M,H)}\mu_k(p_i),
\tag{5}$$ where $A$ maps $L$, $M$ and $H$ to Monitor, Soft-block and Escalate. The firing strengths are logged with every decision and serve as its explanation. For example, an alert with $p=0.60$ has $\mu_M=0.71$ and $\mu_H=0.11$, so it is soft-blocked. Solving $\mu_L=\mu_M$ and $\mu_M=\mu_H$ shows that the rule base switches action at $p=9/32=0.281$ and $p=23/32=0.719$ (Fig. 1). The breakpoints were set by expert judgement before the test data were examined.

![**Fig. 1.** Fuzzy mapping from attack propensity to action on the seed-42 test set. The membership functions of Eqs. (2)–(4) and rules $R_1$–$R_3$ switch action at $p=9/32$ and $p=23/32$ (dotted lines). The Soft-block band holds only 776 alerts (3.9%), yet it produces the entire cost gain over the 0.5 threshold.](../figures/fig1_decision_bands.png)

*Cost model and decision metrics.* Each action is scored with the relative cost matrix $$\begin{aligned}
C(\text{Monitor},y)&=C_{FN}\,y, & C(\text{Soft-block},y)&=1+(1-y),\\
C(\text{Escalate},y)&=3+3(1-y), & C_{FN}&=10.
\end{aligned}
\tag{6}$$ Each policy is evaluated on three metrics. The mean triage cost is $\bar C=\frac1N\sum_i C(a_i,y_i)$. The attack coverage, $\sum_i \mathbf{1}[a_i\neq\text{Monitor}]\,y_i/\sum_i y_i$, is the share of attacks that receive any action other than Monitor. The third metric counts unnecessary escalations (UE), which are benign alerts sent to an analyst. As a reference, we also report the plug-in Bayes cost rule $$a_i^{B}=\arg\min_{a}\;\big[(1-p_i)\,C(a,0)+p_i\,C(a,1)\big].
\tag{7}$$ This rule treats $p_i$ as a calibrated probability, although deep-model scores do not guarantee calibration [26]. Because soft-blocking costs less than escalation for both classes, the Bayes rule never escalates. Instead, it soft-blocks every alert with $p_i>2/11$.

# 3. Experimental design

*Data.* The experiments used the NSL-KDD benchmark [27]. For reproducibility on a standard CPU, we sampled 40,000 training records and split them, with stratification, into 34,000 training and 6,000 validation records. The scaler and encoder were fitted on the training split only. The test set was capped at 20,000 records with an attack rate of 0.568. Because it contains attack types absent from training, the evaluation includes a realistic distribution shift.

*Baselines.* Three detectors were compared: class-balanced logistic regression, gradient boosting (GB) [28] with scikit-learn defaults, and the MLP alone. The proposed policy uses the MLP propensity, so any benefit must come from better decisions rather than better scores. Three decision baselines acted on the same scores. The first escalates every alert scored above 0.5 and monitors the rest. The second tunes this threshold on validation data to minimise cost, a form of threshold moving [24]. The third escalates the 20% highest-scored alerts, mimicking a fixed analyst capacity. Learned policies [17] were excluded because they need interaction data that the benchmark lacks.

*Statistics.* On the seed-42 test set, we report percentile 95% confidence intervals (CIs) from 2,000 paired bootstrap resamples of alerts [29]. Per-alert cost differences were tested with the two-sided Wilcoxon signed-rank test [30]. The full pipeline was repeated for seeds 42–51, and the seed-level costs were compared with a paired Wilcoxon test. Two sweeps varied the missed-attack cost $C_{FN}$ from 2 to 30 and the half-width of the Medium band from 0.25 to 0.45.

# 4. Results and discussion

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

: **Table 2.** Main results on the seed-42 test set ($N=20{,}000$ alerts). Brackets: paired bootstrap 95% CIs. $\Delta\bar C$: cost difference from the proposed policy. $p$: two-sided Wilcoxon signed-rank test on per-alert costs. 10 seeds: mean $\pm$ SD over seeds 42–51. Bold: best value in Panels A and B1.

::: {custom-style="Table Note"}
PR-AUC: area under the precision–recall curve. $\bar C$: mean triage cost, Eq. (6). UE: unnecessary escalations. Panel B2 CIs are given in `results/table2_decision.csv`.
:::

*Prediction versus decision.* Gradient boosting is the best detector on every metric (Table 2, Panel A), and it beats the MLP in ROC-AUC in all ten seeds, so this study makes no claim about detection. On identical MLP scores, however, the rule base lowers the mean triage cost by 0.196 per alert relative to the 0.5 threshold and by 1.242 relative to top-20% escalation (Panel B1). It also raises coverage by 0.031 relative to the threshold (95% CI [0.028, 0.034]). The rule base and the threshold differ only on the 776 Medium-band alerts, which the rule base contains instead of monitoring or escalating them. This small band saves about 3,900 cost units per 20,000 alerts and sends 387 fewer alerts to analysts. Top-20% escalation yields the fewest unnecessary escalations (47), but it covers only 0.348 of attacks.

![**Fig. 2.** Mean triage cost as a function of the missed-attack cost $C_{FN}$ (seed 42). The fuzzy rule policy is cheaper than the 0.5 threshold at every tested value from 2 to 30. Top-20% escalation is cheaper only for $C_{FN}\le3$ (shaded region). Error bars show paired bootstrap 95% CIs at the default $C_{FN}=10$ (dotted line).](../figures/fig2_cost_sensitivity.png)

*Robustness.* These gains are stable across repeated runs and cost settings. Over ten seeds, the rule base has a mean cost of $3.088\pm0.081$, compared with $3.246\pm0.076$ for the threshold and $4.318\pm0.009$ for top-20% escalation. It is cheaper than both crisp rules in every seed (Wilcoxon $p=0.002$ for both), and its coverage exceeds that of the threshold in every seed. Its advantage over the threshold grows from 0.055 at $C_{FN}=2$ to 0.549 at $C_{FN}=30$ (Fig. 2). Top-20% escalation is cheaper only when $C_{FN}\le3$, for example 1.347 against 1.640 at $C_{FN}=2$, because acting on fewer alerts pays off when missed attacks are cheap. Widening the Medium half-width from 0.25 to 0.45 enlarges the Soft-block band from 646 to 900 alerts and lowers the cost steadily from 3.101 to 3.034. Every width beats both crisp rules.

*Sources and limits of the gain.* Crisp bands with cut-points at $p=9/32$ and $23/32$ reproduce every action of the rule base (Panel B2), so the benefit comes from the intermediate containment band rather than from fuzziness as such. What fuzziness adds is a graded explanation of each decision. The rule base is also not cost-optimal. The Bayes cost rule is 0.903 cheaper per alert, because soft-blocking dominates escalation under Eq. (6), and it therefore soft-blocks 8,631 alerts, or 43% of all traffic. Threshold tuning does not match the rule base either. Validation tuning moves the threshold only to $t=0.53$, at a cost of 3.278, because unseen attack types shift the test scores. Only an oracle threshold of 0.10 or lower, chosen on the test set, would win, with a minimum cost of 2.929 at $t=0.01$. Finally, the layer transfers to another detector. On GB scores, the same rules cost 3.004, which is lower than on MLP scores ($p=0.017$), and they cut unnecessary escalations from 603 to 222.

*Implications for SOC decision-makers.* Detectors should be judged by the decisions they enable, since the action rule alone changed the cost by up to 1.242 per alert. Because the cost matrix sets the mix of actions, organisations should price false containment before automating Soft-block. The band boundaries give managers a workload lever that needs no retraining. Named rules with logged firing strengths give each action an audit trail that post hoc attributions lack [15,16], which supports compliance and human–AI teaming [3].

# 5. Limitations and conclusion

The study has four main limitations. First, fixed bands respond to the distribution of the scores, not only to their ranking. Seed 48, the weakest replicate (cost 3.282, coverage 0.626), has a typical ROC-AUC of 0.900 but the lowest accuracy at the 0.5 threshold (0.757). Adaptive tuning in the style of ANFIS [20], or bands derived from the cost matrix, could reduce this sensitivity and narrow the gap to the Bayes rule. Second, NSL-KDD is a laboratory benchmark from 2009 that does not reflect modern enterprise, cloud or Internet of Things traffic [31,32], and the data were subsampled under a single protocol. Third, the cost matrix is illustrative rather than elicited from a real SOC, and the coverage metric counts containment as covering an attack. Fourth, the study involved no deployment, drift analysis or user study.

In conclusion, a fuzzy–neural rule base can serve as a knowledge-based decision layer for alert triage. On NSL-KDD, it achieved lower cost and higher coverage than crisp threshold and capacity rules across seeds, cost ratios and band widths, and its gain comes from the intermediate containment band. Future work will derive the bands from costs elicited in real SOCs and validate the layer in live operation.

# Data availability

The code, result tables and figure-generation scripts are publicly available at <https://github.com/nithin427/fuzzy-neural-cyber-triage>. NSL-KDD is available from the Canadian Institute for Cybersecurity [27].

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# CRediT authorship contribution statement

**Sai Nithin Erigineni:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Grok Bot, an AI-assisted writing and coding tool, to help with language editing, code refactoring, reference checking and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. All results, tables and figures were generated by the author's executable code from the public dataset; no figure was created with generative AI.

# References

::: {custom-style="Reference"}
[1]	M. Vielberth, F. Böhm, I. Fichtinger, G. Pernul, Security operations center: A systematic study and open challenges, IEEE Access 8 (2020) 227756–227779. https://doi.org/10.1109/ACCESS.2020.3045514.
:::

::: {custom-style="Reference"}
[2]	S. Tariq, M. Baruwal Chhetri, S. Nepal, C. Paris, Alert fatigue in security operations centres: Research challenges and opportunities, ACM Comput. Surv. 57 (9) (2025) 1–38. https://doi.org/10.1145/3723158.
:::

::: {custom-style="Reference"}
[3]	M. Baruwal Chhetri, S. Tariq, R. Singh, F. Jalalvand, C. Paris, S. Nepal, Towards human–AI teaming to mitigate alert fatigue in security operations centres, ACM Trans. Internet Technol. 24 (3) (2024) 1–22. https://doi.org/10.1145/3670009.
:::

::: {custom-style="Reference"}
[4]	H.-J. Liao, C.-H. Richard Lin, Y.-C. Lin, K.-Y. Tung, Intrusion detection system: A comprehensive review, J. Netw. Comput. Appl. 36 (1) (2013) 16–24. https://doi.org/10.1016/j.jnca.2012.09.004.
:::

::: {custom-style="Reference"}
[5]	A. L. Buczak, E. Guven, A survey of data mining and machine learning methods for cyber security intrusion detection, IEEE Commun. Surv. Tutor. 18 (2) (2016) 1153–1176. https://doi.org/10.1109/COMST.2015.2494502.
:::

::: {custom-style="Reference"}
[6]	R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, S. Venkatraman, Deep learning approach for intelligent intrusion detection system, IEEE Access 7 (2019) 41525–41550. https://doi.org/10.1109/ACCESS.2019.2895334.
:::

::: {custom-style="Reference"}
[7]	T. Su, H. Sun, J. Zhu, S. Wang, Y. Li, BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset, IEEE Access 8 (2020) 29575–29585. https://doi.org/10.1109/ACCESS.2020.2972627.
:::

::: {custom-style="Reference"}
[8]	G. Vial, Understanding digital transformation: A review and a research agenda, J. Strateg. Inf. Syst. 28 (2) (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003.
:::

::: {custom-style="Reference"}
[9]	Y. Duan, J. S. Edwards, Y. K. Dwivedi, Artificial intelligence for decision making in the era of Big Data – evolution, challenges and research agenda, Int. J. Inf. Manag. 48 (2019) 63–71. https://doi.org/10.1016/j.ijinfomgt.2019.01.021.
:::

::: {custom-style="Reference"}
[10]	A. N. Toosi, M. Kahani, A new approach to intrusion detection based on an evolutionary soft computing model using neuro-fuzzy classifiers, Comput. Commun. 30 (10) (2007) 2201–2212. https://doi.org/10.1016/j.comcom.2007.05.002.
:::

::: {custom-style="Reference"}
[11]	A. H. Hamamoto, L. F. Carvalho, L. D. H. Sampaio, T. Abrão, M. L. Proença, Network anomaly detection system using genetic algorithm and fuzzy logic, Expert Syst. Appl. 92 (2018) 390–402. https://doi.org/10.1016/j.eswa.2017.09.013.
:::

::: {custom-style="Reference"}
[12]	M. Alalhareth, S.-C. Hong, An adaptive intrusion detection system in the Internet of Medical Things using fuzzy-based learning, Sensors 23 (22) (2023) 9247. https://doi.org/10.3390/s23229247.
:::

::: {custom-style="Reference"}
[13]	N. Gupta, V. Jindal, P. Bedi, CSE-IDS: Using cost-sensitive deep learning and ensemble algorithms to handle class imbalance in network-based intrusion detection systems, Comput. Secur. 112 (2022) 102499. https://doi.org/10.1016/j.cose.2021.102499.
:::

::: {custom-style="Reference"}
[14]	A. Telikani, N. E. Rudbardeh, S. Soleymanpour, A. Shahbahrami, J. Shen, G. Gaydadjiev, R. Hassanpour, A cost-sensitive machine learning model with multitask learning for intrusion detection in IoT, IEEE Trans. Ind. Inform. 20 (3) (2024) 3880–3890. https://doi.org/10.1109/TII.2023.3314208.
:::

::: {custom-style="Reference"}
[15]	M. Wang, K. Zheng, Y. Yang, X. Wang, An explainable machine learning framework for intrusion detection systems, IEEE Access 8 (2020) 73127–73141. https://doi.org/10.1109/ACCESS.2020.2988359.
:::

::: {custom-style="Reference"}
[16]	A. Alabbadi, F. Bajaber, An intrusion detection system over the IoT data streams using eXplainable Artificial Intelligence (XAI), Sensors 25 (3) (2025) 847. https://doi.org/10.3390/s25030847.
:::

::: {custom-style="Reference"}
[17]	X. Wang, X. Yang, X. Liang, X. Zhang, W. Zhang, X. Gong, Combating alert fatigue with AlertPro: Context-aware alert prioritization using reinforcement learning for multi-step attack detection, Comput. Secur. 137 (2024) 103583. https://doi.org/10.1016/j.cose.2023.103583.
:::

::: {custom-style="Reference"}
[18]	S. Bag, S. Sarkar, I. Bose, Enhancing cybersecurity risk assessment using temporal knowledge graph-based explainable decision support system, Decis. Support Syst. 198 (2025) 114526. https://doi.org/10.1016/j.dss.2025.114526.
:::

::: {custom-style="Reference"}
[19]	L. A. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X.
:::

::: {custom-style="Reference"}
[20]	J.-S. R. Jang, ANFIS: Adaptive-network-based fuzzy inference system, IEEE Trans. Syst. Man Cybern. 23 (3) (1993) 665–685. https://doi.org/10.1109/21.256541.
:::

::: {custom-style="Reference"}
[21]	P. V. de Campos Souza, Fuzzy neural networks and neuro-fuzzy networks: A review the main techniques and applications used in the literature, Appl. Soft Comput. 92 (2020) 106275. https://doi.org/10.1016/j.asoc.2020.106275.
:::

::: {custom-style="Reference"}
[22]	A. Fernández, F. Herrera, O. Cordón, M. J. del Jesus, F. Marcelloni, Evolutionary fuzzy systems for explainable artificial intelligence: Why, when, what for, and where to?, IEEE Comput. Intell. Mag. 14 (1) (2019) 69–81. https://doi.org/10.1109/MCI.2018.2881645.
:::

::: {custom-style="Reference"}
[23]	C. Elkan, The foundations of cost-sensitive learning, in: Proc. 17th Int. Jt. Conf. Artif. Intell. (IJCAI), 2001, pp. 973–978.
:::

::: {custom-style="Reference"}
[24]	Z.-H. Zhou, X.-Y. Liu, Training cost-sensitive neural networks with methods addressing the class imbalance problem, IEEE Trans. Knowl. Data Eng. 18 (1) (2006) 63–77. https://doi.org/10.1109/TKDE.2006.17.
:::

::: {custom-style="Reference"}
[25]	A. Barredo Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. Garcia, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI, Inf. Fusion 58 (2020) 82–115. https://doi.org/10.1016/j.inffus.2019.12.012.
:::

::: {custom-style="Reference"}
[26]	M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, V. Makarenkov, S. Nahavandi, A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Inf. Fusion 76 (2021) 243–297. https://doi.org/10.1016/j.inffus.2021.05.008.
:::

::: {custom-style="Reference"}
[27]	M. Tavallaee, E. Bagheri, W. Lu, A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: Proc. 2009 IEEE Symp. Comput. Intell. Secur. Def. Appl. (CISDA), 2009, pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528.
:::

::: {custom-style="Reference"}
[28]	J. H. Friedman, Greedy function approximation: A gradient boosting machine, Ann. Stat. 29 (5) (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451.
:::

::: {custom-style="Reference"}
[29]	B. Efron, R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall/CRC, New York, 1994. https://doi.org/10.1201/9780429246593.
:::

::: {custom-style="Reference"}
[30]	F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (6) (1945) 80–83. https://doi.org/10.2307/3001968.
:::

::: {custom-style="Reference"}
[31]	M. Ring, S. Wunderlich, D. Scheuring, D. Landes, A. Hotho, A survey of network-based intrusion detection data sets, Comput. Secur. 86 (2019) 147–167. https://doi.org/10.1016/j.cose.2019.06.005.
:::

::: {custom-style="Reference"}
[32]	M. A. Al-Garadi, A. Mohamed, A. K. Al-Ali, X. Du, I. Ali, M. Guizani, A survey of machine and deep learning methods for Internet of Things (IoT) security, IEEE Commun. Surv. Tutor. 22 (3) (2020) 1646–1685. https://doi.org/10.1109/COMST.2020.2988293.
:::
