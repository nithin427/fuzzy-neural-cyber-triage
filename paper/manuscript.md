# Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage

**Article type:** Short Communication — *Knowledge-Based Systems* (Elsevier)  
**Author:** Sai Nithin Erigineni, DBA, Midwest University, 851 Parr Rd, Wentzville, MO 63385, United States (corresponding author: serigineni@midwest.edu)  
**Code and results:** GitHub repository, to be made public on acceptance; available to the editor and reviewers on reasonable request

> Markdown mirror of `paper/elsevier/main.tex` (the LaTeX file is the submission source). Generated with pandoc; every number comes from `results/summary.json` and `results/*.csv`.

## Abstract

Security operations centres (SOCs) receive risk scores from machine-learning detectors, yet the step from score to action is usually a single hand-set threshold. This Short Communication treats cyber alert triage as a knowledge-based decision problem. A multilayer perceptron estimates attack propensity; triangular fuzzy membership functions map it to Low, Medium and High risk bands; and a three-rule production base assigns Monitor, Soft-block or Escalate, scored under an explicit relative cost matrix. On the NSL-KDD benchmark (seed 42; 20,000 test alerts), gradient boosting remains the strongest detector (ROC-AUC 0.947 versus 0.908 for the neural model). The fuzzy rule layer lowers the mean triage cost to 3.068 [95% bootstrap CI 3.016–3.118], against 3.264 for a 0.5 threshold and 4.309 for a top-20% escalation rule, and raises attack coverage from 0.655 to 0.686 (paired Wilcoxon $p<10^{-100}$). It is cheaper than both crisp rules in all ten repeated runs ($p=0.002$) and at every missed-attack cost from 2 to 30, except that top-20% escalation is cheaper when missed attacks cost 2–3 units. Ablations bound the claim. The gain comes from the intermediate Soft-block band, which is decision-equivalent to crisp cut-points at $p=9/32$ and $23/32$, and a Bayes rule that uses the cost matrix directly is cheaper still (2.164). The contribution is therefore an interpretable, auditable decision layer and a decision-centric evaluation protocol, not a new detector.

**Keywords:** Fuzzy logic; Neural networks; Decision intelligence; Intrusion detection; Alert triage; Cost-sensitive decisions; Knowledge-based systems

# Introduction

SOC analysts face far more alerts than they can investigate, and alert fatigue is now recognised as a central operational risk of security operations [1,2]. Recent work prioritises alerts with reinforcement learning or human–AI teaming [2,3], but most intrusion-detection research still ends at a score or a label and is judged by accuracy or ROC-AUC [4–7]. For a SOC manager the business question is different: given finite analyst capacity and asymmetric costs of missed attacks and needless escalations, which alerts should be monitored, contained automatically, or escalated? Answering it explicitly is part of the digital transformation of security operations from tool-centric to data-driven decision processes [8,9].

Fuzzy and neuro-fuzzy systems are a natural vehicle for this step because they express learned evidence in linguistic terms that operators can audit [10–12]. In intrusion detection, however, fuzzy–neural hybrids have mainly been used to raise detection rates [13–15]; cost-sensitive learning has been applied to the classifier rather than to the action [16,17]; and explainable-AI work explains scores without choosing actions [18–20]. Knowledge-based decision support for cyber risk, such as explainable knowledge-graph risk assessment [21], shows the value of explicit knowledge representations, but a learned-score-to-action rule base for alert triage that is evaluated with decision metrics, paired tests and ablations is still missing.

This Short Communication closes that gap with three contributions: (i) a compact knowledge-based triage layer that converts neural propensity into fuzzy risk bands and then into SOC actions through a stated production-rule base (Section 2); (ii) a decision-centric evaluation protocol, comprising mean triage cost, attack coverage and unnecessary escalations, with paired bootstrap confidence intervals, Wilcoxon tests, a 10-seed repetition, and cost-ratio and membership sweeps (Sections 3–4); and (iii) an ablation that locates where the gain comes from and how far the rule base is from a cost-optimal reference. The design differs structurally from the strongest baseline, gradient boosting [22]: boosting is a detector whose output still requires a threshold, whereas the proposed layer is detector-agnostic and acts on any propensity score. Section 4 demonstrates this by applying the same rules to boosting scores.

# Proposed fuzzy–neural decision framework

![**Fig. 1.** Proposed framework. (a) Pipeline from alert features to neural propensity, fuzzy risk bands and the production-rule base $R_1$–$R_3$. (b) Membership functions of Eqs. (2)–(4); dotted lines mark the dominant-band switch points $p=9/32$ and $23/32$ that separate Monitor, Soft-block and Escalate.](../figures/fig1_architecture.png)

## Neural propensity model

Each alert $i$ has features $x_i$ (38 numeric NSL-KDD fields standardised; protocol, service and flag one-hot encoded) and label $y_i\in\{0,1\}$ (normal, attack). A multilayer perceptron $g_\theta$ with ReLU layers of width 128 and 64, dropout 0.2 and one logit output [23] gives the attack propensity $$\begin{split}
p_i&=\sigma\big(g_\theta(x_i)\big),\\
\mathcal{L}(\theta)&=-\frac{1}{N}\sum_{i=1}^{N}\Big[w_{+}\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big],\qquad w_{+}=\frac{N_{-}}{N_{+}},
\end{split}
\tag{1}$$ trained with Adam (learning rate $10^{-3}$, batch 512, 20 epochs), keeping the epoch with the best validation ROC-AUC.

## Fuzzy risk bands and production-rule base

Propensity is mapped to three linguistic risk bands [10] with shoulder and triangular membership functions, using $[z]_0^1=\min(1,\max(0,z))$: $$\begin{align}
\mu_L(p)&=\big[(0.45-p)/0.45\big]_0^1, \tag{2}\\
\mu_M(p)&=\begin{cases}\big[(p-0.15)/0.35\big]_0^1, & p<0.5,\\ \big[(0.85-p)/0.35\big]_0^1, & p\ge 0.5,\end{cases} \tag{3}\\
\mu_H(p)&=\big[(p-0.55)/0.45\big]_0^1. \tag{4}
\end{align}$$ The knowledge layer is the production-rule base $\mathcal{R}=\{R_1,R_2,R_3\}$:

|  |  |
|:-|:-----------------------------------------------------------|
| $R_1$: | IF risk IS Low THEN action $=$ Monitor (log the alert; no intervention); |
| $R_2$: | IF risk IS Medium THEN action $=$ Soft-block (automatic containment, e.g., rate-limiting); |
| $R_3$: | IF risk IS High THEN action $=$ Escalate (hand over to a human analyst). |


Each rule fires with strength $\mu_k(p_i)$, and conflicts are resolved by winner-take-all, with ties going to the less severe action: $$a_i = A\big(k_i^{*}\big),\qquad k_i^{*}=\arg\max_{k\in(L,M,H)}\mu_k(p_i),
\tag{5}$$ where $A$ maps $(L,M,H)$ to (Monitor, Soft-block, Escalate). The firing strengths are kept with every decision as its explanation (e.g., “$p=0.60$: $\mu_M=0.71$, $\mu_H=0.11$, Soft-block”). Solving $\mu_L=\mu_M$ and $\mu_M=\mu_H$ shows that Eq. (5) switches action at $p=9/32=0.281$ and $p=23/32=0.719$ (Fig. 1b).

## Cost model and decision metrics

Actions are scored with a relative cost matrix $C(a,y)$ that charges a missed attack $C_{FN}=10$, an analyst escalation 3 units (plus 3 if the alert is benign), and a soft-block 1 unit (plus 1 if benign): $$\begin{aligned}
C(\text{Monitor},y)&=C_{FN}\,y, & C(\text{Soft-block},y)&=1+(1-y),\\
C(\text{Escalate},y)&=3+3(1-y), & C_{FN}&=10.
\end{aligned}
\tag{6}$$ Policies are compared by the mean triage cost $\bar C=\frac1N\sum_i C(a_i,y_i)$, the attack coverage $\sum_i \mathbf{1}[a_i\neq\text{Monitor}]\,y_i/\sum_i y_i$, and the count of unnecessary escalations (benign alerts that are escalated). As an analytical reference we also report the plug-in Bayes rule $$a_i^{B}=\arg\min_{a}\;\big[(1-p_i)\,C(a,0)+p_i\,C(a,1)\big],
\tag{7}$$ which treats $p_i$ as calibrated. Because $C(\text{Soft-block},y)<C(\text{Escalate},y)$ for both classes, Eq. (7) never escalates and soft-blocks whenever $p_i>2/11$. Algorithm 1 summarises the pipeline.

**Algorithm 1.** Fuzzy–neural cost-aware alert triage

**Input:** labelled alerts $\mathcal{D}$; alert stream $\mathcal{S}$; breakpoints of Eqs. (2)–(4); cost matrix $C$, Eq. (6)

**Output:** action $a_i\in\{$Monitor, Soft-block, Escalate$\}$ and firing strengths for every $i\in\mathcal{S}$

Step 1: Split $\mathcal{D}$ into training and validation sets (85/15, stratified); fit scaler and one-hot encoder on training data only

Step 2: Train $g_\theta$ with the weighted loss of Eq. (1); keep the epoch with the best validation ROC-AUC

Step 3: for each alert $i\in\mathcal{S}$ do

Step 4: $p_i\gets\sigma(g_\theta(x_i))$ (neural propensity)

Step 5: compute $\mu_L(p_i),\mu_M(p_i),\mu_H(p_i)$ (Eqs. (2)–(4))

Step 6: fire $R_1$–$R_3$; $a_i\gets$ action of the strongest rule (Eq. (5))

Step 7: log $(p_i,\mu_L,\mu_M,\mu_H,a_i)$ for analyst audit

Step 8: end for

Step 9: **evaluation only:** report $\bar C$, coverage and unnecessary escalations under $C$

# Experimental design

*Data.* We use NSL-KDD [24], a public benchmark of labelled network-connection records, with labels binarised as normal versus attack. For CPU-scale reproducibility, 40,000 training records are sampled (34,000 for training and 6,000 for validation) and the test set is capped at 20,000 records (test attack rate 0.568). The test file contains attack types absent from training, so a train/test distribution shift is part of the task.

*Baselines.* Detectors are logistic regression (class-balanced), gradient boosting [22] (scikit-learn defaults), and the MLP without the rule layer (neural ablation). The proposed system shares the neural propensity, so it has the same predictive metrics by construction, and its contribution must show at the decision level. Decision baselines on the same neural scores are a Monitor/Escalate threshold at 0.5, the same threshold tuned on validation data to minimise cost, and escalation of the top 20% of scores (a fixed analyst-capacity rule). The Bayes rule of Eq. (7) is a reference, not a competitor, because it requires the cost matrix and calibrated probabilities at run time. Learned triage policies (e.g., reinforcement learning [3]) are excluded because they need interaction data and a matched reward protocol.

*Statistics.* On the seed-42 test set we report percentile 95% confidence intervals from 2,000 paired bootstrap resamples of alerts [25] and two-sided Wilcoxon signed-rank tests on per-alert cost differences [26]. The full pipeline (subsample, split, initialisation and training) is then repeated for seeds 42–51 and compared with a paired Wilcoxon test across seeds. Sensitivity is examined by sweeping $C_{FN}\in[2,30]$ and the half-width of the Medium band (0.25–0.45). All numbers are produced by a single script (Data Availability), run with one PyTorch thread so that results are bit-identical between runs.

# Results and discussion

| Model | Acc. | F1 | ROC-AUC | PR-AUC | AUC (10 seeds) |
|:---|:--:|:--:|:--:|:--:|:--:|
| Logistic regression | 0.752 | 0.740 | 0.818 | 0.879 | $0.812\pm0.008$ |
| Gradient boosting | **0.786** | **0.774** | **0.947** | **0.951** | $\mathbf{0.936\pm0.010}$ |
| Neural net (ablation) | 0.773 | 0.766 | 0.908 | 0.923 | $0.900\pm0.007$ |
| Fuzzy–neural (proposed) | 0.773 | 0.766 | 0.908 | 0.923 | $0.900\pm0.007$ |

: Predictive performance on the NSL-KDD test sample. Point estimates are for seed 42; the last column gives the mean $\pm$ SD of ROC-AUC over seeds 42–51. The proposed system uses the same propensity as the neural ablation (its rule layer changes actions, not scores), so its predictive metrics are identical by construction.

| Policy | Scores | $\bar C$ | Coverage | UE | $\Delta\bar C$ | $p$ | 10 seeds |
|:---|:---|:--:|:--:|:--:|:--:|:--:|:--:|
| Threshold 0.5 | NN | 3.264 [3.210, 3.316] | 0.655 [0.646, 0.664] | 623 | $+0.196$ [0.180, 0.213] | $7.3{\times}10^{-124}$ | $3.246\pm0.076$ |
| Top-20% escalate | NN | 4.309 [4.246, 4.372] | 0.348 [0.339, 0.357] | **47** | $+1.242$ [1.196, 1.285] | $<10^{-300}$ | $4.318\pm0.009$ |
| **Fuzzy (proposed)** | NN | **3.068** [3.016, 3.118] | **0.686** [0.677, 0.694] | 603 | – | – | $\mathbf{3.088\pm0.081}$ |
| *Ablation and reference policies* |  |  |  |  |  |  |  |
| Crisp bands$^{c}$ | NN | 3.068 | 0.686 | 603 | 0 | – | – |
| Tuned thr. $t{=}0.53$ | NN | 3.278 [3.224, 3.330] | 0.651 [0.641, 0.659] | 617 | $+0.210$ [0.194, 0.228] | $2.3{\times}10^{-123}$ | – |
| Fuzzy | GB | 3.004 [2.952, 3.055] | 0.676 [0.667, 0.684] | 222 | $-0.064$ [$-0.091$, $-0.037$] | 0.017 | $2.967\pm0.100$ |
| Threshold 0.5 | GB | 3.189 [3.137, 3.241] | 0.644 [0.635, 0.654] | 241 | $+0.122$ [0.093, 0.149] | $6.4{\times}10^{-40}$ | – |
| Bayes rule$^{b}$ | NN | 2.164 [2.113, 2.216] | 0.701 [0.692, 0.709] | 0 | $-0.903$ [$-0.921$, $-0.885$] | $<10^{-300}$ | $2.207\pm0.100$ |

: Decision quality on the seed-42 test sample ($N=20{,}000$). Brackets give paired bootstrap 95% CIs; $\Delta\bar C$ is the mean cost relative to the proposed policy (positive values mean the alternative is more expensive); $p$ is the two-sided Wilcoxon signed-rank test on per-alert costs; the last column gives the mean $\pm$ SD of $\bar C$ over seeds 42–51. UE = unnecessary escalations; NN = neural propensity, GB = gradient-boosting propensity. $^{b}$Reference only: soft-blocks every alert with $p>2/11$ (8,631 of 20,000) and never escalates. $^{c}$Crisp cut-points at $p=9/32$ and $23/32$; actions identical to the proposed policy.

![**Fig. 2.** Predictive and decision evidence. (a) ROC curves (seed 42). (b) Mean triage cost of a Monitor/Escalate threshold policy as the threshold varies, with the proposed, top-20% and Bayes-rule costs shown as horizontal lines. (c) Cost-ratio sensitivity as the missed-attack cost $C_{FN}$ varies (the paper default, 10, is dotted). (d) Paired per-seed costs for seeds 42–51; black markers show the mean $\pm$ SD.](../figures/fig2_roc_and_cost.png)

*Prediction versus decision.* Table 1 and Fig. 2a confirm that gradient boosting is the best detector: it has a higher ROC-AUC than the MLP in all ten seeds ($0.936\pm0.010$ versus $0.900\pm0.007$). We make no detection claim for the proposed system.

*Triage cost and coverage.* Table 2 shows that, on identical neural scores, the rule base reduces the mean triage cost by 0.196 units per alert relative to the 0.5 threshold (95% CI 0.180–0.213; $p=7.3\times10^{-124}$) and by 1.242 relative to top-20% escalation, while raising attack coverage by 0.031 (CI 0.028–0.034) and lowering unnecessary escalations from 623 to 603. The two policies disagree only on the 776 alerts placed in the Medium band. The gain is not an artefact of a poorly chosen threshold: tuning the threshold on validation data to minimise cost selects $t=0.53$ and gives a higher cost (3.278). Across ten full re-runs the rule base is cheaper than both crisp rules in every seed (Fig. 2d; $3.088\pm0.081$ versus $3.246\pm0.076$ and $4.318\pm0.009$; Wilcoxon $p=0.002$ for both) and has higher coverage than the 0.5 threshold in every seed. For a SOC manager, at $C_{FN}=10$ a saving of 0.196 units per alert amounts to about 3,900 cost units on this 20,000-alert sample, with fewer analyst hand-offs.

*Sensitivity.* The advantage over the 0.5 threshold holds for every $C_{FN}$ from 2 to 30 (Fig. 2c). Top-20% escalation becomes cheaper than the rule base when missed attacks are cheap ($C_{FN}=2$: 1.347 versus 1.640; $C_{FN}=3$: 1.718 versus 1.818), because a capacity cap is then the economical choice. Varying the half-width of the Medium band from 0.25 to 0.45 changes the cost monotonically from 3.101 to 3.034 and the coverage from 0.680 to 0.691; every setting remains cheaper than both threshold baselines.

*What drives the gain (ablation).* Two ablations bound the contribution. First, replacing the fuzzy sets with crisp bands at $9/32$ and $23/32$ reproduces every action, so the decision benefit comes from the intermediate Soft-block band defined by the rule base, not from fuzziness itself. The memberships add graded, linguistic explanations and a principled way to set and audit the bands. Second, the Bayes rule of Eq. (7), which reads the cost matrix directly, is much cheaper (2.164; $2.207\pm0.100$ over seeds) because under Eq. (6) soft-blocking dominates escalation. It achieves this by automatically containing 43% of all traffic and never involving an analyst. We report this gap as a managerial finding rather than hide it: when containment is priced this cheaply, the cost matrix, not the detector, dictates the action mix, so organisations must calibrate $C$, including business disruption from false containment, before automating. Fig. 2b also shows that a test-set threshold of 0.10 or lower would beat the rule base (minimum 2.929 at $t=0.01$). Validation tuning cannot find such a threshold because of the train/test shift, so it is not an attainable baseline, but it confirms that the band positions matter.

*Detector-agnostic use and explainability.* Applying the unchanged rules to gradient-boosting scores lowers the cost further (3.004; $2.967\pm0.100$ over seeds) and cuts unnecessary escalations from 603 to 222. The knowledge layer and a stronger detector are therefore complementary. Explainability comes from the rule structure rather than from post-hoc attribution [18,19]: every action is traceable to one named rule and its firing strength, which supports audit trails and analyst trust in human–AI teaming [2].

# Limitations

*Failure modes.* The sensitivity results already show two failure boundaries: below $C_{FN}\approx3$ the capacity-capped top-20% rule is cheaper, and seed 48 is the weakest replicate (cost 3.282, coverage 0.626): its MLP has a typical ROC-AUC (0.900) but the lowest accuracy at 0.5 (0.757), showing that fixed band positions are sensitive to how scores are distributed, not only to how well they rank. More fundamentally, the rule base is not cost-optimal under its own matrix. The Bayes reference shows the size of the gap, and adaptive memberships (e.g., ANFIS-style tuning [11]) or cost-derived band positions are the obvious next step. The present bands were set by expert judgement and not tuned on test data.

*Data and scope.* NSL-KDD (2009) is a dated laboratory benchmark whose traffic does not reflect modern enterprise, cloud or IoT telemetry [27,28]. The training data were subsampled to 40,000 records and the test set was capped at 20,000 for CPU-scale reproducibility, and a single train/test protocol is used. The cost matrix is illustrative, not elicited from a SOC, and coverage counts automatic containment as covering an attack. No live SOC deployment, concept-drift analysis or user study was performed, so the operational value for analysts remains to be validated.

# Conclusion

This Short Communication framed cyber alert triage as a knowledge-based decision problem and proposed a fuzzy–neural layer that turns neural propensity into linguistic risk bands and then into Monitor, Soft-block or Escalate actions through an explicit production-rule base. On NSL-KDD the layer lowered the mean triage cost and raised attack coverage relative to crisp threshold and capacity rules, with tight paired confidence intervals, consistency across ten re-runs and robustness to cost and membership settings. It also improved further when combined with a stronger gradient-boosting detector. Ablations show that the benefit comes from the intermediate containment band and that a cost-derived Bayes rule is cheaper still. For SOC managers, the practical message is to evaluate detectors by the decisions they enable and to make the action rules and their cost assumptions explicit and auditable. Future work will elicit cost matrices from practitioners, learn band positions, and test the layer on recent datasets and live alert streams.

# Data availability

The NSL-KDD dataset is publicly available from the Canadian Institute for Cybersecurity, University of New Brunswick [24]. The code and results (experiment script, result tables in CSV/JSON and figure-generation code) are held in a GitHub repository that will be made public on acceptance; until then they are available to the editor and reviewers on reasonable request.

# Declaration of competing interest

The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Grok Bot, an AI-assisted writing and coding tool, to help with language editing, code refactoring, reference checking and formatting. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. All results, tables and figures were generated by the author’s executable code from the public dataset; no figure was created with generative AI.

# Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

# CRediT authorship contribution statement

**Sai Nithin Erigineni:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing.

# References

[1] S. Tariq, M. Baruwal Chhetri, S. Nepal, C. Paris, Alert fatigue in security operations centres: Research challenges and opportunities, ACM Computing Surveys 57 (2025) 1–38. https://doi.org/10.1145/3723158.

[2] M. Baruwal Chhetri, S. Tariq, R. Singh, F. Jalalvand, C. Paris, S. Nepal, Towards human–AI teaming to mitigate alert fatigue in security operations centres, ACM Transactions on Internet Technology 24 (2024) 1–22. https://doi.org/10.1145/3670009.

[3] X. Wang, X. Yang, X. Liang, X. Zhang, W. Zhang, X. Gong, Combating alert fatigue with AlertPro: Context-aware alert prioritization using reinforcement learning for multi-step attack detection, Computers & Security 137 (2024) 103583. https://doi.org/10.1016/j.cose.2023.103583.

[4] A.L. Buczak, E. Guven, A survey of data mining and machine learning methods for cyber security intrusion detection, IEEE Communications Surveys & Tutorials 18 (2016) 1153–1176. https://doi.org/10.1109/COMST.2015.2494502.

[5] A. Khraisat, I. Gondal, P. Vamplew, J. Kamruzzaman, Survey of intrusion detection systems: Techniques, datasets and challenges, Cybersecurity 2 (2019) 20. https://doi.org/10.1186/s42400-019-0038-7.

[6] T. Su, H. Sun, J. Zhu, S. Wang, Y. Li, BAT: Deep learning methods on network intrusion detection using NSL-KDD dataset, IEEE Access 8 (2020) 29575–29585. https://doi.org/10.1109/ACCESS.2020.2972627.

[7] N. Sun, J. Zhang, P. Rimba, S. Gao, L.Y. Zhang, Y. Xiang, Data-driven cybersecurity incident prediction: A survey, IEEE Communications Surveys & Tutorials 21 (2019) 1744–1772. https://doi.org/10.1109/COMST.2018.2885561.

[8] G. Vial, Understanding digital transformation: A review and a research agenda, The Journal of Strategic Information Systems 28 (2019) 118–144. https://doi.org/10.1016/j.jsis.2019.01.003.

[9] F. Provost, T. Fawcett, Data science and its relationship to big data and data-driven decision making, Big Data 1 (2013) 51–59. https://doi.org/10.1089/big.2013.1508.

[10] L.A. Zadeh, Fuzzy sets, Information and Control 8 (1965) 338–353. https://doi.org/10.1016/S0019-9958(65)90241-X90241-X).

[11] J.-S.R. Jang, ANFIS: Adaptive-network-based fuzzy inference system, IEEE Transactions on Systems, Man, and Cybernetics 23 (1993) 665–685. https://doi.org/10.1109/21.256541.

[12] P.V. de Campos Souza, Fuzzy neural networks and neuro-fuzzy networks: A review the main techniques and applications used in the literature, Applied Soft Computing 92 (2020) 106275. https://doi.org/10.1016/j.asoc.2020.106275.

[13] D. Srilatha, G.K. Shyam, Cloud-based intrusion detection using kernel fuzzy clustering and optimal type-2 fuzzy neural network, Cluster Computing 24 (2021) 2657–2672. https://doi.org/10.1007/s10586-021-03281-9.

[14] J. Liu, W. Yinchai, T.C. Siong, X. Li, L. Zhao, F. Wei, On the combination of adaptive neuro-fuzzy inference system and deep residual network for improving detection rates on intrusion detection, PLOS ONE 17 (2022) e0278819. https://doi.org/10.1371/journal.pone.0278819.

[15] J. Liu, W. Yinchai, T.C. Siong, X. Li, L. Zhao, F. Wei, A hybrid interpretable deep structure based on adaptive neuro-fuzzy inference system, decision tree, and K-means for intrusion detection, Scientific Reports 12 (2022) 20770. https://doi.org/10.1038/s41598-022-23765-x.

[16] C. Elkan, The foundations of cost-sensitive learning, in: Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI), 2001: pp. 973–978.

[17] A. Telikani, N.E. Rudbardeh, S. Soleymanpour, A. Shahbahrami, J. Shen, G. Gaydadjiev, R. Hassanpour, A cost-sensitive machine learning model with multitask learning for intrusion detection in IoT, IEEE Transactions on Industrial Informatics 20 (2024) 3880–3890. https://doi.org/10.1109/TII.2023.3314208.

[18] S.M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, Advances in Neural Information Processing Systems 30 (2017). https://doi.org/10.48550/arXiv.1705.07874.

[19] V.Z. Mohale, I.C. Obagbuwa, A systematic review on the integration of explainable artificial intelligence in intrusion detection systems to enhancing transparency and interpretability in cybersecurity, Frontiers in Artificial Intelligence 8 (2025) 1526221. https://doi.org/10.3389/frai.2025.1526221.

[20] S. AL, S. Sagiroglu, Explainable artificial intelligence models in intrusion detection systems, Engineering Applications of Artificial Intelligence 144 (2025) 110145. https://doi.org/10.1016/j.engappai.2025.110145.

[21] S. Bag, S. Sarkar, I. Bose, Enhancing cybersecurity risk assessment using temporal knowledge graph-based explainable decision support system, Decision Support Systems 198 (2025) 114526. https://doi.org/10.1016/j.dss.2025.114526.


[22] J.H. Friedman, Greedy function approximation: A gradient boosting machine, The Annals of Statistics 29 (2001) 1189–1232. https://doi.org/10.1214/aos/1013203451.

[23] I. Goodfellow, Y. Bengio, A. Courville, Deep learning, MIT Press, Cambridge, MA, 2016.

[24] M. Tavallaee, E. Bagheri, W. Lu, A.A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, in: Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009: pp. 1–6. https://doi.org/10.1109/CISDA.2009.5356528.

[25] B. Efron, R.J. Tibshirani, An introduction to the bootstrap, Chapman & Hall/CRC, New York, 1994. https://doi.org/10.1201/9780429246593.

[26] F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin 1 (1945) 80–83. https://doi.org/10.2307/3001968.

[27] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, A. Hotho, A survey of network-based intrusion detection data sets, Computers & Security 86 (2019) 147–167. https://doi.org/10.1016/j.cose.2019.06.005.

[28] M.A. Al-Garadi, A. Mohamed, A.K. Al-Ali, X. Du, I. Ali, M. Guizani, A survey of machine and deep learning methods for Internet of Things (IoT) security, IEEE Communications Surveys & Tutorials 22 (2020) 1646–1685. https://doi.org/10.1109/COMST.2020.2988293.
