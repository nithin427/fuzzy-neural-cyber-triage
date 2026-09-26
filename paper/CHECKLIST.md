# Instructor-style final checklist (Topic 2: fuzzy–neural cyber triage)

| Check | Status | Notes |
|-------|--------|-------|
| Research gap | Locked | Accuracy-centric IDS vs missing score→fuzzy knowledge→cost-aware actions loop |
| Novelty narrative | Locked | Decision intelligence + linguistic bands; not SOTA AUC claim |
| Formal model | Locked | MLP + triangular fuzzy + Monitor/Soft/Escalate costs |
| Baselines | Locked | LR, GB, neural ablation, proposed |
| Ablations | Locked | Neural-only; fuzzy on logistic left as extension; no-cost vs cost in Table 2 proxies |
| Sensitivity analysis | Partial | Cost-vs-threshold curve in Fig. 2; full cost-grid sweep optional next |
| Scalability | Locked for short communication | Documented 40k/20k subsample; full NSL-KDD easy extension |
| Claims vs evidence | Locked | Boosting wins AUC; fuzzy–neural wins cost/coverage |
| Limitations | Locked | Section 6 |
| References | Near lock | 20–25 slots; finish DOI-exact rows marked select-one before submission |
| Figures (exactly 2) | Locked | Fig.1 architecture; Fig.2 ROC+cost from code |
| Tables (exactly 2) | Locked | Table 1 predictive; Table 2 decision |
| More experiments | No (for now) | Unless instructor demands phishing transfer study |
| More scientific rewriting | Optional | One more ChatGPT regression pass if desired |
| Repo dump | Done | https://github.com/nithin427/fuzzy-neural-cyber-triage |
