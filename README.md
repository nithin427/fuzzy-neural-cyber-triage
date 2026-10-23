# Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage

Code, results and the manuscript for a *Knowledge-Based Systems* (Elsevier) Short Communication.
Author: Sai Nithin Erigineni (DBA, Midwest University). serigineni@midwest.edu

A multilayer perceptron scores each NSL-KDD alert. Fuzzy Low/Medium/High risk bands and a three-rule
production base then turn that score into **Monitor / Soft-block / Escalate**, and every policy is scored under
an explicit cost matrix. The paper's claim is about the decision layer. It does not claim a better detector:
gradient boosting has the best ROC-AUC.

## Data and code availability
This repository is public. The code, result tables and figure-generation scripts are available at https://github.com/nithin427/fuzzy-neural-cyber-triage.
The NSL-KDD dataset is publicly available from the Canadian Institute for Cybersecurity, University of New Brunswick
(https://www.unb.ca/cic/datasets/nsl.html; Tavallaee et al., CISDA 2009, https://doi.org/10.1109/CISDA.2009.5356528).
The data files are not redistributed in this repository.

## Reproduce

```bash
git clone https://github.com/nithin427/fuzzy-neural-cyber-triage.git
cd fuzzy-neural-cyber-triage
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch matplotlib tabulate
# NSL-KDD (Canadian Institute for Cybersecurity, UNB: https://www.unb.ca/cic/datasets/nsl.html)
# put KDDTrain+.txt and KDDTest+.txt in data/   (files are not redistributed here)
python src/experiment.py            # full run: seed 42 + seeds 42-51, about 3 min on CPU
python src/experiment.py --quick    # seed 42 only (Table 2 headline numbers and figures)
python src/make_figures.py          # redraw Fig. 1 and Fig. 2 from results/ (no retraining)
```

Data files used (sha256):
`KDDTrain+.txt 1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`,
`KDDTest+.txt fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`.

The script fixes all seeds and uses one PyTorch thread (`torch.set_num_threads(1)`). Multi-threaded
CPU kernels changed the MLP numbers in the third decimal between runs; with one thread, reruns are bit-identical.
Tested with Python 3.13, torch 2.14.1+cpu, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0.

## What the script does (`src/experiment.py`, one file)
1. Subsamples NSL-KDD (40,000 train records split 85/15 into train/validation, stratified; 20,000 test records).
   Encoders are fitted on the training split only.
2. Trains logistic regression, gradient boosting and a 128-64-1 MLP (class-weighted BCE).
3. Applies the fuzzy memberships, Eqs. (2)–(4), and the rule base R1–R3, Eq. (5), then scores policies with the cost matrix, Eq. (6).
4. Evidence:
   - paired bootstrap 95% CIs (2,000 resamples) and per-alert Wilcoxon tests
   - full re-runs for seeds 42–51 with a Wilcoxon test across seeds
   - a missed-attack cost sweep (C_FN = 2–30) and a Medium-band width sweep
   - ablations: crisp bands, a validation-tuned threshold, the rules on GB scores, the Bayes cost rule, and a test-oracle threshold (reported as invalid)
5. Writes the result tables (merged into Table 2 of the paper) and the graphical abstract, then calls `src/make_figures.py`, which draws Fig. 1 and Fig. 2 (PNG 300 dpi + vector PDF) from `results/summary.json` and `results/table2_decision.csv` and copies them into `paper/elsevier/figures/`.

## Outputs
| File | Content |
|---|---|
| `results/table1_predictive.{csv,md}` | Table 2, Panel A: Acc/F1/ROC-AUC/PR-AUC (seed 42) + 10-seed AUC |
| `results/table2_decision.{csv,md}` | Table 2, Panels B1-B2: cost, coverage, unnecessary escalations, CIs, tests, ablations |
| `results/evidence_extended.csv` | Long-format sensitivity, ablation and per-seed numbers |
| `results/summary.json` | Every number quoted in the paper |
| `figures/fig1_decision_bands.{png,pdf}` | Fig. 1: membership functions, rule bands R1-R3 with alert counts, switch points and Bayes threshold |
| `figures/fig2_cost_sensitivity.{png,pdf}` | Fig. 2: mean triage cost vs missed-attack cost C_FN with 95% CIs at C_FN = 10 |
| `figures/graphical_abstract.png` | Graphical abstract (2656×1062 px; 1328×531 ratio) |

## Paper
- `paper/elsevier/` holds the self-contained Overleaf/LaTeX submission folder: `main.tex`, `refs.bib`, figures, `highlights.txt`,
  `manuscript.docx` and the compiled `fuzzy-neural-cyber-triage-kbs.pdf`. See its README.
- `paper/manuscript.md` is a Markdown mirror of `main.tex`.
- Table 1 of the paper (related work by design property, with citations in the cells) is written directly in `main.tex`.
- `paper/CHECKLIST.md` contains the journal-compliance checklist, the reference/DOI report and the open issues.
