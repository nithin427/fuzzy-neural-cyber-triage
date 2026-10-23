# Fuzzy–Neural Decision Intelligence for Cost-Aware Cyber Alert Triage

Code, results and the manuscript for a *Knowledge-Based Systems* (Elsevier) Short Communication.
Author: Sai Nithin Erigineni (DBA, Midwest University). serigineni@midwest.edu

A multilayer perceptron scores each NSL-KDD alert. Fuzzy Low/Medium/High risk bands and a three-rule
production base then turn that score into **Monitor / Soft-block / Escalate**, and every policy is scored under
an explicit cost matrix. The paper's claim is about the decision layer. It does not claim a better detector:
gradient boosting has the best ROC-AUC.

## Reproduce

```bash
git clone https://github.com/nithin427/fuzzy-neural-cyber-triage.git
cd fuzzy-neural-cyber-triage
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch matplotlib tabulate
# NSL-KDD (Canadian Institute for Cybersecurity, UNB: https://www.unb.ca/cic/datasets/nsl.html)
# put KDDTrain+.txt and KDDTest+.txt in data/   (files are not redistributed here)
python src/experiment.py            # full run: seed 42 + seeds 42-51, about 3 min on CPU
python src/experiment.py --quick    # seed 42 only (Tables 1-2 headline numbers and figures)
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
5. Writes the two tables, the two figures (PNG 300 dpi + vector PDF) and the graphical abstract. It also copies the figures into `paper/elsevier/figures/`.

## Outputs
| File | Content |
|---|---|
| `results/table1_predictive.{csv,md}` | Table 1: Acc/F1/ROC-AUC/PR-AUC (seed 42) + 10-seed AUC |
| `results/table2_decision.{csv,md}` | Table 2: cost, coverage, unnecessary escalations, CIs, tests, ablations |
| `results/evidence_extended.csv` | Long-format sensitivity, ablation and per-seed numbers |
| `results/summary.json` | Every number quoted in the paper |
| `figures/fig1_architecture.{png,pdf}` | Fig. 1: pipeline + membership functions |
| `figures/fig2_roc_and_cost.{png,pdf}` | Fig. 2: ROC, cost vs threshold, C_FN sweep, 10-seed paired costs |
| `figures/graphical_abstract.png` | Graphical abstract (2656×1062 px; 1328×531 ratio) |

## Paper
- `paper/elsevier/` holds the self-contained Overleaf/LaTeX submission folder: `main.tex`, `refs.bib`, figures, `highlights.txt`,
  `manuscript.docx` and the compiled `fuzzy-neural-cyber-triage-kbs.pdf`. See its README.
- `paper/manuscript.md` is a Markdown mirror of `main.tex`.
- `paper/CHECKLIST.md` contains the journal-compliance checklist, the reference/DOI report and the open issues.
