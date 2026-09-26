# Fuzzy–Neural Decision Intelligence for Cyber Alert Triage

Reproducible experiment for a *Knowledge-Based Systems* short communication.

## What this does
1. Loads public **NSL-KDD**
2. Trains baselines: logistic regression, gradient boosting, MLP
3. Applies **fuzzy risk bands** + cost-aware triage actions (Monitor / SoftBlock / Escalate)
4. Writes **Table I** (prediction), **Table II** (decision quality), **Fig. 1** (architecture), **Fig. 2** (ROC + cost)

## Setup
```bash
cd fuzzy-neural-cyber-triage
python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas scikit-learn torch numpy tabulate
python src/experiment.py
```

## Outputs
- `results/table1_predictive.csv`
- `results/table2_decision.csv`
- `results/summary.json`
- `figures/fig1_architecture.png`
- `figures/fig2_roc_and_cost.png`

## Data
NSL-KDD files under `data/` (downloaded from a public GitHub mirror of the dataset).
