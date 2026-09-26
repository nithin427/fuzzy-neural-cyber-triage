#!/usr/bin/env python3
"""Fuzzy-neural decision intelligence for cyber alert triage (NSL-KDD).
Reproducible experiment for a Knowledge-Based Systems short communication.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Cost model (relative units for decision evaluation)
C_MONITOR = 0.0
C_SOFT = 1.0
C_ESC = 3.0
C_FN = 10.0   # miss an attack
C_FP_SOFT = 1.0
C_FP_ESC = 3.0
B_SOFT = 6.0  # benefit if attack and soft-blocked
B_ESC = 9.0


def load_nsl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["y"] = (df["label"].astype(str).str.lower() != "normal").astype(int)
    return df


def fuzzy_memberships(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangular fuzzy sets on propensity in [0,1]."""
    p = np.asarray(p, dtype=float)
    # Low: 1 at 0, 0 at 0.45
    mu_l = np.clip((0.45 - p) / 0.45, 0, 1)
    # Medium: peak at 0.5, zero at 0.15 and 0.85
    mu_m = np.where(p < 0.5, np.clip((p - 0.15) / 0.35, 0, 1), np.clip((0.85 - p) / 0.35, 0, 1))
    # High: 0 at 0.55, 1 at 1.0
    mu_h = np.clip((p - 0.55) / 0.45, 0, 1)
    return mu_l, mu_m, mu_h


def fuzzy_actions(p: np.ndarray) -> np.ndarray:
    mu_l, mu_m, mu_h = fuzzy_memberships(p)
    stack = np.vstack([mu_l, mu_m, mu_h])
    return stack.argmax(axis=0)  # 0 Monitor, 1 SoftBlock, 2 Escalate


def threshold_actions(p: np.ndarray, t: float = 0.5) -> np.ndarray:
    # 0 monitor if p<t else escalate (binary policy)
    return np.where(p < t, 0, 2)


def topk_actions(p: np.ndarray, k_frac: float = 0.2) -> np.ndarray:
    n = len(p)
    k = max(1, int(n * k_frac))
    idx = np.argsort(-p)[:k]
    a = np.zeros(n, dtype=int)
    a[idx] = 2
    return a


def expected_cost(y: np.ndarray, a: np.ndarray) -> dict:
    """Compute decision-quality metrics."""
    y = np.asarray(y)
    a = np.asarray(a)
    cost = 0.0
    for yi, ai in zip(y, a):
        if ai == 0:
            cost += C_MONITOR + (C_FN if yi == 1 else 0.0)
        elif ai == 1:
            cost += C_SOFT + (0.0 if yi == 1 else C_FP_SOFT)
        else:
            cost += C_ESC + (0.0 if yi == 1 else C_FP_ESC)
    attacks = y == 1
    covered = ((a >= 1) & attacks).sum() / max(attacks.sum(), 1)
    unnec_esc = ((a == 2) & (y == 0)).sum()
    return {
        "expected_cost": float(cost / len(y)),
        "attack_coverage": float(covered),
        "unnecessary_escalations": int(unnec_esc),
        "n_monitor": int((a == 0).sum()),
        "n_soft": int((a == 1).sum()),
        "n_escalate": int((a == 2).sum()),
    }


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, epochs=25, lr=1e-3, batch=512):
    device = torch.device("cpu")
    model = MLP(X_train.shape[1]).to(device)
    # class weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)

    xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = y_val

    best_auc, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).cpu().numpy()
        auc = roc_auc_score(yv, pv)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba_mlp(model, X):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32)
    return torch.sigmoid(model(x)).numpy()


def metrics_pack(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.0, 1.8, 1.2, "Alert\nfeatures\n(NSL-KDD)"),
        (2.5, 1.0, 1.8, 1.2, "Neural\npropensity\nMLP"),
        (4.7, 1.0, 1.8, 1.2, "Fuzzy risk\nbands\nL / M / H"),
        (6.9, 1.0, 2.4, 1.2, "Cost-aware\ntriage actions\nMonitor / Soft / Escalate"),
    ]
    for x, y, w, h, t in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor="#e8f1fb", edgecolor="#1f4e79", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
    for x0 in [2.1, 4.3, 6.5]:
        ax.annotate("", xy=(x0 + 0.35, 1.6), xytext=(x0, 1.6),
                    arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.5))
    ax.set_title("Fig. 1. Fuzzy–neural decision intelligence pipeline for cyber alert triage", fontsize=11, pad=8)
    fig.tight_layout()
    out = FIG / "fig1_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def draw_decision_figure(y_test, curves: dict):
    """Fig 2: ROC + expected cost vs threshold for fuzzy vs baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    for name, p in curves.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = roc_auc_score(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    thresholds = np.linspace(0.05, 0.95, 19)
    # cost curves for neural probs under threshold policy vs fuzzy fixed policy cost as horizontal
    p_nn = curves["Neural (ablation)"]
    thr_costs = []
    for t in thresholds:
        a = threshold_actions(p_nn, t)
        thr_costs.append(expected_cost(y_test, a)["expected_cost"])
    ax.plot(thresholds, thr_costs, label="Neural + hard threshold", color="#c0392b")
    fuzzy_cost = expected_cost(y_test, fuzzy_actions(p_nn))["expected_cost"]
    ax.axhline(fuzzy_cost, color="#1f4e79", linestyle="--", label=f"Fuzzy–neural policy ({fuzzy_cost:.3f})")
    topk_cost = expected_cost(y_test, topk_actions(p_nn, 0.2))["expected_cost"]
    ax.axhline(topk_cost, color="#27ae60", linestyle=":", label=f"Top-20% escalate ({topk_cost:.3f})")
    ax.set_xlabel("Decision threshold on propensity")
    ax.set_ylabel("Mean expected decision cost")
    ax.set_title("Decision cost comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Fig. 2. Predictive ROC and cost-aware triage comparison", fontsize=11)
    fig.tight_layout()
    out = FIG / "fig2_roc_and_cost.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main():
    train_df = load_nsl(DATA / "KDDTrain+.txt")
    test_df = load_nsl(DATA / "KDDTest+.txt")

    # subsample for speed while keeping signal (still scientific; document in paper)
    train_df = train_df.sample(n=min(40000, len(train_df)), random_state=SEED)
    # keep full test or cap
    if len(test_df) > 20000:
        test_df = test_df.sample(n=20000, random_state=SEED)

    feature_cols = [c for c in COLUMNS if c not in ("label", "difficulty")]
    cat_cols = ["protocol_type", "service", "flag"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X_train_raw = train_df[feature_cols]
    y_train = train_df["y"].values
    X_test_raw = test_df[feature_cols]
    y_test = test_df["y"].values

    # validation split from train
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_raw, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    X_te_p = pre.transform(X_test_raw)

    results = {}

    # Logistic regression
    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    lr.fit(X_tr_p, y_tr)
    p_lr = lr.predict_proba(X_te_p)[:, 1]
    results["Logistic regression"] = metrics_pack(y_test, p_lr)

    # Gradient boosting
    gb = GradientBoostingClassifier(random_state=SEED)
    gb.fit(X_tr_p, y_tr)
    p_gb = gb.predict_proba(X_te_p)[:, 1]
    results["Gradient boosting"] = metrics_pack(y_test, p_gb)

    # Neural ablation
    model = train_mlp(X_tr_p, y_tr, X_va_p, y_va, epochs=20)
    p_nn = predict_proba_mlp(model, X_te_p)
    results["Neural net (ablation)"] = metrics_pack(y_test, p_nn)

    # Proposed: same neural scores + fuzzy decision (prediction metrics identical to neural;
    # decision metrics differ — report prediction of neural component + mark proposed)
    results["Fuzzy–neural (proposed)"] = metrics_pack(y_test, p_nn)

    # Decision table
    decision = {
        "Prob. threshold 0.5": expected_cost(y_test, threshold_actions(p_nn, 0.5)),
        "Top-20% by score": expected_cost(y_test, topk_actions(p_nn, 0.2)),
        "Fuzzy–cost (proposed)": expected_cost(y_test, fuzzy_actions(p_nn)),
    }

    # Tables
    table1 = pd.DataFrame(results).T
    table1.index.name = "Model"
    table1.to_csv(RES / "table1_predictive.csv")
    table1.to_markdown(RES / "table1_predictive.md")

    table2 = pd.DataFrame(decision).T
    table2.index.name = "Policy"
    table2.to_csv(RES / "table2_decision.csv")
    table2.to_markdown(RES / "table2_decision.md")

    fig1 = draw_architecture()
    fig2 = draw_decision_figure(
        y_test,
        {
            "Logistic regression": p_lr,
            "Gradient boosting": p_gb,
            "Neural (ablation)": p_nn,
        },
    )

    summary = {
        "seed": SEED,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_test": int(len(y_test)),
        "attack_rate_test": float(y_test.mean()),
        "table1": results,
        "table2": decision,
        "figures": {"fig1": fig1, "fig2": fig2},
        "notes": [
            "Proposed model shares neural propensity with ablation; novelty is fuzzy+cost decision layer.",
            "Train subsampled to 40k for <2h CPU reproducibility; document in paper limitations.",
        ],
    }
    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(table1.to_string())
    print()
    print(table2.to_string())
    print("FIG1", fig1)
    print("FIG2", fig2)
    print("Wrote results to", RES)


if __name__ == "__main__":
    main()
