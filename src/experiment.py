#!/usr/bin/env python3
"""Fuzzy-neural decision intelligence for cost-aware cyber alert triage (NSL-KDD).

Single reproducible script behind the Knowledge-Based Systems short communication.
It (i) trains the propensity models, (ii) applies the fuzzy production-rule triage
layer, (iii) writes the predictive and decision result tables (merged into Table 2 of the
paper) and calls make_figures.py for Fig. 1 / Fig. 2, and (iv) produces the
robustness evidence reported in the paper: paired bootstrap 95% CIs, Wilcoxon tests,
a 10-seed repetition, a cost-ratio sweep, a membership-width sweep and decision-layer
ablations. The seed-42 headline run is identical to the original pipeline.

Usage:  python src/experiment.py            (full run: seed 42 + seeds 42-51, about 3 min wall-clock;
                                             2 min 47-50 s measured on an 8-vCPU Intel Xeon)
        python src/experiment.py --quick    (seed 42 only; skips the 10-seed loop)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
RES = ROOT / "results"
PAPER_FIG = ROOT / "paper" / "elsevier" / "figures"


def _rel(path) -> str:
    """Path relative to the repository root (keeps machine paths out of summary.json)."""
    return Path(path).resolve().relative_to(ROOT).as_posix()


SEED = 42                        # headline run (Table 2 point estimates)
SEEDS = list(range(42, 52))      # 10-seed repetition (subsample, split, model init)
N_TRAIN, N_TEST_CAP = 40_000, 20_000
N_BOOT = 2000                    # paired bootstrap resamples

# Multi-threaded CPU reductions in PyTorch are not bit-reproducible; one intra-op thread
# makes every run identical (and reproduces the locked seed-42 numbers exactly).
torch.set_num_threads(1)

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
CAT_COLS = ["protocol_type", "service", "flag"]

# ---------------------------------------------------------------------------
# Cost model (relative units), Eq. (6) of the paper.
#   Monitor : C_MONITOR + C_FN * y            -> 0 (benign), 10 (missed attack)
#   SoftBlk : C_SOFT + C_FP_SOFT * (1 - y)    -> 2 (benign), 1 (attack)
#   Escalate: C_ESC + C_FP_ESC * (1 - y)      -> 6 (benign), 3 (attack)
# ---------------------------------------------------------------------------
C_MONITOR, C_SOFT, C_ESC = 0.0, 1.0, 3.0
C_FN = 10.0
C_FP_SOFT, C_FP_ESC = 1.0, 3.0
ACTIONS = ("Monitor", "Soft-block", "Escalate")

# Triangular membership breakpoints on propensity p, Eqs. (2)-(4).
MF = {"low_end": 0.45, "med_lo": 0.15, "med_peak": 0.50, "med_hi": 0.85, "high_start": 0.55}

COL = {"lr": "#7f7f7f", "gb": "#2a9d8f", "nn": "#1f4e79", "thr": "#c0392b",
       "top": "#e69f00", "fz": "#1f4e79", "bayes": "#6a3d9a"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------
def load_nsl(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["y"] = (df["label"].astype(str).str.lower() != "normal").astype(int)
    return df


class MLP(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, epochs=20, lr=1e-3, batch=512):
    """Class-weighted BCE; keeps the epoch with the best validation ROC-AUC."""
    model = MLP(X_train.shape[1])
    pos = y_train.sum()
    pos_weight = torch.tensor([(len(y_train) - pos) / max(pos, 1)], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch, shuffle=True)
    xv = torch.tensor(X_val, dtype=torch.float32)
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(y_val, torch.sigmoid(model(xv)).numpy())
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_proba_mlp(model, X):
    model.eval()
    return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def metrics_pack(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def run_pipeline(seed: int, train_full: pd.DataFrame, test_full: pd.DataFrame) -> dict:
    """One complete replicate: subsample, split, preprocess, train LR / GB / MLP."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_df = train_full.sample(n=min(N_TRAIN, len(train_full)), random_state=seed)
    test_df = test_full.sample(n=N_TEST_CAP, random_state=seed) if len(test_full) > N_TEST_CAP else test_full
    feats = [c for c in COLUMNS if c not in ("label", "difficulty")]
    nums = [c for c in feats if c not in CAT_COLS]
    X_tr, X_va, y_tr, y_va = train_test_split(
        train_df[feats], train_df["y"].values, test_size=0.15, random_state=seed,
        stratify=train_df["y"].values)
    pre = ColumnTransformer([
        ("num", StandardScaler(), nums),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)])
    X_tr_p, X_va_p = pre.fit_transform(X_tr), pre.transform(X_va)
    X_te_p, y_te = pre.transform(test_df[feats]), test_df["y"].values

    lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed).fit(X_tr_p, y_tr)
    gb = GradientBoostingClassifier(random_state=seed).fit(X_tr_p, y_tr)
    mlp = train_mlp(X_tr_p, y_tr, X_va_p, y_va, epochs=20)
    return {
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
        "y_val": y_va, "y_test": y_te,
        "p_val": {"nn": predict_proba_mlp(mlp, X_va_p)},
        "p_test": {"lr": lr.predict_proba(X_te_p)[:, 1], "gb": gb.predict_proba(X_te_p)[:, 1],
                   "nn": predict_proba_mlp(mlp, X_te_p)},
    }


# ---------------------------------------------------------------------------
# Fuzzy knowledge layer and triage policies
# ---------------------------------------------------------------------------
def fuzzy_memberships(p, low_end=MF["low_end"], med_lo=MF["med_lo"], med_peak=MF["med_peak"],
                      med_hi=MF["med_hi"], high_start=MF["high_start"]):
    """Triangular / shoulder memberships, Eqs. (2)-(4)."""
    p = np.asarray(p, dtype=float)
    mu_l = np.clip((low_end - p) / low_end, 0, 1)
    mu_m = np.where(p < med_peak, np.clip((p - med_lo) / (med_peak - med_lo), 0, 1),
                    np.clip((med_hi - p) / (med_hi - med_peak), 0, 1))
    mu_h = np.clip((p - high_start) / (1.0 - high_start), 0, 1)
    return mu_l, mu_m, mu_h


def fuzzy_actions(p, **mf):
    """Production rules R1-R3: dominant band -> action (0 Monitor, 1 Soft-block, 2 Escalate)."""
    return np.vstack(fuzzy_memberships(p, **mf)).argmax(axis=0)


def crisp_cutpoints(**mf):
    """Propensity values at which the dominant fuzzy band changes (numerical)."""
    grid = np.linspace(0, 1, 100_001)
    a = fuzzy_actions(grid, **mf)
    idx = np.flatnonzero(np.diff(a)) + 1
    return [float(grid[i]) for i in idx]


def threshold_actions(p, t=0.5):
    return np.where(np.asarray(p) < t, 0, 2)


def topk_actions(p, k_frac=0.2):
    a = np.zeros(len(p), dtype=int)
    a[np.argsort(-np.asarray(p))[:max(1, int(len(p) * k_frac))]] = 2
    return a


def cost_matrix(c_fn=C_FN):
    """Rows: actions; columns: y = 0 (benign), y = 1 (attack)."""
    return np.array([[C_MONITOR, C_MONITOR + c_fn],
                     [C_SOFT + C_FP_SOFT, C_SOFT],
                     [C_ESC + C_FP_ESC, C_ESC]])


def bayes_actions(p, c_fn=C_FN):
    """Minimum expected-cost action if p were a calibrated probability, Eq. (7)."""
    M = cost_matrix(c_fn)
    p = np.asarray(p, dtype=float)
    exp_cost = M[:, [0]] * (1 - p) + M[:, [1]] * p
    return exp_cost.argmin(axis=0)


def alert_costs(y, a, c_fn=C_FN):
    return cost_matrix(c_fn)[np.asarray(a), np.asarray(y)]


def expected_cost(y, a, c_fn=C_FN) -> dict:
    y, a = np.asarray(y), np.asarray(a)
    attacks = y == 1
    return {
        "expected_cost": float(alert_costs(y, a, c_fn).mean()),
        "attack_coverage": float(((a >= 1) & attacks).sum() / max(attacks.sum(), 1)),
        "unnecessary_escalations": int(((a == 2) & (y == 0)).sum()),
        "n_monitor": int((a == 0).sum()),
        "n_soft": int((a == 1).sum()),
        "n_escalate": int((a == 2).sum()),
    }


def tune_threshold(y_val, p_val):
    """Monitor/Escalate threshold minimising validation cost (fair crisp baseline)."""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = [alert_costs(y_val, threshold_actions(p_val, t)).mean() for t in grid]
    return float(grid[int(np.argmin(costs))])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def paired_bootstrap(y, a_ref, a_alt, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over alerts; returns CIs for both policies and the paired difference."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    c_ref, c_alt = alert_costs(y, a_ref), alert_costs(y, a_alt)
    cov_ref = ((np.asarray(a_ref) >= 1) & (y == 1)).astype(float)
    cov_alt = ((np.asarray(a_alt) >= 1) & (y == 1)).astype(float)
    n = len(y)
    stats = np.empty((n_boot, 6))
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        att = max(y[i].sum(), 1)
        stats[b] = (c_ref[i].mean(), c_alt[i].mean(), c_alt[i].mean() - c_ref[i].mean(),
                    cov_ref[i].sum() / att, cov_alt[i].sum() / att,
                    (cov_alt[i].sum() - cov_ref[i].sum()) / att)
    lo, hi = np.percentile(stats, [2.5, 97.5], axis=0)
    d_cost = stats[:, 2]
    p_boot = float(min(1.0, 2 * min((d_cost <= 0).mean(), (d_cost >= 0).mean())))
    nonzero = int((c_alt != c_ref).sum())
    w = wilcoxon(c_alt, c_ref, zero_method="wilcox") if nonzero else None
    keys = ["cost_ref", "cost_alt", "d_cost", "cov_ref", "cov_alt", "d_cov"]
    out = {f"{k}_ci": [float(lo[j]), float(hi[j])] for j, k in enumerate(keys)}
    out.update({"p_bootstrap_cost": p_boot if p_boot > 0 else 1.0 / n_boot,  # 1/B = "p < 1/B"
                "p_bootstrap_is_upper_bound": bool(p_boot == 0),
                "wilcoxon_stat": None if w is None else float(w.statistic),
                "p_wilcoxon_cost": None if w is None else float(w.pvalue),
                "n_nonzero_pairs": nonzero})
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, stem: str) -> list[str]:
    outs = []
    for ext in ("png", "pdf"):
        out = FIG / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outs.append(str(out))
    plt.close(fig)
    return outs


def draw_graphical_abstract(t2, coverage) -> str:
    """Graphical abstract, 1328 x 531 px ratio (2.5:1), rendered at 2x."""
    fig = plt.figure(figsize=(13.28, 5.31))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(12.5, 9.3, "Fuzzy-neural decision intelligence for cost-aware cyber alert triage",
            ha="center", fontsize=17, weight="bold", color="#1f4e79")
    steps = [("SOC alerts", "NSL-KDD\nconnection\nrecords"), ("Neural MLP", "attack\npropensity\n$p\\in[0,1]$"),
             ("Fuzzy bands", "Low / Medium /\nHigh membership\ndegrees"),
             ("Rule base", "Monitor /\nSoft-block /\nEscalate")]
    for i, (h, sub_t) in enumerate(steps):
        x = 0.4 + i * 3.25
        ax.add_patch(FancyBboxPatch((x, 4.3), 2.7, 3.6, boxstyle="round,pad=0.05,rounding_size=0.25",
                                    facecolor="#e8f1fb" if i < 3 else "#fdf2e9",
                                    edgecolor="#1f4e79" if i < 3 else "#a04000", lw=2))
        ax.text(x + 1.35, 7.35, h, ha="center", va="center", fontsize=13.5, weight="bold")
        ax.text(x + 1.35, 5.6, sub_t, ha="center", va="center", fontsize=11.5, linespacing=1.35)
        if i < 3:
            ax.annotate("", xy=(x + 3.22, 6.1), xytext=(x + 2.78, 6.1),
                        arrowprops=dict(arrowstyle="-|>", color="#1f4e79", lw=2.5))
    sub = fig.add_axes([0.66, 0.24, 0.31, 0.56])
    names = ["Threshold 0.5", "Top-20%", "Fuzzy-cost"]
    vals = [t2[k]["expected_cost"] for k in ("Prob. threshold 0.5", "Top-20% by score", "Fuzzy–cost (proposed)")]
    bars = sub.barh(names, vals, color=[COL["thr"], COL["top"], COL["fz"]])
    for b, v, c in zip(bars, vals, coverage):
        sub.text(v + 0.08, b.get_y() + b.get_height() / 2, f"{v:.3f}\ncov. {c:.3f}", va="center", fontsize=11)
    sub.set_xlim(0, max(vals) * 1.45)
    sub.invert_yaxis()
    sub.set_xlabel("Mean triage cost per alert (lower = better)", fontsize=11)
    sub.tick_params(labelsize=11)
    ax.text(6.95, 2.85, "A decision layer, not a new detector: gradient boosting keeps the best ROC-AUC (0.947).",
            ha="center", fontsize=10.5, style="italic")
    ax.text(6.95, 2.05, "The rule base lowers triage cost and raises attack coverage vs. crisp score-to-action rules",
            ha="center", fontsize=10.5, style="italic")
    ax.text(6.95, 1.25, "(NSL-KDD, seed 42; 10-seed and bootstrap evidence in the paper).",
            ha="center", fontsize=10.5, style="italic")
    out = FIG / "graphical_abstract.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def fmt_ci(ci, nd=3):
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def main(quick: bool = False):
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    train_full = load_nsl(DATA / "KDDTrain+.txt")
    test_full = load_nsl(DATA / "KDDTest+.txt")
    long_rows = []  # one tidy CSV with all supplementary evidence

    # ---- 1. Headline run (seed 42): unchanged Table 2 point estimates ----
    run = run_pipeline(SEED, train_full, test_full)
    y, P = run["y_test"], run["p_test"]
    p_nn = P["nn"]
    table1 = {"Logistic regression": metrics_pack(y, P["lr"]),
              "Gradient boosting": metrics_pack(y, P["gb"]),
              "Neural net (ablation)": metrics_pack(y, p_nn),
              "Fuzzy–neural (proposed)": metrics_pack(y, p_nn)}
    acts = {"Prob. threshold 0.5": threshold_actions(p_nn, 0.5),
            "Top-20% by score": topk_actions(p_nn, 0.2),
            "Fuzzy–cost (proposed)": fuzzy_actions(p_nn)}
    table2 = {k: expected_cost(y, a) for k, a in acts.items()}

    # ---- 2. Paired inference vs the proposed policy (seed 42 test set) ----
    ref = acts["Fuzzy–cost (proposed)"]
    inference = {k: paired_bootstrap(y, ref, a) for k, a in acts.items() if k != "Fuzzy–cost (proposed)"}
    own = paired_bootstrap(y, ref, ref)
    ci = {"Fuzzy–cost (proposed)": (own["cost_ref_ci"], own["cov_ref_ci"])}
    for k, r in inference.items():
        ci[k] = (r["cost_alt_ci"], r["cov_alt_ci"])

    # ---- 3. Decision-layer ablations / reference policies (seed 42) ----
    cuts = crisp_cutpoints()
    t_val = tune_threshold(run["y_val"], run["p_val"]["nn"])
    abl_acts = {
        f"Crisp bands at fuzzy switch points ({cuts[0]:.3f}/{cuts[1]:.3f})":
            np.digitize(p_nn, cuts),
        f"Validation-tuned threshold (t={t_val:.2f})": threshold_actions(p_nn, t_val),
        "Bayes cost rule (reference)": bayes_actions(p_nn),
        "Fuzzy–cost on GB scores": fuzzy_actions(P["gb"]),
        "Threshold 0.5 on GB scores": threshold_actions(P["gb"], 0.5),
    }
    ablation = {k: expected_cost(y, a) for k, a in abl_acts.items()}
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    test_thr_costs = [alert_costs(y, threshold_actions(p_nn, t)).mean() for t in grid]
    test_oracle = {"threshold": float(grid[int(np.argmin(test_thr_costs))]),
                   "expected_cost": float(np.min(test_thr_costs)),
                   "thresholds_beating_fuzzy": [float(t) for t, c in zip(grid, test_thr_costs)
                                                if c < table2["Fuzzy–cost (proposed)"]["expected_cost"]]}
    ablation_inf = {k: paired_bootstrap(y, ref, a) for k, a in abl_acts.items()}
    identical = bool(np.array_equal(abl_acts[next(iter(abl_acts))], ref))
    M = cost_matrix()
    bayes_soft_threshold = float((M[1, 0] - M[0, 0]) / ((M[1, 0] - M[0, 0]) + (M[0, 1] - M[1, 1])))

    # ---- 4. Cost-ratio sweep (C_FN), seed 42 neural scores ----
    sweep = []
    sweep_pols = {"Prob. threshold 0.5": lambda c: threshold_actions(p_nn, 0.5),
                  "Top-20% by score": lambda c: topk_actions(p_nn, 0.2),
                  "Fuzzy–cost (proposed)": lambda c: fuzzy_actions(p_nn),
                  "Bayes cost rule (reference)": lambda c: bayes_actions(p_nn, c)}
    for c_fn in [2, 3, 5, 7.5, 10, 15, 20, 30]:
        for pol, f in sweep_pols.items():
            r = expected_cost(y, f(c_fn), c_fn)
            sweep.append({"c_fn": c_fn, "policy": pol, **r})
            long_rows.append({"analysis": "cost_ratio_sweep", "setting": f"C_FN={c_fn}", "policy": pol,
                              "expected_cost": r["expected_cost"], "attack_coverage": r["attack_coverage"]})
    sw = pd.DataFrame(sweep).pivot(index="c_fn", columns="policy", values="expected_cost")
    fz_beats_thr = [float(c) for c in sw.index if sw.loc[c, "Fuzzy–cost (proposed)"] < sw.loc[c, "Prob. threshold 0.5"]]

    # ---- 5. Membership-width sensitivity (Medium half-width), seed 42 ----
    mf_sens = []
    for hw in [0.25, 0.30, 0.35, 0.40, 0.45]:
        mf = dict(med_lo=0.5 - hw, med_hi=0.5 + hw)
        r = expected_cost(y, fuzzy_actions(p_nn, **mf))
        mf_sens.append({"medium_half_width": hw, "switch_points": crisp_cutpoints(**mf), **r})
        long_rows.append({"analysis": "membership_width", "setting": f"half_width={hw}",
                          "policy": "Fuzzy–cost (proposed)", "expected_cost": r["expected_cost"],
                          "attack_coverage": r["attack_coverage"]})

    # ---- 6. Ten-seed repetition ----
    seed_rows = []
    for s in ([SEED] if quick else SEEDS):
        rr = run if s == SEED else run_pipeline(s, train_full, test_full)
        ys, ps = rr["y_test"], rr["p_test"]
        row = {"seed": s}
        for key in ("lr", "gb", "nn"):
            m = metrics_pack(ys, ps[key])
            row[f"auc_{key}"], row[f"acc_{key}"] = m["roc_auc"], m["accuracy"]
        for key, a in [("thr05", threshold_actions(ps["nn"])), ("top20", topk_actions(ps["nn"])),
                       ("fuzzy", fuzzy_actions(ps["nn"])), ("bayes", bayes_actions(ps["nn"])),
                       ("fuzzy_gb", fuzzy_actions(ps["gb"]))]:
            r = expected_cost(ys, a)
            row[key], row[f"cov_{key}"] = r["expected_cost"], r["attack_coverage"]
        seed_rows.append(row)
        print(f"seed {s}: " + ", ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "seed"), flush=True)
    sr = pd.DataFrame(seed_rows)
    for _, row in sr.iterrows():
        for k in sr.columns:
            if k != "seed":
                long_rows.append({"analysis": "multi_seed", "setting": f"seed={int(row.seed)}", "policy": k,
                                  "expected_cost": row[k] if not k.startswith(("auc", "acc", "cov")) else np.nan,
                                  "attack_coverage": row[k] if k.startswith("cov") else np.nan,
                                  "value": row[k]})

    def agg(col):
        return {"mean": float(sr[col].mean()), "sd": float(sr[col].std(ddof=1)) if len(sr) > 1 else 0.0,
                "min": float(sr[col].min()), "max": float(sr[col].max())}
    multi = {"seeds": [int(s) for s in sr.seed], "per_seed": seed_rows,
             "summary": {c: agg(c) for c in sr.columns if c != "seed"}}
    if len(sr) >= 5:
        for other in ("thr05", "top20", "bayes"):
            w = wilcoxon(sr["fuzzy"], sr[other])
            multi[f"wilcoxon_fuzzy_vs_{other}"] = {
                "p": float(w.pvalue), "statistic": float(w.statistic),
                "fuzzy_lower_in_n_seeds": int((sr["fuzzy"] < sr[other]).sum()), "n": int(len(sr))}
        multi["gb_auc_higher_than_nn_in_n_seeds"] = int((sr["auc_gb"] > sr["auc_nn"]).sum())
        multi["fuzzy_cov_higher_than_thr05_in_n_seeds"] = int((sr["cov_fuzzy"] > sr["cov_thr05"]).sum())

    # ---- Tables (CSV + Markdown) ----
    t1 = pd.DataFrame(table1).T
    t1.index.name = "Model"
    auc_key = {"Logistic regression": "auc_lr", "Gradient boosting": "auc_gb",
               "Neural net (ablation)": "auc_nn", "Fuzzy–neural (proposed)": "auc_nn"}
    t1["roc_auc_10seed_mean"] = [multi["summary"][auc_key[m]]["mean"] for m in t1.index]
    t1["roc_auc_10seed_sd"] = [multi["summary"][auc_key[m]]["sd"] for m in t1.index]
    t1.to_csv(RES / "table1_predictive.csv")
    t1.to_markdown(RES / "table1_predictive.md")

    cost_key = {"Prob. threshold 0.5": "thr05", "Top-20% by score": "top20", "Fuzzy–cost (proposed)": "fuzzy"}
    t2_rows = []
    for k, r in table2.items():
        inf = inference.get(k)
        t2_rows.append({"Policy": k, "group": "main", **r,
                        "cost_ci95": fmt_ci(ci[k][0]), "coverage_ci95": fmt_ci(ci[k][1]),
                        "d_cost_vs_proposed": None if inf is None else r["expected_cost"] - table2["Fuzzy–cost (proposed)"]["expected_cost"],
                        "d_cost_ci95": None if inf is None else fmt_ci(inf["d_cost_ci"]),
                        "p_bootstrap": None if inf is None else inf["p_bootstrap_cost"],
                        "p_wilcoxon": None if inf is None else inf["p_wilcoxon_cost"],
                        "cost_10seed_mean": multi["summary"][cost_key[k]]["mean"],
                        "cost_10seed_sd": multi["summary"][cost_key[k]]["sd"]})
    for k, r in ablation.items():
        inf = ablation_inf[k]
        t2_rows.append({"Policy": k, "group": "ablation/reference", **r,
                        "cost_ci95": fmt_ci(inf["cost_alt_ci"]), "coverage_ci95": fmt_ci(inf["cov_alt_ci"]),
                        "d_cost_vs_proposed": r["expected_cost"] - table2["Fuzzy–cost (proposed)"]["expected_cost"],
                        "d_cost_ci95": fmt_ci(inf["d_cost_ci"]),
                        "p_bootstrap": inf["p_bootstrap_cost"] if inf["n_nonzero_pairs"] else None,
                        "p_wilcoxon": inf["p_wilcoxon_cost"] if inf["n_nonzero_pairs"] else None,
                        "cost_10seed_mean": multi["summary"]["bayes"]["mean"] if k.startswith("Bayes") else
                        (multi["summary"]["fuzzy_gb"]["mean"] if k.startswith("Fuzzy–cost on GB") else None),
                        "cost_10seed_sd": multi["summary"]["bayes"]["sd"] if k.startswith("Bayes") else
                        (multi["summary"]["fuzzy_gb"]["sd"] if k.startswith("Fuzzy–cost on GB") else None)})
    t2 = pd.DataFrame(t2_rows).set_index("Policy")
    t2.to_csv(RES / "table2_decision.csv")
    t2.to_markdown(RES / "table2_decision.md")
    pd.DataFrame(long_rows).to_csv(RES / "evidence_extended.csv", index=False)

    # ---- Graphical abstract (Fig. 1 / Fig. 2 are drawn from the saved results by make_figures.py) ----
    cov = [table2[k]["attack_coverage"] for k in ("Prob. threshold 0.5", "Top-20% by score", "Fuzzy–cost (proposed)")]
    ga = draw_graphical_abstract(table2, cov)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ga, PAPER_FIG / Path(ga).name)
    fig1 = [str(FIG / f"fig1_decision_bands.{e}") for e in ("png", "pdf")]
    fig2 = [str(FIG / f"fig2_cost_sensitivity.{e}") for e in ("png", "pdf")]

    summary = {
        "seed": SEED, "n_train": run["n_train"], "n_val": run["n_val"], "n_test": run["n_test"],
        "attack_rate_test": float(y.mean()),
        "table1": table1, "table2": table2,
        "cost_matrix": {"rows": list(ACTIONS), "cols": ["benign", "attack"], "values": cost_matrix().tolist()},
        "membership_breakpoints": MF, "fuzzy_switch_points": cuts,
        "inference_vs_proposed_seed42": inference,
        "ci95_seed42": {k: {"cost": v[0], "coverage": v[1]} for k, v in ci.items()},
        "ablation_seed42": {"policies": ablation, "inference_vs_proposed": ablation_inf,
                            "crisp_bands_identical_to_fuzzy": identical,
                            "validation_tuned_threshold": t_val,
                            "test_oracle_threshold_not_a_valid_baseline": test_oracle,
                            "bayes_soft_block_threshold": bayes_soft_threshold},
        "cost_ratio_sweep": sweep, "fuzzy_beats_thr05_at_c_fn": fz_beats_thr,
        "membership_width_sensitivity": mf_sens,
        "multi_seed": multi,
        "figures": {"fig1": [_rel(f) for f in fig1], "fig2": [_rel(f) for f in fig2], "graphical_abstract": _rel(ga)},
        "config": {"n_boot": N_BOOT, "seeds": SEEDS if not quick else [SEED], "n_train_cap": N_TRAIN,
                   "n_test_cap": N_TEST_CAP, "torch": torch.__version__},
        "notes": [
            "Proposed model shares the neural propensity with the ablation; the contribution is the fuzzy rule-based decision layer.",
            "Train subsampled to 40k (34k train / 6k validation) and test capped at 20k so the full run (seed 42 + seeds 42-51) takes about 3 min on a CPU (2 min 47-50 s measured on 8 vCPUs).",
            "Under the Eq. (6) cost matrix Soft-block dominates Escalate for every p, so the Bayes cost rule never escalates.",
        ],
    }
    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    import make_figures  # single-panel Fig. 1 and Fig. 2, read back from results/summary.json
    make_figures.main()

    print(t1.round(4).to_string())
    print(t2.drop(columns=["group"]).to_string())
    print("switch points", cuts, "identical", identical, "t_val", t_val)
    print(json.dumps({k: v for k, v in multi.items() if k.startswith(("wilcoxon", "gb_", "fuzzy_"))}, indent=1))
    print("Wrote results to", RES)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="seed 42 only (skip 10-seed repetition)")
    main(ap.parse_args().quick)
