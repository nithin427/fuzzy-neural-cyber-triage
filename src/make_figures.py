#!/usr/bin/env python3
"""Fig. 1 and Fig. 2 of the paper, drawn only from the saved result files.

Fig. 1  Score-to-action mapping: fuzzy memberships (Eqs. 2-4), the actions chosen by
        rules R1-R3, the switch points 9/32 and 23/32, the Bayes soft-block threshold
        2/11 and the number of seed-42 test alerts routed to each action.
Fig. 2  Mean triage cost versus the missed-attack cost C_FN for the four policies
        (seed 42), with paired-bootstrap 95% CIs at the default C_FN = 10.

Inputs : results/summary.json, results/table2_decision.csv
Outputs: figures/fig1_decision_bands.{png,pdf}, figures/fig2_cost_sensitivity.{png,pdf}
         (300 dpi PNG + vector PDF), copied to paper/elsevier/figures/.
Usage  : python src/make_figures.py   (also called at the end of src/experiment.py)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
PAPER_FIG = ROOT / "paper" / "elsevier" / "figures"
PROPOSED = "Fuzzy\u2013cost (proposed)"
COL = {"thr": "#c0392b", "top": "#e69f00", "fz": "#1f4e79", "bayes": "#6a3d9a",
       "L": "#2a9d8f", "M": "#e69f00", "H": "#c0392b"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})


def memberships(p, mf):
    """Eqs. (2)-(4) from the breakpoints stored in summary.json."""
    mu_l = np.clip((mf["low_end"] - p) / mf["low_end"], 0, 1)
    mu_m = np.where(p < mf["med_peak"], np.clip((p - mf["med_lo"]) / (mf["med_peak"] - mf["med_lo"]), 0, 1),
                    np.clip((mf["med_hi"] - p) / (mf["med_hi"] - mf["med_peak"]), 0, 1))
    mu_h = np.clip((p - mf["high_start"]) / (1.0 - mf["high_start"]), 0, 1)
    return mu_l, mu_m, mu_h


def _save(fig, stem):
    outs = []
    for ext in ("png", "pdf"):
        out = FIG / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outs.append(out)
    plt.close(fig)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for o in outs:
        shutil.copy2(o, PAPER_FIG / o.name)
    return [str(o) for o in outs]


def fig1(summary):
    mf, cuts = summary["membership_breakpoints"], summary["fuzzy_switch_points"]
    bayes_t = summary["ablation_seed42"]["bayes_soft_block_threshold"]
    t2 = summary["table2"][PROPOSED]
    n = summary["n_test"]
    p = np.linspace(0, 1, 2001)
    mu_l, mu_m, mu_h = memberships(p, mf)
    fig, ax = plt.subplots(figsize=(4.0, 1.9))
    spans = [(0, cuts[0], COL["L"], "R1: Monitor", t2["n_monitor"]),
             (cuts[0], cuts[1], COL["M"], "R2: Soft-block", t2["n_soft"]),
             (cuts[1], 1, COL["H"], "R3: Escalate", t2["n_escalate"])]
    for a, b, c, lab, cnt in spans:
        ax.axvspan(a, b, color=c, alpha=0.08, lw=0)
        ax.text((a + b) / 2, 1.19, lab, ha="center", va="bottom", fontsize=8, weight="bold", color=c)
        ax.text((a + b) / 2, 1.09, f"{cnt:,} ({cnt / n:.1%})", ha="center", va="bottom", fontsize=7)
    ax.plot(p, mu_l, color=COL["L"], lw=1.9, label=r"Low  $\mu_L$")
    ax.plot(p, mu_m, color=COL["M"], lw=1.9, label=r"Medium  $\mu_M$")
    ax.plot(p, mu_h, color=COL["H"], lw=1.9, label=r"High  $\mu_H$")
    for c, txt in zip(cuts, ("9/32", "23/32")):
        ax.axvline(c, color="k", ls=":", lw=0.9)
        ax.text(c + 0.008, 0.93, f"switch point\np = {txt}", fontsize=7, va="center")
    ax.axvline(bayes_t, ymax=1.08 / 1.32, color=COL["bayes"], ls="--", lw=1.0)
    ax.text(bayes_t - 0.008, 0.03, "Bayes rule: p = 2/11", fontsize=7,
            rotation=90, va="bottom", ha="right", color=COL["bayes"],
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Neural attack propensity p")
    ax.set_ylabel("Membership degree")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.95, fontsize=7)
    return _save(fig, "fig1_decision_bands")


def fig2(summary):
    sw = pd.DataFrame(summary["cost_ratio_sweep"])
    t2 = pd.read_csv(RES / "table2_decision.csv").set_index("Policy")
    c10 = summary["config"].get("c_fn_default", 10)
    fig, ax = plt.subplots(figsize=(4.0, 2.05))
    pols = [("Prob. threshold 0.5", "thr", "s", "Threshold 0.5"),
            ("Top-20% by score", "top", "^", "Top-20% escalation"),
            (PROPOSED, "fz", "o", "Fuzzy rule policy (proposed)"),
            ("Bayes cost rule (reference)", "bayes", "d", "Bayes cost rule (reference)")]
    lo_cross = sw[(sw.policy == "Top-20% by score")].set_index("c_fn").expected_cost < \
        sw[(sw.policy == PROPOSED)].set_index("c_fn").expected_cost
    xmax_top = max(lo_cross[lo_cross].index) if lo_cross.any() else None
    if xmax_top is not None:
        ax.axvspan(1.9, xmax_top * 1.12, color="#e69f00", alpha=0.10, lw=0)
        ax.text(2.0, 12.4, f"top-20%\ncheaper only\nfor $C_{{FN}}\\leq{xmax_top:g}$", fontsize=6.5, va="top")
    for pol, key, mk, lab in pols:
        d = sw[sw.policy == pol].sort_values("c_fn")
        ls = "--" if key == "bayes" else "-"
        ax.plot(d.c_fn, d.expected_cost, marker=mk, ms=4, lw=1.5 if key == "fz" else 1.2, ls=ls,
                color=COL[key], label=lab, zorder=3 if key == "fz" else 2)
        lo, hi = [float(v) for v in t2.loc[pol, "cost_ci95"].strip("[]").split(",")]
        m = t2.loc[pol, "expected_cost"]
        ax.errorbar(c10, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor="k", elinewidth=1.0, capsize=3,
                    zorder=4)
    ax.axvline(c10, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 7.5, 10, 15, 20, 30])
    ax.set_xticklabels(["2", "3", "5", "7.5", "10", "15", "20", "30"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.9, 32)
    ax.set_xlabel("Missed-attack cost $C_{FN}$ (log scale; default 10)")
    ax.set_ylabel("Mean triage cost per alert")
    ax.errorbar([], [], yerr=[], fmt="none", ecolor="k", capsize=3, label="95% bootstrap CI ($C_{FN}$ = 10)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.205, 1.02), frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=7)
    return _save(fig, "fig2_cost_sensitivity")


def main():
    with open(RES / "summary.json") as f:
        summary = json.load(f)
    return fig1(summary) + fig2(summary)


if __name__ == "__main__":
    for path in main():
        print("wrote", Path(path).relative_to(ROOT))
