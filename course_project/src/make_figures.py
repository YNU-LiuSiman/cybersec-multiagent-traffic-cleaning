"""Generate 5 publication-quality figures for P7 report and presentation slides.

Figures generated under `course_project/figures/`:
1. fig1_architecture.png          -- LLM Control Plane x Python Data Plane architecture diagram
2. fig2_main_results.png          -- Main 10-class Macro-F1 (RF & HistGB, mean +/- SD across 10 eval seeds)
3. fig3_ablation_control_plane.png -- Ablation delta + Control-plane accuracy/protection heatmap
4. fig4_rare4_retention_recall.png -- Rare-4 training retention & downstream test recall comparison
5. fig5_cicids_burst_vs_fault.png  -- CICIDS-2017 9-slice Burst DDoS vs True Fault attribution heatmap
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Global clean publication style (supports CJK fallback if needed, clean sans-serif English labels)
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Microsoft YaHei", "SimHei"],
    "axes.unicode_minus": False,
    "figure.dpi": 220,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

METHOD_LABELS = {
    "Raw-Merge": "Raw-Merge",
    "IForest-Clean": "IForest-Clean",
    "Global-CL": "Global-CL",
    "Static-Pipeline": "Static-Pipeline\n(Same Tools, No LLM)",
    "Ours-wo-Coordinator": "Ours w/o\nCoordinator",
    "Ours-wo-Anchor": "Ours w/o\nAnchor",
    "Ours-wo-RareExempt": "Ours w/o\nRare-Exempt",
    "Ours-Full": "Ours (Full)",
    "Oracle-Clean": "Oracle-Clean\n(Upper Bound)",
}

METHOD_SHORT = {
    "Raw-Merge": "Raw-Merge",
    "IForest-Clean": "IForest",
    "Global-CL": "Global-CL",
    "Static-Pipeline": "Static-Pipe",
    "Ours-wo-Coordinator": "w/o Coord",
    "Ours-wo-Anchor": "w/o Anchor",
    "Ours-wo-RareExempt": "w/o RareEx",
    "Ours-Full": "Ours (Full)",
    "Oracle-Clean": "Oracle",
}


def fig1_architecture() -> None:
    """Clean schematic diagram of LLM Control Plane x Python Data Plane."""
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_color="#111827", lw=1.8, title_fs=10.5, body_fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.16",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color=title_color)
        if body:
            ax.text(x + w / 2, y + h - 0.62, body, ha="center", va="top",
                    fontsize=body_fs, color="#374151", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", color="#334155", rad=0.0, dy=0.16, ls="-"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}"),
        )
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + dy
            ax.text(mx, my, label, ha="center", va="center", fontsize=8.3,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))

    # Title banner
    ax.text(7.0, 7.35, "LLM Control Plane × Python Data Plane: Multi-Agent IDS Data Governance Architecture",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#0F172A")
    ax.text(7.0, 6.98, "Collaboration lives in Control-Plane Diagnosis & Negotiation (Zero Raw Rows Shared); Python Executes All Deterministic Math & Fixes",
            ha="center", va="center", fontsize=9.2, color="#475569", style="italic")

    # Zone backgrounds
    z_dp = mpatches.FancyBboxPatch((0.25, 0.25), 13.5, 2.30, boxstyle="round,pad=0.05,rounding_size=0.2",
                                   fc="#F0F9FF", ec="#7DD3FC", lw=1.3, ls="--")
    ax.add_patch(z_dp)
    ax.text(0.45, 2.38, "PYTHON DATA PLANE (Deterministic Compute, Hard Isolation & Execution)",
            fontsize=8.8, fontweight="bold", color="#0369A1")

    z_cp = mpatches.FancyBboxPatch((0.25, 3.60), 13.5, 3.15, boxstyle="round,pad=0.05,rounding_size=0.2",
                                   fc="#FEFCE8", ec="#FCD34D", lw=1.3, ls="--")
    ax.add_patch(z_cp)
    ax.text(0.45, 6.56, "LLM CONTROL PLANE (Semantic Attribution, Cross-Node Triangulation & Auditable Planning)",
            fontsize=8.8, fontweight="bold", color="#B45309")

    # Privacy boundary line (cleanly separated channel between y=2.60 and y=3.55)
    ax.plot([0.25, 13.75], [3.08, 3.08], color="#DC2626", lw=2.0, ls=":")
    ax.text(7.0, 3.08, "  HARD PRIVACY BOUNDARY: Whitelisted Statistical Summaries Only (Zero Row-Level Data / Indices)  ",
            ha="center", va="center", fontsize=8.3, fontweight="bold", color="#DC2626",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FEF2F2", ec="#FCA5A5"), zorder=5)

    # Data Plane boxes
    box(0.5, 0.50, 2.5, 1.55, "Shared Infra Layer",
        "All methods share:\n• Header & bad-row cleanup\n• Inf -> NaN repair\n• Regex schema alignment",
        "#E0F2FE", "#0284C7")
    box(3.55, 0.50, 3.55, 1.55, "Isolated Node Sandboxes (×4)",
        "Local-only Python tools per node:\nschema_profile · quantile_profile (Benign)\nconservation_check · label_histogram\nfeature_label_mi · row_health · local CL",
        "#DBEAFE", "#2563EB")
    box(7.75, 0.50, 2.85, 1.55, "Deterministic Executor",
        "Whitelist action engine:\nrename · drop_rows · rescale\ndrop_column · local_cl_clean\n(w/ Rare-Exempt) · no_op",
        "#DCFCE7", "#16A34A")
    box(11.15, 0.50, 2.35, 1.55, "Downstream IDS Eval",
        "Merge cleaned nodes\nTrain RF & HistGB\nEvaluate on untouched\nofficial unsw_test",
        "#F3E8FF", "#9333EA")

    # Control Plane boxes
    box(0.6, 3.95, 3.6, 2.25, "Node Agents (×4, Isolated)",
        "Role: Local Data-Quality Auditor\n• Calls ONLY own sandbox tools\n• Reports factual local evidence\n• Zero cross-node visibility\n• Prompt: frozen node_agent.md",
        "#FFFBEB", "#D97706")
    box(4.95, 3.95, 4.35, 2.25, "Coordinator Agent (≤3 Query Rounds)",
        "Operates ONLY on JSON summaries:\n• compare_anchor_quantiles() (Benign anchor)\n• compare_mi() (Single-node shortcut hunter)\n• aggregate_histograms() (Global rare counts)\n• query_node() -> shortcut-free CL verification\n• Prompt: frozen coordinator_agent.md",
        "#FEF3C7", "#B45309")
    box(10.05, 3.95, 3.45, 2.25, "Remediation Plan (JSON)",
        "Auditable structured contract:\n• Strict Executor schema check\n• Mandatory numerical evidence\n• Explicit no_op for Node D\n• Full JSONL transcripts logged",
        "#ECFDF5", "#059669")

    # Arrows (with labels positioned cleanly ABOVE the privacy banner, never overlapping it!)
    arrow(3.0, 1.28, 3.55, 1.28, "")
    arrow(4.6, 2.05, 2.4, 3.95, "", color="#2563EB")
    ax.text(2.45, 3.45, "Local tool JSON", ha="center", va="center", fontsize=8.3, fontweight="bold",
            color="#2563EB", bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#93C5FD", alpha=0.95), zorder=6)
    arrow(4.2, 5.08, 4.95, 5.08, "Node\nReports", color="#B45309", dy=0.35)
    arrow(6.8, 3.95, 5.8, 2.05, "", color="#DC2626", ls="--")
    ax.text(6.95, 3.48, "query_node() (exclude shortcut)", ha="center", va="center", fontsize=8.2, fontweight="bold",
            color="#DC2626", bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#FCA5A5", alpha=0.95), zorder=6)
    arrow(9.30, 5.08, 10.05, 5.08, "Validated\nPlan", color="#059669", dy=0.35)
    arrow(11.3, 3.95, 9.4, 2.05, "", color="#16A34A")
    ax.text(11.45, 3.45, "Plan JSON → Execute", ha="center", va="center", fontsize=8.3, fontweight="bold",
            color="#16A34A", bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#86EFAC", alpha=0.95), zorder=6)
    arrow(10.60, 1.28, 11.15, 1.28, "")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig2_main_results(main_df: pd.DataFrame, stats: Dict) -> None:
    """Grouped bar chart with +/- SD error bars for RF and HistGB 10-class Macro-F1."""
    order = [
        "Raw-Merge", "IForest-Clean", "Global-CL", "Static-Pipeline",
        "Ours-wo-Coordinator", "Ours-wo-Anchor", "Ours-wo-RareExempt",
        "Ours-Full", "Oracle-Clean",
    ]
    df = main_df.set_index("method").loc[[m for m in order if m in main_df["method"].values]].reset_index()

    x = np.arange(len(df))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12.8, 5.4))

    rf_means = df["RF_macro_f1_mean"].to_numpy()
    rf_sds = df["RF_macro_f1_sd"].to_numpy()
    hgb_means = df["HistGB_macro_f1_mean"].to_numpy()
    hgb_sds = df["HistGB_macro_f1_sd"].to_numpy()

    colors_rf = ["#94A3B8" if m not in ("Ours-Full", "Oracle-Clean") else ("#1D4ED8" if m == "Ours-Full" else "#0F766E") for m in df["method"]]
    colors_hgb = ["#CBD5E1" if m not in ("Ours-Full", "Oracle-Clean") else ("#EA580C" if m == "Ours-Full" else "#14B8A6") for m in df["method"]]

    bars_rf = ax.bar(x - w / 2, rf_means, w, yerr=rf_sds, capsize=3.5, color=colors_rf, edgecolor="#1E293B", linewidth=0.8, label="Random Forest (RF)")
    bars_hgb = ax.bar(x + w / 2, hgb_means, w, yerr=hgb_sds, capsize=3.5, color=colors_hgb, edgecolor="#1E293B", linewidth=0.8, hatch="//", label="HistGradientBoosting (HistGB)")

    # Highlight Ours (Full) column softly
    full_idx = df.index[df["method"] == "Ours-Full"].tolist()
    if full_idx:
        fi = full_idx[0]
        ax.axvspan(fi - 0.48, fi + 0.48, color="#EFF6FF", alpha=0.7, zorder=0)

    # Value labels + significance markers vs Ours-Full
    rf_stats = stats.get("classifiers", {}).get("RF", {}).get("macro_f1", {}).get("comparisons", {})
    hgb_stats = stats.get("classifiers", {}).get("HistGB", {}).get("macro_f1", {}).get("comparisons", {})

    for i, m in enumerate(df["method"]):
        star_rf = ""
        if m in rf_stats:
            p = rf_stats[m]["p_holm"]
            star_rf = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        star_hgb = ""
        if m in hgb_stats:
            p = hgb_stats[m]["p_holm"]
            star_hgb = "**" if p < 0.01 else ("*" if p < 0.05 else "")

        ax.text(x[i] - w / 2, rf_means[i] + rf_sds[i] + 0.003, f"{rf_means[i]:.3f}{star_rf}",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold" if m == "Ours-Full" else "normal", color="#1E3A8A")
        ax.text(x[i] + w / 2, hgb_means[i] + hgb_sds[i] + 0.003, f"{hgb_means[i]:.3f}{star_hgb}",
                ha="center", va="bottom", fontsize=7.8, fontweight="bold" if m == "Ours-Full" else "normal", color="#9A3412")

    ymin = max(0.30, min(rf_means.min(), hgb_means.min()) - 0.04)
    ymax = max(rf_means.max(), hgb_means.max()) + 0.045
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in df["method"]], fontsize=8.8)
    ax.set_ylabel("UNSW-NB15 Official Test 10-Class Macro-F1", fontsize=10.5, fontweight="bold")
    ax.set_title("Main Evaluation Results across 10 Randomized Multi-Node Fault Scenarios (Seeds 100–109, Mean ± SD)",
                 fontsize=12.0, fontweight="bold", pad=12)
    ax.legend(loc="upper left", frameon=True, fontsize=9.2)
    fig.text(0.5, -0.015, "* p_Holm < 0.05, ** p_Holm < 0.01 (one-sided paired Wilcoxon signed-rank test: Ours-Full > comparator, n=10 eval seeds)",
             ha="center", va="top", fontsize=8.6, color="#475569")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_main_results.png", bbox_inches="tight")
    plt.close(fig)


def fig3_ablation_control_plane(main_df: pd.DataFrame, cp_df: pd.DataFrame) -> None:
    """Dual panel: Left = Ablation Macro-F1 Delta vs Ours-Full; Right = Control-Plane Heatmap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw={"width_ratios": [1.0, 1.35]})

    # Left panel: Delta relative to Ours-Full
    m_map = main_df.set_index("method")
    full_rf = float(m_map.loc["Ours-Full", "RF_macro_f1_mean"])
    full_hgb = float(m_map.loc["Ours-Full", "HistGB_macro_f1_mean"])

    ab_list = ["Ours-wo-RareExempt", "Ours-wo-Anchor", "Ours-wo-Coordinator", "Static-Pipeline", "Global-CL", "Raw-Merge"]
    ab_list = [m for m in ab_list if m in m_map.index]
    d_rf = [float(m_map.loc[m, "RF_macro_f1_mean"]) - full_rf for m in ab_list]
    d_hgb = [float(m_map.loc[m, "HistGB_macro_f1_mean"]) - full_hgb for m in ab_list]

    y = np.arange(len(ab_list))
    h = 0.35
    ax1.barh(y - h / 2, d_rf, h, color="#2563EB", alpha=0.88, label="RF ΔMacro-F1")
    ax1.barh(y + h / 2, d_hgb, h, color="#EA580C", alpha=0.88, label="HistGB ΔMacro-F1")
    ax1.axvline(0.0, color="#0F172A", lw=1.2)
    for i in range(len(ab_list)):
        ax1.text(d_rf[i] - 0.001 if d_rf[i] < 0 else d_rf[i] + 0.001, y[i] - h / 2,
                 f"{d_rf[i]:+.4f}", va="center", ha="right" if d_rf[i] < 0 else "left", fontsize=8.0, color="#1E3A8A")
        ax1.text(d_hgb[i] - 0.001 if d_hgb[i] < 0 else d_hgb[i] + 0.001, y[i] + h / 2,
                 f"{d_hgb[i]:+.4f}", va="center", ha="right" if d_hgb[i] < 0 else "left", fontsize=8.0, color="#9A3412")
    ax1.set_yticks(y)
    ax1.set_yticklabels([METHOD_SHORT[m] for m in ab_list], fontsize=9.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("Macro-F1 Difference vs Ours (Full)", fontsize=9.8, fontweight="bold")
    ax1.set_title("(a) Ablation & Control Impact (Δ vs Ours-Full)", fontsize=10.8, fontweight="bold")
    ax1.legend(loc="lower left", fontsize=8.5)
    xmin = min(min(d_rf), min(d_hgb), -0.01) - 0.014
    xmax = max(max(d_rf), max(d_hgb), 0.005) + 0.012
    ax1.set_xlim(xmin, xmax)

    # Right panel: Control-plane performance heatmap
    cp_order = ["Static-Pipeline", "Ours-wo-Coordinator", "Ours-wo-Anchor", "Ours-Full"]
    cp_sub = cp_df.set_index("method").loc[[m for m in cp_order if m in cp_df["method"].values]]
    n_eval = cp_sub["n_eval"].to_numpy(dtype=float)

    mat_cols = [
        ("Unit Factor\nAcc", cp_sub["unit_factor_acc"].to_numpy(dtype=float)),
        ("Shortcut\nExact Acc", cp_sub["shortcut_exact_acc"].to_numpy(dtype=float)),
        ("Residual\nAlias Acc", cp_sub["residual_alias_acc"].to_numpy(dtype=float)),
        ("Noise Node\nRecall", cp_sub["noise_node_recall"].to_numpy(dtype=float)),
        ("Noise\nSample F1", cp_sub["noise_sample_f1"].to_numpy(dtype=float)),
        ("Node D\nSafe Rate", np.clip(1.0 - cp_sub["node_d_false_remediation_sum"].to_numpy(dtype=float) / np.maximum(n_eval, 1.0), 0.0, 1.0)),
        ("Clean Ctrl\nSpecificity", np.clip(1.0 - cp_sub["clean_control_fpr"].fillna(0.0).to_numpy(dtype=float), 0.0, 1.0)),
    ]
    mat = np.column_stack([vals for _, vals in mat_cols])
    im = ax2.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax2.set_xticks(np.arange(len(mat_cols)))
    ax2.set_xticklabels([name for name, _ in mat_cols], fontsize=8.5)
    ax2.set_yticks(np.arange(len(cp_sub)))
    ax2.set_yticklabels([METHOD_SHORT[m] for m in cp_sub.index], fontsize=9.2, fontweight="bold")
    ax2.grid(False)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            txt_col = "white" if val < 0.35 or val > 0.85 else "#111827"
            ax2.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=9.0, fontweight="bold", color=txt_col)
    ax2.set_title("(b) Control-Plane Accuracy & Safety Matrix (Seeds 100–109 & 200–202)", fontsize=10.8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.042, pad=0.03)
    cbar.set_label("Score (higher = better)", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_ablation_control_plane.png", bbox_inches="tight")
    plt.close(fig)


def fig4_rare4_retention_recall(main_df: pd.DataFrame) -> None:
    """Dual panel: Left = Rare-4 Training Retention Rate; Right = Rare-4 Test Recall (RF & HistGB)."""
    order = [
        "Raw-Merge", "IForest-Clean", "Global-CL", "Static-Pipeline",
        "Ours-wo-Coordinator", "Ours-wo-Anchor", "Ours-wo-RareExempt",
        "Ours-Full", "Oracle-Clean",
    ]
    df = main_df.set_index("method").loc[[m for m in order if m in main_df["method"].values]].reset_index()
    x = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0))

    # Left: Rare-4 train retention
    ret = df["rare4_train_retention_mean"].to_numpy() * 100.0
    cols1 = ["#DC2626" if v < 85 else ("#16A34A" if m in ("Ours-Full", "Oracle-Clean") else "#64748B")
             for m, v in zip(df["method"], ret)]
    ax1.bar(x, ret, color=cols1, edgecolor="#1E293B", linewidth=0.8)
    for i, v in enumerate(ret):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax1.set_ylim(0, 114)
    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_SHORT[m] for m in df["method"]], rotation=25, ha="right", fontsize=8.8)
    ax1.set_ylabel("Rare-4 Training Samples Retained (%)", fontsize=10.0, fontweight="bold")
    ax1.set_title("(a) Rare-4 Training Set Preservation (Out of 1,682 Samples)", fontsize=10.8, fontweight="bold")

    # Right: Rare-4 Test Recall
    w = 0.36
    r4_rf = df["RF_rare4_recall_mean"].to_numpy()
    r4_hgb = df["HistGB_rare4_recall_mean"].to_numpy()
    ax2.bar(x - w / 2, r4_rf, w, color="#2563EB", edgecolor="#1E293B", linewidth=0.7, label="RF Rare-4 Recall")
    ax2.bar(x + w / 2, r4_hgb, w, color="#F97316", edgecolor="#1E293B", linewidth=0.7, label="HistGB Rare-4 Recall")
    for i in range(len(df)):
        ax2.text(i - w / 2, r4_rf[i] + 0.005, f"{r4_rf[i]:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
        ax2.text(i + w / 2, r4_hgb[i] + 0.005, f"{r4_hgb[i]:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax2.set_ylim(0, max(r4_rf.max(), r4_hgb.max()) + 0.08)
    ax2.set_xticks(x)
    ax2.set_xticklabels([METHOD_SHORT[m] for m in df["method"]], rotation=25, ha="right", fontsize=8.8)
    ax2.set_ylabel("Official unsw_test Rare-4 Mean Recall", fontsize=10.0, fontweight="bold")
    ax2.set_title("(b) Downstream Rare-4 Attack Recall on unsw_test", fontsize=10.8, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_rare4_retention_recall.png", bbox_inches="tight")
    plt.close(fig)


def fig5_cicids_burst_vs_fault(cic: Dict) -> None:
    """CICIDS-2017 9-slice attribution heatmap: Burst DDoS vs True Fault."""
    tel = pd.DataFrame(cic["slice_telemetry"])
    slices = tel["slice"].tolist()
    n = len(slices)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12.2, 6.2), gridspec_kw={"height_ratios": [1.0, 2.2]}, sharex=True
    )

    # Top: Natural DDoS rate across chronological slices + Tuesday Benign slice_9
    ddos_pct = tel["ddos_rate"].to_numpy() * 100.0
    bar_cols = [
        "#DC2626" if gt != "clean" else ("#0D9488" if s == "slice_9" else "#3B82F6")
        for s, gt in zip(tel["slice"], tel["ground_truth"])
    ]
    ax_top.bar(np.arange(n), ddos_pct, color=bar_cols, edgecolor="#1E293B", linewidth=0.8)
    for i, (v, gt) in enumerate(zip(ddos_pct, tel["ground_truth"])):
        tag = f"{v:.1f}%" + (f"\n[{gt}]" if gt != "clean" else ("\n[Tue Benign]" if i == n - 1 else ""))
        ax_top.text(i, v + 2.5, tag, ha="center", va="bottom", fontsize=7.8,
                    fontweight="bold" if gt != "clean" else "normal",
                    color="#B91C1C" if gt != "clean" else "#1E293B")
    ax_top.set_ylim(0, 102)
    ax_top.set_ylabel("DDoS Rate (%)", fontsize=9.8, fontweight="bold")
    ax_top.set_title(
        "Cross-Dataset Generalization (CICIDS-2017, Zero Prompt Edits): Burst Attack vs True Fault Attribution",
        fontsize=11.8, fontweight="bold", pad=10,
    )

    # Bottom: Diagnostic signal + decision heatmap
    rows_labels = [
        "Whole-Table Max |log10 Ratio|",
        "Benign-Anchor Max |log10 Ratio|",
        "Conservation Violation Rate",
        "CL Suspected Noise Rate",
        "Ground Truth (1=Fault)",
        "Static-Pipeline Flagged",
        "Ours (Full) Flagged",
    ]
    raw_mat = np.vstack([
        tel["whole_table_max_abs_log10"].to_numpy(dtype=float),
        tel["benign_anchor_max_abs_log10"].to_numpy(dtype=float),
        tel["conservation_fwd_violation"].to_numpy(dtype=float),
        tel["cl_suspected_noise_rate"].to_numpy(dtype=float),
        (tel["ground_truth"] != "clean").astype(float).to_numpy(),
        tel["static_flagged"].astype(float).to_numpy(),
        tel["ours_flagged"].astype(float).to_numpy(),
    ])
    # Normalize each row into [0, 1] for visual clarity while printing exact raw values
    norm_mat = np.zeros_like(raw_mat)
    row_scales = [3.0, 3.0, 1.0, 0.30, 1.0, 1.0, 1.0]
    for r, sc in enumerate(row_scales):
        norm_mat[r] = np.clip(raw_mat[r] / sc, 0.0, 1.0)

    ax_bot.imshow(norm_mat, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax_bot.grid(False)
    ax_bot.set_yticks(np.arange(len(rows_labels)))
    ax_bot.set_yticklabels(rows_labels, fontsize=9.2, fontweight="bold")
    ax_bot.set_xticks(np.arange(n))
    ax_bot.set_xticklabels([f"{s}\n({'Tue' if s=='slice_9' else 'Fri'})" for s in slices], fontsize=8.8)

    for r in range(raw_mat.shape[0]):
        for c in range(raw_mat.shape[1]):
            v = raw_mat[r, c]
            if r >= 4:
                lbl = "FAULT" if v >= 0.5 else "clean"
            elif r in (0, 1):
                lbl = f"{v:.2f}"
            else:
                lbl = f"{v:.3f}"
            txt_c = "white" if norm_mat[r, c] > 0.62 else "#111827"
            ax_bot.text(c, r, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=txt_c)

    # Separator line between diagnostic features and decisions
    ax_bot.axhline(3.5, color="#0F172A", lw=2.0)

    sp_fpr = cic["static_pipeline"]["clean_slice_fpr"] * 100
    sp_tpr = cic["static_pipeline"]["fault_slice_tpr"] * 100
    ou_fpr = cic["ours_full"]["clean_slice_fpr"] * 100
    ou_tpr = cic["ours_full"]["fault_slice_tpr"] * 100
    ax_bot.set_xlabel(
        f"Verdict — Static-Pipeline: FPR={sp_fpr:.1f}% ({cic['static_pipeline']['fp_count']}/7), TPR={sp_tpr:.0f}%   |   "
        f"Ours (Full, Frozen Prompt): FPR={ou_fpr:.1f}% ({cic['ours_full']['fp_count']}/7), TPR={ou_tpr:.0f}%",
        fontsize=9.3, fontweight="bold", labelpad=8, color="#0F172A",
    )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_cicids_burst_vs_fault.png", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig1_architecture()
    main_df = pd.read_csv(RESULTS_DIR / "main_table.csv")
    cp_df = pd.read_csv(RESULTS_DIR / "control_plane.csv")
    stats = json.loads((RESULTS_DIR / "stats.json").read_text(encoding="utf-8"))
    fig2_main_results(main_df, stats)
    fig3_ablation_control_plane(main_df, cp_df)
    fig4_rare4_retention_recall(main_df)
    cic_path = RESULTS_DIR / "cicids_demo.json"
    if cic_path.exists():
        cic = json.loads(cic_path.read_text(encoding="utf-8"))
        fig5_cicids_burst_vs_fault(cic)
    print("Generated all figures in", FIGURES_DIR)


if __name__ == "__main__":
    generate_all_figures()
