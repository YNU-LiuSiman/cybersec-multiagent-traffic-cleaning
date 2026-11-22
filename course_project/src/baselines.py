"""Deterministic baseline methods sharing the exact same infrastructure layer.

Methods implemented:
- Raw-Merge: infra-cleaned nodes merged directly (lower bound)
- IForest-Clean: merged + IsolationForest (5% row contamination removal)
- Global-CL: merged + global Confident Learning without rare-class exemption
- Static-Pipeline: SAME Python tools as Ours, fixed execution order + fixed thresholds
  (whole-table drift threshold, fixed MI outlier threshold, fixed CL threshold w/o rare exemption,
   naive quantile-distance alias heuristic) — key non-LLM control
- Oracle-Clean: perfect ground-truth remediation (upper bound)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OrdinalEncoder

from .executor import execute_remediation_plan
from .infra import InfraReport
from .tools_coord import CoordinatorToolkit
from .tools_node import NodeSandbox, run_local_confident_learning


def run_raw_merge(cleaned_nodes: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Baseline 1: Direct merge after shared infrastructure cleaning."""
    plan = {
        "diagnosis_summary": "Raw-Merge baseline (no remediation).",
        "actions": [
            {"action": "no_op", "node": nid, "evidence": "Raw-Merge baseline"}
            for nid in sorted(cleaned_nodes.keys())
        ],
    }
    merged, audit = execute_remediation_plan(cleaned_nodes, plan)
    return merged, plan, audit


def run_iforest_clean(
    cleaned_nodes: Dict[str, pd.DataFrame],
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Baseline 2: Classic unsupervised row-level Isolation Forest (5% removal) on merged table."""
    merged = pd.concat([cleaned_nodes[k] for k in sorted(cleaned_nodes.keys())], ignore_index=True)
    feat_cols = [c for c in merged.columns if c not in ("attack_cat", "label")]
    X = merged[feat_cols].copy()
    cat_cols = [c for c in ("proto", "service", "state") if c in feat_cols]
    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("__MISSING__")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])
    for c in feat_cols:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = X[c].median()
            X[c] = X[c].fillna(0.0 if pd.isna(med) else float(med))

    iso = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    preds = iso.fit_predict(X.to_numpy(dtype=np.float32))
    keep = preds == 1
    out = merged.loc[keep].reset_index(drop=True)
    plan = {
        "diagnosis_summary": f"IForest-Clean baseline (contamination={contamination}).",
        "actions": [
            {"action": "no_op", "node": nid, "evidence": "IForest operates globally after merge"}
            for nid in sorted(cleaned_nodes.keys())
        ],
    }
    audit = {
        "method": "IForest-Clean",
        "rows_in": int(len(merged)),
        "rows_removed": int((~keep).sum()),
        "merged_rows": int(len(out)),
    }
    return out, plan, audit


def run_global_cl(
    cleaned_nodes: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Baseline 3: Global Confident Learning on merged table without rare-class exemption."""
    merged = pd.concat([cleaned_nodes[k] for k in sorted(cleaned_nodes.keys())], ignore_index=True)
    flagged, stats = run_local_confident_learning(
        df=merged,
        exempt_classes=set(),
        exclude_cols={"attack_cat", "label"},
    )
    keep = np.ones(len(merged), dtype=bool)
    if flagged:
        keep[flagged] = False
    out = merged.loc[keep].reset_index(drop=True)
    plan = {
        "diagnosis_summary": "Global-CL baseline (merged Confident Learning without rare-class exemption).",
        "actions": [
            {"action": "no_op", "node": nid, "evidence": "Global-CL operates globally after merge"}
            for nid in sorted(cleaned_nodes.keys())
        ],
    }
    audit = {
        "method": "Global-CL",
        "rows_in": int(len(merged)),
        "rows_removed": int(len(flagged)),
        "merged_rows": int(len(out)),
        "cl_stats": stats,
    }
    return out, plan, audit


def run_static_pipeline(
    cleaned_nodes: Dict[str, pd.DataFrame],
    infra_reports: Dict[str, InfraReport],
    drift_log10_thresh: float = 0.70,      # >= ~5x whole-table median shift
    mi_gap_thresh: float = 0.35,           # fixed cross-node MI gap threshold
    mi_max_thresh: float = 0.50,           # fixed peak MI threshold
    cl_noise_thresh: float = 0.12,         # fixed suspected noise rate threshold
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Baseline 4 (Critical Non-LLM Control): Static-Pipeline using the EXACT same Python tools.

    Fixed execution order + fixed thresholds:
      1. row_health -> drop_rows if corrupted_rows > 0
      2. schema_profile + quantile_profile(ALL) -> naive quantile-distance alias match (no semantics)
      3. compare_anchor_quantiles(ALL) -> rescale any col with |log10_ratio| >= drift_log10_thresh
      4. compare_mi -> drop_column if max_mi >= mi_max_thresh and gap >= mi_gap_thresh
      5. confident_learning (excluding dropped cols, NO rare-class exemption) -> local_cl_clean if rate >= cl_noise_thresh
    """
    sandboxes = {
        nid: NodeSandbox(node_id=nid, df=cleaned_nodes[nid], infra_report=infra_reports.get(nid))
        for nid in sorted(cleaned_nodes.keys())
    }
    coord = CoordinatorToolkit(sandboxes)
    node_ids = sorted(sandboxes.keys())

    actions: List[Dict[str, Any]] = []
    touched_nodes = set()

    # 1. Row health check
    for nid in node_ids:
        rh = coord.query_node(nid, "row_health")
        if rh["total_corrupted_rows"] > 0:
            actions.append(
                {
                    "action": "drop_rows",
                    "node": nid,
                    "rule": "nan_or_negative_physical",
                    "evidence": f"Static rule: total_corrupted_rows={rh['total_corrupted_rows']} > 0",
                }
            )
            touched_nodes.add(nid)

    # 2. Whole-table quantile drift & naive quantile-only residual alias matching
    sp_sums = {nid: coord.query_node(nid, "schema_profile") for nid in node_ids}
    q_sums_all = {nid: coord.query_node(nid, "quantile_profile", {"label_filter": "ALL"}) for nid in node_ids}
    cmp_q = coord.compare_anchor_quantiles(q_sums_all, sp_sums)

    # Naive alias matching: purely by closest p90/p50 distance (no LLM semantic reasoning)
    for nid, hint in cmp_q.get("unmapped_vs_missing_candidates", {}).items():
        unmapped_dict = hint.get("unmapped_columns", {})
        missing_refs = hint.get("missing_canonical_refs", {})
        # Exclude columns present in ALL nodes (like metadata/shortcut col)
        common_cols = set.intersection(*(set(sp_sums[x]["columns"]) for x in node_ids))
        cand_unmapped = [u for u in unmapped_dict if u not in common_cols]
        mapping: Dict[str, str] = {}
        used_targets = set()
        for u in cand_unmapped:
            up50 = unmapped_dict[u].get("p50")
            up90 = unmapped_dict[u].get("p90")
            best_t, best_d = None, 1e9
            for mc, ref in missing_refs.items():
                if mc in used_targets:
                    continue
                rp50, rp90 = ref.get("ref_p50"), ref.get("ref_p90")
                if up50 is None or rp50 is None or up90 is None or rp90 is None:
                    continue
                d = abs(math.log10(abs(up50) + 1e-4) - math.log10(abs(rp50) + 1e-4)) + abs(
                    math.log10(abs(up90) + 1e-4) - math.log10(abs(rp90) + 1e-4)
                )
                if d < best_d:
                    best_d, best_t = d, mc
            if best_t is not None and best_d < 0.25:
                mapping[u] = best_t
                used_targets.add(best_t)
        if mapping:
            actions.append(
                {
                    "action": "rename_columns",
                    "node": nid,
                    "mapping": mapping,
                    "evidence": f"Static rule: closest quantile distance match {mapping}",
                }
            )
            touched_nodes.add(nid)

    # Whole-table scale drift detection (fixed |log10_ratio| >= drift_log10_thresh)
    per_node_drift: Dict[str, List[Tuple[str, float]]] = {}
    for col, info in cmp_q.get("flagged_scale_shifts", {}).items():
        for nid, stats in info.get("per_node", {}).items():
            l10 = abs(float(stats.get("log10_ratio", 0.0)))
            ratio = float(stats.get("ratio_to_ref", 1.0))
            if l10 >= drift_log10_thresh and ratio > 0:
                per_node_drift.setdefault(nid, []).append((col, ratio))

    for nid, col_ratios in sorted(per_node_drift.items()):
        # Group by similar ratio (within 20%) or issue per-column rescale
        for col, ratio in col_ratios:
            actions.append(
                {
                    "action": "rescale",
                    "node": nid,
                    "cols": [col],
                    "factor": round(1.0 / ratio, 6),
                    "evidence": f"Static whole-table rule: {col} ratio={ratio:.4f} (|log10|>={drift_log10_thresh})",
                }
            )
            touched_nodes.add(nid)

    # 3. Cross-node MI outlier check
    mi_sums = {nid: coord.query_node(nid, "feature_label_mi") for nid in node_ids}
    cmp_mi = coord.compare_mi(mi_sums)
    dropped_cols: List[str] = []
    for row in cmp_mi.get("top_cross_node_mi_outliers", []):
        if row["max_mi_norm"] >= mi_max_thresh and row["gap_max_minus_median"] >= mi_gap_thresh:
            col = row["col"]
            dropped_cols.append(col)
            actions.append(
                {
                    "action": "drop_column",
                    "col": col,
                    "evidence": (
                        f"Static MI rule: {col} max_mi={row['max_mi_norm']} on {row['max_node']}, "
                        f"gap={row['gap_max_minus_median']} >= {mi_gap_thresh}"
                    ),
                }
            )

    # 4. Confident Learning check (with dropped cols excluded, NO rare-class exemption)
    for nid in node_ids:
        cl = coord.query_node(nid, "confident_learning", {"exclude_cols": dropped_cols, "exempt_classes": []})
        if cl["suspected_noise_rate"] >= cl_noise_thresh:
            actions.append(
                {
                    "action": "local_cl_clean",
                    "node": nid,
                    "exempt_classes": [],
                    "evidence": (
                        f"Static CL rule: suspected_noise_rate={cl['suspected_noise_rate']} >= {cl_noise_thresh} "
                        "(no rare-class exemption)"
                    ),
                }
            )
            touched_nodes.add(nid)

    for nid in node_ids:
        if nid not in touched_nodes:
            actions.append({"action": "no_op", "node": nid, "evidence": "Static rule: no threshold exceeded"})

    plan = {
        "diagnosis_summary": "Static-Pipeline deterministic fixed-threshold baseline.",
        "actions": actions,
    }
    merged, audit = execute_remediation_plan(cleaned_nodes, plan)
    return merged, plan, audit


def run_oracle_clean(
    cleaned_nodes: Dict[str, pd.DataFrame],
    ground_truth: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Upper bound: exact remediation using hidden ground truth."""
    if ground_truth.get("clean_control"):
        return run_raw_merge(cleaned_nodes)

    out_nodes: Dict[str, pd.DataFrame] = {}
    for nid in sorted(cleaned_nodes.keys()):
        df = cleaned_nodes[nid].copy().reset_index(drop=True)
        if nid == ground_truth["residual_aliases"]["node"]:
            df = df.rename(columns=ground_truth["residual_aliases"]["alias_to_canonical"])
        if nid == ground_truth["dirty_rows"]["node"]:
            bad = df.isna().any(axis=1)
            for c in ("dur", "sbytes", "spkts", "dbytes", "sload", "dload", "rate", "smean", "dmean"):
                if c in df.columns:
                    bad = bad | (pd.to_numeric(df[c], errors="coerce") < 0)
            df = df.loc[~bad].reset_index(drop=True)
        if nid == ground_truth["unit_drift"]["node"]:
            rfac = ground_truth["unit_drift"]["recovery_factor"]
            for c in ground_truth["unit_drift"]["cols"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce") * rfac
        if nid == ground_truth["label_noise"]["node"]:
            noisy_set = set(ground_truth["label_noise"]["noisy_indices"])
            keep = [i not in noisy_set for i in range(len(df))]
            df = df.loc[keep].reset_index(drop=True)
        df = df.drop(columns=[ground_truth["shortcut_col"]], errors="ignore")
        out_nodes[nid] = df

    merged = pd.concat([out_nodes[k] for k in sorted(out_nodes.keys())], ignore_index=True)
    plan = {
        "diagnosis_summary": "Oracle-Clean upper bound using hidden ground truth.",
        "actions": [
            {"action": "no_op", "node": nid, "evidence": "Oracle ground truth"}
            for nid in sorted(cleaned_nodes.keys())
        ],
    }
    audit = {
        "method": "Oracle-Clean",
        "merged_rows": int(len(merged)),
        "per_node_final_rows": {k: int(len(v)) for k, v in out_nodes.items()},
    }
    return merged, plan, audit
