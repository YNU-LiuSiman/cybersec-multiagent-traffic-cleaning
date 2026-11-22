"""Downstream IDS evaluation (RF & HistGB on official unsw_test) and control-plane scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.preprocessing import OrdinalEncoder

from .infra import UNSW_CATEGORICAL_COLS, UNSW_RARE4_CLASSES
from .scenario_gen import ALL_UNSW_CLASSES

CAT_FEATURE_COLS = ("proto", "service", "state")


def _prepare_train_test_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    for_rf: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Align feature columns between merged training data and official test set."""
    drop_cols = {"attack_cat", "label"}
    train_feat_cols = [c for c in train_df.columns if c not in drop_cols]
    # Keep features present in train; if an unmapped alias exists only in train,
    # fill it with NaN in test (representing unseen/misaligned column at inference time).
    feat_cols = sorted(train_feat_cols)

    X_tr = train_df[feat_cols].copy()
    X_te = pd.DataFrame(index=test_df.index)
    for c in feat_cols:
        if c in test_df.columns:
            X_te[c] = test_df[c].values
        else:
            X_te[c] = np.nan

    cat_cols = [c for c in CAT_FEATURE_COLS if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    if cat_cols:
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str).fillna("__MISSING__")
            X_te[c] = X_te[c].astype(str).fillna("__MISSING__")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols])
        X_te[cat_cols] = enc.transform(X_te[cat_cols])

    for c in num_cols:
        X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_te[c] = pd.to_numeric(X_te[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if for_rf:
            med = X_tr[c].median()
            fill_val = 0.0 if pd.isna(med) else float(med)
            X_tr[c] = X_tr[c].fillna(fill_val)
            X_te[c] = X_te[c].fillna(fill_val)

    X_tr_arr = X_tr.to_numpy(dtype=np.float32)
    X_te_arr = X_te.to_numpy(dtype=np.float32)
    y_tr = train_df["attack_cat"].astype(str).to_numpy()
    y_te = test_df["attack_cat"].astype(str).to_numpy()
    return X_tr_arr, y_tr, X_te_arr, y_te, feat_cols


def evaluate_merged_training(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_rare4_count: int = 1682,
    clf_seed: int = 42,
    classifiers: Sequence[str] = ("RF", "HistGB"),
) -> Dict[str, Any]:
    """Train RF and HistGB on merged `train_df` and evaluate on untouched official `unsw_test`."""
    # Filter any rows with missing/invalid attack_cat
    valid_mask = train_df["attack_cat"].astype(str).isin(ALL_UNSW_CLASSES)
    train_clean = train_df.loc[valid_mask].reset_index(drop=True)

    rare4_kept = int(train_clean["attack_cat"].isin(UNSW_RARE4_CLASSES).sum())
    rare4_retention = float(rare4_kept / max(1, original_rare4_count))

    results: Dict[str, Any] = {
        "train_rows": int(len(train_clean)),
        "rare4_kept": rare4_kept,
        "rare4_train_retention": round(rare4_retention, 4),
    }

    labels_list = list(ALL_UNSW_CLASSES)

    for clf_name in classifiers:
        for_rf = clf_name == "RF"
        X_tr, y_tr, X_te, y_te, feat_cols = _prepare_train_test_matrices(
            train_clean, test_df, for_rf=for_rf
        )
        if clf_name == "RF":
            model = RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=clf_seed,
            )
        elif clf_name == "HistGB":
            model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=10,
                min_samples_leaf=10,
                learning_rate=0.08,
                random_state=clf_seed,
            )
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        macro_f1 = float(f1_score(y_te, y_pred, labels=labels_list, average="macro", zero_division=0))
        per_f1 = f1_score(y_te, y_pred, labels=labels_list, average=None, zero_division=0)
        per_rec = recall_score(y_te, y_pred, labels=labels_list, average=None, zero_division=0)

        per_f1_dict = {cls: round(float(v), 4) for cls, v in zip(labels_list, per_f1)}
        per_rec_dict = {cls: round(float(v), 4) for cls, v in zip(labels_list, per_rec)}
        rare4_rec = float(np.mean([per_rec_dict[c] for c in UNSW_RARE4_CLASSES]))

        y_te_bin = (y_te != "Normal").astype(int)
        y_pr_bin = (y_pred != "Normal").astype(int)
        bin_f1 = float(f1_score(y_te_bin, y_pr_bin, zero_division=0))

        results[clf_name] = {
            "macro_f1": round(macro_f1, 4),
            "rare4_recall": round(rare4_rec, 4),
            "binary_attack_f1": round(bin_f1, 4),
            "n_features": len(feat_cols),
            "per_class_f1": per_f1_dict,
            "per_class_recall": per_rec_dict,
        }

    return results


def evaluate_control_plane(
    plan: Dict[str, Any],
    ground_truth: Dict[str, Any],
    exec_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute control-plane precision/recall/accuracy metrics against hidden ground truth."""
    actions = plan.get("actions", [])
    clean_control = bool(ground_truth.get("clean_control", False))
    node_d = ground_truth["hard_negative_node"]

    # 1. Node D false remediation count (any destructive/transform action on Node D:
    #    rescale, quarantine_node, local_cl_clean, drop_rows on Node D)
    node_d_bad_actions = [
        a for a in actions
        if a.get("node") == node_d
        and a.get("action") in ("rescale", "quarantine_node", "local_cl_clean", "drop_rows", "rename_columns")
    ]
    node_d_false_remediation = len(node_d_bad_actions)

    # Clean control false remediation: any non-no_op action
    non_noop_actions = [a for a in actions if a.get("action") != "no_op"]
    if clean_control:
        return {
            "clean_control": True,
            "false_remediation_actions": len(non_noop_actions),
            "clean_false_remediation": int(len(non_noop_actions) > 0),
            "node_d_false_remediation": node_d_false_remediation,
        }

    # 2. Unit drift detection P/R + factor recovery accuracy
    gt_ud = ground_truth["unit_drift"]
    gt_ud_pairs = {(gt_ud["node"], c) for c in gt_ud["cols"]}
    pred_ud_pairs = set()
    pred_factor_matches = 0
    for a in actions:
        if a.get("action") == "rescale":
            nid = a.get("node")
            cols = a.get("cols") or []
            factor = float(a.get("factor", 1.0))
            for c in cols:
                pred_ud_pairs.add((nid, c))
                if (nid, c) in gt_ud_pairs:
                    rel_err = abs(factor - gt_ud["recovery_factor"]) / max(gt_ud["recovery_factor"], 1e-12)
                    if rel_err <= 0.15:
                        pred_factor_matches += 1

    ud_tp = len(gt_ud_pairs & pred_ud_pairs)
    ud_prec = float(ud_tp / len(pred_ud_pairs)) if pred_ud_pairs else 0.0
    ud_rec = float(ud_tp / len(gt_ud_pairs)) if gt_ud_pairs else 1.0
    ud_factor_acc = float(pred_factor_matches / len(gt_ud_pairs)) if gt_ud_pairs else 1.0

    # 3. Shortcut column detection accuracy
    gt_sc = ground_truth["shortcut_col"]
    dropped_cols = {a.get("col") for a in actions if a.get("action") == "drop_column" and a.get("col")}
    shortcut_tp = int(gt_sc in dropped_cols)
    shortcut_fp = len(dropped_cols - {gt_sc})
    shortcut_exact = int(dropped_cols == {gt_sc})

    # 4. Residual alias mapping accuracy
    gt_alias = ground_truth["residual_aliases"]["alias_to_canonical"]
    gt_alias_node = ground_truth["residual_aliases"]["node"]
    pred_alias_map: Dict[str, str] = {}
    for a in actions:
        if a.get("action") == "rename_columns" and a.get("node") == gt_alias_node:
            m = a.get("mapping") or {}
            pred_alias_map.update(m)
    alias_correct = sum(1 for k, v in gt_alias.items() if pred_alias_map.get(k) == v)
    alias_acc = float(alias_correct / len(gt_alias)) if gt_alias else 1.0

    # 5. Label noise node targeting & sample-level noise detection F1 (on Node C)
    gt_ln = ground_truth["label_noise"]
    gt_ln_node = gt_ln["node"]
    cl_nodes = [a.get("node") for a in actions if a.get("action") == "local_cl_clean"]
    noise_node_detected = int(gt_ln_node in cl_nodes)
    noise_node_fp = sum(1 for n in cl_nodes if n != gt_ln_node)

    noise_f1 = 0.0
    noise_prec = 0.0
    noise_rec = 0.0
    if exec_audit and "cl_flagged_indices" in exec_audit and gt_ln_node in exec_audit["cl_flagged_indices"]:
        flagged = set(exec_audit["cl_flagged_indices"][gt_ln_node])
        true_noisy = set(gt_ln["noisy_indices"])
        tp = len(flagged & true_noisy)
        noise_prec = float(tp / len(flagged)) if flagged else 0.0
        noise_rec = float(tp / len(true_noisy)) if true_noisy else 0.0
        noise_f1 = float(2 * noise_prec * noise_rec / (noise_prec + noise_rec)) if (noise_prec + noise_rec) > 0 else 0.0

    # 6. Dirty rows drop check on Node B
    gt_dirty_node = ground_truth["dirty_rows"]["node"]
    dirty_dropped = int(
        any(a.get("action") == "drop_rows" and a.get("node") == gt_dirty_node for a in actions)
    )

    # 7. Quarantine check (no node should be quarantined since all faults are remediable!)
    quarantined_nodes = [a.get("node") for a in actions if a.get("action") == "quarantine_node"]

    return {
        "clean_control": False,
        "unit_drift_precision": round(ud_prec, 4),
        "unit_drift_recall": round(ud_rec, 4),
        "unit_drift_factor_acc": round(ud_factor_acc, 4),
        "shortcut_detected": shortcut_tp,
        "shortcut_fp_drops": shortcut_fp,
        "shortcut_exact": shortcut_exact,
        "residual_alias_acc": round(alias_acc, 4),
        "noise_node_detected": noise_node_detected,
        "noise_node_fp": noise_node_fp,
        "noise_sample_precision": round(noise_prec, 4),
        "noise_sample_recall": round(noise_rec, 4),
        "noise_sample_f1": round(noise_f1, 4),
        "dirty_rows_remediated": dirty_dropped,
        "node_d_false_remediation": node_d_false_remediation,
        "quarantined_nodes": quarantined_nodes,
    }
