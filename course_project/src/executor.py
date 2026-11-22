"""Deterministic Remediation Plan Executor (Python data plane).

HARD ISOLATION RULE #2:
- LLMs NEVER mutate DataFrames directly. They emit a structured JSON Remediation Plan.
- All data transformations are validated against a strict action whitelist, executed
  deterministically in canonical order, and fully audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .tools_node import NON_NEGATIVE_PHYSICAL_COLS, run_local_confident_learning

ALLOWED_ACTIONS: Set[str] = {
    "rename_columns",
    "drop_rows",
    "rescale",
    "drop_column",
    "local_cl_clean",
    "quarantine_node",
    "no_op",
}

ALLOWED_DROP_RULES: Set[str] = {
    "nan_or_negative_physical",
    "negative_physical",
    "nan_any",
}

ACTION_EXECUTION_ORDER: Dict[str, int] = {
    "rename_columns": 1,
    "drop_rows": 2,
    "rescale": 3,
    "drop_column": 4,
    "local_cl_clean": 5,
    "quarantine_node": 6,
    "no_op": 7,
}


class PlanValidationError(ValueError):
    """Raised when a Remediation Plan violates the Executor schema or whitelist."""


def validate_plan(plan: Dict[str, Any], valid_nodes: Sequence[str]) -> List[Dict[str, Any]]:
    """Strictly validate a Remediation Plan JSON against the Executor whitelist."""
    if not isinstance(plan, dict):
        raise PlanValidationError("Remediation plan must be a JSON object.")
    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise PlanValidationError("Remediation plan must contain an 'actions' list.")

    node_set = set(valid_nodes)
    validated: List[Dict[str, Any]] = []

    for idx, act in enumerate(actions):
        if not isinstance(act, dict):
            raise PlanValidationError(f"Action #{idx} must be a dict.")
        atype = act.get("action")
        if atype not in ALLOWED_ACTIONS:
            raise PlanValidationError(
                f"Action #{idx} has disallowed action '{atype}'. Allowed: {sorted(ALLOWED_ACTIONS)}"
            )
        evidence = act.get("evidence")
        if not evidence or (isinstance(evidence, str) and not evidence.strip()):
            raise PlanValidationError(f"Action #{idx} ('{atype}') is missing required 'evidence'.")

        if atype == "drop_column":
            col = act.get("col")
            if not isinstance(col, str) or not col.strip():
                raise PlanValidationError(f"Action #{idx} (drop_column) requires non-empty string 'col'.")
            if col in ("attack_cat", "label"):
                raise PlanValidationError(f"Action #{idx} cannot drop label column '{col}'.")
        else:
            node = act.get("node")
            if node not in node_set:
                raise PlanValidationError(
                    f"Action #{idx} ('{atype}') references unknown node '{node}'. Valid: {sorted(node_set)}"
                )

        if atype == "rename_columns":
            mapping = act.get("mapping")
            if not isinstance(mapping, dict) or not mapping:
                raise PlanValidationError(f"Action #{idx} (rename_columns) requires non-empty dict 'mapping'.")
            for k, v in mapping.items():
                if not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip():
                    raise PlanValidationError(f"Action #{idx} (rename_columns) invalid pair {k}->{v}.")
                if v in ("attack_cat", "label"):
                    raise PlanValidationError(f"Action #{idx} cannot rename into label column '{v}'.")

        elif atype == "rescale":
            cols = act.get("cols")
            factor = act.get("factor")
            if not isinstance(cols, list) or not cols or not all(isinstance(c, str) for c in cols):
                raise PlanValidationError(f"Action #{idx} (rescale) requires non-empty list of column strings 'cols'.")
            try:
                fval = float(factor)
            except (TypeError, ValueError) as exc:
                raise PlanValidationError(f"Action #{idx} (rescale) has non-numeric factor '{factor}'.") from exc
            if not np.isfinite(fval) or fval <= 0:
                raise PlanValidationError(f"Action #{idx} (rescale) factor must be finite > 0, got {fval}.")

        elif atype == "drop_rows":
            rule = act.get("rule", "nan_or_negative_physical")
            if rule not in ALLOWED_DROP_RULES:
                raise PlanValidationError(
                    f"Action #{idx} (drop_rows) invalid rule '{rule}'. Allowed: {sorted(ALLOWED_DROP_RULES)}"
                )

        elif atype == "local_cl_clean":
            exempt = act.get("exempt_classes", [])
            if not isinstance(exempt, list) or not all(isinstance(c, str) for c in exempt):
                raise PlanValidationError(f"Action #{idx} (local_cl_clean) 'exempt_classes' must be list[str].")

        validated.append(dict(act))

    return validated


def execute_remediation_plan(
    nodes: Dict[str, pd.DataFrame],
    plan: Dict[str, Any],
    disable_rare_exempt: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute a validated Remediation Plan deterministically and return (merged_df, audit_log)."""
    validated = validate_plan(plan, valid_nodes=list(nodes.keys()))

    # Work on clean copies; attach original row position tracker for CL noise F1 audit
    work: Dict[str, pd.DataFrame] = {}
    orig_pos: Dict[str, np.ndarray] = {}
    for nid, df in nodes.items():
        work[nid] = df.copy().reset_index(drop=True)
        orig_pos[nid] = np.arange(len(work[nid]), dtype=int)

    quarantined: Set[str] = set()
    audit_steps: List[Dict[str, Any]] = []
    cl_flagged_indices: Dict[str, List[int]] = {}

    # Sort deterministically by execution phase while preserving order within phase
    ordered = sorted(
        enumerate(validated),
        key=lambda pair: (ACTION_EXECUTION_ORDER[pair[1]["action"]], pair[0]),
    )

    for _, act in ordered:
        atype = act["action"]

        if atype == "rename_columns":
            nid = act["node"]
            df = work[nid]
            mapping = {
                k: v for k, v in act["mapping"].items()
                if k in df.columns and v not in df.columns
            }
            work[nid] = df.rename(columns=mapping)
            audit_steps.append({
                "action": atype,
                "node": nid,
                "applied_mapping": mapping,
                "evidence": act["evidence"],
            })

        elif atype == "drop_rows":
            nid = act["node"]
            rule = act.get("rule", "nan_or_negative_physical")
            df = work[nid]
            bad = pd.Series(False, index=df.index)
            if rule in ("nan_or_negative_physical", "nan_any"):
                bad = bad | df.isna().any(axis=1)
            if rule in ("nan_or_negative_physical", "negative_physical"):
                for c in NON_NEGATIVE_PHYSICAL_COLS:
                    if c in df.columns:
                        vals = pd.to_numeric(df[c], errors="coerce")
                        bad = bad | (vals < 0)
            n_before = len(df)
            keep_mask = (~bad).to_numpy(dtype=bool)
            work[nid] = df.loc[keep_mask].reset_index(drop=True)
            orig_pos[nid] = orig_pos[nid][keep_mask]
            audit_steps.append({
                "action": atype,
                "node": nid,
                "rule": rule,
                "rows_removed": int(n_before - len(work[nid])),
                "rows_remaining": int(len(work[nid])),
                "evidence": act["evidence"],
            })

        elif atype == "rescale":
            nid = act["node"]
            factor = float(act["factor"])
            df = work[nid]
            applied_cols = []
            for c in act["cols"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce") * factor
                    applied_cols.append(c)
            work[nid] = df
            audit_steps.append({
                "action": atype,
                "node": nid,
                "cols": applied_cols,
                "factor": factor,
                "evidence": act["evidence"],
            })

        elif atype == "drop_column":
            col = act["col"]
            dropped_from = []
            for nid in sorted(work.keys()):
                if col in work[nid].columns:
                    work[nid] = work[nid].drop(columns=[col])
                    dropped_from.append(nid)
            audit_steps.append({
                "action": atype,
                "col": col,
                "dropped_from_nodes": dropped_from,
                "evidence": act["evidence"],
            })

        elif atype == "local_cl_clean":
            nid = act["node"]
            if nid in quarantined:
                continue
            df = work[nid]
            exempt = set() if disable_rare_exempt else set(act.get("exempt_classes", []))
            flagged_local_pos, cl_stats = run_local_confident_learning(
                df=df,
                exempt_classes=exempt,
                exclude_cols={"attack_cat", "label"},
            )
            # Map local positional indices back to original pre-drop indices for GT evaluation
            mapped_orig = [int(orig_pos[nid][i]) for i in flagged_local_pos]
            cl_flagged_indices[nid] = mapped_orig

            keep_mask = np.ones(len(df), dtype=bool)
            if flagged_local_pos:
                keep_mask[flagged_local_pos] = False
            work[nid] = df.loc[keep_mask].reset_index(drop=True)
            orig_pos[nid] = orig_pos[nid][keep_mask]
            audit_steps.append({
                "action": atype,
                "node": nid,
                "exempt_classes": sorted(exempt),
                "disable_rare_exempt": bool(disable_rare_exempt),
                "rows_removed": int(len(flagged_local_pos)),
                "rows_remaining": int(len(work[nid])),
                "cl_stats": cl_stats,
                "evidence": act["evidence"],
            })

        elif atype == "quarantine_node":
            nid = act["node"]
            quarantined.add(nid)
            audit_steps.append({
                "action": atype,
                "node": nid,
                "rows_excluded": int(len(work[nid])),
                "evidence": act["evidence"],
            })

        elif atype == "no_op":
            nid = act["node"]
            audit_steps.append({
                "action": atype,
                "node": nid,
                "evidence": act["evidence"],
            })

    active_dfs = [work[nid] for nid in sorted(work.keys()) if nid not in quarantined]
    if not active_dfs:
        raise RuntimeError("All nodes were quarantined; cannot merge empty training set.")

    merged = pd.concat(active_dfs, ignore_index=True)
    audit = {
        "n_actions_executed": len(audit_steps),
        "quarantined_nodes": sorted(quarantined),
        "per_node_final_rows": {nid: (0 if nid in quarantined else int(len(work[nid]))) for nid in sorted(work.keys())},
        "merged_rows": int(len(merged)),
        "merged_cols": int(len(merged.columns)),
        "cl_flagged_indices": cl_flagged_indices,
        "steps": audit_steps,
    }
    return merged, audit
