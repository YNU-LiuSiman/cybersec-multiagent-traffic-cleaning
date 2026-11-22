"""Node-side isolated diagnostic tools (Python data plane).

HARD ISOLATION RULE #1:
- Each NodeSandbox encapsulates ONLY its own local DataFrame (`_df`).
- Every tool returns strictly whitelisted aggregate statistics (quantiles, counts, MI, residual rates).
- NO row-level data or sample indices may cross the node boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from .infra import (
    InfraReport,
    UNSW_CANONICAL_COLS,
    UNSW_CATEGORICAL_COLS,
)

NON_NEGATIVE_PHYSICAL_COLS = (
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload", "smean", "dmean",
)

KEY_QUANTILE_COLS = (
    "dur", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload", "smean", "dmean",
    "sinpkt", "dinpkt", "sjit", "djit", "tcprtt",
    "synack", "ackdat", "response_body_len",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
)

# Allowed top-level keys in any summary payload crossing node boundary
SUMMARY_WHITELIST_KEYS: Set[str] = {
    "tool", "node_id", "n_rows", "n_cols", "columns", "dtypes",
    "missing_rate_nonzero", "unmapped_columns", "missing_canonical",
    "unmapped_profiles", "label_filter", "n_rows_filtered", "quantiles",
    "laws", "class_counts", "class_fractions", "binary_attack_rate",
    "label_entropy_nats", "mi_scores", "mi_normalized", "top_mi_features",
    "exempt_classes", "excluded_cols", "n_evaluated", "suspected_noise_count",
    "suspected_noise_rate", "high_conf_conflict_count", "high_conf_conflict_rate",
    "mean_self_confidence", "cv_macro_f1", "per_class_suspected_rate", "bad_header_rows_removed",
    "inf_cells_replaced", "rows_with_nan", "rows_with_negative_physical",
    "total_corrupted_rows", "corrupted_row_rate", "offending_columns",
    "status", "note",
}


class IsolationViolationError(RuntimeError):
    """Raised if a tool attempts to emit forbidden/row-level payload across node boundary."""


def assert_summary_safe(payload: Dict[str, Any], max_list_len: int = 80) -> None:
    """Recursively validate that a payload contains only whitelisted summary stats, no row data."""
    if not isinstance(payload, dict):
        raise IsolationViolationError(f"Summary payload must be dict, got {type(payload)}")

    unknown_top = set(payload.keys()) - SUMMARY_WHITELIST_KEYS
    if unknown_top:
        raise IsolationViolationError(f"Non-whitelisted top-level summary keys: {sorted(unknown_top)}")

    def _walk(obj: Any, path: str) -> None:
        if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)):
            raise IsolationViolationError(f"Raw tabular/array object forbidden at {path}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(tok in str(k).lower() for tok in ("row_indices", "sample_ids", "raw_rows", "records")):
                    raise IsolationViolationError(f"Forbidden key '{k}' at {path}")
                _walk(v, f"{path}.{k}")
        elif isinstance(obj, (list, tuple)):
            if len(obj) > max_list_len:
                raise IsolationViolationError(
                    f"List length {len(obj)} exceeds summary limit {max_list_len} at {path}"
                )
            for idx, item in enumerate(obj):
                _walk(item, f"{path}[{idx}]")
        elif isinstance(obj, (str, int, float, bool, np.integer, np.floating)) or obj is None:
            return
        else:
            raise IsolationViolationError(f"Unsupported type {type(obj)} at {path}")

    _walk(payload, "root")


def _safe_float(v: Any, ndigits: int = 6) -> Optional[float]:
    if v is None or pd.isna(v) or np.isinf(v):
        return None
    return round(float(v), ndigits)


def _compute_col_quantiles(series: pd.Series) -> Dict[str, Optional[float]]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p50_pos": None, "zero_frac": None}
    arr = s.to_numpy(dtype=np.float64)
    pos = arr[arr > 0]
    return {
        "p10": _safe_float(np.percentile(arr, 10)),
        "p25": _safe_float(np.percentile(arr, 25)),
        "p50": _safe_float(np.percentile(arr, 50)),
        "p75": _safe_float(np.percentile(arr, 75)),
        "p90": _safe_float(np.percentile(arr, 90)),
        "p50_pos": _safe_float(np.median(pos)) if len(pos) > 0 else 0.0,
        "zero_frac": _safe_float(float((arr == 0).mean()), 4),
    }


class NodeSandbox:
    """Strict per-node sandbox binding all diagnostic tools to ONE local DataFrame."""

    ALLOWED_TOOLS = (
        "schema_profile",
        "quantile_profile",
        "conservation_check",
        "label_histogram",
        "feature_label_mi",
        "confident_learning",
        "row_health",
    )

    def __init__(
        self,
        node_id: str,
        df: pd.DataFrame,
        infra_report: Optional[InfraReport] = None,
        canonical_cols: Sequence[str] = UNSW_CANONICAL_COLS,
    ) -> None:
        self.node_id = node_id
        self._df = df.copy()  # private; never returned directly
        self._infra_report = infra_report
        self._canonical_cols = list(canonical_cols)
        self._last_cl_flagged_indices: List[int] = []  # kept strictly local!

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        if tool_name not in self.ALLOWED_TOOLS:
            raise ValueError(f"Unknown or disallowed node tool: {tool_name}")
        fn = getattr(self, tool_name)
        result = fn(**kwargs)
        assert_summary_safe(result)
        return result

    def schema_profile(self) -> Dict[str, Any]:
        df = self._df
        cols = list(df.columns)
        dtypes = {c: str(df[c].dtype) for c in cols}
        miss = {
            c: round(float(df[c].isna().mean()), 5)
            for c in cols
            if df[c].isna().any()
        }
        canon_set = set(self._canonical_cols)
        unmapped = [c for c in cols if c not in canon_set]
        missing_canon = [c for c in self._canonical_cols if c not in cols]

        # Compute Benign-anchor summary stats for unmapped columns so Coordinator can compare
        # against missing canonical columns on healthy nodes
        normal_df = df[df["attack_cat"].astype(str) == "Normal"] if "attack_cat" in df.columns else df
        unmapped_profiles: Dict[str, Any] = {}
        for c in unmapped:
            unmapped_profiles[c] = _compute_col_quantiles(normal_df[c])

        out = {
            "tool": "schema_profile",
            "node_id": self.node_id,
            "n_rows": int(len(df)),
            "n_cols": int(len(cols)),
            "columns": cols,
            "dtypes": dtypes,
            "missing_rate_nonzero": miss,
            "unmapped_columns": unmapped,
            "missing_canonical": missing_canon,
            "unmapped_profiles": unmapped_profiles,
        }
        assert_summary_safe(out)
        return out

    def quantile_profile(
        self,
        cols: Optional[Sequence[str]] = None,
        label_filter: str = "Normal",
    ) -> Dict[str, Any]:
        df = self._df
        if label_filter and label_filter.upper() != "ALL" and "attack_cat" in df.columns:
            sub = df[df["attack_cat"].astype(str) == label_filter]
        else:
            sub = df

        if cols is None:
            target_cols = [c for c in KEY_QUANTILE_COLS if c in sub.columns]
            # Also include any unmapped numeric columns
            canon_set = set(self._canonical_cols)
            for c in sub.columns:
                if c not in canon_set and c not in target_cols and pd.api.types.is_numeric_dtype(sub[c]):
                    target_cols.append(c)
        else:
            target_cols = [c for c in cols if c in sub.columns]

        qmap: Dict[str, Dict[str, Optional[float]]] = {}
        for c in target_cols:
            qmap[c] = _compute_col_quantiles(sub[c])

        out = {
            "tool": "quantile_profile",
            "node_id": self.node_id,
            "label_filter": label_filter,
            "n_rows_filtered": int(len(sub)),
            "quantiles": qmap,
        }
        assert_summary_safe(out)
        return out

    def conservation_check(self) -> Dict[str, Any]:
        """Verify physical flow conservation laws on positive finite rows.

        Primary laws (5% tol):
          1. sbytes ≈ spkts * smean
          2. dbytes ≈ dpkts * dmean
        Auxiliary order-of-magnitude laws (50% tol, due to UNSW dur rounding):
          3. sload ≈ sbytes * 8 / dur
          4. dload ≈ dbytes * 8 / dur
        """
        df = self._df
        laws: Dict[str, Any] = {}

        def _eval_law(
            name: str,
            req_cols: Sequence[str],
            lhs_fn,
            rhs_fn,
            tol: float,
            role: str,
        ) -> None:
            missing = [c for c in req_cols if c not in df.columns]
            if missing:
                laws[name] = {"status": "missing_columns", "missing": missing, "role": role}
                return
            num = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64) for c in req_cols}
            lhs = lhs_fn(num)
            rhs = rhs_fn(num)
            valid = np.isfinite(lhs) & np.isfinite(rhs) & (lhs > 0) & (rhs > 0)
            if "dur" in req_cols:
                valid = valid & (num["dur"] > 0.001)
            n_val = int(valid.sum())
            if n_val < 20:
                laws[name] = {"status": "insufficient_rows", "n_checked": n_val, "role": role}
                return
            l_v = lhs[valid]
            r_v = rhs[valid]
            rel_err = np.abs(l_v - r_v) / np.maximum(r_v, 1e-12)
            ratios = l_v / np.maximum(r_v, 1e-12)
            laws[name] = {
                "status": "ok",
                "role": role,
                "tolerance": tol,
                "n_checked": n_val,
                "median_rel_err": _safe_float(np.median(rel_err), 4),
                "p90_rel_err": _safe_float(np.percentile(rel_err, 90), 4),
                "violation_rate": _safe_float(float((rel_err > tol).mean()), 4),
                "median_ratio_lhs_over_rhs": _safe_float(np.median(ratios), 6),
            }

        _eval_law(
            "sbytes_eq_spkts_x_smean",
            ("sbytes", "spkts", "smean"),
            lambda d: d["sbytes"],
            lambda d: d["spkts"] * d["smean"],
            tol=0.05,
            role="primary",
        )
        _eval_law(
            "dbytes_eq_dpkts_x_dmean",
            ("dbytes", "dpkts", "dmean"),
            lambda d: d["dbytes"],
            lambda d: d["dpkts"] * d["dmean"],
            tol=0.05,
            role="primary",
        )
        _eval_law(
            "sload_approx_sbytes_x8_div_dur",
            ("sload", "sbytes", "dur"),
            lambda d: d["sload"],
            lambda d: d["sbytes"] * 8.0 / np.where(d["dur"] > 0, d["dur"], np.nan),
            tol=0.50,
            role="auxiliary_order_of_magnitude",
        )
        _eval_law(
            "dload_approx_dbytes_x8_div_dur",
            ("dload", "dbytes", "dur"),
            lambda d: d["dload"],
            lambda d: d["dbytes"] * 8.0 / np.where(d["dur"] > 0, d["dur"], np.nan),
            tol=0.50,
            role="auxiliary_order_of_magnitude",
        )

        out = {
            "tool": "conservation_check",
            "node_id": self.node_id,
            "laws": laws,
        }
        assert_summary_safe(out)
        return out

    def label_histogram(self) -> Dict[str, Any]:
        df = self._df
        vc = df["attack_cat"].astype(str).value_counts()
        total = max(1, len(df))
        counts = {str(k): int(v) for k, v in vc.items()}
        fracs = {str(k): round(float(v / total), 5) for k, v in vc.items()}
        normal_cnt = counts.get("Normal", 0)
        out = {
            "tool": "label_histogram",
            "node_id": self.node_id,
            "n_rows": int(len(df)),
            "class_counts": counts,
            "class_fractions": fracs,
            "binary_attack_rate": round(float(1.0 - normal_cnt / total), 5),
        }
        assert_summary_safe(out)
        return out

    def feature_label_mi(
        self,
        cols: Optional[Sequence[str]] = None,
        sample_size: int = 8000,
    ) -> Dict[str, Any]:
        """Compute mutual information between each numeric feature and `attack_cat`."""
        df = self._df
        drop_cols = {"attack_cat", "label"} | (UNSW_CATEGORICAL_COLS - {"attack_cat"})
        if cols is None:
            feat_cols = [
                c for c in df.columns
                if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
            ]
        else:
            feat_cols = [c for c in cols if c in df.columns and c not in drop_cols]

        rng = np.random.default_rng(42)
        if len(df) > sample_size:
            idx = rng.choice(len(df), size=sample_size, replace=False)
            sub = df.iloc[idx]
        else:
            sub = df

        y_cat = pd.Categorical(sub["attack_cat"].astype(str))
        y_codes = y_cat.codes
        # Label entropy H(Y) in nats for scale-free normalized MI
        probs = np.bincount(y_codes) / max(1, len(y_codes))
        probs = probs[probs > 0]
        h_y = float(-np.sum(probs * np.log(probs)))

        X_num = sub[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(X_num.median()).fillna(0.0)
        mi_vals = mutual_info_classif(
            X_num.to_numpy(dtype=np.float64),
            y_codes,
            discrete_features=False,
            random_state=42,
        )

        mi_scores = {c: round(float(v), 5) for c, v in zip(feat_cols, mi_vals)}
        mi_norm = {
            c: round(float(v / max(h_y, 1e-6)), 5)
            for c, v in zip(feat_cols, mi_vals)
        }
        top_feats = sorted(mi_norm.items(), key=lambda kv: kv[1], reverse=True)[:10]

        out = {
            "tool": "feature_label_mi",
            "node_id": self.node_id,
            "label_entropy_nats": round(h_y, 5),
            "mi_scores": mi_scores,
            "mi_normalized": mi_norm,
            "top_mi_features": [{"col": k, "mi_norm": v, "mi": mi_scores[k]} for k, v in top_feats],
        }
        assert_summary_safe(out)
        return out

    def confident_learning(
        self,
        exempt_classes: Optional[Sequence[str]] = None,
        exclude_cols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run local 3-fold CV Confident Learning. Flagged row indices stay strictly local!"""
        df = self._df
        exempt_set = set(exempt_classes or [])
        excl_set = set(exclude_cols or []) | {"attack_cat", "label"}

        flagged_idx, stats = run_local_confident_learning(
            df=df,
            exempt_classes=exempt_set,
            exclude_cols=excl_set,
        )
        # Keep row indices strictly private on the node!
        self._last_cl_flagged_indices = flagged_idx

        out = {
            "tool": "confident_learning",
            "node_id": self.node_id,
            "exempt_classes": sorted(exempt_set),
            "excluded_cols": sorted(set(exclude_cols or [])),
            "n_evaluated": stats["n_evaluated"],
            "suspected_noise_count": stats["suspected_noise_count"],
            "suspected_noise_rate": stats["suspected_noise_rate"],
            "high_conf_conflict_count": stats["high_conf_conflict_count"],
            "high_conf_conflict_rate": stats["high_conf_conflict_rate"],
            "mean_self_confidence": stats["mean_self_confidence"],
            "cv_macro_f1": stats["cv_macro_f1"],
            "per_class_suspected_rate": stats["per_class_suspected_rate"],
        }
        assert_summary_safe(out)
        return out

    def row_health(self) -> Dict[str, Any]:
        df = self._df
        nan_mask = df.isna().any(axis=1)
        neg_mask = pd.Series(False, index=df.index)
        offending: Dict[str, int] = {}

        for c in df.columns:
            nc = int(df[c].isna().sum())
            if nc > 0:
                offending[f"{c}:nan"] = nc
        for c in NON_NEGATIVE_PHYSICAL_COLS:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                m = vals < 0
                cnt = int(m.sum())
                if cnt > 0:
                    offending[f"{c}:negative"] = cnt
                    neg_mask = neg_mask | m

        tot_bad = int((nan_mask | neg_mask).sum())
        out = {
            "tool": "row_health",
            "node_id": self.node_id,
            "n_rows": int(len(df)),
            "bad_header_rows_removed": int(self._infra_report.bad_header_rows) if self._infra_report else 0,
            "inf_cells_replaced": int(self._infra_report.inf_cells) if self._infra_report else 0,
            "rows_with_nan": int(nan_mask.sum()),
            "rows_with_negative_physical": int(neg_mask.sum()),
            "total_corrupted_rows": tot_bad,
            "corrupted_row_rate": round(float(tot_bad / max(1, len(df))), 5),
            "offending_columns": offending,
        }
        assert_summary_safe(out)
        return out


def run_local_confident_learning(
    df: pd.DataFrame,
    exempt_classes: Set[str],
    exclude_cols: Set[str],
    n_splits: int = 3,
    random_state: int = 42,
) -> tuple[List[int], Dict[str, Any]]:
    """Deterministic local Confident Learning engine (shared by NodeSandbox and Executor).

    Returns (flagged_positional_indices, summary_stats_dict).
    """
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feat_cols].copy()
    cat_cols = [c for c in ("proto", "service", "state") if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    if cat_cols:
        for c in cat_cols:
            X[c] = X[c].astype(str).fillna("__MISSING__")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    X_arr = X.to_numpy(dtype=np.float32)
    y_str = df["attack_cat"].astype(str).to_numpy()
    classes, y_int = np.unique(y_str, return_inverse=True)
    n_classes = len(classes)
    n_rows = len(df)

    # Filter classes with fewer than n_splits samples from stratification issues
    counts = np.bincount(y_int, minlength=n_classes)
    if n_rows < 30 or n_classes < 2 or int(counts.min()) < 2:
        return [], {
            "n_evaluated": n_rows,
            "suspected_noise_count": 0,
            "suspected_noise_rate": 0.0,
            "high_conf_conflict_count": 0,
            "high_conf_conflict_rate": 0.0,
            "mean_self_confidence": 0.0,
            "cv_macro_f1": 0.0,
            "per_class_suspected_rate": {},
        }

    eff_splits = min(n_splits, int(counts.min()))
    skf = StratifiedKFold(n_splits=eff_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros((n_rows, n_classes), dtype=np.float64)

    for tr_idx, va_idx in skf.split(X_arr, y_int):
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
        )
        clf.fit(X_arr[tr_idx], y_int[tr_idx])
        probs = clf.predict_proba(X_arr[va_idx])
        # Align probability columns with global `classes`
        for local_j, cls_code in enumerate(clf.classes_):
            oof_probs[va_idx, int(cls_code)] = probs[:, local_j]

    # Per-class CL self-confidence thresholds t_j
    thresholds = np.zeros(n_classes, dtype=np.float64)
    for j in range(n_classes):
        mask_j = y_int == j
        thresholds[j] = float(oof_probs[mask_j, j].mean()) if mask_j.any() else 0.5

    obs_prob = oof_probs[np.arange(n_rows), y_int]
    pred_cls = np.argmax(oof_probs, axis=1)
    pred_prob = oof_probs[np.arange(n_rows), pred_cls]

    # Standard Confident Learning error rule (noise-calibrated via per-class thresholds t_j):
    # predicted class != observed class AND pred_prob >= t_{pred_cls} AND obs_prob < 0.5 * t_{obs_cls}
    is_cl_error = (
        (pred_cls != y_int)
        & (pred_prob >= thresholds[pred_cls])
        & (obs_prob < 0.5 * thresholds[y_int])
    )
    # Threshold-calibrated high-confidence conflict indicator (remains valid even at high noise rates)
    is_high_conf_conflict = (
        (pred_cls != y_int)
        & (pred_prob >= thresholds[pred_cls])
        & (obs_prob < 0.25 * thresholds[y_int])
    )

    # Apply rare-class exemption
    if exempt_classes:
        not_exempt = np.array([cls not in exempt_classes for cls in y_str], dtype=bool)
        is_cl_error = is_cl_error & not_exempt
        is_high_conf_conflict = is_high_conf_conflict & not_exempt

    flagged_indices = np.where(is_cl_error)[0].tolist()
    cv_macro_f1 = float(f1_score(y_int, pred_cls, average="macro", zero_division=0))

    per_cls_rate: Dict[str, float] = {}
    for j, cname in enumerate(classes):
        mask_j = y_int == j
        if mask_j.sum() >= 20:
            per_cls_rate[str(cname)] = round(float(is_cl_error[mask_j].mean()), 4)

    stats = {
        "n_evaluated": int(n_rows),
        "suspected_noise_count": int(len(flagged_indices)),
        "suspected_noise_rate": round(float(len(flagged_indices) / max(1, n_rows)), 4),
        "high_conf_conflict_count": int(is_high_conf_conflict.sum()),
        "high_conf_conflict_rate": round(float(is_high_conf_conflict.mean()), 4),
        "mean_self_confidence": round(float(obs_prob.mean()), 4),
        "cv_macro_f1": round(cv_macro_f1, 4),
        "per_class_suspected_rate": per_cls_rate,
    }
    return flagged_indices, stats
