"""Coordinator-side summary comparison tools (strictly zero row-level data access).

All tools operate EXCLUSIVELY on node JSON summaries returned by NodeSandbox.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .tools_node import NodeSandbox, assert_summary_safe


class CoordinatorToolkit:
    """Stateless/summary-only diagnostic tools for the Coordinator Agent."""

    def __init__(self, node_sandboxes: Dict[str, NodeSandbox]) -> None:
        # Coordinator holds references ONLY to NodeSandbox.call_tool (never accesses _df directly)
        self._node_ids = sorted(node_sandboxes.keys())
        self._dispatch = {nid: sb.call_tool for nid, sb in node_sandboxes.items()}

    def query_node(self, node: str, tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a whitelisted read-only diagnostic tool on a single node and return its summary."""
        if node not in self._dispatch:
            return {"error": f"Unknown node '{node}'. Valid nodes: {self._node_ids}"}
        call_args = dict(args or {})
        res = self._dispatch[node](tool, **call_args)
        assert_summary_safe(res)
        return res

    @staticmethod
    def compare_anchor_quantiles(
        quantile_summaries: Dict[str, Dict[str, Any]],
        schema_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compare per-node anchor quantiles across nodes via robust cross-node median ratios.

        Returns:
          - `column_ratios`: for each column with any notable ratio deviation (|log10| >= 0.35),
            per-node median, reference median, ratio, and log10_ratio.
          - `all_columns_max_abs_log10`: compact overview across all shared columns.
          - `unmapped_vs_missing_candidates`: for each node with unmapped columns, compares
            their Benign quantiles against the reference Benign quantiles of missing canonical cols.
        """
        node_ids = sorted(quantile_summaries.keys())
        # Collect all columns present in >=2 nodes
        col_nodes: Dict[str, Dict[str, float]] = {}
        col_nodes_p75: Dict[str, Dict[str, float]] = {}

        for nid in node_ids:
            qmap = quantile_summaries[nid].get("quantiles", {})
            for col, stats in qmap.items():
                if not isinstance(stats, dict):
                    continue
                p50 = stats.get("p50")
                p50_pos = stats.get("p50_pos")
                p75 = stats.get("p75")
                # Prefer p50 if > 0, else p50_pos (handles zero-inflated columns like dbytes/dload)
                rep_val = p50 if (p50 is not None and p50 > 0) else p50_pos
                if rep_val is not None and rep_val > 0:
                    col_nodes.setdefault(col, {})[nid] = float(rep_val)
                if p75 is not None and p75 > 0:
                    col_nodes_p75.setdefault(col, {})[nid] = float(p75)

        flagged_shifts: Dict[str, Any] = {}
        overview_max_log10: Dict[str, float] = {}
        ref_medians: Dict[str, float] = {}

        for col, nvals in sorted(col_nodes.items()):
            if len(nvals) < 2:
                continue
            vals = list(nvals.values())
            ref = float(np.median(vals))
            if ref <= 0:
                continue
            ref_medians[col] = round(ref, 6)
            per_node: Dict[str, Any] = {}
            max_abs = 0.0
            for nid, v in sorted(nvals.items()):
                ratio = v / ref
                l10 = math.log10(max(ratio, 1e-12))
                max_abs = max(max_abs, abs(l10))
                per_node[nid] = {
                    "val": round(v, 4),
                    "ratio_to_ref": round(ratio, 5),
                    "log10_ratio": round(l10, 4),
                }
            overview_max_log10[col] = round(max_abs, 4)
            if max_abs >= 0.35:
                flagged_shifts[col] = {
                    "ref_median": round(ref, 4),
                    "max_abs_log10": round(max_abs, 4),
                    "per_node": per_node,
                }

        # Match unmapped columns against missing canonical columns using Benign quantile similarity
        alias_hints: Dict[str, Any] = {}
        if schema_summaries:
            for nid, sp in sorted(schema_summaries.items()):
                unmapped = sp.get("unmapped_columns", [])
                missing_canon = sp.get("missing_canonical", [])
                if not unmapped or not missing_canon:
                    continue
                unmapped_profs = sp.get("unmapped_profiles", {})
                # Gather reference profiles of missing canonical cols from OTHER nodes
                missing_refs: Dict[str, Dict[str, Optional[float]]] = {}
                for mc in missing_canon:
                    other_p50s, other_p90s, other_z = [], [], []
                    for oid in node_ids:
                        if oid == nid:
                            continue
                        oq = quantile_summaries[oid].get("quantiles", {}).get(mc)
                        if isinstance(oq, dict):
                            if oq.get("p50") is not None:
                                other_p50s.append(oq["p50"])
                            if oq.get("p90") is not None:
                                other_p90s.append(oq["p90"])
                            if oq.get("zero_frac") is not None:
                                other_z.append(oq["zero_frac"])
                    missing_refs[mc] = {
                        "ref_p50": round(float(np.median(other_p50s)), 6) if other_p50s else None,
                        "ref_p90": round(float(np.median(other_p90s)), 6) if other_p90s else None,
                        "ref_zero_frac": round(float(np.median(other_z)), 4) if other_z else None,
                    }
                alias_hints[nid] = {
                    "unmapped_columns": {u: unmapped_profs.get(u, {}) for u in unmapped},
                    "missing_canonical_refs": missing_refs,
                }

        return {
            "label_filter_used": list({q.get("label_filter", "unknown") for q in quantile_summaries.values()}),
            "flagged_scale_shifts": flagged_shifts,
            "all_columns_max_abs_log10": overview_max_log10,
            "unmapped_vs_missing_candidates": alias_hints,
        }

    @staticmethod
    def compare_mi(mi_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-node feature-label MI outlier analysis (detects single-node shortcut columns)."""
        node_ids = sorted(mi_summaries.keys())
        col_mi: Dict[str, Dict[str, float]] = {}
        for nid in node_ids:
            mn = mi_summaries[nid].get("mi_normalized", {})
            for col, val in mn.items():
                col_mi.setdefault(col, {})[nid] = float(val)

        rows: List[Dict[str, Any]] = []
        for col, nvals in col_mi.items():
            if len(nvals) < 2:
                continue
            vals = np.array([nvals[n] for n in sorted(nvals.keys())], dtype=np.float64)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            scale = max(1.4826 * mad, 0.02)  # floor prevents /0 when 3 nodes have ~0 MI
            max_node = max(nvals.items(), key=lambda kv: kv[1])[0]
            max_val = float(nvals[max_node])
            min_val = float(np.min(vals))
            gap = max_val - med
            robust_z = (max_val - med) / scale
            other_vals = [v for k, v in nvals.items() if k != max_node]
            other_max = float(max(other_vals)) if other_vals else 0.0

            rows.append(
                {
                    "col": col,
                    "max_node": max_node,
                    "max_mi_norm": round(max_val, 4),
                    "median_mi_norm": round(med, 4),
                    "other_nodes_max_mi_norm": round(other_max, 4),
                    "gap_max_minus_median": round(gap, 4),
                    "robust_z": round(robust_z, 2),
                    "per_node_mi_norm": {k: round(v, 4) for k, v in sorted(nvals.items())},
                }
            )

        rows.sort(key=lambda r: (r["gap_max_minus_median"], r["robust_z"]), reverse=True)
        return {
            "n_features_compared": len(rows),
            "top_cross_node_mi_outliers": rows[:10],
        }

    @staticmethod
    def aggregate_histograms(hist_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate per-node class counts into a global histogram (counts only, zero samples)."""
        global_counts: Dict[str, int] = {}
        per_class_node_share: Dict[str, Dict[str, float]] = {}
        total_rows = 0

        for nid, hs in sorted(hist_summaries.items()):
            total_rows += int(hs.get("n_rows", 0))
            for cls, cnt in hs.get("class_counts", {}).items():
                global_counts[cls] = global_counts.get(cls, 0) + int(cnt)

        for cls, tot_c in sorted(global_counts.items()):
            shares = {}
            for nid, hs in sorted(hist_summaries.items()):
                c = int(hs.get("class_counts", {}).get(cls, 0))
                if c > 0:
                    shares[nid] = round(float(c / max(1, tot_c)), 4)
            per_class_node_share[cls] = shares

        global_fracs = {
            cls: round(float(cnt / max(1, total_rows)), 5)
            for cls, cnt in sorted(global_counts.items(), key=lambda kv: kv[1], reverse=True)
        }
        globally_rare = [
            cls for cls, cnt in sorted(global_counts.items())
            if cnt < 1000 or (cnt / max(1, total_rows)) < 0.01
        ]
        # Identify nodes holding unique/concentrated rare classes
        rare_host_nodes = sorted(
            {
                nid
                for cls in globally_rare
                for nid, sh in per_class_node_share.get(cls, {}).items()
                if sh >= 0.80
            }
        )

        return {
            "total_rows": total_rows,
            "global_class_counts": dict(sorted(global_counts.items(), key=lambda kv: kv[1], reverse=True)),
            "global_class_fractions": global_fracs,
            "globally_rare_classes": globally_rare,
            "per_class_node_share": per_class_node_share,
            "rare_class_concentrated_nodes": rare_host_nodes,
        }
