"""CICIDS cross-dataset generalization demonstration (P6 highlight, §4.4).

Reuses the EXACT SAME frozen `node_agent.md` and `coordinator_agent.md` prompts with ZERO edits:
1. Burst Attack vs True Fault:
   - `cic2017_Friday-DDos.csv` chronologically split into 8 slices (DDoS 0.06% -> 77.43%)
   - `cic2017_tuesday_withIP.csv` Benign rows as 9th slice (cross-day Benign stress test)
   - Inject real faults into 2 slices:
       * slice_2: 1000x unit drift on `totlen_fwd_pkts` and `flow_byts_s`
       * slice_5: 25% label flip (BENIGN <-> DDoS)
   - Compare Static-Pipeline (whole-table quantile drift) vs Ours (Full) (Benign-anchor +
     physical conservation + CL via frozen prompts) on slice-level FPR / TPR.
2. Cross-version Audit:
   - `cic2018_head.csv` bad header row + Inf detection + CIC17 regex column alignment count.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .agents import COORD_PROMPT_PATH, NODE_PROMPT_PATH, get_prompt_hashes
from .infra import clean_cic_csv_infra
from .llm_client import LLMClient, extract_json_object
from .tools_coord import CoordinatorToolkit
from .tools_node import assert_summary_safe, run_local_confident_learning

DATA_DIR = Path(r"D:\WARE_HOUSE\desktop_file\云大学科\网安实训中级\网安实训选题\data")
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts" / "cicids_demo"

CIC_KEY_COLS = (
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "totlen_fwd_pkts",
    "totlen_bwd_pkts",
    "fwd_pkts_len_mean",
    "bwd_pkts_len_mean",
    "flow_byts_s",
    "flow_pkts_s",
    "pkts_len_mean",
)


class CICSliceSandbox:
    """Thin adapter exposing the exact same whitelisted NodeSandbox tool interface for CICIDS slices."""

    ALLOWED_TOOLS = (
        "schema_profile",
        "quantile_profile",
        "conservation_check",
        "label_histogram",
        "feature_label_mi",
        "confident_learning",
        "row_health",
    )

    def __init__(self, slice_id: str, df: pd.DataFrame) -> None:
        self.node_id = slice_id
        self._df = df.copy().reset_index(drop=True)

    def call_tool(self, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
        fn = getattr(self, tool_name)
        res = fn(**kwargs)
        assert_summary_safe(res)
        return res

    def schema_profile(self) -> Dict[str, Any]:
        cols = [c for c in CIC_KEY_COLS if c in self._df.columns] + ["attack_cat"]
        return {
            "tool": "schema_profile",
            "node_id": self.node_id,
            "n_rows": int(len(self._df)),
            "n_cols": int(len(cols)),
            "columns": cols,
            "unmapped_columns": [],
            "missing_canonical": [],
            "missing_rate_nonzero": {},
        }

    def quantile_profile(self, label_filter: str = "BENIGN", **_: Any) -> Dict[str, Any]:
        df = self._df
        if label_filter and label_filter.upper() != "ALL":
            sub = df[df["attack_cat"].astype(str).str.upper() == label_filter.upper()]
        else:
            sub = df
        qmap = {}
        for c in CIC_KEY_COLS:
            if c not in sub.columns:
                continue
            s = pd.to_numeric(sub[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            arr = s.to_numpy(dtype=np.float64)
            pos = arr[arr > 0]
            qmap[c] = {
                "p50": round(float(np.median(arr)), 4) if len(arr) else None,
                "p75": round(float(np.percentile(arr, 75)), 4) if len(arr) else None,
                "p90": round(float(np.percentile(arr, 90)), 4) if len(arr) else None,
                "p50_pos": round(float(np.median(pos)), 4) if len(pos) else 0.0,
            }
        return {
            "tool": "quantile_profile",
            "node_id": self.node_id,
            "label_filter": label_filter,
            "n_rows_filtered": int(len(sub)),
            "quantiles": qmap,
        }

    def conservation_check(self) -> Dict[str, Any]:
        df = self._df
        laws = {}
        for name, c_tot, c_cnt, c_mean in [
            ("totlen_fwd_eq_pkts_x_mean", "totlen_fwd_pkts", "tot_fwd_pkts", "fwd_pkts_len_mean"),
            ("totlen_bwd_eq_pkts_x_mean", "totlen_bwd_pkts", "tot_bwd_pkts", "bwd_pkts_len_mean"),
        ]:
            lhs = pd.to_numeric(df[c_tot], errors="coerce").to_numpy(dtype=np.float64)
            rhs = (
                pd.to_numeric(df[c_cnt], errors="coerce").to_numpy(dtype=np.float64)
                * pd.to_numeric(df[c_mean], errors="coerce").to_numpy(dtype=np.float64)
            )
            valid = np.isfinite(lhs) & np.isfinite(rhs) & (lhs > 0) & (rhs > 0)
            if valid.sum() < 20:
                laws[name] = {"status": "insufficient_rows"}
                continue
            rel = np.abs(lhs[valid] - rhs[valid]) / np.maximum(rhs[valid], 1e-12)
            rat = lhs[valid] / np.maximum(rhs[valid], 1e-12)
            laws[name] = {
                "status": "ok",
                "role": "primary",
                "tolerance": 0.05,
                "n_checked": int(valid.sum()),
                "median_rel_err": round(float(np.median(rel)), 4),
                "violation_rate": round(float((rel > 0.05).mean()), 4),
                "median_ratio_lhs_over_rhs": round(float(np.median(rat)), 4),
            }
        return {"tool": "conservation_check", "node_id": self.node_id, "laws": laws}

    def label_histogram(self) -> Dict[str, Any]:
        vc = self._df["attack_cat"].astype(str).value_counts()
        tot = max(1, len(self._df))
        counts = {str(k): int(v) for k, v in vc.items()}
        ben = counts.get("BENIGN", 0)
        return {
            "tool": "label_histogram",
            "node_id": self.node_id,
            "n_rows": int(len(self._df)),
            "class_counts": counts,
            "class_fractions": {k: round(v / tot, 4) for k, v in counts.items()},
            "binary_attack_rate": round(1.0 - ben / tot, 4),
        }

    def feature_label_mi(self) -> Dict[str, Any]:
        return {"tool": "feature_label_mi", "node_id": self.node_id, "mi_normalized": {}}

    def row_health(self) -> Dict[str, Any]:
        return {
            "tool": "row_health",
            "node_id": self.node_id,
            "n_rows": int(len(self._df)),
            "total_corrupted_rows": 0,
            "corrupted_row_rate": 0.0,
        }

    def confident_learning(
        self,
        exempt_classes: Optional[Sequence[str]] = None,
        exclude_cols: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        # Run CL on up to 6000 rows for fast slice audit
        sub = self._df if len(self._df) <= 6000 else self._df.sample(n=6000, random_state=42).reset_index(drop=True)
        feat_cols = [c for c in CIC_KEY_COLS if c in sub.columns]
        use_df = sub[feat_cols + ["attack_cat"]].copy()
        _, stats = run_local_confident_learning(
            df=use_df,
            exempt_classes=set(exempt_classes or []),
            exclude_cols=set(exclude_cols or []) | {"attack_cat", "label"},
        )
        return {
            "tool": "confident_learning",
            "node_id": self.node_id,
            "exempt_classes": sorted(set(exempt_classes or [])),
            "excluded_cols": sorted(set(exclude_cols or [])),
            **stats,
        }


def run_cicids_demonstration() -> Dict[str, Any]:
    """Execute both CICIDS cross-dataset highlights and save results to `results/cicids_demo.json`."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Part 2: Cross-version Audit (CIC18 vs CIC17) --------
    df18_raw = pd.read_csv(DATA_DIR / "cic2018_head.csv", low_memory=False)
    df18_clean, rep18 = clean_cic_csv_infra(df18_raw)
    df17_head = pd.read_csv(DATA_DIR / "cic2017_Friday-DDos.csv", nrows=100)
    df17_clean_head, _ = clean_cic_csv_infra(df17_head)
    raw_overlap = len(set(c.strip() for c in df18_raw.columns) & set(c.strip() for c in df17_head.columns))
    aligned_overlap = len(set(df18_clean.columns) & set(df17_clean_head.columns))
    cross_version = {
        "cic18_rows_in": rep18.rows_in,
        "cic18_rows_out": rep18.rows_out,
        "cic18_bad_header_rows": rep18.bad_header_rows,
        "cic18_bad_header_indices": rep18.bad_header_indices,
        "cic18_inf_cells": rep18.inf_cells,
        "cic17_vs_cic18_raw_exact_overlap": raw_overlap,
        "cic17_vs_cic18_regex_aligned_overlap": aligned_overlap,
    }

    # -------- Part 1: Burst DDoS (8 Fri slices) + Cross-Day Tue Benign (slice_9) --------
    df_fri = pd.read_csv(DATA_DIR / "cic2017_Friday-DDos.csv", low_memory=False)
    c_fri, _ = clean_cic_csv_infra(df_fri)
    c_fri["attack_cat"] = c_fri["label"].astype(str).str.strip().str.upper()

    df_tue = pd.read_csv(DATA_DIR / "cic2017_tuesday_withIP.csv", low_memory=False)
    c_tue, _ = clean_cic_csv_infra(df_tue)
    c_tue["attack_cat"] = c_tue["label"].astype(str).str.strip().str.upper()
    tue_benign = c_tue[c_tue["attack_cat"] == "BENIGN"].head(25000).reset_index(drop=True)

    keep_cols = [c for c in CIC_KEY_COLS if c in c_fri.columns and c in tue_benign.columns] + ["attack_cat"]
    slices: Dict[str, pd.DataFrame] = {}
    n_fri = len(c_fri)
    for i in range(8):
        sub = c_fri.iloc[i * n_fri // 8 : (i + 1) * n_fri // 8][keep_cols].copy().reset_index(drop=True)
        slices[f"slice_{i+1}"] = sub
    slices["slice_9"] = tue_benign[keep_cols].copy().reset_index(drop=True)

    # Inject true faults into 2 slices:
    # - slice_2: 1000x unit drift on totlen_fwd_pkts and flow_byts_s
    # - slice_5: 25% label flip (BENIGN <-> DDOS)
    rng = np.random.default_rng(2026)
    slices["slice_2"]["totlen_fwd_pkts"] = slices["slice_2"]["totlen_fwd_pkts"] * 1000.0
    slices["slice_2"]["flow_byts_s"] = slices["slice_2"]["flow_byts_s"] * 1000.0

    n5 = len(slices["slice_5"])
    flip_idx = rng.choice(n5, size=int(round(0.25 * n5)), replace=False)
    cur_y = slices["slice_5"]["attack_cat"].to_numpy(copy=True)
    cur_y[flip_idx] = np.where(cur_y[flip_idx] == "BENIGN", "DDOS", "BENIGN")
    slices["slice_5"]["attack_cat"] = cur_y

    faulty_slices = {"slice_2": "unit_drift_1000x", "slice_5": "label_flip_25pct"}
    clean_slices = [s for s in sorted(slices.keys()) if s not in faulty_slices]

    sandboxes = {sid: CICSliceSandbox(sid, df) for sid, df in sorted(slices.items())}
    coord = CoordinatorToolkit(sandboxes)
    slice_ids = sorted(sandboxes.keys())

    # ---- Static-Pipeline (whole-table drift >= 5x i.e. |log10|>=0.70, plus CL >= 0.12) ----
    q_all = {sid: sandboxes[sid].call_tool("quantile_profile", label_filter="ALL") for sid in slice_ids}
    cmp_all = coord.compare_anchor_quantiles(q_all)
    static_flagged: Dict[str, List[str]] = {sid: [] for sid in slice_ids}
    for col, info in cmp_all.get("flagged_scale_shifts", {}).items():
        for sid, st in info.get("per_node", {}).items():
            if abs(float(st.get("log10_ratio", 0.0))) >= 0.70:
                static_flagged[sid].append(f"whole_table_drift:{col}(ratio={st['ratio_to_ref']})")
    for sid in slice_ids:
        cl = sandboxes[sid].call_tool("confident_learning")
        if cl["suspected_noise_rate"] >= 0.12:
            static_flagged[sid].append(f"cl_noise:{cl['suspected_noise_rate']}")

    static_fp = sum(1 for s in clean_slices if len(static_flagged[s]) > 0)
    static_tp = sum(1 for s in faulty_slices if len(static_flagged[s]) > 0)
    static_res = {
        "flagged_per_slice": static_flagged,
        "clean_slice_fpr": round(static_fp / len(clean_slices), 4),
        "fault_slice_tpr": round(static_tp / len(faulty_slices), 4),
        "fp_count": static_fp,
        "tp_count": static_tp,
    }

    # ---- Ours (Full): EXACT SAME frozen `coordinator_agent.md` prompt (ZERO edits!) ----
    q_ben = {sid: sandboxes[sid].call_tool("quantile_profile", label_filter="BENIGN") for sid in slice_ids}
    cons_all = {sid: sandboxes[sid].call_tool("conservation_check") for sid in slice_ids}
    hist_all = {sid: sandboxes[sid].call_tool("label_histogram") for sid in slice_ids}
    cmp_ben = coord.compare_anchor_quantiles(q_ben)
    agg_h = coord.aggregate_histograms(hist_all)

    # Also compute per-slice summary metrics for heatmap visualization
    slice_telemetry: List[Dict[str, Any]] = []
    for sid in slice_ids:
        cl_s = sandboxes[sid].call_tool("confident_learning")
        # Max |log10_ratio| in ALL vs BENIGN across shared cols
        max_all = max(
            (abs(float(info["per_node"][sid]["log10_ratio"])) for info in cmp_all.get("flagged_scale_shifts", {}).values() if sid in info["per_node"]),
            default=0.0,
        )
        max_ben = max(
            (abs(float(info["per_node"][sid]["log10_ratio"])) for info in cmp_ben.get("flagged_scale_shifts", {}).values() if sid in info["per_node"]),
            default=0.0,
        )
        fwd_law = cons_all[sid]["laws"].get("totlen_fwd_eq_pkts_x_mean", {})
        slice_telemetry.append(
            {
                "slice": sid,
                "ddos_rate": hist_all[sid]["binary_attack_rate"],
                "ground_truth": faulty_slices.get(sid, "clean"),
                "whole_table_max_abs_log10": round(max_all, 3),
                "benign_anchor_max_abs_log10": round(max_ben, 3),
                "conservation_fwd_violation": fwd_law.get("violation_rate", 0.0),
                "conservation_fwd_ratio": fwd_law.get("median_ratio_lhs_over_rhs", 1.0),
                "cl_suspected_noise_rate": cl_s["suspected_noise_rate"],
                "cl_mean_self_confidence": cl_s["mean_self_confidence"],
                "static_flagged": len(static_flagged[sid]) > 0,
            }
        )

    # Call Coordinator Agent using EXACT frozen coordinator_agent.md (zero prompt edits!)
    tpath = TRANSCRIPTS_DIR / "ours_full_cicids.jsonl"
    if tpath.exists():
        tpath.unlink()
    client = LLMClient(transcript_path=tpath)
    system_prompt = COORD_PROMPT_PATH.read_text(encoding="utf-8")

    coord_packet = {
        "nodes": slice_ids,
        "quantile_anchor_mode": "BENIGN",
        "per_node_label_histograms": hist_all,
        "per_node_conservation": cons_all,
        "compare_anchor_quantiles": cmp_ben,
        "aggregate_histograms": agg_h,
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_node",
                "description": "Dispatch a read-only diagnostic tool call to one isolated node/slice.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node": {"type": "string", "enum": slice_ids},
                        "tool": {"type": "string", "enum": list(CICSliceSandbox.ALLOWED_TOOLS)},
                        "args": {"type": "object"},
                    },
                    "required": ["node", "tool"],
                },
            },
        }
    ]
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Here are the cross-slice statistical summaries from 9 isolated time-ordered network traffic slices "
                "(benign anchor label='BENIGN'). Run any needed follow-up `query_node` calls (such as `confident_learning` "
                "on slices with two classes) and emit your JSON Remediation Plan.\n\n"
                + json.dumps(coord_packet, ensure_ascii=False)
            ),
        },
    ]

    final_content = ""
    coord_rounds = 0
    for rnd in range(3):
        res = client.chat(messages=messages, tools=tools, tool_choice="auto", max_tokens=6144, tag=f"cic_coord_r{rnd}")
        if res.tool_calls:
            coord_rounds += 1
            messages.append({"role": "assistant", "content": res.content or "", "tool_calls": res.tool_calls})
            for tc in res.tool_calls:
                fn_args = json.loads(tc["function"].get("arguments") or "{}")
                out = coord.query_node(fn_args.get("node", ""), fn_args.get("tool", ""), fn_args.get("args") or {})
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "name": "query_node", "content": json.dumps(out, ensure_ascii=False)}
                )
        else:
            final_content = res.content
            break
    else:
        messages.append({"role": "user", "content": "Output ONLY the final JSON Remediation Plan now."})
        final_content = client.chat(messages=messages, max_tokens=6144, tag="cic_coord_final").content

    plan = extract_json_object(final_content)
    ours_flagged_slices = {
        a.get("node")
        for a in plan.get("actions", [])
        if a.get("action") in ("rescale", "local_cl_clean", "quarantine_node", "drop_rows")
    }
    for row in slice_telemetry:
        row["ours_flagged"] = row["slice"] in ours_flagged_slices

    ours_fp = sum(1 for s in clean_slices if s in ours_flagged_slices)
    ours_tp = sum(1 for s in faulty_slices if s in ours_flagged_slices)

    result = {
        "prompt_hashes_unchanged": get_prompt_hashes(),
        "cross_version_cic18": cross_version,
        "faulty_slices_gt": faulty_slices,
        "clean_slices_gt": clean_slices,
        "static_pipeline": static_res,
        "ours_full": {
            "plan": plan,
            "flagged_slices": sorted(ours_flagged_slices),
            "clean_slice_fpr": round(ours_fp / len(clean_slices), 4),
            "fault_slice_tpr": round(ours_tp / len(faulty_slices), 4),
            "fp_count": ours_fp,
            "tp_count": ours_tp,
            "coord_rounds": coord_rounds,
            "llm_usage": client.get_usage_summary(),
        },
        "slice_telemetry": slice_telemetry,
    }
    (RESULTS_DIR / "cicids_demo.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = run_cicids_demonstration()
    print("CIC18 Cross-Version:", res["cross_version_cic18"])
    print("Static-Pipeline FPR/TPR:", res["static_pipeline"]["clean_slice_fpr"], res["static_pipeline"]["fault_slice_tpr"])
    print("Ours (Full) FPR/TPR:", res["ours_full"]["clean_slice_fpr"], res["ours_full"]["fault_slice_tpr"])
