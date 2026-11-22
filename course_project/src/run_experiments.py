"""Master reproducible experiment runner for P4 (baselines), P5 (main eval + clean control),
and P6 (stability repeats + statistical tests + result tables).

All results are checkpointed per (seed, method) under `course_project/runs/` and transcripts
under `course_project/transcripts/`.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .agents import get_prompt_hashes, run_full_mas, run_without_coordinator
from .baselines import (
    run_global_cl,
    run_iforest_clean,
    run_oracle_clean,
    run_raw_merge,
    run_static_pipeline,
)
from .evaluate import evaluate_control_plane, evaluate_merged_training
from .executor import execute_remediation_plan
from .infra import clean_unsw_infra
from .llm_client import LLMClient
from .scenario_gen import generate_scenario, save_ground_truth
from .stats import paired_wilcoxon_suite
from .tools_node import NodeSandbox

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
RESULTS_DIR = BASE_DIR / "results"

ALL_METHODS: Tuple[str, ...] = (
    "Raw-Merge",
    "IForest-Clean",
    "Global-CL",
    "Static-Pipeline",
    "Ours-wo-Coordinator",
    "Ours-wo-Anchor",
    "Ours-wo-RareExempt",
    "Ours-Full",
    "Oracle-Clean",
)

NON_LLM_METHODS = ("Raw-Merge", "IForest-Clean", "Global-CL", "Static-Pipeline", "Oracle-Clean")
LLM_METHODS = ("Ours-Full", "Ours-wo-Anchor", "Ours-wo-Coordinator", "Ours-wo-RareExempt")


def _scenario_dir(seed: int) -> Path:
    d = RUNS_DIR / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prepare_cleaned_scenario(seed: int, clean_control: bool):
    sc = generate_scenario(seed=seed, clean_control=clean_control)
    cleaned_nodes = {}
    infra_reports = {}
    for nid, df in sc.nodes.items():
        cdf, rep = clean_unsw_infra(df)
        cleaned_nodes[nid] = cdf
        infra_reports[nid] = rep
    save_ground_truth(sc.ground_truth, _scenario_dir(seed) / "ground_truth.json")
    return sc, cleaned_nodes, infra_reports


def _save_method_result(seed: int, method: str, payload: Dict[str, Any]) -> None:
    out_dir = _scenario_dir(seed)
    (out_dir / f"metrics_{method}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "plan" in payload:
        (out_dir / f"plan_{method}.json").write_text(
            json.dumps(payload["plan"], ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _load_cached(seed: int, method: str) -> Optional[Dict[str, Any]]:
    p = _scenario_dir(seed) / f"metrics_{method}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def run_llm_plans_for_seed(
    seed: int,
    clean_control: bool = False,
    methods: Sequence[str] = ("Ours-Full", "Ours-wo-Anchor", "Ours-wo-Coordinator"),
    force: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Run LLM control-plane planning (can be parallelized across seeds via threads)."""
    sc, cleaned_nodes, infra_reports = _prepare_cleaned_scenario(seed, clean_control)
    plans_out: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        plan_cache = _scenario_dir(seed) / f"llm_raw_{method}.json"
        if plan_cache.exists() and not force:
            plans_out[method] = json.loads(plan_cache.read_text(encoding="utf-8"))
            continue

        sandboxes = {
            nid: NodeSandbox(node_id=nid, df=cleaned_nodes[nid], infra_report=infra_reports[nid])
            for nid in sorted(cleaned_nodes.keys())
        }
        tpath = TRANSCRIPTS_DIR / f"seed_{seed}" / f"{method}.jsonl"
        if tpath.exists():
            tpath.unlink()
        client = LLMClient(transcript_path=tpath)

        if method == "Ours-Full":
            mas_out = run_full_mas(sandboxes, client, label_filter="Normal", max_coord_rounds=3)
        elif method == "Ours-wo-Anchor":
            mas_out = run_full_mas(sandboxes, client, label_filter="ALL", max_coord_rounds=3)
        elif method == "Ours-wo-Coordinator":
            mas_out = run_without_coordinator(sandboxes, client)
        else:
            continue

        plan_cache.write_text(json.dumps(mas_out, ensure_ascii=False, indent=2), encoding="utf-8")
        plans_out[method] = mas_out
        print(
            f"[LLM DONE] seed={seed} method={method} tokens={mas_out['llm_usage']['total_tokens']} "
            f"rounds={mas_out['coord_query_rounds']} elapsed={mas_out['elapsed_s']}s",
            flush=True,
        )

    return plans_out


def evaluate_all_methods_for_seed(
    seed: int,
    clean_control: bool = False,
    methods: Sequence[str] = ALL_METHODS,
    force_eval: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Execute plans and run RF + HistGB downstream evaluation for all requested methods on one seed."""
    sc, cleaned_nodes, infra_reports = _prepare_cleaned_scenario(seed, clean_control)
    gt = sc.ground_truth
    out: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        cached = _load_cached(seed, method)
        if cached is not None and not force_eval:
            out[method] = cached
            continue

        t0 = time.perf_counter()
        llm_meta: Dict[str, Any] = {}

        if method == "Raw-Merge":
            merged, plan, audit = run_raw_merge(cleaned_nodes)
        elif method == "IForest-Clean":
            merged, plan, audit = run_iforest_clean(cleaned_nodes)
        elif method == "Global-CL":
            merged, plan, audit = run_global_cl(cleaned_nodes)
        elif method == "Static-Pipeline":
            merged, plan, audit = run_static_pipeline(cleaned_nodes, infra_reports)
        elif method == "Oracle-Clean":
            merged, plan, audit = run_oracle_clean(cleaned_nodes, gt)
        elif method in ("Ours-Full", "Ours-wo-Anchor", "Ours-wo-Coordinator"):
            raw_path = _scenario_dir(seed) / f"llm_raw_{method}.json"
            if not raw_path.exists():
                run_llm_plans_for_seed(seed, clean_control=clean_control, methods=[method])
            mas_out = json.loads(raw_path.read_text(encoding="utf-8"))
            plan = mas_out["plan"]
            merged, audit = execute_remediation_plan(cleaned_nodes, plan, disable_rare_exempt=False)
            llm_meta = {
                "llm_usage": mas_out.get("llm_usage", {}),
                "coord_query_rounds": mas_out.get("coord_query_rounds", 0),
                "llm_elapsed_s": mas_out.get("elapsed_s", 0.0),
                "prompt_hashes": mas_out.get("prompt_hashes", {}),
            }
        elif method == "Ours-wo-RareExempt":
            # Reuses Ours-Full LLM plan, disables rare-class exemption at Executor level (plan spec §4.1)
            raw_path = _scenario_dir(seed) / "llm_raw_Ours-Full.json"
            if not raw_path.exists():
                run_llm_plans_for_seed(seed, clean_control=clean_control, methods=["Ours-Full"])
            mas_out = json.loads(raw_path.read_text(encoding="utf-8"))
            plan = mas_out["plan"]
            merged, audit = execute_remediation_plan(cleaned_nodes, plan, disable_rare_exempt=True)
            llm_meta = {
                "reused_plan_from": "Ours-Full",
                "disable_rare_exempt": True,
            }
        else:
            raise ValueError(f"Unknown method: {method}")

        cp = evaluate_control_plane(plan, gt, exec_audit=audit)
        # On clean control scenarios we only need control-plane FPR + optionally downstream metrics
        downstream = evaluate_merged_training(merged, sc.test_df)

        # Strip huge cl_flagged_indices list before saving metrics json
        audit_compact = {k: v for k, v in audit.items() if k != "cl_flagged_indices"}
        if "cl_flagged_indices" in audit:
            audit_compact["cl_flagged_counts"] = {k: len(v) for k, v in audit["cl_flagged_indices"].items()}

        payload = {
            "seed": seed,
            "clean_control": clean_control,
            "method": method,
            "downstream": downstream,
            "control_plane": cp,
            "plan": plan,
            "audit": audit_compact,
            "llm_meta": llm_meta,
            "eval_elapsed_s": round(time.perf_counter() - t0, 3),
        }
        _save_method_result(seed, method, payload)
        out[method] = payload
        print(
            f"[EVAL] seed={seed} {method:20s} | RF={downstream['RF']['macro_f1']:.4f} (R4={downstream['RF']['rare4_recall']:.4f}) "
            f"| HGB={downstream['HistGB']['macro_f1']:.4f} (R4={downstream['HistGB']['rare4_recall']:.4f}) "
            f"| R4_ret={downstream['rare4_train_retention']:.3f} | D_fp={cp.get('node_d_false_remediation', 0)}",
            flush=True,
        )

    return out


def run_stability_repeats(
    seeds: Sequence[int] = (100, 101, 102),
    n_repeats: int = 3,
    max_workers: int = 3,
) -> Dict[str, Any]:
    """Run Ours-Full `n_repeats` times on each of `seeds` and measure decision agreement rate."""
    stab_dir = RUNS_DIR / "stability"
    stab_dir.mkdir(parents=True, exist_ok=True)

    def _one_run(seed: int, rep_idx: int) -> Dict[str, Any]:
        cache_p = stab_dir / f"seed_{seed}_rep_{rep_idx}.json"
        if cache_p.exists():
            return json.loads(cache_p.read_text(encoding="utf-8"))
        # rep 0 can reuse the main Ours-Full run if present
        main_p = _scenario_dir(seed) / "llm_raw_Ours-Full.json"
        sc, cleaned_nodes, infra_reports = _prepare_cleaned_scenario(seed, clean_control=False)
        if rep_idx == 0 and main_p.exists():
            mas_out = json.loads(main_p.read_text(encoding="utf-8"))
        else:
            sandboxes = {
                nid: NodeSandbox(node_id=nid, df=cleaned_nodes[nid], infra_report=infra_reports[nid])
                for nid in sorted(cleaned_nodes.keys())
            }
            tpath = TRANSCRIPTS_DIR / "stability" / f"seed_{seed}_rep_{rep_idx}.jsonl"
            if tpath.exists():
                tpath.unlink()
            client = LLMClient(transcript_path=tpath)
            mas_out = run_full_mas(sandboxes, client, label_filter="Normal", max_coord_rounds=3)

        plan = mas_out["plan"]
        merged, audit = execute_remediation_plan(cleaned_nodes, plan)
        cp = evaluate_control_plane(plan, sc.ground_truth, exec_audit=audit)
        # Canonical action signature for exact agreement comparison
        sig = _canonical_action_signature(plan)
        rec = {
            "seed": seed,
            "repeat": rep_idx,
            "signature": sig,
            "control_plane": cp,
            "llm_usage": mas_out.get("llm_usage", {}),
            "coord_query_rounds": mas_out.get("coord_query_rounds", 0),
        }
        cache_p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[STAB] seed={seed} rep={rep_idx} sig_len={len(sig)} cp_ok={cp['unit_drift_factor_acc']==1.0 and cp['shortcut_exact']==1 and cp['noise_node_detected']==1 and cp['node_d_false_remediation']==0}", flush=True)
        return rec

    tasks = [(s, r) for s in seeds for r in range(n_repeats)]
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one_run, s, r) for s, r in tasks]
        for fut in as_completed(futs):
            records.append(fut.result())

    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for rec in records:
        by_seed.setdefault(rec["seed"], []).append(rec)

    summary_per_seed: Dict[str, Any] = {}
    exact_agreements = []
    cp_agreements = []
    for s in seeds:
        reps = sorted(by_seed[s], key=lambda x: x["repeat"])
        sigs = [json.dumps(r["signature"], sort_keys=True) for r in reps]
        # Pairwise agreement across the 3 repeats
        pairs = [(i, j) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
        exact_rate = float(np.mean([sigs[i] == sigs[j] for i, j in pairs]))
        # Key fault-level decision agreement (unit drift, shortcut, noise node, alias, Node D)
        key_vecs = [
            (
                r["control_plane"]["unit_drift_factor_acc"],
                r["control_plane"]["shortcut_exact"],
                r["control_plane"]["residual_alias_acc"],
                r["control_plane"]["noise_node_detected"],
                r["control_plane"]["noise_node_fp"],
                r["control_plane"]["dirty_rows_remediated"],
                r["control_plane"]["node_d_false_remediation"],
            )
            for r in reps
        ]
        cp_rate = float(np.mean([key_vecs[i] == key_vecs[j] for i, j in pairs]))
        exact_agreements.append(exact_rate)
        cp_agreements.append(cp_rate)
        summary_per_seed[str(s)] = {
            "exact_plan_pairwise_agreement": round(exact_rate, 4),
            "control_plane_pairwise_agreement": round(cp_rate, 4),
            "repeats": reps,
        }

    out = {
        "seeds": list(seeds),
        "n_repeats": n_repeats,
        "mean_exact_plan_agreement": round(float(np.mean(exact_agreements)), 4),
        "mean_control_plane_agreement": round(float(np.mean(cp_agreements)), 4),
        "per_seed": summary_per_seed,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "stability.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _canonical_action_signature(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    sig = []
    for a in plan.get("actions", []):
        atype = a.get("action")
        if atype == "no_op":
            continue
        item: Dict[str, Any] = {"action": atype}
        if "node" in a:
            item["node"] = a["node"]
        if "col" in a:
            item["col"] = a["col"]
        if "cols" in a:
            item["cols"] = sorted(a["cols"])
        if "factor" in a:
            item["factor"] = round(float(a["factor"]), 6)
        if "mapping" in a:
            item["mapping"] = dict(sorted(a["mapping"].items()))
        if "rule" in a:
            item["rule"] = a["rule"]
        if "exempt_classes" in a:
            item["exempt_classes"] = sorted(a["exempt_classes"])
        sig.append(item)
    sig.sort(key=lambda x: json.dumps(x, sort_keys=True))
    return sig


def aggregate_and_run_stats(
    eval_seeds: Sequence[int] = tuple(range(100, 110)),
    clean_seeds: Sequence[int] = (200, 201, 202),
) -> Dict[str, Any]:
    """Aggregate all eval_seeds + clean_seeds into CSV tables and stats.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all eval metrics
    records: Dict[str, List[Dict[str, Any]]] = {m: [] for m in ALL_METHODS}
    for s in eval_seeds:
        for m in ALL_METHODS:
            p = _scenario_dir(s) / f"metrics_{m}.json"
            if p.exists():
                records[m].append(json.loads(p.read_text(encoding="utf-8")))

    # 1. Main table (all methods, mean +/- SD across eval seeds)
    main_rows = []
    for m in ALL_METHODS:
        recs = records[m]
        if not recs:
            continue
        rf_f1 = [r["downstream"]["RF"]["macro_f1"] for r in recs]
        rf_r4 = [r["downstream"]["RF"]["rare4_recall"] for r in recs]
        rf_bin = [r["downstream"]["RF"]["binary_attack_f1"] for r in recs]
        hgb_f1 = [r["downstream"]["HistGB"]["macro_f1"] for r in recs]
        hgb_r4 = [r["downstream"]["HistGB"]["rare4_recall"] for r in recs]
        hgb_bin = [r["downstream"]["HistGB"]["binary_attack_f1"] for r in recs]
        r4_ret = [r["downstream"]["rare4_train_retention"] for r in recs]
        main_rows.append(
            {
                "method": m,
                "n_seeds": len(recs),
                "RF_macro_f1_mean": round(float(np.mean(rf_f1)), 4),
                "RF_macro_f1_sd": round(float(np.std(rf_f1, ddof=1)) if len(rf_f1) > 1 else 0.0, 4),
                "RF_rare4_recall_mean": round(float(np.mean(rf_r4)), 4),
                "RF_binary_f1_mean": round(float(np.mean(rf_bin)), 4),
                "HistGB_macro_f1_mean": round(float(np.mean(hgb_f1)), 4),
                "HistGB_macro_f1_sd": round(float(np.std(hgb_f1, ddof=1)) if len(hgb_f1) > 1 else 0.0, 4),
                "HistGB_rare4_recall_mean": round(float(np.mean(hgb_r4)), 4),
                "HistGB_binary_f1_mean": round(float(np.mean(hgb_bin)), 4),
                "rare4_train_retention_mean": round(float(np.mean(r4_ret)), 4),
            }
        )
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(RESULTS_DIR / "main_table.csv", index=False)

    # 2. Ablation table subset
    ablation_methods = ("Ours-Full", "Ours-wo-RareExempt", "Ours-wo-Anchor", "Ours-wo-Coordinator", "Static-Pipeline")
    ablation_df = main_df[main_df["method"].isin(ablation_methods)].copy()
    ablation_df.to_csv(RESULTS_DIR / "ablation.csv", index=False)

    # 3. Control-plane table
    cp_methods = ("Static-Pipeline", "Ours-wo-Coordinator", "Ours-wo-Anchor", "Ours-Full")
    cp_rows = []
    for m in cp_methods:
        recs = records[m]
        if not recs:
            continue
        cps = [r["control_plane"] for r in recs]
        # Clean control FPR
        clean_recs = []
        for cs in clean_seeds:
            p = _scenario_dir(cs) / f"metrics_{m}.json"
            if p.exists():
                clean_recs.append(json.loads(p.read_text(encoding="utf-8")))
        clean_fpr = (
            round(float(np.mean([r["control_plane"]["clean_false_remediation"] for r in clean_recs])), 4)
            if clean_recs
            else None
        )
        # Cost stats
        usages = [r.get("llm_meta", {}).get("llm_usage", {}) for r in recs if r.get("llm_meta", {}).get("llm_usage")]
        avg_tokens = round(float(np.mean([u.get("total_tokens", 0) for u in usages])), 1) if usages else 0.0
        avg_rounds = (
            round(float(np.mean([r.get("llm_meta", {}).get("coord_query_rounds", 0) for r in recs])), 2)
            if usages
            else 0.0
        )
        avg_time = (
            round(float(np.mean([r.get("llm_meta", {}).get("llm_elapsed_s", 0.0) for r in recs])), 2)
            if usages
            else 0.0
        )
        cp_rows.append(
            {
                "method": m,
                "n_eval": len(recs),
                "unit_drift_P": round(float(np.mean([c["unit_drift_precision"] for c in cps])), 4),
                "unit_drift_R": round(float(np.mean([c["unit_drift_recall"] for c in cps])), 4),
                "unit_factor_acc": round(float(np.mean([c["unit_drift_factor_acc"] for c in cps])), 4),
                "shortcut_exact_acc": round(float(np.mean([c["shortcut_exact"] for c in cps])), 4),
                "residual_alias_acc": round(float(np.mean([c["residual_alias_acc"] for c in cps])), 4),
                "noise_node_recall": round(float(np.mean([c["noise_node_detected"] for c in cps])), 4),
                "noise_node_fp_sum": int(sum(c["noise_node_fp"] for c in cps)),
                "noise_sample_f1": round(float(np.mean([c["noise_sample_f1"] for c in cps])), 4),
                "dirty_rows_remediated": round(float(np.mean([c["dirty_rows_remediated"] for c in cps])), 4),
                "node_d_false_remediation_sum": int(sum(c["node_d_false_remediation"] for c in cps)),
                "clean_control_fpr": clean_fpr,
                "avg_tokens": avg_tokens,
                "avg_coord_rounds": avg_rounds,
                "avg_llm_elapsed_s": avg_time,
            }
        )
    cp_df = pd.DataFrame(cp_rows)
    cp_df.to_csv(RESULTS_DIR / "control_plane.csv", index=False)

    # 4. Per-seed detailed table (for plotting paired differences and box/bar charts)
    per_seed_rows = []
    for s in eval_seeds:
        for m in ALL_METHODS:
            p = _scenario_dir(s) / f"metrics_{m}.json"
            if not p.exists():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            per_seed_rows.append(
                {
                    "seed": s,
                    "method": m,
                    "RF_macro_f1": r["downstream"]["RF"]["macro_f1"],
                    "RF_rare4_recall": r["downstream"]["RF"]["rare4_recall"],
                    "RF_binary_f1": r["downstream"]["RF"]["binary_attack_f1"],
                    "HistGB_macro_f1": r["downstream"]["HistGB"]["macro_f1"],
                    "HistGB_rare4_recall": r["downstream"]["HistGB"]["rare4_recall"],
                    "HistGB_binary_f1": r["downstream"]["HistGB"]["binary_attack_f1"],
                    "rare4_train_retention": r["downstream"]["rare4_train_retention"],
                    "node_d_false_remediation": r["control_plane"].get("node_d_false_remediation", 0),
                }
            )
    pd.DataFrame(per_seed_rows).to_csv(RESULTS_DIR / "per_seed_metrics.csv", index=False)

    # 5. Statistical tests: Full vs each comparator, SEPARATELY for RF and HistGB (never pooled!)
    comparators = [
        "Raw-Merge",
        "IForest-Clean",
        "Global-CL",
        "Static-Pipeline",
        "Ours-wo-Coordinator",
        "Ours-wo-Anchor",
        "Ours-wo-RareExempt",
    ]
    stats_out: Dict[str, Any] = {
        "eval_seeds": list(eval_seeds),
        "clean_seeds": list(clean_seeds),
        "prompt_hashes": get_prompt_hashes(),
        "primary_endpoint": "unsw_test_10class_macro_f1",
        "classifiers": {},
    }

    full_recs = sorted(records["Ours-Full"], key=lambda x: x["seed"])
    full_seed_order = [r["seed"] for r in full_recs]
    if len(full_recs) >= 3:
        for clf in ("RF", "HistGB"):
            full_f1 = [r["downstream"][clf]["macro_f1"] for r in full_recs]
            full_r4 = [r["downstream"][clf]["rare4_recall"] for r in full_recs]
            comp_f1: Dict[str, List[float]] = {}
            comp_r4: Dict[str, List[float]] = {}
            for cm in comparators:
                cm_map = {r["seed"]: r for r in records[cm]}
                if all(s in cm_map for s in full_seed_order):
                    comp_f1[cm] = [cm_map[s]["downstream"][clf]["macro_f1"] for s in full_seed_order]
                    comp_r4[cm] = [cm_map[s]["downstream"][clf]["rare4_recall"] for s in full_seed_order]
            stats_out["classifiers"][clf] = {
                "macro_f1": paired_wilcoxon_suite(full_f1, comp_f1, alternative="greater"),
                "rare4_recall": paired_wilcoxon_suite(full_r4, comp_r4, alternative="greater"),
            }

    (RESULTS_DIR / "stats.json").write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["p4_baselines", "p5_llm", "p5_eval", "p6_stability", "p6_aggregate"], required=True)
    parser.add_argument("--seeds", type=str, default="100-109")
    parser.add_argument("--clean-seeds", type=str, default="200-202")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def _parse_seeds(spec: str) -> List[int]:
        out = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    eval_seeds = _parse_seeds(args.seeds)
    clean_seeds = _parse_seeds(args.clean_seeds)

    if args.stage == "p4_baselines":
        for s in eval_seeds:
            evaluate_all_methods_for_seed(s, clean_control=False, methods=NON_LLM_METHODS, force_eval=args.force)
        for s in clean_seeds:
            evaluate_all_methods_for_seed(s, clean_control=True, methods=("Raw-Merge", "Static-Pipeline", "Oracle-Clean"), force_eval=args.force)

    elif args.stage == "p5_llm":
        jobs = [(s, False, ("Ours-Full", "Ours-wo-Anchor", "Ours-wo-Coordinator")) for s in eval_seeds] + [
            (s, True, ("Ours-Full", "Ours-wo-Anchor", "Ours-wo-Coordinator")) for s in clean_seeds
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_llm_plans_for_seed, s, cc, ms, args.force) for s, cc, ms in jobs]
            for fut in as_completed(futs):
                fut.result()

    elif args.stage == "p5_eval":
        for s in eval_seeds:
            evaluate_all_methods_for_seed(s, clean_control=False, methods=ALL_METHODS, force_eval=args.force)
        for s in clean_seeds:
            evaluate_all_methods_for_seed(
                s,
                clean_control=True,
                methods=("Raw-Merge", "Static-Pipeline", "Ours-wo-Coordinator", "Ours-wo-Anchor", "Ours-Full", "Oracle-Clean"),
                force_eval=args.force,
            )

    elif args.stage == "p6_stability":
        res = run_stability_repeats(seeds=eval_seeds[:3], n_repeats=3, max_workers=args.workers)
        print("Stability summary:", json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=2))

    elif args.stage == "p6_aggregate":
        stats = aggregate_and_run_stats(eval_seeds=eval_seeds, clean_seeds=clean_seeds)
        print("Saved results to", RESULTS_DIR)
        print("Main table:\n", pd.read_csv(RESULTS_DIR / "main_table.csv").to_string(index=False))
        print("\nControl-plane table:\n", pd.read_csv(RESULTS_DIR / "control_plane.csv").to_string(index=False))


if __name__ == "__main__":
    main()
