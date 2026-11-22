"""Randomized multi-node UNSW-NB15 fault scenario generator with strict ground-truth separation.

Splits:
- Development scenarios: seeds 0..2 (prompt tuning only)
- Evaluation scenarios: seeds 100..109 (run once after prompt freeze)
- Clean control scenarios: seeds 200..202 (zero faults injected; measures false remediation rate)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .infra import UNSW_CANONICAL_COLS, UNSW_RARE4_CLASSES

DATA_DIR = Path(r"D:\WARE_HOUSE\desktop_file\云大学科\网安实训中级\网安实训选题\data")
UNSW_TRAIN_PATH = DATA_DIR / "unsw_train.parquet"
UNSW_TEST_PATH = DATA_DIR / "unsw_test.parquet"

UNIT_DRIFT_FACTORS: Tuple[float, ...] = (1000.0, 0.125, 1024.0)
UNIT_DRIFT_CANDIDATES: Tuple[str, ...] = ("sbytes", "dbytes", "sload", "dload")

SHORTCUT_COL_POOL: Tuple[str, ...] = (
    "collector_tag_score",
    "probe_qos_hint",
    "sensor_meta_v2",
    "ingest_flag_val",
    "telemetry_bucket_id",
    "tap_priority_metric",
    "export_shard_code",
    "mirror_queue_hint",
    "flow_audit_tag",
    "capture_profile_idx",
)

# Non-regex semantic residual aliases: canonical -> list of plausible domain aliases
RESIDUAL_ALIAS_POOL: Dict[str, List[str]] = {
    "dur": ["session_timespan", "flow_lifespan_val", "conn_elapsed_qty"],
    "rate": ["wire_cadence_idx", "packet_velocity_val", "flow_tempo_metric"],
    "sinpkt": ["egress_interarrival_avg", "tx_gap_millis"],
    "dinpkt": ["ingress_interarrival_avg", "rx_gap_millis"],
    "sjit": ["egress_timing_variance", "tx_jitter_index"],
    "djit": ["ingress_timing_variance", "rx_jitter_index"],
    "tcprtt": ["handshake_roundtrip_val", "tcp_setup_latency"],
    "synack": ["initial_ack_delay", "syn_reply_interval"],
    "ackdat": ["payload_confirm_delay", "ack_data_interval"],
    "response_body_len": ["http_reply_payload_sz", "server_body_octets"],
    "ct_src_dport_ltm": ["recent_src_port_fanout", "window_origin_dport_cnt"],
    "ct_dst_sport_ltm": ["recent_dst_sport_fanout", "window_target_sport_cnt"],
}

ALL_UNSW_CLASSES: Tuple[str, ...] = (
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
)

_CACHED_TRAIN: Optional[pd.DataFrame] = None
_CACHED_TEST: Optional[pd.DataFrame] = None
_CACHED_BASE_SPLIT: Optional[Dict[str, pd.DataFrame]] = None


def load_unsw_train() -> pd.DataFrame:
    global _CACHED_TRAIN
    if _CACHED_TRAIN is None:
        df = pd.read_parquet(UNSW_TRAIN_PATH)
        for c in ("proto", "service", "state", "attack_cat"):
            df[c] = df[c].astype(str)
        _CACHED_TRAIN = df
    return _CACHED_TRAIN.copy()


def load_unsw_test() -> pd.DataFrame:
    global _CACHED_TEST
    if _CACHED_TEST is None:
        df = pd.read_parquet(UNSW_TEST_PATH)
        for c in ("proto", "service", "state", "attack_cat"):
            df[c] = df[c].astype(str)
        _CACHED_TEST = df
    return _CACHED_TEST.copy()


def build_base_role_split(base_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Fixed deterministic 4-role Non-IID partition of unsw_train.

    - Role A (~25k): healthy baseline, stratified non-Rare-4
    - Role B (~24k): target for schema/unit/dirty-row faults, stratified non-Rare-4
    - Role C (~24k): target for label noise + shortcut fault, stratified non-Rare-4
    - Role D (~8.5k): ALL Rare-4 (1,682) + Normal (~3,500) + skewed non-Generic attacks (~3,300)
                      Designed as a genuine Non-IID hard negative: whole-table distribution
                      differs sharply from A/B/C, while Normal-only Benign anchor matches A/C.
    """
    global _CACHED_BASE_SPLIT
    if _CACHED_BASE_SPLIT is not None:
        return {k: v.copy() for k, v in _CACHED_BASE_SPLIT.items()}

    df = load_unsw_train()
    rng = np.random.default_rng(base_seed)

    rare_mask = df["attack_cat"].isin(UNSW_RARE4_CLASSES)
    rare_idx = df.index[rare_mask].to_numpy()
    non_rare_df = df.loc[~rare_mask]

    role_indices: Dict[str, List[int]] = {"A": [], "B": [], "C": [], "D": list(rare_idx)}

    # Allocate Normal: 3,500 to D (Benign anchor), remainder split A:35%, B:32.5%, C:32.5%
    normal_idx = rng.permutation(non_rare_df.index[non_rare_df["attack_cat"] == "Normal"].to_numpy())
    d_norm = 3500
    role_indices["D"].extend(normal_idx[:d_norm].tolist())
    rem_norm = normal_idx[d_norm:]
    n_a = int(round(len(rem_norm) * 0.35))
    n_b = int(round(len(rem_norm) * 0.325))
    role_indices["A"].extend(rem_norm[:n_a].tolist())
    role_indices["B"].extend(rem_norm[n_a : n_a + n_b].tolist())
    role_indices["C"].extend(rem_norm[n_a + n_b :].tolist())

    # Allocate non-Rare-4 attack classes:
    # Give D a realistic skewed slice of Exploits (20%), Fuzzers (15%), DoS (15%), Reconnaissance (10%)
    # and 0% Generic -> creates strong legitimate Non-IID whole-table shift on D while Normal is IID!
    d_attack_fracs = {
        "Generic": 0.0,
        "Exploits": 0.18,
        "Fuzzers": 0.12,
        "DoS": 0.12,
        "Reconnaissance": 0.08,
    }
    for cls, d_frac in d_attack_fracs.items():
        cls_idx = rng.permutation(non_rare_df.index[non_rare_df["attack_cat"] == cls].to_numpy())
        n_d = int(round(len(cls_idx) * d_frac))
        role_indices["D"].extend(cls_idx[:n_d].tolist())
        rem = cls_idx[n_d:]
        na = int(round(len(rem) * 0.35))
        nb = int(round(len(rem) * 0.325))
        role_indices["A"].extend(rem[:na].tolist())
        role_indices["B"].extend(rem[na : na + nb].tolist())
        role_indices["C"].extend(rem[na + nb :].tolist())

    split: Dict[str, pd.DataFrame] = {}
    for role, idxs in role_indices.items():
        idxs_arr = np.array(idxs, dtype=int)
        rng.shuffle(idxs_arr)
        sub = df.loc[idxs_arr].reset_index(drop=True)
        split[role] = sub

    _CACHED_BASE_SPLIT = split
    return {k: v.copy() for k, v in split.items()}


@dataclass
class Scenario:
    seed: int
    clean_control: bool
    nodes: Dict[str, pd.DataFrame]          # node_id -> raw (fault-injected) DataFrame
    test_df: pd.DataFrame                   # unsw_test with random shortcut column appended
    ground_truth: Dict[str, Any]            # NEVER passed to agents or tools!


def generate_scenario(seed: int, clean_control: bool = False) -> Scenario:
    """Generate one randomized multi-node fault scenario deterministically from `seed`."""
    rng = np.random.default_rng(seed)
    base_split = build_base_role_split(base_seed=42)

    node_ids = ["node_1", "node_2", "node_3", "node_4"]
    roles = ["A", "B", "C", "D"]
    perm_nodes = [str(x) for x in rng.permutation(node_ids)]
    role_to_node = dict(zip(roles, perm_nodes))
    node_to_role = {v: k for k, v in role_to_node.items()}

    # Choose shortcut column name & class->numeric encoding (per-scenario random)
    shortcut_col = str(rng.choice(SHORTCUT_COL_POOL))
    class_perm = [str(x) for x in rng.permutation(ALL_UNSW_CLASSES)]
    class_to_code = {cls: float(i) for i, cls in enumerate(class_perm)}
    # Marginal code probabilities for random shortcut values on non-C nodes & test set
    n_classes = len(ALL_UNSW_CLASSES)

    def _random_shortcut(n: int) -> np.ndarray:
        base = rng.integers(0, n_classes, size=n).astype(np.float64)
        return np.round(base + rng.normal(0.0, 0.08, size=n), 4)

    nodes_out: Dict[str, pd.DataFrame] = {}
    for role in roles:
        nid = role_to_node[role]
        df_n = base_split[role].copy()
        # Ensure float64 on numeric columns that may be scaled / hold NaN
        for c in ("dur", "sbytes", "dbytes", "sload", "dload", "smean", "dmean", "spkts", "dpkts", "rate"):
            if c in df_n.columns:
                df_n[c] = df_n[c].astype(np.float64)
        df_n[shortcut_col] = _random_shortcut(len(df_n))
        nodes_out[nid] = df_n

    # Prepare test set with completely random shortcut column
    test_df = load_unsw_test()
    test_df[shortcut_col] = _random_shortcut(len(test_df))

    gt: Dict[str, Any] = {
        "seed": int(seed),
        "clean_control": bool(clean_control),
        "role_to_node": role_to_node,
        "node_to_role": node_to_role,
        "shortcut_col": shortcut_col,
        "shortcut_corrupted_node": None,
        "unit_drift": None,
        "residual_aliases": {},
        "dirty_rows": None,
        "label_noise": None,
        "hard_negative_node": role_to_node["D"],
        "rare4_classes": list(UNSW_RARE4_CLASSES),
    }

    if clean_control:
        return Scenario(
            seed=seed,
            clean_control=True,
            nodes=nodes_out,
            test_df=test_df,
            ground_truth=gt,
        )

    # ---------------- Inject Faults into Role B ----------------
    node_b = role_to_node["B"]
    df_b = nodes_out[node_b].copy()

    # B1: Unit drift on 2-4 byte/load columns (always includes at least one of sbytes/dbytes)
    n_drift = int(rng.integers(2, 5))  # 2, 3, or 4
    drift_factor = float(rng.choice(UNIT_DRIFT_FACTORS))
    if n_drift == 4:
        drift_cols = list(UNIT_DRIFT_CANDIDATES)
    else:
        must_col = str(rng.choice(["sbytes", "dbytes"]))
        others = [c for c in UNIT_DRIFT_CANDIDATES if c != must_col]
        chosen_others = [str(x) for x in rng.choice(others, size=n_drift - 1, replace=False)]
        drift_cols = sorted([must_col] + chosen_others)

    for c in drift_cols:
        df_b[c] = df_b[c].astype(np.float64) * drift_factor

    gt["unit_drift"] = {
        "node": node_b,
        "cols": drift_cols,
        "injected_factor": drift_factor,
        "recovery_factor": float(1.0 / drift_factor),
    }

    # B2: Dirty rows (1-2%): negative sentinel values (-999) + Inf/NaN + 1 repeated header row
    dirty_rate = float(rng.uniform(0.012, 0.020))
    n_dirty = max(10, int(round(len(df_b) * dirty_rate)))
    dirty_idx = rng.choice(len(df_b), size=n_dirty, replace=False)
    dirty_idx_sorted = sorted(int(i) for i in dirty_idx)
    half = n_dirty // 2
    neg_idx = dirty_idx_sorted[:half]
    inf_idx = dirty_idx_sorted[half:]

    # Negative sentinels in non-negative physical columns
    df_b.loc[neg_idx, "dur"] = -999.0
    df_b.loc[neg_idx, "sbytes"] = -999.0
    df_b.loc[neg_idx, "spkts"] = -1.0
    # Inf / NaN burst
    df_b.loc[inf_idx[: len(inf_idx) // 2], "sload"] = np.inf
    df_b.loc[inf_idx[len(inf_idx) // 2 :], "dbytes"] = np.nan

    # Add 1 repeated header row at a random position
    hdr_row = pd.DataFrame([{c: c for c in df_b.columns}])
    insert_pos = int(rng.integers(50, min(500, len(df_b) - 1)))
    df_b = pd.concat([df_b.iloc[:insert_pos], hdr_row, df_b.iloc[insert_pos:]], ignore_index=True)

    gt["dirty_rows"] = {
        "node": node_b,
        "dirty_rate": round(dirty_rate, 4),
        "n_dirty_rows": n_dirty,
        "n_neg_sentinel": len(neg_idx),
        "n_inf_nan": len(inf_idx),
        "n_repeated_header": 1,
    }

    # B3: Residual semantic aliases (2-3 columns not matched by infra regex)
    n_alias = int(rng.integers(2, 4))  # 2 or 3
    alias_canonicals = [str(x) for x in rng.choice(list(RESIDUAL_ALIAS_POOL.keys()), size=n_alias, replace=False)]
    alias_map_old_to_canon: Dict[str, str] = {}
    rename_canon_to_alias: Dict[str, str] = {}
    for canon in alias_canonicals:
        alias_name = str(rng.choice(RESIDUAL_ALIAS_POOL[canon]))
        rename_canon_to_alias[canon] = alias_name
        alias_map_old_to_canon[alias_name] = canon

    df_b = df_b.rename(columns=rename_canon_to_alias)
    gt["residual_aliases"] = {
        "node": node_b,
        "alias_to_canonical": alias_map_old_to_canon,
    }
    nodes_out[node_b] = df_b

    # ---------------- Inject Faults into Role C ----------------
    node_c = role_to_node["C"]
    df_c = nodes_out[node_c].copy()

    # C1: Label noise (22% - 35% flipped to other classes present in Node C;
    # calibrated in P1 so single-node fault yields >= 0.02 global Macro-F1 degradation across seeds)
    noise_rate = float(rng.uniform(0.22, 0.35))
    n_c = len(df_c)
    n_noisy = int(round(n_c * noise_rate))
    noisy_indices = np.sort(rng.choice(n_c, size=n_noisy, replace=False))
    local_classes = sorted(df_c["attack_cat"].unique().tolist())

    true_labels = df_c["attack_cat"].to_numpy(copy=True)
    obs_labels = true_labels.copy()
    for idx in noisy_indices:
        orig = obs_labels[idx]
        candidates = [cls for cls in local_classes if cls != orig]
        obs_labels[idx] = str(rng.choice(candidates))

    df_c["attack_cat"] = obs_labels
    df_c["label"] = (df_c["attack_cat"] != "Normal").astype(np.int64)

    # C2: Shortcut column strongly correlated with OBSERVED label on Node C
    obs_codes = np.array([class_to_code[c] for c in obs_labels], dtype=np.float64)
    df_c[shortcut_col] = np.round(obs_codes + rng.normal(0.0, 0.05, size=n_c), 4)

    gt["label_noise"] = {
        "node": node_c,
        "noise_rate": round(noise_rate, 4),
        "n_noisy_rows": int(n_noisy),
        "noisy_indices": [int(i) for i in noisy_indices],
        "true_attack_cat": true_labels.tolist(),
    }
    gt["shortcut_corrupted_node"] = node_c
    nodes_out[node_c] = df_c

    return Scenario(
        seed=seed,
        clean_control=False,
        nodes=nodes_out,
        test_df=test_df,
        ground_truth=gt,
    )


def save_ground_truth(gt: Dict[str, Any], path: Path) -> None:
    """Save ground truth atomically (compact: omits huge true_attack_cat array)."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = dict(gt)
    if compact.get("label_noise") is not None:
        ln = dict(compact["label_noise"])
        ln.pop("true_attack_cat", None)
        compact["label_noise"] = ln
    tmp = path.with_suffix(f".tmp.{gt.get('seed', 0)}")
    tmp.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
