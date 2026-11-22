"""Shared infrastructure layer: header cleanup, bad-row & Inf/NaN repair, regex schema alignment.

Used identically by ALL methods (baseline & proposed) so that regex alias alignment and basic
CSV/header sanitization are NOT misattributed as LLM innovations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Canonical UNSW-NB15 schema (36 columns)
UNSW_CANONICAL_COLS: List[str] = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports",
    "attack_cat", "label",
]

UNSW_CATEGORICAL_COLS = {"proto", "service", "state", "attack_cat"}
UNSW_LABEL_COLS = {"attack_cat", "label"}
UNSW_RARE4_CLASSES = ("Analysis", "Backdoor", "Shellcode", "Worms")

# Standard deterministic regex patterns for UNSW-NB15 (common naming variations only)
UNSW_REGEX_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(dur|duration|flow_dur|flow_duration)$", re.I), "dur"),
    (re.compile(r"^(proto|protocol)$", re.I), "proto"),
    (re.compile(r"^(service|svc)$", re.I), "service"),
    (re.compile(r"^(state|conn_state)$", re.I), "state"),
    (re.compile(r"^(spkts|src_pkts|s_pkts|src_packets)$", re.I), "spkts"),
    (re.compile(r"^(dpkts|dst_pkts|d_pkts|dst_packets)$", re.I), "dpkts"),
    (re.compile(r"^(sbytes|src_bytes|s_bytes)$", re.I), "sbytes"),
    (re.compile(r"^(dbytes|dst_bytes|d_bytes)$", re.I), "dbytes"),
    (re.compile(r"^(rate|pkt_rate|flow_rate)$", re.I), "rate"),
    (re.compile(r"^(sload|src_load|s_load)$", re.I), "sload"),
    (re.compile(r"^(dload|dst_load|d_load)$", re.I), "dload"),
    (re.compile(r"^(sloss|src_loss)$", re.I), "sloss"),
    (re.compile(r"^(dloss|dst_loss)$", re.I), "dloss"),
    (re.compile(r"^(sinpkt|src_inpkt|s_inpkt)$", re.I), "sinpkt"),
    (re.compile(r"^(dinpkt|dst_inpkt|d_inpkt)$", re.I), "dinpkt"),
    (re.compile(r"^(sjit|src_jit|src_jitter)$", re.I), "sjit"),
    (re.compile(r"^(djit|dst_jit|dst_jitter)$", re.I), "djit"),
    (re.compile(r"^(swin|src_win)$", re.I), "swin"),
    (re.compile(r"^(dwin|dst_win)$", re.I), "dwin"),
    (re.compile(r"^(stcpb|src_tcpb)$", re.I), "stcpb"),
    (re.compile(r"^(dtcpb|dst_tcpb)$", re.I), "dtcpb"),
    (re.compile(r"^(tcprtt|tcp_rtt)$", re.I), "tcprtt"),
    (re.compile(r"^(synack|syn_ack)$", re.I), "synack"),
    (re.compile(r"^(ackdat|ack_dat)$", re.I), "ackdat"),
    (re.compile(r"^(smean|src_mean|s_mean)$", re.I), "smean"),
    (re.compile(r"^(dmean|dst_mean|d_mean)$", re.I), "dmean"),
    (re.compile(r"^(trans_depth)$", re.I), "trans_depth"),
    (re.compile(r"^(response_body_len|resp_body_len)$", re.I), "response_body_len"),
    (re.compile(r"^(ct_src_dport_ltm)$", re.I), "ct_src_dport_ltm"),
    (re.compile(r"^(ct_dst_sport_ltm)$", re.I), "ct_dst_sport_ltm"),
    (re.compile(r"^(is_ftp_login)$", re.I), "is_ftp_login"),
    (re.compile(r"^(ct_ftp_cmd)$", re.I), "ct_ftp_cmd"),
    (re.compile(r"^(ct_flw_http_mthd)$", re.I), "ct_flw_http_mthd"),
    (re.compile(r"^(is_sm_ips_ports)$", re.I), "is_sm_ips_ports"),
    (re.compile(r"^(attack_cat|attack_category|category)$", re.I), "attack_cat"),
    (re.compile(r"^(label|is_attack)$", re.I), "label"),
]


@dataclass
class InfraReport:
    rows_in: int = 0
    rows_out: int = 0
    bad_header_rows: int = 0
    bad_header_indices: List[int] = field(default_factory=list)
    inf_cells: int = 0
    nan_cells_before: int = 0
    nan_cells_after_inf: int = 0
    corrupted_numeric_rows: int = 0
    regex_mapped: Dict[str, str] = field(default_factory=dict)
    unmapped_columns: List[str] = field(default_factory=list)
    missing_canonical: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "bad_header_rows": self.bad_header_rows,
            "bad_header_indices": self.bad_header_indices[:20],
            "inf_cells": self.inf_cells,
            "nan_cells_before": self.nan_cells_before,
            "nan_cells_after_inf": self.nan_cells_after_inf,
            "corrupted_numeric_rows": self.corrupted_numeric_rows,
            "regex_mapped": dict(self.regex_mapped),
            "unmapped_columns": list(self.unmapped_columns),
            "missing_canonical": list(self.missing_canonical),
        }


def _normalize_token(col: str) -> str:
    s = str(col).strip()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def clean_unsw_infra(
    df: pd.DataFrame,
    canonical_cols: Sequence[str] = UNSW_CANONICAL_COLS,
) -> Tuple[pd.DataFrame, InfraReport]:
    """Run shared infrastructure cleaning on a UNSW-NB15 node DataFrame.

    Steps:
    1. Strip/normalize header whitespace.
    2. Detect and remove repeated header rows.
    3. Apply deterministic regex schema alignment (leaves unknown/residual aliases untouched).
    4. Replace +/-Inf with NaN (recorded in report); coerce numeric columns safely while
       preserving non-numeric categoricals and leaving NaN/dirty rows countable for diagnosis.
    """
    report = InfraReport(rows_in=len(df))
    out = df.copy()

    # 1. Strip header whitespace
    out.columns = [str(c).strip() for c in out.columns]

    # 2. Detect repeated header rows
    col_names_lower = {c.lower() for c in out.columns}
    if len(out) > 0:
        first_col = out.columns[0]
        last_col = out.columns[-1]
        is_rep_hdr = (
            out[first_col].astype(str).str.strip().str.lower() == first_col.lower()
        ) | (
            out[last_col].astype(str).str.strip().str.lower() == last_col.lower()
        )
        # Also catch rows where >=3 cells equal their own column names
        if not is_rep_hdr.any():
            sample_cols = list(out.columns[: min(6, len(out.columns))])
            matches = sum(
                (out[c].astype(str).str.strip().str.lower() == c.lower()).astype(int)
                for c in sample_cols
            )
            is_rep_hdr = matches >= 3
        bad_idx = out.index[is_rep_hdr].tolist()
        if bad_idx:
            report.bad_header_rows = len(bad_idx)
            report.bad_header_indices = [int(i) for i in bad_idx]
            out = out.loc[~is_rep_hdr].reset_index(drop=True)

    # 3. Regex schema alignment
    rename_map: Dict[str, str] = {}
    used_targets = set()
    for col in out.columns:
        norm = _normalize_token(col)
        matched = None
        for pat, target in UNSW_REGEX_PATTERNS:
            if pat.match(norm):
                matched = target
                break
        if matched and matched not in used_targets:
            if col != matched:
                report.regex_mapped[col] = matched
            rename_map[col] = matched
            used_targets.add(matched)
        else:
            rename_map[col] = col

    out = out.rename(columns=rename_map)

    canonical_set = set(canonical_cols)
    report.unmapped_columns = [c for c in out.columns if c not in canonical_set]
    report.missing_canonical = [c for c in canonical_cols if c not in out.columns]

    # 4. Numeric coercion and Inf -> NaN
    report.nan_cells_before = int(out.isna().sum().sum())
    inf_total = 0
    for col in out.columns:
        if col in UNSW_CATEGORICAL_COLS:
            out[col] = out[col].astype(str)
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            vals = out[col].to_numpy(dtype=np.float64, copy=False)
            inf_mask = np.isinf(vals)
            cnt = int(inf_mask.sum())
            if cnt > 0:
                inf_total += cnt
                out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        else:
            # Try numeric coercion if mostly numeric (e.g., corrupted string rows in a numeric col)
            coerced = pd.to_numeric(out[col], errors="coerce")
            non_null_orig = out[col].notna().sum()
            if non_null_orig > 0 and coerced.notna().sum() / non_null_orig >= 0.5:
                inf_mask = np.isinf(coerced.to_numpy(dtype=np.float64, copy=False))
                cnt = int(inf_mask.sum())
                if cnt > 0:
                    inf_total += cnt
                    coerced = coerced.replace([np.inf, -np.inf], np.nan)
                out[col] = coerced

    report.inf_cells = inf_total
    report.nan_cells_after_inf = int(out.isna().sum().sum())

    # Count rows that have NaN or negative values in non-negative physical columns
    non_neg_cols = [
        c for c in ("dur", "spkts", "dpkts", "sbytes", "dbytes", "sload", "dload", "smean", "dmean")
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]
    bad_num_mask = out.isna().any(axis=1)
    for c in non_neg_cols:
        bad_num_mask = bad_num_mask | (out[c] < 0)
    report.corrupted_numeric_rows = int(bad_num_mask.sum())
    report.rows_out = len(out)
    return out, report


# ---------------- CICIDS cross-version infra ----------------

def _cic_canonical_key(col: str) -> str:
    """Deterministic regex canonicalization for CICIDS-2017 / CICIDS-2018 column headers."""
    s = str(col).strip().lower()
    s = re.sub(r"[/\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    replacements = [
        (r"\bdestination\b", "dst"),
        (r"\bsource\b", "src"),
        (r"\btotal\b", "tot"),
        (r"\btotlen\b", "tot len"),
        (r"\blength\b", "len"),
        (r"\bforward\b", "fwd"),
        (r"\bbackward\b", "bwd"),
        (r"\bpackets?\b", "pkts"),
        (r"\bpkts?\b", "pkts"),
        (r"\bbytes?\b", "byts"),
        (r"\bbyts?\b", "byts"),
        (r"\bheader\b", "hdr"),
        (r"\bsegment\b", "seg"),
        (r"\baverage\b", "avg"),
        (r"\bvariance\b", "var"),
        (r"\bminimum\b", "min"),
        (r"\bmaximum\b", "max"),
        (r"\bflags?\b", "flag"),
        (r"\bcount\b", "cnt"),
        (r"\bcounts\b", "cnt"),
        (r"\binit\b", "init"),
        (r"\bwin\b", "win"),
        (r"\bwindow\b", "win"),
        (r"\bsubflow\b", "subflow"),
        (r"\bdown\s+up\b", "down up"),
        (r"\bof\b", ""),
        (r"\bin\b", ""),
        (r"\bsize\b", "size"),
        (r"\bbulk\b", "blk"),
        (r"\brate\b", "rate"),
        (r"\bact\s+data\s+pkt\b", "act data pkts"),
        (r"\bmin\s+seg\s+size\b", "seg size min"),
        (r"\bavg\s+fwd\s+seg\s+size\b", "fwd seg size avg"),
        (r"\bavg\s+bwd\s+seg\s+size\b", "bwd seg size avg"),
        (r"\bfwd\s+avg\s+byts\s+blk\b", "fwd byts b avg"),
        (r"\bfwd\s+avg\s+pkts\s+blk\b", "fwd pkts b avg"),
        (r"\bfwd\s+avg\s+blk\s+rate\b", "fwd blk rate avg"),
        (r"\bbwd\s+avg\s+byts\s+blk\b", "bwd byts b avg"),
        (r"\bbwd\s+avg\s+pkts\s+blk\b", "bwd pkts b avg"),
        (r"\bbwd\s+avg\s+blk\s+rate\b", "bwd blk rate avg"),
        (r"\binit\s+win\s+byts\s+fwd\b", "init fwd win byts"),
        (r"\binit\s+win\s+byts\s+bwd\b", "init bwd win byts"),
        (r"\binit\s+fwd\s+win\s+byts\b", "init fwd win byts"),
        (r"\binit\s+bwd\s+win\s+byts\b", "init bwd win byts"),
        (r"\bfwd\s+act\s+data\s+pkts\b", "fwd act data pkts"),
        (r"\bact\s+data\s+pkts\s+fwd\b", "fwd act data pkts"),
        (r"\bfwd\s+seg\s+size\s+min\b", "fwd seg size min"),
        (r"\bseg\s+size\s+min\s+fwd\b", "fwd seg size min"),
        (r"\bcwe\s+flag\s+cnt\b", "cwe flag cnt"),
        (r"\bcwr\s+flag\s+cnt\b", "cwe flag cnt"),
        (r"\bfwd\s+hdr\s+len\.\d+\b", "fwd hdr len dup"),
        (r"\bpacket\b", "pkts"),
    ]
    for pat, repl in replacements:
        s = re.sub(pat, repl, s)
    s = re.sub(r"\s+", "_", s).strip("_")
    # Post-normalize common word-order inversions between CIC17 and CIC18
    order_fixes = {
        "tot_len_fwd_pkts": "totlen_fwd_pkts",
        "tot_len_bwd_pkts": "totlen_bwd_pkts",
        "fwd_pkt_len_max": "fwd_pkts_len_max",
        "fwd_pkt_len_min": "fwd_pkts_len_min",
        "fwd_pkt_len_mean": "fwd_pkts_len_mean",
        "fwd_pkt_len_std": "fwd_pkts_len_std",
        "bwd_pkt_len_max": "bwd_pkts_len_max",
        "bwd_pkt_len_min": "bwd_pkts_len_min",
        "bwd_pkt_len_mean": "bwd_pkts_len_mean",
        "bwd_pkt_len_std": "bwd_pkts_len_std",
        "min_pkts_len": "pkts_len_min",
        "max_pkts_len": "pkts_len_max",
        "avg_pkts_size": "pkts_size_avg",
        "pkts_len_avg": "pkts_len_mean",
        "fwd_byts_blk_avg": "fwd_byts_b_avg",
        "fwd_pkts_blk_avg": "fwd_pkts_b_avg",
        "bwd_byts_blk_avg": "bwd_byts_b_avg",
        "bwd_pkts_blk_avg": "bwd_pkts_b_avg",
        "fwd_bulk_rate_avg": "fwd_blk_rate_avg",
        "bwd_bulk_rate_avg": "bwd_blk_rate_avg",
        "fwd_init_win_byts": "init_fwd_win_byts",
        "bwd_init_win_byts": "init_bwd_win_byts",
        "init_win_bytes_forward": "init_fwd_win_byts",
        "init_win_bytes_backward": "init_bwd_win_byts",
        "act_data_pkt_fwd": "fwd_act_data_pkts",
        "min_seg_size_forward": "fwd_seg_size_min",
    }
    return order_fixes.get(s, s)


def clean_cic_csv_infra(df: pd.DataFrame) -> Tuple[pd.DataFrame, InfraReport]:
    """Shared infrastructure cleaning for CICIDS-2017 / CICIDS-2018 CSV tables."""
    report = InfraReport(rows_in=len(df))
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    # Detect repeated header rows (such as CIC18 row 413)
    if len(out) > 0:
        first_c = out.columns[0]
        last_c = out.columns[-1]
        is_bad_hdr = (
            (out[first_c].astype(str).str.strip().str.lower() == first_c.lower())
            | (out[last_c].astype(str).str.strip().str.lower() == last_c.lower())
        )
        bad_idx = out.index[is_bad_hdr].tolist()
        if bad_idx:
            report.bad_header_rows = len(bad_idx)
            report.bad_header_indices = [int(i) for i in bad_idx]
            out = out.loc[~is_bad_hdr].reset_index(drop=True)

    # Canonicalize column names deterministically via regex
    new_cols: List[str] = []
    seen: Dict[str, int] = {}
    for col in out.columns:
        key = _cic_canonical_key(col)
        if key in seen:
            seen[key] += 1
            key = f"{key}_{seen[key]}"
        else:
            seen[key] = 0
        if col != key:
            report.regex_mapped[col] = key
        new_cols.append(key)
    out.columns = new_cols

    # Coerce numeric columns and detect Inf / NaN
    non_num = {"label", "timestamp", "src_ip", "dst_ip", "flow_id"}
    report.nan_cells_before = int(out.isna().sum().sum())
    inf_total = 0
    for col in out.columns:
        if col in non_num:
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        vals = coerced.to_numpy(dtype=np.float64, copy=False)
        inf_mask = np.isinf(vals)
        cnt = int(inf_mask.sum())
        if cnt > 0:
            inf_total += cnt
            coerced = coerced.replace([np.inf, -np.inf], np.nan)
        out[col] = coerced

    report.inf_cells = inf_total
    report.nan_cells_after_inf = int(out.isna().sum().sum())
    report.rows_out = len(out)
    return out, report
