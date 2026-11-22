"""Statistical testing utilities for P6:
- Paired one-sided Wilcoxon signed-rank test (preregistered direction: Full > baseline)
- Holm-Bonferroni step-down multiple-testing correction
- Bootstrap 95% CI for paired mean difference
- Matched-pairs rank-biserial correlation effect size (r_rb)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import wilcoxon


def matched_pairs_rank_biserial(diffs: np.ndarray) -> float:
    """Compute matched-pairs rank-biserial correlation r = (W+ - W-) / (W+ + W-)."""
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    abs_nz = np.abs(nz)
    # Average ranks for ties
    order = np.argsort(abs_nz)
    ranks = np.empty_like(abs_nz, dtype=np.float64)
    sorted_vals = abs_nz[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    w_pos = float(ranks[nz > 0].sum())
    w_neg = float(ranks[nz < 0].sum())
    denom = w_pos + w_neg
    return float((w_pos - w_neg) / denom) if denom > 0 else 0.0


def bootstrap_paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap 95% CI for mean(a - b)."""
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diffs)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return {
        "mean_diff": round(float(diffs.mean()), 5),
        "sd_diff": round(float(diffs.std(ddof=1)) if n > 1 else 0.0, 5),
        "ci95_low": round(lo, 5),
        "ci95_high": round(hi, 5),
    }


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values preserving original order."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = round(float(running_max), 6)
    return adjusted


def paired_wilcoxon_suite(
    full_scores: Sequence[float],
    comparator_scores: Dict[str, Sequence[float]],
    alternative: str = "greater",
) -> Dict[str, Any]:
    """Run paired Wilcoxon signed-rank tests (Full vs each comparator) + Holm correction + bootstrap CI."""
    full_arr = np.asarray(full_scores, dtype=np.float64)
    names = list(comparator_scores.keys())
    raw_ps: List[float] = []
    entries: Dict[str, Dict[str, Any]] = {}

    for name in names:
        comp_arr = np.asarray(comparator_scores[name], dtype=np.float64)
        diffs = full_arr - comp_arr
        ci = bootstrap_paired_diff_ci(full_arr, comp_arr)
        r_rb = matched_pairs_rank_biserial(diffs)

        if np.all(diffs == 0):
            stat, pval = 0.0, 1.0
        else:
            res = wilcoxon(diffs, alternative=alternative, zero_method="wilcox")
            stat, pval = float(res.statistic), float(res.pvalue)

        raw_ps.append(pval)
        entries[name] = {
            "n_pairs": int(len(diffs)),
            "full_mean": round(float(full_arr.mean()), 5),
            "full_sd": round(float(full_arr.std(ddof=1)) if len(full_arr) > 1 else 0.0, 5),
            "comp_mean": round(float(comp_arr.mean()), 5),
            "comp_sd": round(float(comp_arr.std(ddof=1)) if len(comp_arr) > 1 else 0.0, 5),
            "wins": int((diffs > 0).sum()),
            "ties": int((diffs == 0).sum()),
            "losses": int((diffs < 0).sum()),
            **ci,
            "wilcoxon_stat": round(stat, 4),
            "p_raw": round(pval, 6),
            "rank_biserial_r": round(r_rb, 4),
        }

    adj_ps = holm_bonferroni(raw_ps)
    for name, adj_p in zip(names, adj_ps):
        entries[name]["p_holm"] = adj_p
        entries[name]["significant_005"] = bool(adj_p < 0.05)

    return {
        "alternative": alternative,
        "comparisons": entries,
    }
