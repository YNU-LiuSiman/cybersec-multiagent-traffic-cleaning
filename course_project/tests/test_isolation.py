"""Hard isolation & summary-whitelist tests (P2 acceptance gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from course_project.src.infra import clean_unsw_infra
from course_project.src.scenario_gen import generate_scenario
from course_project.src.tools_coord import CoordinatorToolkit
from course_project.src.tools_node import (
    IsolationViolationError,
    NodeSandbox,
    assert_summary_safe,
)


@pytest.fixture(scope="module")
def sample_sandboxes():
    sc = generate_scenario(seed=0, clean_control=False)
    sandboxes = {}
    for nid, df in sc.nodes.items():
        cdf, rep = clean_unsw_infra(df)
        # Keep small slice for fast unit test while preserving all classes
        sub = cdf.head(1200).reset_index(drop=True)
        sandboxes[nid] = NodeSandbox(node_id=nid, df=sub, infra_report=rep)
    return sandboxes


def test_node_sandbox_isolation_and_all_tools_pass_whitelist(sample_sandboxes):
    """Every node tool returns strictly whitelisted summary stats, zero row-level data."""
    for nid, sb in sample_sandboxes.items():
        # Verify sandbox does not hold references to any other node
        for attr_val in sb.__dict__.values():
            if isinstance(attr_val, dict):
                for other_nid in sample_sandboxes:
                    if other_nid != nid:
                        assert other_nid not in attr_val

        res_schema = sb.call_tool("schema_profile")
        res_q = sb.call_tool("quantile_profile", label_filter="Normal")
        res_cons = sb.call_tool("conservation_check")
        res_hist = sb.call_tool("label_histogram")
        res_mi = sb.call_tool("feature_label_mi", sample_size=600)
        res_cl = sb.call_tool("confident_learning", exempt_classes=["Worms"])
        res_health = sb.call_tool("row_health")

        for res in (res_schema, res_q, res_cons, res_hist, res_mi, res_cl, res_health):
            assert res["node_id"] == nid
            assert_summary_safe(res)
            # Ensure no raw indices leaked from confident_learning
            assert "flagged_indices" not in res
            assert "row_indices" not in res


def test_node_tool_rejects_unknown_tool(sample_sandboxes):
    sb = next(iter(sample_sandboxes.values()))
    with pytest.raises(ValueError):
        sb.call_tool("dump_raw_rows")


def test_coordinator_has_no_dataframe_access(sample_sandboxes):
    coord = CoordinatorToolkit(sample_sandboxes)
    # Coordinator must not store any DataFrame or NodeSandbox instance directly
    for attr_val in coord.__dict__.values():
        assert not isinstance(attr_val, pd.DataFrame)
        if isinstance(attr_val, dict):
            for v in attr_val.values():
                assert not isinstance(v, pd.DataFrame)
                assert not isinstance(v, NodeSandbox)

    # Coordinator cross-node summary tools work strictly on JSON summaries
    q_sums = {nid: coord.query_node(nid, "quantile_profile", {"label_filter": "Normal"}) for nid in sample_sandboxes}
    sp_sums = {nid: coord.query_node(nid, "schema_profile") for nid in sample_sandboxes}
    mi_sums = {nid: coord.query_node(nid, "feature_label_mi", {"sample_size": 500}) for nid in sample_sandboxes}
    h_sums = {nid: coord.query_node(nid, "label_histogram") for nid in sample_sandboxes}

    cmp_q = coord.compare_anchor_quantiles(q_sums, sp_sums)
    cmp_mi = coord.compare_mi(mi_sums)
    agg_h = coord.aggregate_histograms(h_sums)

    assert "flagged_scale_shifts" in cmp_q
    assert "top_cross_node_mi_outliers" in cmp_mi
    assert "global_class_counts" in agg_h


def test_summary_whitelist_blocks_leaks():
    # Raw DataFrame
    with pytest.raises(IsolationViolationError):
        assert_summary_safe({"tool": "x", "node_id": "n1", "quantiles": pd.DataFrame({"a": [1]})})

    # Numpy array
    with pytest.raises(IsolationViolationError):
        assert_summary_safe({"tool": "x", "node_id": "n1", "quantiles": np.array([1, 2, 3])})

    # Forbidden key name
    with pytest.raises(IsolationViolationError):
        assert_summary_safe({"tool": "x", "node_id": "n1", "quantiles": {"row_indices": [1, 2]}})

    # Non-whitelisted top-level key
    with pytest.raises(IsolationViolationError):
        assert_summary_safe({"tool": "x", "node_id": "n1", "secret_dump": [1, 2]})

    # Over-long list (row-level dump disguised as list)
    with pytest.raises(IsolationViolationError):
        assert_summary_safe({"tool": "x", "node_id": "n1", "columns": [f"r{i}" for i in range(200)]})
