"""Deterministic Remediation Plan Executor unit tests (P2 acceptance gate)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from course_project.src.executor import (
    PlanValidationError,
    execute_remediation_plan,
    validate_plan,
)


def _make_synthetic_nodes():
    rng = np.random.default_rng(7)
    n = 120

    def _base(n_rows, classes):
        y = rng.choice(classes, size=n_rows)
        sbytes = rng.uniform(200, 1000, size=n_rows)
        # Make features separable by class for CL test
        cls_offset = np.array([{"Normal": 1.0, "Exploits": 50.0, "Worms": 120.0}[c] for c in y])
        return pd.DataFrame({
            "dur": rng.uniform(0.1, 2.0, size=n_rows) + cls_offset * 0.1,
            "proto": ["tcp"] * n_rows,
            "service": ["http"] * n_rows,
            "state": ["FIN"] * n_rows,
            "spkts": np.full(n_rows, 4.0),
            "dpkts": np.full(n_rows, 4.0),
            "sbytes": sbytes + cls_offset * 20.0,
            "dbytes": rng.uniform(100, 500, size=n_rows),
            "sload": rng.uniform(1000, 5000, size=n_rows),
            "dload": rng.uniform(1000, 5000, size=n_rows),
            "smean": (sbytes + cls_offset * 20.0) / 4.0,
            "dmean": np.full(n_rows, 50.0),
            "probe_qos_hint": rng.uniform(0, 5, size=n_rows),
            "attack_cat": y,
            "label": (y != "Normal").astype(int),
        })

    n1 = _base(n, ["Normal", "Exploits"])
    # Node 2: alias + 1000x unit drift + 5 dirty rows
    n2 = _base(n, ["Normal", "Exploits"]).rename(columns={"dur": "session_timespan"})
    n2["sbytes"] = n2["sbytes"] * 1000.0
    n2.loc[0:2, "sbytes"] = -999.0
    n2.loc[3:4, "dbytes"] = np.nan

    # Node 3: 25% label noise + shortcut column
    n3 = _base(n, ["Normal", "Exploits"])
    noisy_idx = list(range(0, 30))
    n3.loc[noisy_idx, "attack_cat"] = np.where(n3.loc[noisy_idx, "attack_cat"] == "Normal", "Exploits", "Normal")
    n3["probe_qos_hint"] = np.where(n3["attack_cat"] == "Normal", 0.0, 1.0)

    # Node 4: Rare class "Worms" + Normal (hard negative)
    n4 = _base(80, ["Normal", "Worms"])
    n4.loc[:19, "attack_cat"] = "Worms"
    return {"node_1": n1, "node_2": n2, "node_3": n3, "node_4": n4}


def test_validate_plan_rejects_bad_actions():
    nodes = ["node_1", "node_2", "node_3", "node_4"]

    with pytest.raises(PlanValidationError):
        validate_plan({"actions": [{"action": "exec_python", "node": "node_1", "evidence": "x"}]}, nodes)

    with pytest.raises(PlanValidationError):
        validate_plan({"actions": [{"action": "no_op", "node": "node_1", "evidence": ""}]}, nodes)

    with pytest.raises(PlanValidationError):
        validate_plan({"actions": [{"action": "rescale", "node": "node_1", "cols": ["sbytes"], "factor": -2, "evidence": "e"}]}, nodes)

    with pytest.raises(PlanValidationError):
        validate_plan({"actions": [{"action": "drop_column", "col": "attack_cat", "evidence": "e"}]}, nodes)

    with pytest.raises(PlanValidationError):
        validate_plan({"actions": [{"action": "no_op", "node": "node_99", "evidence": "e"}]}, nodes)


def test_execute_remediation_plan_end_to_end():
    nodes = _make_synthetic_nodes()
    orig_n4_worms = int((nodes["node_4"]["attack_cat"] == "Worms").sum())

    plan = {
        "actions": [
            {"action": "no_op", "node": "node_1", "evidence": "healthy baseline"},
            {"action": "rename_columns", "node": "node_2", "mapping": {"session_timespan": "dur"}, "evidence": "alias match"},
            {"action": "drop_rows", "node": "node_2", "rule": "nan_or_negative_physical", "evidence": "5 dirty rows"},
            {"action": "rescale", "node": "node_2", "cols": ["sbytes"], "factor": 0.001, "evidence": "1000x unit drift"},
            {"action": "drop_column", "col": "probe_qos_hint", "evidence": "single-node MI outlier"},
            {"action": "local_cl_clean", "node": "node_3", "exempt_classes": ["Worms"], "evidence": "high CL noise"},
            {"action": "no_op", "node": "node_4", "evidence": "Benign anchor normal; holds rare class"},
        ]
    }

    merged, audit = execute_remediation_plan(nodes, plan)

    assert "session_timespan" not in merged.columns
    assert "dur" in merged.columns
    assert "probe_qos_hint" not in merged.columns
    # Node 2 had 5 dirty rows dropped -> 115 left
    assert audit["per_node_final_rows"]["node_2"] == 115
    # Node 2 sbytes rescaled back to ~[200, 2000]
    assert merged["sbytes"].max() < 5000.0
    assert merged["sbytes"].min() > 0.0
    # Node 3 CL caught noisy rows (after shortcut dropped!)
    assert len(audit["cl_flagged_indices"]["node_3"]) >= 15
    # Node 4 untouched -> 100% Worms preserved
    assert audit["per_node_final_rows"]["node_4"] == 80
    assert int((merged["attack_cat"] == "Worms").sum()) == orig_n4_worms


def test_rare_exempt_vs_disable_rare_exempt():
    nodes = _make_synthetic_nodes()
    # Deliberately run local_cl_clean on node_4 with Worms exempt vs disabled
    plan = {
        "actions": [
            {"action": "local_cl_clean", "node": "node_4", "exempt_classes": ["Worms"], "evidence": "test"}
        ]
    }
    orig_worms = int((nodes["node_4"]["attack_cat"] == "Worms").sum())
    merged_exempt, _ = execute_remediation_plan(nodes, plan, disable_rare_exempt=False)
    assert int((merged_exempt["attack_cat"] == "Worms").sum()) == orig_worms

    # Inject a corrupted label on Worms rows to ensure disable_rare_exempt can drop them
    nodes_bad = _make_synthetic_nodes()
    nodes_bad["node_4"].loc[0:9, "sbytes"] = 300.0  # looks like Normal while labeled Worms
    nodes_bad["node_4"].loc[0:9, "dur"] = 0.5
    merged_ex, _ = execute_remediation_plan(nodes_bad, plan, disable_rare_exempt=False)
    merged_noex, _ = execute_remediation_plan(nodes_bad, plan, disable_rare_exempt=True)
    assert int((merged_ex["attack_cat"] == "Worms").sum()) == orig_worms
    assert int((merged_noex["attack_cat"] == "Worms").sum()) < orig_worms
