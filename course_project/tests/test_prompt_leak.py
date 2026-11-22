"""Prompt anti-leakage audit tests (P3 acceptance gate).

Fails build if ANY forbidden dataset-specific column name, unit factor, node ID/role,
threshold literal, dataset name, class name, shortcut name, or alias name appears in prompts.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from course_project.src.infra import UNSW_CANONICAL_COLS, UNSW_RARE4_CLASSES
from course_project.src.scenario_gen import (
    ALL_UNSW_CLASSES,
    RESIDUAL_ALIAS_POOL,
    SHORTCUT_COL_POOL,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Explicit forbidden tokens (word-boundary case-insensitive regexes)
FORBIDDEN_LITERALS = [
    # Dataset names
    "unsw", "nb15", "cicids", "cic17", "cic18", "nsl-kdd", "kddtrain", "kddtest",
    # Specific node IDs / roles
    "node_1", "node_2", "node_3", "node_4",
    "node a", "node b", "node c", "node d",
    "node 1", "node 2", "node 3", "node 4",
    "node1", "node2", "node3", "node4",
    "role a", "role b", "role c", "role d",
    # Specific unit drift factors
    "1000", "1024", "0.125", "0.001", "1/8", "1e-3", "1e3", "10^3",
    # Specific threshold literals
    "0.15", "0.30", "0.25", "0.05", "0.50", "0.75", "0.20", "0.35",
    "5%", "15%", "25%", "30%", "50%", "75%",
    # Rare-4 label shortcut
    "rare-4", "rare4",
]

# All UNSW canonical columns (except completely generic English words like "rate", "state", "service", "label")
GENERIC_ENGLISH_WORDS = {"rate", "state", "service", "label"}
FORBIDDEN_COLS = [c for c in UNSW_CANONICAL_COLS if c not in GENERIC_ENGLISH_WORDS]

# All shortcut pool names and residual alias names
FORBIDDEN_INJECTED = list(SHORTCUT_COL_POOL) + [
    alias for aliases in RESIDUAL_ALIAS_POOL.values() for alias in aliases
]

# All UNSW class names (except generic English word "Normal" / "Analysis" -> actually block ALL attack names!)
FORBIDDEN_CLASSES = [c for c in ALL_UNSW_CLASSES if c != "Normal"]


def _prompt_files():
    files = sorted(PROMPTS_DIR.glob("*.md"))
    assert files, f"No prompt files found in {PROMPTS_DIR}"
    return files


@pytest.mark.parametrize("prompt_path", _prompt_files(), ids=lambda p: p.name)
def test_no_forbidden_leakage_tokens(prompt_path: Path):
    text = prompt_path.read_text(encoding="utf-8")
    violations = []

    for tok in FORBIDDEN_LITERALS:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok) + r"(?![A-Za-z0-9_])", re.I)
        if pat.search(text):
            violations.append(f"literal:{tok}")

    for col in FORBIDDEN_COLS:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(col) + r"(?![A-Za-z0-9_])", re.I)
        if pat.search(text):
            violations.append(f"col:{col}")

    for inj in FORBIDDEN_INJECTED:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(inj) + r"(?![A-Za-z0-9_])", re.I)
        if pat.search(text):
            violations.append(f"injected:{inj}")

    for cls in FORBIDDEN_CLASSES:
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(cls) + r"(?![A-Za-z0-9_])", re.I)
        if pat.search(text):
            violations.append(f"class:{cls}")

    assert not violations, f"Prompt leakage detected in {prompt_path.name}: {violations}"
