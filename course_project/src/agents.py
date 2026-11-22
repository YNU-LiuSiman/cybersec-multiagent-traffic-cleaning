"""Multi-Agent Control Plane (Node Agents + Coordinator Agent) over strict isolated sandboxes.

Supports:
- Ours (Full): Isolated Node Agents (Benign anchor) + Coordinator Agent (<=3 query rounds)
- Ours w/o Anchor: Same pipeline using whole-table quantiles (`label_filter="ALL"`) instead of Benign anchor
- Ours w/o Coordinator: Local-only Node Agents with zero cross-node summary comparison
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .executor import PlanValidationError, validate_plan
from .llm_client import LLMClient, extract_json_object
from .tools_coord import CoordinatorToolkit
from .tools_node import NodeSandbox

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
NODE_PROMPT_PATH = PROMPTS_DIR / "node_agent.md"
COORD_PROMPT_PATH = PROMPTS_DIR / "coordinator_agent.md"


def get_prompt_hashes() -> Dict[str, str]:
    """Return SHA256 hashes of frozen prompt files for reproducibility auditing."""
    hashes = {}
    for p in (NODE_PROMPT_PATH, COORD_PROMPT_PATH):
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _node_tool_schemas(default_label_filter: str = "Normal") -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "schema_profile",
                "description": "Inspect node columns, dtypes, missing rates, unmapped columns, and missing canonical columns.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "quantile_profile",
                "description": f"Compute numeric feature quantiles (default label_filter='{default_label_filter}').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_filter": {
                            "type": "string",
                            "description": "Class filter for anchor quantiles (e.g. 'Normal' or 'ALL').",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "conservation_check",
                "description": "Check physical flow conservation laws (bytes vs pkts*mean, load vs bytes*8/dur).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "label_histogram",
                "description": "Compute local class counts, fractions, and binary attack rate.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "feature_label_mi",
                "description": "Compute normalized mutual information between each numeric feature and the target label.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "row_health",
                "description": "Inspect counts of NaN rows, negative physical quantities, and infra-repaired header/Inf rows.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "confident_learning",
                "description": "Run local 3-fold CV Confident Learning label noise audit (flagged indices stay strictly local).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "exempt_classes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Classes exempted from flagging.",
                        },
                        "exclude_cols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Columns to exclude during CV (e.g. suspected shortcut columns).",
                        },
                    },
                },
            },
        },
    ]


def _coord_tool_schemas(node_ids: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "query_node",
                "description": (
                    "Dispatch a read-only diagnostic tool call to one isolated node and receive its "
                    "whitelisted statistical summary (zero raw rows). Ideal for running "
                    "confident_learning with exclude_cols (suspected shortcut) and exempt_classes (globally rare classes)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node": {
                            "type": "string",
                            "enum": node_ids,
                            "description": "Target node identifier.",
                        },
                        "tool": {
                            "type": "string",
                            "enum": list(NodeSandbox.ALLOWED_TOOLS),
                            "description": "Whitelisted node diagnostic tool name.",
                        },
                        "args": {
                            "type": "object",
                            "description": "Optional keyword arguments for the node tool (e.g. exempt_classes, exclude_cols, label_filter).",
                        },
                    },
                    "required": ["node", "tool"],
                },
            },
        }
    ]


def run_node_agent(
    node_id: str,
    sandbox: NodeSandbox,
    client: LLMClient,
    label_filter: str = "Normal",
    max_tool_rounds: int = 5,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Run one isolated Node Agent over its own NodeSandbox.

    Returns:
        (parsed_node_report_json, collected_tool_summaries)
    """
    system_prompt = NODE_PROMPT_PATH.read_text(encoding="utf-8")
    tools = _node_tool_schemas(default_label_filter=label_filter)

    user_msg = (
        f"Audit your assigned isolated node: `{node_id}`. "
        f"Use label_filter='{label_filter}' when calling `quantile_profile`. "
        "Run your diagnostic tools and return your final JSON audit report."
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    collected: Dict[str, Dict[str, Any]] = {}
    final_content = ""

    for rnd in range(max_tool_rounds):
        res = client.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
            tag=f"node_{node_id}_r{rnd}",
        )
        if res.tool_calls:
            # Append assistant message with tool_calls
            messages.append(
                {
                    "role": "assistant",
                    "content": res.content or "",
                    "tool_calls": res.tool_calls,
                }
            )
            for tc in res.tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    fn_args = {}
                # Enforce label_filter consistency for ablation (`w/o Anchor` uses "ALL")
                if fn_name == "quantile_profile":
                    fn_args["label_filter"] = label_filter
                try:
                    tool_out = sandbox.call_tool(fn_name, **fn_args)
                    collected[fn_name] = tool_out
                except Exception as exc:
                    tool_out = {"error": str(exc)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": json.dumps(tool_out, ensure_ascii=False),
                    }
                )
        else:
            final_content = res.content
            break
    else:
        # Force final JSON summary if max tool rounds reached
        messages.append(
            {
                "role": "user",
                "content": "You have finished gathering tool evidence. Now output ONLY the final JSON audit report.",
            }
        )
        res = client.chat(messages=messages, max_tokens=4096, tag=f"node_{node_id}_final")
        final_content = res.content

    # Parse JSON with up to 2 retries
    parsed: Optional[Dict[str, Any]] = None
    last_err = ""
    for attempt in range(3):
        try:
            parsed = extract_json_object(final_content)
            break
        except Exception as exc:
            last_err = str(exc)
            if attempt < 2:
                client.json_retries += 1
                messages.append({"role": "assistant", "content": final_content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your response was not valid JSON ({last_err}). Reply ONLY with a single concise valid JSON object.",
                    }
                )
                res = client.chat(messages=messages, max_tokens=4096, tag=f"node_{node_id}_json_retry_{attempt+1}")
                final_content = res.content
    if parsed is None:
        parsed = {
            "node_id": node_id,
            "node_summary": {"parse_error": last_err},
            "suspicions": [],
            "confidence": 0.0,
        }

    # Ensure all 6 core summaries exist in `collected` (in case LLM skipped one optional call)
    for req_tool, req_kwargs in [
        ("schema_profile", {}),
        ("quantile_profile", {"label_filter": label_filter}),
        ("conservation_check", {}),
        ("label_histogram", {}),
        ("feature_label_mi", {}),
        ("row_health", {}),
    ]:
        if req_tool not in collected:
            collected[req_tool] = sandbox.call_tool(req_tool, **req_kwargs)

    return parsed, collected


def run_full_mas(
    sandboxes: Dict[str, NodeSandbox],
    client: LLMClient,
    label_filter: str = "Normal",
    max_coord_rounds: int = 3,
) -> Dict[str, Any]:
    """Run the full LLM Control Plane (Node Agents -> Coordinator Agent with <=3 query rounds).

    Set `label_filter="ALL"` for the `Ours w/o Anchor` ablation.
    """
    t0 = time.perf_counter()
    node_ids = sorted(sandboxes.keys())
    coord_tools = CoordinatorToolkit(sandboxes)

    # Step 1: Isolated Node Agents audit their own sandboxes
    node_reports: Dict[str, Dict[str, Any]] = {}
    node_summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for nid in node_ids:
        rep, sums = run_node_agent(
            node_id=nid,
            sandbox=sandboxes[nid],
            client=client,
            label_filter=label_filter,
        )
        node_reports[nid] = rep
        node_summaries[nid] = sums

    # Step 2: Coordinator-side cross-node summary comparisons (strictly zero raw rows)
    q_sums = {nid: node_summaries[nid]["quantile_profile"] for nid in node_ids}
    sp_sums = {nid: node_summaries[nid]["schema_profile"] for nid in node_ids}
    mi_sums = {nid: node_summaries[nid]["feature_label_mi"] for nid in node_ids}
    h_sums = {nid: node_summaries[nid]["label_histogram"] for nid in node_ids}

    cmp_quantiles = coord_tools.compare_anchor_quantiles(q_sums, sp_sums)
    cmp_mi = coord_tools.compare_mi(mi_sums)
    agg_hist = coord_tools.aggregate_histograms(h_sums)

    # Build compact, evidence-rich packet for Coordinator Agent
    coord_packet = {
        "nodes": node_ids,
        "quantile_anchor_mode": label_filter,
        "node_agent_reports": node_reports,
        "per_node_row_health": {nid: node_summaries[nid]["row_health"] for nid in node_ids},
        "per_node_conservation": {nid: node_summaries[nid]["conservation_check"] for nid in node_ids},
        "per_node_schema_issues": {
            nid: {
                "unmapped_columns": sp_sums[nid]["unmapped_columns"],
                "missing_canonical": sp_sums[nid]["missing_canonical"],
                "missing_rate_nonzero": sp_sums[nid]["missing_rate_nonzero"],
            }
            for nid in node_ids
        },
        "compare_anchor_quantiles": cmp_quantiles,
        "compare_mi": cmp_mi,
        "aggregate_histograms": agg_hist,
    }

    system_prompt = COORD_PROMPT_PATH.read_text(encoding="utf-8")
    tools = _coord_tool_schemas(node_ids)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Here are the cross-node statistical summaries and isolated Node Agent reports. "
                "You may call `query_node` (up to 3 rounds) for follow-up diagnostics (for example, "
                "running `confident_learning` with any suspected shortcut column excluded and globally "
                "rare classes exempted to compare true label noise across nodes). "
                "Then emit your final JSON Remediation Plan.\n\n"
                + json.dumps(coord_packet, ensure_ascii=False)
            ),
        },
    ]

    coord_rounds_used = 0
    query_log: List[Dict[str, Any]] = []
    final_content = ""

    for rnd in range(max_coord_rounds):
        res = client.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=6144,
            tag=f"coord_r{rnd}",
        )
        if res.tool_calls:
            coord_rounds_used += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": res.content or "",
                    "tool_calls": res.tool_calls,
                }
            )
            for tc in res.tool_calls:
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    fn_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    fn_args = {}
                target_node = fn_args.get("node", "")
                target_tool = fn_args.get("tool", "")
                tool_kwargs = fn_args.get("args") or {}
                if target_tool == "quantile_profile":
                    tool_kwargs["label_filter"] = label_filter
                try:
                    out = coord_tools.query_node(target_node, target_tool, tool_kwargs)
                except Exception as exc:
                    out = {"error": str(exc)}
                query_log.append(
                    {
                        "round": coord_rounds_used,
                        "node": target_node,
                        "tool": target_tool,
                        "args": tool_kwargs,
                        "summary_keys": list(out.keys()),
                        "result": out,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": "query_node",
                        "content": json.dumps(out, ensure_ascii=False),
                    }
                )
        else:
            final_content = res.content
            break
    else:
        messages.append(
            {
                "role": "user",
                "content": "Maximum follow-up query rounds reached. Output ONLY the final JSON Remediation Plan now.",
            }
        )
        res = client.chat(messages=messages, max_tokens=6144, tag="coord_final")
        final_content = res.content

    # Parse and strictly validate Remediation Plan JSON (<= 2 retries)
    plan: Optional[Dict[str, Any]] = None
    last_err = ""
    for attempt in range(3):
        try:
            cand = extract_json_object(final_content)
            validate_plan(cand, valid_nodes=node_ids)
            plan = cand
            break
        except (ValueError, PlanValidationError) as exc:
            last_err = str(exc)
            if attempt < 2:
                client.json_retries += 1
                messages.append({"role": "assistant", "content": final_content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Remediation Plan validation failed: {last_err}. "
                            "Fix the JSON strictly following the allowed action whitelist and concise 'evidence' fields. "
                            "Reply ONLY with the corrected JSON object."
                        ),
                    }
                )
                res = client.chat(messages=messages, max_tokens=6144, tag=f"coord_json_retry_{attempt+1}")
                final_content = res.content

    if plan is None:
        # Fallback safe no_op plan if all retries fail (recorded honestly as failure!)
        plan = {
            "diagnosis_summary": f"PLAN_VALIDATION_FAILED: {last_err}",
            "globally_rare_classes": agg_hist.get("globally_rare_classes", []),
            "actions": [
                {"action": "no_op", "node": nid, "evidence": f"fallback due to parse error: {last_err}"}
                for nid in node_ids
            ],
            "validation_failed": True,
        }

    elapsed_s = round(time.perf_counter() - t0, 3)
    return {
        "plan": plan,
        "node_reports": node_reports,
        "coord_packet_overview": {
            "flagged_scale_shifts": cmp_quantiles.get("flagged_scale_shifts", {}),
            "top_mi_outliers": cmp_mi.get("top_cross_node_mi_outliers", [])[:5],
            "globally_rare_classes": agg_hist.get("globally_rare_classes", []),
        },
        "query_log": query_log,
        "coord_query_rounds": coord_rounds_used,
        "elapsed_s": elapsed_s,
        "llm_usage": client.get_usage_summary(),
        "prompt_hashes": get_prompt_hashes(),
    }


def run_without_coordinator(
    sandboxes: Dict[str, NodeSandbox],
    client: LLMClient,
) -> Dict[str, Any]:
    """Ablation `Ours w/o Coordinator`: each Node Agent decides its own local actions in isolation.

    Zero cross-node comparison, zero global histogram (uses local histogram only).
    """
    t0 = time.perf_counter()
    node_ids = sorted(sandboxes.keys())
    base_node_prompt = NODE_PROMPT_PATH.read_text(encoding="utf-8")
    all_actions: List[Dict[str, Any]] = []
    per_node_outputs: Dict[str, Any] = {}

    for nid in node_ids:
        sb = sandboxes[nid]
        # Gather all local tools (including local confident_learning with local rare classes only)
        sp = sb.call_tool("schema_profile")
        rh = sb.call_tool("row_health")
        cons = sb.call_tool("conservation_check")
        hist = sb.call_tool("label_histogram")
        mi = sb.call_tool("feature_label_mi")
        local_rare = [c for c, cnt in hist["class_counts"].items() if cnt < 1000]
        cl = sb.call_tool("confident_learning", exempt_classes=local_rare)

        local_packet = {
            "node_id": nid,
            "schema_profile": {
                "unmapped_columns": sp["unmapped_columns"],
                "missing_canonical": sp["missing_canonical"],
                "missing_rate_nonzero": sp["missing_rate_nonzero"],
            },
            "row_health": rh,
            "conservation_check": cons,
            "label_histogram": hist,
            "top_mi_features": mi["top_mi_features"],
            "confident_learning": cl,
            "local_rare_classes": local_rare,
        }

        messages = [
            {"role": "system", "content": base_node_prompt},
            {
                "role": "user",
                "content": (
                    f"You are operating in STANDALONE LOCAL MODE on node `{nid}` with NO access to any other node. "
                    "Based ONLY on your local diagnostic summaries below, decide what local remediation actions "
                    "(from: `rename_columns`, `drop_rows` with rule `nan_or_negative_physical`, `rescale`, `drop_column`, "
                    "`local_cl_clean`, `no_op`) should be applied to your own node. "
                    "Reply ONLY with valid JSON: `{\"actions\": [...]}` where every action includes `node` (except `drop_column`) "
                    "and a concrete `evidence` string.\n\n"
                    + json.dumps(local_packet, ensure_ascii=False)
                ),
            },
        ]

        res = client.chat(messages=messages, max_tokens=3072, tag=f"local_only_{nid}")
        content = res.content
        node_actions: List[Dict[str, Any]] = []
        for attempt in range(3):
            try:
                obj = extract_json_object(content)
                validate_plan(obj, valid_nodes=[nid])
                node_actions = obj["actions"]
                break
            except (ValueError, PlanValidationError) as exc:
                if attempt < 2:
                    client.json_retries += 1
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Validation error: {exc}. Return ONLY valid JSON `{{\"actions\": [...]}}` for node `{nid}`.",
                        }
                    )
                    res = client.chat(messages=messages, max_tokens=3072, tag=f"local_only_{nid}_retry{attempt+1}")
                    content = res.content
                else:
                    node_actions = [{"action": "no_op", "node": nid, "evidence": f"parse fallback: {exc}"}]

        per_node_outputs[nid] = node_actions
        all_actions.extend(node_actions)

    plan = {
        "diagnosis_summary": "Standalone local-only Node Agent decisions without cross-node Coordinator.",
        "actions": all_actions,
    }
    elapsed_s = round(time.perf_counter() - t0, 3)
    return {
        "plan": plan,
        "per_node_outputs": per_node_outputs,
        "coord_query_rounds": 0,
        "elapsed_s": elapsed_s,
        "llm_usage": client.get_usage_summary(),
        "prompt_hashes": get_prompt_hashes(),
    }
