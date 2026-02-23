"""Application service for campaign action enrichment use case."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.domain.models import CampaignSignals
from src.domain.recommendation import action_checklist, confirmed_primary_driver, recommended_action


def _format_branch_trace(branch_trace: Any, dimensions: Sequence[str]) -> str:
    if not isinstance(branch_trace, list):
        return ""
    parts: list[str] = []
    for node in branch_trace:
        if not isinstance(node, dict):
            continue
        selected = node.get("selected_path", {})
        if isinstance(selected, dict):
            selected_text = "/".join(str(selected.get(dim, "")) for dim in dimensions if selected.get(dim) is not None)
        else:
            selected_text = ""
        parts.append(f"{node.get('level', '')}:{node.get('rule', '')}:{selected_text}")
    return " -> ".join(parts)


def enrich_campaign_actions(rows: Sequence[Mapping[str, Any]], dimensions: Sequence[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)

        signals = CampaignSignals.from_row(enriched)
        primary_driver = confirmed_primary_driver(signals)
        policy_input = signals.with_primary_driver(primary_driver)

        enriched["primary_driver"] = primary_driver
        enriched["diagnosis_path"] = {dim: enriched.get(dim) for dim in dimensions}
        enriched["campaign_diagnosis_mode"] = "MANUAL"
        enriched["recommended_actions"] = recommended_action(policy_input)
        enriched["action_checklist"] = action_checklist(policy_input)
        enriched["drill_trace_text"] = _format_branch_trace(enriched.get("drill_trace"), dimensions=dimensions)
        output.append(enriched)
    return output
