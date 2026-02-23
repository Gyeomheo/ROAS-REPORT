"""Topic selection and prioritization helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from src.application.reporting.metrics import safe_pct_change, to_float


def top3_by_subsidiary(
    rows: List[Dict[str, Any]],
    descending: bool,
    min_topic_impact_pct: float = 0.03,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        sub = str(row.get("diagnosis_subsidiary", "UNKNOWN"))
        grouped.setdefault(sub, []).append(row)

    for sub in grouped:
        def _metric(item: Dict[str, Any]) -> float:
            pct = item.get("impact_contribution_pct")
            if pct is not None:
                return to_float(pct)
            raw = item.get("impact_adj")
            if raw is None:
                raw = item.get("Impact_adj")
            return abs(to_float(raw))

        def _scope_parts(item: Dict[str, Any]) -> tuple[str, str, str]:
            path = item.get("path", {})
            if not isinstance(path, dict):
                path = {}
            channel = str(path.get("CHANNEL", item.get("CHANNEL", "")))
            division = str(path.get("DIVISION", item.get("DIVISION", "")))
            product = str(path.get("PRODUCT", item.get("PRODUCT", "")))
            return channel, division, product

        def _has_yoy_basis(item: Dict[str, Any]) -> bool:
            pairs = [("CVR_curr", "CVR_prev"), ("AOV_curr", "AOV_prev"), ("CPC_curr", "CPC_prev")]
            for curr_key, prev_key in pairs:
                curr = item.get(curr_key)
                prev = item.get(prev_key)
                if curr is None or prev is None:
                    continue
                prev_val = to_float(prev)
                if prev_val <= 0:
                    continue
                curr_val = to_float(curr)
                if safe_pct_change(curr_val, prev_val) is not None:
                    return True
            return False

        def _is_actionable(item: Dict[str, Any]) -> bool:
            pct = item.get("impact_contribution_pct")
            low_impact = pct is None or to_float(pct) < min_topic_impact_pct
            return _has_yoy_basis(item) or not low_impact

        candidates = [item for item in grouped[sub] if _is_actionable(item)]

        ordered = sorted(
            candidates,
            key=lambda row: (
                -_metric(row) if descending else _metric(row),
                *_scope_parts(row),
            )
        )
        selected: List[Dict[str, Any]] = []
        seen_triplets: set[tuple[str, str, str]] = set()
        used_channel_bu: set[tuple[str, str]] = set()
        used_channels: set[str] = set()
        used_bus: set[str] = set()

        def _pick(stage: str) -> None:
            for item in ordered:
                if len(selected) >= 3:
                    return
                channel, division, product = _scope_parts(item)
                triplet = (channel, division, product)
                channel_bu = (channel, division)
                if triplet in seen_triplets:
                    continue

                is_new_pair = channel_bu not in used_channel_bu
                is_new_channel = channel not in used_channels
                is_new_bu = division not in used_bus

                if stage == "strict" and not (is_new_pair and is_new_channel and is_new_bu):
                    continue
                if stage == "semi" and not (is_new_pair and (is_new_channel or is_new_bu)):
                    continue
                if stage == "pair" and not is_new_pair:
                    continue

                selected.append(item)
                seen_triplets.add(triplet)
                used_channel_bu.add(channel_bu)
                used_channels.add(channel)
                used_bus.add(division)

        _pick("strict")
        _pick("semi")
        _pick("pair")
        _pick("fill")
        grouped[sub] = selected[:3]
    return grouped


def choose_subsidiary_mode(
    subsidiary: str,
    perf: Dict[str, Any],
    issues_by_sub: Dict[str, List[Dict[str, Any]]],
    improves_by_sub: Dict[str, List[Dict[str, Any]]],
) -> tuple[str, List[Dict[str, Any]]]:
    rev_delta = to_float(perf.get("rev_delta"))
    issues = issues_by_sub.get(subsidiary, [])
    improves = improves_by_sub.get(subsidiary, [])

    if rev_delta < 0:
        if issues:
            return "ISSUE", issues
        return "IMPROVE", improves

    if improves:
        return "IMPROVE", improves
    if issues:
        return "ISSUE", issues
    return "IMPROVE", improves

