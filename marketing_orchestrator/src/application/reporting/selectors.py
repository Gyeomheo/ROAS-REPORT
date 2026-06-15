"""Topic selection and prioritization helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from src.application.reporting.metrics import safe_pct_change, to_float

ISSUE_MODE_ROAS_DELTA_THRESHOLD = float(os.getenv("ROAS_ISSUE_THRESHOLD", "-0.03"))
ISSUE_CANDIDATE_ROAS_YOY_THRESHOLD = float(os.getenv("ROAS_ISSUE_MIN_YOY", "-0.15"))
ISSUE_CANDIDATE_REVENUE_YOY_THRESHOLD = float(os.getenv("REVENUE_ISSUE_MIN_YOY", "-0.20"))
ISSUE_CANDIDATE_MIN_IMPACT_CONTRIBUTION = float(os.getenv("ROAS_ISSUE_MIN_IMPACT_CONTRIBUTION", "0.10"))


def _topic_roas_delta(item: Dict[str, Any]) -> float | None:
    direct = item.get("topic_roas_delta")
    if direct is not None:
        return to_float(direct)

    roas_curr = item.get("ROAS_curr")
    roas_prev = item.get("ROAS_prev")
    if roas_curr is None or roas_prev is None:
        return None
    return to_float(roas_curr) - to_float(roas_prev)


def _safe_yoy(curr: Any, prev: Any) -> float | None:
    if curr is None or prev is None:
        return None
    return safe_pct_change(to_float(curr), to_float(prev))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _topic_roas_yoy(item: Dict[str, Any]) -> float | None:
    explicit = item.get("roas_yoy")
    if explicit is not None:
        return to_float(explicit)

    for curr_key, prev_key in [("topic_roas_curr", "topic_roas_prev"), ("ROAS_curr", "ROAS_prev")]:
        yoy = _safe_yoy(item.get(curr_key), item.get(prev_key))
        if yoy is not None:
            return yoy
    return None


def _revenue_yoy(item: Dict[str, Any]) -> float | None:
    for key in ["revenue_yoy", "rev_yoy"]:
        value = item.get(key)
        if value is not None:
            return to_float(value)
    return _safe_yoy(item.get("Revenue_curr_sum"), item.get("Revenue_prev_sum"))


def _spend_yoy(item: Dict[str, Any]) -> float | None:
    value = item.get("spend_yoy")
    if value is not None:
        return to_float(value)
    return _safe_yoy(item.get("Spend_curr_sum"), item.get("Spend_prev_sum"))


def _passes_issue_candidate_gate(item: Dict[str, Any]) -> bool:
    tag = str(item.get("tag", "") or "").upper()
    if tag in {"NEW", "LOW_VOL", "GONE", "UNDEFINED"}:
        return False

    roas_yoy = _topic_roas_yoy(item)
    revenue_yoy = _revenue_yoy(item)
    spend_yoy = _spend_yoy(item)
    impact = item.get("impact_contribution_pct")

    topic_roas_curr = item.get("topic_roas_curr", item.get("ROAS_curr"))
    topic_roas_prev = item.get("topic_roas_prev", item.get("ROAS_prev"))
    has_valid_roas_base = (
        topic_roas_curr is not None
        and topic_roas_prev is not None
        and to_float(topic_roas_prev) > 0
    )

    revenue_curr = _optional_float(item.get("Revenue_curr_sum"))
    spend_curr = _optional_float(item.get("Spend_curr_sum"))
    has_current_activity = (
        (revenue_curr is not None and revenue_curr > 0)
        or (spend_curr is not None and spend_curr > 0)
    )

    has_roas_drop = roas_yoy is not None and roas_yoy <= ISSUE_CANDIDATE_ROAS_YOY_THRESHOLD
    has_revenue_drop_with_nonnegative_spend = (
        revenue_yoy is not None
        and spend_yoy is not None
        and revenue_yoy <= ISSUE_CANDIDATE_REVENUE_YOY_THRESHOLD
        and spend_yoy >= 0
    )
    has_minimum_impact_contribution = (
        impact is not None
        and to_float(impact) >= ISSUE_CANDIDATE_MIN_IMPACT_CONTRIBUTION
        and has_valid_roas_base
        and has_current_activity
    )
    return has_roas_drop or has_revenue_drop_with_nonnegative_spend or has_minimum_impact_contribution


def top3_by_subsidiary(
    rows: List[Dict[str, Any]],
    descending: bool,
    min_topic_roas_abs_delta: float = 0.03,
    issue_candidate_gate: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        sub = str(row.get("diagnosis_subsidiary", "UNKNOWN"))
        grouped.setdefault(sub, []).append(row)

    for sub in grouped:
        def _metric(item: Dict[str, Any]) -> float:
            roas_delta = _topic_roas_delta(item)
            if roas_delta is not None:
                return abs(roas_delta)
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
            if issue_candidate_gate:
                return _passes_issue_candidate_gate(item)
            roas_delta = _topic_roas_delta(item)
            if roas_delta is None:
                return _has_yoy_basis(item)
            return _has_yoy_basis(item) or abs(roas_delta) >= min_topic_roas_abs_delta

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
    roas_delta = perf.get("roas_delta")
    if roas_delta is None:
        roas_delta = perf.get("rev_delta")
    roas_delta_value = to_float(roas_delta)
    issues = issues_by_sub.get(subsidiary, [])
    improves = improves_by_sub.get(subsidiary, [])

    if roas_delta_value < ISSUE_MODE_ROAS_DELTA_THRESHOLD:
        if issues:
            return "ISSUE", issues
        return "IMPROVE", improves

    if improves:
        return "IMPROVE", improves
    if issues:
        return "ISSUE", issues
    return "IMPROVE", improves
