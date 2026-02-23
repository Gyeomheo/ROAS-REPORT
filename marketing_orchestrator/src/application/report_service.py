"""Marketing Orchestrator entrypoint."""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import polars as pl

from src.application.analysis_service import run_subsidiary_analysis
from src.infrastructure.excel_repository import load_input_frame, save_output_workbook
from src.infrastructure.report_exporter import save_summary_html, save_summary_json


DIMENSIONS: List[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"]
CURRENT_YEAR = 2026
PREVIOUS_YEAR = 2025
CHANNEL_DIMS: List[str] = ["SUBSIDIARY", "CHANNEL"]
DIVISION_DIMS: List[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION"]
MIN_TOPIC_IMPACT_PCT = 0.03
ScopeMaps = tuple[
    Dict[tuple[Any, ...], Dict[str, Any]],
    Dict[tuple[Any, ...], Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _safe_pct_change(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev <= 0:
        return None
    return (curr - prev) / prev


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "$0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${abs_value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"${abs_value / 1_000:.0f}K"
    return f"${abs_value:.0f}"


def _fmt_money_signed(value: float | None) -> str:
    if value is None:
        return "$0"
    sign = "-" if value < 0 else "+"
    return f"{sign}{_fmt_money(abs(value))}"


def _fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    pct = value * 100
    if signed:
        return f"{pct:+.0f}%".replace("+", "")
    return f"{pct:.0f}%"


def _fmt_pct_yoy(value: float | None, signed: bool = True) -> str:
    base = _fmt_pct(value, signed=signed)
    if base == "N/A":
        return base
    return f"{base} YoY"


def _fmt_roas(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def _fmt_pp(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.0f}%p"


def _fmt_driver_metric(metric: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if metric == "CPC":
        return f"${value:.2f}"
    if metric == "CVR":
        return f"{value * 100:.2f}%"
    if metric == "AOV":
        return _fmt_money(value)
    return f"{value:.3f}"



def _trend(value: float | None, eps: float = 1e-9) -> str:
    if value is None:
        return "unknown"
    if value > eps:
        return "up"
    if value < -eps:
        return "down"
    return "flat"


def _direct_contribution_pct(child_impact: float, parent_delta: float, mode_tag: str) -> float | None:
    parent_abs = abs(parent_delta)
    if parent_abs <= 0:
        return None
    mode = (mode_tag or "").upper()
    if mode == "ISSUE":
        return max(0.0, -child_impact) / parent_abs
    if mode in {"IMPROVE", "DEFENSE"}:
        return max(0.0, child_impact) / parent_abs
    return abs(child_impact) / parent_abs

def _sum_aggregations() -> list[pl.Expr]:
    return [
        pl.col("Revenue_curr").sum().alias("Revenue_curr_sum"),
        pl.col("Revenue_prev").sum().alias("Revenue_prev_sum"),
        pl.col("Spend_curr").sum().alias("Spend_curr_sum"),
        pl.col("Spend_prev").sum().alias("Spend_prev_sum"),
    ]


def _log_mean(curr: float, prev: float) -> float | None:
    if curr <= 0 or prev <= 0:
        return None
    if abs(curr - prev) < 1e-12:
        return curr
    denom = math.log(curr) - math.log(prev)
    if abs(denom) < 1e-12:
        return None
    return (curr - prev) / denom


def _sum_aggregations_full() -> list[pl.Expr]:
    return [
        pl.col("Revenue_curr").sum().alias("Revenue_curr_sum"),
        pl.col("Revenue_prev").sum().alias("Revenue_prev_sum"),
        pl.col("Spend_curr").sum().alias("Spend_curr_sum"),
        pl.col("Spend_prev").sum().alias("Spend_prev_sum"),
        pl.col("Clicks_curr").sum().alias("Clicks_curr_sum"),
        pl.col("Clicks_prev").sum().alias("Clicks_prev_sum"),
        pl.col("Orders_curr").sum().alias("Orders_curr_sum"),
        pl.col("Orders_prev").sum().alias("Orders_prev_sum"),
    ]


def _to_key(row: Dict[str, Any], dims: List[str]) -> tuple[Any, ...]:
    return tuple(row.get(dim) for dim in dims)


def _scope_maps(
    df: pl.DataFrame,
) -> tuple[Dict[tuple[Any, ...], Dict[str, Any]], Dict[tuple[Any, ...], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    subsidiary_rows = df.group_by(["SUBSIDIARY"]).agg(_sum_aggregations_full()).to_dicts()
    channel_rows = df.group_by(CHANNEL_DIMS).agg(_sum_aggregations_full()).to_dicts()
    division_rows = df.group_by(DIVISION_DIMS).agg(_sum_aggregations_full()).to_dicts()
    subsidiary_map = {str(row.get("SUBSIDIARY", "")): row for row in subsidiary_rows}
    channel_map = {_to_key(row, CHANNEL_DIMS): row for row in channel_rows}
    division_map = {_to_key(row, DIVISION_DIMS): row for row in division_rows}
    return channel_map, division_map, subsidiary_map


def _division_parallel_contribution(df: pl.DataFrame) -> List[Dict[str, Any]]:
    parent_rows = df.group_by(["SUBSIDIARY"]).agg(_sum_aggregations_full()).sort(["SUBSIDIARY"]).to_dicts()
    division_rows = (
        df.group_by(["SUBSIDIARY", "DIVISION"])
        .agg(_sum_aggregations_full())
        .sort(["SUBSIDIARY", "DIVISION"])
        .to_dicts()
    )

    by_sub: Dict[str, List[Dict[str, Any]]] = {}
    for row in division_rows:
        key = str(row.get("SUBSIDIARY", ""))
        by_sub.setdefault(key, []).append(row)

    output: List[Dict[str, Any]] = []
    for parent in parent_rows:
        subsidiary = str(parent.get("SUBSIDIARY", ""))
        children = by_sub.get(subsidiary, [])

        parent_curr = _to_float(parent.get("Revenue_curr_sum"))
        parent_prev = _to_float(parent.get("Revenue_prev_sum"))
        parent_delta = parent_curr - parent_prev
        parent_abs_delta = abs(parent_delta)
        parent_log_mean = _log_mean(parent_curr, parent_prev)
        parent_log_ratio: float | None = None
        if parent_curr > 0 and parent_prev > 0:
            parent_log_ratio = math.log(parent_curr / parent_prev)

        staged: List[Dict[str, Any]] = []
        raw_sum = 0.0
        for child in children:
            child_curr = _to_float(child.get("Revenue_curr_sum"))
            child_prev = _to_float(child.get("Revenue_prev_sum"))
            child_delta = child_curr - child_prev
            child_log_mean = _log_mean(child_curr, child_prev)
            valid_lmdi = (
                child_curr > 0
                and child_prev > 0
                and parent_curr > 0
                and parent_prev > 0
                and child_log_mean is not None
                and parent_log_mean is not None
                and parent_log_ratio is not None
            )
            if valid_lmdi:
                impact_raw = (child_log_mean / parent_log_mean) * parent_log_ratio * parent_delta
                impact_method = "LMDI"
            else:
                impact_raw = child_delta
                impact_method = "DELTA_FALLBACK"
            raw_sum += impact_raw
            staged.append(
                {
                    **child,
                    "impact_raw": impact_raw,
                    "impact_raw_method": impact_method,
                }
            )

        residual = parent_delta - raw_sum
        residual_idx = -1
        if staged:
            residual_idx = max(
                range(len(staged)),
                key=lambda idx: (abs(_to_float(staged[idx].get("impact_raw"))), str(staged[idx].get("DIVISION", ""))),
            )

        rows: List[Dict[str, Any]] = []
        for idx, row in enumerate(staged):
            rev_curr = _to_float(row.get("Revenue_curr_sum"))
            rev_prev = _to_float(row.get("Revenue_prev_sum"))
            spend_curr = _to_float(row.get("Spend_curr_sum"))
            spend_prev = _to_float(row.get("Spend_prev_sum"))
            impact_raw = _to_float(row.get("impact_raw"))
            impact_adj = impact_raw + residual if idx == residual_idx else impact_raw

            roas_curr = _safe_ratio(rev_curr, spend_curr)
            roas_prev = _safe_ratio(rev_prev, spend_prev)
            roas_yoy = _safe_pct_change(roas_curr, roas_prev)
            rev_yoy = _safe_pct_change(rev_curr, rev_prev)
            spend_yoy = _safe_pct_change(spend_curr, spend_prev)

            contribution_to_total_delta_pct = (
                abs(impact_adj) / parent_abs_delta if parent_abs_delta > 0 else 0.0
            )
            contribution_to_decline_pct = (
                max(0.0, -impact_adj) / parent_abs_delta if parent_delta < 0 and parent_abs_delta > 0 else 0.0
            )
            contribution_to_growth_pct = (
                max(0.0, impact_adj) / parent_abs_delta if parent_delta > 0 and parent_abs_delta > 0 else 0.0
            )

            if parent_delta < 0:
                contribution_label = "DECLINE_DRIVER" if impact_adj < 0 else "DECLINE_OFFSET"
            elif parent_delta > 0:
                contribution_label = "GROWTH_DRIVER" if impact_adj > 0 else "GROWTH_OFFSET"
            else:
                contribution_label = "NEUTRAL"

            division_name = str(row.get("DIVISION", "DIVISION"))
            if parent_delta < 0:
                summary_text = (
                    f"{division_name} 매출 { _fmt_pct(rev_yoy) } / ROAS {_fmt_roas(roas_curr)}({_fmt_pct(roas_yoy)}) "
                    f"기준, 전체 하락 기여도 {_fmt_pct(contribution_to_decline_pct, signed=False)}"
                )
            elif parent_delta > 0:
                summary_text = (
                    f"{division_name} 매출 { _fmt_pct(rev_yoy) } / ROAS {_fmt_roas(roas_curr)}({_fmt_pct(roas_yoy)}) "
                    f"기준, 전체 성장 기여도 {_fmt_pct(contribution_to_growth_pct, signed=False)}"
                )
            else:
                summary_text = (
                    f"{division_name} 매출 { _fmt_pct(rev_yoy) } / ROAS {_fmt_roas(roas_curr)}({_fmt_pct(roas_yoy)})"
                )

            rows.append(
                {
                    "SUBSIDIARY": subsidiary,
                    "DIVISION": row.get("DIVISION"),
                    "Revenue_curr_sum": rev_curr,
                    "Revenue_prev_sum": rev_prev,
                    "Spend_curr_sum": spend_curr,
                    "Spend_prev_sum": spend_prev,
                    "Revenue_delta": rev_curr - rev_prev,
                    "Revenue_yoy_pct": rev_yoy,
                    "Spend_yoy_pct": spend_yoy,
                    "ROAS_curr": roas_curr,
                    "ROAS_prev": roas_prev,
                    "ROAS_yoy_pct": roas_yoy,
                    "Impact_raw": impact_raw,
                    "Impact_adj": impact_adj,
                    "Impact_raw_method": row.get("impact_raw_method"),
                    "Residual_parent": residual,
                    "Residual_applied_flag": idx == residual_idx,
                    "Parent_revenue_delta": parent_delta,
                    "Contribution_to_total_delta_pct": contribution_to_total_delta_pct,
                    "Contribution_to_decline_pct": contribution_to_decline_pct,
                    "Contribution_to_growth_pct": contribution_to_growth_pct,
                    "Contribution_label": contribution_label,
                    "summary_text": summary_text,
                }
            )

        if parent_delta < 0:
            rows.sort(
                key=lambda item: (
                    -_to_float(item.get("Contribution_to_decline_pct")),
                    -abs(_to_float(item.get("Impact_adj"))),
                    str(item.get("DIVISION", "")),
                )
            )
        elif parent_delta > 0:
            rows.sort(
                key=lambda item: (
                    -_to_float(item.get("Contribution_to_growth_pct")),
                    -abs(_to_float(item.get("Impact_adj"))),
                    str(item.get("DIVISION", "")),
                )
            )
        else:
            rows.sort(key=lambda item: (-abs(_to_float(item.get("Impact_adj"))), str(item.get("DIVISION", ""))))

        output.append(
            {
                "SUBSIDIARY": subsidiary,
                "Parent_revenue_curr_sum": parent_curr,
                "Parent_revenue_prev_sum": parent_prev,
                "Parent_revenue_delta": parent_delta,
                "Parent_revenue_yoy_pct": _safe_pct_change(parent_curr, parent_prev),
                "rows": rows,
            }
        )

    return output


def _final_drill_node(drill_trace: Any) -> Dict[str, Any]:
    if not isinstance(drill_trace, list) or not drill_trace:
        return {}
    node = drill_trace[-1]
    return node if isinstance(node, dict) else {}


def _trace_event_by_dim(drill_trace: Any, target_dim: str) -> Dict[str, Any]:
    if target_dim not in DIMENSIONS:
        return {}
    if not isinstance(drill_trace, list):
        return {}

    target_depth = DIMENSIONS.index(target_dim) + 1
    nodes: List[Dict[str, Any]] = []
    for item in drill_trace:
        if isinstance(item, dict):
            nodes.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict):
                    nodes.append(sub)

    for node in reversed(nodes):
        if not isinstance(node, dict):
            continue
        selected = node.get("selected_path", {})
        if not isinstance(selected, dict):
            continue
        depth = 0
        for dim in DIMENSIONS:
            if selected.get(dim) is None:
                break
            depth += 1
        if depth == target_depth and selected.get(target_dim) is not None:
            return node
    return {}


def _insight_text(
    row: Dict[str, Any],
    channel_map: Dict[tuple[Any, ...], Dict[str, Any]],
    division_map: Dict[tuple[Any, ...], Dict[str, Any]],
    subsidiary_map: Dict[str, Dict[str, Any]],
    impact_contribution_pct: float | None,
) -> tuple[str, str]:
    channel_key = tuple(row.get(dim) for dim in CHANNEL_DIMS)
    division_key = tuple(row.get(dim) for dim in DIVISION_DIMS)
    channel_row = channel_map.get(channel_key, {})
    division_row = division_map.get(division_key, {})

    channel_spend_curr = _to_float(channel_row.get("Spend_curr_sum"))
    channel_spend_prev = _to_float(channel_row.get("Spend_prev_sum"))
    channel_rev_curr = _to_float(channel_row.get("Revenue_curr_sum"))
    channel_rev_prev = _to_float(channel_row.get("Revenue_prev_sum"))
    channel_spend_yoy = _safe_pct_change(channel_spend_curr, channel_spend_prev)
    channel_rev_yoy = _safe_pct_change(channel_rev_curr, channel_rev_prev)
    channel_roas_curr = _safe_ratio(channel_rev_curr, channel_spend_curr)
    channel_roas_prev = _safe_ratio(channel_rev_prev, channel_spend_prev)
    channel_roas_yoy = _safe_pct_change(channel_roas_curr, channel_roas_prev)

    mode_tag = str(row.get("mode_tag", "") or "").upper()
    product_name = str(row.get("PRODUCT", "PRODUCT"))
    product_rev_curr = _to_float(row.get("Revenue_curr_sum"))
    product_rev_prev = _to_float(row.get("Revenue_prev_sum"))
    product_spend_curr = _to_float(row.get("Spend_curr_sum"))
    product_spend_prev = _to_float(row.get("Spend_prev_sum"))
    product_rev_yoy = _safe_pct_change(product_rev_curr, product_rev_prev)
    product_spend_yoy = _safe_pct_change(product_spend_curr, product_spend_prev)
    product_roas_curr = _safe_ratio(product_rev_curr, product_spend_curr)
    product_roas_prev = _safe_ratio(product_rev_prev, product_spend_prev)
    product_roas_yoy = _safe_pct_change(product_roas_curr, product_roas_prev)

    channel_name = str(row.get("CHANNEL", "CHANNEL"))
    headline = (
        f"{product_name} | 매출 {_fmt_money(product_rev_curr)}({_fmt_pct_yoy(product_rev_yoy)}) "
        f"| 투자 {_fmt_money(product_spend_curr)}({_fmt_pct_yoy(product_spend_yoy)}) "
        f"| ROAS {_fmt_roas(product_roas_curr)}({_fmt_pct_yoy(product_roas_yoy)})"
    )
    if mode_tag == "ISSUE":
        headline = f"{headline} [하락 유발]"
    elif mode_tag in {"IMPROVE", "DEFENSE"} and (channel_rev_yoy is not None and channel_rev_yoy < 0):
        headline = f"{headline} [채널 하락 상쇄]"

    division_spend_curr = _to_float(division_row.get("Spend_curr_sum"))
    division_spend_prev = _to_float(division_row.get("Spend_prev_sum"))
    division_rev_curr = _to_float(division_row.get("Revenue_curr_sum"))
    division_rev_prev = _to_float(division_row.get("Revenue_prev_sum"))
    division_clicks_curr = _to_float(division_row.get("Clicks_curr_sum"))
    division_clicks_prev = _to_float(division_row.get("Clicks_prev_sum"))
    division_orders_curr = _to_float(division_row.get("Orders_curr_sum"))
    division_orders_prev = _to_float(division_row.get("Orders_prev_sum"))
    division_spend_yoy = _safe_pct_change(division_spend_curr, division_spend_prev)
    division_rev_yoy = _safe_pct_change(division_rev_curr, division_rev_prev)
    division_roas_curr = _safe_ratio(division_rev_curr, division_spend_curr)
    division_roas_prev = _safe_ratio(division_rev_prev, division_spend_prev)
    division_roas_yoy = _safe_pct_change(division_roas_curr, division_roas_prev)
    division_cpc_curr = _safe_ratio(division_spend_curr, division_clicks_curr)
    division_cpc_prev = _safe_ratio(division_spend_prev, division_clicks_prev)
    division_cvr_curr = _safe_ratio(division_orders_curr, division_clicks_curr)
    division_cvr_prev = _safe_ratio(division_orders_prev, division_clicks_prev)
    division_aov_curr = _safe_ratio(division_rev_curr, division_orders_curr)
    division_aov_prev = _safe_ratio(division_rev_prev, division_orders_prev)
    division_rev_share = _safe_ratio(division_rev_curr, channel_rev_curr)
    channel_delta = channel_rev_curr - channel_rev_prev
    channel_abs_delta = abs(channel_delta)
    impact_adj = _to_float(row.get("Impact_adj"))
    division_name = str(row.get("DIVISION", "DIVISION"))
    subsidiary_name = str(row.get("SUBSIDIARY", ""))
    subsidiary_row = subsidiary_map.get(subsidiary_name, {})
    subsidiary_delta = _to_float(subsidiary_row.get("Revenue_curr_sum")) - _to_float(subsidiary_row.get("Revenue_prev_sum"))
    division_delta = division_rev_curr - division_rev_prev

    if impact_contribution_pct is None and channel_abs_delta > 0:
        impact_contribution_pct = _direct_contribution_pct(impact_adj, channel_delta, mode_tag)

    if mode_tag == "ISSUE":
        contribution_rate_label = "상위 하락 기여율"
    elif mode_tag in {"IMPROVE", "DEFENSE"}:
        contribution_rate_label = "상위 개선 기여율"
    else:
        contribution_rate_label = "상위 기여율"

    channel_event = _trace_event_by_dim(row.get("drill_trace"), "CHANNEL")
    division_event = _trace_event_by_dim(row.get("drill_trace"), "DIVISION")
    channel_impact = _to_float(channel_event.get("impact_adj"))
    division_impact = _to_float(division_event.get("impact_adj"))
    channel_direct_pct = _direct_contribution_pct(channel_impact, subsidiary_delta, mode_tag)
    division_direct_pct = _direct_contribution_pct(division_impact, channel_delta, mode_tag)
    product_direct_pct = _direct_contribution_pct(impact_adj, division_delta, mode_tag)
    if product_direct_pct is None:
        product_direct_pct = impact_contribution_pct

    product_clicks_curr = _to_float(row.get("Clicks_curr_sum"))
    product_clicks_prev = _to_float(row.get("Clicks_prev_sum"))
    product_orders_curr = _to_float(row.get("Orders_curr_sum"))
    product_orders_prev = _to_float(row.get("Orders_prev_sum"))

    product_cpc_curr = row.get("CPC_curr")
    product_cpc_prev = row.get("CPC_prev")
    product_cvr_curr = row.get("CVR_curr")
    product_cvr_prev = row.get("CVR_prev")
    product_aov_curr = row.get("AOV_curr")
    product_aov_prev = row.get("AOV_prev")
    if product_cpc_curr is None:
        product_cpc_curr = _safe_ratio(product_spend_curr, product_clicks_curr)
    if product_cpc_prev is None:
        product_cpc_prev = _safe_ratio(product_spend_prev, product_clicks_prev)
    if product_cvr_curr is None:
        product_cvr_curr = _safe_ratio(product_orders_curr, product_clicks_curr)
    if product_cvr_prev is None:
        product_cvr_prev = _safe_ratio(product_orders_prev, product_clicks_prev)
    if product_aov_curr is None:
        product_aov_curr = _safe_ratio(product_rev_curr, product_orders_curr)
    if product_aov_prev is None:
        product_aov_prev = _safe_ratio(product_rev_prev, product_orders_prev)

    driver = str(row.get("primary_driver", "CPC") or "CPC").upper()
    if driver == "AOV":
        product_driver_curr = product_aov_curr
        product_driver_prev = product_aov_prev
        division_driver_curr = division_aov_curr
        division_driver_prev = division_aov_prev
        driver_dlog = row.get("dlog_AOV")
    elif driver == "CVR":
        product_driver_curr = product_cvr_curr
        product_driver_prev = product_cvr_prev
        division_driver_curr = division_cvr_curr
        division_driver_prev = division_cvr_prev
        driver_dlog = row.get("dlog_CVR")
    else:
        driver = "CPC"
        product_driver_curr = product_cpc_curr
        product_driver_prev = product_cpc_prev
        division_driver_curr = division_cpc_curr
        division_driver_prev = division_cpc_prev
        driver_dlog = row.get("dlog_CPC")

    driver_yoy = _safe_pct_change(product_driver_curr, product_driver_prev)
    division_driver_yoy = _safe_pct_change(division_driver_curr, division_driver_prev)
    driver_gap_pp = None
    if driver_yoy is not None and division_driver_yoy is not None:
        driver_gap_pp = driver_yoy - division_driver_yoy

    dlog_cvr_abs = abs(_to_float(row.get("dlog_CVR")))
    dlog_aov_abs = abs(_to_float(row.get("dlog_AOV")))
    dlog_cpc_abs = abs(_to_float(row.get("dlog_CPC")))
    driver_dlog_abs = abs(_to_float(driver_dlog))
    signal_max_abs = max(dlog_cvr_abs, dlog_aov_abs, dlog_cpc_abs)
    tag = str(row.get("tag", "") or "").upper()

    channel_text = (
        f"1) 채널 {channel_name}: 매출 {_fmt_money(channel_rev_curr)}({_fmt_pct_yoy(channel_rev_yoy)}), "
        f"투자 {_fmt_money(channel_spend_curr)}({_fmt_pct_yoy(channel_spend_yoy)}), "
        f"ROAS {_fmt_roas(channel_roas_curr)}({_fmt_pct_yoy(channel_roas_yoy)}), "
        f"{contribution_rate_label} {_fmt_pct(channel_direct_pct, signed=False)}"
    )
    division_text = (
        f"2) BU {division_name}: 채널 내 매출 비중 {_fmt_pct(division_rev_share, signed=False)}, "
        f"매출 {_fmt_money(division_rev_curr)}({_fmt_pct_yoy(division_rev_yoy)}), "
        f"투자 {_fmt_money(division_spend_curr)}({_fmt_pct_yoy(division_spend_yoy)}), "
        f"ROAS {_fmt_roas(division_roas_curr)}({_fmt_pct_yoy(division_roas_yoy)}), "
        f"{contribution_rate_label} {_fmt_pct(division_direct_pct, signed=False)}"
    )
    product_text = (
        f"3) 제품 {product_name}: {contribution_rate_label} {_fmt_pct(product_direct_pct, signed=False)}, "
        f"핵심 지표 {driver}"
    )
    if tag == "NEW":
        evidence_text = (
            f"4) 지표 근거: 신규 항목으로 전년 기준이 부족해 YoY/dlog 신뢰도는 낮음. "
            f"현재 {driver} {_fmt_driver_metric(driver, product_driver_curr)}, "
            f"BU {driver} {_fmt_driver_metric(driver, division_driver_curr)} 기준으로 관찰."
        )
        interpretation_text = (
            "5) 해석: 신규 항목은 원인 확정보다 초기 안정화(클릭·전환·매출 추세) 모니터링이 우선."
        )
    elif tag == "LOW_VOL":
        evidence_text = (
            f"4) 지표 근거: 저볼륨 구간으로 YoY 변동성이 큼. "
            f"{driver} {_fmt_driver_metric(driver, product_driver_curr)} vs BU {driver} {_fmt_driver_metric(driver, division_driver_curr)} 참고."
        )
        interpretation_text = (
            "5) 해석: 저볼륨 구간은 원인 단정 대신 데이터 누적 후 재판단이 적절."
        )
    else:
        evidence_text = (
            f"4) 지표 근거: {driver} {_fmt_driver_metric(driver, product_driver_curr)}"
            f"({_fmt_pct_yoy(driver_yoy)}), BU {driver} {_fmt_driver_metric(driver, division_driver_curr)}"
            f"({_fmt_pct_yoy(division_driver_yoy)}), YoY 격차 {_fmt_pp(driver_gap_pp)}. "
            f"|dlog| {driver} {driver_dlog_abs:.3f} (CVR {dlog_cvr_abs:.3f}, AOV {dlog_aov_abs:.3f}, CPC {dlog_cpc_abs:.3f})"
        )

        if signal_max_abs < 0.05:
            interpretation_text = "5) 해석: 지표 신호 강도가 약해 원인 단정보다 추세 관찰이 우선."
        elif driver == "CPC":
            if mode_tag == "ISSUE":
                interpretation_text = (
                    "5) 해석: 유입 효율 이슈 신호. 보조 확인은 CTR(IMP/CLICK), 클릭 볼륨, 유입단가 추세."
                )
            else:
                interpretation_text = (
                    "5) 해석: 유입 효율 개선 신호. 보조 확인은 CTR(IMP/CLICK) 유지, 클릭 볼륨 유지 여부."
                )
        elif driver == "CVR":
            if mode_tag == "ISSUE":
                interpretation_text = (
                    "5) 해석: 전환 효율 이슈 신호. 유입 트래픽 품질(키워드/오디언스/랜딩 정합성) 점검 필요."
                )
            else:
                interpretation_text = (
                    "5) 해석: 전환 효율 개선 신호. 유입 트래픽 품질 개선 및 랜딩/오퍼 적합도 개선 가능성."
                )
        else:
            if mode_tag == "ISSUE":
                interpretation_text = (
                    "5) 해석: AOV 이슈는 유입 트래픽 믹스(저의도/저가 유입 비중) 문제 신호."
                )
            else:
                interpretation_text = (
                    "5) 해석: AOV 개선은 고의도/고가 상품 유입 비중 개선 신호."
                )

    detail = "\n".join([channel_text, division_text, product_text, evidence_text, interpretation_text])
    return headline, detail

def _to_report_rows(
    rows: List[Dict[str, Any]],
    df: pl.DataFrame,
    scope_maps: ScopeMaps | None = None,
) -> List[Dict[str, Any]]:
    group_map: Dict[tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = {}
    for item in rows:
        gkey = (
            item.get("diagnosis_subsidiary"),
            item.get("CHANNEL"),
            item.get("DIVISION"),
            str(item.get("mode_tag", "")).upper(),
        )
        group_map.setdefault(gkey, []).append(item)

    rank_map: Dict[tuple[Any, Any, Any, Any, Any], Dict[str, int]] = {}
    for gkey, items in group_map.items():
        mode = gkey[3]
        if mode == "ISSUE":
            ordered = sorted(items, key=lambda r: (_to_float(r.get("Impact_adj")), str(r.get("PRODUCT", ""))))
        else:
            ordered = sorted(items, key=lambda r: (-_to_float(r.get("Impact_adj")), str(r.get("PRODUCT", ""))))
        pool_size = len(ordered)
        for idx, it in enumerate(ordered, start=1):
            rkey = (
                it.get("diagnosis_subsidiary"),
                it.get("CHANNEL"),
                it.get("DIVISION"),
                it.get("PRODUCT"),
                str(it.get("mode_tag", "")).upper(),
            )
            rank_map[rkey] = {"rank": idx, "pool_size": pool_size}

    channel_map, division_map, subsidiary_map = scope_maps if scope_maps is not None else _scope_maps(df)
    report_rows: List[Dict[str, Any]] = []
    for row in rows:
        channel_key = tuple(row.get(dim) for dim in CHANNEL_DIMS)
        division_key = tuple(row.get(dim) for dim in DIVISION_DIMS)
        channel_row = channel_map.get(channel_key, {})
        division_row = division_map.get(division_key, {})
        channel_delta = _to_float(channel_row.get("Revenue_curr_sum")) - _to_float(channel_row.get("Revenue_prev_sum"))
        division_delta = _to_float(division_row.get("Revenue_curr_sum")) - _to_float(division_row.get("Revenue_prev_sum"))
        impact_adj = _to_float(row.get("Impact_adj"))
        mode_tag = str(row.get("mode_tag", "")).upper()
        impact_contribution_pct = _direct_contribution_pct(impact_adj, division_delta, mode_tag)
        summary_text, detail_text = _insight_text(
            row,
            channel_map=channel_map,
            division_map=division_map,
            subsidiary_map=subsidiary_map,
            impact_contribution_pct=impact_contribution_pct,
        )
        final_node = _final_drill_node(row.get("drill_trace"))
        rank_info = rank_map.get(
            (
                row.get("diagnosis_subsidiary"),
                row.get("CHANNEL"),
                row.get("DIVISION"),
                row.get("PRODUCT"),
                mode_tag,
            ),
            {"rank": 1, "pool_size": 1},
        )
        if mode_tag == "ISSUE":
            contribution_type = "하락 기여율"
        else:
            contribution_type = "개선 기여율"

        caution = ""
        tag = str(row.get("tag", "") or "")
        if tag == "NEW":
            caution = " 신규(tag=NEW)라 전년 기저효과를 함께 확인하세요."
        elif tag == "LOW_VOL":
            caution = " 저볼륨(tag=LOW_VOL)이라 변동성 해석에 주의가 필요합니다."

        why_selected_text = (
            f"{row.get('CHANNEL')}/{row.get('DIVISION')} 내 {rank_info['pool_size']}개 제품 중 "
            f"{rank_info['rank']}위, {contribution_type} { _fmt_pct(impact_contribution_pct, signed=False) } "
            f"기준으로 선정.{caution}"
        )

        report_rows.append(
            {
                "diagnosis_subsidiary": row.get("diagnosis_subsidiary"),
                "path": row.get("diagnosis_path"),
                "final_drill_result": {
                    "level": final_node.get("level"),
                    "rule": final_node.get("rule"),
                    "selected_path": final_node.get("selected_path"),
                },
                "impact_contribution_pct": impact_contribution_pct,
                "impact_contribution_text": _fmt_pct(impact_contribution_pct, signed=False),
                "mode_tag": row.get("mode_tag"),
                "primary_driver": row.get("primary_driver"),
                "tag": row.get("tag"),
                "selection_rank": rank_info["rank"],
                "selection_pool_size": rank_info["pool_size"],
                "why_selected_text": why_selected_text,
                "campaign_diagnosis_mode": "MANUAL",
                "summary_text": summary_text,
                "detail_text": detail_text,
                "recommended_actions": row.get("recommended_actions"),
                "action_checklist": row.get("action_checklist"),
                "dlog_CVR": row.get("dlog_CVR"),
                "dlog_AOV": row.get("dlog_AOV"),
                "dlog_CPC": row.get("dlog_CPC"),
                "CVR_curr": row.get("CVR_curr"),
                "CVR_prev": row.get("CVR_prev"),
                "CPC_curr": row.get("CPC_curr"),
                "CPC_prev": row.get("CPC_prev"),
                "AOV_curr": row.get("AOV_curr"),
                "AOV_prev": row.get("AOV_prev"),
            }
        )
    return report_rows


def _top3_by_subsidiary(rows: List[Dict[str, Any]], descending: bool) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        sub = str(row.get("diagnosis_subsidiary", "UNKNOWN"))
        grouped.setdefault(sub, []).append(row)

    for sub in grouped:
        # Prefer percentage contribution in report rows; fall back to absolute impact amount.
        def _metric(item: Dict[str, Any]) -> float:
            pct = item.get("impact_contribution_pct")
            if pct is not None:
                return _to_float(pct)
            raw = item.get("impact_adj")
            if raw is None:
                raw = item.get("Impact_adj")
            return abs(_to_float(raw))

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
                prev_val = _to_float(prev)
                if prev_val <= 0:
                    continue
                curr_val = _to_float(curr)
                if _safe_pct_change(curr_val, prev_val) is not None:
                    return True
            return False

        def _is_actionable(item: Dict[str, Any]) -> bool:
            pct = item.get("impact_contribution_pct")
            low_impact = pct is None or _to_float(pct) < MIN_TOPIC_IMPACT_PCT
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

        # Diversification priority:
        # 1) new channel + new BU, 2) at least one new among channel/BU, 3) no duplicate channel/BU pair, 4) fill.
        _pick("strict")
        _pick("semi")
        _pick("pair")
        _pick("fill")
        grouped[sub] = selected[:3]
    return grouped


def _subsidiary_perf_map(df: pl.DataFrame) -> Dict[str, Dict[str, Any]]:
    rows = df.group_by(["SUBSIDIARY"]).agg(_sum_aggregations_full()).to_dicts()
    return {str(row.get("SUBSIDIARY", "")): row for row in rows}


def _subsidiary_metrics(scope_row: Dict[str, Any]) -> Dict[str, Any]:
    rev_curr = _to_float(scope_row.get("Revenue_curr_sum"))
    rev_prev = _to_float(scope_row.get("Revenue_prev_sum"))
    spend_curr = _to_float(scope_row.get("Spend_curr_sum"))
    spend_prev = _to_float(scope_row.get("Spend_prev_sum"))
    clicks_curr = _to_float(scope_row.get("Clicks_curr_sum"))
    clicks_prev = _to_float(scope_row.get("Clicks_prev_sum"))
    orders_curr = _to_float(scope_row.get("Orders_curr_sum"))
    orders_prev = _to_float(scope_row.get("Orders_prev_sum"))

    cpc_curr = _safe_ratio(spend_curr, clicks_curr)
    cpc_prev = _safe_ratio(spend_prev, clicks_prev)
    cvr_curr = _safe_ratio(orders_curr, clicks_curr)
    cvr_prev = _safe_ratio(orders_prev, clicks_prev)
    roas_curr = _safe_ratio(rev_curr, spend_curr)
    roas_prev = _safe_ratio(rev_prev, spend_prev)

    return {
        "rev_curr": rev_curr,
        "rev_prev": rev_prev,
        "rev_delta": rev_curr - rev_prev,
        "rev_yoy": _safe_pct_change(rev_curr, rev_prev),
        "spend_curr": spend_curr,
        "spend_prev": spend_prev,
        "spend_yoy": _safe_pct_change(spend_curr, spend_prev),
        "cpc_curr": cpc_curr,
        "cpc_prev": cpc_prev,
        "cpc_yoy": _safe_pct_change(cpc_curr, cpc_prev),
        "cvr_curr": cvr_curr,
        "cvr_prev": cvr_prev,
        "cvr_yoy": _safe_pct_change(cvr_curr, cvr_prev),
        "roas_curr": roas_curr,
        "roas_prev": roas_prev,
        "roas_yoy": _safe_pct_change(roas_curr, roas_prev),
    }


def _choose_subsidiary_mode(
    subsidiary: str,
    perf: Dict[str, Any],
    issues_by_sub: Dict[str, List[Dict[str, Any]]],
    improves_by_sub: Dict[str, List[Dict[str, Any]]],
) -> tuple[str, List[Dict[str, Any]]]:
    rev_delta = _to_float(perf.get("rev_delta"))
    issues = issues_by_sub.get(subsidiary, [])
    improves = improves_by_sub.get(subsidiary, [])

    if rev_delta < 0:
        if issues:
            return "ISSUE", issues
        return "IMPROVE", improves

    # Flat or positive trend: prefer improve-topics unless unavailable.
    if improves:
        return "IMPROVE", improves
    if issues:
        return "ISSUE", issues
    return "IMPROVE", improves


def _subsidiary_overall_comment(subsidiary: str, perf: Dict[str, Any], mode: str) -> str:
    rev_delta = _to_float(perf.get("rev_delta"))
    if rev_delta < 0:
        trend = "하락"
    elif rev_delta > 0:
        trend = "개선"
    else:
        trend = "보합"

    mode_kr = "이슈" if mode == "ISSUE" else "개선"
    return (
        f"{subsidiary} 총매출 {_fmt_money(perf.get('rev_curr'))}({_fmt_pct_yoy(perf.get('rev_yoy'))})로 {trend}. "
        f"투자 {_fmt_money(perf.get('spend_curr'))}({_fmt_pct_yoy(perf.get('spend_yoy'))}), "
        f"ROAS {_fmt_roas(perf.get('roas_curr'))}({_fmt_pct_yoy(perf.get('roas_yoy'))}). "
        f"이번 리포트는 {mode_kr} Top3 기준."
    )


def _subsidiary_detail_comment(mode: str, top3: List[Dict[str, Any]]) -> str:
    if not top3:
        return "선별된 Top3가 없습니다."

    contrib_label = "하락 기여율" if mode == "ISSUE" else "개선 기여율"
    lines: List[str] = []
    for idx, row in enumerate(top3, start=1):
        path = row.get("path", {})
        if not isinstance(path, dict):
            path = {}
        channel = str(path.get("CHANNEL", ""))
        division = str(path.get("DIVISION", ""))
        product = str(path.get("PRODUCT", ""))
        contrib = str(row.get("impact_contribution_text", "N/A"))
        driver = str(row.get("primary_driver", "NONE"))
        lines.append(
            f"{idx}) {channel}/{division}/{product}: {contrib_label} {contrib}, 주요 지표 {driver}"
        )
    return " | ".join(lines)


def _build_subsidiary_reports(
    df: pl.DataFrame,
    subsidiaries: List[str],
    issues_top3_by_sub: Dict[str, List[Dict[str, Any]]],
    improvements_top3_by_sub: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    perf_map = _subsidiary_perf_map(df)
    reports: List[Dict[str, Any]] = []

    for subsidiary in subsidiaries:
        sub = str(subsidiary)
        perf = _subsidiary_metrics(perf_map.get(sub, {}))
        mode, selected_top3 = _choose_subsidiary_mode(
            sub,
            perf,
            issues_top3_by_sub,
            improvements_top3_by_sub,
        )
        reports.append(
            {
                "SUBSIDIARY": sub,
                "mode": mode,
                "overall_comment": _subsidiary_overall_comment(sub, perf, mode),
                "detailed_comment": _subsidiary_detail_comment(mode, selected_top3),
                "performance": perf,
                "top3": selected_top3,
            }
        )
    return reports


def _campaign_sheet_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    columns = [
        "diagnosis_subsidiary",
        *DIMENSIONS,
        "Revenue_curr_sum",
        "Revenue_prev_sum",
        "Spend_curr_sum",
        "Spend_prev_sum",
        "Clicks_curr_sum",
        "Clicks_prev_sum",
        "Orders_curr_sum",
        "Orders_prev_sum",
        "Impact_raw",
        "Impact_adj",
        "Share_childsum",
        "Share_parentdelta",
        "Residual_parent",
        "Residual_applied_flag",
        "mode_tag",
        "primary_driver",
        "tag",
        "dlog_CVR",
        "dlog_AOV",
        "dlog_CPC",
        "total_dlog_roas",
        "CPC_curr",
        "CPC_prev",
        "CVR_curr",
        "CVR_prev",
        "AOV_curr",
        "AOV_prev",
        "ROAS_curr",
        "ROAS_prev",
        "campaign_diagnosis_mode",
        "recommended_actions",
        "action_checklist",
        "drill_trace_text",
    ]
    if not rows:
        return pl.DataFrame({col: [] for col in columns})

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized_rows.append({col: row.get(col) for col in columns})
    return pl.DataFrame(normalized_rows).select(columns)


def _campaign_pretty_sheet_df(rows: List[Dict[str, Any]]) -> pl.DataFrame:
    columns = [
        "subsidiary",
        "media",
        "bu",
        "product",
        "mode",
        "driver",
        "impact_revenue",
        "revenue_2026",
        "revenue_2025",
        "revenue_yoy",
        "spend_2026",
        "spend_2025",
        "spend_yoy",
        "roas_2026",
        "roas_2025",
        "roas_yoy",
        "tag",
        "action_one",
        "checklist",
        "trace",
    ]
    if not rows:
        return pl.DataFrame({col: [] for col in columns})

    pretty_rows: List[Dict[str, Any]] = []
    for row in rows:
        rev_curr = _to_float(row.get("Revenue_curr_sum"))
        rev_prev = _to_float(row.get("Revenue_prev_sum"))
        spend_curr = _to_float(row.get("Spend_curr_sum"))
        spend_prev = _to_float(row.get("Spend_prev_sum"))
        roas_curr = row.get("ROAS_curr")
        roas_prev = row.get("ROAS_prev")
        if roas_curr is None:
            roas_curr = _safe_ratio(rev_curr, spend_curr)
        if roas_prev is None:
            roas_prev = _safe_ratio(rev_prev, spend_prev)

        pretty_rows.append(
            {
                "subsidiary": row.get("diagnosis_subsidiary"),
                "media": row.get("CHANNEL"),
                "bu": row.get("DIVISION"),
                "product": row.get("PRODUCT"),
                "mode": row.get("mode_tag"),
                "driver": row.get("primary_driver"),
                "impact_revenue": _fmt_money_signed(row.get("Impact_adj")),
                "revenue_2026": _fmt_money(rev_curr),
                "revenue_2025": _fmt_money(rev_prev),
                "revenue_yoy": _fmt_pct(_safe_pct_change(rev_curr, rev_prev)),
                "spend_2026": _fmt_money(spend_curr),
                "spend_2025": _fmt_money(spend_prev),
                "spend_yoy": _fmt_pct(_safe_pct_change(spend_curr, spend_prev)),
                "roas_2026": _fmt_roas(roas_curr),
                "roas_2025": _fmt_roas(roas_prev),
                "roas_yoy": _fmt_pct(_safe_pct_change(roas_curr, roas_prev)),
                "tag": row.get("tag"),
                "action_one": row.get("recommended_actions"),
                "checklist": row.get("action_checklist"),
                "trace": row.get("drill_trace_text"),
            }
        )

    return pl.DataFrame(pretty_rows).select(columns)


def _division_sheet_df(divisions: List[Dict[str, Any]]) -> pl.DataFrame:
    columns = [
        "SUBSIDIARY",
        "DIVISION",
        "Revenue_curr_sum",
        "Revenue_prev_sum",
        "Spend_curr_sum",
        "Spend_prev_sum",
        "Revenue_delta",
        "Revenue_yoy_pct",
        "Spend_yoy_pct",
        "ROAS_curr",
        "ROAS_prev",
        "ROAS_yoy_pct",
        "Impact_raw",
        "Impact_adj",
        "Impact_raw_method",
        "Residual_parent",
        "Residual_applied_flag",
        "Parent_revenue_delta",
        "Contribution_to_total_delta_pct",
        "Contribution_to_decline_pct",
        "Contribution_to_growth_pct",
        "Contribution_label",
        "summary_text",
    ]
    flat_rows: List[Dict[str, Any]] = []
    for block in divisions:
        if not isinstance(block, dict):
            continue
        rows = block.get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            flat_rows.append({col: row.get(col) for col in columns})
    if not flat_rows:
        return pl.DataFrame({col: [] for col in columns})
    return pl.DataFrame(flat_rows).select(columns)


def _flatten_trace(drill_trace: Dict[str, Any]) -> pl.DataFrame:
    columns = [
        "step",
        "focus_SUBSIDIARY",
        "level",
        "rule",
        *[f"parent_{dim}" for dim in DIMENSIONS],
        "parent_delta",
        "top1_share_childsum",
        "top2_share_childsum",
        "impact_floor",
        "selected_paths",
    ]
    events = drill_trace.get("events", []) if isinstance(drill_trace, dict) else []
    if not isinstance(events, list) or not events:
        return pl.DataFrame({col: [] for col in columns})

    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        parent_path = event.get("parent_path", {})
        if not isinstance(parent_path, dict):
            parent_path = {}
        selected_paths = event.get("selected_paths", [])
        selected_tokens: list[str] = []
        if isinstance(selected_paths, list):
            for path_obj in selected_paths:
                if isinstance(path_obj, dict):
                    token = "/".join(
                        str(path_obj.get(dim, "")) for dim in DIMENSIONS if path_obj.get(dim) is not None
                    )
                    selected_tokens.append(token)
        row_data = {
            "step": idx,
            "focus_SUBSIDIARY": event.get("focus_subsidiary"),
            "level": event.get("level"),
            "rule": event.get("rule"),
            "parent_delta": event.get("parent_delta"),
            "top1_share_childsum": event.get("top1_share_childsum"),
            "top2_share_childsum": event.get("top2_share_childsum"),
            "impact_floor": event.get("impact_floor"),
            "selected_paths": " ; ".join(selected_tokens),
        }
        for dim in DIMENSIONS:
            row_data[f"parent_{dim}"] = parent_path.get(dim)
        rows.append(row_data)

    if not rows:
        return pl.DataFrame({col: [] for col in columns})
    return pl.DataFrame(rows).select(columns)


def run_reporting_pipeline() -> None:
    pipeline_start = perf_counter()
    stage_start = pipeline_start
    stage_timings: list[tuple[str, float]] = []

    def _mark(stage_name: str) -> None:
        nonlocal stage_start
        now = perf_counter()
        stage_timings.append((stage_name, now - stage_start))
        stage_start = now

    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "input.xlsx"
    output_json_path = project_root / "output" / "summary.json"
    output_excel_path = project_root / "output" / "summary.xlsx"
    output_html_path = project_root / "output" / "summary.html"

    df, comparison_meta = load_input_frame(input_path, curr_year=CURRENT_YEAR, prev_year=PREVIOUS_YEAR)
    _mark("load_input_frame")
    analysis = run_subsidiary_analysis(df, dimensions=DIMENSIONS)
    _mark("run_subsidiary_analysis")
    subsidiaries = analysis.subsidiaries
    issues_with_actions = analysis.issues_with_actions
    improvements_with_actions = analysis.improvements_with_actions
    trace_by_subsidiary = analysis.trace_by_subsidiary

    scope_maps = _scope_maps(df)
    issues_report_rows = _to_report_rows(issues_with_actions, df=df, scope_maps=scope_maps)
    improvements_report_rows = _to_report_rows(improvements_with_actions, df=df, scope_maps=scope_maps)
    issues_top3_by_sub = _top3_by_subsidiary(issues_report_rows, descending=True)
    improvements_top3_by_sub = _top3_by_subsidiary(improvements_report_rows, descending=True)
    division_parallel = _division_parallel_contribution(df)
    subsidiary_reports = _build_subsidiary_reports(
        df,
        subsidiaries=[str(sub) for sub in subsidiaries],
        issues_top3_by_sub=issues_top3_by_sub,
        improvements_top3_by_sub=improvements_top3_by_sub,
    )
    _mark("build_report_rows")

    summary = {
        "comparison_meta": comparison_meta,
        "subsidiaries_analyzed": subsidiaries,
        "decomposition_method": {
            "impact_raw": "LMDI(log-mean) with direct delta fallback when log condition is invalid",
            "impact_adj": "Impact_raw + residual correction allocated to max(|Impact_raw|) child",
        "division_contribution_pct": "Contribution_to_decline_pct/growth_pct is based on Impact_adj vs parent revenue delta",
        },
        "division_parallel_contribution": division_parallel,
        "subsidiary_reports": subsidiary_reports,
        "issues_top3_products_by_subsidiary": issues_top3_by_sub,
        "improvements_top3_products_by_subsidiary": improvements_top3_by_sub,
        "issues_top_products": issues_report_rows,
        "improvements_top_products": improvements_report_rows,
    }
    _mark("build_summary")

    save_summary_json(output_json_path, summary)
    save_summary_html(output_html_path, summary, df)
    _mark("save_json_html")

    issues_df = _campaign_sheet_df(issues_with_actions)
    improvements_df = _campaign_sheet_df(improvements_with_actions)
    issues_view_df = _campaign_pretty_sheet_df(issues_with_actions)
    improvements_view_df = _campaign_pretty_sheet_df(improvements_with_actions)
    division_df = _division_sheet_df(division_parallel)
    combined_events: list[dict[str, Any]] = []
    for subsidiary, drill_trace in trace_by_subsidiary.items():
        events = drill_trace.get("events", []) if isinstance(drill_trace, dict) else []
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            item = dict(event)
            item["focus_subsidiary"] = subsidiary
            combined_events.append(item)
    trace_df = _flatten_trace({"focus_path": {"SUBSIDIARY": "ALL"}, "events": combined_events})
    _mark("build_excel_frames")
    excel_saved, excel_error_message = save_output_workbook(
        output_excel_path,
        {
            "issues": issues_df,
            "improvements": improvements_df,
            "issues_view": issues_view_df,
            "improvements_view": improvements_view_df,
            "division_parallel": division_df,
            "trace": trace_df,
        },
    )
    _mark("save_excel")
    total_elapsed = perf_counter() - pipeline_start

    print(
        "Summary prepared: "
        f"subsidiaries={len(subsidiaries)}, "
        f"issue_topics={len(issues_report_rows)}, "
        f"improve_topics={len(improvements_report_rows)}"
    )
    stage_text = ", ".join([f"{name}={seconds:.3f}s" for name, seconds in stage_timings])
    print(f"Stage Timing: {stage_text}")
    print(f"Total Elapsed: {total_elapsed:.3f}s")
    print(f"Comparison Meta: {comparison_meta}")
    print(f"Saved JSON: {output_json_path}")
    print(f"Saved HTML: {output_html_path}")
    if excel_saved:
        print(f"Saved Excel: {output_excel_path}")
    else:
        print(f"Excel save skipped (file may be open/locked): {excel_error_message}")
