"""HTML report generator for issue + driver topics by SUBSIDIARY."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl


DIMENSIONS: List[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"]
TOPIC_LIMIT = 3
MIN_TOPIC_IMPACT_PCT = 0.03


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


def _fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    pct = value * 100
    if signed:
        return f"{pct:+.0f}%".replace("+", "")
    return f"{pct:.0f}%"


def _fmt_pct_yoy(value: float | None) -> str:
    base = _fmt_pct(value, signed=True)
    if base == "N/A":
        return base
    return f"{base} YoY"


def _fmt_roas(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def _fmt_cpc(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.3f}"


def _fmt_cvr(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _sum_aggs() -> list[pl.Expr]:
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


def _metric_pack(scope_row: Dict[str, Any]) -> Dict[str, Any]:
    revenue_curr = _to_float(scope_row.get("Revenue_curr_sum"))
    revenue_prev = _to_float(scope_row.get("Revenue_prev_sum"))
    cost_curr = _to_float(scope_row.get("Spend_curr_sum"))
    cost_prev = _to_float(scope_row.get("Spend_prev_sum"))
    clicks_curr = _to_float(scope_row.get("Clicks_curr_sum"))
    clicks_prev = _to_float(scope_row.get("Clicks_prev_sum"))
    orders_curr = _to_float(scope_row.get("Orders_curr_sum"))
    orders_prev = _to_float(scope_row.get("Orders_prev_sum"))

    roas_curr = _safe_ratio(revenue_curr, cost_curr)
    roas_prev = _safe_ratio(revenue_prev, cost_prev)
    cpc_curr = _safe_ratio(cost_curr, clicks_curr)
    cpc_prev = _safe_ratio(cost_prev, clicks_prev)
    cvr_curr = _safe_ratio(orders_curr, clicks_curr)
    cvr_prev = _safe_ratio(orders_prev, clicks_prev)
    aov_curr = _safe_ratio(revenue_curr, orders_curr)
    aov_prev = _safe_ratio(revenue_prev, orders_prev)

    return {
        "cost_curr": cost_curr,
        "cost_prev": cost_prev,
        "cost_yoy": _safe_pct_change(cost_curr, cost_prev),
        "revenue_curr": revenue_curr,
        "revenue_prev": revenue_prev,
        "revenue_delta": revenue_curr - revenue_prev,
        "revenue_yoy": _safe_pct_change(revenue_curr, revenue_prev),
        "roas_curr": roas_curr,
        "roas_prev": roas_prev,
        "roas_yoy": _safe_pct_change(roas_curr, roas_prev),
        "cpc_curr": cpc_curr,
        "cpc_prev": cpc_prev,
        "cpc_yoy": _safe_pct_change(cpc_curr, cpc_prev),
        "cvr_curr": cvr_curr,
        "cvr_prev": cvr_prev,
        "cvr_yoy": _safe_pct_change(cvr_curr, cvr_prev),
        "aov_curr": aov_curr,
        "aov_prev": aov_prev,
        "aov_yoy": _safe_pct_change(aov_curr, aov_prev),
    }


def _path_parts(row: Dict[str, Any]) -> tuple[str, str, str, str]:
    path = row.get("path", {})
    if not isinstance(path, dict):
        path = {}
    return (
        str(path.get("SUBSIDIARY", "")) or str(row.get("diagnosis_subsidiary", "")),
        str(path.get("CHANNEL", "")),
        str(path.get("DIVISION", "")),
        str(path.get("PRODUCT", "")),
    )


def _has_yoy_basis(row: Dict[str, Any]) -> bool:
    for curr_key, prev_key in [("CVR_curr", "CVR_prev"), ("AOV_curr", "AOV_prev"), ("CPC_curr", "CPC_prev")]:
        curr = row.get(curr_key)
        prev = row.get(prev_key)
        if curr is None or prev is None:
            continue
        prev_val = _to_float(prev)
        if prev_val <= 0:
            continue
        curr_val = _to_float(curr)
        if _safe_pct_change(curr_val, prev_val) is not None:
            return True
    return False


def _is_actionable_topic(row: Dict[str, Any]) -> bool:
    pct = row.get("impact_contribution_pct")
    low_impact = pct is None or _to_float(pct) < MIN_TOPIC_IMPACT_PCT
    return _has_yoy_basis(row) or not low_impact


def _scope_maps(
    df: pl.DataFrame,
) -> tuple[
    Dict[Tuple[Any, ...], Dict[str, Any]],
    Dict[Tuple[Any, ...], Dict[str, Any]],
    Dict[Tuple[Any, ...], Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
]:
    channel_rows = df.group_by(["SUBSIDIARY", "CHANNEL"]).agg(_sum_aggs()).to_dicts()
    bu_rows = df.group_by(["SUBSIDIARY", "CHANNEL", "DIVISION"]).agg(_sum_aggs()).to_dicts()
    product_rows = df.group_by(DIMENSIONS).agg(_sum_aggs()).to_dicts()

    channel_map = {(r.get("SUBSIDIARY"), r.get("CHANNEL")): r for r in channel_rows}
    bu_map = {(r.get("SUBSIDIARY"), r.get("CHANNEL"), r.get("DIVISION")): r for r in bu_rows}
    product_map = {(r.get("SUBSIDIARY"), r.get("CHANNEL"), r.get("DIVISION"), r.get("PRODUCT")): r for r in product_rows}

    product_rows_by_sub: Dict[str, List[Dict[str, Any]]] = {}
    for r in product_rows:
        sub = str(r.get("SUBSIDIARY", ""))
        product_rows_by_sub.setdefault(sub, []).append(r)
    return channel_map, bu_map, product_map, product_rows_by_sub


def _group_rows_by_sub(summary: Dict[str, Any], dict_key: str, list_key: str, mode_tag: str) -> Dict[str, List[Dict[str, Any]]]:
    top3 = summary.get(dict_key, {})
    if isinstance(top3, dict) and top3:
        output: Dict[str, List[Dict[str, Any]]] = {}
        for sub, rows in top3.items():
            if not isinstance(rows, list):
                continue
            output[str(sub)] = [r for r in rows if isinstance(r, dict) and _is_actionable_topic(r)][:TOPIC_LIMIT]
        return output

    rows = summary.get(list_key, [])
    if not isinstance(rows, list):
        return {}
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("mode_tag", "")).upper() != mode_tag:
            continue
        if not _is_actionable_topic(row):
            continue
        sub = str(row.get("diagnosis_subsidiary", ""))
        if not sub:
            continue
        grouped.setdefault(sub, []).append(row)
    for sub, items in grouped.items():
        items.sort(key=lambda x: -_to_float(x.get("impact_contribution_pct")))
        grouped[sub] = items[:TOPIC_LIMIT]
    return grouped


def _driver_from_metrics(metrics: Dict[str, Any], mode: str) -> str:
    cvr_yoy = metrics.get("cvr_yoy")
    aov_yoy = metrics.get("aov_yoy")
    cpc_yoy = metrics.get("cpc_yoy")

    if mode == "ISSUE":
        scores = {
            "CVR": max(0.0, -_to_float(cvr_yoy)),
            "AOV": max(0.0, -_to_float(aov_yoy)),
            "CPC": max(0.0, _to_float(cpc_yoy)),
        }
    else:
        scores = {
            "CVR": max(0.0, _to_float(cvr_yoy)),
            "AOV": max(0.0, _to_float(aov_yoy)),
            "CPC": max(0.0, -_to_float(cpc_yoy)),
        }
    if max(scores.values()) > 0:
        return max(scores.items(), key=lambda x: (x[1], x[0]))[0]
    return max(
        {
            "CVR": abs(_to_float(cvr_yoy)),
            "AOV": abs(_to_float(aov_yoy)),
            "CPC": abs(_to_float(cpc_yoy)),
        }.items(),
        key=lambda x: (x[1], x[0]),
    )[0]


def _driver_reason(driver: str, mode: str, topic_row: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[str, List[str]]:
    cvr_yoy = _safe_pct_change(topic_row.get("CVR_curr"), topic_row.get("CVR_prev"))
    aov_yoy = _safe_pct_change(topic_row.get("AOV_curr"), topic_row.get("AOV_prev"))
    cpc_yoy = _safe_pct_change(topic_row.get("CPC_curr"), topic_row.get("CPC_prev"))
    if cvr_yoy is None:
        cvr_yoy = metrics.get("cvr_yoy")
    if aov_yoy is None:
        aov_yoy = metrics.get("aov_yoy")
    if cpc_yoy is None:
        cpc_yoy = metrics.get("cpc_yoy")

    yoy_map = {"CVR": cvr_yoy, "AOV": aov_yoy, "CPC": cpc_yoy}
    dlog_map = {
        "CVR": abs(_to_float(topic_row.get("dlog_CVR"))),
        "AOV": abs(_to_float(topic_row.get("dlog_AOV"))),
        "CPC": abs(_to_float(topic_row.get("dlog_CPC"))),
    }

    if mode == "ISSUE":
        yoy_score = {
            "CVR": max(0.0, -_to_float(cvr_yoy)),
            "AOV": max(0.0, -_to_float(aov_yoy)),
            "CPC": max(0.0, _to_float(cpc_yoy)),
        }
        mode_word = "악화"
    else:
        yoy_score = {
            "CVR": max(0.0, _to_float(cvr_yoy)),
            "AOV": max(0.0, _to_float(aov_yoy)),
            "CPC": max(0.0, -_to_float(cpc_yoy)),
        }
        mode_word = "개선"

    top_yoy = max(yoy_score.items(), key=lambda x: (x[1], x[0]))[0] if max(yoy_score.values()) > 0 else None

    if top_yoy == driver and yoy_map.get(driver) is not None:
        yoy_text = f"{driver} YoY {_fmt_pct(yoy_map.get(driver))}로 3개 지표 중 {mode_word}폭 최대"
    elif yoy_map.get(driver) is not None and top_yoy is not None:
        yoy_text = f"{driver} YoY {_fmt_pct(yoy_map.get(driver))}, YoY 최대 {mode_word}는 {top_yoy}({_fmt_pct(yoy_map.get(top_yoy))})"
    elif yoy_map.get(driver) is not None:
        yoy_text = f"{driver} YoY {_fmt_pct(yoy_map.get(driver))}"
    else:
        yoy_text = f"{driver} YoY N/A"

    metric_desc = {
        "AOV": "객단가(AOV) 변동",
        "CPC": "클릭 단가(CPC) 변동",
        "CVR": "전환율(CVR) 변동",
    }

    def _effect_direction(metric: str, mode_name: str, dlog_value: Any) -> str:
        metric_key = str(metric).upper()
        mode_key = str(mode_name).upper()
        val = _to_float(dlog_value)
        if abs(val) < 1e-12:
            return "중립"
        if mode_key == "ISSUE":
            if metric_key in {"CVR", "AOV"}:
                return "하락 유발" if val < 0 else "하락 완화"
            if metric_key == "CPC":
                return "하락 유발" if val > 0 else "하락 완화"
            return "하락 영향"
        if metric_key in {"CVR", "AOV"}:
            return "개선 유발" if val > 0 else "개선 저해"
        if metric_key == "CPC":
            return "개선 유발" if val < 0 else "개선 저해"
        return "개선 영향"

    dlog_signed_map = {
        "CVR": topic_row.get("dlog_CVR"),
        "AOV": topic_row.get("dlog_AOV"),
        "CPC": topic_row.get("dlog_CPC"),
    }
    dlog_lines: List[str] = []
    if max(dlog_map.values()) > 0:
        ranked = sorted(dlog_map.items(), key=lambda item: (-item[1], item[0]))
        for rank, (metric, val) in enumerate(ranked, start=1):
            approx_pct = int(round(val * 100))
            direction = _effect_direction(metric, mode, dlog_signed_map.get(metric))
            suffix = " (영향력 1위)" if rank == 1 else ""
            dlog_lines.append(
                f"{metric} ({val:.3f}): {direction}, 성과 변화 설명력 약 {approx_pct}% ({metric_desc.get(metric, metric)}){suffix}"
            )
    else:
        dlog_lines.append("|dlog| 값이 없어 영향력 분해 근거가 부족")

    return yoy_text, dlog_lines


def _action_text(mode: str, driver: str) -> str:
    mode = str(mode or "").upper()
    driver = str(driver or "NONE").upper()

    if mode == "ISSUE":
        if driver == "CPC":
            return "High-intent 키워드/오디언스 비중 확대 + 저CTR·고CPC 쿼리 제외로 유입 효율 복원"
        if driver == "CVR":
            return "유입 의도(키워드/오디언스)와 랜딩 메시지 정합성 개선으로 전환 효율 복원"
        if driver == "AOV":
            return "tROAS 상향/세분화 + 고가 SKU(예: S25 Ultra, 1TB) 유입 비중 확대"
        return "핵심 퍼널 지표 재점검 후 타게팅/입찰 규칙 재설정"
    if driver == "CPC":
        return "저CPC·고CTR 세그먼트 점진 증액으로 효율 유지"
    if driver == "CVR":
        return "고CVR 키워드/오디언스·랜딩 조합 복제 확장"
    if driver == "AOV":
        return "tROAS 기반 고AOV SKU/번들 트래픽 확대"
    return "개선 패턴 재현 테스트"


def _checklist_text(mode: str, driver: str) -> str:
    mode = str(mode or "").upper()
    driver = str(driver or "NONE").upper()

    if driver == "CPC":
        if mode == "ISSUE":
            return "1) 저의도 쿼리 제외/네거티브 확장; 2) CTR(IMP/CLICK)·CPC 동시 점검; 3) High-intent 키워드/오디언스 비중 확대"
        return "1) 저CPC·고CTR 세그먼트 예산 확대; 2) 점유율 손실 여부 점검; 3) CPC 급등 알림/입찰 상한 유지"
    if driver == "CVR":
        if mode == "ISSUE":
            return "1) 키워드/오디언스 의도와 랜딩 일치 점검; 2) CVR 낮은 검색어·오디언스는 제외/입찰 하향하고 고CVR 세그먼트로 예산 재배분; 3) 전환 가능성 낮은 유입 지면·매체 비중 축소"
        return "1) 고CVR 키워드/오디언스 확장; 2) 승자 랜딩/소재 복제; 3) 스케일 시 CVR 하락 가드레일 설정"
    if driver == "AOV":
        if mode == "ISSUE":
            return "1) tROAS 목표 상향/세분화; 2) 고가 SKU(예: S25 Ultra, 1TB) 키워드/오디언스 확대; 3) 저객단가 쿼리/지면 입찰 하향"
        return "1) tROAS 유지로 고가 구매 유저 탐색 확대; 2) 고AOV SKU/번들 키워드 점유율 유지; 3) 할인 중심 유입 비중 과다 점검"
    return "1) 타게팅/입찰/소재 기본 세팅 점검; 2) 트래픽 품질 점검; 3) 예산 배분 재검토"


def _yoy_css_class(metric_name: str, value: float | None) -> str:
    if value is None:
        return "yoy-na"
    if abs(value) < 1e-12:
        return "yoy-neutral"

    key = str(metric_name).lower()
    if key == "cpc":
        return "yoy-pos" if value < 0 else "yoy-neg"
    return "yoy-pos" if value > 0 else "yoy-neg"


def _render_yoy_cell(metric_name: str, value: float | None) -> str:
    css = _yoy_css_class(metric_name, value)
    return f"<td class=\"{css}\">{escape(_fmt_pct(value))}</td>"


def _render_metric_table(metrics: Dict[str, Any]) -> str:
    return (
        "<table class=\"metric-table\">"
        "<thead><tr><th>Year</th><th>Cost</th><th>Revenue</th><th>ROAS</th><th>CPC</th><th>CVR</th></tr></thead>"
        "<tbody>"
        f"<tr><td>2026</td><td>{escape(_fmt_money(metrics.get('cost_curr')))}</td>"
        f"<td>{escape(_fmt_money(metrics.get('revenue_curr')))}</td>"
        f"<td>{escape(_fmt_roas(metrics.get('roas_curr')))}</td>"
        f"<td>{escape(_fmt_cpc(metrics.get('cpc_curr')))}</td>"
        f"<td>{escape(_fmt_cvr(metrics.get('cvr_curr')))}</td></tr>"
        f"<tr><td>2025</td><td>{escape(_fmt_money(metrics.get('cost_prev')))}</td>"
        f"<td>{escape(_fmt_money(metrics.get('revenue_prev')))}</td>"
        f"<td>{escape(_fmt_roas(metrics.get('roas_prev')))}</td>"
        f"<td>{escape(_fmt_cpc(metrics.get('cpc_prev')))}</td>"
        f"<td>{escape(_fmt_cvr(metrics.get('cvr_prev')))}</td></tr>"
        "<tr class=\"yoy\"><td>YoY</td>"
        f"{_render_yoy_cell('cost', metrics.get('cost_yoy'))}"
        f"{_render_yoy_cell('revenue', metrics.get('revenue_yoy'))}"
        f"{_render_yoy_cell('roas', metrics.get('roas_yoy'))}"
        f"{_render_yoy_cell('cpc', metrics.get('cpc_yoy'))}"
        f"{_render_yoy_cell('cvr', metrics.get('cvr_yoy'))}</tr>"
        "</tbody></table>"
    )


def _fallback_topics_for_sub(sub: str, rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        channel = str(row.get("CHANNEL", ""))
        bu = str(row.get("DIVISION", ""))
        product = str(row.get("PRODUCT", ""))
        metrics = _metric_pack(row)
        revenue_delta = _to_float(metrics.get("revenue_delta"))
        enriched.append(
            {
                "path": {"SUBSIDIARY": sub, "CHANNEL": channel, "DIVISION": bu, "PRODUCT": product},
                "metrics": metrics,
                "revenue_delta": revenue_delta,
            }
        )

    if mode == "ISSUE":
        primary = [x for x in enriched if x["revenue_delta"] < 0]
        secondary = [x for x in enriched if x["revenue_delta"] >= 0]
        primary.sort(key=lambda x: abs(_to_float(x["revenue_delta"])), reverse=True)
        secondary.sort(key=lambda x: abs(_to_float(x["revenue_delta"])), reverse=True)
        denom = sum(abs(_to_float(x["revenue_delta"])) for x in primary)
    else:
        primary = [x for x in enriched if x["revenue_delta"] > 0]
        secondary = [x for x in enriched if x["revenue_delta"] <= 0]
        primary.sort(key=lambda x: abs(_to_float(x["revenue_delta"])), reverse=True)
        secondary.sort(key=lambda x: abs(_to_float(x["revenue_delta"])), reverse=True)
        denom = sum(abs(_to_float(x["revenue_delta"])) for x in primary)

    selected = primary + secondary
    out: List[Dict[str, Any]] = []
    for item in selected:
        path = item["path"]
        metrics = item["metrics"]
        revenue_delta = _to_float(item["revenue_delta"])
        driver = _driver_from_metrics(metrics, mode)
        contribution_pct = abs(revenue_delta) / denom if denom > 0 else None
        if mode == "ISSUE":
            label = "하락 기여율" if revenue_delta < 0 else "상쇄 기여율"
        else:
            label = "개선 기여율" if revenue_delta > 0 else "역풍 기여율"

        fake_row = {
            "path": path,
            "impact_contribution_pct": contribution_pct,
            "impact_contribution_text": _fmt_pct(contribution_pct, signed=False),
            "primary_driver": driver,
            "mode_tag": mode,
            "CVR_curr": metrics.get("cvr_curr"),
            "CVR_prev": metrics.get("cvr_prev"),
            "CPC_curr": metrics.get("cpc_curr"),
            "CPC_prev": metrics.get("cpc_prev"),
            "AOV_curr": metrics.get("aov_curr"),
            "AOV_prev": metrics.get("aov_prev"),
            "dlog_CVR": None,
            "dlog_AOV": None,
            "dlog_CPC": None,
            "summary_text": (
                f"{path['PRODUCT']} | Revenue {_fmt_money(metrics.get('revenue_curr'))}({_fmt_pct_yoy(metrics.get('revenue_yoy'))}) "
                f"| Cost {_fmt_money(metrics.get('cost_curr'))}({_fmt_pct_yoy(metrics.get('cost_yoy'))}) "
                f"| ROAS {_fmt_roas(metrics.get('roas_curr'))}({_fmt_pct_yoy(metrics.get('roas_yoy'))})"
            ),
            "why_selected_text": f"{label} { _fmt_pct(contribution_pct, signed=False) } 기준 보강 토픽",
            "recommended_actions": _action_text(mode, driver),
            "action_checklist": _checklist_text(mode, driver),
            "contribution_label": label,
        }
        out.append(fake_row)
    return out


def _merge_topics(
    sub: str,
    existing_by_sub: Dict[str, List[Dict[str, Any]]],
    fallback_by_sub: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    existing = existing_by_sub.get(sub, [])
    fallback = fallback_by_sub.get(sub, [])
    candidates: List[Dict[str, Any]] = []
    seen_triplets: set[tuple[str, str, str]] = set()
    for row in [*existing, *fallback]:
        if not isinstance(row, dict):
            continue
        if not _is_actionable_topic(row):
            continue
        _, channel, bu, product = _path_parts(row)
        triplet = (channel, bu, product)
        if triplet in seen_triplets:
            continue
        seen_triplets.add(triplet)
        candidates.append(row)

    merged: List[Dict[str, Any]] = []
    used_channels: set[str] = set()
    used_bus: set[str] = set()
    used_channel_bu: set[tuple[str, str]] = set()
    picked_triplets: set[tuple[str, str, str]] = set()

    def _pick(stage: str) -> None:
        for row in candidates:
            if len(merged) >= TOPIC_LIMIT:
                return
            _, channel, bu, product = _path_parts(row)
            triplet = (channel, bu, product)
            channel_bu = (channel, bu)
            if triplet in picked_triplets:
                continue

            is_new_pair = channel_bu not in used_channel_bu
            is_new_channel = channel not in used_channels
            is_new_bu = bu not in used_bus

            if stage == "strict" and not (is_new_pair and is_new_channel and is_new_bu):
                continue
            if stage == "semi" and not (is_new_pair and (is_new_channel or is_new_bu)):
                continue
            if stage == "pair" and not is_new_pair:
                continue

            merged.append(row)
            picked_triplets.add(triplet)
            used_channels.add(channel)
            used_bus.add(bu)
            used_channel_bu.add(channel_bu)

    # Keep existing ranking priority while minimizing repeated media/BU exposures.
    _pick("strict")
    _pick("semi")
    _pick("pair")
    _pick("fill")
    return merged[:TOPIC_LIMIT]


def _build_topic_html(
    row: Dict[str, Any],
    mode: str,
    idx: int,
    channel_map: Dict[Tuple[Any, ...], Dict[str, Any]],
    bu_map: Dict[Tuple[Any, ...], Dict[str, Any]],
    product_map: Dict[Tuple[Any, ...], Dict[str, Any]],
) -> str:
    sub, channel, bu, product = _path_parts(row)
    channel_metrics = _metric_pack(channel_map.get((sub, channel), {}))
    bu_metrics = _metric_pack(bu_map.get((sub, channel, bu), {}))
    product_metrics = _metric_pack(product_map.get((sub, channel, bu, product), {}))

    contribution_text = str(row.get("impact_contribution_text", "N/A"))
    contribution_label = str(row.get("contribution_label", "하락 기여율" if mode == "ISSUE" else "개선 기여율"))
    driver = str(row.get("primary_driver", _driver_from_metrics(product_metrics, mode)))
    reason_head, dlog_lines = _driver_reason(driver, mode, row, product_metrics)
    action = str(row.get("recommended_actions", _action_text(mode, driver)))
    checklist = str(row.get("action_checklist", _checklist_text(mode, driver)))

    what_line = (
        f"What: {channel}에서 Cost {_fmt_money(channel_metrics.get('cost_curr'))}"
        f"({_fmt_pct_yoy(channel_metrics.get('cost_yoy'))}), Revenue {_fmt_money(channel_metrics.get('revenue_curr'))}"
        f"({_fmt_pct_yoy(channel_metrics.get('revenue_yoy'))}), ROAS {_fmt_roas(channel_metrics.get('roas_curr'))}"
        f"({_fmt_pct_yoy(channel_metrics.get('roas_yoy'))}). "
        f"미디어 내 BU {bu}는 Revenue {_fmt_money(bu_metrics.get('revenue_curr'))}"
        f"({_fmt_pct_yoy(bu_metrics.get('revenue_yoy'))}) / ROAS {_fmt_roas(bu_metrics.get('roas_curr'))}"
        f"({_fmt_pct_yoy(bu_metrics.get('roas_yoy'))})."
    )
    so_what_line = (
        f"So What: 제품 {product}가 {contribution_label} {contribution_text}로 핵심 토픽이며 "
        f"주요 지표는 {driver}."
    )
    now_what_line = f"Now What: {action}"
    overall_html = "<br>".join([escape(what_line), escape(so_what_line), escape(now_what_line)])

    detail_lines: List[str] = [f"근거: {reason_head}"]
    detail_lines.extend(dlog_lines)
    detail_lines.append(f"선정: {row.get('why_selected_text', '')}")
    detail_lines.append(f"실행: {action}")
    detail_lines.append(f"확인: {checklist}")
    detail_html = "<br>".join(escape(line) for line in detail_lines if line)

    topic_name = "Issue" if mode == "ISSUE" else "Driver"
    return (
        "<article class=\"topic\">"
        f"<h4>{topic_name} {idx}: {escape(channel)} / {escape(bu)} / {escape(product)}</h4>"
        "<div class=\"grid\">"
        "<section>"
        f"<h5>미디어 실적 ({escape(channel)})</h5>"
        f"{_render_metric_table(channel_metrics)}"
        "</section>"
        "<section>"
        f"<h5>미디어 내 BU 실적 ({escape(bu)})</h5>"
        f"{_render_metric_table(bu_metrics)}"
        "</section>"
        "</div>"
        f"<p class=\"overall\"><strong>전체 코멘트:</strong> {overall_html}</p>"
        f"<p class=\"detail\"><strong>상세 코멘트:</strong> {detail_html}</p>"
        "</article>"
    )


def _safe_head_text(value: Any) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[1].strip()
    return text


def _lmdi_driver_for_sub(summary: Dict[str, Any], sub: str, mode: str, channel: str = "") -> str:
    target_label = "DECLINE_DRIVER" if mode == "ISSUE" else "GROWTH_DRIVER"
    target_key = "Contribution_to_decline_pct" if mode == "ISSUE" else "Contribution_to_growth_pct"
    lead_text = "하락 동인" if mode == "ISSUE" else "성장 동인"

    best_division = ""
    best_pct = 0.0
    blocks = summary.get("division_parallel_contribution", [])
    if not isinstance(blocks, list):
        return "LMDI 상위 동인 분산"

    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("SUBSIDIARY", "")) != sub:
            continue

        rows = block.get("rows", [])
        if not isinstance(rows, list):
            break

        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("Contribution_label", "")).upper() != target_label:
                continue
            pct = _to_float(row.get(target_key))
            if pct > best_pct:
                best_pct = pct
                best_division = str(row.get("DIVISION", "")) or "N/A"
        break

    channel_name = str(channel or "").strip()
    if best_division and best_pct > 0:
        if channel_name:
            return f"LMDI {lead_text} {channel_name}의 {best_division}({_fmt_pct(best_pct, signed=False)})"
        return f"LMDI {lead_text} {best_division}({_fmt_pct(best_pct, signed=False)})"
    return "LMDI 상위 동인 분산"


def _short_now_what(mode: str, driver: str) -> str:
    mode_name = str(mode or "").upper()
    driver_name = str(driver or "").upper()

    if mode_name == "ISSUE":
        if driver_name == "CVR":
            return "전환지표(CVR) 파악"
        if driver_name == "CPC":
            return "유입단가(CPC) 통제"
        if driver_name == "AOV":
            return "객단가(AOV) 파악"
        return "핵심지표 원인 파악"

    if driver_name == "CVR":
        return "전환지표(CVR) 확장"
    if driver_name == "CPC":
        return "유입단가(CPC) 효율 유지"
    if driver_name == "AOV":
        return "객단가(AOV) 확장"
    return "핵심지표 성과 확장"


def _subsidiary_summary_line(
    sub: str,
    mode: str,
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
    selected_topics: List[Dict[str, Any]],
) -> str:
    what = (
        f"What: 매출 {_fmt_money(metrics.get('revenue_curr'))}({_fmt_pct_yoy(metrics.get('revenue_yoy'))}), "
        f"ROAS {_fmt_roas(metrics.get('roas_curr'))}({_fmt_pct_yoy(metrics.get('roas_yoy'))})"
    )

    top_channel = ""
    if selected_topics:
        top_path = selected_topics[0].get("path", {})
        if isinstance(top_path, dict):
            top_channel = str(top_path.get("CHANNEL", "")).strip()

    so_what = f"So What: {_lmdi_driver_for_sub(summary, sub, mode, top_channel)}"

    primary_driver = ""
    if selected_topics:
        primary_driver = str(selected_topics[0].get("primary_driver", "")).upper()
    if not primary_driver:
        primary_driver = _driver_from_metrics(metrics, mode)
    now_what = f"Now What: {_short_now_what(mode, primary_driver)}"

    return f"{sub} | {what} | {so_what} | {now_what}"


def write_html_report(output_path: Path, summary: Dict[str, Any], df: pl.DataFrame) -> None:
    channel_map, bu_map, product_map, product_rows_by_sub = _scope_maps(df)

    issues_existing = _group_rows_by_sub(
        summary,
        dict_key="issues_top3_products_by_subsidiary",
        list_key="issues_top_products",
        mode_tag="ISSUE",
    )
    improves_existing = _group_rows_by_sub(
        summary,
        dict_key="improvements_top3_products_by_subsidiary",
        list_key="improvements_top_products",
        mode_tag="IMPROVE",
    )

    fallback_issues: Dict[str, List[Dict[str, Any]]] = {}
    fallback_improves: Dict[str, List[Dict[str, Any]]] = {}
    for sub, rows in product_rows_by_sub.items():
        fallback_issues[sub] = _fallback_topics_for_sub(sub, rows, mode="ISSUE")
        fallback_improves[sub] = _fallback_topics_for_sub(sub, rows, mode="IMPROVE")

    subsidiaries = summary.get("subsidiaries_analyzed", [])
    if not isinstance(subsidiaries, list) or not subsidiaries:
        subsidiaries = sorted(product_rows_by_sub.keys())
    subsidiaries = [str(s) for s in subsidiaries]

    subsidiary_reports = summary.get("subsidiary_reports", [])
    mode_by_sub: Dict[str, str] = {}
    if isinstance(subsidiary_reports, list):
        for item in subsidiary_reports:
            if not isinstance(item, dict):
                continue
            sub_name = str(item.get("SUBSIDIARY", ""))
            mode_name = str(item.get("mode", "")).upper()
            if sub_name and mode_name in {"ISSUE", "IMPROVE"}:
                mode_by_sub[sub_name] = mode_name

    sub_metric_rows = df.group_by(["SUBSIDIARY"]).agg(_sum_aggs()).to_dicts()
    sub_metrics_by_sub: Dict[str, Dict[str, Any]] = {}
    rev_delta_by_sub: Dict[str, float] = {}
    for row in sub_metric_rows:
        sub = str(row.get("SUBSIDIARY", ""))
        metrics = _metric_pack(row)
        sub_metrics_by_sub[sub] = metrics
        rev_delta_by_sub[sub] = _to_float(metrics.get("revenue_delta"))

    summary_lines: List[str] = []
    cards: List[str] = []

    for sub in subsidiaries:
        issue_topics = _merge_topics(sub, issues_existing, fallback_issues)
        improve_topics = _merge_topics(sub, improves_existing, fallback_improves)

        selected_mode = mode_by_sub.get(sub)
        if selected_mode not in {"ISSUE", "IMPROVE"}:
            selected_mode = "ISSUE" if rev_delta_by_sub.get(sub, 0.0) < 0 else "IMPROVE"

        if selected_mode == "ISSUE":
            selected_topics = issue_topics
            block_class = "issue-block"
            block_title = "Issue Top3"
            empty_message = "이슈 토픽 없음"
        else:
            selected_topics = improve_topics
            block_class = "improve-block"
            block_title = "Driver Top3"
            empty_message = "Driver 토픽 없음"

        selected_html = "".join(
            _build_topic_html(row, selected_mode, idx + 1, channel_map, bu_map, product_map)
            for idx, row in enumerate(selected_topics[:TOPIC_LIMIT])
        )

        if not selected_html:
            selected_html = f"<p class=\"muted\">{empty_message}</p>"

        sub_metrics = sub_metrics_by_sub.get(sub, _metric_pack({}))
        summary_lines.append(
            _subsidiary_summary_line(
                sub=sub,
                mode=selected_mode,
                metrics=sub_metrics,
                summary=summary,
                selected_topics=selected_topics,
            )
        )

        cards.append(
            "<section class=\"subsidiary-card\">"
            f"<h2>{escape(sub)}</h2>"
            f"<section class=\"mode-block {block_class}\">"
            f"<h3>{block_title}</h3>"
            f"{selected_html}"
            "</section>"
            "</section>"
        )

    comparison_meta = summary.get("comparison_meta", {})
    curr_year = comparison_meta.get("curr_year", "")
    prev_year = comparison_meta.get("prev_year", "")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    summary_lines_html = "".join(f"<li class=\"summary-line\">{escape(line)}</li>" for line in summary_lines)
    if not summary_lines_html:
        summary_lines_html = "<li class=\"summary-line muted\">요약할 국가 데이터가 없습니다.</li>"

    cards_html = "".join(cards) or "<section class=\"subsidiary-card\"><p class=\"muted\">표시할 토픽이 없습니다.</p></section>"

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weekly ROAS Topic Report</title>
  <style>
    :root {{
      --bg: #f3f6fb;
      --panel: #ffffff;
      --line: #d5dce8;
      --text: #0f172a;
      --sub: #475569;
      --brand: #0f766e;
      --issue: #b91c1c;
      --improve: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #e9efff 0%, var(--bg) 35%);
      color: var(--text);
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
    }}
    .wrap {{
      max-width: 1700px;
      margin: 0 auto;
      padding: 20px;
    }}
    .panel, .subsidiary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
      padding: 14px 18px;
      margin-bottom: 14px;
    }}
    h1 {{
      margin: 0 0 8px;
      color: var(--brand);
      font-size: 28px;
    }}
    .meta {{
      color: var(--sub);
      font-size: 13px;
    }}
    .subsidiary-card h2 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    .mode-block {{
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 10px;
    }}
    .mode-block h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .issue-block h3 {{ color: var(--issue); }}
    .improve-block h3 {{ color: var(--improve); }}
    .topic {{
      border-top: 1px dashed var(--line);
      margin-top: 10px;
      padding-top: 10px;
    }}
    .topic:first-of-type {{
      border-top: none;
      margin-top: 0;
      padding-top: 0;
    }}
    .topic h4 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .topic h5 {{
      margin: 0 0 6px;
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(320px, 1fr));
      gap: 10px;
      margin-bottom: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 6px 8px;
      text-align: left;
      word-break: break-word;
    }}
    th {{
      background: #eef4ff;
      font-weight: 700;
    }}
    .yoy td {{
      font-weight: 700;
      background: #f8fafc;
    }}
    .yoy td.yoy-pos {{
      color: #1d4ed8;
    }}
    .yoy td.yoy-neg {{
      color: #b91c1c;
    }}
    .yoy td.yoy-neutral {{
      color: #475569;
    }}
    .yoy td.yoy-na {{
      color: #94a3b8;
    }}
    .overall, .detail {{
      margin: 0 0 8px;
      line-height: 1.55;
    }}
    .summary-list {{
      margin: 0;
      padding-left: 18px;
    }}
    .summary-line {{
      margin: 0 0 6px;
      line-height: 1.6;
    }}
    .muted {{ color: var(--sub); }}
    @media (max-width: 1080px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Weekly ROAS Topic Report</h1>
      <div class="meta">기준: OBJECTIVE=CONVERSION | 비교: {escape(str(curr_year))} vs {escape(str(prev_year))} | 생성시각: {escape(generated_at)}</div>
    </section>
    <section class="panel">
      <h2>Global Summary</h2>
      <ul class="summary-list">
        {summary_lines_html}
      </ul>
    </section>
    {cards_html}
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

