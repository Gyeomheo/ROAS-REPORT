"""Text rendering helpers (Now what layer)."""

from __future__ import annotations

from typing import Any, Dict, List

from src.application.reporting.metrics import fmt_money, fmt_pct_yoy, fmt_roas, to_float


def subsidiary_overall_comment(subsidiary: str, perf: Dict[str, Any], mode: str) -> str:
    roas_delta = perf.get("roas_delta")
    if roas_delta is None:
        roas_delta = 0.0
    roas_delta_value = to_float(roas_delta)

    if roas_delta_value < 0:
        trend = "하락"
    elif roas_delta_value > 0:
        trend = "개선"
    else:
        trend = "보합"

    mode_kr = "이슈" if mode == "ISSUE" else "개선"
    return (
        f"{subsidiary} ROAS {fmt_roas(perf.get('roas_curr'))}({fmt_pct_yoy(perf.get('roas_yoy'))}) 기준 {trend}. "
        f"매출 {fmt_money(perf.get('rev_curr'))}({fmt_pct_yoy(perf.get('rev_yoy'))}), "
        f"투자 {fmt_money(perf.get('spend_curr'))}({fmt_pct_yoy(perf.get('spend_yoy'))}). "
        f"이번 리포트는 ROAS 기준 {mode_kr} Top3를 제시합니다."
    )


def subsidiary_detail_comment(mode: str, top3: List[Dict[str, Any]]) -> str:
    if not top3:
        return "선별된 Top3가 없습니다."

    contrib_label = "ROAS 하락 기여율" if mode == "ISSUE" else "ROAS 개선 기여율"
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

