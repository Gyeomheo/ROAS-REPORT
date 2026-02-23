"""Text rendering helpers (Now what layer)."""

from __future__ import annotations

from typing import Any, Dict, List

from src.application.reporting.metrics import fmt_money, fmt_pct_yoy, fmt_roas, to_float


def subsidiary_overall_comment(subsidiary: str, perf: Dict[str, Any], mode: str) -> str:
    rev_delta = to_float(perf.get("rev_delta"))
    if rev_delta < 0:
        trend = "하락"
    elif rev_delta > 0:
        trend = "개선"
    else:
        trend = "보합"

    mode_kr = "이슈" if mode == "ISSUE" else "개선"
    return (
        f"{subsidiary} 총매출 {fmt_money(perf.get('rev_curr'))}({fmt_pct_yoy(perf.get('rev_yoy'))})로 {trend}. "
        f"투자 {fmt_money(perf.get('spend_curr'))}({fmt_pct_yoy(perf.get('spend_yoy'))}), "
        f"ROAS {fmt_roas(perf.get('roas_curr'))}({fmt_pct_yoy(perf.get('roas_yoy'))}). "
        f"이번 리포트는 {mode_kr} Top3 기준."
    )


def subsidiary_detail_comment(mode: str, top3: List[Dict[str, Any]]) -> str:
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

