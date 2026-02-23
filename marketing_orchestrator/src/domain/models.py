"""Domain models for campaign diagnosis rules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CampaignSignals:
    """Minimal signal set used by domain recommendation policies."""

    mode_tag: str
    tag: str
    channel: str
    primary_driver: str
    dlog_cvr: float | None
    dlog_aov: float | None
    dlog_cpc: float | None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "CampaignSignals":
        return cls(
            mode_tag=str(row.get("mode_tag", "") or "").upper(),
            tag=str(row.get("tag", "NORMAL") or "NORMAL").upper(),
            channel=str(row.get("CHANNEL", "") or ""),
            primary_driver=str(row.get("primary_driver", "NONE") or "NONE").upper(),
            dlog_cvr=_to_optional_float(row.get("dlog_CVR")),
            dlog_aov=_to_optional_float(row.get("dlog_AOV")),
            dlog_cpc=_to_optional_float(row.get("dlog_CPC")),
        )

    def with_primary_driver(self, primary_driver: str) -> "CampaignSignals":
        driver = str(primary_driver or "CPC").upper()
        return replace(self, primary_driver=driver)
