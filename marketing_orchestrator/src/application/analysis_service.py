"""Application service for cross-subsidiary ROAS diagnosis use case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import polars as pl

from src.application.campaign_action_service import enrich_campaign_actions
from src.impact import ImpactEngine
from src.root_cause import RootCauseEngine


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class AnalysisResult:
    subsidiaries: list[Any]
    issues_with_actions: list[dict[str, Any]]
    improvements_with_actions: list[dict[str, Any]]
    trace_by_subsidiary: dict[str, dict[str, Any]]


def run_subsidiary_analysis(df: pl.DataFrame, dimensions: Sequence[str]) -> AnalysisResult:
    """Run impact + root-cause analysis per subsidiary and enrich with domain actions."""
    impact_engine = ImpactEngine(rel_floor=0.01, abs_floor_amount=0.0, phantom_drop_threshold=0.30)
    root_cause_engine = RootCauseEngine()

    subsidiaries = (
        df.select(pl.col("SUBSIDIARY").cast(pl.Utf8).drop_nulls().unique().sort())
        .to_series(0)
        .to_list()
    )

    issues_with_actions: list[dict[str, Any]] = []
    improvements_with_actions: list[dict[str, Any]] = []
    trace_by_subsidiary: dict[str, dict[str, Any]] = {}
    subsidiary_frames: dict[str, pl.DataFrame] = {}
    try:
        partitioned = df.partition_by("SUBSIDIARY", maintain_order=True)
    except Exception:
        partitioned = []
    for scoped_df in partitioned:
        if scoped_df.is_empty():
            continue
        subsidiary_name = scoped_df.select(pl.col("SUBSIDIARY").cast(pl.Utf8)).to_series(0)[0]
        if subsidiary_name is None:
            continue
        subsidiary_frames[str(subsidiary_name)] = scoped_df

    for subsidiary in subsidiaries:
        subsidiary_key = str(subsidiary)
        scoped_df = subsidiary_frames.get(subsidiary_key)
        if scoped_df is None:
            scoped_df = df.filter(pl.col("SUBSIDIARY") == pl.lit(subsidiary))
        if scoped_df.is_empty():
            trace_by_subsidiary[subsidiary_key] = {"focus_path": {"SUBSIDIARY": subsidiary}, "events": []}
            continue

        focus_path = {"SUBSIDIARY": subsidiary}
        issues_top_campaigns, improvements_top_campaigns, drill_trace = impact_engine.run(scoped_df, focus_path=focus_path)
        if issues_top_campaigns or improvements_top_campaigns:
            mode_maps = root_cause_engine.build_mode_maps(scoped_df)
        else:
            mode_maps = {"ISSUE": {}, "IMPROVE": {}}

        issues_with_rca = root_cause_engine.run(
            scoped_df,
            issues_top_campaigns,
            mode="ISSUE",
            rca_map=mode_maps["ISSUE"],
        )["campaigns_with_rca"]
        improvements_with_rca = root_cause_engine.run(
            scoped_df,
            improvements_top_campaigns,
            mode="IMPROVE",
            rca_map=mode_maps["IMPROVE"],
        )[
            "campaigns_with_rca"
        ]

        scoped_issues = enrich_campaign_actions(issues_with_rca, dimensions=dimensions)
        for row in scoped_issues:
            row["diagnosis_subsidiary"] = subsidiary
        issues_with_actions.extend(scoped_issues)

        scoped_improvements = enrich_campaign_actions(improvements_with_rca, dimensions=dimensions)
        for row in scoped_improvements:
            row["diagnosis_subsidiary"] = subsidiary
        improvements_with_actions.extend(scoped_improvements)

        trace_by_subsidiary[subsidiary_key] = drill_trace

    issues_with_actions.sort(
        key=lambda row: (str(row.get("diagnosis_subsidiary", "")), _to_float(row.get("Impact_adj")))
    )
    improvements_with_actions.sort(
        key=lambda row: (str(row.get("diagnosis_subsidiary", "")), -_to_float(row.get("Impact_adj")))
    )

    return AnalysisResult(
        subsidiaries=subsidiaries,
        issues_with_actions=issues_with_actions,
        improvements_with_actions=improvements_with_actions,
        trace_by_subsidiary=trace_by_subsidiary,
    )
