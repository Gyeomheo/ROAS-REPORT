"""LMDI and aggregation helpers (So what layer)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import polars as pl

from src.application.reporting.metrics import (
    fmt_pct,
    fmt_roas,
    log_mean_expr,
    safe_log_ratio_expr,
    safe_pct_change,
    safe_pct_change_expr,
    safe_ratio,
    safe_ratio_expr,
    to_float,
)


CHANNEL_DIMS: List[str] = ["SUBSIDIARY", "CHANNEL"]
DIVISION_DIMS: List[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION"]


def sum_aggregations() -> list[pl.Expr]:
    return [
        pl.col("Revenue_curr").sum().alias("Revenue_curr_sum"),
        pl.col("Revenue_prev").sum().alias("Revenue_prev_sum"),
        pl.col("Spend_curr").sum().alias("Spend_curr_sum"),
        pl.col("Spend_prev").sum().alias("Spend_prev_sum"),
    ]


def sum_aggregations_full() -> list[pl.Expr]:
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


def _to_key(row: Dict[str, Any], dims: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(dim) for dim in dims)


def scope_maps(
    df: pl.DataFrame,
    channel_dims: Sequence[str] = CHANNEL_DIMS,
    division_dims: Sequence[str] = DIVISION_DIMS,
) -> tuple[Dict[tuple[Any, ...], Dict[str, Any]], Dict[tuple[Any, ...], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    subsidiary_rows = df.group_by(["SUBSIDIARY"]).agg(sum_aggregations_full()).to_dicts()
    channel_rows = df.group_by(list(channel_dims)).agg(sum_aggregations_full()).to_dicts()
    division_rows = df.group_by(list(division_dims)).agg(sum_aggregations_full()).to_dicts()
    subsidiary_map = {str(row.get("SUBSIDIARY", "")): row for row in subsidiary_rows}
    channel_map = {_to_key(row, channel_dims): row for row in channel_rows}
    division_map = {_to_key(row, division_dims): row for row in division_rows}
    return channel_map, division_map, subsidiary_map


def division_parallel_contribution(df: pl.DataFrame) -> List[Dict[str, Any]]:
    parent_df = (
        df.group_by(["SUBSIDIARY"])
        .agg(sum_aggregations_full())
        .with_columns(
            [
                pl.col("Revenue_curr_sum").alias("Parent_revenue_curr_sum"),
                pl.col("Revenue_prev_sum").alias("Parent_revenue_prev_sum"),
            ]
        )
        .with_columns(
            [
                (pl.col("Parent_revenue_curr_sum") - pl.col("Parent_revenue_prev_sum")).alias("Parent_revenue_delta"),
                safe_pct_change_expr(pl.col("Parent_revenue_curr_sum"), pl.col("Parent_revenue_prev_sum")).alias(
                    "Parent_revenue_yoy_pct"
                ),
            ]
        )
        .sort(["SUBSIDIARY"])
    )
    parent_rows = parent_df.select(
        [
            "SUBSIDIARY",
            "Parent_revenue_curr_sum",
            "Parent_revenue_prev_sum",
            "Parent_revenue_delta",
            "Parent_revenue_yoy_pct",
        ]
    ).to_dicts()

    division_df = (
        df.group_by(["SUBSIDIARY", "DIVISION"])
        .agg(sum_aggregations_full())
        .sort(["SUBSIDIARY", "DIVISION"])
    )
    if division_df.is_empty():
        return [
            {
                "SUBSIDIARY": str(row.get("SUBSIDIARY", "")),
                "Parent_revenue_curr_sum": to_float(row.get("Parent_revenue_curr_sum")),
                "Parent_revenue_prev_sum": to_float(row.get("Parent_revenue_prev_sum")),
                "Parent_revenue_delta": to_float(row.get("Parent_revenue_delta")),
                "Parent_revenue_yoy_pct": row.get("Parent_revenue_yoy_pct"),
                "rows": [],
            }
            for row in parent_rows
        ]

    parent_join_df = parent_df.select(
        [
            "SUBSIDIARY",
            "Parent_revenue_curr_sum",
            "Parent_revenue_prev_sum",
            "Parent_revenue_delta",
        ]
    )
    joined = division_df.join(parent_join_df, on="SUBSIDIARY", how="left")
    child_log_mean = log_mean_expr(pl.col("Revenue_curr_sum"), pl.col("Revenue_prev_sum"))
    parent_log_mean = log_mean_expr(pl.col("Parent_revenue_curr_sum"), pl.col("Parent_revenue_prev_sum"))
    parent_log_ratio = safe_log_ratio_expr(pl.col("Parent_revenue_curr_sum"), pl.col("Parent_revenue_prev_sum"))
    valid_lmdi = (
        (pl.col("Revenue_curr_sum") > 0)
        & (pl.col("Revenue_prev_sum") > 0)
        & (pl.col("Parent_revenue_curr_sum") > 0)
        & (pl.col("Parent_revenue_prev_sum") > 0)
        & child_log_mean.is_not_null()
        & parent_log_mean.is_not_null()
        & parent_log_ratio.is_not_null()
    )
    joined = joined.with_columns(
        [
            pl.when(valid_lmdi)
            .then((child_log_mean / parent_log_mean) * parent_log_ratio * pl.col("Parent_revenue_delta"))
            .otherwise(pl.col("Revenue_curr_sum") - pl.col("Revenue_prev_sum"))
            .alias("Impact_raw"),
            pl.when(valid_lmdi).then(pl.lit("LMDI")).otherwise(pl.lit("DELTA_FALLBACK")).alias("Impact_raw_method"),
        ]
    )

    residual_df = (
        joined.group_by(["SUBSIDIARY"])
        .agg(
            [
                pl.col("Impact_raw").sum().alias("Impact_raw_sum"),
                pl.col("Impact_raw").abs().sum().alias("Abs_impact_raw_sum"),
                pl.len().alias("Child_count"),
            ]
        )
        .join(parent_join_df.select(["SUBSIDIARY", "Parent_revenue_delta"]), on="SUBSIDIARY", how="left")
        .with_columns((pl.col("Parent_revenue_delta") - pl.col("Impact_raw_sum")).alias("Residual_parent"))
        .select(["SUBSIDIARY", "Residual_parent", "Abs_impact_raw_sum", "Child_count"])
    )

    contribution_label_expr = (
        pl.when(pl.col("Parent_revenue_delta") < 0)
        .then(
            pl.when(pl.col("Impact_adj") < 0)
            .then(pl.lit("DECLINE_DRIVER"))
            .otherwise(pl.lit("DECLINE_OFFSET"))
        )
        .when(pl.col("Parent_revenue_delta") > 0)
        .then(
            pl.when(pl.col("Impact_adj") > 0)
            .then(pl.lit("GROWTH_DRIVER"))
            .otherwise(pl.lit("GROWTH_OFFSET"))
        )
        .otherwise(pl.lit("NEUTRAL"))
    )

    contribution_df = (
        joined.join(residual_df, on="SUBSIDIARY", how="left")
        .with_columns(
            pl.when(pl.col("Child_count") <= 0)
            .then(0.0)
            .when(pl.col("Abs_impact_raw_sum") > 0)
            .then((pl.col("Impact_raw").abs() / pl.col("Abs_impact_raw_sum")) * pl.col("Residual_parent"))
            .otherwise(pl.col("Residual_parent") / pl.col("Child_count"))
            .alias("__residual_allocated")
        )
        .with_columns(
            [
                (pl.col("Impact_raw") + pl.col("__residual_allocated")).alias("Impact_adj"),
                (pl.col("__residual_allocated").abs() > pl.lit(1e-12)).alias("Residual_applied_flag"),
                (pl.col("Revenue_curr_sum") - pl.col("Revenue_prev_sum")).alias("Revenue_delta"),
                safe_pct_change_expr(pl.col("Revenue_curr_sum"), pl.col("Revenue_prev_sum")).alias("Revenue_yoy_pct"),
                safe_pct_change_expr(pl.col("Spend_curr_sum"), pl.col("Spend_prev_sum")).alias("Spend_yoy_pct"),
                safe_ratio_expr(pl.col("Revenue_curr_sum"), pl.col("Spend_curr_sum")).alias("ROAS_curr"),
                safe_ratio_expr(pl.col("Revenue_prev_sum"), pl.col("Spend_prev_sum")).alias("ROAS_prev"),
            ]
        )
        .with_columns(
            [
                safe_pct_change_expr(pl.col("ROAS_curr"), pl.col("ROAS_prev")).alias("ROAS_yoy_pct"),
                pl.col("Parent_revenue_delta").abs().alias("__parent_abs_delta"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("__parent_abs_delta") > 0)
                .then(pl.col("Impact_adj").abs() / pl.col("__parent_abs_delta"))
                .otherwise(0.0)
                .alias("Contribution_to_total_delta_pct"),
                pl.when((pl.col("Parent_revenue_delta") < 0) & (pl.col("__parent_abs_delta") > 0))
                .then(
                    pl.when(pl.col("Impact_adj") < 0).then(-pl.col("Impact_adj")).otherwise(0.0)
                    / pl.col("__parent_abs_delta")
                )
                .otherwise(0.0)
                .alias("Contribution_to_decline_pct"),
                pl.when((pl.col("Parent_revenue_delta") > 0) & (pl.col("__parent_abs_delta") > 0))
                .then(
                    pl.when(pl.col("Impact_adj") > 0).then(pl.col("Impact_adj")).otherwise(0.0)
                    / pl.col("__parent_abs_delta")
                )
                .otherwise(0.0)
                .alias("Contribution_to_growth_pct"),
                contribution_label_expr.alias("Contribution_label"),
            ]
        )
        .drop("__residual_allocated", "__parent_abs_delta")
        .select(
            [
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
            ]
        )
    )

    rows_by_sub: Dict[str, List[Dict[str, Any]]] = {}
    for row in contribution_df.sort(["SUBSIDIARY", "DIVISION"]).to_dicts():
        subsidiary = str(row.get("SUBSIDIARY", ""))
        parent_delta = to_float(row.get("Parent_revenue_delta"))
        division_name = str(row.get("DIVISION", "DIVISION"))
        rev_yoy = row.get("Revenue_yoy_pct")
        roas_curr = row.get("ROAS_curr")
        roas_yoy = row.get("ROAS_yoy_pct")
        if parent_delta < 0:
            summary_text = (
                f"{division_name} 매출 {fmt_pct(rev_yoy)} / ROAS {fmt_roas(roas_curr)}({fmt_pct(roas_yoy)}) "
                f"기준, 전체 하락 기여도 {fmt_pct(row.get('Contribution_to_decline_pct'), signed=False)}"
            )
        elif parent_delta > 0:
            summary_text = (
                f"{division_name} 매출 {fmt_pct(rev_yoy)} / ROAS {fmt_roas(roas_curr)}({fmt_pct(roas_yoy)}) "
                f"기준, 전체 성장 기여도 {fmt_pct(row.get('Contribution_to_growth_pct'), signed=False)}"
            )
        else:
            summary_text = f"{division_name} 매출 {fmt_pct(rev_yoy)} / ROAS {fmt_roas(roas_curr)}({fmt_pct(roas_yoy)})"
        row["summary_text"] = summary_text
        rows_by_sub.setdefault(subsidiary, []).append(row)

    output: List[Dict[str, Any]] = []
    for parent in parent_rows:
        subsidiary = str(parent.get("SUBSIDIARY", ""))
        parent_delta = to_float(parent.get("Parent_revenue_delta"))
        rows = rows_by_sub.get(subsidiary, [])
        if parent_delta < 0:
            rows.sort(
                key=lambda item: (
                    -to_float(item.get("Contribution_to_decline_pct")),
                    -abs(to_float(item.get("Impact_adj"))),
                    str(item.get("DIVISION", "")),
                )
            )
        elif parent_delta > 0:
            rows.sort(
                key=lambda item: (
                    -to_float(item.get("Contribution_to_growth_pct")),
                    -abs(to_float(item.get("Impact_adj"))),
                    str(item.get("DIVISION", "")),
                )
            )
        else:
            rows.sort(key=lambda item: (-abs(to_float(item.get("Impact_adj"))), str(item.get("DIVISION", ""))))

        output.append(
            {
                "SUBSIDIARY": subsidiary,
                "Parent_revenue_curr_sum": to_float(parent.get("Parent_revenue_curr_sum")),
                "Parent_revenue_prev_sum": to_float(parent.get("Parent_revenue_prev_sum")),
                "Parent_revenue_delta": parent_delta,
                "Parent_revenue_yoy_pct": safe_pct_change(
                    to_float(parent.get("Parent_revenue_curr_sum")),
                    to_float(parent.get("Parent_revenue_prev_sum")),
                ),
                "rows": rows,
            }
        )
    return output

