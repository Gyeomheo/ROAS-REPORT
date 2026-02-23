"""Root Cause Engine (Step 4): delta-log ROAS decomposition and primary driver."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import polars as pl


class RootCauseEngine:
    """Root-cause decomposition engine for top campaigns."""

    DIMENSIONS: List[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"]
    INPUT_METRICS: List[str] = [
        "Spend_curr",
        "Spend_prev",
        "Revenue_curr",
        "Revenue_prev",
        "Clicks_curr",
        "Clicks_prev",
        "Orders_curr",
        "Orders_prev",
    ]
    SUM_METRICS: List[str] = [
        "Revenue_curr_sum",
        "Revenue_prev_sum",
        "Spend_curr_sum",
        "Spend_prev_sum",
        "Clicks_curr_sum",
        "Clicks_prev_sum",
        "Orders_curr_sum",
        "Orders_prev_sum",
    ]
    RCA_FIELDS: List[str] = [
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
    ]

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        return float(value)

    def _campaign_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(row.get(dim) for dim in self.DIMENSIONS)  # type: ignore[return-value]

    def _dims_sort_key(self, row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(str(row.get(dim, "")) for dim in self.DIMENSIONS)  # type: ignore[return-value]

    @staticmethod
    def _safe_ratio_expr(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        safe_den = pl.when(den > 0).then(den).otherwise(None)
        return num / safe_den

    @staticmethod
    def _safe_log_ratio_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
        safe_curr = pl.when(curr > 0).then(curr).otherwise(None)
        safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
        return (safe_curr / safe_prev).log()

    def _validate_schema(self, df: pl.DataFrame) -> None:
        required = set(self.DIMENSIONS + self.INPUT_METRICS)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _sum_aggregations(self) -> List[pl.Expr]:
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

    def _aggregate_campaign(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.group_by(self.DIMENSIONS).agg(self._sum_aggregations()).sort(self.DIMENSIONS)

    def _to_rca_map(self, rca_df: pl.DataFrame) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
        return {self._campaign_key(row): row for row in rca_df.to_dicts()}

    def build_mode_maps(self, df: pl.DataFrame) -> Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]]:
        """Precompute RCA lookup maps for ISSUE/IMPROVE from one campaign aggregation."""
        self._validate_schema(df)
        agg_df = self._aggregate_campaign(df)
        issue_map = self._to_rca_map(self._compute_rca_table(agg_df, "ISSUE"))
        improve_map = self._to_rca_map(self._compute_rca_table(agg_df, "IMPROVE"))
        return {"ISSUE": issue_map, "IMPROVE": improve_map}

    def _join_campaign_rca(
        self,
        top_campaigns: List[Dict[str, Any]],
        mode_upper: str,
        rca_map: Mapping[Tuple[Any, ...], Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        campaigns_with_rca: List[Dict[str, Any]] = []
        for campaign in top_campaigns:
            item = dict(campaign)
            key = self._campaign_key(campaign)
            rca = rca_map.get(key)

            if rca is None:
                item["primary_driver"] = "NONE"
                item["tag"] = "UNDEFINED"
                item["dlog_CVR"] = None
                item["dlog_AOV"] = None
                item["dlog_CPC"] = None
                item["total_dlog_roas"] = None
                item["CPC_curr"] = None
                item["CPC_prev"] = None
                item["CVR_curr"] = None
                item["CVR_prev"] = None
                item["AOV_curr"] = None
                item["AOV_prev"] = None
                item["ROAS_curr"] = None
                item["ROAS_prev"] = None
            else:
                for field in self.SUM_METRICS + self.RCA_FIELDS:
                    item[field] = rca.get(field)

            campaigns_with_rca.append(item)

        if mode_upper == "ISSUE":
            campaigns_with_rca.sort(
                key=lambda row: (self._to_float(row.get("Impact_adj")),) + self._dims_sort_key(row)
            )
        else:
            campaigns_with_rca.sort(
                key=lambda row: (-self._to_float(row.get("Impact_adj")),) + self._dims_sort_key(row)
            )

        return {"campaigns_with_rca": campaigns_with_rca}

    def _compute_rca_table(self, agg_df: pl.DataFrame, mode: str) -> pl.DataFrame:
        base = (
            agg_df.with_columns(
                [
                    self._safe_ratio_expr(pl.col("Spend_curr_sum"), pl.col("Clicks_curr_sum")).alias("CPC_curr"),
                    self._safe_ratio_expr(pl.col("Spend_prev_sum"), pl.col("Clicks_prev_sum")).alias("CPC_prev"),
                    self._safe_ratio_expr(pl.col("Orders_curr_sum"), pl.col("Clicks_curr_sum")).alias("CVR_curr"),
                    self._safe_ratio_expr(pl.col("Orders_prev_sum"), pl.col("Clicks_prev_sum")).alias("CVR_prev"),
                    self._safe_ratio_expr(pl.col("Revenue_curr_sum"), pl.col("Orders_curr_sum")).alias("AOV_curr"),
                    self._safe_ratio_expr(pl.col("Revenue_prev_sum"), pl.col("Orders_prev_sum")).alias("AOV_prev"),
                    self._safe_ratio_expr(pl.col("Revenue_curr_sum"), pl.col("Spend_curr_sum")).alias("ROAS_curr"),
                    self._safe_ratio_expr(pl.col("Revenue_prev_sum"), pl.col("Spend_prev_sum")).alias("ROAS_prev"),
                ]
            )
            .with_columns(
                [
                    self._safe_log_ratio_expr(pl.col("CVR_curr"), pl.col("CVR_prev")).alias("dlog_CVR"),
                    self._safe_log_ratio_expr(pl.col("AOV_curr"), pl.col("AOV_prev")).alias("dlog_AOV"),
                    self._safe_log_ratio_expr(pl.col("CPC_curr"), pl.col("CPC_prev")).alias("dlog_CPC"),
                ]
            )
            .with_columns(
                pl.when(
                    pl.any_horizontal(
                        [
                            pl.col("dlog_CVR").is_null(),
                            pl.col("dlog_AOV").is_null(),
                            pl.col("dlog_CPC").is_null(),
                        ]
                    )
                )
                .then(None)
                .otherwise(pl.col("dlog_CVR") + pl.col("dlog_AOV") - pl.col("dlog_CPC"))
                .alias("total_dlog_roas")
            )
            .with_columns(
                pl.when((pl.col("Clicks_curr_sum") < 100) | (pl.col("Orders_curr_sum") < 5))
                .then(pl.lit("LOW_VOL"))
                .when((pl.col("Revenue_prev_sum") == 0) & (pl.col("Revenue_curr_sum") > 0))
                .then(pl.lit("NEW"))
                .when((pl.col("Revenue_prev_sum") > 0) & (pl.col("Revenue_curr_sum") == 0))
                .then(pl.lit("GONE"))
                .when(pl.col("total_dlog_roas").is_null())
                .then(pl.lit("UNDEFINED"))
                .otherwise(pl.lit("NORMAL"))
                .alias("tag")
            )
        )

        if mode == "ISSUE":
            valid_total = pl.col("total_dlog_roas") < 0
            score_cvr = pl.when(valid_total & (pl.col("dlog_CVR") < 0)).then(pl.col("dlog_CVR").abs()).otherwise(0.0)
            score_aov = pl.when(valid_total & (pl.col("dlog_AOV") < 0)).then(pl.col("dlog_AOV").abs()).otherwise(0.0)
            score_cpc = pl.when(valid_total & (pl.col("dlog_CPC") > 0)).then(pl.col("dlog_CPC").abs()).otherwise(0.0)
        else:
            valid_total = pl.col("total_dlog_roas") > 0
            score_cvr = pl.when(valid_total & (pl.col("dlog_CVR") > 0)).then(pl.col("dlog_CVR").abs()).otherwise(0.0)
            score_aov = pl.when(valid_total & (pl.col("dlog_AOV") > 0)).then(pl.col("dlog_AOV").abs()).otherwise(0.0)
            score_cpc = pl.when(valid_total & (pl.col("dlog_CPC") < 0)).then(pl.col("dlog_CPC").abs()).otherwise(0.0)

        base = base.with_columns(
            [
                score_cvr.alias("score_CVR"),
                score_aov.alias("score_AOV"),
                score_cpc.alias("score_CPC"),
            ]
        )

        driver_expr = (
            pl.when(pl.col("tag").is_in(["LOW_VOL", "NEW", "GONE", "UNDEFINED"]) | (~valid_total))
            .then(pl.lit("NONE"))
            .when(
                (pl.col("score_CVR") >= pl.col("score_AOV"))
                & (pl.col("score_CVR") >= pl.col("score_CPC"))
                & (pl.col("score_CVR") > 0)
            )
            .then(pl.lit("CVR"))
            .when((pl.col("score_AOV") >= pl.col("score_CPC")) & (pl.col("score_AOV") > 0))
            .then(pl.lit("AOV"))
            .when(pl.col("score_CPC") > 0)
            .then(pl.lit("CPC"))
            .otherwise(pl.lit("NONE"))
        )

        return (
            base.with_columns(driver_expr.alias("primary_driver"))
            .select(self.DIMENSIONS + self.SUM_METRICS + self.RCA_FIELDS)
            .sort(self.DIMENSIONS)
        )

    def run(
        self,
        df: pl.DataFrame,
        top_campaigns: List[Dict[str, Any]],
        mode: str,
        rca_map: Mapping[Tuple[Any, ...], Dict[str, Any]] | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run RCA for selected campaigns."""
        mode_upper = mode.upper()
        if mode_upper not in {"ISSUE", "IMPROVE"}:
            raise ValueError("mode must be 'ISSUE' or 'IMPROVE'")
        if not top_campaigns:
            return {"campaigns_with_rca": []}

        if rca_map is None:
            mode_maps = self.build_mode_maps(df)
            mode_map = mode_maps[mode_upper]
        else:
            mode_map = rca_map

        return self._join_campaign_rca(top_campaigns, mode_upper, mode_map)
