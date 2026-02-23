"""Impact Engine (Step 3): LMDI + fallback + residual + drill-down + override."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import polars as pl


class ImpactEngine:
    """Impact decomposition and drill-down engine."""

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
    LEVELS: List[List[str]] = [
        ["SUBSIDIARY"],
        ["SUBSIDIARY", "CHANNEL"],
        ["SUBSIDIARY", "CHANNEL", "DIVISION"],
        ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"],
    ]
    RESULT_COLUMNS: List[str] = (
        DIMENSIONS
        + SUM_METRICS
        + [
            "Impact_raw",
            "Impact_adj",
            "Share_childsum",
            "Share_parentdelta",
            "Residual_parent",
            "Residual_applied_flag",
        ]
    )

    def __init__(
        self,
        rel_floor: float = 0.01,
        abs_floor_amount: float = 0.0,
        phantom_drop_threshold: float = 0.30,
    ) -> None:
        self.rel_floor = rel_floor
        self.abs_floor_amount = abs_floor_amount
        self.phantom_drop_threshold = phantom_drop_threshold
        self._level_cache: Dict[int, pl.DataFrame] = {}
        self._root_totals: Dict[str, Any] = {}

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        return float(value)

    @staticmethod
    def _safe_ratio_expr(num: pl.Expr, den: pl.Expr) -> pl.Expr:
        safe_den = pl.when(den > 0).then(den).otherwise(None)
        return num / safe_den

    @staticmethod
    def _safe_pct_change_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
        safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
        return (curr - safe_prev) / safe_prev

    @staticmethod
    def _safe_log_ratio_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
        safe_curr = pl.when(curr > 0).then(curr).otherwise(None)
        safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
        return (safe_curr / safe_prev).log()

    @staticmethod
    def _log_mean_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
        safe_curr = pl.when(curr > 0).then(curr).otherwise(None)
        safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
        safe_den = pl.when(safe_curr != safe_prev).then(safe_curr.log() - safe_prev.log()).otherwise(None)
        return (
            pl.when(safe_curr.is_null() | safe_prev.is_null())
            .then(None)
            .when(safe_curr == safe_prev)
            .then(safe_curr)
            .otherwise((safe_curr - safe_prev) / safe_den)
        )

    @staticmethod
    def _path_key_expr(dims: List[str]) -> pl.Expr:
        return pl.concat_str([pl.col(dim).cast(pl.Utf8) for dim in dims], separator="|")

    def _campaign_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(row.get(dim) for dim in self.DIMENSIONS)  # type: ignore[return-value]

    def _dims_sort_key(self, row: Dict[str, Any]) -> Tuple[str, ...]:
        return tuple(str(row.get(dim, "")) for dim in self.DIMENSIONS)  # type: ignore[return-value]

    def _validate_schema(self, df: pl.DataFrame) -> None:
        required = set(self.DIMENSIONS + self.INPUT_METRICS)
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _normalize_focus_path(self, focus_path: Dict[str, Any] | None) -> Dict[str, Any]:
        if focus_path is None:
            return {}
        unknown = [key for key in focus_path if key not in self.DIMENSIONS]
        if unknown:
            raise ValueError(f"Invalid focus_path keys: {unknown}")

        normalized = {dim: focus_path[dim] for dim in self.DIMENSIONS if dim in focus_path}
        expected_prefix = self.DIMENSIONS[: len(normalized)]
        if list(normalized.keys()) != expected_prefix:
            raise ValueError(
                f"focus_path must follow hierarchy prefix {expected_prefix}, got {list(normalized.keys())}"
            )
        return normalized

    def _apply_focus(self, df: pl.DataFrame, focus_path: Dict[str, Any]) -> pl.DataFrame:
        focused = df
        for dim, value in focus_path.items():
            focused = focused.filter(pl.col(dim) == pl.lit(value))
        return focused

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

    def _materialize_level_cache(self, df: pl.DataFrame) -> None:
        self._level_cache = {}
        for idx, dims in enumerate(self.LEVELS, start=1):
            self._level_cache[idx] = df.group_by(dims).agg(self._sum_aggregations()).sort(dims)
        self._root_totals = df.select(self._sum_aggregations()).to_dicts()[0]

    def _get_parent_totals(self, parent_depth: int, parent_ctx: Dict[str, Any]) -> Dict[str, Any]:
        if parent_depth == 0:
            return self._root_totals

        parent_dims = self.LEVELS[parent_depth - 1]
        parent_df = self._level_cache[parent_depth]
        for dim in parent_dims:
            parent_df = parent_df.filter(pl.col(dim) == pl.lit(parent_ctx[dim]))

        if parent_df.is_empty():
            return {name: 0.0 for name in self.SUM_METRICS}
        return parent_df.select(self.SUM_METRICS).to_dicts()[0]

    def _compute_child_impacts(
        self, parent_depth: int, parent_ctx: Dict[str, Any]
    ) -> Tuple[pl.DataFrame, Dict[str, float]]:
        child_level = parent_depth + 1
        child_dims = self.LEVELS[child_level - 1]
        child_df = self._level_cache[child_level]

        for dim in self.DIMENSIONS[:parent_depth]:
            child_df = child_df.filter(pl.col(dim) == pl.lit(parent_ctx[dim]))

        if child_df.is_empty():
            return child_df, {"parent_delta": 0.0, "parent_prev_revenue": 0.0}

        parent_totals = self._get_parent_totals(parent_depth, parent_ctx)
        parent_curr = self._to_float(parent_totals.get("Revenue_curr_sum"))
        parent_prev = self._to_float(parent_totals.get("Revenue_prev_sum"))
        parent_delta = parent_curr - parent_prev

        log_mean_child = self._log_mean_expr(pl.col("Revenue_curr_sum"), pl.col("Revenue_prev_sum"))
        log_mean_parent = self._log_mean_expr(pl.lit(parent_curr), pl.lit(parent_prev))
        log_ratio_parent = self._safe_log_ratio_expr(pl.lit(parent_curr), pl.lit(parent_prev))

        valid_lmdi = (
            (pl.col("Revenue_curr_sum") > 0)
            & (pl.col("Revenue_prev_sum") > 0)
            & (pl.lit(parent_curr) > 0)
            & (pl.lit(parent_prev) > 0)
            & log_mean_child.is_not_null()
            & log_mean_parent.is_not_null()
            & log_ratio_parent.is_not_null()
        )

        child_imp = (
            child_df.with_columns(
                [
                    log_mean_child.alias("__log_mean_child"),
                    log_mean_parent.alias("__log_mean_parent"),
                    log_ratio_parent.alias("__parent_log_ratio"),
                    self._path_key_expr(child_dims).alias("__path_key"),
                ]
            )
            .with_columns(valid_lmdi.alias("__valid_lmdi"))
            .with_columns(
                pl.when(pl.col("__valid_lmdi"))
                .then(
                    (pl.col("__log_mean_child") / pl.col("__log_mean_parent"))
                    * pl.col("__parent_log_ratio")
                    * pl.lit(parent_delta)
                )
                .otherwise(pl.col("Revenue_curr_sum") - pl.col("Revenue_prev_sum"))
                .alias("Impact_raw")
            )
        )

        impact_sum_raw = child_imp.select(pl.col("Impact_raw").sum()).to_series(0)[0]
        impact_sum = self._to_float(impact_sum_raw)
        residual = parent_delta - impact_sum

        child_count = child_imp.height
        abs_impact_sum_raw = child_imp.select(pl.col("Impact_raw").abs().sum()).to_series(0)[0]
        abs_impact_sum = self._to_float(abs_impact_sum_raw)
        if child_count <= 0:
            residual_alloc_expr = pl.lit(0.0)
        elif abs_impact_sum > 0:
            residual_alloc_expr = (pl.col("Impact_raw").abs() / pl.lit(abs_impact_sum)) * pl.lit(residual)
        else:
            residual_alloc_expr = pl.lit(residual / child_count)

        child_adj = child_imp.with_columns(
            [
                pl.lit(residual).alias("Residual_parent"),
                residual_alloc_expr.alias("__residual_allocated"),
            ]
        ).with_columns(
            [
                (pl.col("__residual_allocated").abs() > pl.lit(1e-12)).alias("Residual_applied_flag"),
                (pl.col("Impact_raw") + pl.col("__residual_allocated")).alias("Impact_adj"),
            ]
        ).drop("__residual_allocated")

        abs_sum_value = child_adj.select(pl.col("Impact_adj").abs().sum()).to_series(0)[0]
        abs_sum = self._to_float(abs_sum_value)
        abs_parent_delta = abs(parent_delta)

        child_adj = (
            child_adj.with_columns(
                [
                    pl.when(pl.lit(abs_sum) > 0)
                    .then(pl.col("Impact_adj").abs() / pl.lit(abs_sum))
                    .otherwise(0.0)
                    .alias("Share_childsum"),
                    pl.when(pl.lit(abs_parent_delta) > 0)
                    .then(pl.col("Impact_adj").abs() / pl.lit(abs_parent_delta))
                    .otherwise(0.0)
                    .alias("Share_parentdelta"),
                ]
            )
        )

        missing_dims = [dim for dim in self.DIMENSIONS if dim not in child_dims]
        if missing_dims:
            child_adj = child_adj.with_columns([pl.lit(None).alias(dim) for dim in missing_dims])

        child_adj = child_adj.select(self.RESULT_COLUMNS + ["__path_key"]).sort(child_dims)

        return child_adj, {"parent_delta": parent_delta, "parent_prev_revenue": parent_prev}

    def _select_children(
        self,
        child_df: pl.DataFrame,
        child_dims: List[str],
        parent_prev_revenue: float,
        parent_delta: float,
    ) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        if child_df.is_empty():
            return child_df, {"rule": "NO_CHILDREN", "top1_share": 0.0, "top2_share": 0.0, "impact_floor": 0.0}

        share_sorted = child_df.sort(
            ["Share_childsum", *child_dims],
            descending=[True] + [False] * len(child_dims),
        )

        top_rows = share_sorted.head(2).to_dicts()
        top1_share = self._to_float(top_rows[0].get("Share_childsum"))
        top2_share = self._to_float(top_rows[1].get("Share_childsum")) if len(top_rows) > 1 else 0.0
        top1_impact_abs = abs(self._to_float(top_rows[0].get("Impact_adj")))
        impact_floor = max(self.abs_floor_amount, max(parent_prev_revenue, 0.0) * self.rel_floor)

        single_drill = (
            top1_share >= 0.45
            and (top1_share - top2_share) >= 0.15
            and top1_impact_abs >= impact_floor
        )

        if single_drill:
            selected = share_sorted.head(1)
            rule = "SINGLE"
        else:
            if parent_delta < 0:
                selected = (
                    child_df.filter(pl.col("Impact_adj") < 0)
                    .sort(["Impact_adj", *child_dims], descending=[False] + [False] * len(child_dims))
                    .head(3)
                )
                rule = "WIDE_NEG"
            elif parent_delta > 0:
                selected = (
                    child_df.filter(pl.col("Impact_adj") > 0)
                    .sort(["Impact_adj", *child_dims], descending=[True] + [False] * len(child_dims))
                    .head(3)
                )
                rule = "WIDE_POS"
            else:
                selected = child_df.head(0)
                rule = "STOP_ZERO_DELTA"

        return selected, {
            "rule": rule,
            "top1_share": top1_share,
            "top2_share": top2_share,
            "impact_floor": impact_floor,
        }

    def _drill(
        self, start_depth: int, initial_ctx: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        contexts: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = [(initial_ctx, [])]
        campaign_rows: List[Dict[str, Any]] = []
        trace_events: List[Dict[str, Any]] = []

        for parent_depth in range(start_depth, len(self.LEVELS)):
            child_dims = self.LEVELS[parent_depth]
            next_contexts: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []

            for parent_ctx, branch_trace in contexts:
                child_imp, parent_meta = self._compute_child_impacts(parent_depth, parent_ctx)
                selected, selection_meta = self._select_children(
                    child_imp,
                    child_dims,
                    parent_prev_revenue=parent_meta["parent_prev_revenue"],
                    parent_delta=parent_meta["parent_delta"],
                )

                selected_rows = selected.to_dicts()
                selected_paths = [{dim: row[dim] for dim in child_dims} for row in selected_rows]

                trace_events.append(
                    {
                        "level": f"LEVEL{parent_depth + 1}",
                        "parent_path": {dim: parent_ctx.get(dim) for dim in self.DIMENSIONS[:parent_depth]},
                        "rule": selection_meta["rule"],
                        "parent_delta": parent_meta["parent_delta"],
                        "top1_share_childsum": selection_meta["top1_share"],
                        "top2_share_childsum": selection_meta["top2_share"],
                        "impact_floor": selection_meta["impact_floor"],
                        "selected_paths": selected_paths,
                    }
                )

                if not selected_rows:
                    continue

                for row in selected_rows:
                    event = {
                        "level": f"LEVEL{parent_depth + 1}",
                        "rule": selection_meta["rule"],
                        "selected_path": {dim: row[dim] for dim in child_dims},
                        "impact_adj": row["Impact_adj"],
                        "share_childsum": row["Share_childsum"],
                    }
                    new_trace = branch_trace + [event]

                    if parent_depth == len(self.LEVELS) - 1:
                        campaign = {col: row.get(col) for col in self.RESULT_COLUMNS}
                        campaign["drill_trace"] = new_trace
                        campaign_rows.append(campaign)
                    else:
                        next_ctx = {dim: row[dim] for dim in child_dims}
                        next_contexts.append((next_ctx, new_trace))

            contexts = next_contexts
            if not contexts and parent_depth < len(self.LEVELS) - 1:
                break

        return campaign_rows, trace_events

    def _terminal_focus_campaign(self, focus_path: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if len(focus_path) < len(self.DIMENSIONS):
            return [], []

        parent_depth = len(self.DIMENSIONS) - 1
        parent_ctx = {dim: focus_path[dim] for dim in self.DIMENSIONS[:parent_depth]}
        child_imp, parent_meta = self._compute_child_impacts(parent_depth, parent_ctx)
        if child_imp.is_empty():
            return [], []

        terminal_dim = self.DIMENSIONS[-1]
        one_campaign_df = child_imp.filter(pl.col(terminal_dim) == pl.lit(focus_path[terminal_dim])).head(1)
        if one_campaign_df.is_empty():
            return [], []

        level_label = f"LEVEL{len(self.DIMENSIONS)}"
        row = one_campaign_df.to_dicts()[0]
        campaign = {col: row.get(col) for col in self.RESULT_COLUMNS}
        campaign["drill_trace"] = [
            {
                "level": level_label,
                "rule": "FOCUS_TERMINAL",
                "selected_path": {dim: focus_path[dim] for dim in self.DIMENSIONS},
                "impact_adj": row.get("Impact_adj"),
                "share_childsum": row.get("Share_childsum"),
            }
        ]

        trace = [
            {
                "level": level_label,
                "parent_path": parent_ctx,
                "rule": "FOCUS_TERMINAL",
                "parent_delta": parent_meta["parent_delta"],
                "selected_paths": [{dim: focus_path[dim] for dim in self.DIMENSIONS}],
            }
        ]
        return [campaign], trace

    def _dedupe_campaigns(self, campaign_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

        for row in campaign_rows:
            key = self._campaign_key(row)
            if key not in merged:
                merged[key] = dict(row)
                merged[key]["drill_trace"] = [row.get("drill_trace", [])]
                continue

            current = merged[key]
            for col in self.SUM_METRICS + ["Impact_raw", "Impact_adj", "Residual_parent"]:
                current[col] = self._to_float(current.get(col)) + self._to_float(row.get(col))
            current["Residual_applied_flag"] = bool(current.get("Residual_applied_flag", False)) or bool(
                row.get("Residual_applied_flag", False)
            )
            current["drill_trace"].append(row.get("drill_trace", []))

        results: List[Dict[str, Any]] = []
        for key in sorted(merged.keys(), key=lambda item: tuple(str(x) for x in item)):
            item = merged[key]
            traces = item.get("drill_trace", [])
            item["drill_trace"] = traces[0] if len(traces) == 1 else traces
            results.append(item)
        return results

    def _attach_candidate_metrics(self, candidate_df: pl.DataFrame) -> pl.DataFrame:
        if candidate_df.is_empty():
            return candidate_df
        return (
            candidate_df.with_columns(
                [
                    self._safe_pct_change_expr(pl.col("Revenue_curr_sum"), pl.col("Revenue_prev_sum")).alias("rev_yoy_pct"),
                    self._safe_pct_change_expr(pl.col("Spend_curr_sum"), pl.col("Spend_prev_sum")).alias("spend_yoy_pct"),
                    self._safe_ratio_expr(pl.col("Revenue_curr_sum"), pl.col("Spend_curr_sum")).alias("roas_curr"),
                    self._safe_ratio_expr(pl.col("Revenue_prev_sum"), pl.col("Spend_prev_sum")).alias("roas_prev"),
                ]
            )
            .with_columns(
                [
                    self._safe_pct_change_expr(pl.col("roas_curr"), pl.col("roas_prev")).alias("roas_yoy_pct"),
                    (pl.col("rev_yoy_pct") - pl.col("spend_yoy_pct")).alias("defense_score_pct"),
                ]
            )
        )

    def _scope_eligibility_map(self, df: pl.DataFrame, focus_path: Dict[str, Any]) -> Dict[Any, bool]:
        if df.is_empty():
            return {}

        if focus_path:
            scope_df = df.select(self._sum_aggregations()).with_columns(pl.lit("FOCUS").alias("__scope_key"))
            key_col = "__scope_key"
        else:
            scope_dim = self.DIMENSIONS[0]
            scope_df = df.group_by([scope_dim]).agg(self._sum_aggregations())
            key_col = scope_dim

        scope_df = (
            scope_df.with_columns(
                [
                    self._safe_pct_change_expr(pl.col("Revenue_curr_sum"), pl.col("Revenue_prev_sum")).alias("rev_yoy_pct"),
                    self._safe_pct_change_expr(pl.col("Spend_curr_sum"), pl.col("Spend_prev_sum")).alias("spend_yoy_pct"),
                    self._safe_ratio_expr(pl.col("Revenue_curr_sum"), pl.col("Spend_curr_sum")).alias("roas_curr"),
                    self._safe_ratio_expr(pl.col("Revenue_prev_sum"), pl.col("Spend_prev_sum")).alias("roas_prev"),
                ]
            )
            .with_columns(
                [
                    self._safe_pct_change_expr(pl.col("roas_curr"), pl.col("roas_prev")).alias("roas_yoy_pct"),
                    (pl.col("rev_yoy_pct") - pl.col("spend_yoy_pct")).alias("defense_score_pct"),
                ]
            )
            .with_columns(
                (
                    (pl.col("defense_score_pct") > 0)
                    & (pl.col("roas_yoy_pct") > 0)
                    & (pl.col("rev_yoy_pct") > pl.lit(-self.phantom_drop_threshold))
                )
                .fill_null(False)
                .alias("scope_defense_eligible")
            )
            .select([key_col, "scope_defense_eligible"])
        )

        return {row[key_col]: bool(row["scope_defense_eligible"]) for row in scope_df.to_dicts()}

    def _select_issues(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        issues = [row for row in candidates if self._to_float(row.get("Impact_adj")) < 0]
        issues.sort(key=lambda row: (self._to_float(row.get("Impact_adj")),) + self._dims_sort_key(row))
        out: List[Dict[str, Any]] = []
        for row in issues[:3]:
            item = dict(row)
            item["mode_tag"] = "ISSUE"
            out.append(item)
        return out

    def _select_improvements(
        self,
        candidates: List[Dict[str, Any]],
        scope_map: Dict[Any, bool],
        focus_path: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        growth = [row for row in candidates if self._to_float(row.get("Impact_adj")) > 0]
        growth.sort(key=lambda row: (-self._to_float(row.get("Impact_adj")),) + self._dims_sort_key(row))
        selected_growth: List[Dict[str, Any]] = []
        for row in growth[:3]:
            item = dict(row)
            item["mode_tag"] = "IMPROVE"
            selected_growth.append(item)

        remain = 3 - len(selected_growth)
        if remain <= 0:
            return selected_growth

        defense: List[Dict[str, Any]] = []
        for row in candidates:
            if self._to_float(row.get("Impact_adj")) > 0:
                continue

            scope_key = "FOCUS" if focus_path else row.get(self.DIMENSIONS[0])
            scope_ok = bool(scope_map.get(scope_key, False))
            rev_yoy = row.get("rev_yoy_pct")
            roas_yoy = row.get("roas_yoy_pct")
            defense_score = row.get("defense_score_pct")

            qualifies = (
                scope_ok
                and rev_yoy is not None
                and roas_yoy is not None
                and defense_score is not None
                and float(defense_score) > 0
                and float(roas_yoy) > 0
                and float(rev_yoy) > -self.phantom_drop_threshold
            )
            if qualifies:
                item = dict(row)
                item["mode_tag"] = "DEFENSE"
                defense.append(item)

        defense.sort(
            key=lambda row: (
                -self._to_float(row.get("defense_score_pct"), default=-1e9),
                -self._to_float(row.get("Impact_adj")),
            )
            + self._dims_sort_key(row)
        )

        return selected_growth + defense[:remain]

    def run(
        self,
        df: pl.DataFrame,
        focus_path: Dict[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Run impact analysis and return issues, improvements, and drill trace."""
        self._validate_schema(df)
        normalized_focus = self._normalize_focus_path(focus_path)
        working_df = self._apply_focus(df, normalized_focus)

        if working_df.is_empty():
            return [], [], {"focus_path": normalized_focus, "events": []}

        self._materialize_level_cache(working_df)

        start_depth = len(normalized_focus)
        if start_depth >= len(self.LEVELS):
            campaigns, trace_events = self._terminal_focus_campaign(normalized_focus)
        else:
            campaigns, trace_events = self._drill(start_depth, normalized_focus)

        deduped = self._dedupe_campaigns(campaigns)
        if not deduped:
            return [], [], {"focus_path": normalized_focus, "events": trace_events}

        candidate_df = pl.DataFrame([{k: v for k, v in row.items() if k != "drill_trace"} for row in deduped])
        candidate_df = self._attach_candidate_metrics(candidate_df)
        metric_map = {self._campaign_key(row): row for row in candidate_df.to_dicts()}

        enriched_candidates: List[Dict[str, Any]] = []
        for row in deduped:
            key = self._campaign_key(row)
            metrics = metric_map.get(key, {})
            item = dict(row)
            item["rev_yoy_pct"] = metrics.get("rev_yoy_pct")
            item["spend_yoy_pct"] = metrics.get("spend_yoy_pct")
            item["roas_curr"] = metrics.get("roas_curr")
            item["roas_prev"] = metrics.get("roas_prev")
            item["roas_yoy_pct"] = metrics.get("roas_yoy_pct")
            item["defense_score_pct"] = metrics.get("defense_score_pct")
            enriched_candidates.append(item)

        scope_map = self._scope_eligibility_map(working_df, normalized_focus)
        issues = self._select_issues(enriched_candidates)
        improvements = self._select_improvements(enriched_candidates, scope_map, normalized_focus)

        return issues, improvements, {"focus_path": normalized_focus, "events": trace_events}
