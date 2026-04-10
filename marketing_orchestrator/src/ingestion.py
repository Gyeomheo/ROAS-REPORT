"""Excel ingestion/output helpers with Polars-first and openpyxl fallback."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, Sequence

import polars as pl


DIMENSIONS: list[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"]
RAW_METRIC_MAP: dict[str, str] = {
    "PLATFORM_SPEND_USD": "Spend",
    "PLATFORM_REVENUE_USD": "Revenue",
    "PLATFORM_CLICKS": "Clicks",
    "GROSS_ORDERS": "Orders",
}
METRICS: list[str] = ["Spend", "Revenue", "Clicks", "Orders"]
OBJECTIVE_CANDIDATES: tuple[str, ...] = ("OBJECTIVE", "Objective", "objective")
TARGET_OBJECTIVE_VALUE = "CONVERSION"
TARGET_DIVISIONS: tuple[str, ...] = ("MX", "VD", "DA")
ENGINE_METRIC_COLUMNS: list[str] = [
    "Spend_curr",
    "Spend_prev",
    "Revenue_curr",
    "Revenue_prev",
    "Clicks_curr",
    "Clicks_prev",
    "Orders_curr",
    "Orders_prev",
]
ENGINE_COLUMNS: list[str] = [*DIMENSIONS, *ENGINE_METRIC_COLUMNS]
LONG_COLUMNS_YEAR_WITH_OBJECTIVE: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "OBJECTIVE", "Year", "Month", "Day"]
LONG_COLUMNS_YEAR_ALT_WITH_OBJECTIVE: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "OBJECTIVE", "YEAR", "Month", "Day"]
LONG_COLUMNS_YEAR_UPPER_WITH_OBJECTIVE: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "OBJECTIVE", "YEAR", "MONTH", "DAY"]
LONG_COLUMNS_YEAR_MIN_WITH_OBJECTIVE: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "OBJECTIVE", "Year"]
LONG_COLUMNS_YEAR_ALT_MIN_WITH_OBJECTIVE: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "OBJECTIVE", "YEAR"]
LONG_COLUMNS_YEAR: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "Year", "Month", "Day"]
LONG_COLUMNS_YEAR_ALT: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "YEAR", "Month", "Day"]
LONG_COLUMNS_YEAR_UPPER: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "YEAR", "MONTH", "DAY"]
LONG_COLUMNS_YEAR_MIN: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "Year"]
LONG_COLUMNS_YEAR_ALT_MIN: list[str] = [*DIMENSIONS, *RAW_METRIC_MAP.keys(), "YEAR"]
PREFERRED_COLUMN_SETS: list[list[str]] = [
    LONG_COLUMNS_YEAR_WITH_OBJECTIVE,
    LONG_COLUMNS_YEAR_ALT_WITH_OBJECTIVE,
    LONG_COLUMNS_YEAR_UPPER_WITH_OBJECTIVE,
    LONG_COLUMNS_YEAR_MIN_WITH_OBJECTIVE,
    LONG_COLUMNS_YEAR_ALT_MIN_WITH_OBJECTIVE,
    LONG_COLUMNS_YEAR,
    LONG_COLUMNS_YEAR_ALT,
    LONG_COLUMNS_YEAR_UPPER,
    LONG_COLUMNS_YEAR_MIN,
    LONG_COLUMNS_YEAR_ALT_MIN,
    ENGINE_COLUMNS,
]
SOURCE_RAW_HEADERS: tuple[str, ...] = (
    "Year",
    "Month",
    "Day",
    "PLATFORM",
    "SUBSIDIARY",
    "REGION",
    "COUNTRY",
    "DIVISION",
    "CAMPAIGN_NAME",
    "CAMPAIGN_ID",
    "PLATFORM_CAMPAIGN_TYPE",
    "CHANNEL",
    "PRODUCT",
    "SOCIAL_CONTENT_TYPE",
    "BID_STRATEGY_TYPE",
    "CAMPAIGN_OBJECTIVE",
    "OBJECTIVE",
    "TAXO_PHASE_PH",
    "PUBLISHER",
    "PLATFORM_LANDING_PAGE_URL",
    "TARGET_TYPE",
    "AI_AD_PRODUCT",
    "MX_FLAGSHIP",
    "MX_FLAGSHIP_S",
    "MX_FLAGSHIP_S_DETAIL",
    "MX_FLAGSHIP_Z",
    "MX_FLAGSHIP_Z_DETAIL",
    "GOOGLE_ADS_NETWORK",
    "GOOGLE_ADS_EXTERNAL_CONVERSION_SOURCE",
    "ACCOUNT",
    "ACCOUNT_ID",
    "PLATFORM_SPEND_USD",
    "PLATFORM_IMPRESSIONS",
    "PLATFORM_CLICKS",
    "VISITS",
    "QUALIFY_VISITS",
    "GROSS_ORDERS",
    "GROSS_REVENUE",
    "PLATFORM_TOTAL_CONVERSIONS",
    "PLATFORM_REVENUE_USD",
    "META_MOBILE_APP_PURCHASE",
    "META_WEB_PURCHASE",
    "PLATFORM_VIDEO_VIEWS",
)
SOURCE_HEADER_MAP: dict[str, str] = {header.upper(): header for header in SOURCE_RAW_HEADERS}
SOURCE_HEADER_SCAN_MAX_ROWS = 120
REQUIRED_LONG_HEADERS_UPPER: set[str] = {
    "YEAR",
    "SUBSIDIARY",
    "CHANNEL",
    "DIVISION",
    "PRODUCT",
    "PLATFORM_SPEND_USD",
    "PLATFORM_CLICKS",
    "GROSS_ORDERS",
    "PLATFORM_REVENUE_USD",
}
REQUIRED_WIDE_HEADERS_UPPER: set[str] = {column.upper() for column in ENGINE_COLUMNS}
DEFAULT_CURR_YEAR = date.today().year
DEFAULT_PREV_YEAR = DEFAULT_CURR_YEAR - 1


def _parse_error_threshold() -> float:
    raw = os.getenv("ROAS_PARSE_ERROR_THRESHOLD", "0.01")
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ROAS_PARSE_ERROR_THRESHOLD: {raw}") from exc
    if threshold < 0 or threshold > 1:
        raise ValueError(f"ROAS_PARSE_ERROR_THRESHOLD must be in [0, 1], got {threshold}")
    return threshold


METRIC_PARSE_ERROR_THRESHOLD = _parse_error_threshold()


def _import_openpyxl() -> tuple[Any, Any]:
    try:
        from openpyxl import Workbook, load_workbook
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openpyxl is required for Excel fallback I/O.") from exc
    return Workbook, load_workbook


def _normalize_headers(raw_headers: Sequence[Any]) -> list[str]:
    headers: list[str] = []
    seen: dict[str, int] = {}
    for idx, value in enumerate(raw_headers):
        base = str(value).strip() if value not in (None, "") else f"column_{idx + 1}"
        count = seen.get(base, 0)
        name = base if count == 0 else f"{base}_{count + 1}"
        seen[base] = count + 1
        headers.append(name)
    return headers


def _canonical_header_name(name: str) -> str:
    cleaned = str(name).strip()
    if cleaned == "":
        return cleaned
    return SOURCE_HEADER_MAP.get(cleaned.upper(), cleaned)


def _standardize_known_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() and not df.columns:
        return df
    rename_map: dict[str, str] = {}
    future_cols = set(df.columns)
    for col in df.columns:
        canonical = _canonical_header_name(col)
        if canonical == col:
            continue
        if canonical in future_cols:
            continue
        rename_map[col] = canonical
        future_cols.add(canonical)
    if not rename_map:
        return df
    return df.rename(rename_map)


def _looks_like_supported_input(columns: Sequence[str]) -> bool:
    upper = {str(column).strip().upper() for column in columns}
    return REQUIRED_WIDE_HEADERS_UPPER.issubset(upper) or REQUIRED_LONG_HEADERS_UPPER.issubset(upper)


def _select_sheet_name(path: Path, preferred_sheet: str) -> str:
    try:
        _, load_workbook = _import_openpyxl()
    except Exception:
        return preferred_sheet

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet_names = list(workbook.sheetnames)
    workbook.close()
    if not sheet_names:
        raise ValueError(f"No sheets found in {path}")
    if preferred_sheet in sheet_names:
        return preferred_sheet
    return sheet_names[0]


def _read_excel_polars(path: Path, **kwargs: Any) -> Any:
    """Use larger schema sampling when supported to avoid dtype inference warnings."""
    try:
        return pl.read_excel(path, infer_schema_length=10000, **kwargs)  # type: ignore[arg-type]
    except TypeError:
        return pl.read_excel(path, **kwargs)  # type: ignore[arg-type]


def _frame_from_polars_result(frame: Any, preferred_sheet: str) -> pl.DataFrame:
    if isinstance(frame, dict):
        if preferred_sheet in frame:
            return frame[preferred_sheet]
        first_key = next(iter(frame.keys()), None)
        if first_key is None:
            return pl.DataFrame()
        return frame[first_key]
    return frame


def _read_with_polars(path: Path, preferred_sheet: str, target_sheet: str) -> pl.DataFrame:
    if not hasattr(pl, "read_excel"):
        raise RuntimeError("polars.read_excel is not available in this environment.")

    tried: list[str] = []
    for sheet_name in [target_sheet, preferred_sheet]:
        if not sheet_name or sheet_name in tried:
            continue
        tried.append(sheet_name)

        try:
            frame = _read_excel_polars(path, sheet_name=sheet_name)
            normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
            if _looks_like_supported_input(normalized.columns):
                return normalized
        except Exception:
            pass

        for columns in PREFERRED_COLUMN_SETS:
            try:
                frame = _read_excel_polars(path, sheet_name=sheet_name, columns=columns)
                normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
                if _looks_like_supported_input(normalized.columns):
                    return normalized
            except Exception:
                continue

    try:
        frame = _read_excel_polars(path, sheet_name=preferred_sheet)
        normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
        if _looks_like_supported_input(normalized.columns):
            return normalized
    except Exception:
        pass

    frame = _read_excel_polars(path)
    normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
    return normalized


def _read_with_openpyxl(path: Path, target_sheet: str, preferred_sheet: str) -> pl.DataFrame:
    _, load_workbook = _import_openpyxl()
    workbook = load_workbook(path, read_only=True, data_only=True)

    candidate_sheets: list[str] = []
    for name in [target_sheet, preferred_sheet, *list(workbook.sheetnames)]:
        if name and name in workbook.sheetnames and name not in candidate_sheets:
            candidate_sheets.append(name)

    def _try_sheet(sheet_name: str) -> pl.DataFrame | None:
        worksheet = workbook[sheet_name]
        header_row_idx: int | None = None
        headers: list[str] = []
        for idx, row_values in enumerate(
            worksheet.iter_rows(values_only=True, max_row=SOURCE_HEADER_SCAN_MAX_ROWS), start=1
        ):
            if row_values is None:
                continue
            normalized_headers = [_canonical_header_name(h) for h in _normalize_headers(row_values)]
            if _looks_like_supported_input(normalized_headers):
                header_row_idx = idx
                headers = normalized_headers
                break
        if header_row_idx is None:
            return None

        records: list[dict[str, Any]] = []
        for values in worksheet.iter_rows(values_only=True, min_row=header_row_idx + 1):
            if values is None or all(value is None for value in values):
                continue
            row_data: dict[str, Any] = {}
            for col_idx, name in enumerate(headers):
                row_data[name] = values[col_idx] if col_idx < len(values) else None
            records.append(row_data)

        if not records:
            return pl.DataFrame({name: [] for name in headers})
        return pl.DataFrame(records)

    selected_df: pl.DataFrame | None = None
    for sheet_name in candidate_sheets:
        frame = _try_sheet(sheet_name)
        if frame is None:
            continue
        selected_df = _standardize_known_columns(frame)
        break

    workbook.close()
    if selected_df is not None:
        return selected_df

    # Legacy fallback: first row as header from target/first sheet.
    fallback_sheet = target_sheet if target_sheet in workbook.sheetnames else (
        preferred_sheet if preferred_sheet in workbook.sheetnames else workbook.sheetnames[0]
    )
    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook[fallback_sheet]
    row_iter = worksheet.iter_rows(values_only=True)
    header_row = next(row_iter, None)
    if header_row is None:
        workbook.close()
        return pl.DataFrame()
    headers = [_canonical_header_name(h) for h in _normalize_headers(header_row)]
    records: list[dict[str, Any]] = []
    for values in row_iter:
        if values is None or all(value is None for value in values):
            continue
        row_data: dict[str, Any] = {}
        for idx, name in enumerate(headers):
            row_data[name] = values[idx] if idx < len(values) else None
        records.append(row_data)
    workbook.close()
    if not records:
        return pl.DataFrame({name: [] for name in headers})
    return _standardize_known_columns(pl.DataFrame(records))


def _read_raw_input_excel_frame(path: Path, preferred_sheet: str = "raw") -> pl.DataFrame:
    excel_path = Path(path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {excel_path}")

    try:
        raw_df = _read_with_polars(excel_path, preferred_sheet, preferred_sheet)
    except Exception:
        target_sheet = _select_sheet_name(excel_path, preferred_sheet)
        raw_df = _read_with_openpyxl(excel_path, target_sheet, preferred_sheet)
    return _standardize_known_columns(raw_df)


def _year_expr(column_name: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column_name).cast(pl.Int64, strict=False),
            pl.col(column_name)
            .cast(pl.Utf8, strict=False)
            .str.extract(r"(\d{4})", group_index=1)
            .cast(pl.Int64, strict=False),
        ]
    )


def _metric_text_expr(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars()


def _metric_parsed_expr(column_name: str) -> pl.Expr:
    return _metric_text_expr(column_name).str.replace_all(",", "").cast(pl.Float64, strict=False)


def _metric_parse_error_expr(column_name: str) -> pl.Expr:
    text_expr = _metric_text_expr(column_name)
    parsed_expr = _metric_parsed_expr(column_name)
    return (
        (text_expr.is_not_null() & (text_expr != "") & parsed_expr.is_null())
        .cast(pl.UInt32)
        .alias(f"__parse_error_{column_name}")
    )


def _metric_expr(column_name: str) -> pl.Expr:
    return _metric_parsed_expr(column_name).fill_null(0.0).alias(column_name)


def _validate_metric_parse_errors(
    df: pl.DataFrame,
    metric_columns: Sequence[str],
    context: str,
    threshold: float = METRIC_PARSE_ERROR_THRESHOLD,
) -> None:
    if df.is_empty() or threshold <= 0:
        return
    targets = [column for column in metric_columns if column in df.columns]
    if not targets:
        return

    checks_df = df.select([_metric_parse_error_expr(column) for column in targets])
    row_count = int(df.height)
    failures: list[str] = []
    for column in targets:
        check_col = f"__parse_error_{column}"
        count_value = checks_df.select(pl.col(check_col).sum()).to_series(0)[0]
        parse_error_count = int(count_value or 0)
        parse_error_ratio = parse_error_count / row_count if row_count > 0 else 0.0
        if parse_error_ratio > threshold:
            failures.append(f"{column}={parse_error_ratio:.2%} ({parse_error_count}/{row_count})")

    if failures:
        joined = ", ".join(failures)
        raise ValueError(
            f"Data quality check failed in {context}: metric parse error ratio exceeds {threshold:.2%} ({joined})"
        )


def _int_expr(column_name: str) -> pl.Expr:
    return pl.coalesce(
        [
            pl.col(column_name).cast(pl.Int64, strict=False),
            pl.col(column_name)
            .cast(pl.Utf8, strict=False)
            .str.extract(r"(\d+)", group_index=1)
            .cast(pl.Int64, strict=False),
        ]
    )


def _dimension_expr(column_name: str) -> pl.Expr:
    return (
        pl.col(column_name)
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .fill_null("UNKNOWN")
        .alias(column_name)
    )


def _normalized_text_expr(column_name: str) -> pl.Expr:
    return pl.col(column_name).cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()


def _ext_revenue_source_expr(columns: Sequence[str]) -> pl.Expr:
    base_revenue = _metric_parsed_expr("Revenue").fill_null(0.0)
    if "GROSS_REVENUE" not in columns:
        return base_revenue.alias("Ext Revenue")

    gross_revenue = _metric_parsed_expr("GROSS_REVENUE").fill_null(0.0)
    use_sec_revenue = (_normalized_text_expr("SUBSIDIARY") == pl.lit("SEC")).fill_null(False)
    if "PLATFORM" in columns:
        use_tiktok_revenue = _normalized_text_expr("PLATFORM").str.contains("TIKTOK").fill_null(False)
    else:
        use_tiktok_revenue = pl.lit(False)
    return pl.when(use_sec_revenue | use_tiktok_revenue).then(gross_revenue).otherwise(base_revenue).alias("Ext Revenue")


def _raw_ext_revenue_expr(columns: Sequence[str]) -> pl.Expr:
    base_column = "PLATFORM_REVENUE_USD" if "PLATFORM_REVENUE_USD" in columns else "Revenue"
    if base_column in columns:
        base_revenue = _metric_parsed_expr(base_column).fill_null(0.0)
    else:
        base_revenue = pl.lit(0.0)

    if "GROSS_REVENUE" not in columns:
        return base_revenue.alias("Ext Revenue")

    gross_revenue = _metric_parsed_expr("GROSS_REVENUE").fill_null(0.0)
    if "SUBSIDIARY" in columns:
        use_sec_revenue = (_normalized_text_expr("SUBSIDIARY") == pl.lit("SEC")).fill_null(False)
    else:
        use_sec_revenue = pl.lit(False)
    if "PLATFORM" in columns:
        use_tiktok_revenue = _normalized_text_expr("PLATFORM").str.contains("TIKTOK").fill_null(False)
    else:
        use_tiktok_revenue = pl.lit(False)
    return pl.when(use_sec_revenue | use_tiktok_revenue).then(gross_revenue).otherwise(base_revenue).alias("Ext Revenue")


def _empty_engine_frame() -> pl.DataFrame:
    return pl.DataFrame({column: [] for column in ENGINE_COLUMNS})


def _has_activity_expr() -> pl.Expr:
    return (
        (pl.col("Spend_curr") > 0)
        | (pl.col("Spend_prev") > 0)
        | (pl.col("Revenue_curr") > 0)
        | (pl.col("Revenue_prev") > 0)
    )


def _filter_conversion_objective(df: pl.DataFrame) -> tuple[pl.DataFrame, Dict[str, Any]]:
    objective_column = next((col for col in OBJECTIVE_CANDIDATES if col in df.columns), None)
    meta: Dict[str, Any] = {
        "objective_filter_applied": objective_column is not None,
        "objective_column": objective_column,
        "objective_value": TARGET_OBJECTIVE_VALUE,
    }
    if objective_column is None:
        return df, meta

    before_rows = int(df.height)
    filtered = df.filter(
        pl.col(objective_column).cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
        == pl.lit(TARGET_OBJECTIVE_VALUE)
    )
    meta["objective_rows_before"] = before_rows
    meta["objective_rows_after"] = int(filtered.height)
    return filtered, meta


def _filter_target_divisions(df: pl.DataFrame) -> tuple[pl.DataFrame, Dict[str, Any]]:
    has_division = "DIVISION" in df.columns
    meta: Dict[str, Any] = {
        "division_filter_applied": has_division,
        "division_filter_values": list(TARGET_DIVISIONS),
    }
    if not has_division:
        return df, meta

    before_rows = int(df.height)
    filtered = df.filter(
        pl.col("DIVISION")
        .cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .is_in(list(TARGET_DIVISIONS))
    )
    meta["division_rows_before"] = before_rows
    meta["division_rows_after"] = int(filtered.height)
    return filtered, meta


def _normalize_wide_engine_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return _empty_engine_frame()

    missing = sorted(set(ENGINE_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected = df.select(ENGINE_COLUMNS)
    _validate_metric_parse_errors(
        selected,
        metric_columns=ENGINE_METRIC_COLUMNS,
        context="wide_engine_frame",
    )

    normalized = (
        selected
        .with_columns(
            [_dimension_expr(dim) for dim in DIMENSIONS]
            + [pl.col(metric).cast(pl.Float64, strict=False).fill_null(0.0).alias(metric) for metric in ENGINE_METRIC_COLUMNS]
        )
        .select(ENGINE_COLUMNS)
    )
    return normalized.filter(_has_activity_expr())


def _normalize_long_frame(df: pl.DataFrame) -> pl.DataFrame:
    year_column = "Year" if "Year" in df.columns else "YEAR" if "YEAR" in df.columns else None
    month_column = "Month" if "Month" in df.columns else "MONTH" if "MONTH" in df.columns else None
    day_column = "Day" if "Day" in df.columns else "DAY" if "DAY" in df.columns else None
    required_raw = [*DIMENSIONS, *RAW_METRIC_MAP.keys()]
    if year_column is None:
        raise ValueError("Missing required columns: ['Year']")

    missing = sorted(set(required_raw).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    renamed = df.rename(RAW_METRIC_MAP)
    selected_columns = [*DIMENSIONS, year_column, *METRICS]
    if "GROSS_REVENUE" in renamed.columns:
        selected_columns.append("GROSS_REVENUE")
    if "PLATFORM" in renamed.columns:
        selected_columns.append("PLATFORM")
    if month_column is not None:
        selected_columns.append(month_column)
    if day_column is not None:
        selected_columns.append(day_column)

    selected = renamed.select(selected_columns)
    metric_quality_columns = list(METRICS)
    if "GROSS_REVENUE" in selected.columns:
        metric_quality_columns.append("GROSS_REVENUE")
    _validate_metric_parse_errors(
        selected,
        metric_columns=metric_quality_columns,
        context="long_engine_frame",
    )

    exprs = (
        [_dimension_expr(dim) for dim in DIMENSIONS]
        + [_metric_expr("Spend"), _metric_expr("Clicks"), _metric_expr("Orders"), _ext_revenue_source_expr(selected.columns)]
        + [_year_expr(year_column).alias("Year")]
    )

    output_columns = [*DIMENSIONS, "Year"]
    if month_column is not None:
        exprs.append(_int_expr(month_column).alias("Month"))
        output_columns.append("Month")
    if day_column is not None:
        exprs.append(_int_expr(day_column).alias("Day"))
        output_columns.append("Day")

    normalized = selected.with_columns(exprs).with_columns(pl.col("Ext Revenue").alias("Revenue"))
    output_columns.extend([*METRICS, "Ext Revenue"])
    return normalized.select(output_columns)


def _filter_target_years(df: pl.DataFrame, curr_year: int, prev_year: int) -> pl.DataFrame:
    return df.filter(pl.col("Year").is_in([curr_year, prev_year]))


def _apply_mtd_alignment(
    df: pl.DataFrame,
    curr_year: int,
    prev_year: int,
) -> tuple[pl.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "curr_year": curr_year,
        "prev_year": prev_year,
        "mtd_applied": False,
    }
    scoped = _filter_target_years(df, curr_year=curr_year, prev_year=prev_year)
    if scoped.is_empty():
        return scoped, meta
    if "Month" not in scoped.columns:
        return scoped, meta

    curr_scope = scoped.filter(pl.col("Year") == curr_year)
    if curr_scope.is_empty():
        return scoped, meta

    max_month = curr_scope.select(pl.col("Month").max()).to_series(0)[0]
    if max_month is None:
        return scoped, meta

    meta["mtd_month_start"] = int(max_month)
    meta["mtd_month_cutoff"] = int(max_month)

    has_day = "Day" in scoped.columns
    max_day = None
    if has_day:
        max_day = (
            curr_scope.filter(pl.col("Month") == pl.lit(max_month))
            .select(pl.col("Day").max())
            .to_series(0)[0]
        )
        if max_day is not None:
            meta["mtd_day_start"] = 1
            meta["mtd_day_cutoff"] = int(max_day)

    if has_day and max_day is not None:
        in_window = (
            (pl.col("Month") == pl.lit(max_month))
            & (pl.col("Day") >= pl.lit(1))
            & (pl.col("Day") <= pl.lit(max_day))
        )
    else:
        in_window = pl.col("Month") == pl.lit(max_month)

    meta["mtd_applied"] = True
    return scoped.filter(in_window), meta


def _pivot_long_to_engine(df: pl.DataFrame, curr_year: int, prev_year: int) -> pl.DataFrame:
    if df.is_empty():
        return _empty_engine_frame()

    agg_exprs: list[pl.Expr] = []
    for metric in METRICS:
        agg_exprs.append(pl.col(metric).filter(pl.col("Year") == curr_year).sum().alias(f"{metric}_curr"))
        agg_exprs.append(pl.col(metric).filter(pl.col("Year") == prev_year).sum().alias(f"{metric}_prev"))

    pivoted = df.group_by(DIMENSIONS).agg(agg_exprs).with_columns(
        [pl.col(metric).cast(pl.Float64, strict=False).fill_null(0.0).alias(metric) for metric in ENGINE_METRIC_COLUMNS]
    )

    return pivoted.select(ENGINE_COLUMNS).filter(_has_activity_expr())


def _to_engine_frame(
    df: pl.DataFrame,
    curr_year: int,
    prev_year: int,
    mtd_only: bool,
) -> tuple[pl.DataFrame, Dict[str, Any]]:
    filtered_input, objective_meta = _filter_conversion_objective(df)
    scoped_input, division_meta = _filter_target_divisions(filtered_input)
    if set(ENGINE_COLUMNS).issubset(scoped_input.columns):
        meta = {
            "curr_year": curr_year,
            "prev_year": prev_year,
            "mtd_applied": False,
            "source_format": "wide",
        }
        meta.update(division_meta)
        meta.update(objective_meta)
        return _normalize_wide_engine_frame(scoped_input), meta

    long_df = _normalize_long_frame(scoped_input)
    if mtd_only:
        filtered_df, meta = _apply_mtd_alignment(long_df, curr_year=curr_year, prev_year=prev_year)
    else:
        filtered_df = _filter_target_years(long_df, curr_year=curr_year, prev_year=prev_year)
        meta = {"curr_year": curr_year, "prev_year": prev_year, "mtd_applied": False}
    meta["source_format"] = "long"
    meta.update(division_meta)
    meta.update(objective_meta)
    return _pivot_long_to_engine(filtered_df, curr_year=curr_year, prev_year=prev_year), meta


def _filter_raw_frame_for_html_window(
    df: pl.DataFrame,
    curr_year: int,
    prev_year: int,
    mtd_only: bool,
) -> tuple[pl.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "curr_year": curr_year,
        "prev_year": prev_year,
        "mtd_applied": False,
    }
    year_column = "Year" if "Year" in df.columns else "YEAR" if "YEAR" in df.columns else None
    if year_column is None:
        return df, meta

    scoped = df.with_columns(_year_expr(year_column).alias("__calc_year")).filter(pl.col("__calc_year").is_in([curr_year, prev_year]))
    if not mtd_only:
        return scoped.drop("__calc_year"), meta

    month_column = "Month" if "Month" in scoped.columns else "MONTH" if "MONTH" in scoped.columns else None
    if month_column is None:
        return scoped.drop("__calc_year"), meta

    scoped = scoped.with_columns(_int_expr(month_column).alias("__calc_month"))
    curr_scope = scoped.filter(pl.col("__calc_year") == curr_year)
    if curr_scope.is_empty():
        drop_columns = [column for column in ["__calc_year", "__calc_month"] if column in scoped.columns]
        return scoped.drop(*drop_columns), meta

    max_month = curr_scope.select(pl.col("__calc_month").max()).to_series(0)[0]
    if max_month is None:
        drop_columns = [column for column in ["__calc_year", "__calc_month"] if column in scoped.columns]
        return scoped.drop(*drop_columns), meta

    meta["mtd_month_start"] = int(max_month)
    meta["mtd_month_cutoff"] = int(max_month)

    day_column = "Day" if "Day" in scoped.columns else "DAY" if "DAY" in scoped.columns else None
    max_day = None
    if day_column is not None:
        scoped = scoped.with_columns(_int_expr(day_column).alias("__calc_day"))
        max_day = (
            curr_scope.with_columns(_int_expr(day_column).alias("__calc_day"))
            .filter(pl.col("__calc_month") == pl.lit(max_month))
            .select(pl.col("__calc_day").max())
            .to_series(0)[0]
        )
        if max_day is not None:
            meta["mtd_day_start"] = 1
            meta["mtd_day_cutoff"] = int(max_day)

    if day_column is not None and max_day is not None:
        in_window = (
            (pl.col("__calc_month") == pl.lit(max_month))
            & (pl.col("__calc_day") >= pl.lit(1))
            & (pl.col("__calc_day") <= pl.lit(max_day))
        )
    else:
        in_window = pl.col("__calc_month") == pl.lit(max_month)

    filtered = scoped.filter(in_window)
    meta["mtd_applied"] = True
    drop_columns = [column for column in ["__calc_year", "__calc_month", "__calc_day"] if column in filtered.columns]
    return filtered.drop(*drop_columns), meta


def build_html_calc_raw_sheets(
    path: str | Path,
    curr_year: int = DEFAULT_CURR_YEAR,
    prev_year: int = DEFAULT_PREV_YEAR,
    preferred_sheet: str = "raw",
    mtd_only: bool = True,
) -> tuple[Dict[str, pl.DataFrame], Dict[str, Any]]:
    raw_df = _read_raw_input_excel_frame(path, preferred_sheet=preferred_sheet)
    filtered_objective_df, objective_meta = _filter_conversion_objective(raw_df)
    scoped_df, division_meta = _filter_target_divisions(filtered_objective_df)

    if set(ENGINE_COLUMNS).issubset(scoped_df.columns):
        normalized_df = _normalize_wide_engine_frame(scoped_df)
        engine_df = normalized_df
        long_meta: Dict[str, Any] = {
            "curr_year": curr_year,
            "prev_year": prev_year,
            "mtd_applied": False,
            "source_format": "wide",
        }
    else:
        long_df = _normalize_long_frame(scoped_df)
        if mtd_only:
            normalized_df, long_meta = _apply_mtd_alignment(long_df, curr_year=curr_year, prev_year=prev_year)
        else:
            normalized_df = _filter_target_years(long_df, curr_year=curr_year, prev_year=prev_year)
            long_meta = {"curr_year": curr_year, "prev_year": prev_year, "mtd_applied": False}
        long_meta["source_format"] = "long"
        engine_df = _pivot_long_to_engine(normalized_df, curr_year=curr_year, prev_year=prev_year)

    raw_full_df, raw_full_meta = _filter_raw_frame_for_html_window(
        scoped_df,
        curr_year=curr_year,
        prev_year=prev_year,
        mtd_only=mtd_only,
    )
    raw_full_df = raw_full_df.with_columns(_raw_ext_revenue_expr(raw_full_df.columns))

    merged_meta: Dict[str, Any] = {"curr_year": curr_year, "prev_year": prev_year}
    merged_meta.update(objective_meta)
    merged_meta.update(division_meta)
    merged_meta.update(long_meta)
    merged_meta.update(raw_full_meta)
    merged_meta["raw_rows_input"] = int(raw_df.height)
    merged_meta["raw_rows_objective_filtered"] = int(filtered_objective_df.height)
    merged_meta["raw_rows_division_filtered"] = int(scoped_df.height)
    merged_meta["raw_rows_mtd_html_calc"] = int(normalized_df.height)
    merged_meta["raw_rows_mtd_html_calc_full"] = int(raw_full_df.height)
    merged_meta["engine_rows_html_calc"] = int(engine_df.height)
    merged_meta["input_path"] = str(Path(path).expanduser().resolve())

    meta_df = pl.DataFrame(
        {
            "key": list(merged_meta.keys()),
            "value": [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for v in merged_meta.values()],
        }
    )
    sheets = {
        "raw_mtd_html_calc_full": raw_full_df,
        "raw_mtd_html_calc": normalized_df,
        "engine_html_input": engine_df,
        "meta": meta_df,
    }
    return sheets, merged_meta


def read_input_excel(
    path: str | Path,
    preferred_sheet: str = "raw",
    curr_year: int = DEFAULT_CURR_YEAR,
    prev_year: int = DEFAULT_PREV_YEAR,
    mtd_only: bool = True,
    return_meta: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, Dict[str, Any]]:
    """Read input Excel and return engine-ready wide schema with curr/prev metrics."""
    raw_df = _read_raw_input_excel_frame(path, preferred_sheet=preferred_sheet)

    engine_df, meta = _to_engine_frame(
        raw_df,
        curr_year=curr_year,
        prev_year=prev_year,
        mtd_only=mtd_only,
    )
    if return_meta:
        return engine_df, meta
    return engine_df


def _excel_cell_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, bool, str)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_with_polars(path: Path, sheets: Dict[str, pl.DataFrame]) -> bool:
    if not sheets:
        return False

    try:
        if hasattr(pl, "write_excel"):
            pl.write_excel(  # type: ignore[attr-defined]
                workbook=path,
                worksheets=sheets,
            )
            return True
    except Exception:
        pass

    first_df = next(iter(sheets.values()))
    if not hasattr(first_df, "write_excel"):
        return False

    try:
        first = True
        for sheet_name, frame in sheets.items():
            if first:
                frame.write_excel(path, worksheet=sheet_name)  # type: ignore[arg-type]
                first = False
            else:
                frame.write_excel(path, worksheet=sheet_name, mode="a")  # type: ignore[arg-type]
        return True
    except Exception:
        return False


def _write_with_openpyxl(path: Path, sheets: Dict[str, pl.DataFrame]) -> None:
    Workbook, _ = _import_openpyxl()
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)

    for sheet_name, frame in sheets.items():
        worksheet = workbook.create_sheet(title=str(sheet_name)[:31])
        worksheet.append(frame.columns)
        for row in frame.iter_rows(named=False):
            worksheet.append([_excel_cell_value(value) for value in row])

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def write_output_excel(path: str | Path, sheets: Dict[str, pl.DataFrame]) -> None:
    """Write output Excel with Polars-first and openpyxl fallback."""
    excel_path = Path(path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    if _write_with_polars(excel_path, sheets):
        return
    _write_with_openpyxl(excel_path, sheets)
