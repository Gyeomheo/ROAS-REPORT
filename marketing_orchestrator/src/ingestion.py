"""Excel ingestion/output helpers with Polars-first and openpyxl fallback."""

from __future__ import annotations

import json
import os
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Dict, Sequence

# raw DataFrame parquet 캐시 경로 (ingestion.py 기준 상위 2단계 = marketing_orchestrator/)
_RAW_PARQUET_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"


def _raw_parquet_path(source_path: Path) -> Path:
    """input.xlsx에 대응하는 raw parquet 캐시 경로."""
    return _RAW_PARQUET_CACHE_DIR / f"{source_path.stem}_raw.parquet"


def _raw_parquet_valid(cache_path: Path, source_path: Path) -> bool:
    """캐시가 존재하고 소스 파일보다 최신이면 유효."""
    try:
        return cache_path.exists() and cache_path.stat().st_mtime >= source_path.stat().st_mtime
    except OSError:
        return False


def _save_raw_parquet(df: pl.DataFrame, cache_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path)
    except Exception:
        pass  # 캐시 저장 실패는 non-fatal


def _load_raw_parquet(cache_path: Path) -> pl.DataFrame:
    return pl.read_parquet(cache_path)

import polars as pl


DIMENSIONS: list[str] = ["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"]
RAW_METRIC_MAP: dict[str, str] = {
    "PLATFORM_SPEND_USD": "Spend",
    "PLATFORM_REVENUE_USD": "Revenue",
    "PLATFORM_CLICKS": "Clicks",
    "GROSS_ORDERS": "Orders",
}
RAW_ZERO_ACTIVITY_METRIC_COLUMNS: tuple[str, ...] = (
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
    "Ext Revenue",
)
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

# ---------------------------------------------------------------------------
# Product enrichment constants
# ---------------------------------------------------------------------------

# PRODUCT values eligible for enrichment (null / vague / cross-product labels)
_NULL_LIKE_PRODUCT_VALUES: frozenset[str] = frozenset({
    "", "N/A", "NULL", "NONE", "-", "NA",
    "OTHERS",
    "MX OTHERS", "DA OTHERS", "VD OTHERS", "TAB OTHERS", "TV OTHERS",
    "MX CROSS PRODUCTS", "DA CROSS PRODUCTS", "VD CROSS PRODUCTS",
    "TAB CROSS PRODUCTS", "CROSS PRODUCTS", "CROSS DIVISION",
})

# SB~ → single canonical PRODUCT  (≥85 % concentration in validated rows)
_SB_TO_PRODUCT: dict[str, str] = {
    # DA — Home Appliance
    "refrig":    "REFRIGERATOR",
    "washmach":  "WASHER",
    "aircon":    "AIR CONDITIONER",
    "airpur":    "AIR PURIFIER",
    "aird":      "AIR DRESSER/SHOE DRESSER",
    "dish":      "DISHWASHER",
    "micro":     "MICROWAVE/OTR/QOOKER",
    "oven":      "OVEN/COMBI OVEN",
    "ldy":       "DRYER",
    "wpf":       "WATER PURIFIER",
    # VD — Display
    "mon":       "ESSENTIAL MONITOR",
    "wa":        "LIFESTYLE TV",
    # MX — Mobile / PC / Wearable
    "pcl":       "NOTEBOOK",
    "npc":       "NOTEBOOK",
    "pca":       "PC ACC",
    "moacc":     "MX ACC",
    "tab":       "TAB S SERIES",
    "wearsmart": "WATCH",
    "buds-pro":  "BUDS",
    "msmul":     "STICK VACUUM",
    "vacuum":    "STICK VACUUM",
    # Audio
    "hometheat": "SOUND DEVICE",
    "aud":       "SOUND DEVICE",
}

# SB~ → fixed cross-product label (원본 taxonomy 값 준수)
_SB_TO_CROSS_LABEL: dict[str, str] = {
    "wearoth":  "MX OTHERS",         # Wearable Others: BUDS/RING/WATCH 혼재
    "wearfit":  "MX OTHERS",         # Wearable Fit: S25/WATCH 혼재
    "vdc":      "VD CROSS PRODUCTS", # VD Cross
    "vdmul":    "VD MULTI",
    "mocross":  "MX CROSS PRODUCTS", # MX Cross
    "samsa":    "MX CROSS PRODUCTS",
    "dcr":      "DA CROSS PRODUCTS", # DA Cross
    "corpb2b":  "B2B",
    "sound":    "SOUND DEVICE",
    # tv은 원본 PRODUCT 값에 따라 분기 → _create_products_column 내 특수 처리
}

# SB~ codes whose label = "{DIVISION} MULTI"  (DIVISION 컬럼 참조)
_SB_DIVISION_MULTI: frozenset[str] = frozenset({"multi", "cptg", "crgbm", "crop"})

# URL path slug → canonical PRODUCT
_URL_SLUG_TO_PRODUCT: dict[str, str] = {
    "air-conditioners":             "AIR CONDITIONER",
    "air-cleaner":                  "AIR PURIFIER",
    "air-care":                     "AIR PURIFIER",
    "air-purifiers":                "AIR PURIFIER",
    "air-purifier":                 "AIR PURIFIER",
    "airdressers-and-shoedressers": "AIR DRESSER/SHOE DRESSER",
    "refrigerators":                "REFRIGERATOR",
    "washers-and-dryers":           "WASHER",
    "washers":                      "WASHER",
    "dishwashers":                  "DISHWASHER",
    "cooking":                      "RANGE/COOKER",
    "microwaves":                   "MICROWAVE/OTR/QOOKER",
    "ovens":                        "OVEN/COMBI OVEN",
    "tablets":                      "TAB S SERIES",
    "galaxy-tab-s":                 "TAB S SERIES",
    "monitors":                     "ESSENTIAL MONITOR",
    "gaming-monitors":              "ODYSSEY",
    "notebook":                     "NOTEBOOK",
    "notebooks":                    "NOTEBOOK",
    "laptops":                      "NOTEBOOK",
    "mobile-accessories":           "MX ACC",
    "galaxy-buds":                  "BUDS",
    "galaxy-watch":                 "WATCH",
    "galaxy-ring":                  "RING",
    "water-purifiers":              "WATER PURIFIER",
    "vacuum-cleaners":              "STICK VACUUM",
    "robot-vacuum":                 "ROBOT VACUUM",
    "soundbars":                    "SOUND DEVICE",
    "soundbar":                     "SOUND DEVICE",
    "dryers":                       "DRYER",
}

# CN~[코드] 패턴 (CAMPAIGN_NAME에서 CN~Ekrfg, CN~Ebqoop 같은 패턴 파싱)
# sec campaign name.xlsx keyword report reverse-traced on 2026-05-07.
_CAMPAIGN_CN_TO_PRODUCT: dict[str, str] = {
    "arc": "AIR CONDITIONER",
    "acl": "AIR PURIFIER",
    "ard": "AIR DRESSER/SHOE DRESSER",
    "jbl": "HARMAN",
    "qoop": "MICROWAVE/OTR/QOOKER",
    "emvip": "LIFESTYLE TV",
    "bsmartm": "DA CROSS PRODUCTS",
    "bsmartp": "DA CROSS PRODUCTS",
    "ebaclm": "AIR PURIFIER",
    "ebaclp": "AIR PURIFIER",
    "ebarcm": "AIR CONDITIONER",
    "ebarcp": "AIR CONDITIONER",
    "ebardm": "AIR DRESSER/SHOE DRESSER",
    "ebardp": "AIR DRESSER/SHOE DRESSER",
    "ebcarem": "DA CROSS PRODUCTS",
    "ebcarep": "DA CROSS PRODUCTS",
    "ebdocm3": "MX CROSS PRODUCTS",
    "ebdocp3": "MX CROSS PRODUCTS",
    "ebdrym": "DRYER",
    "ebdryp": "DRYER",
    "ebdswm": "DISHWASHER",
    "ebdswp": "DISHWASHER",
    "ebhmkm": "HARMAN",
    "ebhmkp": "HARMAN",
    "ebindm": "RANGE/COOKER",
    "ebindp": "RANGE/COOKER",
    "ebkchm": "REFRIGERATOR",
    "ebkchp": "REFRIGERATOR",
    "ebltvm": "LIFESTYLE TV",
    "ebltvp": "LIFESTYLE TV",
    "ebmonm": "ESSENTIAL MONITOR",
    "ebmonp": "ESSENTIAL MONITOR",
    "ebmvim": "LIFESTYLE TV",
    "ebmvip": "LIFESTYLE TV",
    "ebppcm": "NOTEBOOK",
    "ebppcp": "NOTEBOOK",
    "ebprim": "PRINTER",
    "ebprip": "PRINTER",
    "ebps6m": "S SERIES",
    "ebps6p": "S SERIES",
    "ebpz7m1": "Z SERIES",
    "ebpz7p1": "Z SERIES",
    "ebqoom": "MICROWAVE/OTR/QOOKER",
    "ebqoop": "MICROWAVE/OTR/QOOKER",
    "ebrfgm": "REFRIGERATOR",
    "ebrfgp": "REFRIGERATOR",
    "ebsubm": "DA CROSS PRODUCTS",
    "ebsubp": "DA CROSS PRODUCTS",
    "ebtapm": "TAB S SERIES",
    "ebtapp": "TAB S SERIES",
    "ebttvm": "TV",
    "ebttvp": "TV",
    "ebuhdm": "TV",
    "ebuhdp": "TV",
    "ebvcum": "ROBOT VACUUM",
    "ebvcup": "ROBOT VACUUM",
    "ebweam": "MX OTHERS",
    "ebweap": "MX OTHERS",
    "ebwpfm": "WATER PURIFIER",
    "ebwpfp": "WATER PURIFIER",
    "ebwshm": "WASHER",
    "ebwshp": "WASHER",
    "ef1h26": "S SERIES",
    "ekacl": "AIR PURIFIER",
    "ekarc": "AIR CONDITIONER",
    "ekard": "AIR DRESSER/SHOE DRESSER",
    "ekbud4": "BUDS",
    "ekcare": "DA CROSS PRODUCTS",
    "ekdry": "DRYER",
    "ekdsw": "DISHWASHER",
    "ekfit": "MX OTHERS",
    "ekhmk": "SOUND DEVICE",
    "ekind": "RANGE/COOKER",
    "ekjbl": "HARMAN",
    "ekkch": "REFRIGERATOR",
    "ekltv": "LIFESTYLE TV",
    "ekmni": "LIFESTYLE TV",
    "ekppc6": "NOTEBOOK",
    "ekpri": "PRINTER",
    "ekpz7": "Z SERIES",
    "ekrfg": "REFRIGERATOR",
    "eksub": "DA CROSS PRODUCTS",
    "ekta11": "TAB S SERIES",
    "ekttv": "TV",
    "ekvcu": "ROBOT VACUUM",
    "ekwat8": "WATCH",
    "ekwsh": "WASHER",
    "ebuds": "BUDS",
    "ewatch": "WATCH",
    "eaisteam": "FAMILYHUB/AI HOME",
}

def _create_products_column(df: pl.DataFrame) -> pl.DataFrame:
    """PRODUCT 우측에 PRODUCTS 열 추가 (원본 PRODUCT 보존).

    PRODUCT가 _NULL_LIKE_PRODUCT_VALUES인 행에 대해서만 추론 적용.
    추론 우선순위:
      1. MX_FLAGSHIP_S → S SERIES  (S24/S25/S26 포함 시)
      2. MX_FLAGSHIP_Z → Z SERIES
      3. SB~ → _SB_TO_PRODUCT      (단일 제품 고정 매핑)
      4. CN~[코드] → _CAMPAIGN_CN_TO_PRODUCT (CAMPAIGN_NAME 패턴)
      5. SB~ → _SB_TO_CROSS_LABEL  (교차 제품 고정 레이블)
      6. SB~ ∈ _SB_DIVISION_MULTI  → "{DIVISION} MULTI"
      7. URL slug → _URL_SLUG_TO_PRODUCT
      8. fallback → 원본 PRODUCT 그대로
    """
    if "PRODUCT" not in df.columns:
        return df

    null_like = list(_NULL_LIKE_PRODUCT_VALUES)
    product_upper = (
        pl.col("PRODUCT").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
    )
    is_problem = product_upper.is_null() | product_upper.is_in(null_like)

    # --- Level 1 & 2: MX Flagship ---
    mx_s = pl.lit(None, dtype=pl.Utf8)
    if "MX_FLAGSHIP_S" in df.columns:
        fs = pl.col("MX_FLAGSHIP_S").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
        mx_s = pl.when(
            fs.is_not_null()
            & (
                fs.str.contains(r"S2[456]")
                | fs.str.contains("S-SERIES")
                | fs.str.contains("S SERIES")
            )
        ).then(pl.lit("S SERIES")).otherwise(pl.lit(None, dtype=pl.Utf8))

    mx_z = pl.lit(None, dtype=pl.Utf8)
    if "MX_FLAGSHIP_Z" in df.columns:
        fz = pl.col("MX_FLAGSHIP_Z").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
        mx_z = pl.when(
            fz.is_not_null() & fz.str.contains("Z")
        ).then(pl.lit("Z SERIES")).otherwise(pl.lit(None, dtype=pl.Utf8))

    # --- Level 3: CN~[코드] (CAMPAIGN_NAME 파싱) ---
    cn_product = pl.lit(None, dtype=pl.Utf8)
    if "CAMPAIGN_NAME" in df.columns:
        cn_code = (
            pl.col("CAMPAIGN_NAME")
            .cast(pl.Utf8, strict=False)
            .str.extract(r"(?i)CN~([a-z0-9]+)", 1)
            .str.to_lowercase()
        )
        cn_product = cn_code.replace(_CAMPAIGN_CN_TO_PRODUCT, default=None)

    # --- Level 4 ~ 7: SB~ ---
    sb_fixed = pl.lit(None, dtype=pl.Utf8)
    sb_cross = pl.lit(None, dtype=pl.Utf8)
    sb_tv = pl.lit(None, dtype=pl.Utf8)
    sb_div_multi = pl.lit(None, dtype=pl.Utf8)
    if "CAMPAIGN_NAME" in df.columns:
        sb_code = (
            pl.col("CAMPAIGN_NAME")
            .cast(pl.Utf8, strict=False)
            .str.extract(r"(?i)SB~([^_~\s]+)", 1)
            .str.to_lowercase()
        )
        sb_fixed = sb_code.replace(_SB_TO_PRODUCT, default=None)
        sb_cross = sb_code.replace(_SB_TO_CROSS_LABEL, default=None)

        # tv 특수 처리: 원본 PRODUCT가 VD CROSS PRODUCTS면 유지, 나머지는 TV OTHERS
        sb_tv = pl.when(sb_code == pl.lit("tv")).then(
            pl.when(product_upper == pl.lit("VD CROSS PRODUCTS"))
            .then(pl.lit("VD CROSS PRODUCTS"))
            .otherwise(pl.lit("TV OTHERS"))
        ).otherwise(pl.lit(None, dtype=pl.Utf8))

        if "DIVISION" in df.columns:
            div_upper = (
                pl.col("DIVISION").cast(pl.Utf8, strict=False).str.strip_chars().str.to_uppercase()
            )
            sb_div_multi = pl.when(
                sb_code.is_in(list(_SB_DIVISION_MULTI)) & div_upper.is_not_null()
            ).then(
                pl.concat_str([div_upper, pl.lit(" MULTI")])
            ).otherwise(pl.lit(None, dtype=pl.Utf8))

    # --- Level 6: URL slug ---
    url_inferred = pl.lit(None, dtype=pl.Utf8)
    if "PLATFORM_LANDING_PAGE_URL" in df.columns:
        slug = (
            pl.col("PLATFORM_LANDING_PAGE_URL")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .str.extract(r"samsung\.com/[^/]*/([^/?#]+)", 1)
        )
        url_inferred = slug.replace(_URL_SLUG_TO_PRODUCT, default=None)

    # --- Combine & apply ---
    inferred = pl.coalesce([mx_s, mx_z, sb_fixed, cn_product, sb_cross, sb_tv, sb_div_multi, url_inferred])
    products_expr = (
        pl.when(is_problem & inferred.is_not_null())
        .then(inferred)
        .otherwise(pl.col("PRODUCT"))
        .alias("PRODUCTS")
    )

    result = df.with_columns(products_expr)

    # PRODUCTS를 PRODUCT 바로 우측에 배치
    cols = result.columns
    idx = cols.index("PRODUCT")
    ordered = cols[: idx + 1] + ["PRODUCTS"] + [c for c in cols[idx + 1 :] if c != "PRODUCTS"]
    return result.select(ordered)


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


def _parse_excel_engine_candidates() -> tuple[str, ...]:
    # Fast path default: calamine once, then module-level openpyxl fallback in _read_raw_input_excel_frame.
    raw = os.getenv("ROAS_EXCEL_ENGINE", "calamine")
    candidates: list[str] = []
    for token in raw.split(","):
        normalized = token.strip().lower()
        if normalized and normalized not in candidates:
            candidates.append(normalized)
    if not candidates:
        return ("calamine",)
    return tuple(candidates)


def _parse_excel_infer_schema_length() -> int:
    raw = os.getenv("ROAS_EXCEL_INFER_SCHEMA_LENGTH", "10000")
    try:
        length = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid ROAS_EXCEL_INFER_SCHEMA_LENGTH: {raw}") from exc
    if length < 0:
        raise ValueError(f"ROAS_EXCEL_INFER_SCHEMA_LENGTH must be >= 0, got {length}")
    return length


EXCEL_ENGINE_CANDIDATES = _parse_excel_engine_candidates()
EXCEL_INFER_SCHEMA_LENGTH = _parse_excel_infer_schema_length()
EXCEL_SCHEMA_OVERRIDES: dict[str, Any] = {
    "CAMPAIGN_ID": pl.Utf8,
    "ACCOUNT": pl.Utf8,
    "ACCOUNT_ID": pl.Utf8,
}


def _excel_schema_overrides(columns: Any = None) -> dict[str, Any]:
    if columns is None or isinstance(columns, str):
        return dict(EXCEL_SCHEMA_OVERRIDES)
    requested = {str(name).strip().upper() for name in columns if isinstance(name, str)}
    if not requested:
        return {}
    return {
        column: dtype
        for column, dtype in EXCEL_SCHEMA_OVERRIDES.items()
        if column.strip().upper() in requested
    }


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
    """Read once with explicit schema hints; no multi-engine retry loop."""
    read_kwargs = dict(kwargs)
    if "infer_schema_length" not in read_kwargs:
        read_kwargs["infer_schema_length"] = EXCEL_INFER_SCHEMA_LENGTH
    if "schema_overrides" not in read_kwargs:
        overrides = _excel_schema_overrides(read_kwargs.get("columns"))
        if overrides:
            read_kwargs["schema_overrides"] = overrides

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Could not determine dtype for column .*")
        try:
            return pl.read_excel(path, **read_kwargs)  # type: ignore[arg-type]
        except TypeError:
            fallback_kwargs = dict(read_kwargs)
            fallback_kwargs.pop("infer_schema_length", None)
            fallback_kwargs.pop("schema_overrides", None)
            return pl.read_excel(path, **fallback_kwargs)  # type: ignore[arg-type]


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
    """Read Excel with the primary engine only; fallback to openpyxl happens outside."""
    if not hasattr(pl, "read_excel"):
        raise RuntimeError("polars.read_excel is not available in this environment.")

    sheets_to_try: list[str] = []
    for name in [target_sheet, preferred_sheet]:
        if name and name not in sheets_to_try:
            sheets_to_try.append(name)

    primary_engine = EXCEL_ENGINE_CANDIDATES[0] if EXCEL_ENGINE_CANDIDATES else "calamine"
    last_exc: Exception | None = None

    for sheet_name in sheets_to_try:
        try:
            frame = _read_excel_polars(
                path,
                sheet_name=sheet_name,
                engine=primary_engine,
            )
            normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
            if _looks_like_supported_input(normalized.columns):
                return normalized
        except Exception as exc:
            last_exc = exc
            continue

    # Final fallback: let polars pick first sheet once.
    try:
        frame = _read_excel_polars(path, engine=primary_engine)
        normalized = _standardize_known_columns(_frame_from_polars_result(frame, preferred_sheet))
        if _looks_like_supported_input(normalized.columns):
            return normalized
    except Exception as exc:
        last_exc = exc

    if last_exc is not None:
        raise last_exc
    raise ValueError(f"Unable to detect supported input schema via polars ({primary_engine}).")


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
    return _filter_all_zero_raw_metric_rows(_standardize_known_columns(raw_df))


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


def _metric_has_activity_expr(column_name: str) -> pl.Expr:
    text_expr = _metric_text_expr(column_name)
    parsed_expr = _metric_parsed_expr(column_name)
    parse_error = text_expr.is_not_null() & (text_expr != "") & parsed_expr.is_null()
    return (parsed_expr.fill_null(0.0) != 0.0) | parse_error


def _filter_all_zero_raw_metric_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop raw rows where all available activity metrics are zero/blank."""
    if df.is_empty():
        return df
    metric_columns = [column for column in RAW_ZERO_ACTIVITY_METRIC_COLUMNS if column in df.columns]
    if not metric_columns:
        return df
    return df.filter(pl.any_horizontal([_metric_has_activity_expr(column) for column in metric_columns]))


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
        use_meta_revenue = _normalized_text_expr("PLATFORM").str.contains("META|FACEBOOK").fill_null(False)
    else:
        use_tiktok_revenue = pl.lit(False)
        use_meta_revenue = pl.lit(False)
    return pl.when(use_sec_revenue | use_tiktok_revenue | use_meta_revenue).then(gross_revenue).otherwise(base_revenue).alias("Ext Revenue")


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
        use_meta_revenue = _normalized_text_expr("PLATFORM").str.contains("META|FACEBOOK").fill_null(False)
    else:
        use_tiktok_revenue = pl.lit(False)
        use_meta_revenue = pl.lit(False)
    return pl.when(use_sec_revenue | use_tiktok_revenue | use_meta_revenue).then(gross_revenue).otherwise(base_revenue).alias("Ext Revenue")


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
    # MTD month/day window is temporarily disabled.
    # Restore by uncommenting the block below and removing this return.
    return scoped, meta

    # if "Month" not in scoped.columns:
    #     return scoped, meta
    #
    # curr_scope = scoped.filter(pl.col("Year") == curr_year)
    # if curr_scope.is_empty():
    #     return scoped, meta
    #
    # max_month = curr_scope.select(pl.col("Month").max()).to_series(0)[0]
    # if max_month is None:
    #     return scoped, meta
    #
    # meta["mtd_month_start"] = int(max_month)
    # meta["mtd_month_cutoff"] = int(max_month)
    #
    # has_day = "Day" in scoped.columns
    # max_day = None
    # if has_day:
    #     max_day = (
    #         curr_scope.filter(pl.col("Month") == pl.lit(max_month))
    #         .select(pl.col("Day").max())
    #         .to_series(0)[0]
    #     )
    #     if max_day is not None:
    #         meta["mtd_day_start"] = 1
    #         meta["mtd_day_cutoff"] = int(max_day)
    #
    # if has_day and max_day is not None:
    #     in_window = (
    #         (pl.col("Month") == pl.lit(max_month))
    #         & (pl.col("Day") >= pl.lit(1))
    #         & (pl.col("Day") <= pl.lit(max_day))
    #     )
    # else:
    #     in_window = pl.col("Month") == pl.lit(max_month)
    #
    # meta["mtd_applied"] = True
    # return scoped.filter(in_window), meta


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

    # Product enrichment: CAMPAIGN_NAME / MX_FLAGSHIP_S / URL 기반 PRODUCTS 열 추가.
    # 이후 PRODUCTS(enriched) → PRODUCT, PRODUCT(raw) → PRODUCT_RAW 로 swap.
    # PRODUCT_RAW는 DIMENSIONS에 없으므로 normalize 단계에서 자연 탈락.
    enriched_input = _create_products_column(scoped_input)
    if "PRODUCTS" in enriched_input.columns:
        enriched_input = enriched_input.rename({"PRODUCT": "PRODUCT_RAW", "PRODUCTS": "PRODUCT"})

    if set(ENGINE_COLUMNS).issubset(enriched_input.columns):
        meta = {
            "curr_year": curr_year,
            "prev_year": prev_year,
            "mtd_applied": False,
            "source_format": "wide",
        }
        meta.update(division_meta)
        meta.update(objective_meta)
        return _normalize_wide_engine_frame(enriched_input), meta

    long_df = _normalize_long_frame(enriched_input)
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
    # MTD month/day window is temporarily disabled for RAW calc output as well.
    # Restore by uncommenting the block below and removing this return.
    return scoped.drop("__calc_year"), meta

    # if not mtd_only:
    #     return scoped.drop("__calc_year"), meta
    #
    # month_column = "Month" if "Month" in scoped.columns else "MONTH" if "MONTH" in scoped.columns else None
    # if month_column is None:
    #     return scoped.drop("__calc_year"), meta
    #
    # scoped = scoped.with_columns(_int_expr(month_column).alias("__calc_month"))
    # curr_scope = scoped.filter(pl.col("__calc_year") == curr_year)
    # if curr_scope.is_empty():
    #     drop_columns = [column for column in ["__calc_year", "__calc_month"] if column in scoped.columns]
    #     return scoped.drop(*drop_columns), meta
    #
    # max_month = curr_scope.select(pl.col("__calc_month").max()).to_series(0)[0]
    # if max_month is None:
    #     drop_columns = [column for column in ["__calc_year", "__calc_month"] if column in scoped.columns]
    #     return scoped.drop(*drop_columns), meta
    #
    # meta["mtd_month_start"] = int(max_month)
    # meta["mtd_month_cutoff"] = int(max_month)
    #
    # day_column = "Day" if "Day" in scoped.columns else "DAY" if "DAY" in scoped.columns else None
    # max_day = None
    # if day_column is not None:
    #     scoped = scoped.with_columns(_int_expr(day_column).alias("__calc_day"))
    #     max_day = (
    #         curr_scope.with_columns(_int_expr(day_column).alias("__calc_day"))
    #         .filter(pl.col("__calc_month") == pl.lit(max_month))
    #         .select(pl.col("__calc_day").max())
    #         .to_series(0)[0]
    #     )
    #     if max_day is not None:
    #         meta["mtd_day_start"] = 1
    #         meta["mtd_day_cutoff"] = int(max_day)
    #
    # if day_column is not None and max_day is not None:
    #     in_window = (
    #         (pl.col("__calc_month") == pl.lit(max_month))
    #         & (pl.col("__calc_day") >= pl.lit(1))
    #         & (pl.col("__calc_day") <= pl.lit(max_day))
    #     )
    # else:
    #     in_window = pl.col("__calc_month") == pl.lit(max_month)
    #
    # filtered = scoped.filter(in_window)
    # meta["mtd_applied"] = True
    # drop_columns = [column for column in ["__calc_year", "__calc_month", "__calc_day"] if column in filtered.columns]
    # return filtered.drop(*drop_columns), meta


def build_html_calc_raw_sheets(
    path: str | Path,
    curr_year: int = DEFAULT_CURR_YEAR,
    prev_year: int = DEFAULT_PREV_YEAR,
    preferred_sheet: str = "raw",
    mtd_only: bool = True,
) -> tuple[Dict[str, pl.DataFrame], Dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    raw_parquet = _raw_parquet_path(source_path)
    if _raw_parquet_valid(raw_parquet, source_path):
        raw_df = _load_raw_parquet(raw_parquet)
    else:
        raw_df = _read_raw_input_excel_frame(source_path, preferred_sheet=preferred_sheet)
        _save_raw_parquet(raw_df, raw_parquet)
    raw_df = _filter_all_zero_raw_metric_rows(raw_df)

    filtered_objective_df, objective_meta = _filter_conversion_objective(raw_df)
    scoped_df, division_meta = _filter_target_divisions(filtered_objective_df)

    # enriched_df: PRODUCT=원본, PRODUCTS=추론값 (두 열 공존 — raw 출력용)
    # analysis_df: PRODUCTS→PRODUCT 스왑, PRODUCT→PRODUCT_RAW (엔진 분석용)
    enriched_df = _create_products_column(scoped_df)
    analysis_df = (
        enriched_df.rename({"PRODUCT": "PRODUCT_RAW", "PRODUCTS": "PRODUCT"})
        if "PRODUCTS" in enriched_df.columns
        else enriched_df
    )

    if set(ENGINE_COLUMNS).issubset(analysis_df.columns):
        normalized_df = _normalize_wide_engine_frame(analysis_df)
        engine_df = normalized_df
        long_meta: Dict[str, Any] = {
            "curr_year": curr_year,
            "prev_year": prev_year,
            "mtd_applied": False,
            "source_format": "wide",
        }
    else:
        long_df = _normalize_long_frame(analysis_df)
        if mtd_only:
            normalized_df, long_meta = _apply_mtd_alignment(long_df, curr_year=curr_year, prev_year=prev_year)
        else:
            normalized_df = _filter_target_years(long_df, curr_year=curr_year, prev_year=prev_year)
            long_meta = {"curr_year": curr_year, "prev_year": prev_year, "mtd_applied": False}
        long_meta["source_format"] = "long"
        engine_df = _pivot_long_to_engine(normalized_df, curr_year=curr_year, prev_year=prev_year)

    # raw_full_df는 enriched_df 기반 — PRODUCT(원본) + PRODUCTS(추론) 두 열 모두 출력
    raw_full_df, raw_full_meta = _filter_raw_frame_for_html_window(
        enriched_df,
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
    source_path = Path(path).expanduser().resolve()
    raw_df = _read_raw_input_excel_frame(source_path, preferred_sheet=preferred_sheet)

    # raw_df를 parquet으로 캐싱 — build_html_calc_raw_sheets 재읽기 제거
    _save_raw_parquet(raw_df, _raw_parquet_path(source_path))

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
    """Write multi-sheet Excel via xlsxwriter (single Workbook) — ~10x faster than openpyxl.

    94k-row sheet: openpyxl ~340s, xlsxwriter ~30s.
    """
    if not sheets:
        return False

    try:
        import xlsxwriter  # type: ignore
    except ImportError:
        return False

    first_df = next(iter(sheets.values()))
    if not hasattr(first_df, "write_excel"):
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: constant_memory=True는 polars write_excel()과 비호환.
        # polars가 내부적으로 add_table()을 호출하는데, xlsxwriter constant_memory 모드에서는
        # add_table()이 미지원이라 경고만 띄우고 데이터 rows를 통째로 누락시킴 → 6KB 빈 파일 생성.
        with xlsxwriter.Workbook(str(path)) as workbook:
            for sheet_name, frame in sheets.items():
                safe_name = str(sheet_name)[:31] or "Sheet1"
                frame.write_excel(
                    workbook=workbook,
                    worksheet=safe_name,
                    autofit=False,  # autofit scans every cell → skip for speed
                )
        return True
    except Exception:
        # Clean up partial file so openpyxl fallback starts clean
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass
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
    """Write output Excel via openpyxl (no auto-hyperlinks).

    NOTE: polars.write_excel() automatically converts "http://" cells to hyperlinks,
    which hits xlsxwriter's 65,530/sheet limit. openpyxl writes plain text safely.
    Performance: ~35s for 94k rows (acceptable for automation).
    """
    excel_path = Path(path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    _write_with_openpyxl(excel_path, sheets)
