"""Infrastructure adapter for Excel-based data repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from src.ingestion import read_input_excel, write_output_excel

CACHE_SCHEMA_VERSION = 1
CACHE_MTD_ONLY = True
CACHE_PREFERRED_SHEET = "raw"


def _project_root_from_input(path: Path) -> Path:
    resolved = path.resolve()
    if len(resolved.parents) >= 3:
        return resolved.parents[2]
    return resolved.parent


def _cache_paths(path: Path) -> tuple[Path, Path]:
    project_root = _project_root_from_input(path)
    cache_dir = project_root / "output" / ".cache"
    path_hash = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return (
        cache_dir / f"engine_input_{path_hash}.parquet",
        cache_dir / f"engine_input_{path_hash}.meta.json",
    )


def _cache_key(path: Path, curr_year: int, prev_year: int) -> dict[str, Any]:
    stat = path.stat()
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "input_path": str(path.resolve()),
        "input_mtime_ns": stat.st_mtime_ns,
        "input_size": stat.st_size,
        "curr_year": curr_year,
        "prev_year": prev_year,
        "mtd_only": CACHE_MTD_ONLY,
        "preferred_sheet": CACHE_PREFERRED_SHEET,
    }


def _load_cache(
    frame_path: Path,
    meta_path: Path,
    expected_key: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]] | None:
    if not frame_path.exists() or not meta_path.exists():
        return None
    try:
        meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta_obj, dict):
            return None
        cache_key = meta_obj.get("cache_key")
        comparison_meta = meta_obj.get("comparison_meta")
        if not isinstance(cache_key, dict) or cache_key != expected_key:
            return None
        if not isinstance(comparison_meta, dict):
            return None
        return pl.read_parquet(frame_path), comparison_meta
    except Exception:
        return None


def _save_cache(
    frame_path: Path,
    meta_path: Path,
    frame: pl.DataFrame,
    cache_key: dict[str, Any],
    comparison_meta: dict[str, Any],
) -> None:
    try:
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(frame_path, compression="zstd")
        payload = {
            "cache_key": cache_key,
            "comparison_meta": comparison_meta,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Cache write failures should not break report generation.
        return


def load_input_frame(path: Path, curr_year: int, prev_year: int) -> tuple[pl.DataFrame, dict[str, Any]]:
    key = _cache_key(path, curr_year=curr_year, prev_year=prev_year)
    frame_cache_path, meta_cache_path = _cache_paths(path)
    cached = _load_cache(frame_cache_path, meta_cache_path, key)
    if cached is not None:
        cached_frame, cached_meta = cached
        runtime_meta = dict(cached_meta)
        runtime_meta["input_cache_hit"] = True
        return cached_frame, runtime_meta

    loaded = read_input_excel(
        path,
        preferred_sheet=CACHE_PREFERRED_SHEET,
        curr_year=curr_year,
        prev_year=prev_year,
        mtd_only=CACHE_MTD_ONLY,
        return_meta=True,
    )
    if isinstance(loaded, tuple):
        frame, comparison_meta = loaded
        comparison_meta = dict(comparison_meta)
        comparison_meta["input_cache_hit"] = False
        _save_cache(frame_cache_path, meta_cache_path, frame=frame, cache_key=key, comparison_meta=comparison_meta)
        return frame, comparison_meta

    comparison_meta = {
        "curr_year": curr_year,
        "prev_year": prev_year,
        "mtd_applied": False,
        "input_cache_hit": False,
    }
    _save_cache(frame_cache_path, meta_cache_path, frame=loaded, cache_key=key, comparison_meta=comparison_meta)
    return loaded, comparison_meta


def save_output_workbook(path: Path, sheets: dict[str, pl.DataFrame]) -> tuple[bool, str]:
    try:
        write_output_excel(path, sheets)
    except PermissionError as exc:
        return False, str(exc)
    return True, ""
