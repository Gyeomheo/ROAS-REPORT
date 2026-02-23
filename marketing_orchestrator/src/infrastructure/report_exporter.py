"""Infrastructure adapter for summary export targets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from src.reporting import write_html_report


def save_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def save_summary_html(path: Path, summary: dict[str, Any], df: pl.DataFrame) -> None:
    write_html_report(path, summary, df)
