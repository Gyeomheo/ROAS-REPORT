"""Entrypoint for generating GMPD RAW_Cleaned only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.application.cli import select_excel_input_path
from src.application.report_service import run_cleansing_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GMPD RAW_Cleaned workbook only.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional source Excel path. If omitted, a file picker opens.",
    )
    parser.add_argument("--curr-year", type=int, default=None, help="Current comparison year (e.g. 2026).")
    parser.add_argument("--prev-year", type=int, default=None, help="Previous comparison year (e.g. 2025).")
    args = parser.parse_args()

    input_path = args.input_path if args.input_path is not None else select_excel_input_path()
    run_cleansing_pipeline(input_path=input_path, curr_year=args.curr_year, prev_year=args.prev_year)


if __name__ == "__main__":
    main()
