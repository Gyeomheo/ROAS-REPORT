"""Entrypoint for the ROAS reporting application use case."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.application.cli import select_excel_input_path
from src.application.report_service import run_cleansing_pipeline, run_reporting_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly ROAS reporting pipeline.")
    parser.add_argument(
        "--mode",
        choices=["full", "cleanse", "analyze"],
        default="full",
        help=(
            "full: cleanse and analyze in one run; "
            "cleanse: create GMPD RAW_Cleaned only; "
            "analyze: run issue analysis from a cleansed workbook"
        ),
    )
    parser.add_argument("--input-path", type=Path, default=None, help="Optional Excel input path. If omitted, a file picker opens.")
    parser.add_argument("--curr-year", type=int, default=None, help="Current comparison year (e.g. 2026).")
    parser.add_argument("--prev-year", type=int, default=None, help="Previous comparison year (e.g. 2025).")
    args = parser.parse_args()
    input_path = args.input_path if args.input_path is not None else select_excel_input_path()
    if args.mode == "cleanse":
        run_cleansing_pipeline(input_path=input_path, curr_year=args.curr_year, prev_year=args.prev_year)
        return

    if args.mode == "analyze":
        run_reporting_pipeline(
            input_path=input_path,
            curr_year=args.curr_year,
            prev_year=args.prev_year,
            input_sheet="engine_html_input",
            save_cleansed_workbook=False,
        )
        return

    run_reporting_pipeline(input_path=input_path, curr_year=args.curr_year, prev_year=args.prev_year)


if __name__ == "__main__":
    main()
