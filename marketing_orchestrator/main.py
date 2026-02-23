"""Entrypoint for the ROAS reporting application use case."""

from __future__ import annotations

import argparse

from src.application.report_service import run_reporting_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly ROAS reporting pipeline.")
    parser.add_argument("--curr-year", type=int, default=None, help="Current comparison year (e.g. 2026).")
    parser.add_argument("--prev-year", type=int, default=None, help="Previous comparison year (e.g. 2025).")
    args = parser.parse_args()
    run_reporting_pipeline(curr_year=args.curr_year, prev_year=args.prev_year)


if __name__ == "__main__":
    main()
