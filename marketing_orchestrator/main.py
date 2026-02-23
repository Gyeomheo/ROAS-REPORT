"""Entrypoint for the ROAS reporting application use case."""

from __future__ import annotations

from src.application.report_service import run_reporting_pipeline


def main() -> None:
    run_reporting_pipeline()


if __name__ == "__main__":
    main()
