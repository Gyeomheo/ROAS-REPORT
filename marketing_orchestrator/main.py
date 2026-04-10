"""Entrypoint for the ROAS reporting application use case."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.application.report_service import run_reporting_pipeline


def _select_input_path() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tkinter is required to choose an input Excel file.") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    try:
        selected = filedialog.askopenfilename(
            title="Select Weekly ROAS input file",
            filetypes=[
                ("Excel files", "*.xlsx *.xlsm *.xls"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()

    if not selected:
        raise SystemExit("Input file selection cancelled.")
    return Path(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run weekly ROAS reporting pipeline.")
    parser.add_argument("--input-path", type=Path, default=None, help="Optional Excel input path. If omitted, a file picker opens.")
    parser.add_argument("--curr-year", type=int, default=None, help="Current comparison year (e.g. 2026).")
    parser.add_argument("--prev-year", type=int, default=None, help="Previous comparison year (e.g. 2025).")
    args = parser.parse_args()
    input_path = args.input_path if args.input_path is not None else _select_input_path()
    run_reporting_pipeline(input_path=input_path, curr_year=args.curr_year, prev_year=args.prev_year)


if __name__ == "__main__":
    main()
