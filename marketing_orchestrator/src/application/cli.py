"""Shared CLI helpers for local entrypoints."""

from __future__ import annotations

from pathlib import Path


def select_excel_input_path(title: str = "Select Weekly ROAS input file") -> Path:
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
            title=title,
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
