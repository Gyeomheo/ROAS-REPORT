"""Shared numeric/formatting utilities for reporting."""

from __future__ import annotations

from typing import Any

import polars as pl


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def safe_pct_change(curr: float | None, prev: float | None) -> float | None:
    if curr is None or prev is None or prev <= 0:
        return None
    return (curr - prev) / prev


def safe_ratio_expr(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    safe_den = pl.when(den > 0).then(den).otherwise(None)
    return num / safe_den


def safe_pct_change_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
    safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
    return (curr - safe_prev) / safe_prev


def safe_log_ratio_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
    safe_curr = pl.when(curr > 0).then(curr).otherwise(None)
    safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
    return (safe_curr / safe_prev).log()


def log_mean_expr(curr: pl.Expr, prev: pl.Expr) -> pl.Expr:
    safe_curr = pl.when(curr > 0).then(curr).otherwise(None)
    safe_prev = pl.when(prev > 0).then(prev).otherwise(None)
    safe_den = pl.when(safe_curr != safe_prev).then(safe_curr.log() - safe_prev.log()).otherwise(None)
    return (
        pl.when(safe_curr.is_null() | safe_prev.is_null())
        .then(None)
        .when(safe_curr == safe_prev)
        .then(safe_curr)
        .otherwise((safe_curr - safe_prev) / safe_den)
    )


def fmt_money(value: float | None) -> str:
    if value is None:
        return "$0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${abs_value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"${abs_value / 1_000:.0f}K"
    return f"${abs_value:.0f}"


def fmt_money_signed(value: float | None) -> str:
    if value is None:
        return "$0"
    sign = "-" if value < 0 else "+"
    return f"{sign}{fmt_money(abs(value))}"


def fmt_pct(value: float | None, signed: bool = True) -> str:
    if value is None:
        return "N/A"
    pct = value * 100
    if signed:
        return f"{pct:+.0f}%".replace("+", "")
    return f"{pct:.0f}%"


def fmt_pct_yoy(value: float | None, signed: bool = True) -> str:
    base = fmt_pct(value, signed=signed)
    if base == "N/A":
        return base
    return f"{base} YoY"


def fmt_roas(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def fmt_pp(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:+.0f}%p"


def fmt_driver_metric(metric: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if metric == "CPC":
        return f"${value:.2f}"
    if metric == "CVR":
        return f"{value * 100:.2f}%"
    if metric == "AOV":
        return fmt_money(value)
    return f"{value:.3f}"


def trend(value: float | None, eps: float = 1e-9) -> str:
    if value is None:
        return "unknown"
    if value > eps:
        return "up"
    if value < -eps:
        return "down"
    return "flat"


def direct_contribution_pct(child_impact: float, parent_delta: float, mode_tag: str) -> float | None:
    parent_abs = abs(parent_delta)
    if parent_abs <= 0:
        return None
    mode = (mode_tag or "").upper()
    if mode == "ISSUE":
        return max(0.0, -child_impact) / parent_abs
    if mode in {"IMPROVE", "DEFENSE"}:
        return max(0.0, child_impact) / parent_abs
    return abs(child_impact) / parent_abs

