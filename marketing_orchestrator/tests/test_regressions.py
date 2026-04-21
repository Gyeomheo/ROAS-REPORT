from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

import src.ingestion as ingestion
from src.application.campaign_action_service import enrich_campaign_actions
from src.application.report_service import _build_run_metadata, _insight_text
from src.ingestion import _normalize_long_frame, _normalize_wide_engine_frame


class WeeklyRoasRegressionTests(unittest.TestCase):
    def test_special_tag_preserves_none_driver(self) -> None:
        row = {
            "SUBSIDIARY": "SEAU",
            "CHANNEL": "SEARCH",
            "DIVISION": "MX",
            "PRODUCT": "S SERIES",
            "mode_tag": "ISSUE",
            "tag": "NEW",
            "primary_driver": "NONE",
            "dlog_CVR": -1.2,
            "dlog_AOV": 0.2,
            "dlog_CPC": 0.7,
        }

        enriched = enrich_campaign_actions(
            [row],
            dimensions=["SUBSIDIARY", "CHANNEL", "DIVISION", "PRODUCT"],
        )[0]

        self.assertEqual(enriched["primary_driver"], "NONE")
        self.assertIn("신규 캠페인", enriched["recommended_actions"])

    def test_wide_engine_frame_keeps_zero_current_spend_gone_candidate(self) -> None:
        df = pl.DataFrame(
            {
                "SUBSIDIARY": ["SEAU"],
                "CHANNEL": ["SEARCH"],
                "DIVISION": ["MX"],
                "PRODUCT": ["S SERIES"],
                "Spend_curr": [0.0],
                "Spend_prev": [120.0],
                "Revenue_curr": [0.0],
                "Revenue_prev": [500.0],
                "Clicks_curr": [0.0],
                "Clicks_prev": [100.0],
                "Orders_curr": [0.0],
                "Orders_prev": [12.0],
            }
        )

        normalized = _normalize_wide_engine_frame(df)

        self.assertEqual(normalized.height, 1)

    def test_long_engine_frame_applies_ext_revenue_rules_and_gross_orders(self) -> None:
        df = pl.DataFrame(
            {
                "SUBSIDIARY": ["SEAU", "SEC", "SEAU"],
                "CHANNEL": ["SEARCH", "SEARCH", "SEARCH"],
                "DIVISION": ["MX", "MX", "MX"],
                "PRODUCT": ["S SERIES", "S SERIES", "S SERIES"],
                "PLATFORM": ["TIKTOK", "GOOGLE", "GOOGLE"],
                "Year": [2026, 2026, 2026],
                "PLATFORM_SPEND_USD": ["120.5", "140.0", "90.0"],
                "PLATFORM_REVENUE_USD": ["450.0", "300.0", "210.0"],
                "GROSS_REVENUE": ["800.0", "500.0", "999.0"],
                "PLATFORM_CLICKS": ["80", "100", "50"],
                "GROSS_ORDERS": ["9", "7", "5"],
            }
        )

        normalized = _normalize_long_frame(df)

        self.assertEqual(normalized.height, 3)
        rows = {(row["SUBSIDIARY"], row["Spend"]): row for row in normalized.to_dicts()}

        tiktok_row = rows[("SEAU", 120.5)]
        self.assertEqual(tiktok_row["Spend"], 120.5)
        self.assertEqual(tiktok_row["Revenue"], 800.0)
        self.assertEqual(tiktok_row["Ext Revenue"], 800.0)
        self.assertEqual(tiktok_row["Orders"], 9.0)

        sec_row = rows[("SEC", 140.0)]
        self.assertEqual(sec_row["Revenue"], 500.0)
        self.assertEqual(sec_row["Ext Revenue"], 500.0)
        self.assertEqual(sec_row["Orders"], 7.0)

        default_row = rows[("SEAU", 90.0)]
        self.assertEqual(default_row["Revenue"], 210.0)
        self.assertEqual(default_row["Ext Revenue"], 210.0)
        self.assertEqual(default_row["Orders"], 5.0)

    def test_insight_text_uses_clean_korean_strings(self) -> None:
        row = {
            "SUBSIDIARY": "SEAU",
            "CHANNEL": "SEARCH",
            "DIVISION": "MX",
            "PRODUCT": "S SERIES",
            "mode_tag": "ISSUE",
            "tag": "NORMAL",
            "primary_driver": "CVR",
            "Revenue_curr_sum": 150000.0,
            "Revenue_prev_sum": 300000.0,
            "Spend_curr_sum": 62000.0,
            "Spend_prev_sum": 124000.0,
            "Clicks_curr_sum": 5000.0,
            "Clicks_prev_sum": 10000.0,
            "Orders_curr_sum": 35.0,
            "Orders_prev_sum": 140.0,
            "CVR_curr": 0.007,
            "CVR_prev": 0.014,
            "CPC_curr": 12.4,
            "CPC_prev": 12.4,
            "AOV_curr": 4285.714,
            "AOV_prev": 2142.857,
            "dlog_CVR": -1.263,
            "dlog_AOV": 0.061,
            "dlog_CPC": 0.149,
            "Impact_adj": -1.0,
            "drill_trace": {"events": []},
        }
        channel_map = {
            ("SEAU", "SEARCH"): {
                "Spend_curr_sum": 196000.0,
                "Spend_prev_sum": 446000.0,
                "Revenue_curr_sum": 484000.0,
                "Revenue_prev_sum": 3000000.0,
            }
        }
        division_map = {
            ("SEAU", "SEARCH", "MX"): {
                "Spend_curr_sum": 169000.0,
                "Spend_prev_sum": 425000.0,
                "Revenue_curr_sum": 413000.0,
                "Revenue_prev_sum": 2900000.0,
                "Clicks_curr_sum": 7500.0,
                "Clicks_prev_sum": 16000.0,
                "Orders_curr_sum": 53.0,
                "Orders_prev_sum": 390.0,
            }
        }
        subsidiary_map = {
            "SEAU": {
                "Spend_curr_sum": 278000.0,
                "Spend_prev_sum": 541000.0,
                "Revenue_curr_sum": 733000.0,
                "Revenue_prev_sum": 3260000.0,
            }
        }

        summary_text, detail_text = _insight_text(
            row,
            channel_map=channel_map,
            division_map=division_map,
            subsidiary_map=subsidiary_map,
            impact_contribution_pct=0.87,
        )

        combined = f"{summary_text}\n{detail_text}"
        self.assertIn("매출", combined)
        self.assertIn("채널 SEARCH", combined)
        self.assertNotIn("癲", combined)
        self.assertNotIn("???", combined)

    def test_build_run_metadata_includes_hash_and_cutoff_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.xlsx"
            input_path.write_bytes(b"weekly-roas")

            metadata = _build_run_metadata(
                input_path,
                current_year=2026,
                comparison_meta={"mtd_month_cutoff": 2, "mtd_day_cutoff": 20},
            )

        self.assertEqual(metadata["week_key"], "2026-02-20")
        self.assertEqual(metadata["source_hash"], hashlib.sha1(b"weekly-roas").hexdigest())
        self.assertIn("T", metadata["generated_at"])

    def test_build_run_metadata_uses_metadata_hash_when_file_open_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.xlsx"
            input_path.write_bytes(b"weekly-roas")
            stat = input_path.stat()
            fallback_seed = (
                f"{input_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{stat.st_ctime_ns}"
            )
            expected_hash = hashlib.sha1(fallback_seed.encode("utf-8", errors="replace")).hexdigest()

            with patch("pathlib.Path.open", side_effect=PermissionError("locked")):
                metadata = _build_run_metadata(
                    input_path,
                    current_year=2026,
                    comparison_meta={"mtd_month_cutoff": 2, "mtd_day_cutoff": 20},
                )

        self.assertEqual(metadata["week_key"], "2026-02-20")
        self.assertEqual(metadata["source_hash"], expected_hash)
        self.assertIn("T", metadata["generated_at"])

    def test_read_excel_polars_uses_single_engine_call_with_schema_overrides(self) -> None:
        calls: list[dict[str, object]] = []

        def _fake_read_excel(_path: Path, **kwargs: object) -> pl.DataFrame:
            calls.append(kwargs)
            return pl.DataFrame({"SUBSIDIARY": ["SEAU"]})

        with patch("src.ingestion.EXCEL_ENGINE_CANDIDATES", ("calamine",)):
            with patch("src.ingestion.pl.read_excel", side_effect=_fake_read_excel):
                frame = ingestion._read_excel_polars(
                    Path("dummy.xlsx"),
                    sheet_name="raw",
                    engine="calamine",
                    columns=["ACCOUNT", "ACCOUNT_ID"],
                )

        self.assertEqual(frame.height, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].get("engine"), "calamine")
        self.assertIn("schema_overrides", calls[0])

    def test_read_with_polars_avoids_column_set_retry_loop(self) -> None:
        calls: list[dict[str, object]] = []

        def _fake_read_polars(_path: Path, **kwargs: object) -> pl.DataFrame:
            calls.append(kwargs)
            # Unsupported schema -> force fallback path after bounded attempts.
            return pl.DataFrame({"UNKNOWN_COL": ["x"]})

        with patch("src.ingestion.pl.read_excel", side_effect=_fake_read_excel):
            with patch("src.ingestion.EXCEL_ENGINE_CANDIDATES", ("calamine",)):
                with self.assertRaises(ValueError):
                    ingestion._read_with_polars(
                        Path("dummy.xlsx"),
                        preferred_sheet="raw",
                        target_sheet="another",
                    )

        # one read per sheet (another, raw) + one final first-sheet fallback
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call.get("engine") == "calamine" for call in calls))
        self.assertTrue(all("columns" not in call for call in calls))


if __name__ == "__main__":
    unittest.main()


