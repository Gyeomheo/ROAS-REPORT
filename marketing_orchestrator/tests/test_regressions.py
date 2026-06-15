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
import src.application.reporting.selectors as selectors
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
                "SUBSIDIARY": ["SEAU", "SEC", "SEAU", "SEAU"],
                "CHANNEL": ["SEARCH", "SEARCH", "SEARCH", "SEARCH"],
                "DIVISION": ["MX", "MX", "MX", "MX"],
                "PRODUCT": ["S SERIES", "S SERIES", "S SERIES", "S SERIES"],
                "PLATFORM": ["TIKTOK", "GOOGLE", "GOOGLE", "META"],
                "Year": [2026, 2026, 2026, 2026],
                "PLATFORM_SPEND_USD": ["120.5", "140.0", "90.0", "200.0"],
                "PLATFORM_REVENUE_USD": ["450.0", "300.0", "210.0", "600.0"],
                "GROSS_REVENUE": ["800.0", "500.0", "999.0", "1100.0"],
                "PLATFORM_CLICKS": ["80", "100", "50", "70"],
                "GROSS_ORDERS": ["9", "7", "5", "11"],
            }
        )

        normalized = _normalize_long_frame(df)

        self.assertEqual(normalized.height, 4)
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

        meta_row = rows[("SEAU", 200.0)]
        self.assertEqual(meta_row["Revenue"], 1100.0)
        self.assertEqual(meta_row["Ext Revenue"], 1100.0)
        self.assertEqual(meta_row["Orders"], 11.0)

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

        with patch("src.ingestion.pl.read_excel", side_effect=_fake_read_polars):
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

    def test_issue_candidate_gate_passes_on_roas_yoy_drop(self) -> None:
        rows = [
            {
                "diagnosis_subsidiary": "SEC",
                "path": {"CHANNEL": "SEARCH", "DIVISION": "MX", "PRODUCT": "S SERIES"},
                "topic_roas_curr": 0.8,
                "topic_roas_prev": 1.0,  # -20%
                "impact_contribution_pct": 0.02,
            },
            {
                "diagnosis_subsidiary": "SEC",
                "path": {"CHANNEL": "PMAX", "DIVISION": "MX", "PRODUCT": "A SERIES"},
                "topic_roas_curr": 0.99,
                "topic_roas_prev": 1.0,  # -1%
                "impact_contribution_pct": 0.02,
            },
        ]
        result = selectors.top3_by_subsidiary(rows, descending=True, issue_candidate_gate=True)
        self.assertEqual(len(result["SEC"]), 1)
        self.assertEqual(result["SEC"][0]["path"]["CHANNEL"], "SEARCH")

    def test_issue_candidate_gate_passes_on_revenue_drop_with_nonnegative_spend_yoy(self) -> None:
        rows = [
            {
                "diagnosis_subsidiary": "SEAU",
                "path": {"CHANNEL": "SOCIAL", "DIVISION": "MX", "PRODUCT": "S SERIES"},
                "Revenue_curr_sum": 70.0,
                "Revenue_prev_sum": 100.0,  # -30%
                "Spend_curr_sum": 120.0,
                "Spend_prev_sum": 100.0,  # +20%
                "impact_contribution_pct": 0.03,
            }
        ]
        result = selectors.top3_by_subsidiary(rows, descending=True, issue_candidate_gate=True)
        self.assertEqual(len(result["SEAU"]), 1)

    def test_issue_candidate_gate_passes_on_impact_contribution(self) -> None:
        rows = [
            {
                "diagnosis_subsidiary": "SEDA",
                "path": {"CHANNEL": "SEARCH", "DIVISION": "DA", "PRODUCT": "WASHER"},
                "impact_contribution_pct": 0.12,
                "topic_roas_curr": 1.2,
                "topic_roas_prev": 1.0,
                "Revenue_curr_sum": 120000.0,
                "Revenue_prev_sum": 100000.0,
                "Spend_curr_sum": 50000.0,
                "Spend_prev_sum": 40000.0,
                "tag": "NORMAL",
            },
            {
                "diagnosis_subsidiary": "SEDA",
                "path": {"CHANNEL": "PMAX", "DIVISION": "DA", "PRODUCT": "DRYER"},
                "impact_contribution_pct": 0.08,
            },
        ]
        result = selectors.top3_by_subsidiary(rows, descending=True, issue_candidate_gate=True)
        self.assertEqual(len(result["SEDA"]), 1)
        self.assertEqual(result["SEDA"][0]["path"]["PRODUCT"], "WASHER")

    def test_issue_candidate_gate_excludes_low_vol_rows_even_with_high_impact(self) -> None:
        rows = [
            {
                "diagnosis_subsidiary": "SEC",
                "path": {"CHANNEL": "SEARCH", "DIVISION": "MX", "PRODUCT": "MX OTHERS"},
                "impact_contribution_pct": 1.5,
                "tag": "LOW_VOL",
                "topic_roas_curr": None,
                "topic_roas_prev": 30.0,
                "Revenue_curr_sum": 0.0,
                "Revenue_prev_sum": 200000.0,
                "Spend_curr_sum": 0.0,
                "Spend_prev_sum": 60000.0,
            }
        ]
        result = selectors.top3_by_subsidiary(rows, descending=True, issue_candidate_gate=True)
        self.assertEqual(len(result["SEC"]), 0)

    def test_choose_subsidiary_mode_uses_issue_threshold(self) -> None:
        issues_by_sub = {"SEC": [{"topic": "issue"}]}
        improves_by_sub = {"SEC": [{"topic": "improve"}]}
        perf_small_drop = {"roas_delta": -0.001}
        perf_large_drop = {"roas_delta": -0.05}

        with patch("src.application.reporting.selectors.ISSUE_MODE_ROAS_DELTA_THRESHOLD", -0.03):
            mode_small, selected_small = selectors.choose_subsidiary_mode(
                "SEC",
                perf_small_drop,
                issues_by_sub,
                improves_by_sub,
            )
            mode_large, selected_large = selectors.choose_subsidiary_mode(
                "SEC",
                perf_large_drop,
                issues_by_sub,
                improves_by_sub,
            )

        self.assertEqual(mode_small, "IMPROVE")
        self.assertEqual(selected_small, improves_by_sub["SEC"])
        self.assertEqual(mode_large, "ISSUE")
        self.assertEqual(selected_large, issues_by_sub["SEC"])


    def test_products_column_infers_new_cn_campaign_codes(self) -> None:
        df = pl.DataFrame(
            {
                "PRODUCT": ["OTHERS", "DA CROSS PRODUCTS", None, "MX OTHERS", "OTHERS", "OTHERS", "OTHERS"],
                "CAMPAIGN_NAME": [
                    "SEC_CN~arc_launch",
                    "SEC_CN~acl_alwayson",
                    "SEC_CN~ard_promo",
                    "SEC_CN~jbl_speaker",
                    "SEC_CN~ebdswp_q2",
                    "SEC_CN~emvip_sale",
                    "SEC_CN~qoop_kitchen",
                ],
            }
        )

        enriched = ingestion._create_products_column(df)

        self.assertEqual(
            enriched["PRODUCTS"].to_list(),
            [
                "AIR CONDITIONER",
                "AIR PURIFIER",
                "AIR DRESSER/SHOE DRESSER",
                "HARMAN",
                "DISHWASHER",
                "LIFESTYLE TV",
                "MICROWAVE/OTR/QOOKER",
            ],
        )

    def test_products_column_keeps_existing_cn_e_campaign_codes(self) -> None:
        df = pl.DataFrame(
            {
                "PRODUCT": ["OTHERS"],
                "CAMPAIGN_NAME": ["SEC_CN~Ekrfg_launch"],
            }
        )

        enriched = ingestion._create_products_column(df)

        self.assertEqual(enriched["PRODUCTS"].to_list(), ["REFRIGERATOR"])

    def test_products_column_covers_sec_campaign_name_workbook_codes(self) -> None:
        codes = [
            "bsmartm", "bsmartp", "ebaclm", "ebaclp", "ebarcm", "ebarcp", "ebardm", "ebardp",
            "ebcarem", "ebcarep", "ebdocm3", "ebdocp3", "ebdrym", "ebdryp", "ebdswm", "ebdswp",
            "ebhmkm", "ebhmkp", "ebindm", "ebindp", "ebkchm", "ebkchp", "ebltvm", "ebltvp",
            "ebmonm", "ebmonp", "ebmvim", "ebmvip", "ebppcm", "ebppcp", "ebprim", "ebprip",
            "ebps6m", "ebps6p", "ebpz7m1", "ebpz7p1", "ebqoom", "ebqoop", "ebrfgm", "ebrfgp",
            "ebsubm", "ebsubp", "ebtapm", "ebtapp", "ebttvm", "ebttvp", "ebuhdm", "ebuhdp",
            "ebvcum", "ebvcup", "ebweam", "ebweap", "ebwpfm", "ebwpfp", "ebwshm", "ebwshp",
            "ef1h26", "ekacl", "ekarc", "ekard", "ekbud4", "ekcare", "ekdry", "ekdsw",
            "ekfit", "ekhmk", "ekind", "ekjbl", "ekkch", "ekltv", "ekmni", "ekppc6",
            "ekpri", "ekpz7", "ekrfg", "eksub", "ekta11", "ekttv", "ekvcu", "ekwat8", "ekwsh",
        ]
        df = pl.DataFrame(
            {
                "PRODUCT": ["OTHERS"] * len(codes),
                "CAMPAIGN_NAME": [f"SEC_CN~{code}_BS~x" for code in codes],
            }
        )

        enriched = ingestion._create_products_column(df)

        self.assertNotIn("OTHERS", enriched["PRODUCTS"].to_list())

    def test_filter_all_zero_raw_metric_rows_drops_only_zero_activity_rows(self) -> None:
        metric_columns = list(ingestion.RAW_ZERO_ACTIVITY_METRIC_COLUMNS)
        df = pl.DataFrame(
            {
                "CAMPAIGN_NAME": ["all_zero", "has_spend", "has_parse_error"],
                **{column: [0, 0, 0] for column in metric_columns},
            }
        ).with_columns(
            pl.when(pl.col("CAMPAIGN_NAME") == "has_spend")
            .then(pl.lit(12.5))
            .otherwise(pl.col("PLATFORM_SPEND_USD"))
            .alias("PLATFORM_SPEND_USD"),
            pl.when(pl.col("CAMPAIGN_NAME") == "has_parse_error")
            .then(pl.lit("not-a-number"))
            .otherwise(pl.col("PLATFORM_REVENUE_USD").cast(pl.Utf8))
            .alias("PLATFORM_REVENUE_USD"),
        )

        filtered = ingestion._filter_all_zero_raw_metric_rows(df)

        self.assertEqual(filtered["CAMPAIGN_NAME"].to_list(), ["has_spend", "has_parse_error"])


if __name__ == "__main__":
    unittest.main()
