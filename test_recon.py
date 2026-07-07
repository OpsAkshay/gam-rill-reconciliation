"""Tests grounded in the four real sample exports in the repo root.

Run: .venv/bin/python -m pytest test_recon.py -q
"""

from pathlib import Path

import pandas as pd
import pytest

import recon

ROOT = Path(__file__).parent
GB_GAM = ROOT / "GBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_GAM = ROOT / "wBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165350.csv"
GB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165704.csv"

@pytest.fixture(scope="module")
def pairs():
    # Sample exports hold real publisher data and are intentionally not
    # committed — tests needing them skip elsewhere; synthetic tests still run.
    if not all(p.exists() for p in (GB_GAM, WB_GAM, WB_RILL, GB_RILL)):
        pytest.skip("sample export CSVs not present (kept out of git)")
    gb = recon.SitePair(
        "GB News",
        recon.parse_gam(GB_GAM.read_bytes()),
        recon.parse_rill(GB_RILL.read_bytes()),
    )
    wb = recon.SitePair(
        "WeatherBug",
        recon.parse_gam(WB_GAM.read_bytes()),
        recon.parse_rill(WB_RILL.read_bytes()),
    )
    return [gb, wb]


# ── FORMAT DETECTION ──────────────────────────────────────────────────────────

def test_detects_opportunities_metric(pairs):
    for p in pairs:
        assert p.gam.metrics == ["opportunities"]
        assert p.rill.metrics == ["opportunities"]
        assert p.metrics == ["opportunities"]


def test_samples_have_no_date_or_source_group(pairs):
    for p in pairs:
        assert p.dims == set()


def test_unknown_file_rejected_with_clear_error():
    junk = b"foo,bar\n1,2\n"
    with pytest.raises(ValueError, match="GAM"):
        recon.parse_gam(junk)
    with pytest.raises(ValueError, match="Rill"):
        recon.parse_rill(junk)


def test_full_revenue_format_still_parses():
    gam_csv = (
        "Date,Line item type,Order,Ad unit (all levels),Ad unit code,"
        "Ad unit code level 1,Ad unit code level 2,"
        "Total impressions,Total CPM and CPC revenue\n"
        'Jun 26 2026,Price priority,SomeOrder,GB News - Home » MPU_1,mpu_1,'
        'gbnews_home,mpu_1,"1,234",$56.78\n'
        "Jun 26 2026,,,GB News - Home » MPU_2,mpu_2,gbnews_home,mpu_2,10,$1.00\n"
    ).encode()
    rep = recon.parse_gam(gam_csv)
    assert set(rep.metrics) == {"impressions", "revenue"}
    assert rep.dims == {"date", "source_group"}
    assert rep.df["impressions"].sum() == 1244
    # blank type+order → OB
    assert (rep.df["line_item_type"] == "OB").sum() == 1

    rill_csv = (
        "Ts (day),Domain,Revenue Source Type,Ad Unit,Total Impressions,Revenue\n"
        "2026-06-26,gbnews.com,Prebid,/22414545390/gbnews_home/mpu_1,1200,55.0\n"
    ).encode()
    rrep = recon.parse_rill(rill_csv)
    assert set(rrep.metrics) == {"impressions", "revenue"}
    assert rrep.dims == {"date", "source_group"}
    assert rrep.df["key_path"].iloc[0] == "gbnews_home/mpu_1"


# ── PATH JOIN (the acid test: 100% match on both real publishers) ─────────────

def test_every_rill_path_matches_gam(pairs):
    for p in pairs:
        gam_keys = set(p.gam.df["key_path"])
        rill = p.rill.df[~p.rill.df["is_others"]]
        unmatched = set(rill["key_path"]) - gam_keys
        assert unmatched == set(), f"{p.name}: {sorted(unmatched)[:5]}"


def test_rill_id_prefix_with_and_without_leading_slash():
    assert recon.rill_key_path("/22414545390/gbnews_home/mpu_1") == "gbnews_home/mpu_1"
    assert recon.rill_key_path("65299053/weatherbug.com/Alerts/x") == "weatherbug.com/alerts/x"
    assert recon.rill_key_path("Others") == ""
    assert recon.rill_key_path(None) == ""


# ── PROPERTY CLASSIFICATION ───────────────────────────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("gbnews_celebrity",                  "Web"),
    ("weatherbug.com",                    "Web"),
    ("gbnews_app_android",                "App — Android"),
    ("WB_Android_App_Phone",              "App — Android"),
    ("wb_android_app_tablet",             "App — Android Tablet"),
    ("WB_iOS_App_Phone",                  "App — iOS"),
    ("wb_ios_app_tablet",                 "App — iOS Tablet"),
    ("gbnews_video",                      "Video"),
    ("Primis_Video_Android",              "Video"),   # video wins over android
    ("ca-mb-app-pub-8015868500526768-tag", "Ad Exchange In-App"),  # AdX wins over 'app'
])
def test_property_rules(code, expected):
    assert recon.classify_property(code) == expected


def test_gb_is_one_site_many_properties(pairs):
    gb = pairs[0]
    meta = recon.build_unit_meta(gb)
    assert meta.loc["gbnews_celebrity", "property"] == "Web"
    assert meta.loc["gbnews_app_ios", "property"] == "App — iOS"
    assert meta.loc["primis_video_ios", "property"] == "Video"
    # brand prefix stripped for section display
    assert meta.loc["gbnews_celebrity", "unit_display"] == "Celebrity"
    assert meta.loc["gbnews_money", "unit_display"] == "Money"


# ── SECTION EXTRACTION ────────────────────────────────────────────────────────

def test_sections(pairs):
    gb, wb = pairs
    g_gb, r_gb = recon.enrich(gb)
    # GB flat tree: unit itself is the section, brand-stripped
    row = g_gb[g_gb["key_path"] == "gbnews_celebrity/mpu_1"].iloc[0]
    assert (row["property"], row["section"], row["slot"]) == ("Web", "Celebrity", "mpu_1")

    g_wb, r_wb = recon.enrich(wb)
    # WB nested tree: middle segment is the section
    row = g_wb[g_wb["key_path"] == "weatherbug.com/alerts/alerts_300x250_1"].iloc[0]
    assert (row["property"], row["section"], row["slot"]) == ("Web", "alerts", "alerts_300x250_1")
    # Rill rows get the same trio via shared codes
    row = r_wb[r_wb["key_path"] == "weatherbug.com/maps/maps_300x250"].iloc[0]
    assert (row["property"], row["section"]) == ("Web", "maps")
    # Others bucket is explicit, never a fake site
    others = r_wb[r_wb["is_others"]]
    assert (others["property"] == "(unattributed)").all()


# ── TABLES ────────────────────────────────────────────────────────────────────

def test_tables_conserve_volume(pairs):
    tables = recon.build_tables(pairs)
    m = "opportunities"
    ov = tables["overview"]
    for p in pairs:
        row = ov[ov["site"] == p.name].iloc[0]
        assert row[f"GAM_{m}"] == p.gam.df[m].sum()
        assert row[f"Rill_{m}"] == p.rill.df[m].sum()
    # matched + unmatched = total, per site, on the Rill side
    mv, um = tables["overview_matched"], tables["rill_unmatched"]
    for p in pairs:
        matched = mv.loc[mv["site"] == p.name, f"Rill_{m}"].sum()
        residue = um.loc[um["site"] == p.name, f"Rill_{m}"].sum()
        assert matched + residue == pytest.approx(p.rill.df[m].sum())


def test_disc_pct_rill_only_not_hidden():
    import numpy as np
    assert recon.disc_pct(0, 0) == 0.0
    assert np.isnan(recon.disc_pct(0, 100))   # formatter shows 'Rill only'
    assert recon.disc_pct(100, 0) == 100.0
    assert recon.disc_pct(100, 90) == pytest.approx(10.0)


def test_no_date_table_when_samples_lack_dates(pairs):
    tables = recon.build_tables(pairs)
    assert "by_date" not in tables
    assert "by_source_group" not in tables
    assert not tables["by_property"].empty
    assert not tables["by_adunit"].empty


def test_classification_summary(pairs):
    cs = recon.classification_summary(pairs)
    gb = cs[cs["site"] == "GB News"]
    assert set(gb["property"]) == {"Web", "App — Android", "App — iOS", "Video"}
    wb = cs[cs["site"] == "WeatherBug"]
    assert "Ad Exchange In-App" in set(wb["property"])
    # Rill only covers WB web in the sample
    assert wb.loc[wb["property"] == "Web", "in_rill"].iloc[0] == True
    assert wb.loc[wb["property"] == "App — Android", "in_rill"].iloc[0] == False
