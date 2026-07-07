"""Tests grounded in the four real sample exports in the repo root.

Run: .venv/bin/python -m pytest test_recon.py -q

The sample exports hold real publisher data and are intentionally not
committed — tests needing them skip on machines without them; synthetic-input
tests always run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import recon

ROOT = Path(__file__).parent
GB_GAM = ROOT / "GBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_GAM = ROOT / "wBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165350.csv"
GB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165704.csv"


@pytest.fixture(scope="module")
def reports():
    if not all(p.exists() for p in (GB_GAM, WB_GAM, WB_RILL, GB_RILL)):
        pytest.skip("sample export CSVs not present (kept out of git)")
    return {
        "gb_gam": recon.parse_gam(GB_GAM.read_bytes()),
        "gb_rill": recon.parse_rill(GB_RILL.read_bytes()),
        "wb_gam": recon.parse_gam(WB_GAM.read_bytes()),
        "wb_rill": recon.parse_rill(WB_RILL.read_bytes()),
    }


# ── FORMAT DETECTION ──────────────────────────────────────────────────────────

def test_detects_opportunities_metric(reports):
    for key in ("gb", "wb"):
        gam, rill = reports[f"{key}_gam"], reports[f"{key}_rill"]
        assert gam.metrics == ["opportunities"]
        assert rill.metrics == ["opportunities"]
        assert recon.common_metrics(gam, rill) == ["opportunities"]
        assert recon.common_dims(gam, rill) == set()


def test_parse_any_detects_kind(reports):
    assert recon.parse_any(GB_GAM.read_bytes()).kind == "gam"
    assert recon.parse_any(GB_RILL.read_bytes()).kind == "rill"


def test_unknown_file_rejected_with_clear_error():
    junk = b"foo,bar\n1,2\n"
    with pytest.raises(ValueError, match="GAM"):
        recon.parse_gam(junk)
    with pytest.raises(ValueError, match="Rill"):
        recon.parse_rill(junk)
    with pytest.raises(ValueError, match="Unrecognizable"):
        recon.parse_any(junk)


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
    assert rrep.df["domain"].iloc[0] == "gbnews.com"


# ── PATH JOIN (the acid test: 100% match on both real publishers) ─────────────

def test_every_rill_path_matches_gam(reports):
    for key in ("gb", "wb"):
        gam_keys = set(reports[f"{key}_gam"].df["key_path"])
        rill = reports[f"{key}_rill"].df
        unmatched = set(rill.loc[~rill["is_others"], "key_path"]) - gam_keys
        assert unmatched == set(), f"{key}: {sorted(unmatched)[:5]}"


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


def test_unit_meta_brand_stripping(reports):
    meta = recon.build_unit_meta(reports["gb_gam"])
    assert meta.loc["gbnews_celebrity", "property"] == "Web"
    assert meta.loc["gbnews_app_ios", "property"] == "App — iOS"
    assert meta.loc["primis_video_ios", "property"] == "Video"
    assert meta.loc["gbnews_celebrity", "unit_display"] == "Celebrity"
    assert meta.loc["gbnews_money", "unit_display"] == "Money"


# ── SITE IDENTIFICATION ───────────────────────────────────────────────────────

def test_site_map_gb_brand_clustering(reports):
    """No Domain column in the sample → sites come from GAM-side evidence."""
    m = recon.build_site_map(reports["gb_gam"], reports["gb_rill"])
    assert m["gbnews_celebrity"] == "GB News"      # brand prefix in display
    assert m["gbnews_app_android"] == "GB News"    # 'GB News - APP - Android'
    assert m["gbnews_video"] == "GB News"          # code starts with brand
    assert m["primis_video_android"] == "Primis"   # token cluster (4 units)
    assert m["primis_video_ios"] == "Primis"


def test_site_map_wb_domain_and_clusters(reports):
    m = recon.build_site_map(reports["wb_gam"], reports["wb_rill"])
    assert m["weatherbug.com"] == "weatherbug.com"          # domain-like unit
    assert m["wb_android_app_phone"] == "WB"                # token cluster, display casing
    assert m["wb_ios_app_tablet"] == "WB"
    # singleton with no cluster keeps its readable display name
    assert m["ca-mb-app-pub-8015868500526768-tag"] == "Ad Exchange Mobile In-App"


def test_rill_domain_is_authoritative():
    """When Rill reports a unit under a Domain, that beats every heuristic."""
    gam_csv = (
        "Ad unit (all levels),Ad unit code level 1,Ad unit code level 2,"
        "Programmatic eligible ad requests\n"
        "WB_Android_App_Phone » Alerts,WB_Android_App_Phone,alerts,100\n"
        "WB_iOS_App_Phone » Alerts,WB_iOS_App_Phone,alerts,100\n"
    ).encode()
    rill_csv = (
        "Domain,Ad Unit,Total Opportunities\n"
        "www.weatherbug.com,65299053/wb_android_app_phone/alerts,90\n"
        "weatherbug.com,Others,10\n"
    ).encode()
    gam, rill = recon.parse_gam(gam_csv), recon.parse_rill(rill_csv)
    m = recon.build_site_map(gam, rill)
    assert m["wb_android_app_phone"] == "weatherbug.com"   # Domain, www-stripped
    g, r = recon.enrich(gam, rill)
    # 'Others' row still lands on the right site via its own Domain
    assert r.loc[r["is_others"], "site"].iloc[0] == "weatherbug.com"
    # the unit Rill never reported clusters with its sibling's site? No —
    # it has no Domain evidence, so it token-clusters separately and honestly.
    assert m["wb_ios_app_phone"] != ""


# ── SECTION EXTRACTION ────────────────────────────────────────────────────────

def test_sections(reports):
    g_gb, r_gb = recon.enrich(reports["gb_gam"], reports["gb_rill"])
    row = g_gb[g_gb["key_path"] == "gbnews_celebrity/mpu_1"].iloc[0]
    assert (row["site"], row["property"], row["section"], row["slot"]) == \
        ("GB News", "Web", "Celebrity", "mpu_1")

    g_wb, r_wb = recon.enrich(reports["wb_gam"], reports["wb_rill"])
    row = g_wb[g_wb["key_path"] == "weatherbug.com/alerts/alerts_300x250_1"].iloc[0]
    assert (row["site"], row["property"], row["section"], row["slot"]) == \
        ("weatherbug.com", "Web", "alerts", "alerts_300x250_1")
    # Rill rows land on the same site via the shared unit codes
    row = r_wb[r_wb["key_path"] == "weatherbug.com/maps/maps_300x250"].iloc[0]
    assert (row["site"], row["property"], row["section"]) == ("weatherbug.com", "Web", "maps")
    # Others bucket without Domain info: explicitly unattributed, never a fake site
    others = r_wb[r_wb["is_others"]]
    assert (others["site"] == "(unattributed)").all()


# ── TABLES ────────────────────────────────────────────────────────────────────

def test_tables_conserve_volume(reports):
    m = "opportunities"
    for key in ("gb", "wb"):
        gam, rill = reports[f"{key}_gam"], reports[f"{key}_rill"]
        tables = recon.build_tables(gam, rill)
        ov = tables["overview"]
        assert ov[f"GAM_{m}"].sum() == gam.df[m].sum()
        assert ov[f"Rill_{m}"].sum() == rill.df[m].sum()
        # matched + unmatched = total on the Rill side
        matched = tables["overview_matched"][f"Rill_{m}"].sum()
        residue = tables["rill_unmatched"][f"Rill_{m}"].sum()
        assert matched + residue == pytest.approx(rill.df[m].sum())


def test_disc_pct_rill_only_not_hidden():
    assert recon.disc_pct(0, 0) == 0.0
    assert np.isnan(recon.disc_pct(0, 100))   # formatter shows 'Rill only'
    assert recon.disc_pct(100, 0) == 100.0
    assert recon.disc_pct(100, 90) == pytest.approx(10.0)


def test_no_date_table_when_samples_lack_dates(reports):
    tables = recon.build_tables(reports["gb_gam"], reports["gb_rill"])
    assert "by_date" not in tables
    assert "by_date_site" not in tables
    assert "by_source_group" not in tables
    assert not tables["by_property"].empty
    assert not tables["by_adunit"].empty


def test_classification_summary(reports):
    cs = recon.classification_summary(reports["gb_gam"], reports["gb_rill"])
    gb = cs[cs["site"] == "GB News"]
    assert {"Web", "App — Android", "App — iOS", "Video"} <= set(gb["property"])
    assert gb.loc[gb["property"] == "Web", "in_rill"].iloc[0] == True
    assert gb.loc[gb["property"] == "App — Android", "in_rill"].iloc[0] == False

    cs_wb = recon.classification_summary(reports["wb_gam"], reports["wb_rill"])
    web = cs_wb[(cs_wb["site"] == "weatherbug.com") & (cs_wb["property"] == "Web")]
    assert web["in_rill"].iloc[0] == True
    wb_apps = cs_wb[cs_wb["site"] == "WB"]
    assert not wb_apps.empty and not wb_apps["in_rill"].any()
