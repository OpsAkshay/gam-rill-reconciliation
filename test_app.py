"""App-level smoke tests.

AppTest can't inject file uploads, so the boot test checks the empty state and
the pipeline test drives the same functions the app calls after upload.
"""

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import recon

ROOT = Path(__file__).parent
GB_GAM = ROOT / "GBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_GAM = ROOT / "wBAd Manager Report (Jun 26, 2026 - Jun 28, 2026) (1).csv"
WB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165350.csv"
GB_RILL = ROOT / "holistic_revenue_analytics_internal_filtered_20260702165704.csv"


def test_app_boots_to_upload_state():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert any("Upload a GAM + Rill CSV pair" in str(i.value) for i in at.info)


def _pairs():
    if not all(p.exists() for p in (GB_GAM, WB_GAM, WB_RILL, GB_RILL)):
        pytest.skip("sample export CSVs not present (kept out of git)")
    return [
        recon.SitePair("GB News", recon.parse_gam(GB_GAM.read_bytes()),
                       recon.parse_rill(GB_RILL.read_bytes())),
        recon.SitePair("WeatherBug", recon.parse_gam(WB_GAM.read_bytes()),
                       recon.parse_rill(WB_RILL.read_bytes())),
    ]


def test_full_pipeline_matches_app_flow():
    """Same sequence app.py runs after upload: tables → summary → display frames."""
    pairs = _pairs()
    tables = recon.build_tables(pairs)
    class_df = recon.classification_summary(pairs)
    cov = recon.coverage_summary(tables, ["opportunities"])

    # Two sites, exactly as named — never derived from paths
    assert sorted(cov["site"]) == ["GB News", "WeatherBug"]
    assert sorted(class_df["site"].unique()) == ["GB News", "WeatherBug"]

    # Display formatting works on every produced table
    from recon import format_table
    for key in ("overview", "overview_matched", "by_property", "by_section", "by_adunit"):
        disp = format_table(tables[key])
        assert "Site" in disp.columns
        assert "Opportunities Δ%" in disp.columns
        assert len(disp) == len(tables[key])

    # Matched compare covers 100% of attributable Rill volume in the samples
    assert (cov["rill_coverage"] < 100).all()   # Others bucket keeps it below 100
    assert (cov["rill_coverage"] > 60).all()


def test_share_roundtrip():
    from recon import encode_tables, decode_tables

    pairs = _pairs()
    tables = recon.build_tables(pairs)
    keep = {"overview": tables["overview"], "by_property": tables["by_property"]}
    enc = encode_tables(keep, {"date_range": "—", "sites": ["GB News", "WeatherBug"]})
    dec, meta = decode_tables(enc)
    assert set(dec) == {"overview", "by_property"}
    assert meta["sites"] == ["GB News", "WeatherBug"]
    pd.testing.assert_frame_equal(
        dec["overview"][["site"]], tables["overview"][["site"]]
    )
