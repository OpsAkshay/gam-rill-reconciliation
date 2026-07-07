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


def _reports(gam_path, rill_path):
    if not (gam_path.exists() and rill_path.exists()):
        pytest.skip("sample export CSVs not present (kept out of git)")
    return recon.parse_gam(gam_path.read_bytes()), recon.parse_rill(rill_path.read_bytes())


def test_app_boots_to_upload_state():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert any("Upload both CSV files" in str(i.value) for i in at.info)


@pytest.mark.parametrize("gam_path,rill_path,expected_site", [
    (GB_GAM, GB_RILL, "GB News"),
    (WB_GAM, WB_RILL, "weatherbug.com"),
])
def test_full_pipeline_matches_app_flow(gam_path, rill_path, expected_site):
    """Same sequence app.py runs after upload: tables → summary → display frames."""
    gam, rill = _reports(gam_path, rill_path)
    tables = recon.build_tables(gam, rill)
    class_df = recon.classification_summary(gam, rill)
    cov = recon.coverage_summary(tables, recon.common_metrics(gam, rill))

    # The publisher's main site is identified from the data
    assert expected_site in set(cov["site"])
    assert expected_site in set(class_df["site"])

    # Display formatting works on every produced table
    for key in ("overview", "overview_matched", "by_property", "by_section", "by_adunit"):
        disp = recon.format_table(tables[key])
        assert "Site" in disp.columns
        assert "Opportunities Δ%" in disp.columns
        assert len(disp) == len(tables[key])


def test_share_roundtrip():
    gam, rill = _reports(GB_GAM, GB_RILL)
    tables = recon.build_tables(gam, rill)
    keep = {"overview": tables["overview"], "by_property": tables["by_property"]}
    enc = recon.encode_tables(keep, {"date_range": "—", "sites": ["GB News"]})
    dec, meta = recon.decode_tables(enc)
    assert set(dec) == {"overview", "by_property"}
    assert meta["sites"] == ["GB News"]
    pd.testing.assert_frame_equal(
        dec["overview"][["site"]], tables["overview"][["site"]]
    )
