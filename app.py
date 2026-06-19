import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(
    page_title="GAM × Rill Reconciliation",
    page_icon="📊",
    layout="wide",
)

# ── STYLING ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #f0f4f8;
    }
    /* Main header banner */
    .banner {
        background: linear-gradient(90deg, #1a3a5c 0%, #2563a8 100%);
        padding: 1.25rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .banner h1 { margin: 0; font-size: 1.6rem; color: white; }
    .banner p  { margin: 0.25rem 0 0 0; opacity: 0.85; font-size: 0.9rem; }
    /* Section labels */
    .level-label {
        background: #eef2f7;
        border-left: 4px solid #2563a8;
        padding: 0.5rem 1rem;
        border-radius: 0 6px 6px 0;
        font-weight: 600;
        color: #1a3a5c;
        margin: 1.5rem 0 0.5rem 0;
        font-size: 1rem;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #dde3ea;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    /* Download button */
    [data-testid="stDownloadButton"] button {
        background-color: #2563a8;
        color: white;
        border-radius: 6px;
        border: none;
        font-weight: 600;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: #1a3a5c;
        color: white;
    }
    /* Run button */
    .stButton > button[kind="primary"] {
        background-color: #2563a8;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
    }
    /* Table styling */
    [data-testid="stDataFrame"] { border-radius: 8px; }
    /* Remove default padding on expanders */
    .streamlit-expanderContent { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

GAM_REQUIRED = [
    "Date", "Line item type", "Order",
    "Ad unit (all levels)", "Ad unit code",
    "Total impressions", "Total CPM and CPC revenue",
]
RILL_REQUIRED = [
    "Ts (day)", "Domain", "Revenue Source Type",
    "Ad Unit", "Total Impressions", "Revenue",
]
GAM_GROUP = {
    "Price priority": "Prebid + Price Priority",
    "Ad Exchange":    "ADX + OB",
    "OB":             "ADX + OB",
    "AMAZON":         "Amazon",
    "House":          "House",
    "Standard":       "Standard",
}
RILL_GROUP = {
    "Prebid":         "Prebid + Price Priority",
    "Price Priority": "Prebid + Price Priority",
    "OB-ADX":         "ADX + OB",
    "ADX":            "ADX + OB",
    "OB":             "ADX + OB",
    "Amazon":         "Amazon",
    "House":          "House",
    "Standard":       "Standard",
}
WARN_PCT  = 5.0
ALERT_PCT = 20.0

ALL_LEVELS = [
    "Level 1 — Overall by Date",
    "Level 2 — By Date × Site",
    "Level 3 — By Source Group",
    "Level 3b — Source Group × Site",
    "Level 3c — Source Group × Site × Date",
    "Level 4 — By Ad Unit",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_gam(df):
    df = df.copy()
    blank = df["Line item type"].isna() & df["Order"].isna()
    df.loc[blank, "Line item type"] = "OB"
    df.loc[blank, "Order"] = "OB"
    is_amazon = (
        df["Order"].str.contains(r"Amazon|APS|TAM", case=False, na=False)
        & (df["Line item type"] == "Price priority")
    )
    df.loc[is_amazon, "Line item type"] = "AMAZON"
    return df


def site_from_gam(val):
    if pd.isna(val):
        return ""
    s = str(val)
    for sep in [" Â» ", " » ", " > "]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()


def site_and_adunit_from_rill_path(val):
    if pd.isna(val) or str(val).strip() == "Others":
        return "", "Others"
    parts = str(val).strip("/").split("/")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def site_from_rill_domain(domain):
    if pd.isna(domain):
        return ""
    s = str(domain)
    for tld in [".com", ".org", ".net", ".co.uk", ".io", ".de"]:
        if s.endswith(tld):
            return s[: -len(tld)]
    return s


def disc_pct(gam_val, rill_val):
    return None if gam_val == 0 else (gam_val - rill_val) / gam_val * 100


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    icon = "🔴" if abs(v) >= ALERT_PCT else ("⚠️" if abs(v) >= WARN_PCT else "✅")
    return f"{icon} {v:+.2f}%"


def build_disc(gam_agg, rill_agg, keys, sort_by=None):
    merged = gam_agg.merge(rill_agg, on=keys, how="outer").fillna(0)
    merged["IMP_Disc%"] = merged.apply(
        lambda r: disc_pct(r["GAM_IMP"], r["Rill_IMP"]), axis=1
    )
    merged["Rev_Disc%"] = merged.apply(
        lambda r: disc_pct(r["GAM_Rev"], r["Rill_Rev"]), axis=1
    )
    if sort_by:
        merged = merged.sort_values(sort_by)
    return merged


def render_table(merged, label_renames=None):
    disp = merged.copy()
    if "Date" in disp.columns:
        disp["Date"] = disp["Date"].astype(str)
    for c in ["GAM_IMP", "Rill_IMP"]:
        if c in disp:
            disp[c] = disp[c].apply(lambda x: f"{x:,.0f}")
    for c in ["GAM_Rev", "Rill_Rev"]:
        if c in disp:
            disp[c] = disp[c].apply(lambda x: f"${x:,.2f}")
    disp["IMP_Disc%"] = merged["IMP_Disc%"].apply(fmt_pct)
    disp["Rev_Disc%"] = merged["Rev_Disc%"].apply(fmt_pct)

    rename = {
        "GAM_IMP": "GAM Imps", "GAM_Rev": "GAM Rev",
        "Rill_IMP": "Rill Imps", "Rill_Rev": "Rill Rev",
        "site": "Site", "source_group": "Source Group", "ad_unit": "Ad Unit",
        "IMP_Disc%": "IMP Disc%", "Rev_Disc%": "Rev Disc%",
    }
    if label_renames:
        rename.update(label_renames)
    disp = disp.rename(columns=rename)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    max_d = merged[["IMP_Disc%", "Rev_Disc%"]].abs().max().max()
    if not pd.isna(max_d):
        if max_d >= ALERT_PCT:
            st.error(f"🔴 Max discrepancy: {max_d:.1f}% — investigate further.")
        elif max_d >= WARN_PCT:
            st.warning(f"⚠️ Max discrepancy: {max_d:.1f}% — worth reviewing.")
        else:
            st.success(f"✅ All within {WARN_PCT}% threshold.")
    return disp


def to_excel(sheets: dict) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()


def section_header(text):
    st.markdown(f'<div class="level-label">📋 {text}</div>', unsafe_allow_html=True)


# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

# ── BANNER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>📊 GAM × Rill Reconciliation</h1>
  <p>Upload your GAM and Rill exports, configure exclusions, and generate a discrepancy report.</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    # ── FILE UPLOADS ──────────────────────────────────────────────────────────
    st.subheader("1 · Upload Files")

    with st.expander("GAM required columns", expanded=False):
        st.caption(
            "Dimensions: Date, Line item type, Order, "
            "Ad unit (all levels), Ad unit code\n\n"
            "Metrics: Total impressions, Total CPM and CPC revenue"
        )
    gam_file = st.file_uploader("GAM Export (CSV)", type="csv")

    with st.expander("Rill required columns", expanded=False):
        st.caption(
            "Dimensions: Ts (day), Domain, Revenue Source Type, Ad Unit\n\n"
            "Metrics: Total Impressions, Revenue"
        )
    rill_file = st.file_uploader("Rill Export (CSV)", type="csv")

    st.divider()

    # ── REPORT LEVELS ─────────────────────────────────────────────────────────
    st.subheader("2 · Report Levels")
    st.caption("Choose which tables to include in your report.")

    show_levels = {}
    defaults = {
        "Level 1 — Overall by Date": True,
        "Level 2 — By Date × Site": True,
        "Level 3 — By Source Group": True,
        "Level 3b — Source Group × Site": False,
        "Level 3c — Source Group × Site × Date": False,
        "Level 4 — By Ad Unit": True,
    }
    for lvl in ALL_LEVELS:
        show_levels[lvl] = st.checkbox(lvl, value=defaults[lvl])

    st.divider()

    # ── RUN BUTTON ────────────────────────────────────────────────────────────
    run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    if run_clicked:
        st.session_state.report_ready = True

# ── VALIDATE FILES ────────────────────────────────────────────────────────────

if not (gam_file and rill_file):
    st.info("👈 Upload both CSV files in the sidebar to get started.")
    st.stop()

try:
    gam_raw  = pd.read_csv(gam_file,  encoding="latin-1")
    rill_raw = pd.read_csv(rill_file, encoding="utf-8")
except Exception as e:
    st.error(f"Could not read files: {e}")
    st.stop()

gam_raw.columns  = gam_raw.columns.str.strip()
rill_raw.columns = rill_raw.columns.str.strip()

missing_g = [c for c in GAM_REQUIRED  if c not in gam_raw.columns]
missing_r = [c for c in RILL_REQUIRED if c not in rill_raw.columns]
if missing_g:
    st.error(f"GAM file is missing columns: {missing_g}")
    st.stop()
if missing_r:
    st.error(f"Rill file is missing columns: {missing_r}")
    st.stop()

# ── CLEAN GAM ─────────────────────────────────────────────────────────────────

gam = clean_gam(gam_raw)
gam["Total CPM and CPC revenue"] = (
    gam["Total CPM and CPC revenue"].astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce").fillna(0)
)
gam["Total impressions"] = (
    gam["Total impressions"].astype(str)
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce").fillna(0)
)
gam["Date"]         = pd.to_datetime(gam["Date"]).dt.date
gam["site"]         = gam["Ad unit (all levels)"].apply(site_from_gam)
gam["source_group"] = gam["Line item type"].map(GAM_GROUP)

n_ob     = int((gam["Line item type"] == "OB").sum())
n_amazon = int((gam["Line item type"] == "AMAZON").sum())

# ── DATA SUMMARY CARDS ────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("GAM Rows Loaded",     f"{len(gam_raw):,}")
col2.metric("Rill Rows Loaded",    f"{len(rill_raw):,}")
col3.metric("Rows → OB",           f"{n_ob:,}",     help="Blank Line item type + blank Order")
col4.metric("Rows → AMAZON",       f"{n_amazon:,}", help="Price priority with Amazon/APS/TAM orders")

st.divider()

# ── EXCLUSION PICKER ──────────────────────────────────────────────────────────

with st.expander("🚫  Exclude Orders from Analysis", expanded=False):
    st.caption(
        "Tick any orders you want to remove before the comparison runs. "
        "Amazon/APS/TAM orders are already reclassified — only use this for "
        "orders that should be fully ignored."
    )
    all_orders = (
        gam[["Line item type", "Order"]].drop_duplicates()
        .dropna(subset=["Order"])
        .query("Order != 'OB'")
        .sort_values(["Line item type", "Order"])
    )
    excluded_orders: set = set()
    for lit in all_orders["Line item type"].unique():
        orders_in_group = all_orders[all_orders["Line item type"] == lit]["Order"].tolist()
        st.markdown(f"**{lit}**")
        cols = st.columns(2)
        for i, order in enumerate(orders_in_group):
            if cols[i % 2].checkbox(order, key=f"excl_{lit}_{order}", value=False):
                excluded_orders.add(order)

gam_final = gam[~gam["Order"].isin(excluded_orders)].copy() if excluded_orders else gam.copy()
if excluded_orders:
    st.info(f"Excluding **{len(excluded_orders)}** order(s) from the analysis.")

# ── PREPARE RILL ──────────────────────────────────────────────────────────────

rill = rill_raw.copy()
rill["Date"]              = pd.to_datetime(rill["Ts (day)"]).dt.date
rill["Total Impressions"] = pd.to_numeric(rill["Total Impressions"], errors="coerce").fillna(0)
rill["Revenue"]           = pd.to_numeric(rill["Revenue"],           errors="coerce").fillna(0)
rill["source_group"]      = rill["Revenue Source Type"].map(RILL_GROUP)

parsed = rill["Ad Unit"].apply(site_and_adunit_from_rill_path)
rill["site_from_path"] = parsed.apply(lambda t: t[0])
rill["ad_unit_code"]   = parsed.apply(lambda t: t[1])

domain_to_site = (
    rill[rill["site_from_path"] != ""]
    .drop_duplicates("Domain")[["Domain", "site_from_path"]]
    .set_index("Domain")["site_from_path"]
    .to_dict()
)
rill["site"] = rill.apply(
    lambda r: r["site_from_path"] if r["site_from_path"] != ""
    else domain_to_site.get(r["Domain"], site_from_rill_domain(r["Domain"])),
    axis=1,
)

rill_data = rill[
    rill["Revenue Source Type"].notna()
    & (rill["Revenue Source Type"].str.strip() != "")
].copy()

# ── WAIT FOR RUN ──────────────────────────────────────────────────────────────

if not st.session_state.report_ready:
    st.info("👈 Select your report levels in the sidebar, then click **Run Analysis**.")
    st.stop()

# ── BUILD ALL TABLES ──────────────────────────────────────────────────────────

excel_sheets = {}

# Level 1
gam_l1  = gam_final.groupby("Date").agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l1 = rill_data.groupby("Date").agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l1 = build_disc(gam_l1, rill_l1, ["Date"], sort_by=["Date"])

# Level 2
gam_l2  = gam_final.groupby(["Date","site"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l2 = rill_data.groupby(["Date","site"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l2 = build_disc(gam_l2, rill_l2, ["Date","site"], sort_by=["site","Date"])

# Level 3
gam_l3  = gam_final[gam_final["source_group"].notna()].groupby("source_group").agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3 = rill_data[rill_data["source_group"].notna()].groupby("source_group").agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3 = build_disc(gam_l3, rill_l3, ["source_group"])

# Level 3b
gam_l3b  = gam_final[gam_final["source_group"].notna()].groupby(["site","source_group"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3b = rill_data[rill_data["source_group"].notna()].groupby(["site","source_group"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3b = build_disc(gam_l3b, rill_l3b, ["site","source_group"], sort_by=["site","source_group"])

# Level 3c
gam_l3c  = gam_final[gam_final["source_group"].notna()].groupby(["Date","site","source_group"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3c = rill_data[rill_data["source_group"].notna()].groupby(["Date","site","source_group"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3c = build_disc(gam_l3c, rill_l3c, ["Date","site","source_group"], sort_by=["site","Date","source_group"])

# Level 4
gam_ad  = set(gam_final["Ad unit code"].dropna().unique())
rill_ad = set(rill_data[rill_data["ad_unit_code"] != "Others"]["ad_unit_code"].dropna().unique())
common  = gam_ad & rill_ad
gam_l4  = gam_final[gam_final["Ad unit code"].isin(common)].groupby(["site","Ad unit code"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index().rename(columns={"Ad unit code":"ad_unit"})
rill_l4 = rill_data[rill_data["ad_unit_code"].isin(common)].groupby(["site","ad_unit_code"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index().rename(columns={"ad_unit_code":"ad_unit"})
l4 = build_disc(gam_l4, rill_l4, ["site","ad_unit"]).sort_values("GAM_Rev", ascending=False)

# ── DOWNLOAD BUTTON ───────────────────────────────────────────────────────────

def clean_for_excel(df):
    d = df.copy()
    if "Date" in d.columns:
        d["Date"] = d["Date"].astype(str)
    for c in ["IMP_Disc%","Rev_Disc%"]:
        if c in d.columns:
            d[c] = d[c].apply(lambda v: round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None)
    return d

level_map = {
    "Level 1 — Overall by Date":           ("L1 By Date",         l1),
    "Level 2 — By Date × Site":            ("L2 Date x Site",     l2),
    "Level 3 — By Source Group":           ("L3 Source Group",    l3),
    "Level 3b — Source Group × Site":      ("L3b SrcGroup x Site",l3b),
    "Level 3c — Source Group × Site × Date":("L3c Full Drill",    l3c),
    "Level 4 — By Ad Unit":                ("L4 Ad Unit",         l4),
}

for lvl, (sheet_name, df) in level_map.items():
    if show_levels.get(lvl):
        excel_sheets[sheet_name] = clean_for_excel(df)

if excel_sheets:
    excel_bytes = to_excel(excel_sheets)
    st.download_button(
        label="⬇️  Download Report (Excel)",
        data=excel_bytes,
        file_name="gam_rill_reconciliation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.caption("💡 To share as PDF: open the Excel file → File → Print → Save as PDF, or upload to Google Sheets and export.")

st.divider()

# ── RENDER SELECTED LEVELS ────────────────────────────────────────────────────

if show_levels.get("Level 1 — Overall by Date"):
    section_header("Level 1 — Overall by Date")
    render_table(l1)

if show_levels.get("Level 2 — By Date × Site"):
    section_header("Level 2 — By Date × Site")
    render_table(l2)

if show_levels.get("Level 3 — By Source Group"):
    section_header("Level 3 — By Source Group  (all dates & sites combined)")
    st.caption(
        "Mapping: Price priority → Prebid + Price Priority  |  "
        "Ad Exchange + OB → ADX + OB  |  AMAZON → Amazon  |  "
        "House → House  |  Standard → Standard"
    )
    render_table(l3)

if show_levels.get("Level 3b — Source Group × Site"):
    section_header("Level 3b — Source Group × Site")
    render_table(l3b)

if show_levels.get("Level 3c — Source Group × Site × Date"):
    section_header("Level 3c — Source Group × Site × Date")
    render_table(l3c)

if show_levels.get("Level 4 — By Ad Unit"):
    section_header("Level 4 — By Ad Unit  (top revenue units, GAM ∩ Rill only)")
    st.caption("Ad units present in both datasets only. 'Others' excluded. Sorted by GAM revenue.")
    if common:
        render_table(l4)
    else:
        st.info("No matching ad unit codes found between GAM and Rill.")
