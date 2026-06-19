import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(
    page_title="GAM × Rill Reconciliation",
    page_icon="📊",
    layout="wide",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Typography & base ── */
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

/* ── Top banner ── */
.banner {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(37,99,168,0.2);
}
.banner h1 { margin: 0 0 0.3rem 0; font-size: 1.7rem; color: #ffffff; font-weight: 700; }
.banner p  { margin: 0; color: #bfdbfe; font-size: 0.92rem; }

/* ── Section label ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: #eff6ff;
    border-left: 4px solid #2563a8;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.8rem 0 0.6rem 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e40af;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"]  { font-size: 0.78rem; color: #64748b; font-weight: 500; }
[data-testid="stMetricValue"]  { font-size: 1.4rem; color: #1e293b; font-weight: 700; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] h2 {
    color: #1e293b;
    font-size: 1rem;
    font-weight: 700;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2563a8;
    margin-bottom: 0.8rem;
}
[data-testid="stSidebar"] .stCheckbox label { font-size: 0.84rem; color: #334155; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: #2563a8;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.6rem 1.2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover { background: #1e40af; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #16a34a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
    padding: 0.6rem 1.2rem !important;
    font-size: 0.9rem !important;
    transition: background 0.2s !important;
}
[data-testid="stDownloadButton"] > button:hover { background: #15803d !important; }

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] > div {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #ffffff;
}

/* ── Info / warning / success ── */
[data-testid="stAlert"] { border-radius: 8px; }

/* ── Divider ── */
hr { border-color: #e2e8f0; margin: 1rem 0; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #cbd5e1;
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"]:hover { border-color: #2563a8; }
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
LEVEL_DEFAULTS = {
    "Level 1 — Overall by Date":             True,
    "Level 2 — By Date × Site":              True,
    "Level 3 — By Source Group":             True,
    "Level 3b — Source Group × Site":        False,
    "Level 3c — Source Group × Site × Date": False,
    "Level 4 — By Ad Unit":                  True,
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def clean_gam(df: pd.DataFrame) -> pd.DataFrame:
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


def site_from_gam(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val)
    for sep in [" Â» ", " » ", " > "]:
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()


def parse_rill_adunit(val):
    if pd.isna(val) or str(val).strip() == "Others":
        return "", "Others"
    parts = str(val).strip("/").split("/")
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def site_from_domain(domain) -> str:
    if pd.isna(domain):
        return ""
    s = str(domain)
    for tld in [".com", ".org", ".net", ".co.uk", ".io", ".de"]:
        if s.endswith(tld):
            return s[: -len(tld)]
    return s


def disc_pct(gam_val, rill_val):
    return None if gam_val == 0 else (gam_val - rill_val) / gam_val * 100


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    icon = "🔴" if abs(v) >= ALERT_PCT else ("⚠️" if abs(v) >= WARN_PCT else "✅")
    return f"{icon} {v:+.2f}%"


def build_disc(gam_agg, rill_agg, keys, sort_by=None):
    merged = gam_agg.merge(rill_agg, on=keys, how="outer").fillna(0)
    merged["IMP_Disc%"] = merged.apply(lambda r: disc_pct(r["GAM_IMP"], r["Rill_IMP"]), axis=1)
    merged["Rev_Disc%"] = merged.apply(lambda r: disc_pct(r["GAM_Rev"], r["Rill_Rev"]), axis=1)
    if sort_by:
        merged = merged.sort_values(sort_by)
    return merged


def render_table(merged: pd.DataFrame) -> pd.DataFrame:
    disp = merged.copy()
    if "Date" in disp.columns:
        disp["Date"] = disp["Date"].astype(str)
    for c in ["GAM_IMP", "Rill_IMP"]:
        if c in disp:
            disp[c] = disp[c].apply(lambda x: f"{int(x):,}")
    for c in ["GAM_Rev", "Rill_Rev"]:
        if c in disp:
            disp[c] = disp[c].apply(lambda x: f"${x:,.2f}")
    disp["IMP_Disc%"] = merged["IMP_Disc%"].apply(fmt_pct)
    disp["Rev_Disc%"] = merged["Rev_Disc%"].apply(fmt_pct)
    disp = disp.rename(columns={
        "GAM_IMP": "GAM Imps", "GAM_Rev": "GAM Rev",
        "Rill_IMP": "Rill Imps", "Rill_Rev": "Rill Rev",
        "site": "Site", "source_group": "Source Group",
        "ad_unit": "Ad Unit",
        "IMP_Disc%": "IMP Disc%", "Rev_Disc%": "Rev Disc%",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)

    max_d = merged[["IMP_Disc%", "Rev_Disc%"]].abs().max().max()
    if not pd.isna(max_d):
        if max_d >= ALERT_PCT:
            st.error(f"🔴 Largest discrepancy: **{max_d:.1f}%** — this needs investigation.")
        elif max_d >= WARN_PCT:
            st.warning(f"⚠️ Largest discrepancy: **{max_d:.1f}%** — worth reviewing.")
        else:
            st.success(f"✅ All discrepancies within {WARN_PCT}% — looking good.")
    return disp


def section(icon: str, title: str, subtitle: str = ""):
    sub = f"<span style='font-size:0.8rem;color:#64748b;font-weight:400;margin-left:0.5rem'>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f'<div class="section-label">{icon} {title}{sub}</div>',
        unsafe_allow_html=True,
    )


def to_csv(tables: dict) -> str:
    """Combine all selected tables into one CSV with section headers between them."""
    sections = []
    for name, df in tables.items():
        d = df.copy()
        if "Date" in d.columns:
            d["Date"] = d["Date"].astype(str)
        for c in ["IMP_Disc%", "Rev_Disc%"]:
            if c in d.columns:
                d[c] = d[c].apply(
                    lambda v: round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                )
        sections.append(f"### {name}")
        sections.append(d.to_csv(index=False))
        sections.append("")
    return "\n".join(sections)


# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False

# ── BANNER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>📊 GAM × Rill Reconciliation</h1>
  <p>Upload your GAM and Rill CSV exports, configure exclusions, and drill into discrepancies by date, site, source group, and ad unit.</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # ── UPLOADS ───────────────────────────────────────────────────────────────
    st.markdown("## 📁 Upload Files")

    with st.expander("GAM — required columns"):
        st.markdown(
            "**Dimensions**  \nDate · Line item type · Order  \n"
            "Ad unit (all levels) · Ad unit code\n\n"
            "**Metrics**  \nTotal impressions  \nTotal CPM and CPC revenue"
        )
    gam_file = st.file_uploader("GAM Export CSV", type="csv", label_visibility="collapsed")
    if gam_file:
        st.success(f"✅ {gam_file.name}")
    else:
        st.caption("No GAM file uploaded yet.")

    st.markdown("")

    with st.expander("Rill — required columns"):
        st.markdown(
            "**Dimensions**  \nTs (day) · Domain  \n"
            "Revenue Source Type · Ad Unit\n\n"
            "**Metrics**  \nTotal Impressions · Revenue"
        )
    rill_file = st.file_uploader("Rill Export CSV", type="csv", label_visibility="collapsed")
    if rill_file:
        st.success(f"✅ {rill_file.name}")
    else:
        st.caption("No Rill file uploaded yet.")

    st.divider()

    # ── REPORT LEVELS ─────────────────────────────────────────────────────────
    st.markdown("## 📋 Report Levels")
    st.caption("Toggle which tables appear in the report.")

    show_levels = {}
    for lvl in ALL_LEVELS:
        show_levels[lvl] = st.checkbox(lvl, value=LEVEL_DEFAULTS[lvl], key=f"chk_{lvl}")

    st.divider()

    # ── RUN ───────────────────────────────────────────────────────────────────
    run_clicked = st.button("▶  Run Analysis", type="primary")
    if run_clicked:
        st.session_state.report_ready = True

# ── VALIDATE FILES ────────────────────────────────────────────────────────────

if not (gam_file and rill_file):
    st.info("👈  Upload both CSV files in the sidebar to get started.")
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
dates    = sorted(gam["Date"].unique())
sites    = sorted(gam["site"].unique())

# ── PREPARE RILL ──────────────────────────────────────────────────────────────

rill = rill_raw.copy()
rill["Date"]              = pd.to_datetime(rill["Ts (day)"]).dt.date
rill["Total Impressions"] = pd.to_numeric(rill["Total Impressions"], errors="coerce").fillna(0)
rill["Revenue"]           = pd.to_numeric(rill["Revenue"],           errors="coerce").fillna(0)
rill["source_group"]      = rill["Revenue Source Type"].map(RILL_GROUP)

parsed = rill["Ad Unit"].apply(parse_rill_adunit)
rill["site_from_path"] = parsed.apply(lambda t: t[0])
rill["ad_unit_code"]   = parsed.apply(lambda t: t[1])

domain_to_site = (
    rill[rill["site_from_path"] != ""]
    .drop_duplicates("Domain")[["Domain", "site_from_path"]]
    .set_index("Domain")["site_from_path"].to_dict()
)
rill["site"] = rill.apply(
    lambda r: r["site_from_path"] if r["site_from_path"] != ""
    else domain_to_site.get(r["Domain"], site_from_domain(r["Domain"])),
    axis=1,
)
rill_data = rill[
    rill["Revenue Source Type"].notna()
    & (rill["Revenue Source Type"].str.strip() != "")
].copy()

# ── SUMMARY METRICS ───────────────────────────────────────────────────────────

section("📌", "Data Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("GAM Rows",    f"{len(gam_raw):,}")
c2.metric("Rill Rows",   f"{len(rill_raw):,}")
c3.metric("Date Range",  f"{len(dates)} day{'s' if len(dates) != 1 else ''}")
c4.metric("Sites",       f"{len(sites)}")
c5.metric("→ OB",        f"{n_ob:,}",     help="Blank Line item type + blank Order rows")
c6.metric("→ AMAZON",    f"{n_amazon:,}", help="Price priority with Amazon/APS/TAM order")

# ── EXCLUSION PICKER ──────────────────────────────────────────────────────────

section("🚫", "Exclude Orders", "optional — expand to remove specific orders before comparison")
with st.expander("Click to manage order exclusions", expanded=False):
    st.caption(
        "Tick orders you want to remove entirely from the analysis. "
        "Amazon/APS/TAM are already reclassified automatically — "
        "only exclude here if an order should be fully ignored."
    )
    all_orders = (
        gam[["Line item type", "Order"]].drop_duplicates()
        .dropna(subset=["Order"]).query("Order != 'OB'")
        .sort_values(["Line item type", "Order"])
    )
    excluded_orders: set = set()
    for lit in all_orders["Line item type"].unique():
        orders_in_group = all_orders[all_orders["Line item type"] == lit]["Order"].tolist()
        st.markdown(f"**{lit}** ({len(orders_in_group)} orders)")
        cols = st.columns(2)
        for i, order in enumerate(orders_in_group):
            if cols[i % 2].checkbox(order, key=f"excl_{lit}_{order}", value=False):
                excluded_orders.add(order)
        st.markdown("")

gam_final = gam[~gam["Order"].isin(excluded_orders)].copy() if excluded_orders else gam.copy()
if excluded_orders:
    st.info(f"🚫 Excluding **{len(excluded_orders)}** order(s) from all comparisons.")

# ── WAIT FOR RUN ──────────────────────────────────────────────────────────────

if not st.session_state.report_ready:
    st.markdown("")
    st.info("👈  Select your report levels in the sidebar, then click **Run Analysis** to generate the report.")
    st.stop()

# ── BUILD TABLES ──────────────────────────────────────────────────────────────

# L1
gam_l1  = gam_final.groupby("Date").agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l1 = rill_data.groupby("Date").agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l1 = build_disc(gam_l1, rill_l1, ["Date"], sort_by=["Date"])

# L2
gam_l2  = gam_final.groupby(["Date","site"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l2 = rill_data.groupby(["Date","site"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l2 = build_disc(gam_l2, rill_l2, ["Date","site"], sort_by=["site","Date"])

# L3
gam_l3  = gam_final[gam_final["source_group"].notna()].groupby("source_group").agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3 = rill_data[rill_data["source_group"].notna()].groupby("source_group").agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3 = build_disc(gam_l3, rill_l3, ["source_group"])

# L3b
gam_l3b  = gam_final[gam_final["source_group"].notna()].groupby(["site","source_group"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3b = rill_data[rill_data["source_group"].notna()].groupby(["site","source_group"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3b = build_disc(gam_l3b, rill_l3b, ["site","source_group"], sort_by=["site","source_group"])

# L3c
gam_l3c  = gam_final[gam_final["source_group"].notna()].groupby(["Date","site","source_group"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index()
rill_l3c = rill_data[rill_data["source_group"].notna()].groupby(["Date","site","source_group"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index()
l3c = build_disc(gam_l3c, rill_l3c, ["Date","site","source_group"], sort_by=["site","Date","source_group"])

# L4
gam_ad  = set(gam_final["Ad unit code"].dropna().unique())
rill_ad = set(rill_data[rill_data["ad_unit_code"] != "Others"]["ad_unit_code"].dropna().unique())
common  = gam_ad & rill_ad
if common:
    gam_l4  = gam_final[gam_final["Ad unit code"].isin(common)].groupby(["site","Ad unit code"]).agg(GAM_IMP=("Total impressions","sum"), GAM_Rev=("Total CPM and CPC revenue","sum")).reset_index().rename(columns={"Ad unit code":"ad_unit"})
    rill_l4 = rill_data[rill_data["ad_unit_code"].isin(common)].groupby(["site","ad_unit_code"]).agg(Rill_IMP=("Total Impressions","sum"), Rill_Rev=("Revenue","sum")).reset_index().rename(columns={"ad_unit_code":"ad_unit"})
    l4 = build_disc(gam_l4, rill_l4, ["site","ad_unit"]).sort_values("GAM_Rev", ascending=False)
else:
    l4 = pd.DataFrame()

# ── DOWNLOAD & EXPORT ─────────────────────────────────────────────────────────

# All computed tables — available regardless of display toggles
level_map = {
    "Level 1 — Overall by Date":             ("L1 By Date",          l1),
    "Level 2 — By Date × Site":              ("L2 Date x Site",      l2),
    "Level 3 — By Source Group":             ("L3 Source Group",     l3),
    "Level 3b — Source Group × Site":        ("L3b SrcGrp x Site",   l3b),
    "Level 3c — Source Group × Site × Date": ("L3c Full Drill",      l3c),
    "Level 4 — By Ad Unit":                  ("L4 Ad Unit",          l4),
}
available_levels = [lvl for lvl, (_, df) in level_map.items() if not df.empty]

section("⬇️", "Download Report")

# Independent download selector — separate from the display toggles
download_selection = st.multiselect(
    "Select tables to include in your download:",
    options=available_levels,
    default=available_levels,
    key="download_levels",
)

csv_tables = {
    name: df
    for lvl, (name, df) in level_map.items()
    if lvl in download_selection
}

if download_selection:
    col_csv, col_pdf = st.columns(2)

    with col_csv:
        csv_data = to_csv(csv_tables)
        st.download_button(
            label="⬇️  Download as CSV",
            data=csv_data,
            file_name="gam_rill_reconciliation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("💡 To open in Google Sheets: sheets.google.com → File → Import → Upload this CSV.")

    with col_pdf:
        components.html("""
        <button onclick="window.parent.print()" style="
            background-color: #dc2626;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            padding: 0.55rem 1rem;
            width: 100%;
            cursor: pointer;
            font-family: sans-serif;
        ">🖨️  Save as PDF</button>
        """, height=45)
        st.caption("💡 Opens your browser's print dialog — choose 'Save as PDF'.")
else:
    st.warning("Select at least one table above to enable download.")

# ── RENDER REPORT ─────────────────────────────────────────────────────────────

if show_levels.get("Level 1 — Overall by Date"):
    section("📅", "Level 1 — Overall by Date")
    render_table(l1)

if show_levels.get("Level 2 — By Date × Site"):
    section("🌐", "Level 2 — By Date × Site")
    render_table(l2)

if show_levels.get("Level 3 — By Source Group"):
    section("🏷️", "Level 3 — By Source Group", "all dates & sites combined")
    st.caption(
        "**Mapping applied:** Price priority → Prebid + Price Priority  ·  "
        "Ad Exchange + OB → ADX + OB  ·  AMAZON → Amazon  ·  House → House  ·  Standard → Standard"
    )
    render_table(l3)

if show_levels.get("Level 3b — Source Group × Site"):
    section("🏷️", "Level 3b — Source Group × Site")
    render_table(l3b)

if show_levels.get("Level 3c — Source Group × Site × Date"):
    section("🏷️", "Level 3c — Source Group × Site × Date")
    render_table(l3c)

if show_levels.get("Level 4 — By Ad Unit"):
    section("📦", "Level 4 — By Ad Unit", "top revenue units · GAM ∩ Rill only · sorted by GAM revenue")
    st.caption("Only ad units present in **both** GAM and Rill are shown. Rill's 'Others' bucket is excluded here.")
    if not l4.empty:
        render_table(l4)
    else:
        st.info("No matching ad unit codes found between GAM and Rill.")
