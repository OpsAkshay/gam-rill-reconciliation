import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="GAM × Rill Reconciliation",
    page_icon="📊",
    layout="wide",
)

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


def site_from_rill_domain(domain) -> str:
    if pd.isna(domain):
        return ""
    s = str(domain)
    for tld in [".com", ".org", ".net", ".co.uk", ".io", ".de"]:
        if s.endswith(tld):
            return s[: -len(tld)]
    return s


def site_and_adunit_from_rill_path(val):
    """Return (site, ad_unit) from a Rill Ad Unit path like
    '/1030006,21877042/familydestinationsguide/adhesion'.
    For literal 'Others' returns ('', 'Others')."""
    if pd.isna(val) or str(val).strip() == "Others":
        return "", "Others"
    parts = str(val).strip("/").split("/")
    # parts: ["1030006,21877042", "site", "ad_unit"]
    if len(parts) >= 3:
        return parts[-2], parts[-1]
    if len(parts) == 2:
        return parts[0], parts[1]
    return "", parts[0]


def disc_pct(gam_val, rill_val):
    if gam_val == 0:
        return None
    return (gam_val - rill_val) / gam_val * 100


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    abs_v = abs(v)
    icon = "🔴" if abs_v >= ALERT_PCT else ("⚠️" if abs_v >= WARN_PCT else "✅")
    return f"{icon} {v:+.2f}%"


def build_and_show(
    gam_agg: pd.DataFrame,
    rill_agg: pd.DataFrame,
    keys: list,
    sort_by: list = None,
):
    merged = gam_agg.merge(rill_agg, on=keys, how="outer").fillna(0)
    merged["IMP_Disc%"] = merged.apply(
        lambda r: disc_pct(r["GAM_IMP"], r["Rill_IMP"]), axis=1
    )
    merged["Rev_Disc%"] = merged.apply(
        lambda r: disc_pct(r["GAM_Rev"], r["Rill_Rev"]), axis=1
    )
    if sort_by:
        merged = merged.sort_values(sort_by)

    disp = merged.copy()
    if "Date" in disp.columns:
        disp["Date"] = disp["Date"].astype(str)
    disp["GAM_IMP"]   = disp["GAM_IMP"].apply(lambda x: f"{x:,.0f}")
    disp["GAM_Rev"]   = disp["GAM_Rev"].apply(lambda x: f"${x:,.2f}")
    disp["Rill_IMP"]  = disp["Rill_IMP"].apply(lambda x: f"{x:,.0f}")
    disp["Rill_Rev"]  = disp["Rill_Rev"].apply(lambda x: f"${x:,.2f}")
    disp["IMP_Disc%"] = merged["IMP_Disc%"].apply(fmt_pct)
    disp["Rev_Disc%"] = merged["Rev_Disc%"].apply(fmt_pct)

    col_rename = {
        "GAM_IMP":  "GAM Imps",
        "GAM_Rev":  "GAM Rev",
        "Rill_IMP": "Rill Imps",
        "Rill_Rev": "Rill Rev",
        "site":     "Site",
        "source_group": "Source Group",
        "ad_unit":  "Ad Unit",
    }
    disp = disp.rename(columns=col_rename)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    max_disc = merged[["IMP_Disc%", "Rev_Disc%"]].abs().max().max()
    if pd.isna(max_disc):
        return
    if max_disc >= ALERT_PCT:
        st.error(f"🔴 Max discrepancy: {max_disc:.1f}% — investigate further.")
    elif max_disc >= WARN_PCT:
        st.warning(f"⚠️ Max discrepancy: {max_disc:.1f}% — worth reviewing.")
    else:
        st.success(f"✅ All discrepancies within {WARN_PCT}% threshold.")

    return merged


# ── SESSION STATE ─────────────────────────────────────────────────────────────

for k in ["report_ready", "gam_df", "rill_df"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("📊 GAM × Rill Reconciliation")
st.markdown("---")

# ── STEP 1 · UPLOAD ───────────────────────────────────────────────────────────

st.header("1 · Upload Files")

col_g, col_r = st.columns(2)

with col_g:
    st.subheader("GAM Export")
    with st.expander("Required columns & dimensions"):
        st.markdown(
            "**Dimensions (rows):** Date, Line item type, Order, "
            "Ad unit (all levels), Ad unit code  \n"
            "**Metrics:** Total impressions, Total CPM and CPC revenue"
        )
    gam_file = st.file_uploader("Upload GAM CSV", type="csv", key="gam_up")

with col_r:
    st.subheader("Rill Export")
    with st.expander("Required columns & dimensions"):
        st.markdown(
            "**Dimensions:** Ts (day), Domain, Revenue Source Type, Ad Unit  \n"
            "**Metrics:** Total Impressions, Revenue"
        )
    rill_file = st.file_uploader("Upload Rill CSV", type="csv", key="rill_up")

if not (gam_file and rill_file):
    st.info("Upload both files above to continue.")
    st.stop()

# ── LOAD ──────────────────────────────────────────────────────────────────────

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

# ── STEP 2 · CLEAN GAM ────────────────────────────────────────────────────────

st.markdown("---")
st.header("2 · GAM Auto-Cleaning")

gam = clean_gam(gam_raw)

gam["Total CPM and CPC revenue"] = (
    gam["Total CPM and CPC revenue"]
    .astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
    .fillna(0)
)
gam["Total impressions"] = (
    gam["Total impressions"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
    .fillna(0)
)
gam["Date"]         = pd.to_datetime(gam["Date"]).dt.date
gam["site"]         = gam["Ad unit (all levels)"].apply(site_from_gam)
gam["source_group"] = gam["Line item type"].map(GAM_GROUP)

n_ob     = int((gam["Line item type"] == "OB").sum())
n_amazon = int((gam["Line item type"] == "AMAZON").sum())

c1, c2 = st.columns(2)
c1.metric("Rows → OB",     n_ob,     help="Blank Line item type + blank Order rows")
c2.metric("Rows → AMAZON", n_amazon, help="Price priority rows with Amazon/APS/TAM orders")

# ── STEP 3 · EXCLUSION PICKER ─────────────────────────────────────────────────

st.markdown("---")
st.header("3 · Exclude Orders from Analysis")
st.caption(
    "Select any orders you want to drop before reconciliation. "
    "Amazon/APS/TAM are already reclassified — you only need to exclude "
    "orders that should be fully ignored."
)

all_orders = (
    gam[["Line item type", "Order"]]
    .drop_duplicates()
    .dropna(subset=["Order"])
    .query("Order != 'OB'")
    .sort_values(["Line item type", "Order"])
)

# Group checkboxes by line item type inside expanders
excluded_orders: set = set()
for lit in all_orders["Line item type"].unique():
    orders_in_group = all_orders[all_orders["Line item type"] == lit]["Order"].tolist()
    default_open = lit == "Price priority"
    with st.expander(f"{lit}  ({len(orders_in_group)} orders)", expanded=default_open):
        cols = st.columns(2)
        for i, order in enumerate(orders_in_group):
            chk = cols[i % 2].checkbox(order, key=f"excl_{lit}_{order}", value=False)
            if chk:
                excluded_orders.add(order)

gam_final = gam[~gam["Order"].isin(excluded_orders)].copy()

if excluded_orders:
    st.info(f"Excluding **{len(excluded_orders)}** order(s) from the analysis.")

# ── PREPARE RILL ──────────────────────────────────────────────────────────────

rill = rill_raw.copy()
rill["Date"] = pd.to_datetime(rill["Ts (day)"]).dt.date
rill["Total Impressions"] = pd.to_numeric(rill["Total Impressions"], errors="coerce").fillna(0)
rill["Revenue"]           = pd.to_numeric(rill["Revenue"],           errors="coerce").fillna(0)
rill["source_group"]      = rill["Revenue Source Type"].map(RILL_GROUP)

# Extract site + ad_unit_code from the Ad Unit path (same spelling as GAM).
# e.g. "/1030006,21877042/saltandlavendar/adhesion" → site="saltandlavendar", code="adhesion"
parsed = rill["Ad Unit"].apply(site_and_adunit_from_rill_path)
rill["site_from_path"] = parsed.apply(lambda t: t[0])
rill["ad_unit_code"]   = parsed.apply(lambda t: t[1])

# For "Others" ad-unit rows the path has no site segment; fall back to a
# domain→site lookup built from rows that do have a path.
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

# Drop blank Revenue Source Type rows (unfilled / header rows with 0 imps)
rill_data = rill[
    rill["Revenue Source Type"].notna()
    & (rill["Revenue Source Type"].str.strip() != "")
].copy()

# ── STEP 4 · RUN ANALYSIS ─────────────────────────────────────────────────────

st.markdown("---")
st.header("4 · Discrepancy Report")

if st.button("▶  Run Analysis", type="primary", use_container_width=True):
    st.session_state.report_ready = True

if not st.session_state.report_ready:
    st.info("Click **Run Analysis** to generate the report.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 1 · BY DATE
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Level 1 — Overall by Date")

gam_l1 = gam_final.groupby("Date").agg(
    GAM_IMP=("Total impressions", "sum"),
    GAM_Rev=("Total CPM and CPC revenue", "sum"),
).reset_index()

rill_l1 = rill_data.groupby("Date").agg(
    Rill_IMP=("Total Impressions", "sum"),
    Rill_Rev=("Revenue", "sum"),
).reset_index()

build_and_show(gam_l1, rill_l1, ["Date"], sort_by=["Date"])

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 · BY DATE × SITE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Level 2 — By Date × Site")

gam_l2 = gam_final.groupby(["Date", "site"]).agg(
    GAM_IMP=("Total impressions", "sum"),
    GAM_Rev=("Total CPM and CPC revenue", "sum"),
).reset_index()

rill_l2 = rill_data.groupby(["Date", "site"]).agg(
    Rill_IMP=("Total Impressions", "sum"),
    Rill_Rev=("Revenue", "sum"),
).reset_index()

build_and_show(gam_l2, rill_l2, ["Date", "site"], sort_by=["site", "Date"])

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 · BY SOURCE GROUP (all dates + sites combined)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Level 3 — By Revenue Source Group (all dates & sites)")
st.caption(
    "GAM → Rill mapping:  "
    "Price priority → Prebid + Price Priority  |  "
    "Ad Exchange + OB → ADX + OB  |  "
    "AMAZON → Amazon  |  House → House  |  Standard → Standard"
)

gam_l3 = gam_final[gam_final["source_group"].notna()].groupby("source_group").agg(
    GAM_IMP=("Total impressions", "sum"),
    GAM_Rev=("Total CPM and CPC revenue", "sum"),
).reset_index()

rill_l3 = rill_data[rill_data["source_group"].notna()].groupby("source_group").agg(
    Rill_IMP=("Total Impressions", "sum"),
    Rill_Rev=("Revenue", "sum"),
).reset_index()

build_and_show(gam_l3, rill_l3, ["source_group"], sort_by=["source_group"])

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3b · BY SOURCE GROUP × SITE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Level 3b — By Source Group × Site")

gam_l3b = gam_final[gam_final["source_group"].notna()].groupby(["site", "source_group"]).agg(
    GAM_IMP=("Total impressions", "sum"),
    GAM_Rev=("Total CPM and CPC revenue", "sum"),
).reset_index()

rill_l3b = rill_data[rill_data["source_group"].notna()].groupby(["site", "source_group"]).agg(
    Rill_IMP=("Total Impressions", "sum"),
    Rill_Rev=("Revenue", "sum"),
).reset_index()

build_and_show(gam_l3b, rill_l3b, ["site", "source_group"], sort_by=["site", "source_group"])

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3c · BY SOURCE GROUP × SITE × DATE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Level 3c — By Source Group × Site × Date")

gam_l3c = gam_final[gam_final["source_group"].notna()].groupby(
    ["Date", "site", "source_group"]
).agg(
    GAM_IMP=("Total impressions", "sum"),
    GAM_Rev=("Total CPM and CPC revenue", "sum"),
).reset_index()

rill_l3c = rill_data[rill_data["source_group"].notna()].groupby(
    ["Date", "site", "source_group"]
).agg(
    Rill_IMP=("Total Impressions", "sum"),
    Rill_Rev=("Revenue", "sum"),
).reset_index()

build_and_show(
    gam_l3c, rill_l3c,
    ["Date", "site", "source_group"],
    sort_by=["site", "Date", "source_group"],
)

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 4 · BY AD UNIT (top revenue units, intersection only)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Level 4 — By Ad Unit (top units, GAM ∩ Rill only)")
st.caption(
    "Only ad units present in both datasets are shown. "
    "Rill's 'Others' bucket and GAM-only ad units are excluded here. "
    "Sorted by GAM revenue descending."
)

gam_ad_units  = set(gam_final["Ad unit code"].dropna().unique())
rill_ad_units = set(
    rill_data[rill_data["ad_unit_code"] != "Others"]["ad_unit_code"].dropna().unique()
)
common_units = gam_ad_units & rill_ad_units

if not common_units:
    st.info("No matching ad unit codes found between GAM and Rill.")
else:
    gam_l4 = (
        gam_final[gam_final["Ad unit code"].isin(common_units)]
        .groupby(["site", "Ad unit code"])
        .agg(
            GAM_IMP=("Total impressions", "sum"),
            GAM_Rev=("Total CPM and CPC revenue", "sum"),
        )
        .reset_index()
        .rename(columns={"Ad unit code": "ad_unit"})
    )

    rill_l4 = (
        rill_data[rill_data["ad_unit_code"].isin(common_units)]
        .groupby(["site", "ad_unit_code"])
        .agg(
            Rill_IMP=("Total Impressions", "sum"),
            Rill_Rev=("Revenue", "sum"),
        )
        .reset_index()
        .rename(columns={"ad_unit_code": "ad_unit"})
    )

    l4 = gam_l4.merge(rill_l4, on=["site", "ad_unit"], how="outer").fillna(0)
    l4["IMP_Disc%"] = l4.apply(lambda r: disc_pct(r["GAM_IMP"], r["Rill_IMP"]), axis=1)
    l4["Rev_Disc%"] = l4.apply(lambda r: disc_pct(r["GAM_Rev"], r["Rill_Rev"]), axis=1)
    l4 = l4.sort_values("GAM_Rev", ascending=False)

    disp4 = l4.copy()
    disp4["GAM_IMP"]  = disp4["GAM_IMP"].apply(lambda x: f"{x:,.0f}")
    disp4["GAM_Rev"]  = disp4["GAM_Rev"].apply(lambda x: f"${x:,.2f}")
    disp4["Rill_IMP"] = disp4["Rill_IMP"].apply(lambda x: f"{x:,.0f}")
    disp4["Rill_Rev"] = disp4["Rill_Rev"].apply(lambda x: f"${x:,.2f}")
    disp4["IMP_Disc%"] = l4["IMP_Disc%"].apply(fmt_pct)
    disp4["Rev_Disc%"] = l4["Rev_Disc%"].apply(fmt_pct)
    disp4 = disp4.rename(columns={
        "site":     "Site",
        "ad_unit":  "Ad Unit",
        "GAM_IMP":  "GAM Imps",
        "GAM_Rev":  "GAM Rev",
        "Rill_IMP": "Rill Imps",
        "Rill_Rev": "Rill Rev",
    })
    st.dataframe(disp4, use_container_width=True, hide_index=True)
