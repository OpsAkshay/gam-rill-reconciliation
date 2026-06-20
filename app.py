import json
import zlib
import base64

import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

st.set_page_config(
    page_title="GAM × Rill Reconciliation",
    page_icon="📊",
    layout="wide",
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

.banner {
    background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(37,99,168,0.2);
}
.banner h1 { margin: 0 0 0.3rem 0; font-size: 1.7rem; color: #fff; font-weight: 700; }
.banner p  { margin: 0; color: #bfdbfe; font-size: 0.92rem; }

.section-label {
    display: flex; align-items: center; gap: 0.5rem;
    background: #eff6ff; border-left: 4px solid #2563a8;
    padding: 0.6rem 1rem; border-radius: 0 8px 8px 0;
    margin: 1.8rem 0 0.6rem 0; font-size: 0.95rem; font-weight: 600; color: #1e40af;
}
.date-bar {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
    padding: 0.9rem 1.4rem; margin: 1.2rem 0 0.4rem 0;
    display: flex; align-items: center; gap: 1rem;
}
.shared-badge {
    background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px;
    padding: 0.8rem 1.2rem; margin-bottom: 1rem; color: #15803d;
    font-weight: 600; font-size: 0.9rem;
}
.action-panel {
    background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 10px;
    padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
}

[data-testid="stMetric"] {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748b; font-weight: 500; }
[data-testid="stMetricValue"] { font-size: 1.4rem; color: #1e293b; font-weight: 700; }
[data-testid="stSidebar"] { background: #f1f5f9; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] .stCheckbox label { font-size: 0.84rem; color: #334155; }
.stButton > button[kind="primary"] {
    background: #2563a8; color: #fff; border: none; border-radius: 8px;
    font-weight: 600; font-size: 0.9rem; padding: 0.6rem 1.2rem; width: 100%;
}
.stButton > button[kind="primary"]:hover { background: #1e40af; }
[data-testid="stDataFrame"] > div {
    border-radius: 10px; border: 1px solid #e2e8f0;
    overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
[data-testid="stFileUploader"] {
    background: #fff; border: 2px dashed #cbd5e1; border-radius: 10px; padding: 0.5rem;
}
hr { border-color: #e2e8f0; margin: 1rem 0; }
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
RECLASSIFY_TARGETS = [
    "AMAZON", "Price priority", "Ad Exchange", "House", "Standard", "OB",
]

# ── URL ENCODE / DECODE ───────────────────────────────────────────────────────

def encode_tables(tables: dict, meta: dict) -> str:
    payload = {"meta": meta, "tables": {}}
    for name, df in tables.items():
        d = df.copy()
        if "Date" in d.columns:
            d["Date"] = d["Date"].astype(str)
        for col in d.select_dtypes(include=[float]).columns:
            d[col] = d[col].round(4)
        payload["tables"][name] = {"columns": list(d.columns), "data": d.values.tolist()}
    raw = json.dumps(payload, separators=(",", ":"))
    compressed = zlib.compress(raw.encode("utf-8"), level=9)
    return base64.urlsafe_b64encode(compressed).decode()


def decode_tables(encoded: str):
    compressed = base64.urlsafe_b64decode(encoded.encode())
    raw = zlib.decompress(compressed).decode("utf-8")
    payload = json.loads(raw)
    tables = {
        name: pd.DataFrame(v["data"], columns=v["columns"])
        for name, v in payload["tables"].items()
    }
    return tables, payload.get("meta", {})


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
    if pd.isna(val): return ""
    s = str(val)
    for sep in [" Â» ", " » ", " > "]:
        if sep in s: return s.split(sep)[0].strip()
    return s.strip()


def parse_rill_adunit(val):
    if pd.isna(val) or str(val).strip() == "Others": return "", "Others"
    parts = str(val).strip("/").split("/")
    if len(parts) >= 3: return parts[-2], parts[-1]
    if len(parts) == 2: return parts[0], parts[1]
    return "", parts[0]


def site_from_domain(domain):
    if pd.isna(domain): return ""
    s = str(domain)
    for tld in [".com", ".org", ".net", ".co.uk", ".io", ".de"]:
        if s.endswith(tld): return s[:-len(tld)]
    return s


def disc_pct(gam_val, rill_val):
    return None if gam_val == 0 else (gam_val - rill_val) / gam_val * 100


def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    icon = "🔴" if abs(v) >= ALERT_PCT else ("⚠️" if abs(v) >= WARN_PCT else "✅")
    return f"{icon} {v:+.2f}%"


def build_disc(gam_agg, rill_agg, keys, sort_by=None):
    merged = gam_agg.merge(rill_agg, on=keys, how="outer").fillna(0)
    merged["IMP_Disc%"] = merged.apply(lambda r: disc_pct(r["GAM_IMP"], r["Rill_IMP"]), axis=1)
    merged["Rev_Disc%"] = merged.apply(lambda r: disc_pct(r["GAM_Rev"], r["Rill_Rev"]), axis=1)
    if sort_by: merged = merged.sort_values(sort_by)
    return merged


def render_table(df: pd.DataFrame):
    disp = df.copy()
    if "Date" in disp.columns:
        disp["Date"] = disp["Date"].astype(str)
    for c in ["GAM_IMP", "Rill_IMP"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).apply(lambda x: f"{int(x):,}")
    for c in ["GAM_Rev", "Rill_Rev"]:
        if c in disp.columns:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").fillna(0).apply(lambda x: f"${x:,.2f}")
    for c in ["IMP_Disc%", "Rev_Disc%"]:
        if c in disp.columns:
            raw = pd.to_numeric(df[c], errors="coerce")
            disp[c] = raw.apply(fmt_pct)
    disp = disp.rename(columns={
        "GAM_IMP": "GAM Imps", "GAM_Rev": "GAM Rev",
        "Rill_IMP": "Rill Imps", "Rill_Rev": "Rill Rev",
        "site": "Site", "source_group": "Source Group", "ad_unit": "Ad Unit",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)
    raw_disc = pd.to_numeric(df[["IMP_Disc%", "Rev_Disc%"]].stack(), errors="coerce").abs()
    max_d = raw_disc.max() if not raw_disc.empty else float("nan")
    if not pd.isna(max_d):
        if max_d >= ALERT_PCT:
            st.error(f"🔴 Largest discrepancy: **{max_d:.1f}%** — needs investigation.")
        elif max_d >= WARN_PCT:
            st.warning(f"⚠️ Largest discrepancy: **{max_d:.1f}%** — worth reviewing.")
        else:
            st.success(f"✅ All discrepancies within {WARN_PCT}% threshold.")


def section(icon, title, subtitle=""):
    sub = (
        f"<span style='font-size:0.8rem;color:#64748b;font-weight:400;margin-left:0.5rem'>{subtitle}</span>"
        if subtitle else ""
    )
    st.markdown(f'<div class="section-label">{icon} {title}{sub}</div>', unsafe_allow_html=True)


def date_range_bar(date_range_str, sites):
    st.markdown(f"""
    <div class="date-bar">
      <span style="font-size:1.5rem">📅</span>
      <div>
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:0.05em">Analysis Period</div>
        <div style="font-size:1.2rem;font-weight:700;color:#1e40af">{date_range_str}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:0.72rem;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:0.05em">Sites</div>
        <div style="font-size:0.9rem;font-weight:600;color:#1e293b">{" · ".join(sites)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_report(tables: dict, show_levels: dict, date_range_str: str, sites: list):
    date_range_bar(date_range_str, sites)
    label_map = {
        "Level 1 — Overall by Date":             ("📅", "Level 1 — Overall by Date", ""),
        "Level 2 — By Date × Site":              ("🌐", "Level 2 — By Date × Site", ""),
        "Level 3 — By Source Group":             ("🏷️",  "Level 3 — By Source Group", "all dates & sites combined"),
        "Level 3b — Source Group × Site":        ("🏷️",  "Level 3b — Source Group × Site", ""),
        "Level 3c — Source Group × Site × Date": ("🏷️",  "Level 3c — Source Group × Site × Date", ""),
        "Level 4 — By Ad Unit":                  ("📦", "Level 4 — By Ad Unit", "top revenue units · GAM ∩ Rill only"),
    }
    lvl_to_sheet = {
        "Level 1 — Overall by Date":             "L1 By Date",
        "Level 2 — By Date × Site":              "L2 Date x Site",
        "Level 3 — By Source Group":             "L3 Source Group",
        "Level 3b — Source Group × Site":        "L3b SrcGrp x Site",
        "Level 3c — Source Group × Site × Date": "L3c Full Drill",
        "Level 4 — By Ad Unit":                  "L4 Ad Unit",
    }
    for lvl in ALL_LEVELS:
        sheet = lvl_to_sheet[lvl]
        if show_levels.get(lvl) and sheet in tables and not tables[sheet].empty:
            icon, title, sub = label_map[lvl]
            section(icon, title, sub)
            if lvl == "Level 3 — By Source Group":
                st.caption(
                    "**Mapping:** Price priority → Prebid + Price Priority  ·  "
                    "Ad Exchange + OB → ADX + OB  ·  AMAZON → Amazon  ·  "
                    "House → House  ·  Standard → Standard"
                )
            render_table(tables[sheet])


# ── SESSION STATE ─────────────────────────────────────────────────────────────

_ss_defaults = {
    "report_ready":   False,
    "link_generated": False,
    "excl_set":       [],   # list of excluded order names
    "reassignments":  {},   # {order_name: new_line_item_type}
    "_order_sel":     [],   # orders selected in the Orders table (carried across reruns)
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── BANNER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>📊 GAM × Rill Reconciliation</h1>
  <p>Upload your GAM and Rill exports, review order classifications, then run the discrepancy report.</p>
</div>
""", unsafe_allow_html=True)

# ── SHARED REPORT MODE ────────────────────────────────────────────────────────

params = st.query_params
if "r" in params:
    try:
        shared_tables, meta = decode_tables(params["r"])
        st.markdown(f"""
        <div class="shared-badge">
            ✅ Shared report — Analysis period: <strong>{meta.get("date_range", "N/A")}</strong>
            &nbsp;·&nbsp; Generated: {meta.get("generated_at", "N/A")}
        </div>
        """, unsafe_allow_html=True)
        render_report(
            shared_tables,
            {lvl: True for lvl in ALL_LEVELS},
            meta.get("date_range", ""),
            meta.get("sites", []),
        )
    except Exception as e:
        st.error(f"Could not load shared report. The link may be invalid or corrupted. ({e})")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
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

    st.markdown("## 📋 Display Levels")
    st.caption("Choose which tables appear on screen.")
    show_levels = {
        lvl: st.checkbox(lvl, value=LEVEL_DEFAULTS[lvl], key=f"chk_{lvl}")
        for lvl in ALL_LEVELS
    }

    st.divider()

    run_clicked = st.button("▶  Run Analysis", type="primary")
    if run_clicked:
        st.session_state.report_ready = True
        st.session_state.link_generated = False

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
if missing_g: st.error(f"GAM file missing columns: {missing_g}"); st.stop()
if missing_r: st.error(f"Rill file missing columns: {missing_r}"); st.stop()

# ── CLEAN GAM ─────────────────────────────────────────────────────────────────

gam = clean_gam(gam_raw)
gam["Total CPM and CPC revenue"] = (
    gam["Total CPM and CPC revenue"].astype(str)
    .str.replace(r"[$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce").fillna(0)
)
gam["Total impressions"] = (
    gam["Total impressions"].astype(str).str.replace(",", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce").fillna(0)
)
gam["Date"]         = pd.to_datetime(gam["Date"]).dt.date
gam["site"]         = gam["Ad unit (all levels)"].apply(site_from_gam)
gam["source_group"] = gam["Line item type"].map(GAM_GROUP)

dates = sorted(gam["Date"].unique())
sites = sorted(gam["site"].unique())
date_fmt       = lambda d: d.strftime("%-d %b %Y")
date_range_str = date_fmt(dates[0]) if len(dates) == 1 else f"{date_fmt(dates[0])} – {date_fmt(dates[-1])}"

# Original type lookup (before any user overrides)
order_type_map = (
    gam[gam["Order"].notna() & (gam["Order"] != "OB")]
    .drop_duplicates("Order")
    .set_index("Order")["Line item type"]
    .to_dict()
)

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
    else domain_to_site.get(r["Domain"], site_from_domain(r["Domain"])), axis=1,
)
rill_data = rill[
    rill["Revenue Source Type"].notna() & (rill["Revenue Source Type"].str.strip() != "")
].copy()

# ── SUMMARY METRICS ───────────────────────────────────────────────────────────

n_ob     = int((gam["Line item type"] == "OB").sum())
n_amazon = int((gam["Line item type"] == "AMAZON").sum())

section("📌", "Data Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("GAM Rows",   f"{len(gam_raw):,}")
c2.metric("Rill Rows",  f"{len(rill_raw):,}")
c3.metric("Date Range", f"{len(dates)} day{'s' if len(dates) != 1 else ''}")
c4.metric("Sites",      f"{len(sites)}")
c5.metric("→ OB",       f"{n_ob:,}",     help="Blank Line item type + blank Order")
c6.metric("→ AMAZON",   f"{n_amazon:,}", help="Price priority with Amazon/APS/TAM orders")

# ── ORDERS ────────────────────────────────────────────────────────────────────

n_excl = len(st.session_state.excl_set)
n_rc   = len(st.session_state.reassignments)
orders_subtitle = "type in any column header to search · check rows · then Save"
if n_excl or n_rc:
    parts = []
    if n_excl: parts.append(f"{n_excl} excluded")
    if n_rc:   parts.append(f"{n_rc} reclassified")
    orders_subtitle += "  ·  " + " · ".join(parts)

section("📋", "Orders", orders_subtitle)

# Build effective orders table — reflects current reassignments so user sees live state
gam_effective = gam.copy()
for order, new_type in st.session_state.reassignments.items():
    gam_effective.loc[gam_effective["Order"] == order, "Line item type"] = new_type
gam_effective["source_group"] = gam_effective["Line item type"].map(GAM_GROUP)

orders_df = (
    gam_effective[gam_effective["Order"].notna() & (gam_effective["Order"] != "OB")]
    .groupby(["Order", "Line item type"])
    .agg(Revenue=("Total CPM and CPC revenue", "sum"), Impressions=("Total impressions", "sum"))
    .reset_index()
)
orders_df["Bucket"] = orders_df["Line item type"].map(GAM_GROUP).fillna("—")
excl_set_current = set(st.session_state.excl_set)
orders_df["Status"] = orders_df["Order"].apply(
    lambda o: "🚫 Excluded" if o in excl_set_current
    else ("🔄 Reclassified" if o in st.session_state.reassignments else "")
)
orders_df = orders_df.sort_values(["Bucket", "Revenue"], ascending=[True, False]).reset_index(drop=True)

# Keep Revenue/Impressions numeric so AG Grid can sort and filter them correctly
grid_df = orders_df[["Order", "Line item type", "Bucket", "Revenue", "Impressions", "Status"]].copy()
grid_df = grid_df.rename(columns={"Line item type": "Type"})

# ── Action panel (always visible; Save reads AG Grid selection below) ─────────

st.caption("Use the filter row in each column to search, check rows to select, then choose an action:")
ca, cb, cc = st.columns([2, 3, 1])
with ca:
    action_choice = st.radio(
        "Action", ["Exclude", "Reclassify"],
        horizontal=True, key="ord_action",
        label_visibility="collapsed",
    )
with cb:
    if action_choice == "Reclassify":
        reclass_to = st.selectbox(
            "Move to", RECLASSIFY_TARGETS, key="ord_reclass_to",
            label_visibility="collapsed",
        )
    else:
        st.caption("Selected orders will be removed from all comparisons.")
        reclass_to = None
with cc:
    # Render button here for layout; handle it AFTER AgGrid so response is defined
    save_clicked = st.button("💾 Save", type="primary", key="ord_save")

# ── AG Grid table — all interaction is client-side, zero reruns on scroll/select ──

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
gb.configure_default_column(
    filter=True, sortable=True, resizable=True,
    floatingFilter=True, floatingFilterComponentParams={"suppressFilterButton": True},
)
gb.configure_column("Order",       min_width=260, flex=3, pinned="left")
gb.configure_column("Type",        min_width=140, flex=2)
gb.configure_column("Bucket",      min_width=180, flex=2)
gb.configure_column(
    "Revenue", min_width=110, flex=1,
    type=["numericColumn", "numberColumnFilter"],
    valueFormatter="'$' + Number(value).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})",
)
gb.configure_column(
    "Impressions", min_width=120, flex=1,
    type=["numericColumn", "numberColumnFilter"],
    valueFormatter="Number(value).toLocaleString('en-US')",
)
gb.configure_column("Status", min_width=130, flex=1, filter=False, floatingFilter=False)
gb.configure_grid_options(
    rowHeight=36, headerHeight=40,
    suppressMovableColumns=True,
    animateRows=True,
)
grid_opts = gb.build()

response = AgGrid(
    grid_df,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.NO_UPDATE,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    height=430,
    theme="streamlit",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    key="orders_grid",
)

# ── Handle Save (response is now defined) ─────────────────────────────────────

if save_clicked:
    sel_raw = response.get("selected_rows") or []
    if isinstance(sel_raw, pd.DataFrame):
        sel_orders = sel_raw["Order"].tolist() if not sel_raw.empty else []
    else:
        sel_orders = [r["Order"] for r in sel_raw] if sel_raw else []

    if sel_orders:
        if action_choice == "Exclude":
            st.session_state.excl_set = list(set(st.session_state.excl_set + sel_orders))
        else:
            for o in sel_orders:
                st.session_state.reassignments[o] = reclass_to
        st.rerun()
    else:
        st.warning("No orders checked — tick the checkboxes in the table first.")

# ── Active changes ─────────────────────────────────────────────────────────────

if st.session_state.excl_set or st.session_state.reassignments:
    st.markdown("**Active changes** — click Undo to revert individual items:")

    undo_excl = []
    for o in st.session_state.excl_set:
        c1, c2 = st.columns([9, 1])
        c1.markdown(f"🚫 **{o}** — excluded from analysis")
        if c2.button("Undo", key=f"undo_e_{o}"):
            undo_excl.append(o)

    undo_rc = []
    for o, new_t in list(st.session_state.reassignments.items()):
        orig = order_type_map.get(o, "?")
        c1, c2 = st.columns([9, 1])
        c1.markdown(f"🔄 **{o}** — {orig} → **{new_t}**")
        if c2.button("Undo", key=f"undo_r_{o}"):
            undo_rc.append(o)

    if st.button("🗑 Clear all changes", key="clear_all_changes"):
        st.session_state.excl_set = []
        st.session_state.reassignments = {}
        st.rerun()

    if undo_excl:
        st.session_state.excl_set = [o for o in st.session_state.excl_set if o not in undo_excl]
        st.rerun()
    if undo_rc:
        for o in undo_rc:
            del st.session_state.reassignments[o]
        st.rerun()

# ── APPLY CHANGES TO GAM ──────────────────────────────────────────────────────

gam_processed = gam.copy()
if st.session_state.reassignments:
    for order, new_type in st.session_state.reassignments.items():
        gam_processed.loc[gam_processed["Order"] == order, "Line item type"] = new_type
    gam_processed["source_group"] = gam_processed["Line item type"].map(GAM_GROUP)

excl_set_final = set(st.session_state.excl_set)
gam_final = (
    gam_processed[~gam_processed["Order"].isin(excl_set_final)].copy()
    if excl_set_final else gam_processed.copy()
)

# ── WAIT FOR RUN ──────────────────────────────────────────────────────────────

if not st.session_state.report_ready:
    st.info("👈  When ready, click **Run Analysis** in the sidebar to generate the report.")
    st.stop()

# ── BUILD ALL TABLES ──────────────────────────────────────────────────────────

def agg_gam(keys):
    return gam_final.groupby(keys).agg(
        GAM_IMP=("Total impressions",         "sum"),
        GAM_Rev=("Total CPM and CPC revenue", "sum"),
    ).reset_index()

def agg_rill(keys):
    return rill_data.groupby(keys).agg(
        Rill_IMP=("Total Impressions", "sum"),
        Rill_Rev=("Revenue",           "sum"),
    ).reset_index()

def agg_gam_grp(keys):
    return gam_final[gam_final["source_group"].notna()].groupby(keys).agg(
        GAM_IMP=("Total impressions",         "sum"),
        GAM_Rev=("Total CPM and CPC revenue", "sum"),
    ).reset_index()

def agg_rill_grp(keys):
    return rill_data[rill_data["source_group"].notna()].groupby(keys).agg(
        Rill_IMP=("Total Impressions", "sum"),
        Rill_Rev=("Revenue",           "sum"),
    ).reset_index()

l1  = build_disc(agg_gam(["Date"]),
                 agg_rill(["Date"]),
                 ["Date"], sort_by=["Date"])
l2  = build_disc(agg_gam(["Date", "site"]),
                 agg_rill(["Date", "site"]),
                 ["Date", "site"], sort_by=["site", "Date"])
l3  = build_disc(agg_gam_grp(["source_group"]),
                 agg_rill_grp(["source_group"]),
                 ["source_group"])
l3b = build_disc(agg_gam_grp(["site", "source_group"]),
                 agg_rill_grp(["site", "source_group"]),
                 ["site", "source_group"], sort_by=["site", "source_group"])
l3c = build_disc(agg_gam_grp(["Date", "site", "source_group"]),
                 agg_rill_grp(["Date", "site", "source_group"]),
                 ["Date", "site", "source_group"], sort_by=["site", "Date", "source_group"])

gam_ad  = set(gam_final["Ad unit code"].dropna().unique())
rill_ad = set(rill_data[rill_data["ad_unit_code"] != "Others"]["ad_unit_code"].dropna().unique())
common  = gam_ad & rill_ad
if common:
    gam_l4 = (
        gam_final[gam_final["Ad unit code"].isin(common)]
        .groupby(["site", "Ad unit code"])
        .agg(GAM_IMP=("Total impressions", "sum"), GAM_Rev=("Total CPM and CPC revenue", "sum"))
        .reset_index().rename(columns={"Ad unit code": "ad_unit"})
    )
    rill_l4 = (
        rill_data[rill_data["ad_unit_code"].isin(common)]
        .groupby(["site", "ad_unit_code"])
        .agg(Rill_IMP=("Total Impressions", "sum"), Rill_Rev=("Revenue", "sum"))
        .reset_index().rename(columns={"ad_unit_code": "ad_unit"})
    )
    l4 = build_disc(gam_l4, rill_l4, ["site", "ad_unit"]).sort_values("GAM_Rev", ascending=False)
else:
    l4 = pd.DataFrame()

all_tables = {
    "L1 By Date":         l1,
    "L2 Date x Site":     l2,
    "L3 Source Group":    l3,
    "L3b SrcGrp x Site":  l3b,
    "L3c Full Drill":     l3c,
    "L4 Ad Unit":         l4,
}

# ── SHARE SECTION ─────────────────────────────────────────────────────────────

section("🔗", "Share Report")

lvl_to_sheet = {
    "Level 1 — Overall by Date":             "L1 By Date",
    "Level 2 — By Date × Site":              "L2 Date x Site",
    "Level 3 — By Source Group":             "L3 Source Group",
    "Level 3b — Source Group × Site":        "L3b SrcGrp x Site",
    "Level 3c — Source Group × Site × Date": "L3c Full Drill",
    "Level 4 — By Ad Unit":                  "L4 Ad Unit",
}
available_levels = [
    lvl for lvl in ALL_LEVELS
    if not all_tables[lvl_to_sheet[lvl]].empty
]
share_selection = st.multiselect(
    "Select tables to include in the shared link:",
    options=available_levels,
    default=[l for l in available_levels if LEVEL_DEFAULTS.get(l, False)],
    key="share_levels",
)

if share_selection and st.button("🔗  Generate Shareable Link", type="primary"):
    tables_to_share = {lvl_to_sheet[lvl]: all_tables[lvl_to_sheet[lvl]] for lvl in share_selection}
    meta = {
        "date_range":   date_range_str,
        "sites":        sites,
        "generated_at": pd.Timestamp.now().strftime("%-d %b %Y, %H:%M"),
    }
    st.session_state.link_generated = True
    st.query_params["r"] = encode_tables(tables_to_share, meta)

if st.session_state.link_generated:
    st.success(
        "✅ Shareable link generated! Copy the URL from your browser's address bar — "
        "recipients see the full interactive report without uploading any files."
    )
    st.caption("💡 The link encodes your report data. Anyone with it can open the interactive tables.")
elif not share_selection:
    st.warning("Select at least one table above to generate a link.")

# ── RENDER REPORT ─────────────────────────────────────────────────────────────

render_report(all_tables, show_levels, date_range_str, sites)
