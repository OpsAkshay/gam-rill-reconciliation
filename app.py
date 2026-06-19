import json
import zlib
import base64

import streamlit as st
import pandas as pd
import numpy as np

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
.reclass-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.45rem 0.8rem; border-radius: 6px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    margin-bottom: 0.35rem; font-size: 0.85rem;
}
.reclass-order  { flex: 3; font-weight: 500; color: #1e293b; }
.reclass-from   { flex: 2; color: #64748b; }
.reclass-arrow  { color: #94a3b8; }
.reclass-to     { flex: 2; font-weight: 600; color: #1e40af; }

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
    "report_ready":           False,
    "link_generated":         False,
    "excl_multiselect":       [],      # list of excluded order names
    "reassignments":          {},      # {order_name: new_line_item_type}
    "reclass_sel":            [],      # multiselect buffer for reclassify widget
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── BANNER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>📊 GAM × Rill Reconciliation</h1>
  <p>Upload your GAM and Rill exports, configure exclusions and reclassifications,
     then drill into discrepancies by date, site, source group, and ad unit.</p>
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

# Build a deduplicated order list (excluding auto-OB rows) for use in pickers
all_orders_df = (
    gam[gam["Order"].notna() & (gam["Order"] != "OB")]
    [["Order", "Line item type"]]
    .drop_duplicates("Order")
    .sort_values("Order")
)
all_order_names = all_orders_df["Order"].tolist()
order_type_map  = dict(zip(all_orders_df["Order"], all_orders_df["Line item type"]))

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

# ── RECLASSIFY ORDERS ─────────────────────────────────────────────────────────

section(
    "🔄", "Reclassify Orders",
    "optional — move orders from their auto-detected bucket into a different one",
)
with st.expander(
    f"Click to reclassify orders"
    + (f"  ·  {len(st.session_state.reassignments)} active" if st.session_state.reassignments else ""),
    expanded=False,
):
    st.caption(
        "Use this when an order is in the wrong bucket after auto-detection. "
        "For example, orders named A9, A10, K9, K10 that appear under **Price priority** "
        "but should count as **AMAZON**."
    )

    rc_col1, rc_col2, rc_col3 = st.columns([4, 2, 1])
    with rc_col1:
        rc_search = st.text_input(
            "Search orders to reclassify",
            key="rc_search_input",
            placeholder="e.g. A9, K10, Criteo…",
            label_visibility="collapsed",
        )
    with rc_col2:
        rc_target = st.selectbox(
            "Move selected to",
            options=RECLASSIFY_TARGETS,
            key="rc_target",
            label_visibility="collapsed",
        )
    with rc_col3:
        rc_apply = st.button("▶ Apply", key="rc_apply_btn", use_container_width=True)

    # Filter order list by search term
    rc_opts = (
        [o for o in all_order_names if rc_search.lower() in o.lower()]
        if rc_search else all_order_names
    )
    selected_to_move = st.multiselect(
        "Select orders:",
        options=rc_opts,
        format_func=lambda o: f"{o}  [{order_type_map.get(o, '?')}]",
        key="reclass_sel",
        placeholder="Type to search, then select one or more orders…",
        label_visibility="collapsed",
    )

    if rc_apply and selected_to_move:
        for order in selected_to_move:
            st.session_state.reassignments[order] = rc_target
        st.session_state.reclass_sel = []
        st.rerun()

    # Active reassignments table
    if st.session_state.reassignments:
        st.markdown("**Active reassignments:**")
        to_remove = []
        for order, new_type in list(st.session_state.reassignments.items()):
            orig = order_type_map.get(order, "?")
            c_name, c_from, c_arr, c_to, c_rm = st.columns([4, 2, 0.5, 2, 0.8])
            c_name.markdown(f"**{order}**")
            c_from.markdown(f"<span style='color:#64748b'>{orig}</span>", unsafe_allow_html=True)
            c_arr.markdown("→")
            c_to.markdown(f"**{new_type}**")
            if c_rm.button("✕", key=f"rm_rc_{order}"):
                to_remove.append(order)
        for order in to_remove:
            del st.session_state.reassignments[order]
        if to_remove:
            st.rerun()
    else:
        st.caption("No reassignments yet.")

# ── EXCLUDE ORDERS ────────────────────────────────────────────────────────────

section(
    "🚫", "Exclude Orders",
    "optional — remove specific orders from analysis entirely",
)
with st.expander(
    f"Click to manage order exclusions"
    + (f"  ·  {len(st.session_state.excl_multiselect)} excluded" if st.session_state.excl_multiselect else ""),
    expanded=False,
):
    st.caption(
        "Excluded orders are removed from both GAM and Rill comparisons. "
        "Use the search box to find a group of orders quickly, then bulk-exclude them."
    )

    ex_col1, ex_col2 = st.columns([4, 2])
    with ex_col1:
        ex_search = st.text_input(
            "Filter orders",
            key="ex_search_input",
            placeholder="e.g. House, DFP, test…",
            label_visibility="collapsed",
        )
    with ex_col2:
        matching = (
            [o for o in all_order_names if ex_search.lower() in o.lower()]
            if ex_search else []
        )
        ex_bulk_label = f"☑ Exclude all {len(matching)} matching" if matching else "☑ Exclude all matching"
        if st.button(ex_bulk_label, disabled=not matching, key="ex_bulk_btn", use_container_width=True):
            st.session_state.excl_multiselect = list(
                set(st.session_state.excl_multiselect + matching)
            )
            st.rerun()

    excluded_list = st.multiselect(
        "Excluded orders:",
        options=all_order_names,
        key="excl_multiselect",
        placeholder="Type to search, or use the bulk filter above…",
        label_visibility="collapsed",
    )

    # Clear-all shortcut
    if excluded_list and st.button("☐ Clear all exclusions", key="ex_clear_btn"):
        st.session_state.excl_multiselect = []
        st.rerun()

excl_set = set(st.session_state.excl_multiselect)

# ── APPLY RECLASSIFICATIONS & EXCLUSIONS TO GAM ───────────────────────────────

gam_processed = gam.copy()
if st.session_state.reassignments:
    for order, new_type in st.session_state.reassignments.items():
        gam_processed.loc[gam_processed["Order"] == order, "Line item type"] = new_type
    gam_processed["source_group"] = gam_processed["Line item type"].map(GAM_GROUP)

gam_final = (
    gam_processed[~gam_processed["Order"].isin(excl_set)].copy()
    if excl_set else gam_processed.copy()
)

# Status summary
status_parts = []
if excl_set:
    status_parts.append(f"🚫 **{len(excl_set)}** order(s) excluded")
if st.session_state.reassignments:
    status_parts.append(f"🔄 **{len(st.session_state.reassignments)}** order(s) reclassified")
if status_parts:
    st.info("  ·  ".join(status_parts) + "  —  click Run Analysis to refresh the report.")

# ── WAIT FOR RUN ──────────────────────────────────────────────────────────────

if not st.session_state.report_ready:
    st.info("👈  Configure your options above, then click **Run Analysis** in the sidebar.")
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
