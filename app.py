import json
import zlib
import base64

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

st.set_page_config(
    page_title="GAM × Rill Reconciliation",
    page_icon="📊",
    layout="wide",
)

# ── THEME (must come before CSS so dm is available) ───────────────────────────

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
dm = st.session_state.dark_mode

# Palette
BG          = "#0f172a" if dm else "#ffffff"
SURFACE     = "#1e293b" if dm else "#f8fafc"
SURFACE2    = "#334155" if dm else "#f1f5f9"
BORDER      = "#334155" if dm else "#e2e8f0"
TEXT        = "#f1f5f9" if dm else "#1e293b"
MUTED       = "#94a3b8" if dm else "#64748b"
ACCENT      = "#60a5fa" if dm else "#2563a8"
ACCENT_BG   = "#172554" if dm else "#eff6ff"
ACCENT_RING = "#1d4ed8" if dm else "#bfdbfe"
BANNER_A    = "#1e3a8a" if dm else "#1e40af"
BANNER_B    = "#2563a8" if dm else "#3b82f6"
SHADOW      = "rgba(0,0,0,0.35)" if dm else "rgba(0,0,0,0.06)"
SUCCESS_BG  = "#052e16" if dm else "#f0fdf4"
SUCCESS_BDR = "#166534" if dm else "#86efac"
SUCCESS_TXT = "#4ade80" if dm else "#15803d"
NEON1       = "rgba(96,165,250,0.65)"  if dm else "rgba(59,130,246,0.45)"
NEON2       = "rgba(167,139,250,0.55)" if dm else "rgba(139,92,246,0.38)"
NEON3       = "rgba(34,211,238,0.45)"  if dm else "rgba(6,182,212,0.32)"

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── BASE ─────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
*, *::before, *::after {{
    transition: background-color 0.2s ease, color 0.2s ease,
                border-color 0.2s ease, box-shadow 0.2s ease;
}}

/* ── NEON AMBIENT GLOW ────────────────────────────────────────────────────── */
/* html::before at z-index:-1 sits BEHIND everything.                         */
/* .stApp / .main / .block-container are transparent so the neon shows through */
html::before {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: -1;
    background:
        radial-gradient(ellipse 700px 700px at -6% -8%,  {NEON1} 0%, transparent 60%),
        radial-gradient(ellipse 600px 600px at 108% 36%, {NEON2} 0%, transparent 60%),
        radial-gradient(ellipse 560px 560px at 42% 108%, {NEON3} 0%, transparent 60%);
    animation: neonBreathe 10s ease-in-out infinite;
}}
@keyframes neonBreathe {{
    0%,100% {{ opacity: 1; }}
    50%      {{ opacity: 0.5; }}
}}

/* ── APP BACKGROUND ───────────────────────────────────────────────────────── */
html, body {{ background: {BG}; }}
/* Transparent so html::before neon shows through */
.stApp,
[data-testid="stAppViewContainer"],
.main {{
    background: transparent !important;
}}
.main .block-container {{
    background: transparent !important;
    padding-top: 1.5rem;
}}
[data-testid="stHeader"] {{ background-color: {BG} !important; }}
[data-testid="stDecoration"] {{ display: none; }}

/* ── SIDEBAR ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div:not(.stButton) {{
    color: {TEXT} !important;
}}

/* ── TEXT ─────────────────────────────────────────────────────────────────── */
p, .stMarkdown p, div.stMarkdown {{ color: {TEXT}; }}
small, .stCaption, [data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}
h1, h2, h3, h4 {{ color: {TEXT} !important; }}
a {{ color: {ACCENT} !important; }}

/* ── METRICS ──────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 2px 6px {SHADOW};
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.72rem !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
[data-testid="stMetricValue"] {{
    font-size: 1.55rem !important;
    color: {TEXT} !important;
    font-weight: 700 !important;
}}

/* ── BUTTONS ──────────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {{
    background: {ACCENT} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.55rem 1.2rem;
    width: 100%;
    box-shadow: 0 2px 8px {ACCENT_RING};
}}
.stButton > button[kind="primary"]:hover {{ filter: brightness(1.12); }}
.stButton > button:not([kind="primary"]) {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
    font-weight: 500;
}}
.stButton > button:not([kind="primary"]):hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    background: {ACCENT_BG} !important;
}}

/* ── INPUTS ───────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 8px;
    font-size: 0.9rem;
    height: 2.5rem;
}}
[data-testid="stTextInput"] input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px {ACCENT_RING} !important;
}}
[data-testid="stTextInput"] input::placeholder {{ color: {MUTED} !important; }}

/* ── SELECT / MULTISELECT ─────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div {{
    background: {SURFACE} !important;
    border-color: {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 8px;
}}
[data-testid="stSelectbox"] span,
[data-testid="stMultiSelect"] span {{ color: {TEXT} !important; }}

/* ── FILE UPLOADER ────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background: {SURFACE} !important;
    border-radius: 10px;
    padding: 0.3rem;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE2} !important;
    border: 2px dashed {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: {TEXT} !important; }}
[data-testid="stFileUploaderDropzone"] button {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 6px;
}}
[data-testid="stFileUploaderDropzone"] svg {{ fill: {MUTED} !important; }}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {{ color: {MUTED} !important; }}

/* ── EXPANDERS ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px;
    background: {SURFACE} !important;
    overflow: hidden;
}}
[data-testid="stExpander"] > details > summary {{ background: {SURFACE} !important; }}
[data-testid="stExpander"] > details > summary > span {{ color: {TEXT} !important; }}
[data-testid="stExpander"] > details > div {{ background: {BG} !important; }}

/* ── DATAFRAME ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] > div {{
    border-radius: 10px;
    border: 1px solid {BORDER} !important;
    overflow: hidden;
    box-shadow: 0 1px 4px {SHADOW};
}}

/* ── RADIO / CHECKBOX / TOGGLE ────────────────────────────────────────────── */
[data-testid="stRadio"] label span,
[data-testid="stRadio"] p,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] p,
[data-testid="stToggle"] label span,
[data-testid="stToggle"] p {{ color: {TEXT} !important; }}

/* ── ALERTS ───────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{ border-radius: 10px !important; }}

/* ── DIVIDERS ─────────────────────────────────────────────────────────────── */
hr {{ border-color: {BORDER} !important; margin: 1.2rem 0; }}

/* ── CUSTOM COMPONENTS ────────────────────────────────────────────────────── */
.banner {{
    background: linear-gradient(135deg, {BANNER_A} 0%, {BANNER_B} 100%);
    padding: 1.6rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px {'rgba(30,58,138,0.55)' if dm else 'rgba(37,99,168,0.22)'};
    position: relative;
    overflow: hidden;
}}
.banner::before {{
    content: '';
    position: absolute; top: -40%; right: -5%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.banner h1 {{
    margin: 0 0 0.3rem 0;
    font-size: 1.75rem;
    color: #ffffff !important;
    font-weight: 700;
    letter-spacing: -0.02em;
}}
.banner p {{ margin: 0; color: #93c5fd; font-size: 0.9rem; }}

.section-label {{
    display: flex; align-items: center; gap: 0.6rem;
    background: {ACCENT_BG};
    border-left: 3px solid {ACCENT};
    padding: 0.65rem 1.1rem;
    border-radius: 0 10px 10px 0;
    margin: 2rem 0 0.8rem 0;
    font-size: 0.92rem;
    font-weight: 600;
    color: {ACCENT};
    letter-spacing: 0.01em;
}}

.date-bar {{
    background: {ACCENT_BG};
    border: 1px solid {ACCENT_RING};
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 1.2rem 0 0.5rem 0;
    display: flex; align-items: center; gap: 1.2rem;
}}

.shared-badge {{
    background: {SUCCESS_BG};
    border: 1px solid {SUCCESS_BDR};
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    color: {SUCCESS_TXT};
    font-weight: 600; font-size: 0.9rem;
}}

.action-panel {{
    background: {ACCENT_BG};
    border: 1.5px solid {ACCENT_RING};
    border-radius: 12px;
    padding: 1rem 1.3rem;
    margin-bottom: 0.8rem;
}}

.search-box-wrap {{
    position: relative;
    margin-bottom: 0.5rem;
}}

/* ── AG GRID DARK MODE ────────────────────────────────────────────────────── */
.ag-theme-streamlit {{
    --ag-background-color: {SURFACE} !important;
    --ag-header-background-color: {SURFACE2} !important;
    --ag-odd-row-background-color: {BG} !important;
    --ag-foreground-color: {TEXT} !important;
    --ag-header-foreground-color: {TEXT} !important;
    --ag-secondary-foreground-color: {MUTED} !important;
    --ag-border-color: {BORDER} !important;
    --ag-row-hover-color: {ACCENT_BG} !important;
    --ag-selected-row-background-color: {ACCENT_BG} !important;
    --ag-checkbox-checked-color: {ACCENT} !important;
    --ag-range-selection-border-color: {ACCENT} !important;
    --ag-font-family: 'Inter', 'Segoe UI', sans-serif !important;
    --ag-font-size: 13px !important;
}}
.ag-theme-streamlit .ag-header-cell-label,
.ag-theme-streamlit .ag-cell {{ color: {TEXT} !important; }}
.ag-theme-streamlit .ag-paging-panel {{ color: {MUTED} !important; background: {SURFACE} !important; }}
.ag-theme-streamlit .ag-root-wrapper {{ border-radius: 10px; overflow: hidden; border: 1px solid {BORDER} !important; }}

/* ── HIDE "Press Enter to apply" STREAMLIT HINT ──────────────────────────── */
[data-testid="InputInstructions"] {{ display: none !important; }}

</style>
""", unsafe_allow_html=True)

_neon_grad_end = "#1e3a5f" if dm else "#dbeafe"

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
        f"<span style='font-size:0.8rem;color:{MUTED};font-weight:400;margin-left:0.5rem'>{subtitle}</span>"
        if subtitle else ""
    )
    st.markdown(f'<div class="section-label">{icon} {title}{sub}</div>', unsafe_allow_html=True)


def date_range_bar(date_range_str, sites):
    st.markdown(f"""
    <div class="date-bar">
      <span style="font-size:1.5rem">📅</span>
      <div>
        <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Analysis Period</div>
        <div style="font-size:1.2rem;font-weight:700;color:{ACCENT}">{date_range_str}</div>
      </div>
      <div style="margin-left:auto;text-align:right">
        <div style="font-size:0.7rem;color:{MUTED};font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Sites</div>
        <div style="font-size:0.9rem;font-weight:600;color:{TEXT}">{" · ".join(sites)}</div>
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
    "excl_set":       [],
    "reassignments":  {},
    "last_save_msg":  "",
    "grid_version":   0,   # increments on each Save → grid remounts so Status column refreshes
    "_order_sel":     [],  # last known selection from SELECTION_CHANGED rerun
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

    # Dark / Light mode toggle
    dm_label = "☀️  Light mode" if dm else "🌙  Dark mode"
    new_dm = st.toggle(dm_label, value=dm, key="sidebar_dm")
    if new_dm != dm:
        st.session_state.dark_mode = new_dm
        st.rerun()

    st.divider()

    st.markdown("## 📋 Display Levels")
    st.caption("Choose which report tables appear on screen.")
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

order_type_map = (
    gam[gam["Order"].notna() & (gam["Order"] != "OB")]
    .drop_duplicates("Order")
    .set_index("Order")["Line item type"].to_dict()
)

# ── SIDEBAR — CHANGES PANEL (added after order_type_map is available) ─────────

with st.sidebar:
    n_excl_sb = len(st.session_state.excl_set)
    n_rc_sb   = len(st.session_state.reassignments)

    st.divider()
    st.markdown("## 📋 Pending Changes")

    if n_excl_sb == 0 and n_rc_sb == 0:
        st.caption("No changes yet — use the Orders table to exclude or reclassify orders before running analysis.")
    else:
        if n_excl_sb:
            st.markdown(f"**🚫 Excluded ({n_excl_sb})**")
            for o in st.session_state.excl_set:
                st.caption(f"• {o}")

        if n_rc_sb:
            st.markdown(f"**🔄 Reclassified ({n_rc_sb})**")
            for o, new_t in st.session_state.reassignments.items():
                orig = order_type_map.get(o, "?")
                st.caption(f"• {o}  →  {new_t}")

        if st.button("🗑 Clear all", key="sb_clear_all"):
            st.session_state.excl_set = []
            st.session_state.reassignments = {}
            st.session_state.last_save_msg = ""
            st.session_state.grid_version += 1
            st.rerun()

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
orders_sub = "search · check rows · choose Exclude or Reclassify · Save"
if n_excl or n_rc:
    parts = []
    if n_excl: parts.append(f"{n_excl} excluded")
    if n_rc:   parts.append(f"{n_rc} reclassified")
    orders_sub += "  ·  " + " · ".join(parts)

section("📋", "Orders", orders_sub)

# Build effective table (reflects current reassignments)
gam_eff = gam.copy()
for order, new_type in st.session_state.reassignments.items():
    gam_eff.loc[gam_eff["Order"] == order, "Line item type"] = new_type
gam_eff["source_group"] = gam_eff["Line item type"].map(GAM_GROUP)

orders_df = (
    gam_eff[gam_eff["Order"].notna() & (gam_eff["Order"] != "OB")]
    .groupby(["Order", "Line item type"])
    .agg(Revenue=("Total CPM and CPC revenue", "sum"),
         Impressions=("Total impressions", "sum"))
    .reset_index()
)
orders_df["Bucket"] = orders_df["Line item type"].map(GAM_GROUP).fillna("—")
excl_set_cur = set(st.session_state.excl_set)
orders_df["Status"] = orders_df["Order"].apply(
    lambda o: "🚫 Excluded" if o in excl_set_cur
    else ("🔄 Reclassified" if o in st.session_state.reassignments else "")
)
orders_df = orders_df.sort_values(["Bucket", "Revenue"], ascending=[True, False]).reset_index(drop=True)

grid_df = orders_df[["Order", "Line item type", "Bucket", "Revenue", "Impressions", "Status"]].copy()
grid_df = grid_df.rename(columns={"Line item type": "Type"})

# ── Action panel ──────────────────────────────────────────────────────────────

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
    save_clicked = st.button("💾 Save", type="primary", key="ord_save")

# ── Last save confirmation (persists until next save replaces it) ─────────────

if st.session_state.last_save_msg:
    st.success(st.session_state.last_save_msg)

# ── Instruction banner for real-time search ───────────────────────────────────

st.markdown(
    f"<div style='background:{ACCENT_BG};border:1px solid {ACCENT_RING};"
    f"border-radius:8px;padding:0.55rem 1rem;margin:0.6rem 0 0.4rem 0;"
    f"font-size:0.88rem;color:{ACCENT}'>"
    f"🔍 <strong>Type directly in the search row inside the table</strong> "
    f"(the blue row below the column headers) — results filter instantly as you type, "
    f"no Enter needed.&nbsp; Header ☑ selects all visible rows."
    f"</div>",
    unsafe_allow_html=True,
)

# ── AG Grid ───────────────────────────────────────────────────────────────────
# Floating filter on Order column = real-time client-side JS filter, no Enter.
# Key only includes grid_version (no search term — filter is handled client-side).
# On Save: grid_version increments → grid remounts → Status column refreshes.
# Between user typing/selecting and clicking Save, key is STABLE so the
# component keeps its internal floating filter state and selection.

gb = GridOptionsBuilder.from_dataframe(grid_df)
gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
gb.configure_default_column(
    sortable=True, resizable=True,
    filter="agTextColumnFilter",
    floatingFilter=False,
)
gb.configure_column(
    "Order", headerName="Order  ·  🔍 type to search", min_width=300, flex=4,
    filter="agTextColumnFilter",
    floatingFilter=True,
    filterParams={
        "filterOptions": ["contains"],
        "defaultOption": "contains",
        "suppressAndOrCondition": True,
    },
)
gb.configure_column("Type",   min_width=150, flex=2)
gb.configure_column("Bucket", min_width=190, flex=2)
gb.configure_column(
    "Revenue", min_width=115, flex=1,
    filter="agNumberColumnFilter",
    type=["numericColumn"],
    valueFormatter="'$' + Number(value).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})",
)
gb.configure_column(
    "Impressions", min_width=120, flex=1,
    filter="agNumberColumnFilter",
    type=["numericColumn"],
    valueFormatter="Number(value).toLocaleString('en-US')",
)
gb.configure_column("Status", min_width=130, flex=1, filter=False, sortable=False)
gb.configure_grid_options(
    rowHeight=38,
    headerHeight=42,
    floatingFiltersHeight=72,
    suppressMovableColumns=True,
    animateRows=True,
    tooltipShowDelay=300,
)
grid_opts = gb.build()

# JsCode runs INSIDE the AG Grid component iframe — direct DOM access, no
# iframe boundary to cross. Styles the floating filter row and input element.
_on_grid_ready = JsCode(f"""
function(params) {{
    var attempts = 0;
    function applyStyles() {{
        var row = document.querySelector('.ag-header-row-floating-filter');
        var inputs = document.querySelectorAll('.ag-floating-filter-full-body input');
        if ((!row || inputs.length === 0) && attempts < 30) {{
            attempts++;
            setTimeout(applyStyles, 100);
            return;
        }}
        if (row) {{
            row.style.height = '72px';
            row.style.background = 'linear-gradient(90deg,{ACCENT_BG} 0%,{_neon_grad_end} 100%)';
            row.style.borderTop = '3px solid {ACCENT}';
        }}
        var wrappers = document.querySelectorAll(
            '.ag-floating-filter-full-body .ag-wrapper,' +
            '.ag-floating-filter-full-body .ag-input-wrapper,' +
            '.ag-floating-filter-full-body .ag-text-field-input-wrapper'
        );
        wrappers.forEach(function(w) {{
            w.style.height = '52px';
            w.style.border = '2.5px solid {ACCENT}';
            w.style.borderRadius = '12px';
            w.style.background = '{BG}';
            w.style.boxShadow = '0 2px 20px {ACCENT_RING}';
            w.style.paddingLeft = '16px';
            w.style.display = 'flex';
            w.style.alignItems = 'center';
        }});
        inputs.forEach(function(inp) {{
            inp.style.height = '100%';
            inp.style.fontSize = '1rem';
            inp.style.fontWeight = '500';
            inp.style.color = '{TEXT}';
            inp.style.background = 'transparent';
            inp.style.border = 'none';
            inp.style.boxShadow = 'none';
            inp.style.outline = 'none';
            inp.style.width = '100%';
            inp.placeholder = '🔍  Type to search orders…';
        }});
    }}
    setTimeout(applyStyles, 80);
}}
""")
grid_opts["onGridReady"] = _on_grid_ready

ag_key = f"orders_grid_{st.session_state.grid_version}"

# SELECTION_CHANGED fires a Python rerun the moment the user checks/unchecks
# a row, so we capture selection into session state immediately.
# On the Save-button rerun the response may be empty (grid re-renders first),
# so we fall back to the last stored selection from _order_sel.
response = AgGrid(
    grid_df,
    gridOptions=grid_opts,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    height=460,
    theme="streamlit",
    fit_columns_on_grid_load=True,
    allow_unsafe_jscode=True,
    key=ag_key,
)

# Always parse current response selection
_sel_raw = response.get("selected_rows") or []
if isinstance(_sel_raw, pd.DataFrame):
    _current_sel = _sel_raw["Order"].tolist() if not _sel_raw.empty else []
else:
    _current_sel = [r["Order"] for r in _sel_raw] if _sel_raw else []

# Persist selection; only overwrite when not the Save rerun (Save rerun may
# briefly return empty before the grid re-renders with updated Status).
if not save_clicked:
    st.session_state._order_sel = _current_sel

# ── Handle Save ───────────────────────────────────────────────────────────────

if save_clicked:
    # Prefer live response; fall back to last stored selection
    sel_orders = _current_sel or st.session_state._order_sel

    if sel_orders:
        names = ", ".join(sel_orders[:3]) + (f" +{len(sel_orders)-3} more" if len(sel_orders) > 3 else "")
        if action_choice == "Exclude":
            st.session_state.excl_set = list(set(st.session_state.excl_set + sel_orders))
            st.session_state.last_save_msg = f"✅ Excluded {len(sel_orders)} order(s): {names}"
        else:
            for o in sel_orders:
                st.session_state.reassignments[o] = reclass_to
            st.session_state.last_save_msg = f"✅ Reclassified {len(sel_orders)} order(s) → {reclass_to}: {names}"
        st.session_state.grid_version += 1   # force grid remount so Status column refreshes
        st.rerun()
    else:
        st.warning(
            "No rows selected — tick the checkbox on the left side of each row, "
            "then click 💾 Save."
        )

# ── Changes summary (always visible, QA before Run Analysis) ──────────────────

n_excl = len(st.session_state.excl_set)
n_rc   = len(st.session_state.reassignments)
total_changes = n_excl + n_rc

st.divider()

if total_changes == 0:
    st.caption(
        "No changes yet. Use the table above to exclude orders from the analysis "
        "or move them to a different bucket."
    )
else:
    badge_parts = []
    if n_excl: badge_parts.append(f"🚫 {n_excl} excluded")
    if n_rc:   badge_parts.append(f"🔄 {n_rc} reclassified")
    st.markdown(
        f"<div style='background:{ACCENT_BG};border:1.5px solid {ACCENT_RING};"
        f"border-radius:10px;padding:0.7rem 1.1rem;margin-bottom:0.5rem;"
        f"font-weight:600;color:{ACCENT};font-size:0.92rem'>"
        f"📋 Pending changes — {' · '.join(badge_parts)} — "
        f"these will be applied when you click <strong>▶ Run Analysis</strong></div>",
        unsafe_allow_html=True,
    )

    exp_label = f"Review & undo changes ({total_changes} total)"
    with st.expander(exp_label, expanded=True):
        undo_excl, undo_rc = [], []

        if st.session_state.excl_set:
            st.markdown("**Exclusions** — these orders are removed from all comparisons")
            for o in st.session_state.excl_set:
                c1, c2 = st.columns([9, 1])
                c1.markdown(f"🚫 {o}")
                if c2.button("Undo", key=f"undo_e_{o}"): undo_excl.append(o)

        if st.session_state.reassignments:
            if st.session_state.excl_set:
                st.markdown("")
            st.markdown("**Reclassifications** — these orders are moved to a different bucket")
            for o, new_t in list(st.session_state.reassignments.items()):
                orig = order_type_map.get(o, "?")
                c1, c2 = st.columns([9, 1])
                c1.markdown(f"🔄 {o} — {orig} → **{new_t}**")
                if c2.button("Undo", key=f"undo_r_{o}"): undo_rc.append(o)

        st.divider()
        if st.button("🗑 Clear ALL changes", key="clear_all"):
            st.session_state.excl_set = []
            st.session_state.reassignments = {}
            st.session_state.last_save_msg = ""
            st.session_state.grid_version += 1
            st.rerun()

        if undo_excl:
            st.session_state.excl_set = [o for o in st.session_state.excl_set if o not in undo_excl]
            st.session_state.grid_version += 1
            st.rerun()
        if undo_rc:
            for o in undo_rc: del st.session_state.reassignments[o]
            st.session_state.grid_version += 1
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
