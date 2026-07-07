# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

import recon
from recon import (
    WARN_PCT, ALERT_PCT, DIM_LABELS,
    fmt_disc, table_metrics, format_table, encode_tables, decode_tables,
)

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

html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
}}
*, *::before, *::after {{
    transition: background-color 0.2s ease, color 0.2s ease,
                border-color 0.2s ease, box-shadow 0.2s ease;
}}

/* ── NEON AMBIENT GLOW ────────────────────────────────────────────────────── */
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
html {{ background: {BG}; }}
body {{ background: transparent; }}
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
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background: {SURFACE} !important;
    color: {TEXT} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 8px;
    font-size: 0.9rem;
    height: 2.5rem;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {{
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

.format-badge {{
    display: inline-block;
    background: {ACCENT_BG};
    border: 1px solid {ACCENT_RING};
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    margin: 0.1rem 0.2rem 0.1rem 0;
    font-size: 0.72rem;
    font-weight: 600;
    color: {ACCENT};
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

[data-testid="InputInstructions"] {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

URL_WARN_BYTES = 8000

# key → (icon, title, subtitle, on-by-default)
LEVELS = {
    "overview":         ("📊", "Level 1 — Site Overview (full totals)",   "everything in both files, incl. unmatched", True),
    "overview_matched": ("🔗", "Level 1b — Site Overview (matched only)", "paths present on both sides",                True),
    "by_date":          ("📅", "Level 2 — By Date",                        "",                                          True),
    "by_property":      ("🖥️", "Level 3 — Site × Property",               "matched paths",                              True),
    "by_section":       ("🗂️", "Level 4 — Site × Property × Section",     "matched paths",                              True),
    "by_source_group":  ("🏷️", "Level 5 — By Source Group",               "",                                          True),
    "by_adunit":        ("📦", "Level 6 — By Ad Unit",                     "matched · sorted by GAM volume",             True),
}
RECLASSIFY_TARGETS = ["AMAZON", "Price priority", "Ad Exchange", "House", "Standard", "OB"]

# ── CACHED PARSERS ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def parse_any_cached(data: bytes):
    return recon.parse_any(data)


# ── RENDER HELPERS ────────────────────────────────────────────────────────────

def render_table(df: pd.DataFrame, flag_max=True):
    if df is None or df.empty:
        st.caption("No rows.")
        return
    metrics = table_metrics(df)
    st.dataframe(format_table(df), use_container_width=True, hide_index=True)

    if flag_max:
        disc_cols = [f"{m}_disc" for m in metrics]
        raw_disc = pd.to_numeric(df[disc_cols].stack(), errors="coerce").abs()
        max_d = raw_disc.max() if not raw_disc.empty else float("nan")
        n_rill_only = int((df[disc_cols].isna() & (df[[f"Rill_{m}" for m in metrics]].to_numpy() > 0)).to_numpy().sum())
        if n_rill_only:
            st.error(f"🔴 {n_rill_only} cell(s) have Rill volume with zero GAM — investigate mapping.")
        elif not pd.isna(max_d):
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


def render_exec_summary(tables: dict, class_df: pd.DataFrame):
    """The crisp summary: one status line per site + the section classification."""
    metrics = table_metrics(tables["overview"])
    cov = recon.coverage_summary(tables, metrics)

    section("🧭", "Summary", "matched-paths comparison · coverage = share of volume the match covers")
    rows = []
    for _, r in cov.iterrows():
        row = {"Site": r["site"]}
        no_match = r["gam_coverage"] == 0 and r["rill_coverage"] == 0
        worst = 0.0
        for m in metrics:
            d = r[f"{m}_disc"]
            row[f"{recon.METRICS[m][2]} Δ%"] = "—" if no_match else fmt_disc(d, 1 if pd.isna(d) else 0)
            if not pd.isna(d):
                worst = max(worst, abs(d))
        row["GAM coverage"] = f"{r['gam_coverage']:.1f}%"
        row["Rill coverage"] = f"{r['rill_coverage']:.1f}%"
        row["Status"] = "🔴 Investigate" if worst >= ALERT_PCT else ("⚠️ Review" if worst >= WARN_PCT else "✅ OK")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    section("🗂️", "Site & Section Classification", "how each site's ad-unit tree was classified")
    disp = class_df.rename(columns={
        "site": "Site", "property": "Property", "top_level_units": "Top-level Units",
        "sections": "Sections", "slots": "Slots", "in_rill": "Tracked in Rill",
    }).copy()
    disp["Tracked in Rill"] = disp["Tracked in Rill"].map({True: "✅", False: "—"})
    st.dataframe(disp, use_container_width=True, hide_index=True)


def render_report(tables: dict, show_levels: dict, date_range_str: str, sites: list):
    date_range_bar(date_range_str, sites)
    for key, (icon, title, sub, _) in LEVELS.items():
        df = tables.get(key)
        if show_levels.get(key) and df is not None and not df.empty:
            section(icon, title, sub)
            if key == "by_source_group":
                st.caption(
                    "**Mapping:** Price priority → Pre-bid / Price Priority  ·  "
                    "Ad Exchange + OB → ADX + OB  ·  AMAZON → Amazon  ·  "
                    "House → House  ·  Standard → Standard"
                )
            render_table(df)

    # Residue — quantified, never silently dropped.
    rill_um, gam_um = tables.get("rill_unmatched"), tables.get("gam_unmatched")
    if (rill_um is not None and not rill_um.empty) or (gam_um is not None and not gam_um.empty):
        with st.expander("🧩 Unmatched volume — excluded from matched levels above", expanded=False):
            if rill_um is not None and not rill_um.empty:
                st.markdown("**Rill volume with no GAM match** ('Others' = rows Rill itself could not attribute)")
                d = rill_um.copy().rename(columns=DIM_LABELS)
                for m in table_metrics_raw(rill_um, "Rill"):
                    d[f"Rill_{m}"] = d[f"Rill_{m}"].apply(lambda x: f"{int(round(x)):,}") \
                        if recon.METRICS[m][3] == "int" else d[f"Rill_{m}"].apply(lambda x: f"${x:,.2f}")
                    d = d.rename(columns={f"Rill_{m}": f"Rill {recon.METRICS[m][2]}"})
                st.dataframe(d, use_container_width=True, hide_index=True)
            if gam_um is not None and not gam_um.empty:
                st.markdown("**GAM-only inventory** (units Rill doesn't report — expected when Rill tracks a subset)")
                d = gam_um.copy().rename(columns=DIM_LABELS)
                for m in table_metrics_raw(gam_um, "GAM"):
                    d[f"GAM_{m}"] = d[f"GAM_{m}"].apply(lambda x: f"{int(round(x)):,}") \
                        if recon.METRICS[m][3] == "int" else d[f"GAM_{m}"].apply(lambda x: f"${x:,.2f}")
                    d = d.rename(columns={f"GAM_{m}": f"GAM {recon.METRICS[m][2]}"})
                st.dataframe(d, use_container_width=True, hide_index=True)

    # Collapsible issue summary (bottom, scoped to active levels)
    active = {k: tables[k] for k in LEVELS if show_levels.get(k) and k in tables and not tables[k].empty}
    with st.expander(f"🚨 Issue Summary — rows with >{ALERT_PCT:.0f}% discrepancy", expanded=False):
        has_any = False
        for key, df in active.items():
            metrics = table_metrics(df)
            disc = df[[f"{m}_disc" for m in metrics]].apply(pd.to_numeric, errors="coerce")
            rill_only = disc.isna() & (df[[f"Rill_{m}" for m in metrics]].to_numpy() > 0)
            issues = df[(disc.abs() > ALERT_PCT).any(axis=1) | rill_only.any(axis=1)]
            if issues.empty:
                continue
            has_any = True
            st.markdown(
                f"<div style='font-size:0.95rem;font-weight:600;color:{TEXT};margin:0.8rem 0 0.3rem 0'>"
                f"🔴 {LEVELS[key][1]} &nbsp;·&nbsp; "
                f"<span style='color:{ACCENT}'>{len(issues)} row{'s' if len(issues) > 1 else ''} flagged</span></div>",
                unsafe_allow_html=True,
            )
            render_table(issues, flag_max=False)
        if not has_any:
            st.success(f"✅ No discrepancies exceed {ALERT_PCT:.0f}% in any displayed section.")


def table_metrics_raw(df: pd.DataFrame, side: str) -> list:
    return [m for m in recon.METRICS if f"{side}_{m}" in df.columns]


# ── SESSION STATE ─────────────────────────────────────────────────────────────

_ss_defaults = {
    "report_ready":   False,
    "link_generated": False,
    "excl_set":       [],     # list of [site, order]
    "reassignments":  {},     # "site||order" → new type
    "last_save_msg":  "",
    "grid_version":   0,
    "_order_sel":     [],     # last known [site, order] selection
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── BANNER ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="banner">
  <h1>📊 GAM × Rill Reconciliation</h1>
  <p>Upload a GAM + Rill export pair per site. Formats are detected automatically — opportunities, impressions and revenue reports all work.</p>
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
            {k: True for k in LEVELS},
            meta.get("date_range", "—"),
            meta.get("sites", []),
        )
    except Exception as e:
        st.error(f"Could not load shared report. The link may be invalid or corrupted. ({e})")
    st.stop()

# ── SIDEBAR — SITE PAIRS ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📁 Upload Exports")
    st.caption(
        "Dump all your CSVs here — GAM and Rill, one or many sites, in any order. "
        "Files are identified, paired and named automatically from their data."
    )
    uploaded_files = st.file_uploader(
        "CSV exports", type="csv", accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.divider()

    dm_label = "☀️  Light mode" if dm else "🌙  Dark mode"
    new_dm = st.toggle(dm_label, value=dm, key="sidebar_dm")
    if new_dm != dm:
        st.session_state.dark_mode = new_dm
        st.rerun()

    st.divider()

    st.markdown("## 📋 Display Levels")
    st.caption("Levels appear only when the uploaded files contain the needed columns.")
    show_levels = {
        key: st.checkbox(title, value=default, key=f"chk_{key}")
        for key, (_, title, _, default) in LEVELS.items()
    }

    st.divider()

    run_clicked = st.button("▶  Run Analysis", type="primary")
    if run_clicked:
        st.session_state.report_ready = True
        st.session_state.link_generated = False

# ── DETECT, PAIR & VALIDATE ───────────────────────────────────────────────────

problems, gam_files, rill_files = [], [], []
for f in uploaded_files or []:
    try:
        rep = parse_any_cached(f.getvalue())
    except ValueError as e:
        problems.append(f"**{f.name}**: {e}")
        continue
    (gam_files if rep.kind == "gam" else rill_files).append((f.name, rep))

paired, pair_notes = recon.auto_pair(gam_files, rill_files) if (gam_files and rill_files) else ([], [])

pairs = []
for pair, gam_fn, rill_fn, overlap, rill_total in paired:
    if not pair.metrics:
        problems.append(
            f"**{pair.name}**: {gam_fn} and {rill_fn} share no comparable metric — "
            f"GAM has {pair.gam.metrics}, Rill has {pair.rill.metrics}."
        )
        continue
    pairs.append(pair)

for msg in problems:
    st.error(msg)
for msg in pair_notes:
    st.warning(msg)

if not pairs:
    if gam_files and not rill_files:
        st.info("Found only GAM file(s) so far — add the matching Rill export(s).")
    elif rill_files and not gam_files:
        st.info("Found only Rill file(s) so far — add the matching GAM export(s).")
    else:
        st.info("👈  Dump your GAM + Rill CSVs in the sidebar to get started.")
    st.stop()

# Show how files were paired + allow renaming the auto-detected site names.
_pair_meta = {p.name: (g, r, ov, tot) for p, g, r, ov, tot in paired if p in pairs}
with st.expander(f"🔎 File pairing — {len(pairs)} site(s) detected", expanded=False):
    for p in pairs:
        gam_fn, rill_fn, overlap, rill_total = _pair_meta[p.name]
        st.markdown(
            f"**{p.name}** — `{gam_fn}` ⇄ `{rill_fn}` "
            f"<span style='color:{MUTED}'>({overlap}/{rill_total} Rill ad-unit paths matched)</span>",
            unsafe_allow_html=True,
        )
    st.caption("Site names are read from the data (domain or brand prefix). Rename if needed:")
    new_names = {}
    for p in pairs:
        new_names[p.name] = st.text_input(
            f"Name for {p.name}", value=p.name, key=f"rename_{p.name}",
            label_visibility="collapsed",
        )
for p in pairs:
    renamed = new_names.get(p.name, "").strip()
    if renamed:
        p.name = renamed

# ── DATA SUMMARY ──────────────────────────────────────────────────────────────

all_dates = sorted({
    d for p in pairs if "date" in p.dims
    for rep in (p.gam, p.rill) for d in rep.df.get("date", pd.Series(dtype=object)).dropna()
})
if all_dates:
    _fmt = lambda d: d.strftime("%-d %b %Y")
    date_range_str = _fmt(all_dates[0]) if len(all_dates) == 1 else f"{_fmt(all_dates[0])} – {_fmt(all_dates[-1])}"
else:
    date_range_str = "No date dimension in these files"
site_names = [p.name for p in pairs]

section("📌", "Data Summary")
cols = st.columns(2 + len(pairs))
cols[0].metric("Sites", f"{len(pairs)}")
cols[1].metric("Date Range", f"{len(all_dates)} day{'s' if len(all_dates) != 1 else ''}" if all_dates else "—")
for i, p in enumerate(pairs):
    cols[2 + i].metric(p.name, f"{len(p.gam.df):,} / {len(p.rill.df):,}",
                       help="GAM rows / Rill rows")

badges = []
for p in pairs:
    mets = " ".join(f"<span class='format-badge'>{recon.METRICS[m][2]}</span>" for m in p.metrics)
    dims = " ".join(f"<span class='format-badge'>{d.replace('_', ' ')}</span>" for d in sorted(p.dims))
    badges.append(f"<div style='margin:0.2rem 0'><strong style='color:{TEXT}'>{p.name}</strong> — "
                  f"comparing: {mets}{('  ·  dims: ' + dims) if dims else ''}</div>")
st.markdown("\n".join(badges), unsafe_allow_html=True)

# Date-range alignment check — skew here silently corrupts every number.
for p in pairs:
    if "date" in p.gam.dims and "date" in p.rill.dims:
        g_dates = set(p.gam.df["date"].dropna())
        r_dates = set(p.rill.df["date"].dropna())
        if g_dates != r_dates:
            only_g, only_r = sorted(g_dates - r_dates), sorted(r_dates - g_dates)
            st.warning(
                f"⚠️ **{p.name}**: the two files cover different dates — "
                f"GAM-only: {only_g or '—'} · Rill-only: {only_r or '—'}. "
                "Discrepancies will be inflated on those days."
            )

# ── ORDERS (only for revenue-format GAM files) ────────────────────────────────

order_pairs = [p for p in pairs if "order" in p.gam.df.columns]
order_type_map = {}
if order_pairs:
    n_excl = len(st.session_state.excl_set)
    n_rc   = len(st.session_state.reassignments)
    orders_sub = "search · check rows · choose Exclude or Reclassify · Save"
    if n_excl or n_rc:
        parts = []
        if n_excl: parts.append(f"{n_excl} excluded")
        if n_rc:   parts.append(f"{n_rc} reclassified")
        orders_sub += "  ·  " + " · ".join(parts)
    section("📋", "Orders", orders_sub)

    frames = []
    for p in order_pairs:
        df = p.gam.df
        sub = df[df["order"].notna() & (df["order"] != "OB")]
        if sub.empty:
            continue
        agg = sub.groupby(["order", "line_item_type"])[p.gam.metrics].sum().reset_index()
        agg.insert(0, "site", p.name)
        frames.append(agg)
        order_type_map.update(
            sub.drop_duplicates("order").set_index("order")["line_item_type"].to_dict()
        )

    if frames:
        orders_df = pd.concat(frames, ignore_index=True)
        orders_df["Bucket"] = orders_df["line_item_type"].map(recon.GAM_GROUP).fillna("—")
        excl_cur = {(s, o) for s, o in st.session_state.excl_set}
        orders_df["Status"] = [
            "🚫 Excluded" if (s, o) in excl_cur
            else ("🔄 Reclassified" if f"{s}||{o}" in st.session_state.reassignments else "")
            for s, o in zip(orders_df["site"], orders_df["order"])
        ]
        metric_cols = [m for m in recon.METRICS if m in orders_df.columns]
        sort_metric = metric_cols[0] if metric_cols else "order"
        orders_df = orders_df.sort_values(["Bucket", sort_metric], ascending=[True, False]).reset_index(drop=True)

        grid_df = orders_df.rename(columns={
            "site": "Site", "order": "Order", "line_item_type": "Type",
            **{m: recon.METRICS[m][2] for m in metric_cols},
        })[["Site", "Order", "Type", "Bucket", *[recon.METRICS[m][2] for m in metric_cols], "Status"]]

        ca, cb, cc = st.columns([2, 3, 1])
        with ca:
            action_choice = st.radio("Action", ["Exclude", "Reclassify"], horizontal=True,
                                     key="ord_action", label_visibility="collapsed")
        with cb:
            if action_choice == "Reclassify":
                reclass_to = st.selectbox("Move to", RECLASSIFY_TARGETS, key="ord_reclass_to",
                                          label_visibility="collapsed")
            else:
                st.caption("Selected orders will be removed from all comparisons.")
                reclass_to = None
        with cc:
            save_clicked = st.button("💾 Save", type="primary", key="ord_save")

        if st.session_state.last_save_msg:
            st.success(st.session_state.last_save_msg)

        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
        gb.configure_default_column(sortable=True, resizable=True, filter="agTextColumnFilter")
        gb.configure_column("Order", min_width=280, flex=4, floatingFilter=True,
                            filterParams={"filterOptions": ["contains"], "defaultOption": "contains",
                                          "suppressAndOrCondition": True})
        gb.configure_column("Site", min_width=120, flex=1)
        for m in metric_cols:
            label, kind = recon.METRICS[m][2], recon.METRICS[m][3]
            fmt = ("'$' + Number(value).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2})"
                   if kind == "money" else "Number(value).toLocaleString('en-US')")
            gb.configure_column(label, min_width=115, flex=1, filter="agNumberColumnFilter",
                                type=["numericColumn"], valueFormatter=fmt)
        gb.configure_column("Status", min_width=130, flex=1, filter=False, sortable=False)
        gb.configure_grid_options(rowHeight=38, headerHeight=42, suppressMovableColumns=True, animateRows=True)
        grid_opts = gb.build()

        ag_key = f"orders_grid_{st.session_state.grid_version}"
        response = AgGrid(
            grid_df, gridOptions=grid_opts,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=420, theme="streamlit", fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True, key=ag_key,
        )

        _sel_raw = response.get("selected_rows")
        if isinstance(_sel_raw, pd.DataFrame):
            _current_sel = list(zip(_sel_raw["Site"], _sel_raw["Order"])) if not _sel_raw.empty else []
        elif _sel_raw:
            _current_sel = [(r["Site"], r["Order"]) for r in _sel_raw]
        else:
            _current_sel = []
        if not save_clicked:
            st.session_state._order_sel = [list(t) for t in _current_sel]

        if save_clicked:
            sel = _current_sel or [tuple(t) for t in st.session_state._order_sel]
            if sel:
                names = ", ".join(o for _, o in sel[:3]) + (f" +{len(sel) - 3} more" if len(sel) > 3 else "")
                if action_choice == "Exclude":
                    merged = {tuple(t) for t in st.session_state.excl_set} | set(sel)
                    st.session_state.excl_set = [list(t) for t in merged]
                    st.session_state.last_save_msg = f"✅ Excluded {len(sel)} order(s): {names}"
                else:
                    for s, o in sel:
                        st.session_state.reassignments[f"{s}||{o}"] = reclass_to
                    st.session_state.last_save_msg = f"✅ Reclassified {len(sel)} order(s) → {reclass_to}: {names}"
                st.session_state.grid_version += 1
                st.rerun()
            else:
                st.warning("No rows selected — tick the checkbox on the left of each row, then click 💾 Save.")

        # Pending-changes review
        n_excl = len(st.session_state.excl_set)
        n_rc = len(st.session_state.reassignments)
        if n_excl + n_rc:
            with st.expander(f"Review & undo changes ({n_excl + n_rc} total)", expanded=False):
                undo_excl, undo_rc = [], []
                for s, o in st.session_state.excl_set:
                    c1, c2 = st.columns([9, 1])
                    c1.markdown(f"🚫 {s} — {o}")
                    if c2.button("Undo", key=f"undo_e_{s}_{o}"):
                        undo_excl.append([s, o])
                for key_, new_t in list(st.session_state.reassignments.items()):
                    s, o = key_.split("||", 1)
                    c1, c2 = st.columns([9, 1])
                    c1.markdown(f"🔄 {s} — {o}: {order_type_map.get(o, '?')} → **{new_t}**")
                    if c2.button("Undo", key=f"undo_r_{key_}"):
                        undo_rc.append(key_)
                st.divider()
                if st.button("🗑 Clear ALL changes", key="clear_all"):
                    st.session_state.excl_set = []
                    st.session_state.reassignments = {}
                    st.session_state.last_save_msg = ""
                    st.session_state.grid_version += 1
                    st.rerun()
                if undo_excl:
                    st.session_state.excl_set = [t for t in st.session_state.excl_set if t not in undo_excl]
                    st.session_state.grid_version += 1
                    st.rerun()
                if undo_rc:
                    for key_ in undo_rc:
                        del st.session_state.reassignments[key_]
                    st.session_state.grid_version += 1
                    st.rerun()


def apply_order_changes(pairs_in):
    """Order exclusions/reclassifications → new SitePairs (source data untouched)."""
    excl = {tuple(t) for t in st.session_state.excl_set}
    reass = st.session_state.reassignments
    if not excl and not reass:
        return pairs_in
    out = []
    for p in pairs_in:
        df = p.gam.df
        if "order" not in df.columns:
            out.append(p)
            continue
        df = df.copy()
        for key_, new_t in reass.items():
            s, o = key_.split("||", 1)
            if s == p.name:
                df.loc[df["order"] == o, "line_item_type"] = new_t
        df["source_group"] = df["line_item_type"].map(recon.GAM_GROUP)
        drop = {o for s, o in excl if s == p.name}
        if drop:
            df = df[~df["order"].isin(drop)]
        out.append(recon.SitePair(
            p.name,
            recon.Report("gam", df, p.gam.metrics, p.gam.dims, p.gam.warnings),
            p.rill,
        ))
    return out


# ── WAIT FOR RUN ──────────────────────────────────────────────────────────────

if not st.session_state.report_ready:
    st.info("👈  When ready, click **Run Analysis** in the sidebar to generate the report.")
    st.stop()

# ── BUILD & RENDER ────────────────────────────────────────────────────────────

pairs_eff = apply_order_changes(pairs)
all_tables = recon.build_tables(pairs_eff)
class_df = recon.classification_summary(pairs_eff)

render_exec_summary(all_tables, class_df)

# ── SHARE SECTION ─────────────────────────────────────────────────────────────

section("🔗", "Share Report")

available_levels = [k for k in LEVELS if k in all_tables and not all_tables[k].empty]
share_selection = st.multiselect(
    "Select tables to include in the shared link:",
    options=available_levels,
    format_func=lambda k: LEVELS[k][1],
    default=[k for k in available_levels if LEVELS[k][3]],
    key="share_levels",
)

if share_selection and st.button("🔗  Generate Shareable Link", type="primary"):
    meta = {
        "date_range":   date_range_str,
        "sites":        site_names,
        "generated_at": pd.Timestamp.now().strftime("%-d %b %Y, %H:%M"),
    }
    encoded = encode_tables({k: all_tables[k] for k in share_selection}, meta)
    if len(encoded) > URL_WARN_BYTES:
        st.warning(
            f"⚠️ This link is {len(encoded):,} characters — some browsers and chat apps truncate long URLs. "
            "Deselect the Ad Unit level (usually the largest) for a safer link."
        )
    st.session_state.link_generated = True
    st.query_params["r"] = encoded

if st.session_state.link_generated:
    st.success(
        "✅ Shareable link generated! Copy the URL from your browser's address bar — "
        "recipients see the full report without uploading any files."
    )
    st.caption("💡 The link encodes your report data. Anyone with it can open the tables.")
elif not share_selection:
    st.warning("Select at least one table above to generate a link.")

# ── REPORT ────────────────────────────────────────────────────────────────────

render_report(all_tables, show_levels, date_range_str, site_names)
