"""Data + presentation-formatting layer for GAM × Rill reconciliation.

Pure logic — no Streamlit. Everything here is grounded in the observed shape
of real exports (GB News and WeatherBug samples, Jun 2026):

GAM ("Ad Manager Report ...csv")
    Always: "Ad unit (all levels)" (display path, '»'-separated) and
    "Ad unit code level N" columns (the code decomposition of the same path;
    column order in the file is NOT guaranteed to follow N).
    Metrics vary by report: "Programmatic eligible ad requests" and/or
    "Total impressions" / "Total CPM and CPC revenue".
    Optional dims: "Date", "Line item type", "Order".

Rill ("holistic_revenue_...csv")
    Always: "Ad Unit" — '<network id>/<code path>' with or without a leading
    slash, plus an unattributable "Others" row. The code path after the id
    matches GAM's level codes 1:1 (verified 100% on both publishers).
    Metrics vary: "Total Opportunities" and/or "Total Impressions"/"Revenue".
    Optional dims: "Ts (day)", "Revenue Source Type", "Domain".

A "site" is never inferred from paths — each GAM+Rill pair IS one site,
named by the user. Within a site the hierarchy is:
    property (classified top-level unit) → section → slot.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

WARN_PCT = 5.0
ALERT_PCT = 10.0

# ── CANONICAL METRICS ─────────────────────────────────────────────────────────
# canonical name → (GAM column, Rill column, display label, kind)
METRICS = {
    "opportunities": ("Programmatic eligible ad requests", "Total Opportunities", "Opportunities", "int"),
    "impressions":   ("Total impressions",                 "Total Impressions",   "Impressions",   "int"),
    "revenue":       ("Total CPM and CPC revenue",         "Revenue",             "Revenue",       "money"),
}

GAM_DATE, RILL_DATE = "Date", "Ts (day)"
GAM_TYPE_COL, GAM_ORDER_COL, RILL_TYPE_COL = "Line item type", "Order", "Revenue Source Type"

# Source-group buckets (only used when the classification dims are present)
GAM_GROUP = {
    "Price priority": "Pre-bid / Price Priority",
    "Ad Exchange":    "ADX + OB",
    "OB":             "ADX + OB",
    "AMAZON":         "Amazon",
    "House":          "House",
    "Standard":       "Standard",
}
RILL_GROUP = {
    "Prebid":         "Pre-bid / Price Priority",
    "Price Priority": "Pre-bid / Price Priority",
    "OB-ADX":         "ADX + OB",
    "ADX":            "ADX + OB",
    "OB":             "ADX + OB",
    "Amazon":         "Amazon",
    "House":          "House",
    "Standard":       "Standard",
}

OTHERS_LABEL = "Others"


# ── READING ───────────────────────────────────────────────────────────────────

def read_csv_any(data: bytes) -> pd.DataFrame:
    """Read a CSV with unknown encoding; dimensions stay strings.

    GAM downloads are often cp1252 (the '»' separator), Rill is utf-8.
    dtype=str keeps numeric-looking ad-unit codes (e.g. '5926448863') from
    becoming floats; metric columns are converted later.
    """
    last_err = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, dtype=str)
            break
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    else:
        raise ValueError(f"Could not decode file: {last_err}")
    df.columns = df.columns.str.strip()
    return df


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True),
        errors="coerce",
    ).fillna(0)


# ── PATH KEYS (verified: joins 100% of Rill rows on both publishers) ──────────

def gam_level_columns(columns) -> list:
    """'Ad unit code level N' columns present, ordered by N (file order lies:
    the GB export lists level 6 before level 5)."""
    numbered = []
    for col in columns:
        m = re.fullmatch(r"Ad unit code level (\d+)", str(col).strip())
        if m:
            numbered.append((int(m.group(1)), col))
    numbered.sort(key=lambda x: x[0])
    return [c for _, c in numbered]


def _clean_code(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return None
    s = str(v).strip()
    if s == "":
        return None
    if re.fullmatch(r"-?\d+\.0", s):  # float artifact on purely-numeric codes
        s = s[:-2]
    return s.lower()


def gam_key_path(row, level_cols: list) -> str:
    parts = [c for c in (_clean_code(row[col]) for col in level_cols) if c is not None]
    if not parts:
        leaf = _clean_code(row.get("Ad unit code"))
        if leaf is not None:
            parts = [leaf]
    return "/".join(parts)


def rill_key_path(val) -> str:
    """Drop the leading network-id segment; '' for blank/'Others' rows."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if s == "" or s == OTHERS_LABEL:
        return ""
    parts = s.strip("/").split("/")
    if len(parts) <= 1:
        return ""
    return "/".join(p.lower() for p in parts[1:])


# ── PROPERTY CLASSIFICATION ───────────────────────────────────────────────────
# Order matters: AdX before app-rules ('ca-mb-app-pub…' contains 'app'),
# video before android ('Primis_Video_Android' is video inventory).

PROPERTY_RULES = [
    (re.compile(r"ca-.*pub-", re.I),          "Ad Exchange In-App"),
    (re.compile(r"video", re.I),              "Video"),
    (re.compile(r"android.*tablet|tablet.*android", re.I), "App — Android Tablet"),
    (re.compile(r"ios.*tablet|tablet.*ios|ipad", re.I),    "App — iOS Tablet"),
    (re.compile(r"android", re.I),            "App — Android"),
    (re.compile(r"ios|iphone", re.I),         "App — iOS"),
]


def classify_property(unit_code: str, unit_display: str = "") -> str:
    hay = f"{unit_code} {unit_display}"
    for rx, label in PROPERTY_RULES:
        if rx.search(hay):
            return label
    return "Web"


def strip_brand_prefix(displays: pd.Series) -> dict:
    """Map level-1 display name → section-style display with the site brand
    removed, e.g. 'GB News - Celebrity' → 'Celebrity'.

    The brand is detected, not hardcoded: the most common leading token
    (before ' - ') across the site's level-1 units. Only stripped when it
    covers ≥ half the units, so publishers that don't prefix (WeatherBug)
    pass through untouched.
    """
    displays = displays.dropna().astype(str)
    if displays.empty:
        return {}
    leading = displays.str.split(" - ").str[0].str.strip()
    top = leading.value_counts()
    brand, cover = top.index[0], top.iloc[0]
    out = {}
    for d in displays.unique():
        if cover >= len(displays.unique()) / 2 and d.startswith(brand + " - "):
            out[d] = d[len(brand) + 3:].strip()
        else:
            out[d] = d
    return out


# ── PARSED REPORTS ────────────────────────────────────────────────────────────

@dataclass
class Report:
    kind: str                      # 'gam' | 'rill'
    df: pd.DataFrame               # normalized rows
    metrics: list                  # canonical metric names present
    dims: set                      # subset of {'date', 'source_group'}
    warnings: list = field(default_factory=list)


def parse_gam(data: bytes) -> Report:
    raw = read_csv_any(data)
    level_cols = gam_level_columns(raw.columns)
    if not level_cols and "Ad unit code" not in raw.columns:
        raise ValueError(
            "Not a recognizable GAM export: no 'Ad unit code level N' or "
            f"'Ad unit code' columns. Found: {list(raw.columns)}"
        )

    metrics = [m for m, (g, _, _, _) in METRICS.items() if g in raw.columns]
    if not metrics:
        raise ValueError(
            "GAM file has no known metric column "
            f"(expected one of: {[v[0] for v in METRICS.values()]})."
        )

    df = pd.DataFrame()
    df["key_path"] = raw.apply(lambda r: gam_key_path(r, level_cols), axis=1)
    df["unit_code"] = df["key_path"].str.split("/").str[0].fillna("")

    # Display name of the top-level unit, from 'Ad unit (all levels)'.
    if "Ad unit (all levels)" in raw.columns:
        display_l1 = (
            raw["Ad unit (all levels)"].astype(str)
            .str.split("»").str[0].str.strip()
        )
    else:
        display_l1 = df["unit_code"]
    df["unit_display"] = display_l1

    for m in metrics:
        df[m] = _to_num(raw[METRICS[m][0]])

    dims = set()
    warnings = []
    if GAM_DATE in raw.columns:
        parsed = pd.to_datetime(raw[GAM_DATE], errors="coerce")
        if parsed.notna().any():
            df["date"] = parsed.dt.date
            dims.add("date")
        else:
            warnings.append("GAM 'Date' column present but unparseable — ignored.")

    if GAM_TYPE_COL in raw.columns:
        df["line_item_type"] = raw[GAM_TYPE_COL]
        df["order"] = raw.get(GAM_ORDER_COL)
        # Blank type+order rows are Open Bidding; Amazon rides on Price priority.
        blank = df["line_item_type"].isna() & df["order"].isna()
        df.loc[blank, ["line_item_type", "order"]] = "OB"
        is_amazon = (
            df["order"].str.contains(r"Amazon|APS|TAM", case=False, na=False)
            & (df["line_item_type"] == "Price priority")
        )
        df.loc[is_amazon, "line_item_type"] = "AMAZON"
        df["source_group"] = df["line_item_type"].map(GAM_GROUP)
        dims.add("source_group")

    return Report("gam", df, metrics, dims, warnings)


def parse_rill(data: bytes) -> Report:
    raw = read_csv_any(data)
    if "Ad Unit" not in raw.columns:
        raise ValueError(
            f"Not a recognizable Rill export: no 'Ad Unit' column. Found: {list(raw.columns)}"
        )

    metrics = [m for m, (_, r, _, _) in METRICS.items() if r in raw.columns]
    if not metrics:
        raise ValueError(
            "Rill file has no known metric column "
            f"(expected one of: {[v[1] for v in METRICS.values()]})."
        )

    df = pd.DataFrame()
    df["key_path"] = raw["Ad Unit"].map(rill_key_path)
    df["is_others"] = df["key_path"] == ""
    df["unit_code"] = df["key_path"].str.split("/").str[0].fillna("")

    for m in metrics:
        df[m] = _to_num(raw[METRICS[m][1]])

    dims = set()
    warnings = []
    if RILL_DATE in raw.columns:
        parsed = pd.to_datetime(raw[RILL_DATE], errors="coerce")
        if parsed.notna().any():
            df["date"] = parsed.dt.date
            dims.add("date")

    if RILL_TYPE_COL in raw.columns:
        df["source_group"] = raw[RILL_TYPE_COL].map(RILL_GROUP)
        dims.add("source_group")

    if "Domain" in raw.columns:
        df["domain"] = raw["Domain"]

    return Report("rill", df, metrics, dims, warnings)


# ── FILE-KIND DETECTION & AUTO-PAIRING ────────────────────────────────────────

def parse_any(data: bytes) -> Report:
    """Identify a file as GAM or Rill from its columns and parse it.

    GAM exports carry 'Ad unit (all levels)' / 'Ad unit code level N';
    Rill exports carry a single 'Ad Unit' path column. The two never overlap.
    """
    df = read_csv_any(data)
    if gam_level_columns(df.columns) or "Ad unit (all levels)" in df.columns:
        return parse_gam(data)
    if "Ad Unit" in df.columns:
        return parse_rill(data)
    raise ValueError(
        "Unrecognizable file: expected GAM columns ('Ad unit (all levels)' / "
        "'Ad unit code level N') or a Rill 'Ad Unit' column. "
        f"Found: {list(df.columns)}"
    )


def derive_site_name(gam: Report, fallback: str) -> str:
    """Site name from the GAM data itself.

    1. A domain-like top-level unit (contains '.') names the site —
       WeatherBug's tree has 'weatherbug.com'.
    2. Else the dominant brand prefix of the unit display names —
       'GB News - Celebrity', 'GB News - Money', … → 'GB News'.
    3. Else the uploaded file name.
    """
    units = gam.df[["unit_code", "unit_display"]].query("unit_code != ''")
    if units.empty:
        return fallback
    domainish = units.loc[units["unit_code"].str.contains(r"\.", regex=True), "unit_code"]
    if not domainish.empty:
        return domainish.value_counts().index[0]
    displays = units.drop_duplicates("unit_code")["unit_display"].astype(str)
    leading = displays.str.split(" - ").str[0].str.strip()
    top = leading.value_counts()
    if not top.empty and top.iloc[0] >= len(displays) / 2:
        return top.index[0]
    return fallback


def auto_pair(gam_files: list, rill_files: list) -> tuple:
    """Pair GAM and Rill files by ad-unit path overlap (greedy, best first).

    Args are lists of (filename, Report). Returns (pairs, notes) where notes
    flag anything that could not be paired confidently.
    """
    gam_paths = {
        i: set(rep.df["key_path"]) - {""}
        for i, (_, rep) in enumerate(gam_files)
    }
    rill_paths = {
        j: set(rep.df.loc[~rep.df["is_others"], "key_path"]) - {""}
        for j, (_, rep) in enumerate(rill_files)
    }

    candidates = sorted(
        (
            (len(gam_paths[i] & rill_paths[j]), i, j)
            for i in gam_paths for j in rill_paths
        ),
        key=lambda t: -t[0],
    )
    used_g, used_r, pairs, notes, seen_names = set(), set(), [], [], set()
    for overlap, i, j in candidates:
        if overlap == 0 or i in used_g or j in used_r:
            continue
        used_g.add(i)
        used_r.add(j)
        gam_fn, gam_rep = gam_files[i]
        rill_fn, rill_rep = rill_files[j]
        name = derive_site_name(gam_rep, fallback=gam_fn.rsplit(".", 1)[0])
        n, base = 2, name
        while name in seen_names:
            name = f"{base} ({n})"
            n += 1
        seen_names.add(name)
        pairs.append((SitePair(name, gam_rep, rill_rep), gam_fn, rill_fn,
                      overlap, len(rill_paths[j])))

    for i, (fn, _) in enumerate(gam_files):
        if i not in used_g:
            notes.append(f"GAM file **{fn}** matches no uploaded Rill file — not included.")
    for j, (fn, _) in enumerate(rill_files):
        if j not in used_r:
            notes.append(f"Rill file **{fn}** matches no uploaded GAM file — not included.")
    return pairs, notes


# ── SITE PAIR ─────────────────────────────────────────────────────────────────

@dataclass
class SitePair:
    name: str
    gam: Report
    rill: Report

    @property
    def metrics(self) -> list:
        return [m for m in METRICS if m in self.gam.metrics and m in self.rill.metrics]

    @property
    def dims(self) -> set:
        return self.gam.dims & self.rill.dims


def build_unit_meta(pair: SitePair) -> pd.DataFrame:
    """Per top-level unit: property class + brand-stripped display name.
    Derived from GAM (it lists the whole tree); Rill rows reuse it by code."""
    units = (
        pair.gam.df[["unit_code", "unit_display"]]
        .query("unit_code != ''")
        .drop_duplicates("unit_code")
        .copy()
    )
    disp_map = strip_brand_prefix(units["unit_display"])
    units["unit_display"] = units["unit_display"].map(disp_map)
    units["property"] = [
        classify_property(c, d) for c, d in zip(units["unit_code"], units["unit_display"])
    ]
    return units.set_index("unit_code")


def _sectionize(key_path: str, unit_meta: pd.DataFrame) -> tuple:
    """(property, section, slot) for one code path.

    slot = leaf; property = classified top unit. The section is whatever
    sits between them — for flat trees (GB: gbnews_celebrity/mpu_1) the
    top unit itself is the section, shown with its brand-stripped display.
    """
    parts = key_path.split("/") if key_path else []
    if not parts or parts == [""]:
        return ("(unattributed)", "(unattributed)", OTHERS_LABEL)
    unit = parts[0]
    meta = unit_meta.loc[unit] if unit in unit_meta.index else None
    prop = meta["property"] if meta is not None else "Web"
    unit_disp = meta["unit_display"] if meta is not None else unit
    if len(parts) >= 3:
        section = "/".join(parts[1:-1])
    elif len(parts) == 2:
        section = unit_disp
    else:
        section = unit_disp
    slot = parts[-1]
    return (prop, section, slot)


def enrich(pair: SitePair) -> tuple:
    """Return (gam_df, rill_df) with unified site/property/section/slot columns."""
    unit_meta = build_unit_meta(pair)
    out = []
    for rep in (pair.gam, pair.rill):
        df = rep.df.copy()
        trio = df["key_path"].map(lambda p: _sectionize(p, unit_meta))
        df["property"] = trio.str[0]
        df["section"] = trio.str[1]
        df["slot"] = trio.str[2]
        df["site"] = pair.name
        out.append(df)
    return tuple(out)


# ── DISCREPANCY TABLES ────────────────────────────────────────────────────────

def disc_pct(gam_val: float, rill_val: float):
    """(GAM − Rill) / GAM. GAM==0 with Rill>0 has no % — flagged 'Rill only'
    by the formatter instead of being hidden as missing data."""
    if gam_val == 0:
        return 0.0 if rill_val == 0 else np.nan
    return (gam_val - rill_val) / gam_val * 100


def build_disc(gam_agg: pd.DataFrame, rill_agg: pd.DataFrame, keys: list,
               metrics: list, sort_by=None) -> pd.DataFrame:
    merged = gam_agg.merge(rill_agg, on=keys, how="outer")
    for m in metrics:
        for side in ("GAM", "Rill"):
            col = f"{side}_{m}"
            merged[col] = merged.get(col, pd.Series(0, index=merged.index)).fillna(0)
        merged[f"{m}_disc"] = merged.apply(
            lambda r: disc_pct(r[f"GAM_{m}"], r[f"Rill_{m}"]), axis=1
        )
    if sort_by:
        merged = merged.sort_values(sort_by)
    return merged.reset_index(drop=True)


def _agg(df: pd.DataFrame, keys: list, metrics: list, prefix: str) -> pd.DataFrame:
    return (
        df.groupby(keys)[metrics].sum()
        .rename(columns={m: f"{prefix}_{m}" for m in metrics})
        .reset_index()
    )


def build_tables(pairs: list) -> dict:
    """All report tables for a list of SitePairs.

    Matching philosophy: totals are compared per site honestly (full GAM vs
    full Rill including 'Others'); drill-down levels compare matched paths
    only, with the residue quantified in dedicated unmatched tables rather
    than silently inflating per-row discrepancies.
    """
    frames_g, frames_r = [], []
    metrics_by_site, dims_common = {}, None
    for pair in pairs:
        g, r = enrich(pair)
        frames_g.append(g)
        frames_r.append(r)
        metrics_by_site[pair.name] = pair.metrics
        dims_common = pair.dims if dims_common is None else dims_common & pair.dims

    gam_all = pd.concat(frames_g, ignore_index=True)
    rill_all = pd.concat(frames_r, ignore_index=True)
    metrics = sorted(
        {m for ms in metrics_by_site.values() for m in ms},
        key=list(METRICS).index,
    )
    dims_common = dims_common or set()

    gam_paths = gam_all.groupby("site")["key_path"].agg(set).to_dict()
    rill_all["matched"] = [
        (kp != "") and (kp in gam_paths.get(site, set()))
        for site, kp in zip(rill_all["site"], rill_all["key_path"])
    ]
    rill_matched_paths = (
        rill_all[rill_all["matched"]].groupby("site")["key_path"].agg(set).to_dict()
    )
    gam_all["matched"] = [
        kp in rill_matched_paths.get(site, set())
        for site, kp in zip(gam_all["site"], gam_all["key_path"])
    ]
    gam_m, rill_m = gam_all[gam_all["matched"]], rill_all[rill_all["matched"]]

    tables = {}

    # Site overview: full totals AND matched-only compare, side by side.
    g_tot = _agg(gam_all, ["site"], metrics, "GAM")
    r_tot = _agg(rill_all, ["site"], metrics, "Rill")
    overview = build_disc(g_tot, r_tot, ["site"], metrics, sort_by=["site"])
    gm_tot = _agg(gam_m, ["site"], metrics, "GAM") if not gam_m.empty else g_tot.iloc[0:0]
    rm_tot = _agg(rill_m, ["site"], metrics, "Rill") if not rill_m.empty else r_tot.iloc[0:0]
    matched_ov = build_disc(gm_tot, rm_tot, ["site"], metrics, sort_by=["site"])
    tables["overview"] = overview
    tables["overview_matched"] = matched_ov

    if "date" in dims_common:
        tables["by_date"] = build_disc(
            _agg(gam_all, ["site", "date"], metrics, "GAM"),
            _agg(rill_all, ["site", "date"], metrics, "Rill"),
            ["site", "date"], metrics, sort_by=["site", "date"],
        )

    tables["by_property"] = build_disc(
        _agg(gam_m, ["site", "property"], metrics, "GAM"),
        _agg(rill_m, ["site", "property"], metrics, "Rill"),
        ["site", "property"], metrics, sort_by=["site", "property"],
    )
    tables["by_section"] = build_disc(
        _agg(gam_m, ["site", "property", "section"], metrics, "GAM"),
        _agg(rill_m, ["site", "property", "section"], metrics, "Rill"),
        ["site", "property", "section"], metrics,
        sort_by=["site", "property", "section"],
    )
    tables["by_adunit"] = build_disc(
        _agg(gam_m, ["site", "property", "section", "slot", "key_path"], metrics, "GAM"),
        _agg(rill_m, ["site", "property", "section", "slot", "key_path"], metrics, "Rill"),
        ["site", "property", "section", "slot", "key_path"], metrics,
    ).sort_values(f"GAM_{metrics[0]}", ascending=False).reset_index(drop=True)

    if "source_group" in dims_common:
        gam_sg = gam_all[gam_all["source_group"].notna()]
        rill_sg = rill_all[rill_all["source_group"].notna()]
        tables["by_source_group"] = build_disc(
            _agg(gam_sg, ["site", "source_group"], metrics, "GAM"),
            _agg(rill_sg, ["site", "source_group"], metrics, "Rill"),
            ["site", "source_group"], metrics, sort_by=["site", "source_group"],
        )

    # Residue — quantified, never silently dropped.
    rill_um = rill_all[~rill_all["matched"]].copy()
    rill_um["bucket"] = np.where(rill_um["is_others"], "Rill 'Others' (unattributable)",
                                 "Rill path not in GAM")
    tables["rill_unmatched"] = _agg(rill_um, ["site", "bucket"], metrics, "Rill") \
        if not rill_um.empty else pd.DataFrame()
    gam_um = gam_all[~gam_all["matched"]]
    tables["gam_unmatched"] = _agg(gam_um, ["site", "property"], metrics, "GAM") \
        if not gam_um.empty else pd.DataFrame()

    return tables


# ── SUMMARY / CLASSIFICATION ──────────────────────────────────────────────────

def coverage_summary(tables: dict, metrics: list) -> pd.DataFrame:
    """One crisp row per site: matched-compare disc per metric + how much of
    each side's volume the matched compare covers."""
    m0 = metrics[0]
    ov, mv = tables["overview"], tables["overview_matched"]
    rows = []
    for _, r in ov.iterrows():
        site = r["site"]
        mrow = mv[mv["site"] == site]
        mrow = mrow.iloc[0] if not mrow.empty else None
        row = {"site": site}
        for m in metrics:
            row[f"{m}_disc"] = mrow[f"{m}_disc"] if mrow is not None else np.nan
        gam_tot, rill_tot = r[f"GAM_{m0}"], r[f"Rill_{m0}"]
        row["gam_coverage"] = (mrow[f"GAM_{m0}"] / gam_tot * 100) if mrow is not None and gam_tot else 0.0
        row["rill_coverage"] = (mrow[f"Rill_{m0}"] / rill_tot * 100) if mrow is not None and rill_tot else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def classification_summary(pairs: list) -> pd.DataFrame:
    """Site → property breakdown: units, sections, and whether Rill covers it."""
    rows = []
    for pair in pairs:
        g, r = enrich(pair)
        rill_props = set(r.loc[~r["is_others"], "property"].unique())
        for prop, grp in g.groupby("property"):
            rows.append({
                "site": pair.name,
                "property": prop,
                "top_level_units": grp["unit_code"].nunique(),
                "sections": grp["section"].nunique(),
                "slots": grp["slot"].nunique(),
                "in_rill": prop in rill_props,
            })
    return pd.DataFrame(rows).sort_values(["site", "property"]).reset_index(drop=True)


# ── DISPLAY FORMATTING (pure — no Streamlit) ──────────────────────────────────

DIM_LABELS = {
    "site": "Site", "date": "Date", "property": "Property", "section": "Section",
    "slot": "Slot", "source_group": "Source Group", "key_path": "Ad Unit Path",
    "bucket": "Bucket",
}


def fmt_disc(v, rill_val):
    """Discrepancy formatting. NaN means GAM=0 with Rill>0 — that is a
    finding ('Rill only'), never displayed as missing data."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "🔴 Rill only" if rill_val and rill_val > 0 else "—"
    icon = "🔴" if abs(v) >= ALERT_PCT else ("⚠️" if abs(v) >= WARN_PCT else "✅")
    return f"{icon} {v:+.2f}%"


def table_metrics(df: pd.DataFrame) -> list:
    return [m for m in METRICS if f"GAM_{m}" in df.columns]


def fmt_metric_value(x: float, kind: str) -> str:
    return f"${x:,.2f}" if kind == "money" else f"{int(round(x)):,}"


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the human-readable display frame for any discrepancy table."""
    metrics = table_metrics(df)
    disp = df.copy()
    if "date" in disp.columns:
        disp["date"] = disp["date"].astype(str)
    rename = {}
    for m in metrics:
        label, kind = METRICS[m][2], METRICS[m][3]
        for side in ("GAM", "Rill"):
            col = f"{side}_{m}"
            vals = pd.to_numeric(disp[col], errors="coerce").fillna(0)
            disp[col] = vals.apply(lambda x: fmt_metric_value(x, kind))
            rename[col] = f"{side} {label}"
        raw_d = pd.to_numeric(df[f"{m}_disc"], errors="coerce")
        raw_r = pd.to_numeric(df[f"Rill_{m}"], errors="coerce")
        disp[f"{m}_disc"] = [fmt_disc(d, r) for d, r in zip(raw_d, raw_r)]
        rename[f"{m}_disc"] = f"{label} Δ%"
    rename.update(DIM_LABELS)
    return disp.rename(columns=rename)


# ── SHARE-LINK ENCODING (pure — no Streamlit) ─────────────────────────────────

def encode_tables(tables: dict, meta: dict) -> str:
    payload = {"meta": meta, "tables": {}}
    for name, df in tables.items():
        d = df.copy()
        if "date" in d.columns:
            d["date"] = d["date"].astype(str)
        for col in d.select_dtypes(include=[float]).columns:
            d[col] = d[col].round(4)
        payload["tables"][name] = {
            "columns": list(d.columns),
            "data": d.where(pd.notna(d), None).values.tolist(),
        }
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
