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

The workflow is two files: one GAM export + one Rill export, each possibly
covering many sites. Site identification is automatic:

    1. Rill's Domain column is authoritative — each top-level ad unit is
       assigned the domain Rill reports it under.
    2. GAM-only units fall back to evidence in the GAM file itself:
       domain-like unit codes, brand prefixes in display names
       ('GB News - Celebrity' → GB News), and shared code-token clusters.

Within a site the hierarchy is: property (classified top-level unit) →
section → slot.
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
UNATTRIBUTED = "(unattributed)"


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
    (before ' - ') across the level-1 units. Only stripped when it covers
    ≥ half the units, so publishers that don't prefix (WeatherBug) pass
    through untouched.
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


def parse_any(data: bytes) -> Report:
    """Identify a file as GAM or Rill from its columns and parse it.

    GAM exports carry 'Ad unit (all levels)' / 'Ad unit code level N';
    Rill exports carry a single 'Ad Unit' path column. The two never overlap,
    so the user can drop the files in either slot.
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


def common_metrics(gam: Report, rill: Report) -> list:
    return [m for m in METRICS if m in gam.metrics and m in rill.metrics]


def common_dims(gam: Report, rill: Report) -> set:
    return gam.dims & rill.dims


# ── SITE IDENTIFICATION ───────────────────────────────────────────────────────

def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_domain(d) -> str:
    s = str(d).strip().lower()
    return s[4:] if s.startswith("www.") else s


def _site_root(site: str) -> str:
    """'gbnews.com' → 'gbnews'; non-domain names just normalize."""
    return _norm(site.split(".")[0]) if "." in site else _norm(site)


def build_site_map(gam: Report, rill: Report) -> dict:
    """Top-level ad-unit code → site name, learned from the data.

    Priority of evidence:
      1. Rill's Domain column — the authoritative unit→site mapping for
         every unit Rill reports.
      2. Domain-like GAM unit codes (weatherbug.com) are sites themselves.
      3. Brand prefix in the unit display name ('GB News - Celebrity' →
         'GB News'), linked to an already-known site when the names agree
         (brand 'GB News' ↔ domain gbnews.com).
      4. Leftovers attach to a known brand/site their code starts with
         (GBNews_Video → GB News); otherwise units sharing a leading code
         token cluster together (Primis_Video_* → 'Primis').
    """
    site_map: dict = {}

    # 1. Rill Domain (majority vote per unit, in case of stray rows)
    if "domain" in rill.df.columns:
        m = rill.df[
            ~rill.df["is_others"]
            & rill.df["domain"].notna()
            & (rill.df["unit_code"] != "")
        ]
        if not m.empty:
            mode = m.groupby("unit_code")["domain"].agg(
                lambda s: s.value_counts().index[0]
            )
            site_map.update({u: normalize_domain(d) for u, d in mode.items()})

    units = (
        gam.df[["unit_code", "unit_display"]]
        .query("unit_code != ''")
        .drop_duplicates("unit_code")
    )
    todo = [t for t in units.itertuples(index=False) if t.unit_code not in site_map]

    # 2. Domain-like codes
    rest = []
    for t in todo:
        if "." in t.unit_code:
            site_map[t.unit_code] = normalize_domain(t.unit_code)
        else:
            rest.append(t)

    # 3. Brand prefixes, merged with known sites when names agree
    known = set(site_map.values())
    brands: dict = {}
    rest2 = []
    for t in rest:
        disp = str(t.unit_display)
        if " - " in disp:
            brand = disp.split(" - ")[0].strip()
            nb = _norm(brand)
            site = next(
                (s for s in known if _site_root(s) == nb
                 or (nb and nb in _site_root(s))
                 or (_site_root(s) and _site_root(s) in nb)),
                brand,
            )
            site_map[t.unit_code] = site
            brands[nb] = site
        else:
            rest2.append(t)

    # 4. Leftovers: code-prefix containment, else leading-token clusters.
    #    Group by the lowercased code token, but label the cluster with the
    #    display name's casing ('WB', not 'wb').
    known = set(site_map.values())
    token_counts: dict = {}
    token_label: dict = {}
    for t in rest2:
        tl = re.split(r"[_\-.]", str(t.unit_code))[0].lower()
        token_counts[tl] = token_counts.get(tl, 0) + 1
        token_label.setdefault(tl, re.split(r"[_\-.]", str(t.unit_display))[0])
    for t in rest2:
        n = _norm(t.unit_code)
        hit = next((site for b, site in brands.items() if b and n.startswith(b)), None)
        if hit is None:
            hit = next((s for s in known if _site_root(s) and n.startswith(_site_root(s))), None)
        if hit is None:
            tl = re.split(r"[_\-.]", str(t.unit_code))[0].lower()
            # A leading token only names a cluster if ≥2 units share it;
            # singletons keep their readable display name.
            hit = token_label[tl] if token_counts.get(tl, 0) >= 2 else str(t.unit_display)
        site_map[t.unit_code] = hit
    return site_map


# ── UNIT META / SECTION EXTRACTION ────────────────────────────────────────────

def build_unit_meta(gam: Report) -> pd.DataFrame:
    """Per top-level unit: property class + brand-stripped display name.
    Derived from GAM (it lists the whole tree); Rill rows reuse it by code."""
    units = (
        gam.df[["unit_code", "unit_display"]]
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
        return (UNATTRIBUTED, UNATTRIBUTED, OTHERS_LABEL)
    unit = parts[0]
    meta = unit_meta.loc[unit] if unit in unit_meta.index else None
    prop = meta["property"] if meta is not None else "Web"
    unit_disp = meta["unit_display"] if meta is not None else unit
    if len(parts) >= 3:
        section = "/".join(parts[1:-1])
    else:
        section = unit_disp
    slot = parts[-1]
    return (prop, section, slot)


def enrich(gam: Report, rill: Report) -> tuple:
    """Return (gam_df, rill_df) with unified site/property/section/slot columns.

    Site assignment: GAM rows through the learned site map; Rill rows through
    the same map, falling back to their own Domain (this is how 'Others' rows
    still land on the right site when Domain is present)."""
    site_map = build_site_map(gam, rill)
    unit_meta = build_unit_meta(gam)
    out = []
    for rep in (gam, rill):
        df = rep.df.copy()
        trio = df["key_path"].map(lambda p: _sectionize(p, unit_meta))
        df["property"] = trio.str[0]
        df["section"] = trio.str[1]
        df["slot"] = trio.str[2]
        domains = df["domain"] if "domain" in df.columns else pd.Series(np.nan, index=df.index)
        df["site"] = [
            site_map.get(u) if site_map.get(u)
            else (normalize_domain(d) if pd.notna(d) else UNATTRIBUTED)
            for u, d in zip(df["unit_code"], domains)
        ]
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


def build_tables(gam: Report, rill: Report) -> dict:
    """All report tables for one GAM + one Rill export (any number of sites).

    Matching philosophy: totals are compared per site honestly (full GAM vs
    full Rill including 'Others'); drill-down levels compare matched paths
    only, with the residue quantified in dedicated unmatched tables rather
    than silently inflating per-row discrepancies.
    """
    metrics = common_metrics(gam, rill)
    dims = common_dims(gam, rill)
    gam_all, rill_all = enrich(gam, rill)

    gam_paths = set(gam_all["key_path"]) - {""}
    rill_all["matched"] = rill_all["key_path"].isin(gam_paths) & (rill_all["key_path"] != "")
    rill_matched_paths = set(rill_all.loc[rill_all["matched"], "key_path"])
    gam_all["matched"] = gam_all["key_path"].isin(rill_matched_paths)
    gam_m, rill_m = gam_all[gam_all["matched"]], rill_all[rill_all["matched"]]

    tables = {}

    # Site overview: full totals AND matched-only compare, side by side.
    tables["overview"] = build_disc(
        _agg(gam_all, ["site"], metrics, "GAM"),
        _agg(rill_all, ["site"], metrics, "Rill"),
        ["site"], metrics, sort_by=["site"],
    )
    tables["overview_matched"] = build_disc(
        _agg(gam_m, ["site"], metrics, "GAM") if not gam_m.empty else tables["overview"].iloc[0:0][["site"]],
        _agg(rill_m, ["site"], metrics, "Rill") if not rill_m.empty else tables["overview"].iloc[0:0][["site"]],
        ["site"], metrics, sort_by=["site"],
    ) if not (gam_m.empty and rill_m.empty) else pd.DataFrame(columns=tables["overview"].columns)

    if "date" in dims:
        tables["by_date"] = build_disc(
            _agg(gam_all, ["date"], metrics, "GAM"),
            _agg(rill_all, ["date"], metrics, "Rill"),
            ["date"], metrics, sort_by=["date"],
        )
        tables["by_date_site"] = build_disc(
            _agg(gam_all, ["date", "site"], metrics, "GAM"),
            _agg(rill_all, ["date", "site"], metrics, "Rill"),
            ["date", "site"], metrics, sort_by=["site", "date"],
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

    if "source_group" in dims:
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
        mrow = mv[mv["site"] == site] if not mv.empty else mv
        mrow = mrow.iloc[0] if len(mrow) else None
        row = {"site": site}
        for m in metrics:
            row[f"{m}_disc"] = mrow[f"{m}_disc"] if mrow is not None else np.nan
        gam_tot, rill_tot = r[f"GAM_{m0}"], r[f"Rill_{m0}"]
        row["gam_coverage"] = (mrow[f"GAM_{m0}"] / gam_tot * 100) if mrow is not None and gam_tot else 0.0
        row["rill_coverage"] = (mrow[f"Rill_{m0}"] / rill_tot * 100) if mrow is not None and rill_tot else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def classification_summary(gam: Report, rill: Report) -> pd.DataFrame:
    """Site → property breakdown: units, sections, and whether Rill covers it."""
    g, r = enrich(gam, rill)
    rill_cov = set(zip(r.loc[~r["is_others"], "site"], r.loc[~r["is_others"], "property"]))
    rows = []
    for (site, prop), grp in g.groupby(["site", "property"]):
        rows.append({
            "site": site,
            "property": prop,
            "top_level_units": grp["unit_code"].nunique(),
            "sections": grp["section"].nunique(),
            "slots": grp["slot"].nunique(),
            "in_rill": (site, prop) in rill_cov,
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
