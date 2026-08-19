from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

from homeharvest import scrape_property


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="Land Acquisition Analyzer",
    page_icon="🌳",
    layout="wide",
)


# Default demo locations. The UI can override these.
DEFAULT_LOCATION_1 = [
    "Eagleville, TN",
    "Rockvale, TN",
    "Chapel Hill, TN",
    "Thompsons Station, TN",
    "Bethesda, TN",
    "Spring Hill, TN",
    "Unionville, TN",
    "Versailles, TN",
    "Midland, TN",
    "Wilhoite Mills, TN",
    "Holts Corner, TN",
    "Lasea, TN",
    "Santa Fe, TN",
    "Boston, TN",
]

DEFAULT_LOCATION_2 = [
    "Dickson, TN",
    "Bon Aqua, TN",
    "Primm Springs, TN",
    "Centerville, TN",
    "Fairview, TN",
    "Boston, TN",
    "Lyles, TN",
    "Burns, TN",
    "Nunnelly, TN",
]

SOLD_LOOKBACK_DAYS = 180

MODEL_OPTIONS = [
    "Access",
    "Baseline",
    "Site + Access",
    "Land Character",
]


# ============================================================
# HELPERS
# ============================================================


def parse_locations(raw: str) -> list[str]:
    """Parse one location per line or semicolon; commas are preserved."""
    if not raw or not raw.strip():
        return []
    parts = re.split(r"[;\n]+", raw)
    values = []
    seen = set()
    for item in parts:
        value = re.sub(r"\s+", " ", item.strip())
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            values.append(value)
            seen.add(key)
    return values


def normalize_location(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def make_acreage_levels(min_acres: float, max_acres: float) -> list[int]:
    """
    5-acre increments through 30 acres, then 10-acre increments.
    Always include the requested maximum as the final point when needed.
    """
    if min_acres <= 0 or max_acres <= 0:
        raise ValueError("Acreage values must be positive.")
    if min_acres >= max_acres:
        raise ValueError("Maximum acreage must be greater than minimum acreage.")

    start = int(np.ceil(min_acres / 5.0) * 5)
    stop = int(np.floor(max_acres))
    levels = []

    current = start
    while current <= min(stop, 30):
        levels.append(current)
        current += 5

    if max(stop, 0) > 30:
        current = 40
        while current <= stop:
            if current not in levels:
                levels.append(current)
            current += 10

    max_int = int(round(max_acres))
    if max_int not in levels and max_int > 0:
        levels.append(max_int)

    levels = sorted(set(levels))
    if not levels:
        raise ValueError("Unable to create acreage levels from the selected range.")
    return levels


def safe_num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a stable schema from HomeHarvest output."""
    out = df.copy()

    defaults = {
        "property_id": np.nan,
        "property_url": "",
        "status": "",
        "text": "",
        "city": "",
        "county": "",
        "state": "",
        "zip_code": "",
        "street": "",
        "lot_sqft": np.nan,
        "lot_acres": np.nan,
        "list_price": np.nan,
        "sold_price": np.nan,
        "last_sold_date": pd.NaT,
        "list_date": pd.NaT,
        "days_on_mls": np.nan,
        "beds": np.nan,
        "full_baths": np.nan,
        "half_baths": np.nan,
        "sqft": np.nan,
        "year_built": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    # HomeHarvest normally provides lot_sqft rather than lot_acres.
    out["lot_sqft"] = safe_num(out, "lot_sqft")
    out["lot_acres"] = safe_num(out, "lot_acres")
    out.loc[
        out["lot_acres"].isna() & out["lot_sqft"].notna(),
        "lot_acres",
    ] = out.loc[
        out["lot_acres"].isna() & out["lot_sqft"].notna(),
        "lot_sqft",
    ] / 43560.0

    out["list_price"] = safe_num(out, "list_price")
    out["sold_price"] = safe_num(out, "sold_price")
    out["days_on_mls"] = safe_num(out, "days_on_mls")

    # Current/analysis price depends on status.
    status_upper = out["status"].astype(str).str.upper()
    out["analysis_price_model"] = np.where(
        status_upper.eq("SOLD"),
        out["sold_price"],
        out["list_price"],
    )
    out["current_price"] = out["list_price"]

    out["price_per_acre"] = (
        out["current_price"] / out["lot_acres"].replace(0, np.nan)
    )

    # Normalize dates.
    out["list_date"] = pd.to_datetime(out["list_date"], errors="coerce")
    out["last_sold_date"] = pd.to_datetime(out["last_sold_date"], errors="coerce")

    return out


def has_pattern(text: str, pattern: str) -> int:
    return int(bool(re.search(pattern, text, flags=re.I)))


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate the key text-derived variables used in the notebook."""
    out = df.copy()
    out["text_clean"] = (
        out["text"].fillna("").astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    patterns = {
        "perc_available_tf": (
            r"\b(?:approved|passed|existing|installed|functional).{0,80}"
            r"(?:perc|perk|septic).{0,80}\b|"
            r"\b(?:perc|perk|septic).{0,80}"
            r"(?:approved|passed|existing|installed|functional)\b"
        ),
        "perc_preliminary_tf": (
            r"\bpreliminary.{0,80}(?:perc|perk|soil|septic).{0,80}\b|"
            r"\b(?:perc|perk|soil|septic).{0,80}preliminary\b|"
            r"\bsoil\s+(?:work|site|study|evaluation).{0,100}"
            r"(?:supports|identified|suitable|homesite|home)\b|"
            r"\b(?:soil|perk|perc).{0,60}site.{0,60}"
            r"(?:identified|supports|preliminary)\b"
        ),
        "perc_negative_tf": (
            r"\b(?:failed|fail|no|not|does not|doesn't|won't|will not).{0,40}"
            r"(?:perc|perk|septic)\b|"
            r"\b(?:perc|perk|septic).{0,40}"
            r"(?:failed|fail|not suitable|unsuitable)\b"
        ),
        "electric_tf": (
            r"\b(?:electric|electricity|power).{0,60}"
            r"(?:at the road|at road|available|nearby|on site|onsite|to property|to lot|at property|at lot|along road)\b"
        ),
        "public_water_tf": r"\b(?:public water|city water|municipal water|water at the road|water available)\b",
        "well_tf": r"\b(?:private well|water well|well on site|well installed|well needed|well required)\b",
        "sewer_tf": r"\b(?:public sewer|city sewer|municipal sewer|sewer available)\b",
        "road_frontage_tf": r"\b(?:road frontage|frontage on|frontage along|feet of frontage|road front|fronts on|fronting)\b",
        "easement_tf": r"\b(?:easement|utility easement|access easement|shared driveway)\b",
        "survey_tf": r"\b(?:survey|surveyed|new survey|recent survey)\b",
        "creek_tf": r"\b(?:creek|creekfront|creek frontage|creek runs)\b",
        "pond_tf": r"\b(?:pond|ponds|fishing pond)\b",
        "river_tf": r"\b(?:river|riverfront|river frontage|river borders)\b",
        "spring_tf": r"\b(?:spring|spring fed|spring-fed)\b",
        "wooded_tf": r"\b(?:wooded|woods|mature hardwoods|hardwood forest|timber)\b",
        "pasture_tf": r"\b(?:pasture|pastureland|open pasture|hay field|hayfield)\b",
        "fenced_tf": r"\b(?:fenced|fencing|fence)\b",
        "hunting_tf": r"\b(?:hunting|hunt|wildlife|deer|turkey)\b",
        "horse_cattle_tf": r"\b(?:horses|horse|cattle|cow pasture|livestock|mini farm)\b",
        "residence_tf": r"\b(?:home|house|residence|cabin|manufactured home|mobile home|ranch house)\b",
        "manufactured_home_tf": r"\b(?:manufactured home|mobile home|mobile house)\b",
        "barn_tf": r"\b(?:barn|barns|pole barn|stable)\b",
        "other_structure_tf": r"\b(?:shop|garage|shed|workshop|outbuilding|storm shelter)\b",
        "development_tf": r"\b(?:development|developable|development potential|investment opportunity|subdivide|subdivision|multiple homes|multiple homesites|family compound|commercial potential|rezoning|rezoned)\b",
        "multiple_homesites_tf": r"\b(?:multiple homesites|multiple home sites|multiple lots|multiple build sites|two homesites|3 homesites|three homesites|multiple houses)\b",
        "greenbelt_tf": r"\b(?:greenbelt|green belt)\b",
        "view_tf": r"\b(?:views|view of|mountain views|scenic views|panoramic views)\b",
    }

    for name, pattern in patterns.items():
        out[name] = out["text_clean"].map(lambda x, p=pattern: has_pattern(x, p))

    out["site_status"] = np.select(
        [
            out["perc_negative_tf"].eq(1),
            out["perc_available_tf"].eq(1),
            out["perc_preliminary_tf"].eq(1),
        ],
        ["No Perc", "Available", "Preliminary"],
        default="Unknown",
    )

    out["log_price_model"] = np.log(
        out["analysis_price_model"].where(out["analysis_price_model"] > 0)
    )
    out["log_acres_model"] = np.log(
        out["lot_acres"].where(out["lot_acres"] > 0)
    )
    return out


def normalize_grouped_results(frames: list[pd.DataFrame], group_name: str, location_query: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = ensure_columns(result)
    result["_search_group"] = group_name
    result["_search_location"] = location_query
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def scrape_locations(
    location_1: tuple[str, ...],
    location_2: tuple[str, ...],
    min_acres: float,
    max_acres: float,
    sold_days: int,
) -> pd.DataFrame:
    """Pull current for-sale listings + sold listings in last N days from HomeHarvest/Realtor.com."""
    groups = [("Close", list(location_1)), ("Further", list(location_2))]
    all_frames = []

    # Use a modest acreage buffer for the model; the UI applies the final range afterward.
    scrape_min_acres = max(1.0, min_acres * 0.75)
    scrape_max_acres = max(max_acres * 1.25, max_acres + 10)
    lot_min_sqft = int(round(scrape_min_acres * 43560))
    lot_max_sqft = int(round(scrape_max_acres * 43560))

    for group_name, locations in groups:
        for location in locations:
            # Current inventory: all current for-sale listings in the location.
            active = scrape_property(
                location=location,
                listing_type="for_sale",
                lot_sqft_min=lot_min_sqft,
                lot_sqft_max=lot_max_sqft,
                exclude_pending=True,
                parallel=False,
                limit=10000,
                return_type="pandas",
            )
            if active is not None and len(active):
                active = active.copy()
                active["_search_group"] = group_name
                active["_search_location"] = location
                all_frames.append(active)

            # Sold: last 180 days only.
            sold = scrape_property(
                location=location,
                listing_type="sold",
                past_days=int(sold_days),
                lot_sqft_min=lot_min_sqft,
                lot_sqft_max=lot_max_sqft,
                parallel=False,
                limit=10000,
                return_type="pandas",
            )
            if sold is not None and len(sold):
                sold = sold.copy()
                sold["_search_group"] = group_name
                sold["_search_location"] = location
                all_frames.append(sold)

    if not all_frames:
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    raw = ensure_columns(raw)

    # Deduplicate across location queries/status requests.
    # Prefer property_id, then property_url when property_id is missing.
    raw["_dedupe_key"] = raw["property_id"].astype(str)
    missing_id = raw["_dedupe_key"].isin(["nan", "None", ""])
    raw.loc[missing_id, "_dedupe_key"] = raw.loc[missing_id, "property_url"].astype(str)
    raw = raw.drop_duplicates(subset="_dedupe_key", keep="first")

    # Preserve search group assigned by the first matching query.
    raw = build_text_features(raw)
    return raw.reset_index(drop=True)


def train_model(df: pd.DataFrame, model_name: str):
    active = df[
        df["status"].astype(str).str.upper().eq("FOR_SALE")
    ].copy()
    active = active.dropna(subset=["analysis_price_model", "lot_acres", "log_price_model", "log_acres_model"])
    active = active[(active["analysis_price_model"] > 0) & (active["lot_acres"] > 0)]

    counts = active["Distance"].value_counts()
    if counts.get("Close", 0) < 5 or counts.get("Further", 0) < 5:
        raise ValueError(
            "Not enough active listings in both groups. "
            f"Close={int(counts.get('Close', 0))}, Further={int(counts.get('Further', 0))}."
        )

    formulas = {
        "Baseline": "log_price_model ~ log_acres_model + is_close",
        "Access": (
            "log_price_model ~ log_acres_model + is_close + "
            "electric_tf + public_water_tf + well_tf + sewer_tf + "
            "road_frontage_tf + easement_tf + survey_tf"
        ),
        "Site + Access": (
            "log_price_model ~ log_acres_model + is_close + "
            "perc_available_tf + perc_preliminary_tf + perc_negative_tf + "
            "electric_tf + public_water_tf + well_tf + sewer_tf + "
            "road_frontage_tf + easement_tf + survey_tf"
        ),
        "Land Character": (
            "log_price_model ~ log_acres_model + is_close + "
            "wooded_tf + pasture_tf + creek_tf + pond_tf + river_tf + "
            "spring_tf + fenced_tf + hunting_tf + horse_cattle_tf + view_tf"
        ),
    }
    model = smf.ols(formulas[model_name], data=active).fit(cov_type="HC3")
    smearing = float(np.mean(np.exp(model.resid)))
    return model, active, smearing


def build_reference_profile(active: pd.DataFrame, acreage: float, model, window_pct: float = 0.25) -> tuple[dict, int]:
    lower = acreage * (1 - window_pct)
    upper = acreage * (1 + window_pct)
    comps = active[active["lot_acres"].between(lower, upper)].copy()
    if len(comps) < 5:
        comps = active[active["lot_acres"].between(acreage * 0.5, acreage * 1.5)].copy()
    if len(comps) < 5:
        comps = active.copy()

    profile = {"log_acres_model": np.log(acreage), "is_close": 0}
    for term in model.params.index:
        if term in {"Intercept", "log_acres_model", "is_close"} or ":" in term:
            continue
        if term in comps.columns and pd.api.types.is_numeric_dtype(comps[term]):
            value = comps[term].median()
            profile[term] = 0 if pd.isna(value) else float(value)
    return profile, len(comps)


def predict_curve(model, smearing: float, active: pd.DataFrame, acreage_levels: list[int], ci_alpha: float = 0.20) -> pd.DataFrame:
    rows = []
    for acres in acreage_levels:
        profile, n_comp = build_reference_profile(active, acres, model)
        for distance in ["Close", "Further"]:
            row = dict(profile)
            row["is_close"] = 1 if distance == "Close" else 0
            frame = model.get_prediction(pd.DataFrame([row])).summary_frame(alpha=ci_alpha)
            rows.append({
                "acres": acres,
                "Distance": distance,
                "estimate": float(np.exp(frame["mean"].iloc[0]) * smearing),
                "ci_low": float(np.exp(frame["mean_ci_lower"].iloc[0]) * smearing),
                "ci_high": float(np.exp(frame["mean_ci_upper"].iloc[0]) * smearing),
                "comparables": n_comp,
            })
    return pd.DataFrame(rows)


# ============================================================
# LOAD / SIDEBAR
# ============================================================

st.title("🌳 Land Acquisition Analyzer")
st.caption("HomeHarvest-powered current and recent-sold property analysis")

with st.sidebar:
    st.header("1. Locations")
    location_field = st.selectbox(
        "Location matching field",
        ["city", "county", "zip_code"],
        index=0,
        help="Used only for labeling/query validation. HomeHarvest searches the location strings directly."
    )

    location_1_raw = st.text_area(
        "Location 1 / Close",
        value="\n".join(DEFAULT_LOCATION_1),
        height=100,
        help="One HomeHarvest search location per line. Examples: Eagleville, TN; 37060; Williamson County, TN.",
    )

    location_2_raw = st.text_area(
        "Location 2 / Further",
        value="\n".join(DEFAULT_LOCATION_2),
        height=100,
        help="One HomeHarvest search location per line.",
    )

    location_1 = parse_locations(location_1_raw)
    location_2 = parse_locations(location_2_raw)

    st.header("2. Acquisition")
    budget = st.number_input(
        "Budget ($)",
        min_value=50_000,
        max_value=100_000_000,
        value=550_000,
        step=25_000,
        format="%.0f",
    )

    min_acres = st.number_input(
        "Minimum acreage",
        min_value=1.0,
        max_value=5000.0,
        value=15.0,
        step=1.0,
    )

    max_acres = st.number_input(
        "Maximum acreage",
        min_value=2.0,
        max_value=5000.0,
        value=75.0,
        step=5.0,
    )

    model_name = st.selectbox(
        "Valuation model",
        MODEL_OPTIONS,
        index=0,
    )

    ci_level = st.slider(
        "Model confidence level",
        min_value=0.50,
        max_value=0.95,
        value=0.80,
        step=0.05,
        format="%.0f%%",
    )

    sold_days = st.number_input(
        "Sold lookback (days)",
        min_value=30,
        max_value=365,
        value=SOLD_LOOKBACK_DAYS,
        step=30,
    )

    st.header("3. Data")
    refresh = st.button("🔄 Refresh HomeHarvest data", width="stretch")

    if refresh:
        scrape_locations.clear()
        st.rerun()

# ============================================================
# VALIDATION
# ============================================================

location_errors = []
if not location_1:
    location_errors.append("Location 1 is empty.")
if not location_2:
    location_errors.append("Location 2 is empty.")

location_1_norm = {normalize_location(x) for x in location_1}
location_2_norm = {normalize_location(x) for x in location_2}
overlap = sorted(location_1_norm & location_2_norm)
if overlap:
    # Location 1 / Close wins deterministically so the app can still run.
    location_2 = [x for x in location_2 if normalize_location(x) not in set(overlap)]
    location_2_norm = {normalize_location(x) for x in location_2}
    st.warning(
        "These locations were entered in both groups: "
        + ", ".join(overlap)
        + ". They will be assigned to Location 1 / Close to avoid ambiguous classification."
    )

if not location_2:
    location_errors.append("Location 2 is empty after removing overlapping locations.")

if max_acres <= min_acres:
    location_errors.append("Maximum acreage must be greater than minimum acreage.")

if location_errors:
    for err in location_errors:
        st.error(err)
    st.stop()

# ============================================================
# SCRAPE
# ============================================================

with st.spinner("HomeHarvest is pulling current listings and sold properties from Realtor.com..."):
    try:
        raw_df = scrape_locations(
            tuple(location_1),
            tuple(location_2),
            float(min_acres),
            float(max_acres),
            int(sold_days),
        )
    except Exception as exc:
        st.error(
            "HomeHarvest could not complete the scrape. "
            f"Error: {exc}"
        )
        st.info(
            "Try a simpler location string, reduce the acreage range, or use the Refresh button. "
            "HomeHarvest retrieves data directly from Realtor.com and can fail when the upstream site rejects a request."
        )
        st.stop()

if raw_df.empty:
    st.error("HomeHarvest returned no properties for the selected locations and acreage range.")
    st.stop()

# Label Distance directly from the search group returned by the query.
raw_df["Distance"] = raw_df["_search_group"]

# Restrict modeled/search data to the selected acreage window,
# while keeping the full pulled inventory available in the table.
selected_df = raw_df.copy()

# Ensure current/analysis price columns are sane.
selected_df["status_norm"] = selected_df["status"].astype(str).str.upper()
selected_df["days_on_mls_num"] = safe_num(selected_df, "days_on_mls").fillna(0)

# Active + sold in requested lookback; HomeHarvest already filtered sold dates.
active_df = selected_df[selected_df["status_norm"].eq("FOR_SALE")].copy()
sold_df = selected_df[selected_df["status_norm"].eq("SOLD")].copy()

# ============================================================
# DASHBOARD SUMMARY
# ============================================================

st.success(
    f"Pulled {len(selected_df):,} unique properties: "
    f"{len(active_df):,} current listings and {len(sold_df):,} sold in the last {int(sold_days)} days."
)

summary_cols = st.columns(6)
summary_cols[0].metric("Location 1 / Close", len(location_1))
summary_cols[1].metric("Location 2 / Further", len(location_2))
summary_cols[2].metric("Active", f"{len(active_df):,}")
summary_cols[3].metric("Sold", f"{len(sold_df):,}")
summary_cols[4].metric("Budget", f"${budget:,.0f}")
summary_cols[5].metric("Sold lookback", f"{int(sold_days)} days")

# ============================================================
# MODEL DATA
# ============================================================

model_df = active_df[
    active_df["lot_acres"].between(min_acres * 0.75, max_acres * 1.25)
    & active_df["analysis_price_model"].notna()
    & active_df["lot_acres"].notna()
    & active_df["analysis_price_model"].gt(0)
    & active_df["lot_acres"].gt(0)
    & (active_df["days_on_mls_num"] >= 0)
].copy()

counts = model_df["Distance"].value_counts()

if counts.get("Close", 0) < 5 or counts.get("Further", 0) < 5:
    st.warning(
        "There are not enough active observations to fit the selected model reliably. "
        f"Close={int(counts.get('Close', 0))}, Further={int(counts.get('Further', 0))}. "
        "The property table will still work. Add more locations or expand the acreage range."
    )
    model = None
    active_model = model_df
else:
    try:
        model, active_model, smearing = train_model(model_df, model_name)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

# ============================================================
# ACREAGE LEVELS
# ============================================================

try:
    acreage_levels = make_acreage_levels(min_acres, max_acres)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    "Acreage schedule: " + ", ".join(str(x) for x in acreage_levels)
)

# ============================================================
# DECISION TABLE + VISUALS
# ============================================================

if model is not None:
    with st.spinner("Calculating valuation curves and acquisition economics..."):
        curve_df = predict_curve(
            model,
            smearing,
            active_model,
            acreage_levels,
            ci_alpha=1 - ci_level,
        )

    wide = curve_df.pivot(
        index="acres",
        columns="Distance",
        values=["estimate", "ci_low", "ci_high"],
    )
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()

    wide["cost_of_proximity"] = wide["estimate_Close"] - wide["estimate_Further"]
    wide["close_multiple"] = wide["estimate_Close"] / wide["estimate_Further"]
    wide["close_premium_pct"] = wide["close_multiple"] - 1
    wide["additional_cash_close"] = (wide["estimate_Close"] - budget).clip(lower=0)

    # Current cumulative inventory: >= target acreage and <= budget.
    inventory_rows = []
    for acre in acreage_levels:
        eligible = active_df[
            active_df["lot_acres"].ge(acre)
            & active_df["current_price"].le(budget)
        ]
        inventory_rows.append({
            "acres": acre,
            "close_listings_under_budget": int((eligible["Distance"] == "Close").sum()),
            "further_listings_under_budget": int((eligible["Distance"] == "Further").sum()),
        })
    wide = wide.merge(pd.DataFrame(inventory_rows), on="acres", how="left")

    st.subheader("Acquisition decision table")

    display_df = wide.rename(columns={
        "acres": "Acres",
        "estimate_Further": "Further Estimated",
        "estimate_Close": "Close Estimated",
        "ci_low_Further": f"Further {int(ci_level*100)}% CI Low",
        "ci_high_Further": f"Further {int(ci_level*100)}% CI High",
        "ci_low_Close": f"Close {int(ci_level*100)}% CI Low",
        "ci_high_Close": f"Close {int(ci_level*100)}% CI High",
        "cost_of_proximity": "Cost of Proximity",
        "additional_cash_close": "Extra Cash for Close",
        "close_multiple": "Close / Further",
        "close_premium_pct": "Close Premium",
        "close_listings_under_budget": "Close Listings <= Budget",
        "further_listings_under_budget": "Further Listings <= Budget",
    }).copy()

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Acres": st.column_config.NumberColumn(format="%.0f"),
            "Further Estimated": st.column_config.NumberColumn(format="$%,.0f"),
            "Close Estimated": st.column_config.NumberColumn(format="$%,.0f"),
            "Further 80% CI Low": st.column_config.NumberColumn(format="$%,.0f"),
            "Further 80% CI High": st.column_config.NumberColumn(format="$%,.0f"),
            "Close 80% CI Low": st.column_config.NumberColumn(format="$%,.0f"),
            "Close 80% CI High": st.column_config.NumberColumn(format="$%,.0f"),
            "Cost of Proximity": st.column_config.NumberColumn(format="$%,.0f"),
            "Extra Cash for Close": st.column_config.NumberColumn(format="$%,.0f"),
            "Close / Further": st.column_config.NumberColumn(format="%.2fx"),
            "Close Premium": st.column_config.NumberColumn(format="%.1%"),
            "Close Listings <= Budget": st.column_config.NumberColumn(format="%d"),
            "Further Listings <= Budget": st.column_config.NumberColumn(format="%d"),
        },
    )

    st.caption(
        "Inventory counts are actual current listings meeting the cumulative acreage and budget threshold. "
        "Model intervals describe uncertainty around the estimated mean market curve, not individual-property outcomes."
    )

    # ========================================================
    # VISUALS
    # ========================================================
    st.subheader("Valuation visuals")
    tab1, tab2, tab3 = st.tabs([
        "Price / Acre Boxplot",
        "Price vs Acreage",
        "Acquisition Curve",
    ])

    with tab1:
        box_df = active_df[
            active_df["price_per_acre"].gt(0)
        ].copy()
        if len(box_df):
            fig = px.box(
                box_df,
                x="Distance",
                y="price_per_acre",
                points="all",
                hover_data=["property_id", "street", "city", "lot_acres", "current_price", "property_url"],
                log_y=True,
                title="Current Active Price per Acre",
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        scatter_df = active_df[
            active_df["lot_acres"].gt(0)
            & active_df["current_price"].gt(0)
        ].copy()
        if len(scatter_df):
            fig = px.scatter(
                scatter_df,
                x="lot_acres",
                y="current_price",
                color="Distance",
                hover_data=["property_id", "street", "city", "current_price", "price_per_acre", "property_url"],
                log_x=True,
                log_y=True,
                title="Current Price vs Acreage — Log / Log",
            )
            st.plotly_chart(fig, width="stretch")

    with tab3:
        fig = go.Figure()
        for distance in ["Further", "Close"]:
            sub = curve_df[curve_df["Distance"] == distance]
            fig.add_trace(go.Scatter(
                x=sub["acres"],
                y=sub["estimate"],
                mode="lines+markers",
                name=distance,
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([sub["acres"], sub["acres"][::-1]]),
                y=pd.concat([sub["ci_high"], sub["ci_low"][::-1]]),
                fill="toself",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                opacity=0.18,
            ))

        fig.add_hline(
            y=budget,
            line_dash="dash",
            annotation_text=f"Budget: ${budget:,.0f}",
        )
        fig.update_layout(
            title="Modeled Acquisition Cost by Acreage",
            xaxis_title="Target Acres",
            yaxis_title="Estimated Price",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

# ============================================================
# PROPERTY TABLE
# ============================================================

st.subheader("Property inventory")

with st.expander("Property table filters", expanded=False):
    show_status = st.multiselect(
        "Statuses",
        options=["FOR_SALE", "SOLD"],
        default=["FOR_SALE", "SOLD"],
    )
    table_min_acres = st.number_input(
        "Table minimum acres",
        min_value=0.0,
        value=float(min_acres),
        step=1.0,
    )
    table_max_acres = st.number_input(
        "Table maximum acres",
        min_value=0.0,
        value=float(max_acres),
        step=5.0,
    )
    budget_only = st.checkbox(
        "Only show properties at or below budget",
        value=False,
    )
    selected_distance = st.multiselect(
        "Distance groups",
        options=["Close", "Further"],
        default=["Close", "Further"],
    )

property_df = selected_df[
    selected_df["status_norm"].isin(show_status)
    & selected_df["Distance"].isin(selected_distance)
    & selected_df["lot_acres"].between(table_min_acres, table_max_acres)
].copy()

if budget_only:
    property_df = property_df[
        property_df["current_price"].le(budget)
    ].copy()

if property_df.empty:
    st.info("No properties match the current property-table filters.")
else:
    property_df["Listing"] = property_df["property_url"].where(
        property_df["property_url"].astype(str).str.startswith(("http://", "https://")),
        "",
    )

    desired = [
        "Distance",
        "Listing",
        "property_id",
        "status",
        "_search_location",
        "street",
        "city",
        "county",
        "state",
        "zip_code",
        "lot_acres",
        "current_price",
        "price_per_acre",
        "site_status",
        "beds",
        "full_baths",
        "sqft",
        "year_built",
        "days_on_mls",
        "last_sold_date",
        "text",
    ]
    desired = [c for c in desired if c in property_df.columns]
    remaining = [c for c in property_df.columns if c not in desired and not c.startswith("_")]
    property_table = property_df[desired + remaining].copy()

    st.dataframe(
        property_table,
        width="stretch",
        height=700,
        hide_index=True,
        column_config={
            "Listing": st.column_config.LinkColumn(
                "Listing",
                display_text="Open Listing ↗",
                help="Opens the Realtor.com listing URL.",
            ),
            "current_price": st.column_config.NumberColumn("Price", format="$%,.0f"),
            "price_per_acre": st.column_config.NumberColumn("$/Acre", format="$%,.0f"),
            "lot_acres": st.column_config.NumberColumn("Acres", format="%.2f"),
            "beds": st.column_config.NumberColumn("Beds", format="%.0f"),
            "full_baths": st.column_config.NumberColumn("Full Baths", format="%.0f"),
            "sqft": st.column_config.NumberColumn("Sq Ft", format="%,.0f"),
            "days_on_mls": st.column_config.NumberColumn("Days on MLS", format="%.0f"),
        },
    )

    st.caption(
        f"Showing {len(property_table):,} properties. Listing links use the property_url returned by HomeHarvest."
    )

    st.download_button(
        "Download current property table CSV",
        data=property_df.to_csv(index=False).encode("utf-8"),
        file_name="homeharvest_land_properties.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Data source: HomeHarvest, which retrieves structured property data from Realtor.com. "
    f"Last scrape: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
