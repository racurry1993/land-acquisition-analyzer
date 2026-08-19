from __future__ import annotations

import re
from datetime import datetime

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


# ============================================================
# DEFAULT LOCATIONS - MATCH NOTEBOOK
# ============================================================

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
    "Holts Corner, TN",
    "Lasea, TN",
    "Santa Fe, TN",
    "Boston, TN",
]

# NOTE: Boston was NOT in the notebook's original Further list.
# Keep it out here so this exactly matches the notebook comparison.
DEFAULT_LOCATION_2 = [
    "Dickson, TN",
    "Bon Aqua, TN",
    "Primm Springs, TN",
    "Centerville, TN",
    "Fairview, TN",
    "Lyles, TN",
    "Burns, TN",
    "Nunnelly, TN",
    "Wilhoite Mills, TN",
]

SOLD_LOOKBACK_DAYS = 180
CI_LEVEL = 0.80
EXCLUDED_PROPERTY_ID = "7778085034"
MODEL_FORMULA = "np.log(price) ~ np.log(lot_acres) + C(Distance)"


# ============================================================
# LOCATION HELPERS
# ============================================================

def parse_locations(raw: str) -> list[str]:
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


def build_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative text-derived site/perk flags for filtering only.

    These flags are NOT used by the valuation model. They are intended to
    help users inspect how described site characteristics relate to prices.
    """
    out = df.copy()
    text = (
        out.get("text", pd.Series("", index=out.index))
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Conservative definition: language indicating an existing/approved/passed
    # perk/perc/soil/septic site. This intentionally avoids treating every
    # generic mention of septic/perk as a positive site.
    positive = (
        text.str.contains(
            r"(?:approved|passed|existing|installed|functional).{0,100}(?:perc|perk|soil|septic)"
            r"|(?:perc|perk|soil|septic).{0,100}(?:approved|passed|existing|installed|functional)",
            regex=True,
            na=False,
        )
        | text.str.contains(
            r"(?:4\s*br|5\s*br|4-5\s*bedroom|3-4\s*br).{0,80}(?:soil|perk|perc|homesite)",
            regex=True,
            na=False,
        )
    )

    negative = text.str.contains(
        r"(?:failed|fail|no|not|does not|doesn't|won't|will not).{0,50}(?:perc|perk|soil|septic)"
        r"|(?:perc|perk|soil|septic).{0,50}(?:failed|fail|not suitable|unsuitable)",
        regex=True,
        na=False,
    )

    out["has_perked"] = np.select(
        [negative, positive],
        ["No", "Yes"],
        default="Unknown",
    )

    out["has_perked_flag"] = out["has_perked"].eq("Yes").astype(int)
    return out


# ============================================================
# ACREAGE LEVELS
# ============================================================

def make_acreage_levels(min_acres: float, max_acres: float) -> list[int]:
    if min_acres <= 0 or max_acres <= 0:
        raise ValueError("Acreage values must be positive.")
    if min_acres >= max_acres:
        raise ValueError("Maximum acreage must be greater than minimum acreage.")

    start = int(np.ceil(min_acres / 5.0) * 5)
    stop = int(np.floor(max_acres))
    levels = []

    # 5-acre increments through 30 acres.
    current = start
    while current <= min(stop, 30):
        levels.append(current)
        current += 5

    # 10-acre increments above 30 acres.
    if stop > 30:
        current = max(40, ((levels[-1] + 10) if levels else 40))
        while current <= stop:
            levels.append(current)
            current += 10

    max_int = int(round(max_acres))
    if max_int not in levels:
        levels.append(max_int)

    return sorted(set(levels))


# ============================================================
# HOMEHARVEST / NOTEBOOK DATA REPRODUCTION
# ============================================================

def normalize_homeharvest_frame(
    df: pd.DataFrame,
    group_name: str,
    location: str,
    min_acres: float,
    status_query: str,
) -> pd.DataFrame:
    """Replicate the notebook's core transformations as closely as possible."""

    out = df.copy()

    # Ensure expected columns exist.
    for col, default in {
        "property_id": np.nan,
        "property_url": "",
        "status": "",
        "city": "",
        "county": "",
        "state": "",
        "zip_code": "",
        "street": "",
        "text": "",
        "list_price": np.nan,
        "sold_price": np.nan,
        "lot_sqft": np.nan,
        "last_sold_date": pd.NaT,
        "days_on_mls": np.nan,
        "beds": np.nan,
        "full_baths": np.nan,
        "half_baths": np.nan,
        "sqft": np.nan,
        "year_built": np.nan,
        "latitude": np.nan,
        "longitude": np.nan,
    }.items():
        if col not in out.columns:
            out[col] = default

    # EXACT NOTEBOOK APPROACH: lot_acres comes from lot_sqft.
    out["lot_sqft"] = pd.to_numeric(out["lot_sqft"], errors="coerce")
    out["lot_acres"] = out["lot_sqft"] / 43560.0

    # EXACT NOTEBOOK LOT FILTER: minimum only, no maximum scrape filter.
    out = out[out["lot_acres"] >= min_acres].copy()

    if out.empty:
        return out

    # Same sold-date filtering behavior as notebook.
    if status_query == "sold":
        out["last_sold_date"] = pd.to_datetime(
            out["last_sold_date"], errors="coerce"
        )
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=SOLD_LOOKBACK_DAYS)
        out = out[out["last_sold_date"] >= cutoff].copy()

    out["_search_group"] = group_name
    out["_search_location"] = location

    return out


@st.cache_data(ttl=3600, show_spinner=False)
def scrape_locations(
    location_1: tuple[str, ...],
    location_2: tuple[str, ...],
    min_acres: float,
    sold_days: int,
) -> pd.DataFrame:
    """Replicate the notebook's two location-group scraping pattern."""

    groups = [
        ("Close", list(location_1)),
        ("Further", list(location_2)),
    ]

    all_records = []

    for group_name, locations in groups:
        for location in locations:
            for status in ["for_sale", "sold"]:
                try:
                    kwargs = {
                        "location": location,
                        "listing_type": status,
                        "property_type": ["land"],
                        "return_type": "pandas",
                        "parallel": False,
                    }

                    if status == "sold":
                        kwargs["past_days"] = int(sold_days)

                    data = scrape_property(**kwargs)

                    if data is None or data.empty:
                        continue

                    filtered = normalize_homeharvest_frame(
                        data,
                        group_name,
                        location,
                        min_acres,
                        status,
                    )

                    if not filtered.empty:
                        all_records.append(filtered)

                except Exception as exc:
                    # Keep other locations running if one query fails.
                    st.warning(
                        f"HomeHarvest query failed for {location} / {status}: {exc}"
                    )

    if not all_records:
        return pd.DataFrame()

    # EXACT NOTEBOOK STRUCTURE: concatenate records; no dedupe step.
    full_df = pd.concat(all_records, ignore_index=True)

    # EXACT NOTEBOOK GROUP LABEL.
    full_df["Distance"] = full_df["_search_group"]

    # EXACT NOTEBOOK PRICE FUNCTION.
    status_upper = full_df["status"].astype(str).str.upper()
    full_df["price"] = np.where(
        status_upper.isin(["FOR_SALE", "CONTINGENT", "PENDING"]),
        pd.to_numeric(full_df["list_price"], errors="coerce"),
        np.where(
            status_upper.eq("SOLD"),
            pd.to_numeric(full_df["sold_price"], errors="coerce"),
            np.nan,
        ),
    )

    full_df["price_per_acre"] = (
        pd.to_numeric(full_df["price"], errors="coerce")
        / pd.to_numeric(full_df["lot_acres"], errors="coerce")
    )

    full_df = build_text_features(full_df)

    return full_df.reset_index(drop=True)


# ============================================================
# EXACT NOTEBOOK MODEL
# ============================================================

def fit_notebook_model(full_df: pd.DataFrame, min_acres: float, max_acres: float):
    """
    Fit the exact notebook model:

        np.log(price) ~ np.log(lot_acres) + C(Distance)

    No smearing.
    No active-only filter.
    No text features.
    No acreage re-filtering after the notebook scrape.
    One explicit property exclusion matches the notebook.
    """

    model_df = full_df.copy()

    model_df["property_id_str"] = model_df["property_id"].astype(str)

    model_df = model_df[
        model_df["property_id_str"] != EXCLUDED_PROPERTY_ID
    ].copy()

    model_df["price"] = pd.to_numeric(
        model_df["price"], errors="coerce"
    )
    model_df["lot_acres"] = pd.to_numeric(
        model_df["lot_acres"], errors="coerce"
    )

    model_df = model_df[
        (model_df["lot_acres"] >= min_acres)
        & (model_df["lot_acres"] <= max_acres)
        & (model_df["price"] > 0)
        & (model_df["lot_acres"] > 0)
    ].copy()

    model_df = model_df.dropna(
        subset=["price", "lot_acres", "Distance"]
    ).copy()

    # Force Close to be the reference category so that:
    # C(Distance)[T.Further] = Further relative to Close.
    model_df["Distance"] = pd.Categorical(
        model_df["Distance"],
        categories=["Close", "Further"],
        ordered=False,
    )

    model = smf.ols(
        MODEL_FORMULA,
        data=model_df,
    ).fit(cov_type="HC3")

    return model, model_df


# ============================================================
# PREDICTION CURVE - EXACT STATSmodels INPUTS
# ============================================================

def build_prediction_curve(
    model,
    acreage_levels: list[int],
    ci_alpha: float = 0.20,
) -> pd.DataFrame:
    """Generate point estimates plus confidence/prediction intervals."""

    rows = []

    for acres in acreage_levels:
        for distance in ["Close", "Further"]:
            prediction_df = pd.DataFrame({
                "lot_acres": [acres],
                "Distance": pd.Categorical(
                    [distance],
                    categories=["Close", "Further"],
                ),
            })

            pred = model.get_prediction(
                prediction_df
            ).summary_frame(alpha=ci_alpha)

            rows.append({
                "acres": acres,
                "Distance": distance,
                "estimate": float(np.exp(pred["mean"].iloc[0])),
                "ci_low": float(np.exp(pred["mean_ci_lower"].iloc[0])),
                "ci_high": float(np.exp(pred["mean_ci_upper"].iloc[0])),
                "pi_low": float(np.exp(pred["obs_ci_lower"].iloc[0])),
                "pi_high": float(np.exp(pred["obs_ci_upper"].iloc[0])),
            })

    return pd.DataFrame(rows)


# ============================================================
# APP
# ============================================================

st.title("🌳 Land Acquisition Analyzer")
st.caption("HomeHarvest-powered land comparison using the notebook's exact model specification")
st.info(
    "Model used for all valuation outputs: "
    "`np.log(price) ~ np.log(lot_acres) + C(Distance)`  "
    "(no smearing correction)"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("1. Locations")

    location_1_raw = st.text_area(
        "Location 1 / Close",
        value="\n".join(DEFAULT_LOCATION_1),
        height=230,
    )

    location_2_raw = st.text_area(
        "Location 2 / Further",
        value="\n".join(DEFAULT_LOCATION_2),
        height=180,
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

    sold_days = st.number_input(
        "Sold lookback (days)",
        min_value=30,
        max_value=365,
        value=SOLD_LOOKBACK_DAYS,
        step=30,
    )

    refresh = st.button(
        "🔄 Refresh HomeHarvest data",
        width="stretch",
    )

    if refresh:
        scrape_locations.clear()
        st.rerun()


# ============================================================
# VALIDATION
# ============================================================

errors = []

if not location_1:
    errors.append("Location 1 cannot be empty.")

if not location_2:
    errors.append("Location 2 cannot be empty.")

if min_acres >= max_acres:
    errors.append("Maximum acreage must be greater than minimum acreage.")

l1_norm = {normalize_location(x) for x in location_1}
l2_norm = {normalize_location(x) for x in location_2}
overlap = sorted(l1_norm & l2_norm)

if overlap:
    overlap_set = set(overlap)
    location_2 = [
        x for x in location_2
        if normalize_location(x) not in overlap_set
    ]
    st.warning(
        "These locations were entered in both groups: "
        + ", ".join(overlap)
        + ". Location 1 / Close wins so classification is unambiguous."
    )

if not location_2:
    errors.append("Location 2 is empty after removing overlaps.")

if errors:
    for error in errors:
        st.error(error)
    st.stop()


# ============================================================
# SCRAPE
# ============================================================

with st.spinner(
    "HomeHarvest is pulling current land listings and sold land properties..."
):
    full_df = scrape_locations(
        tuple(location_1),
        tuple(location_2),
        float(min_acres),
        int(sold_days),
    )

if full_df.empty:
    st.error("HomeHarvest returned no matching properties.")
    st.stop()


# ============================================================
# SUMMARY
# ============================================================

st.success(
    f"Pulled {len(full_df):,} rows from HomeHarvest: "
    f"{int(full_df['status'].astype(str).str.upper().eq('FOR_SALE').sum()):,} current listings and "
    f"{int(full_df['status'].astype(str).str.upper().eq('SOLD').sum()):,} sold."
)

summary_cols = st.columns(6)
summary_cols[0].metric("Location 1 / Close", len(location_1))
summary_cols[1].metric("Location 2 / Further", len(location_2))
summary_cols[2].metric("Total rows", f"{len(full_df):,}")
summary_cols[3].metric("Budget", f"${budget:,.0f}")
summary_cols[4].metric("Minimum acres", f"{min_acres:g}")
summary_cols[5].metric("Sold lookback", f"{sold_days} days")


# ============================================================
# MODEL
# ============================================================

try:
    final_model, model_df = fit_notebook_model(
        full_df, min_acres, max_acres
    )
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.exception(exc)
    st.stop()

# ============================================================
# MODEL RESULTS
# ============================================================

st.subheader("Model Results")

st.caption(
    "Statistical results are recalculated from the properties returned "
    "for the currently selected locations."
)

# The fitted model is stored as final_model.
distance_term = next(
    (
        term
        for term in final_model.params.index
        if term.startswith("C(Distance)")
    ),
    None,
)

acreage_term = next(
    (
        term
        for term in final_model.params.index
        if "log(lot_acres)" in term
    ),
    None,
)

if distance_term is not None:
    distance_coef = final_model.params[distance_term]
    further_vs_close = np.exp(distance_coef)
    close_vs_further = 1 / further_vs_close
else:
    distance_coef = np.nan
    close_vs_further = np.nan

if acreage_term is not None:
    acreage_elasticity = final_model.params[acreage_term]
else:
    acreage_elasticity = np.nan

# Sold-market statistics for the selected groups.
sold_df = full_df[
    full_df["status"].astype(str).str.upper().eq("SOLD")
].copy()

sold_stats = (
    sold_df
    .groupby("Distance")["price"]
    .agg(["mean", "median", "count"])
)

def sold_value(distance: str, metric: str):
    if distance not in sold_stats.index:
        return np.nan
    value = sold_stats.loc[distance, metric]
    return float(value) if pd.notna(value) else np.nan

close_avg_sold = sold_value("Close", "mean")
close_median_sold = sold_value("Close", "median")
further_avg_sold = sold_value("Further", "mean")
further_median_sold = sold_value("Further", "median")

# ------------------------------------------------------------
# KPI row 1
# ------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Model Observations",
    f"{int(final_model.nobs):,}"
)

m2.metric(
    "R²",
    f"{final_model.rsquared:.3f}"
)

m3.metric(
    "Adjusted R²",
    f"{final_model.rsquared_adj:.3f}"
)

m4.metric(
    "Close / Further",
    f"{close_vs_further:.2f}x" if pd.notna(close_vs_further) else "N/A",
)

# ------------------------------------------------------------
# KPI row 2 - sold market prices
# ------------------------------------------------------------
m5, m6, m7, m8 = st.columns(4)

for col, label, value in [
    (m5, "Close Avg Sold Price", close_avg_sold),
    (m6, "Close Median Sold Price", close_median_sold),
    (m7, "Further Avg Sold Price", further_avg_sold),
    (m8, "Further Median Sold Price", further_median_sold),
]:
    col.metric(
        label,
        f"${value:,.0f}" if pd.notna(value) else "N/A",
    )

if pd.notna(close_vs_further) and pd.notna(acreage_elasticity):
    premium_pct = (close_vs_further - 1) * 100

    st.info(
        f"**Model interpretation:** Holding acreage constant, the model "
        f"estimates that Close properties are approximately **{close_vs_further:.2f}x** "
        f"the price of Further properties, equivalent to an estimated **{premium_pct:.0f}% premium**. "
        f"The acreage elasticity is **{acreage_elasticity:.3f}**, meaning a 1% increase "
        f"in acreage is associated with approximately a **{acreage_elasticity:.3f}% increase in total property price**."
    )

st.markdown("**Model specification**")
st.code("np.log(price) ~ np.log(lot_acres) + C(Distance)", language="python")

with st.expander("View full regression output", expanded=False):
    st.text(final_model.summary().as_text())


# ============================================================
# ACREAGE CURVE
# ============================================================

try:
    acreage_levels = make_acreage_levels(
        min_acres,
        max_acres,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.caption(
    "Acreage schedule: "
    + ", ".join(str(x) for x in acreage_levels)
)


curve_df = build_prediction_curve(
    final_model,
    acreage_levels,
    ci_alpha=1 - CI_LEVEL,
)


# ============================================================
# DECISION TABLE
# ============================================================

wide = curve_df.pivot(
    index="acres",
    columns="Distance",
    values=["estimate", "ci_low", "ci_high"],
)

wide.columns = ["_".join(c) for c in wide.columns]
wide = wide.reset_index()

wide["cost_of_proximity"] = (
    wide["estimate_Close"]
    - wide["estimate_Further"]
)

wide["close_multiple"] = (
    wide["estimate_Close"]
    / wide["estimate_Further"]
)

wide["additional_cash_close"] = (
    wide["estimate_Close"] - budget
).clip(lower=0)

wide["additional_cash_further"] = (
    wide["estimate_Further"] - budget
).clip(lower=0)

# Actual CURRENT inventory by acreage category. The first category is
# 0-min/max target (for a 15-acre target: 0-15), then prior+1 to next.
active_mask = full_df["status"].astype(str).str.upper().eq("FOR_SALE")
active_df = full_df[active_mask].copy()

inventory_rows = []
previous_upper = 0

for acre in acreage_levels:
    category = f"{int(previous_upper) + 1}-{int(acre)}" if previous_upper > 0 else f"0-{int(acre)}"

    category_df = active_df[
        active_df["lot_acres"].gt(previous_upper)
        & active_df["lot_acres"].le(acre)
    ].copy()

    close_df_cat = category_df[category_df["Distance"] == "Close"]
    further_df_cat = category_df[category_df["Distance"] == "Further"]

    close_pct = (
        close_df_cat["price"].le(budget).mean()
        if len(close_df_cat) else np.nan
    )
    further_pct = (
        further_df_cat["price"].le(budget).mean()
        if len(further_df_cat) else np.nan
    )

    inventory_rows.append({
        "acres": acre,
        "acreage_category": category,
        "close_under_budget_pct": close_pct,
        "further_under_budget_pct": further_pct,
        "close_category_n": len(close_df_cat),
        "further_category_n": len(further_df_cat),
    })

    previous_upper = acre

inventory_df = pd.DataFrame(inventory_rows)
wide = wide.merge(inventory_df, on="acres", how="left")

ci_pct = int(CI_LEVEL * 100)
display_df = wide.rename(columns={
    "acres": "Acres",
    "acreage_category": "Acreage Category",
    "estimate_Close": "Close Estimated",
    "estimate_Further": "Further Estimated",
    "ci_low_Close": f"Close {ci_pct}% CI Low",
    "ci_high_Close": f"Close {ci_pct}% CI High",
    "ci_low_Further": f"Further {ci_pct}% CI Low",
    "ci_high_Further": f"Further {ci_pct}% CI High",
    "cost_of_proximity": "Cost of Proximity",
    "close_multiple": "Close / Further",
    "additional_cash_close": "Extra Cash for Close",
    "additional_cash_further": "Extra Cash for Further",
    "close_under_budget_pct": "Close <= Budget %",
    "further_under_budget_pct": "Further <= Budget %",
}).copy()

st.subheader("Acquisition decision table")

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Acres": st.column_config.NumberColumn(format="%.0f"),
        "Close Estimated": st.column_config.NumberColumn(format="$%,.0f"),
        "Further Estimated": st.column_config.NumberColumn(format="$%,.0f"),
        f"Close {ci_pct}% CI Low": st.column_config.NumberColumn(format="$%,.0f"),
        f"Close {ci_pct}% CI High": st.column_config.NumberColumn(format="$%,.0f"),
        f"Further {ci_pct}% CI Low": st.column_config.NumberColumn(format="$%,.0f"),
        f"Further {ci_pct}% CI High": st.column_config.NumberColumn(format="$%,.0f"),
        "Cost of Proximity": st.column_config.NumberColumn(format="$%,.0f"),
        "Close / Further": st.column_config.NumberColumn(format="%.2fx"),
        "Extra Cash for Close": st.column_config.NumberColumn(format="$%,.0f"),
        "Extra Cash for Further": st.column_config.NumberColumn(format="$%,.0f"),
        "Close <= Budget %": st.column_config.NumberColumn(format="%.1%"),
        "Further <= Budget %": st.column_config.NumberColumn(format="%.1%"),
    },
)

# ============================================================
# PROPERTY TABLE
# ============================================================

st.subheader("Property inventory")

with st.expander("Property table filters", expanded=False):
    # Default to the same population used by the model.
    show_status = st.multiselect(
        "Statuses",
        options=["FOR_SALE", "CONTINGENT", "PENDING", "SOLD"],
        default=["FOR_SALE", "CONTINGENT", "PENDING", "SOLD"],
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
        value=True,
    )

    selected_distance = st.multiselect(
        "Distance groups",
        options=["Close", "Further"],
        default=["Close", "Further"],
    )

    perk_filter = st.selectbox(
        "Perk / soil site description",
        ["All", "Yes", "No", "Unknown"],
        index=0,
    )

# Match the model population first, then apply optional presentation filters.
property_df = model_df.copy()
property_df["status_norm"] = property_df["status"].astype(str).str.upper()
property_df = property_df[
    property_df["status_norm"].isin(show_status)
    & property_df["Distance"].isin(selected_distance)
    & property_df["lot_acres"].between(table_min_acres, table_max_acres)
].copy()

if budget_only:
    property_df = property_df[property_df["price"].le(budget)].copy()

if perk_filter != "All":
    property_df = property_df[property_df["has_perked"].eq(perk_filter)].copy()

if property_df.empty:
    st.info("No properties match the current filters.")
else:
    property_df["Listing"] = property_df["property_url"].where(
        property_df["property_url"].astype(str).str.startswith(("http://", "https://")),
        "",
    )

    desired = [
        "Distance", "Listing", "property_id", "status", "_search_location",
        "street", "city", "county", "state", "zip_code", "lot_acres",
        "price", "price_per_acre", "list_price", "sold_price", "has_perked",
        "beds", "full_baths", "half_baths", "sqft", "year_built",
        "days_on_mls", "list_date", "last_sold_date", "latitude", "longitude", "text",
    ]
    desired = [c for c in desired if c in property_df.columns]
    remaining = [
        c for c in property_df.columns
        if c not in desired and not c.startswith("_") and c not in {"property_url", "status_norm"}
    ]
    property_table = property_df[desired + remaining].copy()

    st.dataframe(
        property_table,
        width="stretch",
        height=700,
        hide_index=True,
        column_config={
            "Listing": st.column_config.LinkColumn(
                "Listing", display_text="Open Listing ↗", help="Opens the Realtor.com listing."
            ),
            "lot_acres": st.column_config.NumberColumn("Acres", format="%.2f"),
            "price": st.column_config.NumberColumn("Price", format="$%,.0f"),
            "price_per_acre": st.column_config.NumberColumn("$/Acre", format="$%,.0f"),
            "list_price": st.column_config.NumberColumn("List Price", format="$%,.0f"),
            "sold_price": st.column_config.NumberColumn("Sold Price", format="$%,.0f"),
        },
    )

    st.caption(
        f"Showing {len(property_table):,} properties from the same acreage/model population, after the selected table filters."
    )

    st.download_button(
        "Download filtered property table CSV",
        data=property_df.to_csv(index=False).encode("utf-8"),
        file_name="homeharvest_land_properties.csv",
        mime="text/csv",
    )

# ============================================================
# WHAT-IF CALCULATOR
# ============================================================

st.subheader("What-if calculator")

city_lookup = {}
for group_name, locations in [("Close", location_1), ("Further", location_2)]:
    for raw_location in locations:
        city_name = raw_location.split(",")[0].strip()
        if city_name:
            city_lookup[city_name] = group_name

for _, row in model_df[["city", "Distance"]].dropna().drop_duplicates().iterrows():
    city_name = str(row["city"]).strip()
    if city_name and city_name not in city_lookup:
        city_lookup[city_name] = row["Distance"]

what_if_options = ["Close", "Further"] + sorted(
    city for city in city_lookup.keys() if city not in {"Close", "Further"}
)

w1, w2 = st.columns(2)
with w1:
    what_if_acres = st.number_input(
        "Exact acreage", min_value=0.1, max_value=5000.0, value=float(min_acres), step=0.1,
    )
with w2:
    what_if_location = st.selectbox("Location", options=what_if_options, index=0)

what_if_distance = (
    what_if_location
    if what_if_location in {"Close", "Further"}
    else city_lookup[what_if_location]
)

what_if_prediction_df = pd.DataFrame({
    "lot_acres": [what_if_acres],
    "Distance": pd.Categorical([what_if_distance], categories=["Close", "Further"]),
})

what_if_pred = final_model.get_prediction(what_if_prediction_df).summary_frame(alpha=1 - CI_LEVEL)
what_if_price = float(np.exp(what_if_pred["mean"].iloc[0]))
what_if_ci_low = float(np.exp(what_if_pred["mean_ci_lower"].iloc[0]))
what_if_ci_high = float(np.exp(what_if_pred["mean_ci_upper"].iloc[0]))
what_if_price_per_acre = what_if_price / what_if_acres
what_if_budget_gap = max(0.0, what_if_price - budget)

wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("Model Price", f"${what_if_price:,.0f}")
wc2.metric("Model $ / Acre", f"${what_if_price_per_acre:,.0f}")
wc3.metric("Additional Cash Needed", f"${what_if_budget_gap:,.0f}")
wc4.metric("Model Distance", what_if_distance)

st.caption(
    f"{what_if_location} is modeled as {what_if_distance}; the regression uses acreage and Close/Further location only."
)
st.write(
    f"{int(CI_LEVEL * 100)}% CI for estimated mean price: ${what_if_ci_low:,.0f} – ${what_if_ci_high:,.0f}"
)

# ============================================================
# VISUALS
# ============================================================

st.subheader("Valuation visuals")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Price / Acre Boxplot",
    "Observed Log / Log",
    "Acquisition Curve",
    "City Value Counts",
    "City Price Map",
])

with tab1:
    box_df = model_df[model_df["price_per_acre"].gt(0)].copy()
    if not box_df.empty:
        fig = px.box(
            box_df,
            x="Distance", y="price_per_acre", points="all", log_y=True,
            hover_data=["property_id", "street", "city", "zip_code", "lot_acres", "price", "property_url"],
            title="Model Population Price per Acre",
        )
        st.plotly_chart(fig, width="stretch")

with tab2:
    fig = go.Figure()
    for distance in ["Further", "Close"]:
        sub = model_df[
            (model_df["Distance"] == distance)
            & (model_df["lot_acres"] <= max_acres)
            & (model_df["lot_acres"] >= min_acres)
        ]
        fig.add_trace(go.Scatter(
            x=sub["lot_acres"], y=sub["price"], mode="markers", name=f"{distance} observed",
            text=sub["property_id"].astype(str),
            hovertemplate="Property: %{text}<br>Acres: %{x:.2f}<br>Price: $%{y:,.0f}<extra></extra>",
        ))
    for distance in ["Further", "Close"]:
        sub = curve_df[curve_df["Distance"] == distance]
        fig.add_trace(go.Scatter(
            x=sub["acres"], y=sub["estimate"], mode="lines+markers", name=f"{distance} model",
        ))
    fig.update_xaxes(type="log", title="Acres")
    fig.update_yaxes(type="log", title="Price")
    fig.update_layout(title="Observed Properties + Exact Notebook Model", hovermode="closest")
    st.plotly_chart(fig, width="stretch")

with tab3:
    fig = go.Figure()
    for distance in ["Further", "Close"]:
        sub = curve_df[curve_df["Distance"] == distance]
        fig.add_trace(go.Scatter(
            x=sub["acres"], y=sub["estimate"], mode="lines+markers", name=distance,
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([sub["acres"], sub["acres"][::-1]]),
            y=pd.concat([sub["ci_high"], sub["ci_low"][::-1]]),
            fill="toself", line=dict(width=0), hoverinfo="skip", showlegend=False, opacity=0.15,
        ))
    fig.add_hline(y=budget, line_dash="dash", annotation_text=f"Budget: ${budget:,.0f}")
    fig.update_layout(
        title="Modeled Acquisition Cost by Acreage",
        xaxis_title="Target Acres", yaxis_title="Predicted Price", hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# City value counts: same model population + selected filters.
# ------------------------------------------------------------
with tab4:
    city_count_base = model_df.copy()
    city_count_base["status_norm"] = city_count_base["status"].astype(str).str.upper()
    city_count_base = city_count_base[
        city_count_base["status_norm"].isin(show_status)
        & city_count_base["Distance"].isin(selected_distance)
        & city_count_base["lot_acres"].between(table_min_acres, table_max_acres)
    ].copy()
    if budget_only:
        city_count_base = city_count_base[city_count_base["price"].le(budget)].copy()
    if perk_filter != "All":
        city_count_base = city_count_base[city_count_base["has_perked"].eq(perk_filter)].copy()

    city_counts = (
        city_count_base.groupby(["city", "Distance"])
        .size()
        .reset_index(name="Count")
    )

    if city_counts.empty:
        st.info("No properties match the current criteria.")
    else:
        fig = px.bar(
            city_counts.sort_values(["Count", "city"], ascending=[False, True]),
            x="city", y="Count", color="Distance", barmode="group", text="Count",
            title="Qualifying Property Counts by City",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="City", yaxis_title="Property Count", xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# City map: use ZIP code to locate aggregated city/Distance points.
# ------------------------------------------------------------
with tab5:
    map_base = model_df[
        model_df["status"].astype(str).str.upper().eq("SOLD")
        & model_df["Distance"].isin(selected_distance)
        & model_df["lot_acres"].between(table_min_acres, table_max_acres)
    ].copy()
    if budget_only:
        map_base = map_base[map_base["price"].le(budget)].copy()
    if perk_filter != "All":
        map_base = map_base[map_base["has_perked"].eq(perk_filter)].copy()

    map_base["latitude"] = pd.to_numeric(map_base["latitude"], errors="coerce")
    map_base["longitude"] = pd.to_numeric(map_base["longitude"], errors="coerce")
    map_base["zip_code"] = map_base["zip_code"].astype(str).str.strip()
    map_base = map_base[
        map_base["zip_code"].ne("")
        & map_base["zip_code"].ne("nan")
        & map_base["latitude"].notna()
        & map_base["longitude"].notna()
        & map_base["price"].notna()
    ].copy()

    # Aggregate robustly by ZIP + Distance. Some HomeHarvest/Pandas
    # combinations can raise an error with named aggregation + as_index=False,
    # so build each statistic separately and merge them.
    group_cols = ["zip_code", "Distance"]

    coords = (
        map_base.groupby(group_cols)[["latitude", "longitude"]]
        .mean()
        .reset_index()
    )

    price_stats = (
        map_base.groupby(group_cols)["price"]
        .agg(
            average_price="mean",
            median_price="median",
            sold_count="count",
        )
        .reset_index()
    )

    city_stats = (
        map_base.groupby(group_cols)["city"]
        .agg(
            city=lambda s: (
                s.dropna().astype(str).mode().iat[0]
                if not s.dropna().empty and not s.dropna().astype(str).mode().empty
                else ""
            )
        )
        .reset_index()
    )

    # ============================================================
    # CLEAN ZIP MAP DATA BEFORE PLOTTING
    # ============================================================

    # Make sure numeric fields are actually numeric.
    zip_stats["latitude"] = pd.to_numeric(
        zip_stats["latitude"],
        errors="coerce"
    )

    zip_stats["longitude"] = pd.to_numeric(
        zip_stats["longitude"],
        errors="coerce"
    )

    zip_stats["average_price"] = pd.to_numeric(
        zip_stats["average_price"],
        errors="coerce"
    )

    zip_stats["median_price"] = pd.to_numeric(
        zip_stats["median_price"],
        errors="coerce"
    )

    zip_stats["sold_count"] = pd.to_numeric(
        zip_stats["sold_count"],
        errors="coerce"
    )


    # Remove invalid map observations.
    zip_stats = zip_stats[
        zip_stats["latitude"].notna()
        &
        zip_stats["longitude"].notna()
        &
        zip_stats["average_price"].notna()
        &
        np.isfinite(zip_stats["average_price"])
        &
        (zip_stats["average_price"] > 0)
    ].copy()

    if zip_stats.empty:
        st.info(
            "No valid ZIP-level sold-price observations "
            "are available for the current filters."
        )
    else:
        fig = px.scatter_map(
            zip_stats,
            lat="latitude",
            lon="longitude",
            # ----------------------------------------------------
            # Color = Close vs Further
            # ----------------------------------------------------
            color="Distance",
            # ----------------------------------------------------
            # Bubble size = average sold price
            # ----------------------------------------------------
            size="average_price",
            size_max=40,
            # ----------------------------------------------------
            # Hover
            # ----------------------------------------------------
            hover_name="city",
            hover_data={
                "zip_code": True,
                "average_price": ":$,.0f",
                "median_price": ":$,.0f",
                "sold_count": ":,.0f",
                "latitude": False,
                "longitude": False,
            },
            zoom=8,
            height=600,
            map_style="open-street-map",
            title=(
                "Sold Price by ZIP Code — "
                "Bubble Size = Average Sold Price"
            )
        )
        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Data source: HomeHarvest / Realtor.com. "
    "Model: np.log(price) ~ np.log(lot_acres) + C(Distance). "
    "No smearing correction applied. "
    f"App run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

