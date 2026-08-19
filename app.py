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
# DEFAULT LOCATIONS
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


# ============================================================
# MODEL
# ============================================================

# This is intentionally the ONLY valuation model used
# throughout the application.

MODEL_FORMULA = (
    "log_price_model ~ "
    "log_acres_model + "
    "C(Distance)"
)


# ============================================================
# LOCATION HELPERS
# ============================================================

def parse_locations(raw: str) -> list[str]:
    """
    Parse one location per line or separated by semicolons.

    Commas are preserved so:
        Eagleville, TN

    remains one location.
    """

    if not raw or not raw.strip():
        return []

    parts = re.split(r"[;\n]+", raw)

    values = []
    seen = set()

    for item in parts:

        value = re.sub(
            r"\s+",
            " ",
            item.strip()
        )

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

    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    ).casefold()


# ============================================================
# ACREAGE LEVELS
# ============================================================

def make_acreage_levels(
    min_acres: float,
    max_acres: float
) -> list[int]:

    """
    Create dynamic acreage levels.

    5-acre increments through 30 acres.

    Then 10-acre increments.

    Example:

        15 -> 75

        15
        20
        25
        30
        40
        50
        60
        70
        75
    """

    if min_acres <= 0:
        raise ValueError(
            "Minimum acreage must be positive."
        )

    if max_acres <= 0:
        raise ValueError(
            "Maximum acreage must be positive."
        )

    if min_acres >= max_acres:
        raise ValueError(
            "Maximum acreage must be greater "
            "than minimum acreage."
        )


    start = int(
        np.ceil(min_acres / 5) * 5
    )

    stop = int(
        np.floor(max_acres)
    )


    levels = []


    # --------------------------------------------------------
    # 5-acre increments through 30
    # --------------------------------------------------------

    current = start

    while current <= min(
        stop,
        30
    ):

        levels.append(current)

        current += 5


    # --------------------------------------------------------
    # 10-acre increments after 30
    # --------------------------------------------------------

    if stop > 30:

        current = 40

        while current <= stop:

            if current >= start:

                levels.append(
                    current
                )

            current += 10


    # --------------------------------------------------------
    # Always include exact requested maximum
    # --------------------------------------------------------

    max_int = int(
        round(max_acres)
    )

    if (
        max_int > 0
        and
        max_int not in levels
    ):

        levels.append(
            max_int
        )


    levels = sorted(
        set(levels)
    )


    if not levels:

        raise ValueError(
            "Unable to create acreage levels."
        )


    return levels


# ============================================================
# DATA HELPERS
# ============================================================

def safe_num(
    df: pd.DataFrame,
    column: str
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            np.nan,
            index=df.index,
            dtype=float
        )

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


def ensure_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    """
    Normalize HomeHarvest output into a stable schema.
    """

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


    # --------------------------------------------------------
    # Acreage
    # --------------------------------------------------------

    out["lot_sqft"] = safe_num(
        out,
        "lot_sqft"
    )

    out["lot_acres"] = safe_num(
        out,
        "lot_acres"
    )


    missing_acres = (
        out["lot_acres"].isna()
        &
        out["lot_sqft"].notna()
    )


    out.loc[
        missing_acres,
        "lot_acres"
    ] = (
        out.loc[
            missing_acres,
            "lot_sqft"
        ]
        /
        43560
    )


    # --------------------------------------------------------
    # Prices
    # --------------------------------------------------------

    out["list_price"] = safe_num(
        out,
        "list_price"
    )

    out["sold_price"] = safe_num(
        out,
        "sold_price"
    )


    out["days_on_mls"] = safe_num(
        out,
        "days_on_mls"
    )


    # --------------------------------------------------------
    # Normalize status
    # --------------------------------------------------------

    out["status_norm"] = (
        out["status"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    # --------------------------------------------------------
    # Analysis price
    #
    # SOLD:
    #     sold_price
    #
    # Everything else:
    #     list_price
    # --------------------------------------------------------

    out["analysis_price_model"] = np.where(

        out["status_norm"].eq(
            "SOLD"
        ),

        out["sold_price"],

        out["list_price"]
    )


    # --------------------------------------------------------
    # Current listing price
    # --------------------------------------------------------

    out["current_price"] = (
        out["list_price"]
    )


    # --------------------------------------------------------
    # Price per acre
    # --------------------------------------------------------

    out["price_per_acre"] = (

        out["current_price"]
        /
        out["lot_acres"].replace(
            0,
            np.nan
        )
    )


    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    out["list_date"] = pd.to_datetime(
        out["list_date"],
        errors="coerce"
    )


    out["last_sold_date"] = pd.to_datetime(
        out["last_sold_date"],
        errors="coerce"
    )


    # --------------------------------------------------------
    # Model transformations
    # --------------------------------------------------------

    out["log_price_model"] = np.log(

        out[
            "analysis_price_model"
        ].where(
            out[
                "analysis_price_model"
            ] > 0
        )
    )


    out["log_acres_model"] = np.log(

        out[
            "lot_acres"
        ].where(
            out[
                "lot_acres"
            ] > 0
        )
    )


    return out


# ============================================================
# HOMEHARVEST SCRAPE
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def scrape_locations(
    location_1: tuple[str, ...],
    location_2: tuple[str, ...],
    min_acres: float,
    max_acres: float,
    sold_days: int
) -> pd.DataFrame:

    """
    Pull:

        current FOR_SALE listings

    and

        SOLD listings in the previous N days

    for both location groups.
    """


    groups = [

        (
            "Close",
            list(location_1)
        ),

        (
            "Further",
            list(location_2)
        )
    ]


    all_frames = []


    # --------------------------------------------------------
    # Use an acreage buffer around the requested analysis
    # range.
    # --------------------------------------------------------

    scrape_min_acres = max(
        1,
        min_acres * 0.75
    )


    scrape_max_acres = max(

        max_acres * 1.25,

        max_acres + 10
    )


    lot_min_sqft = int(
        round(
            scrape_min_acres
            *
            43560
        )
    )


    lot_max_sqft = int(
        round(
            scrape_max_acres
            *
            43560
        )
    )


    # ========================================================
    # SCRAPE EACH LOCATION
    # ========================================================

    for group_name, locations in groups:

        for location in locations:


            # ------------------------------------------------
            # ACTIVE LISTINGS
            # ------------------------------------------------

            try:

                active = scrape_property(

                    location=location,

                    listing_type="for_sale",

                    lot_sqft_min=lot_min_sqft,

                    lot_sqft_max=lot_max_sqft,

                    exclude_pending=True,

                    parallel=False,

                    limit=10000,

                    return_type="pandas"
                )

            except Exception:

                active = None


            if (
                active is not None
                and
                len(active) > 0
            ):

                active = active.copy()

                active[
                    "_search_group"
                ] = group_name

                active[
                    "_search_location"
                ] = location

                all_frames.append(
                    active
                )


            # ------------------------------------------------
            # SOLD
            # ------------------------------------------------

            try:

                sold = scrape_property(

                    location=location,

                    listing_type="sold",

                    past_days=int(
                        sold_days
                    ),

                    lot_sqft_min=lot_min_sqft,

                    lot_sqft_max=lot_max_sqft,

                    parallel=False,

                    limit=10000,

                    return_type="pandas"
                )

            except Exception:

                sold = None


            if (
                sold is not None
                and
                len(sold) > 0
            ):

                sold = sold.copy()

                sold[
                    "_search_group"
                ] = group_name

                sold[
                    "_search_location"
                ] = location

                all_frames.append(
                    sold
                )


    # ========================================================
    # COMBINE
    # ========================================================

    if not all_frames:

        return pd.DataFrame()


    raw = pd.concat(
        all_frames,
        ignore_index=True
    )


    raw = ensure_columns(
        raw
    )


    # ========================================================
    # DEDUPLICATE
    # ========================================================

    raw["_dedupe_key"] = (
        raw["property_id"]
        .astype(str)
    )


    missing_id = (
        raw["_dedupe_key"]
        .isin(
            [
                "nan",
                "None",
                ""
            ]
        )
    )


    raw.loc[
        missing_id,
        "_dedupe_key"
    ] = (

        raw.loc[
            missing_id,
            "property_url"
        ]
        .astype(str)
    )


    raw = raw.drop_duplicates(
        subset="_dedupe_key",
        keep="first"
    )


    # ========================================================
    # DISTANCE
    # ========================================================

    raw["Distance"] = (
        raw["_search_group"]
    )


    return (
        raw
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MODEL
# ============================================================

def train_model(
    df: pd.DataFrame
):

    """
    Fit the single valuation model used by the application.

    No smearing correction.

    Model:

        log(price)
            ~
        log(acres)
            +
        Distance
    """


    active = df.copy()


    active = active.dropna(
        subset=[
            "analysis_price_model",
            "lot_acres",
            "log_price_model",
            "log_acres_model",
            "Distance"
        ]
    )


    active = active[
        (
            active[
                "analysis_price_model"
            ] > 0
        )
        &
        (
            active[
                "lot_acres"
            ] > 0
        )
    ].copy()


    # --------------------------------------------------------
    # Require enough observations in both markets
    # --------------------------------------------------------

    counts = (
        active["Distance"]
        .value_counts()
    )


    if (
        counts.get(
            "Close",
            0
        ) < 5
        or
        counts.get(
            "Further",
            0
        ) < 5
    ):

        raise ValueError(

            "Not enough active listings in both groups. "
            f"Close={int(counts.get('Close', 0))}, "
            f"Further={int(counts.get('Further', 0))}. "
            "At least 5 observations are required in each."
        )


    # --------------------------------------------------------
    # EXACT MODEL REQUESTED
    # --------------------------------------------------------

    model = smf.ols(

        MODEL_FORMULA,

        data=active

    ).fit(
        cov_type="HC3"
    )


    return (
        model,
        active
    )


# ============================================================
# PREDICTION CURVE
# ============================================================

def predict_curve(
    model,
    acreage_levels: list[int],
    ci_alpha: float = 0.20
) -> pd.DataFrame:

    """
    Generate Close and Further model estimates.

    IMPORTANT:

    There is NO smearing correction.

    Price is simply:

        exp(predicted log price)
    """


    rows = []


    for acres in acreage_levels:

        for distance in [
            "Close",
            "Further"
        ]:


            prediction_df = pd.DataFrame({

                "log_acres_model": [
                    np.log(acres)
                ],

                "Distance": [
                    distance
                ]
            })


            frame = (

                model
                .get_prediction(
                    prediction_df
                )
                .summary_frame(
                    alpha=ci_alpha
                )
            )


            rows.append({

                "acres":
                    acres,

                "Distance":
                    distance,

                "estimate":
                    float(
                        np.exp(
                            frame[
                                "mean"
                            ].iloc[0]
                        )
                    ),

                "ci_low":
                    float(
                        np.exp(
                            frame[
                                "mean_ci_lower"
                            ].iloc[0]
                        )
                    ),

                "ci_high":
                    float(
                        np.exp(
                            frame[
                                "mean_ci_upper"
                            ].iloc[0]
                        )
                    )
            })


    return pd.DataFrame(
        rows
    )


# ============================================================
# APP HEADER
# ============================================================

st.title(
    "🌳 Land Acquisition Analyzer"
)


st.caption(
    "HomeHarvest-powered Close vs Further "
    "land acquisition analysis"
)


st.info(
    "Valuation model: "
    "`log_price_model ~ log_acres_model + C(Distance)`"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:


    # --------------------------------------------------------
    # LOCATIONS
    # --------------------------------------------------------

    st.header(
        "1. Locations"
    )


    location_1_raw = st.text_area(

        "Location 1 / Close",

        value="\n".join(
            DEFAULT_LOCATION_1
        ),

        height=220,

        help=(
            "One HomeHarvest location per line. "
            "Example: Eagleville, TN"
        )
    )


    location_2_raw = st.text_area(

        "Location 2 / Further",

        value="\n".join(
            DEFAULT_LOCATION_2
        ),

        height=180,

        help=(
            "One HomeHarvest location per line."
        )
    )


    location_1 = parse_locations(
        location_1_raw
    )


    location_2 = parse_locations(
        location_2_raw
    )


    # --------------------------------------------------------
    # ACQUISITION
    # --------------------------------------------------------

    st.header(
        "2. Acquisition"
    )


    budget = st.number_input(

        "Budget ($)",

        min_value=50_000,

        max_value=100_000_000,

        value=550_000,

        step=25_000,

        format="%.0f"
    )


    min_acres = st.number_input(

        "Minimum acreage",

        min_value=1.0,

        max_value=5000.0,

        value=15.0,

        step=1.0
    )


    max_acres = st.number_input(

        "Maximum acreage",

        min_value=2.0,

        max_value=5000.0,

        value=75.0,

        step=5.0
    )


    ci_level = st.slider(

        "Model confidence level",

        min_value=0.50,

        max_value=0.95,

        value=0.80,

        step=0.05,

        format="%.0f%%"
    )


    sold_days = st.number_input(

        "Sold lookback (days)",

        min_value=30,

        max_value=365,

        value=SOLD_LOOKBACK_DAYS,

        step=30
    )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    st.header(
        "3. Data"
    )


    refresh = st.button(
        "🔄 Refresh HomeHarvest data",
        width="stretch"
    )


    if refresh:

        scrape_locations.clear()

        st.rerun()


# ============================================================
# VALIDATION
# ============================================================

errors = []


if not location_1:

    errors.append(
        "Location 1 / Close cannot be empty."
    )


if not location_2:

    errors.append(
        "Location 2 / Further cannot be empty."
    )


if max_acres <= min_acres:

    errors.append(
        "Maximum acreage must be greater "
        "than minimum acreage."
    )


# ------------------------------------------------------------
# DUPLICATE LOCATIONS
# ------------------------------------------------------------

location_1_norm = {
    normalize_location(x)
    for x in location_1
}


location_2_norm = {
    normalize_location(x)
    for x in location_2
}


overlap = sorted(
    location_1_norm
    &
    location_2_norm
)


if overlap:

    # Location 1 wins.

    overlap_set = set(
        overlap
    )


    location_2 = [

        x

        for x in location_2

        if normalize_location(x)
        not in overlap_set
    ]


    st.warning(

        "These locations were entered in both groups: "
        +
        ", ".join(overlap)
        +
        ". They will be assigned to Location 1 / Close."
    )


if not location_2:

    errors.append(
        "Location 2 is empty after removing "
        "overlapping locations."
    )


if errors:

    for error in errors:

        st.error(
            error
        )

    st.stop()


# ============================================================
# HOMEHARVEST
# ============================================================

with st.spinner(
    "HomeHarvest is pulling current listings "
    "and recent sold properties..."
):

    try:

        raw_df = scrape_locations(

            tuple(location_1),

            tuple(location_2),

            float(min_acres),

            float(max_acres),

            int(sold_days)
        )

    except Exception as exc:

        st.error(
            "HomeHarvest could not complete the scrape."
        )

        st.exception(
            exc
        )

        st.stop()


if raw_df.empty:

    st.error(
        "HomeHarvest returned no properties "
        "for the selected locations."
    )

    st.stop()


# ============================================================
# DATA PREPARATION
# ============================================================

selected_df = raw_df.copy()


selected_df[
    "status_norm"
] = (

    selected_df["status"]
    .astype(str)
    .str.upper()
    .str.strip()
)


selected_df[
    "days_on_mls_num"
] = (

    safe_num(
        selected_df,
        "days_on_mls"
    )
    .fillna(0)
)


active_df = selected_df[
    selected_df[
        "status_norm"
    ].eq(
        "FOR_SALE"
    )
].copy()


sold_df = selected_df[
    selected_df[
        "status_norm"
    ].eq(
        "SOLD"
    )
].copy()


# ============================================================
# SUMMARY
# ============================================================

st.success(

    f"Pulled {len(selected_df):,} unique properties: "
    f"{len(active_df):,} current listings and "
    f"{len(sold_df):,} sold properties."
)


summary_cols = st.columns(
    6
)


summary_cols[0].metric(
    "Location 1 / Close",
    len(location_1)
)


summary_cols[1].metric(
    "Location 2 / Further",
    len(location_2)
)


summary_cols[2].metric(
    "Active",
    f"{len(active_df):,}"
)


summary_cols[3].metric(
    "Sold",
    f"{len(sold_df):,}"
)


summary_cols[4].metric(
    "Budget",
    f"${budget:,.0f}"
)


summary_cols[5].metric(
    "Sold lookback",
    f"{int(sold_days)} days"
)


# ============================================================
# MODEL DATA
# ============================================================

model_df = active_df[

    active_df[
        "lot_acres"
    ].between(
        min_acres * 0.75,
        max_acres * 1.25
    )

    &

    active_df[
        "analysis_price_model"
    ].notna()

    &

    active_df[
        "lot_acres"
    ].notna()

    &

    active_df[
        "analysis_price_model"
    ].gt(0)

    &

    active_df[
        "lot_acres"
    ].gt(0)

].copy()


counts = (
    model_df[
        "Distance"
    ]
    .value_counts()
)


# ============================================================
# FIT MODEL
# ============================================================

model = None


if (
    counts.get(
        "Close",
        0
    ) < 5
    or
    counts.get(
        "Further",
        0
    ) < 5
):

    st.warning(

        "There are not enough active observations "
        "to fit the model reliably. "
        f"Close={int(counts.get('Close', 0))}, "
        f"Further={int(counts.get('Further', 0))}."
    )


else:

    try:

        model, active_model = train_model(
            model_df
        )

    except Exception as exc:

        st.error(
            "Model training failed."
        )

        st.exception(
            exc
        )

        st.stop()


# ============================================================
# MODEL SUMMARY
# ============================================================

if model is not None:

    st.subheader(
        "Model summary"
    )


    col1, col2, col3 = st.columns(
        3
    )


    col1.metric(
        "R²",
        f"{model.rsquared:.3f}"
    )


    col2.metric(
        "Adjusted R²",
        f"{model.rsquared_adj:.3f}"
    )


    # --------------------------------------------------------
    # Close/Further multiple
    #
    # Statsmodels will use one Distance category as baseline.
    # --------------------------------------------------------

    distance_coef_name = next(

        (
            x
            for x in model.params.index
            if x.startswith(
                "C(Distance)"
            )
        ),

        None
    )


    if distance_coef_name:

        distance_coef = (
            model.params[
                distance_coef_name
            ]
        )


        distance_multiple = np.exp(
            distance_coef
        )


        col3.metric(
            "Distance Multiple",
            f"{distance_multiple:.2f}x"
        )


    with st.expander(
        "Regression output"
    ):

        st.text(
            model.summary().as_text()
        )


# ============================================================
# ACREAGE LEVELS
# ============================================================

try:

    acreage_levels = make_acreage_levels(

        min_acres,

        max_acres
    )

except ValueError as exc:

    st.error(
        str(exc)
    )

    st.stop()


st.caption(

    "Acreage schedule: "
    +
    ", ".join(
        str(x)
        for x in acreage_levels
    )
)


# ============================================================
# MODEL CURVE + DECISION TABLE
# ============================================================

if model is not None:


    curve_df = predict_curve(

        model,

        acreage_levels,

        ci_alpha=(
            1
            -
            ci_level
        )
    )


    # --------------------------------------------------------
    # Pivot
    # --------------------------------------------------------

    wide = curve_df.pivot(

        index="acres",

        columns="Distance",

        values=[
            "estimate",
            "ci_low",
            "ci_high"
        ]
    )


    wide.columns = [

        "_".join(c)

        for c in wide.columns
    ]


    wide = (
        wide
        .reset_index()
    )


    # --------------------------------------------------------
    # Economics
    # --------------------------------------------------------

    wide[
        "cost_of_proximity"
    ] = (

        wide[
            "estimate_Close"
        ]
        -
        wide[
            "estimate_Further"
        ]
    )


    wide[
        "close_multiple"
    ] = (

        wide[
            "estimate_Close"
        ]
        /
        wide[
            "estimate_Further"
        ]
    )


    wide[
        "close_premium_pct"
    ] = (

        wide[
            "close_multiple"
        ]
        -
        1
    )


    wide[
        "additional_cash_close"
    ] = (

        wide[
            "estimate_Close"
        ]
        -
        budget
    ).clip(
        lower=0
    )


    wide[
        "additional_cash_further"
    ] = (

        wide[
            "estimate_Further"
        ]
        -
        budget
    ).clip(
        lower=0
    )


    # ========================================================
    # ACTUAL CURRENT INVENTORY
    # ========================================================

    inventory_rows = []


    for acre in acreage_levels:


        eligible = active_df[

            active_df[
                "lot_acres"
            ].ge(
                acre
            )

            &

            active_df[
                "current_price"
            ].le(
                budget
            )
        ]


        inventory_rows.append({

            "acres":
                acre,

            "close_listings_under_budget":
                int(
                    (
                        eligible[
                            "Distance"
                        ]
                        ==
                        "Close"
                    ).sum()
                ),

            "further_listings_under_budget":
                int(
                    (
                        eligible[
                            "Distance"
                        ]
                        ==
                        "Further"
                    ).sum()
                )
        })


    inventory_df = pd.DataFrame(
        inventory_rows
    )


    wide = wide.merge(

        inventory_df,

        on="acres",

        how="left"
    )


    # ========================================================
    # DECISION TABLE
    # ========================================================

    st.subheader(
        "Acquisition decision table"
    )


    ci_pct = int(
        ci_level * 100
    )


    display_df = wide.rename(
        columns={

            "acres":
                "Acres",

            "estimate_Further":
                "Further Estimated",

            "estimate_Close":
                "Close Estimated",

            "ci_low_Further":
                f"Further {ci_pct}% CI Low",

            "ci_high_Further":
                f"Further {ci_pct}% CI High",

            "ci_low_Close":
                f"Close {ci_pct}% CI Low",

            "ci_high_Close":
                f"Close {ci_pct}% CI High",

            "cost_of_proximity":
                "Cost of Proximity",

            "additional_cash_close":
                "Extra Cash for Close",

            "additional_cash_further":
                "Extra Cash for Further",

            "close_multiple":
                "Close / Further",

            "close_premium_pct":
                "Close Premium",

            "close_listings_under_budget":
                "Close Listings <= Budget",

            "further_listings_under_budget":
                "Further Listings <= Budget"
        }
    ).copy()


    money_columns = [

        "Further Estimated",

        "Close Estimated",

        f"Further {ci_pct}% CI Low",

        f"Further {ci_pct}% CI High",

        f"Close {ci_pct}% CI Low",

        f"Close {ci_pct}% CI High",

        "Cost of Proximity",

        "Extra Cash for Close",

        "Extra Cash for Further"
    ]


    column_config = {

        "Acres":
            st.column_config.NumberColumn(
                format="%.0f"
            ),

        "Close / Further":
            st.column_config.NumberColumn(
                format="%.2fx"
            ),

        "Close Premium":
            st.column_config.NumberColumn(
                format="%.1%"
            ),

        "Close Listings <= Budget":
            st.column_config.NumberColumn(
                format="%d"
            ),

        "Further Listings <= Budget":
            st.column_config.NumberColumn(
                format="%d"
            )
    }


    for column in money_columns:

        column_config[
            column
        ] = (

            st.column_config.NumberColumn(
                format="$%,.0f"
            )
        )


    st.dataframe(

        display_df,

        width="stretch",

        hide_index=True,

        column_config=column_config
    )


    st.caption(

        "Model estimates use only log acreage and "
        "Close/Further location. No smearing correction "
        "is applied. Inventory counts are actual active "
        "listings at or below the selected budget."
    )


    # ========================================================
    # VISUALS
    # ========================================================

    st.subheader(
        "Valuation visuals"
    )


    tab1, tab2, tab3, tab4 = st.tabs([

        "Price / Acre Boxplot",

        "Price vs Acreage",

        "Log / Log Model",

        "Acquisition Curve"
    ])


    # ========================================================
    # BOXPLOT
    # ========================================================

    with tab1:


        box_df = active_df[

            active_df[
                "price_per_acre"
            ].gt(0)

        ].copy()


        if box_df.empty:

            st.info(
                "No valid price-per-acre observations."
            )


        else:

            fig = px.box(

                box_df,

                x="Distance",

                y="price_per_acre",

                points="all",

                hover_data=[

                    "property_id",

                    "street",

                    "city",

                    "lot_acres",

                    "current_price",

                    "property_url"
                ],

                log_y=True,

                title=(
                    "Current Active Price per Acre"
                )
            )


            fig.update_yaxes(
                title="Price per Acre"
            )


            st.plotly_chart(
                fig,
                width="stretch"
            )


    # ========================================================
    # PRICE VS ACREAGE
    # ========================================================

    with tab2:


        scatter_df = active_model.copy()


        fig = px.scatter(

            scatter_df,

            x="lot_acres",

            y="analysis_price_model",

            color="Distance",

            hover_data=[

                "property_id",

                "street",

                "city",

                "analysis_price_model",

                "price_per_acre",

                "property_url"
            ],

            log_x=True,

            log_y=True,

            title=(
                "Current Price vs Acreage — Log / Log"
            )
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # ========================================================
    # LOG / LOG MODEL
    # ========================================================

    with tab3:


        fig = go.Figure()


        # ----------------------------------------------------
        # Actual observations
        # ----------------------------------------------------

        for distance in [
            "Further",
            "Close"
        ]:


            actual = active_model[

                active_model[
                    "Distance"
                ].eq(
                    distance
                )
            ]


            fig.add_trace(

                go.Scatter(

                    x=actual[
                        "lot_acres"
                    ],

                    y=actual[
                        "analysis_price_model"
                    ],

                    mode="markers",

                    name=(
                        f"{distance} actual"
                    ),

                    text=actual[
                        "property_id"
                    ].astype(str),

                    customdata=np.column_stack([

                        actual[
                            "city"
                        ].astype(str),

                        actual[
                            "property_url"
                        ].astype(str)
                    ]),

                    hovertemplate=(
                        "Property: %{text}<br>"
                        "City: %{customdata[0]}<br>"
                        "Acres: %{x:.2f}<br>"
                        "Price: $%{y:,.0f}<br>"
                        "<extra></extra>"
                    )
                )
            )


        # ----------------------------------------------------
        # Model curves
        # ----------------------------------------------------

        for distance in [
            "Further",
            "Close"
        ]:


            sub = curve_df[

                curve_df[
                    "Distance"
                ].eq(
                    distance
                )
            ]


            fig.add_trace(

                go.Scatter(

                    x=sub[
                        "acres"
                    ],

                    y=sub[
                        "estimate"
                    ],

                    mode="lines+markers",

                    name=(
                        f"{distance} model"
                    )
                )
            )


        fig.update_xaxes(
            type="log",
            title="Acres"
        )


        fig.update_yaxes(
            type="log",
            title="Price"
        )


        fig.update_layout(

            title=(
                "Observed Properties + "
                "Log / Log Model Curves"
            ),

            hovermode="closest"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


    # ========================================================
    # ACQUISITION CURVE
    # ========================================================

    with tab4:


        fig = go.Figure()


        for distance in [
            "Further",
            "Close"
        ]:


            sub = curve_df[

                curve_df[
                    "Distance"
                ].eq(
                    distance
                )
            ]


            # ------------------------------------------------
            # Confidence interval
            # ------------------------------------------------

            fig.add_trace(

                go.Scatter(

                    x=pd.concat([

                        sub[
                            "acres"
                        ],

                        sub[
                            "acres"
                        ][::-1]
                    ]),

                    y=pd.concat([

                        sub[
                            "ci_high"
                        ],

                        sub[
                            "ci_low"
                        ][::-1]
                    ]),

                    fill="toself",

                    line=dict(
                        width=0
                    ),

                    hoverinfo="skip",

                    showlegend=False,

                    opacity=0.15
                )
            )


            # ------------------------------------------------
            # Model line
            # ------------------------------------------------

            fig.add_trace(

                go.Scatter(

                    x=sub[
                        "acres"
                    ],

                    y=sub[
                        "estimate"
                    ],

                    mode="lines+markers",

                    name=distance
                )
            )


        fig.add_hline(

            y=budget,

            line_dash="dash",

            annotation_text=(
                f"Budget: "
                f"${budget:,.0f}"
            )
        )


        fig.update_layout(

            title=(
                "Modeled Acquisition Cost "
                "by Acreage"
            ),

            xaxis_title=(
                "Target Acres"
            ),

            yaxis_title=(
                "Estimated Price"
            ),

            hovermode="x unified"
        )


        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# PROPERTY INVENTORY
# ============================================================

st.subheader(
    "Property inventory"
)


with st.expander(
    "Property table filters",
    expanded=False
):


    show_status = st.multiselect(

        "Statuses",

        options=[
            "FOR_SALE",
            "SOLD"
        ],

        default=[
            "FOR_SALE",
            "SOLD"
        ]
    )


    table_min_acres = st.number_input(

        "Table minimum acres",

        min_value=0.0,

        value=float(
            min_acres
        ),

        step=1.0
    )


    table_max_acres = st.number_input(

        "Table maximum acres",

        min_value=0.0,

        value=float(
            max_acres
        ),

        step=5.0
    )


    budget_only = st.checkbox(

        "Only show properties at or below budget",

        value=False
    )


    selected_distance = st.multiselect(

        "Distance groups",

        options=[
            "Close",
            "Further"
        ],

        default=[
            "Close",
            "Further"
        ]
    )


# ============================================================
# FILTER PROPERTY TABLE
# ============================================================

property_df = selected_df[

    selected_df[
        "status_norm"
    ].isin(
        show_status
    )

    &

    selected_df[
        "Distance"
    ].isin(
        selected_distance
    )

    &

    selected_df[
        "lot_acres"
    ].between(

        table_min_acres,

        table_max_acres
    )

].copy()


if budget_only:


    # SOLD properties should use sold price.
    # FOR_SALE properties should use list price.

    property_df = property_df[

        property_df[
            "analysis_price_model"
        ].le(
            budget
        )

    ].copy()


# ============================================================
# DISPLAY PROPERTY TABLE
# ============================================================

if property_df.empty:


    st.info(
        "No properties match the current filters."
    )


else:


    property_df[
        "Listing"
    ] = (

        property_df[
            "property_url"
        ].where(

            property_df[
                "property_url"
            ]
            .astype(str)
            .str.startswith(
                (
                    "http://",
                    "https://"
                )
            ),

            ""
        )
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

        "list_price",

        "sold_price",

        "analysis_price_model",

        "price_per_acre",

        "beds",

        "full_baths",

        "half_baths",

        "sqft",

        "year_built",

        "days_on_mls",

        "list_date",

        "last_sold_date",

        "latitude",

        "longitude",

        "text"
    ]


    desired = [

        col

        for col in desired

        if col in property_df.columns
    ]


    remaining = [

        col

        for col in property_df.columns

        if (
            col not in desired
            and
            not col.startswith("_")
            and
            col != "property_url"
        )
    ]


    property_table = property_df[

        desired
        +
        remaining

    ].copy()


    st.dataframe(

        property_table,

        width="stretch",

        height=700,

        hide_index=True,

        column_config={

            "Listing":

                st.column_config.LinkColumn(

                    "Listing",

                    display_text=(
                        "Open Listing ↗"
                    ),

                    help=(
                        "Open the Realtor.com listing."
                    )
                ),


            "lot_acres":

                st.column_config.NumberColumn(

                    "Acres",

                    format="%.2f"
                ),


            "list_price":

                st.column_config.NumberColumn(

                    "List Price",

                    format="$%,.0f"
                ),


            "sold_price":

                st.column_config.NumberColumn(

                    "Sold Price",

                    format="$%,.0f"
                ),


            "analysis_price_model":

                st.column_config.NumberColumn(

                    "Analysis Price",

                    format="$%,.0f"
                ),


            "price_per_acre":

                st.column_config.NumberColumn(

                    "$ / Acre",

                    format="$%,.0f"
                ),


            "beds":

                st.column_config.NumberColumn(

                    "Beds",

                    format="%.0f"
                ),


            "full_baths":

                st.column_config.NumberColumn(

                    "Full Baths",

                    format="%.0f"
                ),


            "half_baths":

                st.column_config.NumberColumn(

                    "Half Baths",

                    format="%.0f"
                ),


            "sqft":

                st.column_config.NumberColumn(

                    "Sq Ft",

                    format="%,.0f"
                ),


            "days_on_mls":

                st.column_config.NumberColumn(

                    "Days on MLS",

                    format="%.0f"
                )
        }
    )


    st.caption(

        f"Showing {len(property_table):,} properties. "
        "Listing links use the property_url "
        "returned by HomeHarvest."
    )


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    st.download_button(

        "Download filtered property table CSV",

        data=(
            property_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        ),

        file_name=(
            "homeharvest_land_properties.csv"
        ),

        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(

    "Data source: HomeHarvest / Realtor.com. "
    "Model: log_price_model ~ "
    "log_acres_model + C(Distance). "
    "No smearing correction applied. "
    f"App run: "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)