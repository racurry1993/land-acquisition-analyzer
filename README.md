# Land Acquisition Analyzer - HomeHarvest Streamlit App

This Streamlit app pulls property data directly from HomeHarvest/Realtor.com based on two user-defined location groups. The user does not need to upload a CSV.

## Default locations

### Location 1 / Close
- Eagleville, TN
- Rockvale, TN
- Chapel Hill, TN
- Thompsons Station, TN
- Bethesda, TN
- Spring Hill, TN
- Unionville, TN
- Versailles, TN
- Midland, TN
- Wilhoite Mills, TN
- Holts Corner, TN
- Lasea, TN
- Santa Fe, TN
- Boston, TN

### Location 2 / Further
- Dickson, TN
- Bon Aqua, TN
- Primm Springs, TN
- Centerville, TN
- Fairview, TN
- Boston, TN
- Lyles, TN
- Burns, TN
- Nunnelly, TN

Boston is intentionally present in both user-supplied defaults. The app resolves overlapping search locations deterministically: Location 1 / Close wins and the duplicate is removed from Location 2 / Further, with a warning displayed in the UI.

## Data acquisition

For every search location, the app performs two HomeHarvest searches:

- `for_sale` for current inventory
- `sold` with the selected sold lookback (default 180 days)

The results are combined, deduplicated, feature-engineered from listing text, and labeled Close/Further based on which location group generated the result.

## Run locally

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Community Cloud

Commit `app.py` and `requirements.txt` (and this README if desired) to the repository. No source CSV is required for normal operation.

Streamlit entry point: `app.py`

## Runtime dependency

`homeharvest` is a required runtime dependency because the app pulls data live instead of reading a static CSV.
