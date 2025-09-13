import os
import json
import pandas as pd
import streamlit as st

# Local modules (files in the same repo)
import odds_loader
from parlay_builder import build_parlays

# ---- Page setup ----
st.set_page_config(page_title="NFL Parlay Ideas (Info Only)", layout="wide")
st.title("NFL Parlay Ideas (Information Only)")

# Try to read a default API key from Streamlit Secrets (optional convenience)
ODDS_API_KEY_DEFAULT = None
try:
    ODDS_API_KEY_DEFAULT = st.secrets.get("THE_ODDS_API_KEY", None)
except Exception:
    pass

# Build default markets string from odds_loader constants (new names)
try:
    default_markets_list = (
        getattr(odds_loader, "DEFAULT_STANDARD_MARKETS", [])
        + getattr(odds_loader, "DEFAULT_PROP_MARKETS", [])
    )
    DEFAULT_MARKETS_STR = ",".join(default_markets_list)
except Exception:
    # Fallback in case constants are missing for any reason
    DEFAULT_MARKETS_STR = (
        "h2h,spreads,totals,"
        "player_pass_tds,player_pass_yds,player_rush_yds,player_receptions,player_anytime_td"
    )

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Data Source")
    mode = st.radio(
        "How do you want to load odds today?",
        ["Fetch from The Odds API", "Upload CSV"],
        index=0
    )

    st.markdown("---")
    st.header("Parlay Settings")
    max_legs = st.slider("Max legs", 2, 5, 5)
    min_leg_ev = st.slider("Minimum leg EV", 0.0, 0.15, 0.02, 0.005)
    min_joint_p = st.slider("Minimum joint probability", 0.0, 0.30, 0.05, 0.01)
    max_bad_price = st.number_input(
        "Reject legs worse than (American)", value=-300, step=5
    )

    st.markdown("---")
    st.caption(
        "This app displays model-derived ideas for information/education only. "
        "It does not place bets."
    )

legs = []

# ---- Mode: Fetch from API ----
if mode == "Fetch from The Odds API":
    st.subheader("Fetch Daily Snapshot")

    api_key = st.text_input(
        "The Odds API key",
        type="password",
        value=ODDS_API_KEY_DEFAULT or ""
    )

    col1, col2 = st.columns(2)
    with col1:
        region = st.text_input("Region", value="us")
    with col2:
        books_str = st.text_input(
            "Restrict to bookmakers (comma-separated, optional)", value=""
        )

    markets_str = st.text_input(
        "Markets (comma-separated)",
        value=DEFAULT_MARKETS_STR,
        help=(
            "Standard: h2h, spreads, totals. Player props (e.g., player_pass_tds, "
            "player_pass_yds, player_rush_yds, player_receptions, player_anytime_td). "
            "Props are fetched per-event to avoid 422 errors."
        ),
    )

    if st.button("Fetch snapshot now"):
        if not api_key.strip():
            st.error("Please enter your API key (or add THE_ODDS_API_KEY to Streamlit Secrets).")
        else:
            books = [b.strip() for b in books_str.split(",") if b.strip()] or None
            markets = [m.strip() for m in markets_str.split(",") if m.strip()]
            with st.spinner("Fetching odds and computing consensus probabilities..."):
                try:
                    legs, headers = odds_loader.snapshot_daily(
                        api_key=api_key.strip(), region=region.strip(), books=books, markets=markets
                    )
                    remaining = headers.get("x-requests-remaining")
                    used = headers.get("x-requests-used")
                    st.success(f"Fetched {len(legs)} legs. Requests remaining: {remaining} (used: {used}).")
                except Exception as e:
                    st.error("Fetching failed. See details below.")
                    st.exception(e)

    # Show raw legs and download
    if legs:
        df_legs = pd.DataFrame(legs)
        st.write("### Legs (raw)")
        st.dataframe(df_legs, use_container_width=True, height=420)
        st.download_button(
            "Download legs CSV",
            df_legs.to_csv(index=False).encode("utf-8"),
            file_name="nfl_daily.csv",
            mime="text/csv",
        )

# ---- Mode: Upload CSV ----
elif mode == "Upload CSV":
    st.subheader("Upload legs CSV")
    f = st.file_uploader(
        "Upload CSV with columns: id, description, odds, p_true, game_id",
        type=["csv"]
    )
    if f is not None:
        try:
            df_legs = pd.read_csv(f)
            # basic validation
            expected = {"id", "description", "odds", "p_true"}
            missing = expected - set(df_legs.columns)
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                legs = df_legs.to_dict("records")
                st.write("### Legs (raw)")
                st.dataframe(df_legs, use_container_width=True, height=420)
        except Exception as e:
            st.error("Could not read the CSV. Make sure it has the required columns.")
            st.exception(e)

# ---- Build & show parlay ideas ----
if legs:
    st.markdown("---")
    st.subheader("Top Parlay Ideas")

    try:
        df = build_parlays(
            legs=legs,
            max_legs=max_legs,
            min_leg_ev=min_leg_ev,
            max_bad_price=max_bad_price,
            min_joint_p_true=min_joint_p,
            forbid_same_game_non_sgp=True,   # allow same-game only if you wire correlations later
            correlation_matrix=None,
            n_sims_joint=30000,
        )
    except Exception as e:
        st.error("Failed to build parlays from the provided legs.")
        st.exception(e)
        df = pd.DataFrame()

    if df.empty:
        st.info("No parlays passed the filters. Try lowering the EV or joint probability thresholds.")
    else:
        st.dataframe(df, use_container_width=True, height=520)
        st.download_button(
            "Download parlay ideas CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="parlay_ideas.csv",
            mime="text/csv",
        )
