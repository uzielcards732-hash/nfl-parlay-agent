
import json
import pandas as pd
import streamlit as st

# Local modules you will include in your repo
import odds_loader
from parlay_builder import build_parlays

st.set_page_config(page_title="NFL Parlay Ideas (Info Only)", layout="wide")
st.title("NFL Parlay Ideas (Info Only)")

with st.sidebar:
    st.header("Data Source")
    mode = st.radio("How do you want to load odds today?", ["Fetch from The Odds API", "Upload CSV"])

    st.markdown("---")
    st.header("Parlay Settings")
    max_legs = st.slider("Max legs", 2, 5, 5)
    min_leg_ev = st.slider("Minimum leg EV", 0.0, 0.15, 0.02, 0.005)
    min_joint_p = st.slider("Minimum joint probability", 0.0, 0.30, 0.05, 0.01)
    max_bad_price = st.number_input("Reject legs worse than (American)", value=-300, step=5)

    st.markdown("---")
    st.caption("This app shows model-derived ideas for information/education only. It does not place bets.")

legs = []

if mode == "Fetch from The Odds API":
    st.subheader("Fetch Daily Snapshot")
    api_key = st.text_input("The Odds API key", type="password")
    col1, col2 = st.columns(2)
    with col1:
        region = st.text_input("Region", value="us")
    with col2:
        books_str = st.text_input("Restrict to bookmakers (comma-separated, optional)", value="")

    default_markets = ",".join(odds_loader.DEFAULT_MARKETS)
    markets_str = st.text_input("Markets (comma-separated)", value=default_markets,
                                help="Include player props; each market is one API request.")

    if st.button("Fetch snapshot now"):
        if not api_key:
            st.error("Please enter your API key.")
        else:
            books = [b.strip() for b in books_str.split(",") if b.strip()] or None
            markets = [m.strip() for m in markets_str.split(",") if m.strip()]
            with st.spinner("Fetching odds and computing consensus probabilities..."):
                try:
                    legs, headers = odds_loader.snapshot_daily(api_key=api_key, region=region, books=books, markets=markets)
                    st.success(f"Fetched {len(legs)} legs. Requests remaining: {headers.get('x-requests-remaining')}")
                except Exception as e:
                    st.exception(e)

    if legs:
        df_legs = pd.DataFrame(legs)
        st.write("### Legs (raw)")
        st.dataframe(df_legs, use_container_width=True)
        st.download_button("Download legs CSV", df_legs.to_csv(index=False).encode("utf-8"),
                           file_name="nfl_daily.csv", mime="text/csv")

elif mode == "Upload CSV":
    st.subheader("Upload legs CSV")
    f = st.file_uploader("Upload CSV with columns: id, description, odds, p_true, game_id", type=["csv"])
    if f is not None:
        df_legs = pd.read_csv(f)
        legs = df_legs.to_dict("records")
        st.write("### Legs (raw)")
        st.dataframe(df_legs, use_container_width=True)

# Build and display parlays if we have legs
if legs:
    st.markdown("---")
    st.subheader("Top Parlay Ideas")

    df = build_parlays(
        legs=legs,
        max_legs=max_legs,
        min_leg_ev=min_leg_ev,
        max_bad_price=max_bad_price,
        min_joint_p_true=min_joint_p,
        forbid_same_game_non_sgp=True,
        correlation_matrix=None,   # You can wire JSON correlations here later
        n_sims_joint=30000
    )
    if df.empty:
        st.info("No parlays passed the filters today. Try lowering the EV or joint probability thresholds.")
    else:
        st.dataframe(df, use_container_width=True, height=500)
        st.download_button("Download parlay ideas CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="parlay_ideas.csv", mime="text/csv")
