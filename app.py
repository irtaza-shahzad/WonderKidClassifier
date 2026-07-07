import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "wonderkid_model.joblib"
st.set_page_config(page_title="Wonderkid Detector", page_icon=None, layout="centered")


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


st.title("Wonderkid Detector")
st.caption(
    "Predicts whether a player under 21 will roughly triple their market value within a few years, "
    "using only what we know at age 18-20."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found. Run the notebook end to end first - the final cell "
        "saves the model bundle that this app loads."
    )
    st.stop()

bundle = load_bundle(MODEL_PATH)
model = bundle["model"]
default_threshold = float(bundle["threshold"])
base_rate = float(bundle.get("base_rate", 0.08))
feature_cols = bundle["feature_cols"]
positions = bundle["positions"]
feet = bundle["feet"]
league_groups = bundle["league_groups"]

st.info(
    f"Only about {base_rate*100:.0f}% of young players in the data are wonderkids, so a probability "
    "well above that is a strong signal. The model outputs a calibrated probability; use the "
    "threshold slider to make it stricter or more of a wide scouting net."
)

st.sidebar.header("Player profile")

position = st.sidebar.selectbox("Position", positions)
foot = st.sidebar.selectbox("Preferred foot", feet)
league_group = st.sidebar.selectbox("League group", league_groups)
height_in_cm = st.sidebar.slider("Height (cm)", 150, 210, 180)
age_at_valuation = st.sidebar.slider("Age at first valuation", 15.0, 21.0, 18.0, 0.1)
starting_value = st.sidebar.number_input(
    "Market value at 18-20 (EUR)", min_value=0, value=1_000_000, step=250_000
)

has_appearances = st.sidebar.checkbox("Has senior appearance data", value=True)

st.sidebar.header("Model settings")
threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.90, default_threshold, 0.01,
    help="Lower = a wider scouting net (more players flagged, more false alarms). "
         "Higher = stricter (only the clearest wonderkids). Default is the balanced (F1) value.",
)

if has_appearances:
    total_games = st.sidebar.slider("Senior games (before valuation)", 0, 120, 25)
    goals_per_game = st.sidebar.slider("Goals per game", 0.0, 1.5, 0.3, 0.01)
    assists_per_game = st.sidebar.slider("Assists per game", 0.0, 1.0, 0.15, 0.01)
    minutes_per_game = st.sidebar.slider("Minutes per game", 0, 90, 70)
    cards_per_game = st.sidebar.slider("Cards per game", 0.0, 1.0, 0.1, 0.01)
else:
    total_games = goals_per_game = assists_per_game = minutes_per_game = cards_per_game = 0

profile = {
    "position": position,
    "foot": foot,
    "league_group": league_group,
    "height_in_cm": float(height_in_cm),
    "age_at_valuation": float(age_at_valuation),
    "log_starting_value": float(np.log1p(starting_value)),
    "total_games": float(total_games),
    "goals_per_game": float(goals_per_game),
    "assists_per_game": float(assists_per_game),
    "minutes_per_game": float(minutes_per_game),
    "cards_per_game": float(cards_per_game),
    "has_senior_appearances": int(has_appearances),
}

row = pd.DataFrame([profile])[feature_cols]

if st.button("Predict", type="primary"):
    proba = float(model.predict_proba(row)[:, 1][0])
    is_wonderkid = proba >= threshold

    col1, col2, col3 = st.columns(3)
    col1.metric("Wonderkid probability", f"{proba*100:.1f}%")
    col2.metric("Decision threshold", f"{threshold*100:.1f}%")
    col3.metric("vs base rate", f"{proba/base_rate:.1f}x", help="How many times the average young player's odds")

    if is_wonderkid:
        st.success("Prediction: WONDERKID - flagged as a likely breakout talent.")
    else:
        st.info("Prediction: Not a wonderkid - unlikely to both triple in value and reach 10M+ EUR.")

    st.progress(min(proba, 1.0))
    st.caption(
        "'Wonderkid' here means a player who at least tripled their market value AND reached at "
        "least 10M EUR by ages 21-25. Move the threshold slider to trade off catching more "
        "wonderkids vs fewer false alarms."
    )

with st.expander("What is this model?"):
    st.markdown(
        "- Trained on Kaggle's `davidcariboo/player-scores` dataset.\n"
        "- Label: a player who at least tripled their market value AND reached 10M+ EUR by "
        "ages 21-25 (engineered purely from valuation data, no scout opinions).\n"
        "- Features use only information available at the first under-21 valuation, so there is "
        "no leakage from the future.\n"
        "- Model: tuned XGBoost with calibrated probabilities and a tunable decision threshold."
    )
