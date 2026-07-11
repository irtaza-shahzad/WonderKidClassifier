import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

MODEL_PATH = "wonderkid_model.joblib"
st.set_page_config(page_title="Wonderkid Detector", page_icon=None, layout="centered")

READABLE_FEATURES = {
    "height in cm": "Height (cm)",
    "age at valuation": "Age at valuation",
    "log starting value": "Starting market value (log)",
    "total games": "Senior games",
    "goals per game": "Goals per game",
    "assists per game": "Assists per game",
    "minutes per game": "Minutes per game",
    "cards per game": "Cards per game",
    "has senior appearances": "Has senior appearances",
}


@st.cache_resource
def load_bundle(path, file_mtime):
    return joblib.load(path)


@st.cache_resource
def build_shap_explainer(model_mtime, _explain_model):
    preprocessor = _explain_model.named_steps["preprocessor"]
    classifier = _explain_model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    return preprocessor, explainer, feature_names


def readable_feature_name(name):
    if name.startswith("cat__"):
        body = name.removeprefix("cat__")
        if body.startswith("position_"):
            return f"Position: {body.removeprefix('position_')}"
        if body.startswith("foot_"):
            return f"Foot: {body.removeprefix('foot_')}"
        if body.startswith("league_group_"):
            return f"League: {body.removeprefix('league_group_')}"
        return body.replace("_", " ").title()
    if name.startswith("num__"):
        body = name.removeprefix("num__").replace("_", " ")
        return READABLE_FEATURES.get(body, body.title())
    return name.replace("_", " ").title()


def shap_contributions(explanation, feature_names):
    values = explanation.values[0]
    rows = []
    for feature_name, shap_value in zip(feature_names, values, strict=False):
        rows.append(
            {
                "Feature": readable_feature_name(feature_name),
                "SHAP value": float(shap_value),
                "Impact": "Increases probability" if shap_value > 0 else "Decreases probability",
            }
        )
    return pd.DataFrame(rows).sort_values("SHAP value", key=np.abs, ascending=False)


def render_shap_section(explain_model, row, model_mtime):
    st.subheader("Why this prediction? (SHAP)")
    st.caption(
        "SHAP shows how each input pushed the model's raw score up or down for this player. "
        "Positive values increase wonderkid probability; negative values decrease it."
    )

    preprocessor, explainer, feature_names = build_shap_explainer(model_mtime, explain_model)
    transformed_row = preprocessor.transform(row)
    explanation = explainer(transformed_row)
    contributions = shap_contributions(explanation, feature_names)
    top = contributions.head(10).sort_values("SHAP value")

    fig_bar, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in top["SHAP value"]]
    ax.barh(top["Feature"], top["SHAP value"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on wonderkid score)")
    ax.set_title("Top feature contributions for this player")
    st.pyplot(fig_bar, clear_figure=True)

    st.dataframe(
        contributions.head(10)[["Feature", "SHAP value", "Impact"]],
        use_container_width=True,
        hide_index=True,
    )

    try:
        shap.plots.waterfall(explanation[0], max_display=10, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
    except Exception as waterfall_error:
        st.caption(f"Waterfall plot unavailable: {waterfall_error}")


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

model_mtime = os.path.getmtime(MODEL_PATH)
bundle = load_bundle(MODEL_PATH, model_mtime)
model = bundle["model"]
explain_model = bundle.get("explain_model")
default_threshold = float(bundle["threshold"])
base_rate = float(bundle.get("base_rate", 0.08))
feature_cols = bundle["feature_cols"]
positions = bundle["positions"]
feet = bundle["feet"]
league_groups = bundle["league_groups"]

if explain_model is None:
    st.warning(
        "This model bundle does not include SHAP support yet. Re-run the notebook's final save cell "
        "and push the updated wonderkid_model.joblib to GitHub."
    )

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

    if explain_model is not None:
        try:
            render_shap_section(explain_model, row, model_mtime)
        except Exception as exc:
            st.error("SHAP explanation failed.")
            st.exception(exc)

with st.expander("What is this model?"):
    st.markdown(
        "- Trained on Kaggle's `davidcariboo/player-scores` dataset.\n"
        "- Label: a player who at least tripled their market value AND reached 10M+ EUR by "
        "ages 21-25 (engineered purely from valuation data, no scout opinions).\n"
        "- Features use only information available at the first under-21 valuation, so there is "
        "no leakage from the future.\n"
        "- Model: tuned XGBoost with calibrated probabilities, a tunable decision threshold, "
        "and SHAP explanations in the demo."
    )
