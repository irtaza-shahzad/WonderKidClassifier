# Wonderkid Detector

A machine-learning project that looks at a young footballer (under 21) and predicts whether they
are going to "explode" in market value over the next few years, using only the information you
would have known about them at age 18-20.

The core idea: instead of asking scouts for their opinion, we let cold market-value data define who
counts as a "wonderkid", and train a model to spot the early signals that separate future stars from
players who plateau.

---

## What "wonderkid" means here

A player is labelled a wonderkid if, by ages 21-25, they **both**:

1. **at least tripled** their market value compared to their first valuation under 21, **and**
2. **reached at least EUR 10 million** in value.

Both conditions matter. The absolute EUR 10M floor is what stops trivial cases (for example a
player going from EUR 10k to EUR 30k is technically "3x" but is obviously not a star) from counting.
About **8.3%** of young players in the data meet this bar, so it is genuinely rare.

---

## The dataset

Kaggle's [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores),
downloaded automatically through the `kagglehub` API (no manual downloads). Three files are used:

| File | Rows | Used for |
| --- | --- | --- |
| `players.csv` | ~48k | player bios: position, foot, height, nationality |
| `player_valuations.csv` | ~656k | market-value history over time (used to build the label) |
| `appearances.csv` | ~1.9M | per-game stats: goals, assists, minutes, cards |

---

## Project files

| File | What it is |
| --- | --- |
| `Wonderkid_Classifier.ipynb` | The full project notebook (data cleaning, label, models, demo) |
| `app.py` | Streamlit web app for the live demo |
| `wonderkid_model.joblib` | The trained model saved for the app |
| `requirements.txt` | All Python packages needed |
| `.env.example` | Template for your Kaggle credentials |
| `.gitignore` | Keeps secrets, the virtual env, and datasets out of Git |

---

## Setup and how to run

### 1. Install the packages

```powershell
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Add your Kaggle credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings), open the **API** section, and click
   **Create New Token**. This downloads a `kaggle.json` file.
2. Copy `.env.example` to a new file called `.env` and fill in your details:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

The notebook reads these automatically. Your real credentials never get committed to Git because
`.env` is in `.gitignore`.

### 3. Run the notebook

Open `Wonderkid_Classifier.ipynb`, select the **Python (WonderKidClassifier)** kernel, and run all
cells top to bottom. The last cell saves `wonderkid_model.joblib`.

### 4. Launch the demo app

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

It opens at `http://localhost:8501`. Enter a young player's profile in the sidebar and press
**Predict**.

---

## How the model works (in plain terms)

1. **Build the label** - join the valuations file with itself to find each player's first value
   under 21 and their peak value at ages 21-25, then apply the wonderkid rule above.
2. **Build the features** - only using information available at age 18-20:
   - position, preferred foot, height
   - age at first valuation
   - the starting market value (log-scaled)
   - which league group they play in (big-5 league vs other)
   - early-career per-game stats (goals, assists, minutes, cards)
   - a flag for whether senior appearance data even exists
3. **Train and compare models** - a simple roadmap baseline (Decision Tree, kNN, Naive Bayes) plus
   stronger models (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, and an
   XGBoost + SMOTE version).
4. **Tune and pick a threshold** - the best model is tuned with cross-validation, and the decision
   cut-off is chosen deliberately rather than left at the default 0.5.
5. **Explain, cluster, demo, save** - feature importance, K-Means player archetypes, a live demo on
   real players, and saving the model for the app.

---

## Results (headline)

- Best model: a **tuned, calibrated XGBoost**.
- **ROC-AUC around 0.81** - the model ranks players well.
- At the **balanced threshold**, it flags about **10%** of players as wonderkids (very close to the
  true 8.3% rate) instead of flagging almost everyone.
- On real players it gives Bellingham 0.90, Pedri 0.95, Musiala 0.77, Wirtz 0.80, while ordinary
  cheap players score near zero.
- Honest misses tell the real story: Mbappe scored low because at his first valuation he was a
  EUR 50k unknown - there was simply nothing early to flag. That is the natural ceiling of this
  kind of prediction.

Market-value jumps are inherently noisy (transfers, hype, injuries, coaching), so the goal is not a
perfect score. The goal is a fair, leakage-free model that clearly beats guessing and is easy to
explain.

---

## Changes and fixes made (the full journey)

This section explains everything that was changed from the original version, and why.

### A. Environment and setup

| Change | Why it helps |
| --- | --- |
| Created a virtual environment and registered a named Jupyter kernel | Fixed the "billion errors" from running against the wrong Python. Everything now runs in one clean, reproducible environment. |
| Moved Kaggle credentials into a `.env` file (loaded with `python-dotenv`) | No more hard-coded secrets in the notebook. Safe to push to GitHub. |
| Added `.gitignore`, `.env.example`, `requirements.txt` | Standard, shareable project setup. Secrets, the virtual env, and large CSVs stay out of Git. |
| Switched dataset download to work locally (removed the Colab-only bits) | The notebook now runs on your machine and caches the dataset so it downloads only once. |

### B. Correctness fixes (these were real bugs)

| Change | Why it helps |
| --- | --- |
| **Fixed data leakage in the performance window.** Old code used appearances from 180 days *before* to 730 days *after* the first valuation. The "after" part overlapped the period we are trying to predict. | Now it only uses games played *up to* the first valuation, so the model genuinely predicts the future instead of peeking at it. This is the single most important honesty fix. |
| **Fixed leakage in missing-value filling.** Old code filled missing height/foot using averages computed over the whole dataset (including the test set) before splitting. | Filling now happens inside the model pipeline, so those averages are learned only from training data. |
| **Added a "has senior appearance data" flag.** 71.5% of players had no appearance data and their stats were filled with 0. | The model can now tell "played 40 games and scored 0" apart from "we have no data". Before, those looked identical and confused the model. |

### C. Better features

| Change | Why it helps |
| --- | --- |
| **Added the starting market value as a feature (log-scaled).** It was thrown away before. | This turned out to be the single strongest signal - the market already prices in most of a player's potential by 18. Ignoring it was a big miss. |
| Added age at first valuation and a big-5-league-vs-other feature | A EUR 500k valuation in the Premier League means something different from EUR 500k in a small league. Age also matters. |
| Dropped `international_caps` | It is a career-total stat, so it leaks the future (good players earn more caps *because* they became good). |

### D. Stronger models and better evaluation

| Change | Why it helps |
| --- | --- |
| Added Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost | The roadmap models (Decision Tree, kNN, Naive Bayes) were weak. These handle the data far better. |
| Switched the main metrics to PR-AUC, ROC-AUC, and F2 | Plain accuracy is misleading when only ~8% of players are positive. These metrics actually reflect how well the rare class is found. |
| Added proper hyperparameter tuning (`RandomizedSearchCV`) | Finds better model settings automatically. |

### E. Handling the rare-class problem

| Change | Why it helps |
| --- | --- |
| Used class weighting / `scale_pos_weight` and an XGBoost + SMOTE variant | Makes the model pay attention to the rare wonderkid class instead of ignoring it. |
| **Tuned the decision threshold** (instead of always using 0.5) | This is the biggest lever on recall. We compute a balanced (F1) threshold as the default and a high-recall (F2) "scouting net" threshold as an alternative. |

### F. Fixing "it predicts wonderkid for everyone"

| Change | Why it helps |
| --- | --- |
| **Redefined the label** to require tripling in value *and* reaching EUR 10M | The old rule counted trivial rises (like EUR 10k to 30k) as wonderkids. The new rule is realistic and makes the positive class properly rare (8.3%). |
| **Switched the app to the balanced threshold** | The old deployed threshold was tuned only for recall and flagged about half of all players. The balanced one flags only about 10%. |
| **Calibrated the probabilities** | Raw scores were inflated, so an average player could read "60%". After calibration, a "30%" really means about a 30% chance - ordinary players now score 0.01-0.11 and stars score 0.5-0.95. |

### G. The demo and the app

| Change | Why it helps |
| --- | --- |
| Added a live demo on real players (Bellingham, Pedri, Musiala, and more) | Instantly shows the model works, and its honest misses make great presentation points. |
| Added K-Means + PCA clustering | Finds natural player "archetypes" and checks whether wonderkids cluster together. |
| Built a Streamlit app with a threshold slider and base-rate context | Lets anyone try the model, and dial between "strict" and "wide net". |
| Saved the trained model with `joblib` | So the app loads instantly without retraining. |

---

## Possible next steps

- Add per-player explanations (which features pushed a prediction up or down) to the app.
- Add nationality or club-strength features.
- Deploy the app publicly on Streamlit Community Cloud.
