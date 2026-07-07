# Wonderkid Detector

A machine-learning project that predicts whether a young footballer (under 21) will "explode" in market value over the next few years, using only information available at age 18-20.

The core idea: instead of relying on scout opinions, market-value history itself defines who becomes a "wonderkid" and the model learns the early signals that separate future stars from players who plateau.

---

## What "wonderkid" means here

A player is labelled a wonderkid if, by ages 21-25, they **both**:

1. **at least tripled** their market value compared to their first valuation under 21, **and**
2. **reached at least EUR 10 million** in value.

The EUR 10M floor stops trivial rises (e.g. EUR 10k → 30k) from counting as wonderkids. About **8.3%** of young players in the data meet this bar.

---

## The dataset

Kaggle's [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores), downloaded automatically through the `kagglehub` API. Three files are used:

| File | Rows | Used for |
| --- | --- | --- |
| `players.csv` | ~48k | player bios: position, foot, height |
| `player_valuations.csv` | ~656k | market-value history (used to build the label) |
| `appearances.csv` | ~1.9M | per-game stats: goals, assists, minutes, cards |

---

## Project files

| File | What it is |
| --- | --- |
| `Wonderkid_Classifier.ipynb` | Full project notebook: data cleaning, label engineering, models, demo |
| `app.py` | Streamlit web app for the live demo |
| `wonderkid_model.joblib` | Trained model bundle loaded by the app |
| `requirements.txt` | Full local environment for the notebook |
| `requirements-app.txt` | Lean dependencies for the Streamlit app |
| `.env.example` | Template for Kaggle credentials |
| `.gitignore` | Keeps secrets, virtual environments, and datasets out of Git |

---

## Local setup

### 1. Install the environment

```powershell
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 2. Add Kaggle credentials

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings), open the **API** section, and click **Create New Token** to download `kaggle.json`.
2. Copy `.env.example` to `.env` and fill in your details:

```
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

`.env` is in `.gitignore`, so your real credentials are never committed.

### 3. Run the notebook

Open `Wonderkid_Classifier.ipynb`, select the **Python (WonderKidClassifier)** kernel, and run all cells. The last cell saves `wonderkid_model.joblib`.

### 4. Launch the demo app locally

```powershell
.\venv\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501` in your browser.

---

## How the model works

1. **Build the label** from valuation history: first value under 21 vs peak value at ages 21-25.
2. **Build the features** using only information known at the first valuation:
   - position, foot, height
   - age at first valuation
   - starting market value (log-scaled)
   - league group (big-5 vs other)
   - early-career per-game stats (goals, assists, minutes, cards)
   - a flag for whether senior appearance data exists
3. **Train and compare models**: Decision Tree, kNN, Naive Bayes, Logistic Regression, Random Forest, HistGradientBoosting, and XGBoost.
4. **Tune** hyperparameters and the decision threshold (not just 0.5).
5. **Calibrate** probabilities so displayed scores are trustworthy.
6. **Demo, cluster, and save** the model for the app.

---

## Headline results

- Best model: **tuned, calibrated XGBoost** with ROC-AUC around **0.81**.
- At the balanced threshold, it flags roughly **10%** of players (close to the true 8.3% base rate).
- On real players it gives high scores to Bellingham, Pedri, Musiala, Wirtz; low scores to ordinary cheap players.
- Honest misses are expected: Mbappe scored low at his first valuation because he was a EUR 50k unknown with almost no senior data. Predicting before the market catches on is the hard ceiling of this task.

---


## Key design choices

- **Leakage-free:** all performance features use only games played *up to* the first valuation.
- **Missing-data flag:** distinguishes "no senior data" from actual zeros.
- **Realistic label:** the EUR 10M floor prevents trivial 3x rises from being counted as wonderkids.
- **Calibrated probabilities:** displayed percentages reflect real likelihoods, not inflated XGBoost scores.
