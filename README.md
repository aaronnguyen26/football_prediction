# ⚽ EPL Match Outcome & Score Predictor

A machine learning-powered football prediction model for the English Premier League, using **expected goals (xG)**, **Elo ratings**, **rolling form**, and **advanced team stats** from [FBref](https://fbref.com) to predict match outcomes and scorelines.

Built with **Python, XGBoost**, this project combines data science and web development to create a robust, interactive predictor.

---

## 🔍 Features

- ✅ Predicts **Home Win / Draw / Away Win** probabilities
- ✅ Forecasts **expected goals** and **most likely scorelines** using Poisson simulation
- ✅ Uses **FBref xG and team stats** (2017–2024) for true team strength modeling
- ✅ Implements **Elo ratings** with home advantage
- ✅ Dynamic **weighted rolling averages** (recent form emphasized)
- ✅ **Calibrated probabilities** for more reliable confidence estimates
- ✅ Head-to-head (H2H) stats and historical scorelines

## 📊 Model Performance

| Metric | Score |
|-------|-------|
| Accuracy (H/D/A) | ~54–58% |
| Home Goal MAE | ~1.03 |
| Away Goal MAE | ~0.91 |

---
## 🛠️ Tech Stack

- **Python** – Core logic
- **Pandas, NumPy** – Data processing
- **XGBoost** – Match outcome & goal prediction
- **Scikit-learn** – Calibration, preprocessing
- **BeautifulSoup** – Data scraping (FBref)
- **Requests, lxml** – HTML fetching and parsing

---
## 🗂️ Project Structure

footballprediction/
├── prediction_model.py # Core prediction logic
├── fbref_scraper.py # Scrapes team stats from FBref
├── epl_data/ # Match CSVs (epl2017.csv → epl2024.csv)
├── fbref_data/ # Scraped team stats (scrapped_team_stats_*.csv)
└── README.md

---

## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/epl-predictor.git
   cd epl-predictor
2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
3. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost requests beautifulsoup4 lxml
   





