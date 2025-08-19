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

