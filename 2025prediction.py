import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os
import warnings
from scipy.stats import poisson

# -----------------------------
# 1. Load and Validate Data
# -----------------------------
csv_files = [
    'footballprediction/epl_data/epl2024.csv',
    'footballprediction/epl_data/epl2023.csv',
    'footballprediction/epl_data/epl2022.csv',
    'footballprediction/epl_data/epl2021.csv',
    'footballprediction/epl_data/epl2020.csv',
    'footballprediction/epl_data/epl2019.csv',
    'footballprediction/epl_data/epl2018.csv',
    'footballprediction/epl_data/epl2017.csv',
]

df_list = []
for f in csv_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")
    df = pd.read_csv(f)
    # Extract season from filename (e.g., epl2024.csv -> 2024)
    season_year = f.split('/')[-1].replace('.csv', '').replace('epl', '')
    df['Season'] = int(season_year)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

# Check required columns
required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Parse Date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')  # Handles DD/MM/YYYY or MM/DD/YYYY
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# -----------------------------
# 2. Preprocess Data
# -----------------------------
columns_needed = [
    'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR'
]
df = df[[col for col in columns_needed if col in df.columns]]

# Drop rows where FTR is missing
df = df.dropna(subset=['FTR'])

# -----------------------------
# 3. Rolling Features (Form & Stats)
# -----------------------------
def add_rolling_features(df, window=5):
    df = df.copy()
    stats = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    
    for stat in stats:
        # Home team's rolling average of stat (e.g., FTHG = goals scored at home)
        df[f'home_avg_{stat}'] = (
            df.groupby('HomeTeam')[stat].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        )
        # Away team's rolling average of stat (e.g., FTAG = goals scored away)
        df[f'away_avg_{stat}'] = (
            df.groupby('AwayTeam')[stat].transform(lambda x: x.shift().rolling(window, min_periods=1).mean())
        )
        # Fill NaNs with league average
        df[f'home_avg_{stat}'] = df[f'home_avg_{stat}'].fillna(df[stat].median())
        df[f'away_avg_{stat}'] = df[f'away_avg_{stat}'].fillna(df[stat].median())

    # Form: 3 for win, 1 for draw, 0 for loss
    def rolling_form(results):
        points = results.map({'H': 3, 'D': 1, 'A': 0})
        return points.shift().rolling(window, min_periods=1).sum()
    
    df['home_form'] = df.groupby('HomeTeam')['FTR'].transform(rolling_form).fillna(0)
    df['away_form'] = df.groupby('AwayTeam')['FTR'].transform(rolling_form).fillna(0)
    
    return df

df = add_rolling_features(df)

# -----------------------------
# 4. Team Encoding
# -----------------------------
all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
le_team = LabelEncoder()
le_team.fit(all_teams)
df['HomeTeam_enc'] = le_team.transform(df['HomeTeam'])
df['AwayTeam_enc'] = le_team.transform(df['AwayTeam'])

# -----------------------------
# 5. Elo Rating System (with final state saved)
# -----------------------------
def calculate_elo(df, base_elo=1500, k=20, home_advantage=70):
    elos = {team: base_elo for team in all_teams}
    home_elo_list, away_elo_list = [], []

    for idx, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        home_elo, away_elo = elos[home], elos[away]

        # Apply home advantage to expected score
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        exp_away = 1 - exp_home

        if result == 'H':
            score_home, score_away = 1, 0
        elif result == 'A':
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        # Update Elo
        elos[home] += k * (score_home - exp_home)
        elos[away] += k * (score_away - exp_home)

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    return df, elos  # Return final Elos

df, final_elos = calculate_elo(df)

# -----------------------------
# 6. Target Encoding
# -----------------------------
df['FTR_enc'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# -----------------------------
# 7. Final Feature Set
# -----------------------------
feature_cols = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'home_elo', 'away_elo',
    'home_form', 'away_form',
    # Home stats
    'home_avg_FTHG', 'home_avg_HS', 'home_avg_HST', 'home_avg_HC', 'home_avg_HY', 'home_avg_HR',
    # Away stats (opponent's performance when away)
    'away_avg_FTAG', 'away_avg_AS', 'away_avg_AST', 'away_avg_AC', 'away_avg_AY', 'away_avg_AR',
    # Defensive stats: how many goals conceded
    'home_avg_FTAG',  # home team's avg goals conceded at home
    'away_avg_FTHG',  # away team's avg goals conceded away
]

# Ensure all features exist
feature_cols = [col for col in feature_cols if col in df.columns]
X = df[feature_cols]
y = df['FTR_enc']
score_y_home = df['FTHG']
score_y_away = df['FTAG']

# -----------------------------
# 8. Time-Based Train/Test Split (Critical!)
# -----------------------------
split_year = 2022  # Train on 2017–2022, test on 2023–2024
train_mask = df['Season'] <= split_year
test_mask = df['Season'] > split_year

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

score_X_train = X_train
score_X_test = X_test
score_y_home_train = score_y_home[train_mask]
score_y_home_test = score_y_home[test_mask]
score_y_away_train = score_y_away[train_mask]
score_y_away_test = score_y_away[test_mask]

# If test set is empty, fallback to random (shouldn't happen)
if len(X_test) == 0:
    print("Warning: No test data found after split. Falling back to random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    score_X_train, score_X_test, score_y_home_train, score_y_home_test = train_test_split(X, score_y_home, test_size=0.2, random_state=42)
    _, _, score_y_away_train, score_y_away_test = train_test_split(X, score_y_away, test_size=0.2, random_state=42)

# -----------------------------
# 9. Model Training
# -----------------------------
# Outcome model
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Goal models
home_goal_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
away_goal_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
home_goal_model.fit(score_X_train, score_y_home_train)
away_goal_model.fit(score_X_train, score_y_away_train)

# -----------------------------
# 10. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("=== Match Outcome Prediction ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

print("\n=== Goal Prediction (Regression) ===")
home_goal_pred = home_goal_model.predict(score_X_test)
away_goal_pred = away_goal_model.predict(score_X_test)
print("Home Goal MAE:", mean_absolute_error(score_y_home_test, home_goal_pred))
print("Away Goal MAE:", mean_absolute_error(score_y_away_test, away_goal_pred))

# -----------------------------
# 11. Prediction Helper Functions
# -----------------------------
def get_latest_or_mean(df, col, mask):
    sub = df[mask].dropna(subset=[col])
    return sub[col].iloc[-1] if not sub.empty else df[col].median()

def get_features_for_prediction(home_team, away_team, df, feature_cols, le_team, final_elos):
    if home_team not in le_team.classes_ or away_team not in le_team.classes_:
        raise ValueError("Team not in training data.")

    features = []
    features.append(le_team.transform([home_team])[0])
    features.append(le_team.transform([away_team])[0])

    # Use final Elo (most up-to-date)
    features.append(final_elos[home_team])
    features.append(final_elos[away_team])

    # Form
    features.append(get_latest_or_mean(df, 'home_form', df['HomeTeam'] == home_team))
    features.append(get_latest_or_mean(df, 'away_form', df['AwayTeam'] == away_team))

    # Rolling stats
    for stat in feature_cols[6:]:
        if stat.startswith('home_avg_'):
            base_col = stat
            val = get_latest_or_mean(df, base_col, df['HomeTeam'] == home_team)
        elif stat.startswith('away_avg_'):
            base_col = stat
            val = get_latest_or_mean(df, base_col, df['AwayTeam'] == away_team)
        else:
            val = df[stat].median()
        features.append(val)

    return np.array([features])

# -----------------------------
# 12. Predict Match with Poisson-Based Scorelines
# -----------------------------
def predict_match_proba(home_team, away_team):
    features = get_features_for_prediction(home_team, away_team, df, feature_cols, le_team, final_elos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        proba = model.predict_proba(features)[0]
        home_goals_pred = home_goal_model.predict(features)[0]
        away_goals_pred = away_goal_model.predict(features)[0]

    # Generate top scorelines using Poisson
    scores = []
    for h in range(6):
        for a in range(6):
            p = poisson.pmf(h, home_goals_pred) * poisson.pmf(a, away_goals_pred)
            scores.append((h, a, p))
    scores.sort(key=lambda x: -x[2])
    top3_poisson = scores[:3]

    # Fallback: H2H historical scores
    h2h = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
    if len(h2h) > 0:
        h2h_scores = h2h.groupby(['FTHG', 'FTAG']).size().reset_index(name='count')
        h2h_scores['prob'] = h2h_scores['count'] / h2h_scores['count'].sum()
        top3_h2h = h2h_scores.sort_values('prob', ascending=False).head(3)
    else:
        top3_h2h = None

    most_likely = top3_poisson[0]
    return proba, (int(most_likely[0]), int(most_likely[1]), most_likely[2]), top3_poisson, top3_h2h, home_goals_pred, away_goals_pred

# -----------------------------
# 13. Interactive CLI
# -----------------------------
if __name__ == "__main__":
    print("\n--- EPL Match Outcome & Score Predictor ---")
    valid_teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    print("Available teams:")
    print(", ".join(valid_teams))

    while True:
        home_team = input("\nEnter Home Team (or 'exit' to quit): ").strip()
        if home_team.lower() == 'exit':
            print("Exiting predictor.")
            break
        away_team = input("Enter Away Team (or 'exit' to quit): ").strip()
        if away_team.lower() == 'exit':
            print("Exiting predictor.")
            break

        if home_team not in valid_teams:
            print(f"Error: '{home_team}' not recognized.")
            continue
        if away_team not in valid_teams:
            print(f"Error: '{away_team}' not recognized.")
            continue

        try:
            proba, (hg, ag, score_p), top3_poisson, top3_h2h, pred_hg, pred_ag = predict_match_proba(home_team, away_team)
            
            # H2H Stats
            h2h = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
            h2h_hw = (h2h['FTR'] == 'H').sum()
            h2h_aw = (h2h['FTR'] == 'A').sum()
            h2h_d = (h2h['FTR'] == 'D').sum()
            print(f"\nHead-to-Head: {len(h2h)} matches | {home_team} wins: {h2h_hw} | Draws: {h2h_d} | {away_team} wins: {h2h_aw}")

            print(f"\nPrediction for {home_team} vs {away_team}:")
            print(f"Home Win: {proba[0]*100:.1f}% | Draw: {proba[1]*100:.1f}% | Away Win: {proba[2]*100:.1f}%")
            print(f"Expected Goals: {pred_hg:.2f} - {pred_ag:.2f}")
            print(f"Most Likely Score (Poisson): {hg}-{ag} (prob: {score_p*100:.1f}%)")

            print("\nTop 3 Likely Scorelines (Poisson):")
            for i, (h, a, p) in enumerate(top3_poisson):
                print(f"  {h}-{a}: {p*100:.1f}%")

            if top3_h2h is not None:
                print(f"\nTop 3 H2H Scorelines:")
                for _, row in top3_h2h.iterrows():
                    print(f"  {int(row['FTHG'])}-{int(row['FTAG'])}: {row['prob']*100:.1f}%")

        except Exception as e:
            print(f"Error predicting: {e}")