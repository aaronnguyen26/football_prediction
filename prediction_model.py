import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import os
import warnings
from scipy.stats import poisson

# -----------------------------
# 1. Suppress FutureWarning for CalibratedClassifierCV
# -----------------------------
warnings.filterwarnings(
    "ignore",
    message="The `cv='prefit'` option is deprecated",
    category=FutureWarning
)

# -----------------------------
# 2. Load and Validate Match Data (from 2017 to 2024)
# - data include match by match results with different features 
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
    season_year = int(f.split('/')[-1].replace('.csv', '').replace('epl', ''))
    df['Season'] = season_year
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)

required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# -----------------------------
# 3. Load FBref Seasonal Team Stats
# -----------------------------
fbref_data = {}
fbref_dir = 'footballprediction/fbref_data'

# Team name mapping to standardize across datasets
team_mapping = {
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Newcastle Utd': 'Newcastle United',
    'Tottenham': 'Tottenham Hotspur',
    'West Ham': 'West Ham United',
    'Leicester': 'Leicester City',
    'Wolves': 'Wolverhampton Wanderers',
    'Brighton': 'Brighton and Hove Albion',
    'Nott\'ham Forest': 'Nottingham Forest',
    'Spurs': 'Tottenham Hotspur',
    'Bournemouth': 'AFC Bournemouth',
    'Leeds': 'Leeds United',
    'Sheffield Utd': 'Sheffield United',
    'West Brom': 'West Bromwich Albion',
    'Huddersfield': 'Huddersfield Town',
    'Stoke': 'Stoke City',
    'Swansea': 'Swansea City',
    'Cardiff': 'Cardiff City',
    'Norwich': 'Norwich City',
    'Fulham': 'Fulham',
    'Brentford': 'Brentford',
    'Ipswich': 'Ipswich Town',
    'Aston Villa': 'Aston Villa'
}

for year in range(2017, 2025):
    path = os.path.join(fbref_dir, f'scrapped_team_stats_{year}.csv')
    if not os.path.exists(path):
        print(f"⚠️ FBref data not found for {year}: {path}")
        continue

    try:
        fb_df = pd.read_csv(path)
        # Standardize team names
        fb_df['Squad'] = fb_df['Squad'].replace(team_mapping)
        # Build team stats lookup
        team_stats = {}
        for _, row in fb_df.iterrows():
            team_stats[row['Squad']] = {
                'attack_xG': row['Per 90 Minutes_xG_x'],
                'defense_xGA': row['Per 90 Minutes_xG_y'],
                'attack_G': row['Per 90 Minutes_Gls_x'],
                'defense_GA': row['Per 90 Minutes_Gls_y'],
                'possession': row['Poss_x'],
                'progression': row['Progression_PrgC_x'],
                'xAG': row['Per 90 Minutes_xAG_x'],
                'npxG': row['Per 90 Minutes_npxG_x'],
                'CrdY': row['Performance_CrdY_x'],
            }
        fbref_data[year] = team_stats
        print(f"✅ Loaded FBref data for {year}")
    except Exception as e:
        print(f"❌ Failed to load FBref data for {year}: {e}")

# -----------------------------
# 4. Preprocess Match Data
# -----------------------------
columns_needed = [
    'Date', 'Season', 'HomeTeam', 'AwayTeam', 'FTR', 'HS', 'AS', 'HST', 'AST',
    'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'FTHG', 'FTAG'
]
df = df[[col for col in columns_needed if col in df.columns]].dropna(subset=['FTR'])

# -----------------------------
# 5. Add FBref Features to Each Match
# -----------------------------
def add_fbref_features(row):
    season = row['Season']
    home, away = row['HomeTeam'], row['AwayTeam']
    fb_season = fbref_data.get(season, {})

    h_stats = fb_season.get(home, None)
    a_stats = fb_season.get(away, None)

    if h_stats and a_stats:
        return pd.Series([
            h_stats['attack_xG'], h_stats['defense_xGA'], h_stats['possession'], h_stats['progression'],
            a_stats['attack_xG'], a_stats['defense_xGA'], a_stats['possession'], a_stats['progression'],
            h_stats['xAG'], a_stats['xAG'], h_stats['npxG'], a_stats['npxG'], h_stats['CrdY']
        ])
    else:
        # Fallback: league medians
        return pd.Series([1.5, 1.2, 50.0, 700, 1.5, 1.2, 50.0, 700, 1.0, 1.0, 1.3, 1.3, 1.0])

print("🔧 Adding FBref-enhanced features to match data...")
fbref_features = [
    'home_attack_xG', 'home_defense_xGA', 'home_possession', 'home_progression',
    'away_attack_xG', 'away_defense_xGA', 'away_possession', 'away_progression',
    'home_xAG', 'away_xAG', 'home_npxG', 'away_npxG', 'home_CrdY'
]
df[fbref_features] = df.apply(add_fbref_features, axis=1)

# -----------------------------
# 6. Weighted Rolling Features (Exponential Decay)
# -----------------------------
def weighted_rolling(series, window=5, alpha=0.2):
    """
    Compute exponentially weighted rolling average.
    Recent games have higher weight.
    """
    weights = np.exp(-alpha * np.arange(window)[::-1])
    weights /= weights.sum()  # Normalize
    return series.rolling(window).apply(lambda x: (x * weights).sum(), raw=True)

def add_rolling_features(df, window=5, alpha=0.2):
    df = df.copy()
    stats = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'HR']

    for stat in stats:
        # Weighted rolling average for home team
        df[f'home_avg_{stat}'] = df.groupby('HomeTeam')[stat].transform(
            lambda x: weighted_rolling(x.shift(), window=window, alpha=alpha)
        )
        # Weighted rolling average for away team
        df[f'away_avg_{stat}'] = df.groupby('AwayTeam')[stat].transform(
            lambda x: weighted_rolling(x.shift(), window=window, alpha=alpha)
        )
        # Fill NaNs with league median
        df[f'home_avg_{stat}'] = df[f'home_avg_{stat}'].fillna(df[stat].median())
        df[f'away_avg_{stat}'] = df[f'away_avg_{stat}'].fillna(df[stat].median())

    # Form: 3 for win, 1 for draw, 0 for loss
    def rolling_form(results):
        points = results.map({'H': 3, 'D': 1, 'A': 0})
        return weighted_rolling(points.shift(), window=window, alpha=alpha)
    
    df['home_form'] = df.groupby('HomeTeam')['FTR'].transform(rolling_form).fillna(1)
    df['away_form'] = df.groupby('AwayTeam')['FTR'].transform(rolling_form).fillna(1)

    return df

df = add_rolling_features(df, window=5, alpha=0.2)

# -----------------------------
# 7. Team Encoding
# -----------------------------
all_teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
le_team = LabelEncoder()
le_team.fit(all_teams)
df['HomeTeam_enc'] = le_team.transform(df['HomeTeam'])
df['AwayTeam_enc'] = le_team.transform(df['AwayTeam'])

# -----------------------------
# 8. Elo Rating System
# -----------------------------
def calculate_elo(df, base_elo=1500, k=20, home_advantage=70):
    elos = {team: base_elo for team in all_teams}
    home_elo_list, away_elo_list = [], []
    for idx, row in df.iterrows():
        home, away, result = row['HomeTeam'], row['AwayTeam'], row['FTR']
        home_elo, away_elo = elos[home], elos[away]
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - home_advantage) / 400))
        exp_away = 1 - exp_home
        if result == 'H':
            score_home, score_away = 1, 0
        elif result == 'A':
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5
        elos[home] += k * (score_home - exp_home)
        elos[away] += k * (score_away - exp_away)
        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)
    df['home_elo'] = home_elo_list
    df['away_elo'] = away_elo_list
    return df, elos

df, final_elos = calculate_elo(df)

# -----------------------------
# 9. Target Encoding
# -----------------------------
df['FTR_enc'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# -----------------------------
# 10. Final Feature Set
# -----------------------------
feature_cols = [
    'HomeTeam_enc', 'AwayTeam_enc',
    'home_elo', 'away_elo',
    'home_form', 'away_form',
    # FBref Features
    'home_attack_xG', 'home_defense_xGA', 'home_possession', 'home_progression',
    'away_attack_xG', 'away_defense_xGA', 'away_possession', 'away_progression',
    'home_xAG', 'away_xAG', 'home_npxG', 'away_npxG', 'home_CrdY',
    # Weighted rolling stats
    'home_avg_FTHG', 'away_avg_FTAG', 'home_avg_FTAG', 'away_avg_FTHG'
]
feature_cols = [col for col in feature_cols if col in df.columns]
X = df[feature_cols]
y = df['FTR_enc']
score_y_home = df['FTHG']
score_y_away = df['FTAG']

# -----------------------------
# 11. Time-Based Train/Test Split
# -----------------------------
split_year = 2022
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

if len(X_test) == 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    score_X_train, score_X_test, score_y_home_train, score_y_home_test = train_test_split(X, score_y_home, test_size=0.2, random_state=42)
    _, _, score_y_away_train, score_y_away_test = train_test_split(X, score_y_away, test_size=0.2, random_state=42)

# -----------------------------
# 12. Model Training with Calibration
# -----------------------------
# Hold out part of training data for calibration
calib_size = int(0.2 * len(X_train))
X_calib = X_train[-calib_size:]
y_calib = y_train[-calib_size:]
X_train_fit = X_train[:-calib_size]
y_train_fit = y_train[:-calib_size]

# Train base model
base_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
base_model.fit(X_train_fit, y_train_fit)

# Calibrate probabilities
model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
model.fit(X_calib, y_calib)

# Train goal models
home_goal_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
away_goal_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
home_goal_model.fit(score_X_train, score_y_home_train)
away_goal_model.fit(score_X_train, score_y_away_train)

# -----------------------------
# 13. Prediction Helper Functions
# -----------------------------
def get_features_for_prediction(home_team, away_team, season):
    features = []

    # Encode teams
    if home_team not in le_team.classes_ or away_team not in le_team.classes_:
        raise ValueError("Team not in training data.")
    features.append(le_team.transform([home_team])[0])
    features.append(le_team.transform([away_team])[0])

    # Elo
    features.append(final_elos.get(home_team, 1500))
    features.append(final_elos.get(away_team, 1500))

    # Form (latest)
    home_form = df[df['HomeTeam'] == home_team]['home_form'].iloc[-1] if (df['HomeTeam'] == home_team).any() else 1
    away_form = df[df['AwayTeam'] == away_team]['away_form'].iloc[-1] if (df['AwayTeam'] == away_team).any() else 1
    features.append(home_form)
    features.append(away_form)

    # FBref stats for this season
    fb_season = fbref_data.get(season, {})
    h_stats = fb_season.get(home_team, {})
    a_stats = fb_season.get(away_team, {})

    defaults = {
        'attack_xG': 1.5, 'defense_xGA': 1.2, 'possession': 50.0, 'progression': 700,
        'xAG': 1.0, 'npxG': 1.3, 'CrdY': 1.0
    }

    features.extend([
        h_stats.get('attack_xG', defaults['attack_xG']),
        h_stats.get('defense_xGA', defaults['defense_xGA']),
        h_stats.get('possession', defaults['possession']),
        h_stats.get('progression', defaults['progression']),
        a_stats.get('attack_xG', defaults['attack_xG']),
        a_stats.get('defense_xGA', defaults['defense_xGA']),
        a_stats.get('possession', defaults['possession']),
        a_stats.get('progression', defaults['progression']),
        h_stats.get('xAG', defaults['xAG']),
        a_stats.get('xAG', defaults['xAG']),
        h_stats.get('npxG', defaults['npxG']),
        a_stats.get('npxG', defaults['npxG']),
        h_stats.get('CrdY', defaults['CrdY'])
    ])

    # Rolling fallbacks
    home_avg_FTHG = df[df['HomeTeam'] == home_team]['home_avg_FTHG'].iloc[-1] if (df['HomeTeam'] == home_team).any() else 1.5
    away_avg_FTAG = df[df['AwayTeam'] == away_team]['away_avg_FTAG'].iloc[-1] if (df['AwayTeam'] == away_team).any() else 1.2
    home_avg_FTAG = df[df['HomeTeam'] == home_team]['home_avg_FTAG'].iloc[-1] if (df['HomeTeam'] == home_team).any() else 1.2
    away_avg_FTHG = df[df['AwayTeam'] == away_team]['away_avg_FTHG'].iloc[-1] if (df['AwayTeam'] == away_team).any() else 1.5

    features.extend([home_avg_FTHG, away_avg_FTAG, home_avg_FTAG, away_avg_FTHG])

    return np.array([features])

# -----------------------------
# 14. Predict Match with Poisson
# -----------------------------
def predict_match_proba(home_team, away_team, season=2024):
    features = get_features_for_prediction(home_team, away_team, season)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        proba = model.predict_proba(features)[0]
        home_goals_pred = max(0, home_goal_model.predict(features)[0])
        away_goals_pred = max(0, away_goal_model.predict(features)[0])

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

    return proba, top3_poisson[0], top3_poisson, top3_h2h, home_goals_pred, away_goals_pred

# -----------------------------
# 15. Interactive CLI (Clean Output)
# -----------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ⚽ EPL Match Outcome & Score Predictor (FBref-Enhanced)")
    print("="*60)
    
    valid_teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    print("Available teams (enter exactly as shown):")
    print(", ".join(valid_teams))
    print(f"\nLoaded FBref data for seasons: {sorted(fbref_data.keys())}")

    while True:
        print("\n" + "-"*50)
        home_team = input("Enter Home Team (or 'exit' to quit): ").strip()
        if home_team.lower() == 'exit':
            print("👋 Exiting predictor. Goodbye!")
            break
        away_team = input("Enter Away Team (or 'exit' to quit): ").strip()
        if away_team.lower() == 'exit':
            print("👋 Exiting predictor. Goodbye!")
            break

        if home_team not in valid_teams:
            print(f"❌ Error: '{home_team}' not recognized. Check spelling.")
            continue
        if away_team not in valid_teams:
            print(f"❌ Error: '{away_team}' not recognized. Check spelling.")
            continue

        try:
            season_input = input("Enter season (e.g., 2024) [default: 2024]: ").strip()
            season = int(season_input) if season_input else 2024

            # Get prediction
            proba, likely, top3, h2h_top3, hg, ag = predict_match_proba(home_team, away_team, season)

            # H2H Stats
            h2h = df[(df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team)]
            h2h_hw = (h2h['FTR'] == 'H').sum()
            h2h_d = (h2h['FTR'] == 'D').sum()
            h2h_aw = (h2h['FTR'] == 'A').sum()

            print(f"\n📊 Head-to-Head: {len(h2h)} matches")
            print(f"   {home_team} wins: {h2h_hw} | Draws: {h2h_d} | {away_team} wins: {h2h_aw}")

            print(f"\n🎯 Prediction for {home_team} vs {away_team} ({season}):")
            print(f"   Home Win: {proba[0]*100:.1f}% | Draw: {proba[1]*100:.1f}% | Away Win: {proba[2]*100:.1f}%")
            print(f"   Expected Goals: {hg:.2f} - {ag:.2f}")
            print(f"   Most Likely Score: {int(likely[0])}-{int(likely[1])} (prob: {likely[2]*100:.1f}%)")

            print(f"\n🔝 Top 3 Likely Scorelines (Poisson):")
            for i, (h, a, p) in enumerate(top3):
                print(f"   {i+1}. {int(h)}-{int(a)}: {p*100:.1f}%")

            if h2h_top3 is not None and len(h2h_top3) > 0:
                print(f"\n🔍 Top 3 H2H Scorelines:")
                for _, row in h2h_top3.iterrows():
                    print(f"   {int(row['FTHG'])}-{int(row['FTAG'])}: {row['prob']*100:.1f}%")

        except Exception as e:
            print(f"❌ Error during prediction: {e}")