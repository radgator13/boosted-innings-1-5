import pandas as pd
import joblib

# === Load model and scaler ===
model = joblib.load("models/rf_model_over_4_5.joblib")
scaler = joblib.load("models/scaler_over_4_5.joblib")

# === Load game data ===
games = pd.read_csv("data/mlb_boxscores_full.csv")
games.columns = games.columns.str.strip().str.replace(" ", "_")
games["Game_Date"] = pd.to_datetime(games["Game_Date"])

# === Map team names to 3-letter codes ===
TEAM_NAME_MAP = {
    "Athletics": "OAK", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP", "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
}

games["Home_Team"] = games["Home_Team"].map(TEAM_NAME_MAP)
games["Away_Team"] = games["Away_Team"].map(TEAM_NAME_MAP)

# === Identify pending games
innings_cols = [col for col in games.columns if any(s in col for s in ["1th", "2th", "3th", "4th", "5th"])]
games["is_pending"] = games[innings_cols].apply(lambda row: row.astype(str).str.contains("Pending", case=False).any(), axis=1)

# === Calculate 1-5 inning scores
games["Runs_1_5_Away"] = games[[f"Away_{i}th" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
games["Runs_1_5_Home"] = games[[f"Home_{i}th" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").sum(axis=1)

# === Rolling 7-game averages
long_home = games[["Game_Date", "Home_Team", "Runs_1_5_Home"]].rename(columns={"Home_Team": "Team", "Runs_1_5_Home": "Runs_1_5"})
long_away = games[["Game_Date", "Away_Team", "Runs_1_5_Away"]].rename(columns={"Away_Team": "Team", "Runs_1_5_Away": "Runs_1_5"})
long_games = pd.concat([long_home, long_away]).dropna().sort_values(["Team", "Game_Date"])
long_games["Avg_Runs_1_5_Last7"] = (
    long_games.groupby("Team")["Runs_1_5"].transform(lambda x: x.shift().rolling(7, min_periods=1).mean())
)

games = games.merge(
    long_games[["Game_Date", "Team", "Avg_Runs_1_5_Last7"]],
    left_on=["Game_Date", "Home_Team"], right_on=["Game_Date", "Team"],
    how="left"
).rename(columns={"Avg_Runs_1_5_Last7": "Home_Last7_Runs_1_5"}).drop(columns=["Team"])

games = games.merge(
    long_games[["Game_Date", "Team", "Avg_Runs_1_5_Last7"]],
    left_on=["Game_Date", "Away_Team"], right_on=["Game_Date", "Team"],
    how="left"
).rename(columns={"Avg_Runs_1_5_Last7": "Away_Last7_Runs_1_5"}).drop(columns=["Team"])

# === Load team stats
standard = pd.read_csv("downloads/team_standard.csv")
advanced = pd.read_csv("downloads/team_advanced.csv")
team_stats = pd.merge(standard, advanced, on="Tm", suffixes=("_std", "_adv")).rename(columns={"Tm": "Team"})

# === Merge full-season stats
home_merged = games.merge(team_stats, left_on="Home_Team", right_on="Team", how="left").add_prefix("home_")
away_merged = games.merge(team_stats, left_on="Away_Team", right_on="Team", how="left").add_prefix("away_")

games_pred = pd.concat([
    games.reset_index(drop=True),
    home_merged.drop(columns=["home_Home_Team"], errors="ignore").reset_index(drop=True),
    away_merged.drop(columns=["away_Away_Team"], errors="ignore").reset_index(drop=True)
], axis=1)

# === Build features
model_features = [col for col in games_pred.columns if any(stat in col for stat in [
    "BB%", "K%", "ISO", "wRC+", "OBP", "SLG", "RBI", "AVG", "OPS"
])]
model_features += ["Home_Last7_Runs_1_5", "Away_Last7_Runs_1_5"]
features = games_pred[model_features].fillna(0)

# === Predict
X_scaled = scaler.transform(features)
predictions = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)

games_pred["Predicted_Over_4_5"] = predictions
games_pred["Confidence"] = probs.max(axis=1).round(4)
games_pred["Model_Total"] = (probs[:, 1] * 6).round(2)
games_pred["Actual_Over_4_5"] = None

# === Add actuals for completed games
games_pred.loc[~games_pred["is_pending"], "Runs_1_5"] = games_pred.loc[~games_pred["is_pending"], innings_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
games_pred.loc[~games_pred["is_pending"], "Actual_Over_4_5"] = (games_pred.loc[~games_pred["is_pending"], "Runs_1_5"] > 4.5).astype(int)

# === Save
games_pred[[
    "Game_Date", "Home_Team", "Away_Team",
    "Predicted_Over_4_5", "Actual_Over_4_5", "Runs_1_5",
    "Confidence", "Model_Total", "is_pending"
]].to_csv("mlb_predictions.csv", index=False)


print("✅ Predictions saved to mlb_predictions.csv")

# === Evaluate accuracy
played = games_pred[~games_pred["is_pending"]].dropna(subset=["Actual_Over_4_5"])
acc = (played["Predicted_Over_4_5"] == played["Actual_Over_4_5"]).mean()
print(f"\n🎯 Accuracy on played games: {acc:.2%}")
