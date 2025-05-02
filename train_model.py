import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os

# === Load & Normalize Game Data ===
games = pd.read_csv("data/mlb_boxscores_full.csv")
games.columns = games.columns.str.strip().str.replace(" ", "_")
games["Game_Date"] = pd.to_datetime(games["Game_Date"])

print("✅ Game file columns:", games.columns.tolist())

# === Map full team names to 3-letter codes ===
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

# === Calculate Runs_1_5 for form tracking ===
games["Runs_1_5_Away"] = games[[f"Away_{i}th" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").sum(axis=1)
games["Runs_1_5_Home"] = games[[f"Home_{i}th" for i in range(1, 6)]].apply(pd.to_numeric, errors="coerce").sum(axis=1)

# Build long-form game log
long_home = games[["Game_Date", "Home_Team", "Runs_1_5_Home"]].rename(columns={
    "Home_Team": "Team", "Runs_1_5_Home": "Runs_1_5"
})
long_away = games[["Game_Date", "Away_Team", "Runs_1_5_Away"]].rename(columns={
    "Away_Team": "Team", "Runs_1_5_Away": "Runs_1_5"
})
long_games = pd.concat([long_home, long_away]).dropna().sort_values(["Team", "Game_Date"])

# Rolling average (last 7 games)
long_games["Avg_Runs_1_5_Last7"] = (
    long_games
    .groupby("Team")["Runs_1_5"]
    .transform(lambda x: x.shift().rolling(7, min_periods=1).mean())
)

# Merge back to main games file
games = games.merge(
    long_games[["Game_Date", "Team", "Avg_Runs_1_5_Last7"]],
    left_on=["Game_Date", "Home_Team"],
    right_on=["Game_Date", "Team"],
    how="left"
).rename(columns={"Avg_Runs_1_5_Last7": "Home_Last7_Runs_1_5"}).drop(columns=["Team"])

games = games.merge(
    long_games[["Game_Date", "Team", "Avg_Runs_1_5_Last7"]],
    left_on=["Game_Date", "Away_Team"],
    right_on=["Game_Date", "Team"],
    how="left"
).rename(columns={"Avg_Runs_1_5_Last7": "Away_Last7_Runs_1_5"}).drop(columns=["Team"])

# === Load Team Stats ===
standard = pd.read_csv("downloads/team_standard.csv")
advanced = pd.read_csv("downloads/team_advanced.csv")
team_stats = pd.merge(standard, advanced, on="Tm", suffixes=("_std", "_adv")).rename(columns={"Tm": "Team"})

# === Filter games with full inning data
innings_cols = [col for col in games.columns if any(s in col for s in ["1th", "2th", "3th", "4th", "5th"])]
for col in innings_cols:
    games[col] = pd.to_numeric(games[col], errors="coerce")

games_clean = games[games[innings_cols].notna().all(axis=1)].copy()
games_clean["Runs_1_5"] = games_clean[innings_cols].sum(axis=1)
games_clean["Over_4_5"] = (games_clean["Runs_1_5"] > 4.5).astype(int)

# === Merge Team Stats
home_merged = games_clean.merge(team_stats, left_on="Home_Team", right_on="Team", how="left").add_prefix("home_")
away_merged = games_clean.merge(team_stats, left_on="Away_Team", right_on="Team", how="left").add_prefix("away_")

games_enriched = pd.concat([
    games_clean.reset_index(drop=True),
    home_merged.drop(columns=["home_Home_Team"], errors="ignore").reset_index(drop=True),
    away_merged.drop(columns=["away_Away_Team"], errors="ignore").reset_index(drop=True)
], axis=1)

# === Sanity Check
print("\n🧪 Sample merged features:")
print(games_enriched[[col for col in games_enriched.columns if "wRC+" in col or "OBP" in col]].head())

# === Select Features & Target
numeric_cols = [col for col in games_enriched.columns if any(stat in col for stat in [
    "BB%", "K%", "ISO", "wRC+", "OBP", "SLG", "RBI", "AVG", "OPS"
])]
numeric_cols += ["Home_Last7_Runs_1_5", "Away_Last7_Runs_1_5"]

features = games_enriched[numeric_cols].fillna(0)
target = games_enriched["Over_4_5"]

# === Show Distribution
print("\n📊 Class distribution:")
print(target.value_counts())

print(f"🔍 Nonzero feature rows: {(features != 0).any(axis=1).sum()} / {features.shape[0]}")

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# === Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train Model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# === Evaluate
y_pred = model.predict(X_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# === Feature Importance
importances = pd.Series(model.feature_importances_, index=features.columns)
print("\n🔥 Top 10 Features:")
print(importances.sort_values(ascending=False).head(10))

# === Save model
joblib.dump(model, "rf_model_over_4_5.joblib")
joblib.dump(scaler, "scaler_over_4_5.joblib")
print("💾 Model + scaler saved.")
